"""
quant_kv_triton.py - Phase B: 融合的 int8 量化 kernel (findmax+quant+write 一次过内存)

对比对象: 朴素 PyTorch (abs/amax/div/round/clamp/to 多次读写显存).
量化方案: 对称 int8, K per-channel (沿 S 归约), V per-token (沿 D 归约).
输入张量统一 [B, H, S, D].
"""

import torch
import triton
import triton.language as tl


# ---------- K: per-channel, 沿 S 归约 ----------
# 每个 program 负责一个 (b, h, d) 列, 把该列沿 S 的全部值读进来求 absmax 再量化.
@triton.jit
def _quant_k_perchannel_kernel(
    x_ptr, q_ptr, scale_ptr,
    B, H, S, D,
    stride_b, stride_h, stride_s, stride_d,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)              # 一个 program = 一个 (b,h,d) 列
    d = pid % D
    h = (pid // D) % H
    b = pid // (D * H)

    base = b * stride_b + h * stride_h + d * stride_d
    offs_s = tl.arange(0, BLOCK_S)

    # 第一遍: 沿 S 求 absmax
    amax = 0.0
    for s0 in range(0, S, BLOCK_S):
        s = s0 + offs_s
        mask = s < S
        x = tl.load(x_ptr + base + s * stride_s, mask=mask, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x)))

    scale = tl.maximum(amax, 1e-8) / 127.0
    # scale 形状 [B,H,1,D] -> 线性 index = (b*H + h)*D + d
    tl.store(scale_ptr + (b * H + h) * D + d, scale)

    # 第二遍: 量化写回
    for s0 in range(0, S, BLOCK_S):
        s = s0 + offs_s
        mask = s < S
        x = tl.load(x_ptr + base + s * stride_s, mask=mask, other=0.0).to(tl.float32)
        q = tl.extra.cuda.libdevice.round(x / scale)
        q = tl.minimum(tl.maximum(q, -127.0), 127.0)
        tl.store(q_ptr + base + s * stride_s, q.to(tl.int8), mask=mask)


# ---------- V: per-token, 沿 D 归约 ----------
# 每个 program 负责一个 (b, h, s) 行, 沿 D 求 absmax 再量化.
@triton.jit
def _quant_v_pertoken_kernel(
    x_ptr, q_ptr, scale_ptr,
    B, H, S, D,
    stride_b, stride_h, stride_s, stride_d,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)             # 一个 program = 一个 (b,h,s) 行
    s = pid % S
    h = (pid // S) % H
    b = pid // (S * H)

    base = b * stride_b + h * stride_h + s * stride_s
    offs_d = tl.arange(0, BLOCK_D)

    amax = 0.0
    for d0 in range(0, D, BLOCK_D):
        d = d0 + offs_d
        mask = d < D
        x = tl.load(x_ptr + base + d * stride_d, mask=mask, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x)))

    scale = tl.maximum(amax, 1e-8) / 127.0
    # scale 形状 [B,H,S,1] -> 线性 index = (b*H + h)*S + s
    tl.store(scale_ptr + (b * H + h) * S + s, scale)

    for d0 in range(0, D, BLOCK_D):
        d = d0 + offs_d
        mask = d < D
        x = tl.load(x_ptr + base + d * stride_d, mask=mask, other=0.0).to(tl.float32)
        q = tl.extra.cuda.libdevice.round(x / scale)
        q = tl.minimum(tl.maximum(q, -127.0), 127.0)
        tl.store(q_ptr + base + d * stride_d, q.to(tl.int8), mask=mask)


def quant_k_perchannel_triton(x):
    """x: [B,H,S,D] -> (int8 [B,H,S,D], scale fp16 [B,H,1,D])."""
    B, H, S, D = x.shape
    x = x.contiguous()
    q = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty((B, H, 1, D), device=x.device, dtype=torch.float32)
    grid = (B * H * D,)
    _quant_k_perchannel_kernel[grid](
        x, q, scale, B, H, S, D,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        BLOCK_S=triton.next_power_of_2(S),
    )
    return q, scale.to(torch.float16)


def quant_v_pertoken_triton(x):
    """x: [B,H,S,D] -> (int8 [B,H,S,D], scale fp16 [B,H,S,1])."""
    B, H, S, D = x.shape
    x = x.contiguous()
    q = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty((B, H, S, 1), device=x.device, dtype=torch.float32)
    grid = (B * H * S,)
    _quant_v_pertoken_kernel[grid](
        x, q, scale, B, H, S, D,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        BLOCK_D=triton.next_power_of_2(D),
    )
    return q, scale.to(torch.float16)


# ---------- 朴素 PyTorch 对照 ----------
def quant_perchannel_torch(x):
    amax = x.float().abs().amax(dim=2, keepdim=True).clamp(min=1e-8)
    scale = amax / 127.0
    q = torch.clamp(torch.round(x.float() / scale), -127, 127).to(torch.int8)
    return q, scale.to(torch.float16)

def quant_pertoken_torch(x):
    amax = x.float().abs().amax(dim=3, keepdim=True).clamp(min=1e-8)
    scale = amax / 127.0
    q = torch.clamp(torch.round(x.float() / scale), -127, 127).to(torch.int8)
    return q, scale.to(torch.float16)


# ---------- 正确性 + 速度自测 ----------
def _bench(fn, *args, warmup=20, iters=100):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000  # us


def _self_test():
    device = "cuda"
    # 用接近真实 KV cache 的形状: [B=1, H=8, S, D=128], 取一个有意义的 S
    B, H, D = 1, 8, 128
    for S in [64, 512, 4096]:
        x = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

        # --- K per-channel ---
        qt, st = quant_k_perchannel_triton(x)
        qp, sp = quant_perchannel_torch(x)
        # int8 结果允许 ±1 的 round 边界差异 (浮点 round 实现细节)
        max_int_diff = (qt.int() - qp.int()).abs().max().item()
        scale_ok = torch.allclose(st.float(), sp.float(), rtol=1e-3)
        t_tri = _bench(lambda: quant_k_perchannel_triton(x))
        t_torch = _bench(lambda: quant_perchannel_torch(x))
        print(f"[K per-channel S={S:>4}] int8 max diff={max_int_diff}, scale_ok={scale_ok}  "
              f"triton={t_tri:7.1f}us  torch={t_torch:7.1f}us  speedup={t_torch/t_tri:.2f}x")

        # --- V per-token ---
        qt, st = quant_v_pertoken_triton(x)
        qp, sp = quant_pertoken_torch(x)
        max_int_diff = (qt.int() - qp.int()).abs().max().item()
        scale_ok = torch.allclose(st.float(), sp.float(), rtol=1e-3)
        t_tri = _bench(lambda: quant_v_pertoken_triton(x))
        t_torch = _bench(lambda: quant_pertoken_torch(x))
        print(f"[V per-token  S={S:>4}] int8 max diff={max_int_diff}, scale_ok={scale_ok}  "
              f"triton={t_tri:7.1f}us  torch={t_torch:7.1f}us  speedup={t_torch/t_tri:.2f}x")


if __name__ == "__main__":
    _self_test()