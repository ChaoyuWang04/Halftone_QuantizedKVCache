"""
int8_decode_attn_triton.py - Phase C: 融合 int8 decode attention (flash-style)

decode (Sq=1): q fp16 [H_q,1,D], K/V int8 cache [H_kv,S,D] + scale.
读 int8 (省一半带宽) -> register 反量化 -> fp32 online-softmax attention -> fp16 out.
对照: 同结构的 fp16 版本, 隔离出纯带宽红利.
"""

import torch
import triton
import triton.language as tl


# ---------- int8 版本 ----------
@triton.jit
def _int8_decode_kernel(
    q_ptr, k_ptr, ksc_ptr, v_ptr, vsc_ptr, out_ptr,
    H_q, H_kv, S, D, n_rep, sm_scale,
    sqh, sqd,
    skh, sks, skd,
    svh, svs, svd,
    skch, skcd,            # k_scale [H_kv,1,D] strides
    svch, svcs,            # v_scale [H_kv,S,1] strides
    soh, sod,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    h = tl.program_id(0)          # query head
    kvh = h // n_rep              # 对应的 kv head
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D

    q = tl.load(q_ptr + h * sqh + offs_d * sqd, mask=d_mask, other=0.0).to(tl.float32)
    ksc = tl.load(ksc_ptr + kvh * skch + offs_d * skcd, mask=d_mask, other=0.0).to(tl.float32)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for n0 in range(0, S, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        n_mask = offs_n < S

        # K 瓦片 [BLOCK_N, D]: D 在内连续 -> 合并访存
        k = tl.load(k_ptr + kvh * skh + offs_n[:, None] * sks + offs_d[None, :] * skd,
                    mask=n_mask[:, None] & d_mask[None, :], other=0).to(tl.float32)
        k = k * ksc[None, :]                                  # per-channel 反量化
        scores = tl.sum(q[None, :] * k, axis=1) * sm_scale    # [BLOCK_N]
        scores = tl.where(n_mask, scores, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        p = tl.exp(scores - m_new)
        corr = tl.exp(m_i - m_new)
        l_i = l_i * corr + tl.sum(p, axis=0)

        vsc = tl.load(vsc_ptr + kvh * svch + offs_n * svcs, mask=n_mask, other=0.0).to(tl.float32)
        v = tl.load(v_ptr + kvh * svh + offs_n[:, None] * svs + offs_d[None, :] * svd,
                    mask=n_mask[:, None] & d_mask[None, :], other=0).to(tl.float32)
        v = v * vsc[:, None]                                  # per-token 反量化
        acc = acc * corr + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    acc = acc / l_i
    tl.store(out_ptr + h * soh + offs_d * sod, acc.to(tl.float16), mask=d_mask)


# ---------- fp16 对照版本 (同结构, 不反量化) ----------
@triton.jit
def _fp16_decode_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    H_q, H_kv, S, D, n_rep, sm_scale,
    sqh, sqd, skh, sks, skd, svh, svs, svd, soh, sod,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    h = tl.program_id(0)
    kvh = h // n_rep
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    q = tl.load(q_ptr + h * sqh + offs_d * sqd, mask=d_mask, other=0.0).to(tl.float32)

    m_i = float("-inf"); l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for n0 in range(0, S, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        n_mask = offs_n < S
        k = tl.load(k_ptr + kvh * skh + offs_n[:, None] * sks + offs_d[None, :] * skd,
                    mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        scores = tl.sum(q[None, :] * k, axis=1) * sm_scale
        scores = tl.where(n_mask, scores, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        p = tl.exp(scores - m_new); corr = tl.exp(m_i - m_new)
        l_i = l_i * corr + tl.sum(p, axis=0)
        v = tl.load(v_ptr + kvh * svh + offs_n[:, None] * svs + offs_d[None, :] * svd,
                    mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        acc = acc * corr + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new
    acc = acc / l_i
    tl.store(out_ptr + h * soh + offs_d * sod, acc.to(tl.float16), mask=d_mask)


# ---------- wrappers ----------
def int8_decode_attn(q, k_i8, k_sc, v_i8, v_sc, n_rep):
    H_q, _, D = q.shape; H_kv, S, _ = k_i8.shape
    out = torch.empty((H_q, 1, D), device=q.device, dtype=torch.float16)
    _int8_decode_kernel[(H_q,)](
        q, k_i8, k_sc, v_i8, v_sc, out, H_q, H_kv, S, D, n_rep, 1.0 / (D ** 0.5),
        q.stride(0), q.stride(2),
        k_i8.stride(0), k_i8.stride(1), k_i8.stride(2),
        v_i8.stride(0), v_i8.stride(1), v_i8.stride(2),
        k_sc.stride(0), k_sc.stride(2),
        v_sc.stride(0), v_sc.stride(1),
        out.stride(0), out.stride(2),
        BLOCK_N=64, BLOCK_D=triton.next_power_of_2(D),
    )
    return out

def fp16_decode_attn(q, k, v, n_rep):
    H_q, _, D = q.shape; H_kv, S, _ = k.shape
    out = torch.empty((H_q, 1, D), device=q.device, dtype=torch.float16)
    _fp16_decode_kernel[(H_q,)](
        q, k, v, out, H_q, H_kv, S, D, n_rep, 1.0 / (D ** 0.5),
        q.stride(0), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(2),
        BLOCK_N=64, BLOCK_D=triton.next_power_of_2(D),
    )
    return out


# ---------- torch 参考 + 量化 helper ----------
def _ref(q, k, v, n_rep, sm):
    q = q.float(); k = k.float(); v = v.float()
    k = k.repeat_interleave(n_rep, 0); v = v.repeat_interleave(n_rep, 0)
    a = torch.softmax((q @ k.transpose(1, 2)) * sm, dim=-1)
    return a @ v

def _quant_k(x):   # per-channel, 沿 S (dim 1)
    amax = x.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-8); sc = amax / 127
    return torch.clamp(torch.round(x.float() / sc), -127, 127).to(torch.int8), sc.to(torch.float16)

def _quant_v(x):   # per-token, 沿 D (dim 2)
    amax = x.float().abs().amax(dim=2, keepdim=True).clamp(min=1e-8); sc = amax / 127
    return torch.clamp(torch.round(x.float() / sc), -127, 127).to(torch.int8), sc.to(torch.float16)

def _relerr(ref, x):
    return ((x.float() - ref.float()).norm() / ref.float().norm().clamp(min=1e-8)).item()


def _bench(fn, *a, warmup=20, iters=100):
    for _ in range(warmup): fn(*a)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn(*a)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000  # us


def _self_test():
    dev = "cuda"; H_q, H_kv, D, n_rep = 16, 8, 128, 2; sm = 1.0 / D ** 0.5
    for S in [64, 512, 4096]:
        q = torch.randn(H_q, 1, D, device=dev, dtype=torch.float16)
        k = torch.randn(H_kv, S, D, device=dev, dtype=torch.float16)
        v = torch.randn(H_kv, S, D, device=dev, dtype=torch.float16)
        k_i8, k_sc = _quant_k(k); v_i8, v_sc = _quant_v(v)

        # 正确性: int8 kernel 对 "dequant 后的 torch 参考"; fp16 kernel 对 "原始 torch 参考"
        out_i8 = int8_decode_attn(q, k_i8, k_sc, v_i8, v_sc, n_rep)
        ref_i8 = _ref(q, k_i8.float() * k_sc.float(), v_i8.float() * v_sc.float(), n_rep, sm)
        out_fp = fp16_decode_attn(q, k, v, n_rep)
        ref_fp = _ref(q, k, v, n_rep, sm)

        t_i8 = _bench(int8_decode_attn, q, k_i8, k_sc, v_i8, v_sc, n_rep)
        t_fp = _bench(fp16_decode_attn, q, k, v, n_rep)
        print(f"S={S:>4}  int8 vs ref={_relerr(ref_i8, out_i8)*100:.2f}%  "
              f"fp16 vs ref={_relerr(ref_fp, out_fp)*100:.2f}%  |  "
              f"int8={t_i8:6.1f}us  fp16={t_fp:6.1f}us  speedup={t_fp/t_i8:.2f}x")


if __name__ == "__main__":
    _self_test()