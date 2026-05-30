"""
page_attention_int8_pytorch.py - Step 2: int8 量化版 decode 注意力.

唯一变量: 量化噪声. 注意力数学直接复用已验证的 reference (page_attention_reference),
所以任何精度差都只来自把 q/k/v 量化成 int8 这一件事.

量化维度 (来自 Step 1 数据驱动的结论):
    q: per-tensor   (q 数值少, 单 scale 够)
    K: per-channel  (scale 沿 token 维 reduce -> 每 (head, channel) 一个, 隔离离群 channel)
    V: per-token    (scale 沿 channel 维 reduce -> 每 (head, token) 一个)

为什么 dequant 后再做 matmul (而非 int8 matmul 后乘 scale):
    K per-channel 的 scale 在 QK^T 的求和轴(channel)上, V per-token 的 scale 在 PV 的
    求和轴(token)上, scale 无法外提 -> 必须先反量化. 这与老师伪代码同构, 且精度等价于
    "求和前 dequant 的真 int8 kernel". 提速(真 int8 tensor core)是被外包的 Step 5.
"""

import torch

from page_attention_reference import page_attention_reference, repeat_kv


def quantize_int8(x, reduce_dims):
    """对称 int8 量化. 返回 (q_int8, scale).
    scale 沿 reduce_dims 求 absmax, 其余轴各自独立 (粒度由 reduce_dims 决定)."""
    x = x.float()
    amax = x.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-8)
    scale = amax / 127.0
    q = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)
    return q, scale


def dequantize_int8(q, scale):
    return q.float() * scale


@torch.no_grad()
def page_attention_int8(q, k_cache, v_cache, n_rep=None,
                        q_dims=(0, 1, 2), k_dims=(1,), v_dims=(2,)):
    """
    int8 量化 decode 注意力.
      q:       [H_q, Sq, D]   默认 per-tensor
      k_cache: [H_kv, S, D]   默认 per-channel (reduce token 维)
      v_cache: [H_kv, S, D]   默认 per-token   (reduce channel 维)
    q_dims/k_dims/v_dims 暴露出来是为了做消融 (例如全传 (0,1,2) 即全 per-tensor).
    """
    H_q, _, _ = q.shape
    H_kv = k_cache.shape[0]
    if n_rep is None:
        n_rep = H_q // H_kv

    # 量化 (模拟真实 int8 KV cache 的存储: cache 按 H_kv 个 head 存, 不是 repeat 后的 H_q)
    q_int8, q_scale = quantize_int8(q, q_dims)
    k_int8, k_scale = quantize_int8(k_cache, k_dims)
    v_int8, v_scale = quantize_int8(v_cache, v_dims)

    # 反量化回 fp16 (与 reference 同精度, 隔离掉 fp32-vs-fp16 的干扰, 只留量化噪声)
    q_dq = dequantize_int8(q_int8, q_scale).to(q.dtype)
    k_dq = dequantize_int8(k_int8, k_scale).to(q.dtype)
    v_dq = dequantize_int8(v_int8, v_scale).to(q.dtype)

    # 复用已验证的 reference 数学
    return page_attention_reference(q_dq, k_dq, v_dq, n_rep)


def _rel_err(ref, x):
    return ((x.float() - ref.float()).norm() / ref.float().norm().clamp(min=1e-8)).item()


def _sqnr_db(ref, x):
    ref = ref.float()
    noise = (ref - x.float()).pow(2).sum()
    if noise.item() == 0:
        return float("inf")
    return (10 * torch.log10(ref.pow(2).sum() / noise)).item()


def _self_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    n_rep = 2

    def run_case(name, k, v):
        H_kv, S, D = k.shape
        q = torch.randn(H_kv * n_rep, 1, D, device=device, dtype=dtype) * k.std()
        out_ref = page_attention_reference(q, k, v)

        # 我们 Step 1 选的维度
        out_ours = page_attention_int8(q, k, v)  # K per-channel, V per-token
        # 消融基线: 全 per-tensor
        out_pt = page_attention_int8(q, k, v,
                                     q_dims=(0, 1, 2), k_dims=(0, 1, 2), v_dims=(0, 1, 2))

        print(f"\n[{name}] S={S}")
        print(f"  ours (K per-channel, V per-token):  "
              f"rel_err={_rel_err(out_ref, out_ours)*100:.2f}%   SQNR={_sqnr_db(out_ref, out_ours):.1f} dB")
        print(f"  baseline (all per-tensor):          "
              f"rel_err={_rel_err(out_ref, out_pt)*100:.2f}%   SQNR={_sqnr_db(out_ref, out_pt):.1f} dB")
        return _rel_err(out_ref, out_ours)

    # 真实采集的 KV (有真实离群结构, 才能体现 per-channel 的价值)
    blob = torch.load("data/kvcache_dump.pt", weights_only=False)
    errs = []
    for L in [0, blob["metadata"]["num_layers"] // 2, blob["metadata"]["num_layers"] - 1]:
        k = blob["data"][0]["K_per_layer"][L].to(device, dtype)
        v = blob["data"][0]["V_per_layer"][L].to(device, dtype)
        errs.append(run_case(f"real L={L}", k, v))

    worst = max(errs)
    print(f"\n最差 rel_err = {worst*100:.2f}%  (老师标准: < 5~10%)")
    assert worst < 0.10, f"输出误差 {worst*100:.2f}% 超过 10%, 需要排查"
    print("PASS: int8 量化注意力输出在误差范围内.")


if __name__ == "__main__":
    _self_test()