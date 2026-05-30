"""
page_attention_reference.py - Step 2: fp16 decode 注意力参考实现 (ground truth).

定位:
    这是"金标准", 后面 int8 量化版对着它验精度.
    所以它自己必须先被验证正确 -> __main__ 里和 torch 官方 SDPA 对拍.

只做 decode 阶段:
    q 是当前要生成的 token(s), shape [H_q, Sq, D] (Sq 通常=1)
    K/V cache 是过去所有 token, shape [H_kv, S, D]
    decode token 看得见全部历史 -> 不需要 causal mask
    (prefill 的 causal mask 留到 Step 3 集成时加, 这步只把量化地基打对)

GQA:
    Qwen3-0.6B: 16 个 Q head, 8 个 KV head -> 每个 KV head 服务 2 个 Q head (n_rep=2)
    Q head h 用 KV head h // n_rep
"""

import torch
import torch.nn.functional as F


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[H_kv, S, D] -> [H_kv*n_rep, S, D]. KV head k 复制到 Q head [k*n_rep : (k+1)*n_rep].
    与 HF repeat_kv 等价: Q head h 对应 KV head h // n_rep."""
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=0)


@torch.no_grad()
def page_attention_reference(q, k_cache, v_cache, n_rep=None):
    """
    fp16 decode 注意力.
      q:       [H_q, Sq, D]
      k_cache: [H_kv, S, D]
      v_cache: [H_kv, S, D]
    返回 out: [H_q, Sq, D]
    """
    H_q, Sq, D = q.shape
    H_kv = k_cache.shape[0]
    if n_rep is None:
        assert H_q % H_kv == 0, f"H_q={H_q} 不能被 H_kv={H_kv} 整除"
        n_rep = H_q // H_kv

    K = repeat_kv(k_cache, n_rep)   # [H_q, S, D]
    V = repeat_kv(v_cache, n_rep)   # [H_q, S, D]

    scale = 1.0 / (D ** 0.5)
    scores = torch.bmm(q, K.transpose(1, 2)) * scale          # [H_q, Sq, S]
    # softmax 在 fp32 做: 注意力对数值精度敏感, fp16 直接 softmax 易精度损失
    attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    out = torch.bmm(attn, V)                                  # [H_q, Sq, D]
    return out


def _sqnr_db(ref, x):
    ref = ref.float()
    sig = ref.pow(2).sum()
    noise = (ref - x.float()).pow(2).sum()
    if noise.item() == 0:
        return float("inf")
    return (10 * torch.log10(sig / noise)).item()


def _self_test():
    """对拍 torch 官方 SDPA, 确认我们的 reference 是真 ground truth."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    n_rep = 2  # Qwen3-0.6B: 16 q heads / 8 kv heads

    # ---- 1) 纯随机张量, 验形状 + GQA 逻辑 ----
    H_kv, S, D = 8, 64, 128
    H_q = H_kv * n_rep
    q = torch.randn(H_q, 1, D, device=device, dtype=dtype)
    k = torch.randn(H_kv, S, D, device=device, dtype=dtype)
    v = torch.randn(H_kv, S, D, device=device, dtype=dtype)

    out_ref = page_attention_reference(q, k, v)
    out_sdpa = F.scaled_dot_product_attention(
        q, repeat_kv(k, n_rep), repeat_kv(v, n_rep), is_causal=False
    )
    sqnr = _sqnr_db(out_sdpa, out_ref)
    print(f"[random]   shape={tuple(out_ref.shape)}  SQNR vs SDPA = {sqnr:.1f} dB")
    assert sqnr > 35, f"reference 与官方 SDPA 不一致 (SQNR={sqnr:.1f}), 实现有 bug"

    # ---- 2) 真实采集的 KV, 用真实分布再验一次 ----
    try:
        blob = torch.load("data/kvcache_dump.pt", weights_only=False)
        L = blob["metadata"]["num_layers"] // 2                       # 中间一层
        k_real = blob["data"][0]["K_per_layer"][L].to(device, dtype)  # [H_kv, S, D]
        v_real = blob["data"][0]["V_per_layer"][L].to(device, dtype)
        H_kv_r, S_r, D_r = k_real.shape
        # 没采 q, 合成一个和真实 K 同量级的 q
        q_real = torch.randn(H_kv_r * n_rep, 1, D_r, device=device, dtype=dtype) * k_real.std()

        out_ref = page_attention_reference(q_real, k_real, v_real)
        out_sdpa = F.scaled_dot_product_attention(
            q_real, repeat_kv(k_real, n_rep), repeat_kv(v_real, n_rep), is_causal=False
        )
        sqnr = _sqnr_db(out_sdpa, out_ref)
        print(f"[real L={L}] S={S_r}  SQNR vs SDPA = {sqnr:.1f} dB")
        assert sqnr > 35
    except FileNotFoundError:
        print("[real] 跳过 (没找到 data/kvcache_dump.pt)")

    print("PASS: reference 与官方 SDPA 一致, 可作为 int8 版的 ground truth.")


if __name__ == "__main__":
    _self_test()