"""
qwen_kvcache_test.py - Step 3: 把量化注意力接进真 Qwen3-0.6B, 用真 q 验精度.

做法:
    Qwen3 算注意力时会调用模块级函数 eager_attention_forward(module, q, k, v, ...).
    我们 monkeypatch 它 -> 在真正算之前把 q/K/V 量化->反量化 (模拟 int8 存储),
    其余数学不变. 这样能用"模型自己算的真 q + 真 K/V"验证精度.

量化维度 (来自 Step 1):
    q: per-token   (每 head 一把尺子)
    K: per-channel (每 channel 一把尺子, 隔离离群 channel)
    V: per-token

输出:
    1) fp16 PPL 基准
    2) 只压 KV / 全压 两种配置的 PPL 及与基准的比值
    3) 每层注意力输出相对误差 (真 q), 专盯 L=0

执行 (从 repo 根目录):
    uv run tests/qwen_kvcache_test.py
"""

import math

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen3.modeling_qwen3 as qwen3_mod


# ========== 第一块: 量化版注意力 ==========

# 量化粒度, 4D 张量 [B, H, S, D]:
#   q: per-token   -> 沿 D 求 absmax (dim 3)
#   K: per-channel -> 沿 S 求 absmax (dim 2)
#   V: per-token   -> 沿 D 求 absmax (dim 3)
Q_DIMS = (3,)
K_DIMS = (2,)
V_DIMS = (3,)

# 开关: 分别控制压不压 q / K / V (做对比用)
QUANT_Q = True
QUANT_K = True
QUANT_V = True

# 诊断: 记录每层量化前后的输出误差 (用真 q)
LOG_LAYER_ERRORS = False
LAYER_ERRORS = {}


def _quant_dequant(x, reduce_dims):
    """对称 int8 量化再反量化, 返回 fp 近似 (模拟 int8 存储的精度损失)."""
    amax = x.float().abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-8)
    scale = amax / 127.0
    q = torch.clamp(torch.round(x.float() / scale), -127, 127)
    return (q * scale).to(x.dtype)


def _repeat_kv(x, n_rep):
    """[B, H_kv, S, D] -> [B, H_kv*n_rep, S, D]. GQA 展开, 与 HF 一致."""
    return x if n_rep == 1 else x.repeat_interleave(n_rep, dim=1)


def _core_attention(query, key, value, attention_mask, scaling, n_rep, dropout, training):
    """标准注意力数学 (和我们验证过的 reference 同构). 输出 [B, S, H, D]."""
    k = _repeat_kv(key, n_rep)
    v = _repeat_kv(value, n_rep)
    scores = torch.matmul(query, k.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : k.shape[-2]]
    attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    attn = F.dropout(attn, p=dropout, training=training)
    out = torch.matmul(attn, v)
    return out.transpose(1, 2).contiguous()


def quantized_attention_forward(module, query, key, value, attention_mask,
                                scaling=None, dropout=0.0, **kwargs):
    """替换 HF 的 eager_attention_forward: 先量化 q/K/V 再算注意力."""
    if scaling is None:
        scaling = module.head_dim ** -0.5
    n_rep = module.num_key_value_groups

    q_in = _quant_dequant(query, Q_DIMS) if QUANT_Q else query
    k_in = _quant_dequant(key, K_DIMS) if QUANT_K else key
    v_in = _quant_dequant(value, V_DIMS) if QUANT_V else value

    out = _core_attention(q_in, k_in, v_in, attention_mask, scaling, n_rep,
                          dropout, module.training)

    if LOG_LAYER_ERRORS:
        with torch.no_grad():
            out_fp = _core_attention(query, key, value, attention_mask, scaling,
                                     n_rep, dropout, False)
            err = ((out - out_fp).float().norm()
                   / out_fp.float().norm().clamp(min=1e-8)).item()
            LAYER_ERRORS.setdefault(module.layer_idx, []).append(err)

    return out, None


# ========== 第二块: 接到真模型上测 PPL ==========

EVAL_TEXTS = [
    "The capital of France is Paris, a city known for its museums and history.",
    "To solve a quadratic equation, we use the formula involving the discriminant.",
    "Once upon a time there lived a young blacksmith famous for his magical blades.",
    "Photosynthesis converts light energy into chemical energy inside chloroplasts.",
    "Climate change has raised the Earth's average temperature since pre-industrial times.",
    "A transformer is a neural network architecture built on self-attention mechanisms.",
]


@torch.no_grad()
def compute_ppl(model, tokenizer, texts):
    """整段文字的困惑度 PPL = exp(平均每 token 的负对数似然). 越低越好."""
    total_nll, total_tok = 0.0, 0
    for text in texts:
        ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
        n = ids.shape[1] - 1
        if n <= 0:
            continue
        loss = model(ids, labels=ids).loss   # 每 token 平均 NLL
        total_nll += loss.item() * n
        total_tok += n
    return math.exp(total_nll / total_tok)


def main():
    global QUANT_Q, QUANT_K, QUANT_V, LOG_LAYER_ERRORS

    model_path = "models/Qwen3-0.6B"
    print(f"Loading {model_path} (attn_implementation=eager)...")
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, device_map="cuda",
        attn_implementation="eager",     # 必须 eager, 我们的 patch 才会被调用
    )
    model.eval()

    assert hasattr(qwen3_mod, "eager_attention_forward"), \
        "没找到 eager_attention_forward, transformers 版本可能变了, 把报错贴给我"

    # ---- 1) fp16 基准 (原始函数, 未 patch) ----
    ppl_fp16 = compute_ppl(model, tok, EVAL_TEXTS)

    # ---- 2) 换上量化注意力 ----
    qwen3_mod.eager_attention_forward = quantized_attention_forward

    # 配置 A: 只压 KV (项目核心主张: int8 KV cache)
    QUANT_Q, QUANT_K, QUANT_V = False, True, True
    ppl_kv = compute_ppl(model, tok, EVAL_TEXTS)

    # 配置 B: q/K/V 全压 (老师伪代码的完整 int8 注意力)
    QUANT_Q, QUANT_K, QUANT_V = True, True, True
    ppl_full = compute_ppl(model, tok, EVAL_TEXTS)

    # ---- 3) 每层误差 (真 q), 复查 L=0 ----
    LAYER_ERRORS.clear()
    LOG_LAYER_ERRORS = True
    compute_ppl(model, tok, EVAL_TEXTS[:1])
    LOG_LAYER_ERRORS = False

    # ---- 报告 ----
    print(f"\n=== PPL (越接近 fp16 越好) ===")
    print(f"  fp16 baseline:        {ppl_fp16:.4f}")
    print(f"  int8 KV only (q fp16): {ppl_kv:.4f}   (ratio {ppl_kv/ppl_fp16:.4f})")
    print(f"  int8 q+K+V (全压):     {ppl_full:.4f}   (ratio {ppl_full/ppl_fp16:.4f})")

    print(f"\n=== 每层注意力输出相对误差 (真 q, 全压) ===")
    for L in sorted(LAYER_ERRORS):
        e = LAYER_ERRORS[L]
        mark = "  <-- 之前用假 q 时这层 25%" if L == 0 else ""
        print(f"  L={L:<3} {sum(e)/len(e)*100:5.2f}%{mark}")


if __name__ == "__main__":
    main()