"""
qwen_kvcache_full.py - Phase A: 物理 int8 cache + q 粒度真模型 A/B

新增 vs Step 3:
  1) QuantizedKVCache: HF DynamicCache 子类, K/V 物理以 int8 存储
  2) Cache 体积测量 (int8 vs fp16 equiv)
  3) q 粒度 ablation: per-tensor / per-token / per-channel, 真 q 真模型
"""

import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
import transformers.models.qwen3.modeling_qwen3 as qwen3_mod


# ===== A. 通用对称 int8 量化 =====
def _quant_dequant(x, reduce_dims):
    amax = x.float().abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-8)
    scale = amax / 127.0
    q = torch.clamp(torch.round(x.float() / scale), -127, 127)
    return (q * scale).to(x.dtype)


# ===== B. 物理 int8 KV cache =====
class QuantizedKVCache(DynamicCache):
    """K/V 物理以 int8 存储, 读取时反量化为 fp16 给 attention.
    K per-channel (沿 S 求 amax), V per-token (沿 D 求 amax).
    这是生产里"先量化再存 cache"的最简实现.
    """
    def __init__(self):
        super().__init__()
        self._k_i8, self._v_i8 = {}, {}
        self._k_sc, self._v_sc = {}, {}
        self._lens = {}

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # key/value [B, H, S_new, D] fp16 -> int8 + scale
        # K per-channel: 沿 S (dim 2) 求 amax, scale [B,H,1,D]
        k_amax = key_states.float().abs().amax(dim=2, keepdim=True).clamp(min=1e-8)
        k_scale = k_amax / 127.0
        k_int8 = torch.clamp(torch.round(key_states.float() / k_scale), -127, 127).to(torch.int8)
        # V per-token: 沿 D (dim 3) 求 amax, scale [B,H,S,1]
        v_amax = value_states.float().abs().amax(dim=3, keepdim=True).clamp(min=1e-8)
        v_scale = v_amax / 127.0
        v_int8 = torch.clamp(torch.round(value_states.float() / v_scale), -127, 127).to(torch.int8)

        # 物理存储 (decode 时拼旧的, prefill 一次塞满)
        if layer_idx in self._k_i8:
            self._k_i8[layer_idx] = torch.cat([self._k_i8[layer_idx], k_int8], dim=2)
            self._v_i8[layer_idx] = torch.cat([self._v_i8[layer_idx], v_int8], dim=2)
            self._v_sc[layer_idx] = torch.cat([self._v_sc[layer_idx], v_scale.to(torch.float16)], dim=2)
            # K scale 是 per-channel, 简化版保留旧 scale (生产用 block-wise per-channel)
        else:
            self._k_i8[layer_idx] = k_int8
            self._v_i8[layer_idx] = v_int8
            self._k_sc[layer_idx] = k_scale.to(torch.float16)
            self._v_sc[layer_idx] = v_scale.to(torch.float16)

        self._lens[layer_idx] = self._k_i8[layer_idx].shape[2]

        # 反量化返回 (模拟 fused kernel 在 register 里 dequant)
        k_out = (self._k_i8[layer_idx].float() * self._k_sc[layer_idx].float()).to(key_states.dtype)
        v_out = (self._v_i8[layer_idx].float() * self._v_sc[layer_idx].float()).to(value_states.dtype)
        return k_out, v_out

    def get_seq_length(self, layer_idx=0, cache_position=None):
        return self._lens.get(layer_idx, 0)

    def __len__(self):
        return len(self._k_i8)

    # ---- 体积测量 ----
    def storage_bytes(self):
        """实际占用: int8 数据 1B/元素 + fp16 scale 2B/元素."""
        b = 0
        for d in (self._k_i8, self._v_i8):
            for t in d.values(): b += t.numel()
        for d in (self._k_sc, self._v_sc):
            for t in d.values(): b += t.numel() * 2
        return b

    def fp16_equiv_bytes(self):
        """同样 K/V 用 fp16 存的对照."""
        b = 0
        for d in (self._k_i8, self._v_i8):
            for t in d.values(): b += t.numel() * 2
        return b


# ===== C. 只做 q 量化的 attention patch (K/V 由 cache 处理) =====
Q_DIMS = (3,)   # 默认 per-token; ablation 会覆盖

def quantized_attention_forward(module, query, key, value, attention_mask,
                                scaling=None, dropout=0.0, **kwargs):
    """patch eager_attention_forward: 只对 q 做 quant-dequant.
    K/V 此时已经是从 QuantizedKVCache 反量化出来的 fp16, 不再处理."""
    if scaling is None:
        scaling = module.head_dim ** -0.5
    n_rep = module.num_key_value_groups

    q_in = _quant_dequant(query, Q_DIMS)
    k = key if n_rep == 1 else key.repeat_interleave(n_rep, dim=1)
    v = value if n_rep == 1 else value.repeat_interleave(n_rep, dim=1)

    scores = torch.matmul(q_in, k.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, :k.shape[-2]]
    attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q_in.dtype)
    attn = F.dropout(attn, p=dropout, training=module.training)
    out = torch.matmul(attn, v).transpose(1, 2).contiguous()
    return out, None


# ===== D. PPL + 实验主循环 =====
EVAL_TEXTS = [
    "The capital of France is Paris, a city known for its museums and history.",
    "To solve a quadratic equation, we use the formula involving the discriminant.",
    "Once upon a time there lived a young blacksmith famous for his magical blades.",
    "Photosynthesis converts light energy into chemical energy inside chloroplasts.",
    "Climate change has raised the Earth's average temperature since pre-industrial times.",
    "A transformer is a neural network architecture built on self-attention mechanisms.",
]

@torch.no_grad()
def compute_ppl(model, tok, texts, cache_factory=None):
    total_nll, total_tok = 0.0, 0
    last_cache = None
    for text in texts:
        ids = tok(text, return_tensors="pt").input_ids.to(model.device)
        n = ids.shape[1] - 1
        if n <= 0: continue
        cache = cache_factory() if cache_factory else None
        out = model(ids, labels=ids, past_key_values=cache, use_cache=True)
        total_nll += out.loss.item() * n
        total_tok += n
        if cache is not None: last_cache = cache
    return math.exp(total_nll / total_tok), last_cache


def main():
    global Q_DIMS

    model_path = "models/Qwen3-0.6B"
    print(f"Loading {model_path}...")
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()

    orig_attn = qwen3_mod.eager_attention_forward

    # ---- 1) fp16 基准 ----
    ppl_fp16, _ = compute_ppl(model, tok, EVAL_TEXTS, cache_factory=DynamicCache)
    print(f"\nfp16 baseline PPL = {ppl_fp16:.4f}")

    # ---- 2) 装上 q 量化 patch (K/V 由 cache 处理) ----
    qwen3_mod.eager_attention_forward = quantized_attention_forward

    # ---- 3) q 粒度 ablation ----
    print("\n=== q 粒度 ablation (K per-channel / V per-token 由 QuantizedKVCache 处理) ===")
    print(f"  {'q config':<18}{'PPL':>10}{'ratio':>10}")
    for label, dims in [("per-tensor", (1, 2, 3)), ("per-token", (3,)), ("per-channel", (2,))]:
        Q_DIMS = dims
        ppl, _ = compute_ppl(model, tok, EVAL_TEXTS, cache_factory=QuantizedKVCache)
        print(f"  q {label:<16}{ppl:>10.4f}{ppl/ppl_fp16:>10.4f}")

    # ---- 4) Cache 体积测量 (用最佳 q 配置 = per-token) ----
    Q_DIMS = (3,)
    print("\n=== Physical int8 cache 体积 (per text) ===")
    print(f"  {'seq_len':>8}{'int8 cache':>15}{'fp16 equiv':>15}{'ratio':>9}")
    int8_total, fp16_total = 0, 0
    for text in EVAL_TEXTS:
        ids = tok(text, return_tensors="pt").input_ids.to(model.device)
        cache = QuantizedKVCache()
        model(ids, past_key_values=cache, use_cache=True)
        i8 = cache.storage_bytes()
        fp = cache.fp16_equiv_bytes()
        int8_total += i8; fp16_total += fp
        print(f"  {ids.shape[1]:>8}{i8/1024:>12.1f} KB{fp/1024:>12.1f} KB{i8/fp:>9.3f}")
    print(f"  {'TOTAL':>8}{int8_total/1024:>12.1f} KB{fp16_total/1024:>12.1f} KB{int8_total/fp16_total:>9.3f}")

    qwen3_mod.eager_attention_forward = orig_attn


if __name__ == "__main__":
    main()