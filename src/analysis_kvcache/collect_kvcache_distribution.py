"""
collect_kvcache.py - Step 1: 采集 Qwen3 模型的 KV cache, 用于分布分析.

设计目标:
    1. 输入一组多样化 prompts (知识/数学/创作/代码/对话)
    2. 跑 prefill, 拿到每层每个位置的 K 和 V
    3. 保存为 .pt 文件, 供后续分布分析用

为什么这样设计:
    - 多样 prompt: 让分布覆盖真实场景, 不偏 (避免只采"数学题"导致量化方案偏)
    - 长输入 (200-500 token): 让位置分布有意义
    - 保存全部张量: 后续分析灵活 (per-tensor / per-token / per-head / per-channel 都能算)

采集到的数据结构:
    cache_data = {
        "metadata": {...},
        "data": [
            {"prompt", "seq_len", "K_per_layer", "V_per_layer"}, ...
        ],
    }
    每个 layer_K shape: [num_kv_heads, seq_len, head_dim]
    seq_len 因 prompt 而异.

执行:
    python analysis/collect_kvcache.py
    输出: data/kvcache_dump.pt (约几百 MB)
"""

import argparse
import sys
import time
from pathlib import Path

import torch

# 多样化测试 prompts (覆盖不同 LLM 使用场景, 不限于单一分布)
PROMPTS = [
    # 知识查询
    "The capital of France is Paris. It is known for its iconic Eiffel Tower, "
    "world-class museums like the Louvre, and rich cultural history dating back to Roman times. "
    "The city is divided into 20 arrondissements, each with its own character.",

    # 数学计算
    "To solve a quadratic equation ax^2 + bx + c = 0, we use the quadratic formula: "
    "x = (-b plus or minus square root of (b squared minus 4ac), all divided by 2a. "
    "The discriminant b^2 - 4ac tells us about the nature of the roots.",

    # 长文本叙事
    "Once upon a time, in a small village nestled between two mountains, there lived a young "
    "blacksmith named Elias. He was known throughout the land for his exceptional craftsmanship "
    "and the magical quality of his blades. One day, a mysterious traveler arrived at his forge.",

    # 代码
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n"
    "    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n"
    "    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",

    # 对话/QA
    "User: Can you explain how a transformer model works?\nAssistant: Sure! A transformer is a "
    "neural network architecture that relies on self-attention mechanisms. It processes input "
    "sequences in parallel rather than sequentially like RNNs.",

    # 科学说明
    "Photosynthesis is the process by which plants convert light energy into chemical energy. "
    "It occurs primarily in the chloroplasts of plant cells, where chlorophyll absorbs sunlight. "
    "The overall reaction takes carbon dioxide and water, producing glucose and oxygen.",

    # 列表/结构化
    "Top 5 programming languages in 2024: 1. Python - dominant in AI/ML and data science. "
    "2. JavaScript - essential for web development. 3. Rust - growing for systems programming. "
    "4. Go - popular for cloud infrastructure. 5. TypeScript - JavaScript with type safety.",

    # 长篇分析
    "Climate change is one of the most pressing issues of our time. The Earth's average temperature "
    "has risen by approximately 1.1 degrees Celsius since pre-industrial times, driven primarily "
    "by human activities such as burning fossil fuels and deforestation. The consequences include "
    "rising sea levels, more frequent extreme weather events, and disruptions to ecosystems.",
]


def _extract_kv(past_kv):
    """
    从 past_key_values 取出每层 (K, V), 返回 (K_list, V_list, path_str).
    每个张量已 squeeze batch 维 + 转 cpu, shape [num_kv_heads, seq_len, head_dim].
    分层兼容多个 transformers 版本, 并返回实际命中的路径作为证据.
    """
    K_list, V_list = [], []

    # A) 新版 (>=4.54) DynamicCache: .layers, 每层 CacheLayer(.keys / .values)
    if hasattr(past_kv, "layers"):
        for layer in past_kv.layers:
            k = getattr(layer, "keys", None)
            v = getattr(layer, "values", None)
            if k is None or v is None:  # 属性名若变, 直接暴露真实结构而非静默猜
                raise AttributeError(f"未知 CacheLayer 结构: {dir(layer)}")
            K_list.append(k.squeeze(0).cpu())
            V_list.append(v.squeeze(0).cpu())
        return K_list, V_list, "layers[*].keys/values"

    # B) 通用兜底: to_legacy_cache() -> ((K, V), ...)
    if hasattr(past_kv, "to_legacy_cache"):
        for k, v in past_kv.to_legacy_cache():
            K_list.append(k.squeeze(0).cpu())
            V_list.append(v.squeeze(0).cpu())
        return K_list, V_list, "to_legacy_cache()"

    # C) 中期 DynamicCache: .key_cache / .value_cache
    if hasattr(past_kv, "key_cache"):
        for k, v in zip(past_kv.key_cache, past_kv.value_cache):
            K_list.append(k.squeeze(0).cpu())
            V_list.append(v.squeeze(0).cpu())
        return K_list, V_list, "key_cache/value_cache"

    raise RuntimeError(f"无法识别的 cache 类型: {type(past_kv)}")


@torch.no_grad()
def collect_kvcache_for_prompt(model, tokenizer, prompt: str) -> dict:
    """
    跑一次 prefill, 拿到 KV cache.

    Returns:
        {
            "prompt": str,
            "seq_len": int,
            "cache_type": str,    # cache 对象类名 (证据)
            "cache_path": str,    # _extract_kv 命中的兼容路径 (证据)
            "K_per_layer": [tensor [num_kv_heads, seq_len, head_dim], ...],
            "V_per_layer": [...],
        }
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    seq_len = inputs.input_ids.shape[1]

    # 关键: use_cache=True 让 HF 自动保存 KV cache
    outputs = model(**inputs, use_cache=True, return_dict=True)
    past_kv = outputs.past_key_values

    K_per_layer, V_per_layer, cache_path = _extract_kv(past_kv)

    return {
        "prompt": prompt[:80] + ("..." if len(prompt) > 80 else ""),
        "seq_len": seq_len,
        "cache_type": type(past_kv).__name__,
        "cache_path": cache_path,
        "K_per_layer": K_per_layer,
        "V_per_layer": V_per_layer,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Qwen3-0.6B")
    parser.add_argument("--output", default="data/kvcache_dump.pt")
    parser.add_argument("--n-prompts", type=int, default=len(PROMPTS),
                        help=f"用多少个 prompts (max {len(PROMPTS)})")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"Model:  {args.model}")
    print(f"Output: {args.output}")
    print()

    # ----- 加载模型 -----
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    config = model.config
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    print(f"\nArchitecture:")
    print(f"  num_hidden_layers:    {config.num_hidden_layers}")
    print(f"  num_kv_heads:         {config.num_key_value_heads}")
    print(f"  head_dim:             {head_dim}")
    print(f"  hidden_size:          {config.hidden_size}")
    print()

    # ----- 采集 -----
    prompts_to_use = PROMPTS[:args.n_prompts]
    print(f"Collecting KV cache for {len(prompts_to_use)} prompts...")

    all_data = []
    for i, prompt in enumerate(prompts_to_use):
        t0 = time.time()
        data = collect_kvcache_for_prompt(model, tokenizer, prompt)
        elapsed = time.time() - t0
        if i == 0:
            # 一次性自报 cache 来源, 作为数据可信度的证据
            k0 = data["K_per_layer"][0]
            print(f"  [cache] type={data['cache_type']}, extract path={data['cache_path']}, "
                  f"per-layer K shape={tuple(k0.shape)}")
        print(f"  [{i+1}/{len(prompts_to_use)}] seq_len={data['seq_len']}, "
              f"took {elapsed:.2f}s, '{data['prompt']}'")
        all_data.append(data)

    # ----- 保存 -----
    print()
    print(f"Saving to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "metadata": {
            "model_name": args.model,
            "num_layers": config.num_hidden_layers,
            "num_kv_heads": config.num_key_value_heads,
            "head_dim": head_dim,
            "hidden_size": config.hidden_size,
            "dtype": "fp16",
            "n_prompts": len(prompts_to_use),
            "cache_type": all_data[0]["cache_type"],
            "cache_path": all_data[0]["cache_path"],
            "prompts": [d["prompt"] for d in all_data],
            "seq_lens": [d["seq_len"] for d in all_data],
        },
        "data": all_data,
    }
    torch.save(save_dict, args.output)

    # 文件大小
    size_mb = output_path.stat().st_size / 1e6
    print(f"  Saved: {size_mb:.1f} MB")

    # ----- 统计 -----
    print()
    print("Summary:")
    total_tokens = sum(d["seq_len"] for d in all_data)
    print(f"  Total prompts: {len(all_data)}")
    print(f"  Total tokens:  {total_tokens}")
    print(f"  Avg seq_len:   {total_tokens/len(all_data):.1f}")
    print(f"  Layers x KV heads x head_dim: "
          f"{config.num_hidden_layers} x {config.num_key_value_heads} x {head_dim}")
    print()
    print(f"Done. Next step: python analysis/analyze_distribution.py --input {args.output}")


if __name__ == "__main__":
    main()