"""
analyze_distribution.py - Step 1 分析: 数据驱动地选 KV cache 量化维度.

对 K、V 分别模拟 int8 对称量化, 比较 4 种粒度的保真度 (SQNR dB):
    per-tensor / per-head / per-token / per-channel
并给出 K/V 的离群结构证据 (channel 集中度), 最后输出量化维度建议.

为什么对称量化: 对齐项目一 W8A8 的方案 (scale only, 无 zero-point),
    范围 [-127, 127]. 若 K/V 分布明显偏离 0 (skew 大), 会额外提示 asymmetric 的必要性.

执行:
    python analysis/analyze_distribution.py --input data/kvcache_dump.pt
"""

import argparse
import math
import statistics
from pathlib import Path

import torch

# 粒度 -> absmax 沿哪些轴 reduce (张量布局 [H, S, D] = [head, token, channel])
GRANULARITIES = {
    "per-tensor":  (0, 1, 2),   # 1 个 scale
    "per-head":    (1, 2),      # [H]   每 head 一个
    "per-token":   (2,),        # [H,S] 每 (head, token) 一个, 沿 channel 求 absmax
    "per-channel": (1,),        # [H,D] 每 (head, channel) 一个, 沿 token 求 absmax
}


def fake_quant_int8_symmetric(x: torch.Tensor, reduce_dims) -> torch.Tensor:
    """int8 对称量化-反量化. absmax 沿 reduce_dims 求, 其余轴各自独立 scale."""
    m = x.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-8)
    s = m / 127.0
    q = torch.clamp(torch.round(x / s), -127, 127)
    return q * s


def sqnr_db(x: torch.Tensor, xq: torch.Tensor) -> float:
    """signal-to-quantization-noise ratio, 单位 dB. 越高越好."""
    x = x.float()
    sig = x.pow(2).sum().item()
    noise = (x - xq.float()).pow(2).sum().item()
    if noise == 0:
        return float("inf")
    return 10.0 * math.log10(sig / noise)


def channel_concentration(x: torch.Tensor) -> float:
    """channel 幅值集中度 = max(channel absmax) / median(channel absmax).
    高 -> 存在离群 channel (per-channel 量化收益大). x: [H, S, D]."""
    chan_absmax = x.abs().amax(dim=(0, 1))   # [D], 每个 channel 跨所有 head/token 的最大幅值
    mx = chan_absmax.max().item()
    med = chan_absmax.median().item()
    return mx / max(med, 1e-8)


def token_concentration(x: torch.Tensor) -> float:
    """token 幅值集中度 = max(token absmax) / median(token absmax). x: [H, S, D]."""
    tok_absmax = x.abs().amax(dim=(0, 2))    # [S]
    mx = tok_absmax.max().item()
    med = tok_absmax.median().item()
    return mx / max(med, 1e-8)


def skew_ratio(x: torch.Tensor) -> float:
    """|mean| / std. 接近 0 -> 关于 0 对称 (对称量化够用); 偏大 -> asymmetric 更优."""
    return (x.float().mean().abs() / x.float().std().clamp(min=1e-8)).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/kvcache_dump.pt")
    parser.add_argument("--report", default="data/distribution_report.txt")
    args = parser.parse_args()

    blob = torch.load(args.input, weights_only=False)
    meta = blob["metadata"]
    data = blob["data"]
    num_layers = meta["num_layers"]

    print(f"Loaded {args.input}")
    print(f"  model={meta['model_name']}  layers={num_layers}  "
          f"kv_heads={meta['num_kv_heads']}  head_dim={meta['head_dim']}")
    print(f"  prompts={meta['n_prompts']}  seq_lens={meta['seq_lens']}")
    print()

    # 每个粒度累积各层 SQNR (K、V 分开)
    k_sqnr = {g: [] for g in GRANULARITIES}
    v_sqnr = {g: [] for g in GRANULARITIES}
    k_chan_conc, v_chan_conc = [], []
    k_tok_conc, v_tok_conc = [], []
    k_skew, v_skew = [], []

    for L in range(num_layers):
        # 同一层跨所有 prompt 的 token 拼起来 (沿 seq 维), 让 per-channel 统计更稳
        K = torch.cat([d["K_per_layer"][L] for d in data], dim=1).float()  # [H, S_tot, D]
        V = torch.cat([d["V_per_layer"][L] for d in data], dim=1).float()

        for g, rd in GRANULARITIES.items():
            k_sqnr[g].append(sqnr_db(K, fake_quant_int8_symmetric(K, rd)))
            v_sqnr[g].append(sqnr_db(V, fake_quant_int8_symmetric(V, rd)))

        k_chan_conc.append(channel_concentration(K))
        v_chan_conc.append(channel_concentration(V))
        k_tok_conc.append(token_concentration(K))
        v_tok_conc.append(token_concentration(V))
        k_skew.append(skew_ratio(K))
        v_skew.append(skew_ratio(V))

    def fmt_table(name, sqnr):
        lines = [f"\n=== {name}: SQNR (dB), 跨 {num_layers} 层 mean +/- std (越高越好) ==="]
        lines.append(f"  {'granularity':<14}{'mean':>9}{'std':>9}{'#scales/layer':>16}")
        scales = {"per-tensor": "1", "per-head": "H=8",
                  "per-token": "H*S", "per-channel": "H*128"}
        for g in GRANULARITIES:
            mean = statistics.mean(sqnr[g])
            std = statistics.pstdev(sqnr[g])
            lines.append(f"  {g:<14}{mean:>9.2f}{std:>9.2f}{scales[g]:>16}")
        return "\n".join(lines)

    out = []
    out.append(fmt_table("K cache", k_sqnr))
    out.append(fmt_table("V cache", v_sqnr))

    # ---- 离群结构证据 ----
    out.append("\n\n=== 离群结构 (跨层 mean) ===")
    out.append(f"  {'':<22}{'K':>10}{'V':>10}")
    out.append(f"  {'channel 集中度':<18}{statistics.mean(k_chan_conc):>10.2f}"
               f"{statistics.mean(v_chan_conc):>10.2f}   (max/median channel absmax; 高=有离群channel)")
    out.append(f"  {'token 集中度':<19}{statistics.mean(k_tok_conc):>10.2f}"
               f"{statistics.mean(v_tok_conc):>10.2f}   (max/median token absmax)")
    out.append(f"  {'skew |mean|/std':<18}{statistics.mean(k_skew):>10.3f}"
               f"{statistics.mean(v_skew):>10.3f}   (高=偏离0, asymmetric 更优)")

    # ---- 数据驱动的结论 ----
    def best(sqnr, candidates):
        return max(candidates, key=lambda g: statistics.mean(sqnr[g]))

    k_best = best(k_sqnr, ["per-token", "per-channel"])
    v_best = best(v_sqnr, ["per-token", "per-channel"])
    k_gap = statistics.mean(k_sqnr["per-channel"]) - statistics.mean(k_sqnr["per-token"])
    v_gap = statistics.mean(v_sqnr["per-token"]) - statistics.mean(v_sqnr["per-channel"])
    k_head_gap = statistics.mean(k_sqnr[k_best]) - statistics.mean(k_sqnr["per-head"])
    v_head_gap = statistics.mean(v_sqnr[v_best]) - statistics.mean(v_sqnr["per-head"])

    out.append("\n\n=== 结论 (数据驱动) ===")
    out.append(f"  K cache 最佳细粒度: {k_best}  "
               f"(per-channel 比 per-token 高 {k_gap:+.2f} dB)")
    out.append(f"  V cache 最佳细粒度: {v_best}  "
               f"(per-token 比 per-channel 高 {v_gap:+.2f} dB)")
    kivi_k = "符合" if k_best == "per-channel" else "不符合"
    kivi_v = "符合" if v_best == "per-token" else "不符合"
    out.append(f"  vs KIVI 共识 (K per-channel, V per-token): K {kivi_k}, V {kivi_v}")
    out.append(f"  若工程上用 per-head (scale 能干净外提出 matmul):")
    out.append(f"    K 相比最佳损失 {k_head_gap:.2f} dB, V 损失 {v_head_gap:.2f} dB")
    out.append(f"    -> 这就是 Step 2 算子 '用 per-head 换可实现性' 牺牲掉的精度, 可量化、可辩护")

    report = "\n".join(out)
    print(report)

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report)
    print(f"\n\nReport saved: {args.report}")


if __name__ == "__main__":
    main()