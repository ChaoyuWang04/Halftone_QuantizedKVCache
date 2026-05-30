"""
diagnose_layer0.py - 排查 L=0 量化误差异常 (25.86%) 的根因.

假设: L=0 是 token 离群主导 (attention sink / massive activation),
      与 per-channel K 的设计目标 (channel 离群) 相反, 故 per-channel K 在 L=0 失效.
验三件事:
  1. 各层 K/V 的 channel 集中度 vs token 集中度 (看 L=0 是否 token 离群主导)
  2. L=0 每 token 的 K 范数 (离群 token 是不是第一个 = 经典 sink)
  3. L=0 误差来自 K / V / q 哪个 (只量化一个, 其余 fp16)
"""

import torch

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "reference"))

from page_attention_reference import page_attention_reference
from page_attention_int8_pytorch import quantize_int8, dequantize_int8, _rel_err

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16
n_rep = 2

blob = torch.load("data/kvcache_dump.pt", weights_only=False)
num_layers = blob["metadata"]["num_layers"]


def conc(x, axis):
    """集中度 = max/median. x: [H,S,D]. axis='channel' 看 [D] 分布, 'token' 看 [S]."""
    a = x.abs().amax(dim=(0, 1)) if axis == "channel" else x.abs().amax(dim=(0, 2))
    return (a.max() / a.median().clamp(min=1e-8)).item()


print("=== 1) 各层离群结构: channel 集中度 vs token 集中度 ===")
print(f"  {'layer':<8}{'K chan':>9}{'K tok':>9}{'V chan':>9}{'V tok':>9}")
for L in [0, 1, 2, num_layers // 2, num_layers - 1]:
    k = blob["data"][0]["K_per_layer"][L].float()
    v = blob["data"][0]["V_per_layer"][L].float()
    print(f"  L={L:<6}{conc(k,'channel'):>9.2f}{conc(k,'token'):>9.2f}"
          f"{conc(v,'channel'):>9.2f}{conc(v,'token'):>9.2f}")

print("\n=== 2) L=0 每 token 的 K 范数 (找 sink token) ===")
k0 = blob["data"][0]["K_per_layer"][0].float()    # [H, S, D]
tok_norm = k0.norm(dim=(0, 2))                     # 每 token 跨 head/channel 的范数 -> [S]
med = tok_norm.median().item()
top = torch.topk(tok_norm, k=5)
print(f"  median token norm = {med:.2f}")
print("  top-5 (idx, norm, x median): "
      + ", ".join(f"({i.item()}, {n.item():.1f}, {n.item()/med:.1f}x)"
                  for n, i in zip(top.values, top.indices)))

print("\n=== 3) L=0 误差来源定位 (只量化一个, 其余 fp16) ===")
k = blob["data"][0]["K_per_layer"][0].to(device, dtype)
v = blob["data"][0]["V_per_layer"][0].to(device, dtype)
q = torch.randn(k.shape[0] * n_rep, 1, k.shape[2], device=device, dtype=dtype) * k.std()
ref = page_attention_reference(q, k, v)

def dq(x, dims):
    qi, s = quantize_int8(x, dims)
    return dequantize_int8(qi, s).to(dtype)

print(f"  only-K (per-channel): rel_err={_rel_err(ref, page_attention_reference(q, dq(k,(1,)), v))*100:.2f}%")
print(f"  only-V (per-token):   rel_err={_rel_err(ref, page_attention_reference(q, k, dq(v,(2,))))*100:.2f}%")
print(f"  only-q (per-tensor):  rel_err={_rel_err(ref, page_attention_reference(dq(q,(0,1,2)), k, v))*100:.2f}%")

print(f"\n  [验证假设] L=0 的 K 改 per-token: "
      f"rel_err={_rel_err(ref, page_attention_reference(q, dq(k,(2,)), v))*100:.2f}%")
print("  若 per-token K 明显优于 per-channel K -> 坐实 L=0 是 token 离群主导")