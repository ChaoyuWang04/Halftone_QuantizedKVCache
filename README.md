# QuantizedKVCache — INT8 KV Cache 量化(Qwen3-0.6B)

> 个人项目笔记 / 复盘。从"为什么量化"一路做到"亲手写 kernel 摸到带宽红利"。
> 硬件:RTX 5090 (sm_120) · CUDA 13 · Triton 3.6 · PyTorch 2.11 · uv

---

## 一句话总结

把 Qwen3-0.6B 的 KV cache 用 int8 量化,**精度几乎无损**(PPL +0.2% ~ +1.2%),**显存减半**(0.54×→长序列趋近 0.50×),并用 Triton 写出 fused int8 decode attention kernel,**实测带宽红利 1.56×**(理论上界 2×,差距已定位到占用率)。

| 维度 | 结果 |
|---|---|
| 精度(KV int8,q fp16) | PPL 24.68 → 24.73,**1.002×** |
| 精度(q+K+V 全 int8) | PPL 24.68 → 24.98,**1.012×** |
| 显存(cache int8/fp16) | **0.538×**(短序列),随上下文趋近 **0.50×** |
| 速度(decode attn,S=4096) | int8 vs fp16 kernel **1.56×**(理论 2×) |

量化维度(数据驱动选出):**K per-channel · V per-token · q per-token · 对称 int8**。

---

## 我做了什么

### Step 1 — 分布分析,数据驱动地选量化维度
- `collect_kvcache.py`:加载真模型,用 8 个多样 prompt 跑 prefill,采集**真实** K/V。
- `analyze_distribution.py`:对 K/V 分别测 per-tensor / per-head / per-token / per-channel 的 SQNR。

| 粒度 | K SQNR (dB) | V SQNR (dB) |
|---|---|---|
| per-tensor | 29.18 | 31.51 |
| per-head | 31.14 | 34.38 |
| per-token | 35.64 | **42.04** |
| per-channel | **46.17** | 41.49 |

- K 选 **per-channel**(领先 per-token +10.5 dB,channel 集中度 13.6× → 存在离群 channel)。
- V 选 **per-token**(只领先 per-channel +0.54 dB,基本打平;V 的 token 集中度 3.40 > channel 2.44)。
- skew 0.015 / 0.008 → 分布对称,**对称 int8 够用**,不需要 asymmetric。
- 复现了 KIVI 的共识(K per-channel,V per-token)。

### Step 2 — PyTorch 算子 + L=0 侦探案
- `page_attention_reference.py`:fp16 decode attention,**自测对拍 torch SDPA(64.5/61.8 dB)**,确立可信 ground truth。
- `page_attention_int8_pytorch.py`:int8 版,**dequant-then-matmul**(原因见下方"对偶")。
- 结果:中后层极好(L=14 0.61%,L=27 1.46%),但 **L=0 高达 25.86%**。
- `diagnose_layer0.py` 排查:我的"L=0 是 token 离群(attention sink)"假设**被证伪**——
  - L=0 的 K 是极端**channel** 离群(集中度 91.91,全模型最凶),per-channel 反而最该用;per-token K 会烂到 50%。
  - 没有任何爆表 token(最大也才 1.4× 中位数)。
  - 真凶是 **q**(only-q 17% > only-K 7% > only-V 1%),而当时的 **q 是合成的随机数**。

### Step 3 — 真模型验精度
- `qwen_kvcache_test.py`:monkeypatch `eager_attention_forward`,真模型跑 PPL。
- 用**真 q** 后 L=0 从 25.86% 掉到 **2.55%** → 坐实之前是合成 q 的锅。
- PPL:KV-only **1.002×**,q+K+V 全压 **1.012×**,近乎无损。

### Phase A — 物理 int8 cache + q 粒度 A/B(`qwen_kvcache_full.py`)
- `QuantizedKVCache`(HF DynamicCache 子类):K/V **物理上以 int8 存储**,读时反量化。
- q 三粒度真模型对比:per-tensor 1.0104 / per-token 1.0096 / per-channel 1.0018 —— **全在 1% 内,q 粒度不是有意义的杠杆**。
- cache 体积 int8/fp16 ≈ **0.538**(短序列偏高是 per-channel K scale 的定长开销,长序列趋近 0.50)。

### Phase B — fused 量化 Triton kernel(`quant_kv_triton.py`)
- 把 findmax + 量化 + 写回融合进一个 kernel。
- V per-token 稳定 **2.8–3.3×**;K per-channel 在 S=4096 **退化到 0.45×(反而慢一半)**。
- 暴露的是**访存合并**问题(见下)。

### Phase C — fused int8 decode attention kernel(`int8_decode_attn_triton.py`)
- flash-style online softmax,**读 int8、register 反量化、fp32 计算**。
- 正确性 0.02%(精确)。速度 int8 vs fp16:S=64 0.87× / S=512 1.50× / S=4096 **1.56×**。
- 没到 2×,定位到占用率(见下)。

---

## 我学到了什么(技术)

**SQNR + mean/std 的读法。** SQNR = $10\log_{10}(\sum x^2 / \sum(x-\hat x)^2)$,每 +6 dB ≈ 多 1 bit。**mean 决定选哪个,std 决定多敢信这个选择**;真正判断"两个粒度差异是否真实"看的是 **gap vs std**(K 的 10.5 dB ≫ std → 铁;V 的 0.54 dB < std → 基本打平)。

**"恶霸 + 尺子"是量化的第一性原理。** int8 只有 255 格,尺子量程被组内最大值定死;组里混进离群值 → 尺子变粗 → 小数全遭殃。所以**沿离群值聚集的轴切 scale**:K 离群在 channel → per-channel;V 波动在 token → per-token。

**K/V 的对偶 = 为什么 dequant-then-matmul。** $QK^\top$ 沿 channel 求和、$PV$ 沿 token 求和;K 的 per-channel scale 和 V 的 per-token scale **恰好都落在各自的求和轴上**,无法外提,所以必须先反量化再算。这两者"精度最优粒度 = kernel 最敌视粒度"——正是 KIVI 值得成为论文的原因。

**访存合并(coalescing)。** Phase B 的 K kernel 沿 strided 的 S 轴归约,相邻线程地址差 256B → 不合并 → ~16× 带宽浪费;V kernel 沿连续的 D 轴归约 → 合并。小 S 时被启动开销掩盖,大 S 暴露。**教训:永远沿连续维 stream,2D 瓦片让最内维连续。**

**Roofline:decode 是 memory-bound。** 1000-token 时 KV cache 112MB,读它要 ~63μs,而 attention 算力只要 ~1μs(差 60×)。所以 int8 省的是**访存**,反量化那点计算几乎免费,且**上下文越长红利越大**。

**占用率(occupancy)是另一道墙。** Phase C 只 1.56× 不是 2×,因为 grid 只有 `batch(1)×heads(16)=16` 个 program,而 5090 有 ~170 SM,有效带宽不到峰值 5% → **不是 bandwidth-bound,是 occupancy-bound**。反解:固定开销 $R≈47\mu s$(fp32 softmax)不随 int8 缩小,把 2× 稀释成 1.56×。

---

## 我意识到了什么(更重要的元认知)

1. **反常数据先怀疑 bug,不降标准。** L=0 的 25% 没让我去松 10% 的阈值,而是去查——最后发现是合成 q 的伪影。坚持这条纪律救了结论。

2. **假设错了没关系,诊断方法对就行。** 我赌"token 离群 / sink",数据直接打脸(L=0 反而是史上最强 channel 离群)。但**证伪驱动的诊断脚本**照样把真凶(q)揪出来了。猜错是常态,可证伪的流程才是底气。

3. **真实 vs 合成要分得清。** K/V 是真的(模型真跑出来的),**q 是合成的——因为 KV cache 顾名思义不存 q**。这个区分一开始没讲清,差点让我把"假 q 的 25%"当成真问题。

4. **量化 q 是算力侧需求,不是存储侧。** q 不进 cache,在"反量化再算"的路径里量化它**纯亏无赚**。它只在要喂 int8 tensor core(int8×int8 矩阵乘)时才必需。想清楚"省显存只需量 K/V"这件事,避免了无谓的纠结。

5. **理论上界和实测下界的 gap 要能解释。** 1.56× 不是失败,是"未把自己推到 memory-bound 屋脊上"的中间态;能反解出固定开销 R、能说清是占用率而非带宽,才算真懂这个 kernel。

6. **诚实分层是可辩护性的来源。** 我做的是"int8 KV 量化的**方案设计 + 精度验证 + 性能摸底**";物理 fused kernel 的极致优化、更大模型的能力测试还没做。把"做了什么/没做什么"标清楚,比虚报一个"72B 10%"强得多。

---

## 还能进一步动手做什么

**性能(把 1.56× 推向 2×)**
- [ ] **Flash-Decoding / split-K**:把 S 切段并行,grid 从 16 暴涨,填满 SM,逼近 bandwidth-bound。
- [ ] **batched benchmark**:验证 batch=32 时占用率自然上去、不靠 split-K 也接近 2×。
- [ ] **修 Phase B 的 K-quant kernel**:2D tiling 或先转置,消除非合并访存。

**量化方案完整化**
- [ ] **静态量化**:用 Step 1 的 calibration 分布预存 scale(K/V 静态、q 动态),省掉每步 findmax。
- [ ] **block-wise per-channel**:解决 decode 单 token 时 per-channel K 退化的问题(真实 PagedAttention 的做法)。
- [ ] **int8 tensor-core prefill kernel**:prefill 是 compute-bound,这里 int8 矩阵乘才真有意义(和 decode 的带宽红利互补)。

**可信度 / 泛化**
- [ ] **更大模型**(Qwen3-4B/7B,5090 32GB 跑得动):验证"近乎无损"在规模上是否成立。
- [ ] **真任务 benchmark**(不只 PPL):0.6B 上跑 HellaSwag/ARC,或更大模型上跑 GSM8K。
- [ ] **int4 探索**:4× 压缩 vs 精度损失的权衡点在哪。
