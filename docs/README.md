项目：INT8 KV Cache 量化方案 (面向国产芯片无 FP8 算力场景)

主导 INT8 KV cache 端到端量化方案在 Qwen3 上的实现：
- 完成 KV cache 分布采集与分析，对比 perTensor / perToken / perHead / 
  perChannel 四种量化维度方案的精度-性能 trade-off
- 基于 PyTorch 实现量化版 PageAttention 算子 (Q/K/V 三处量化点 + 两次
  反量化 matmul + softmax), 验证算子等价性
- 在 Qwen3-0.6B 集成动态量化方案, 端到端 PPL 偏差 < X%
- 提出静态量化优化方案 (prefill/decode 阶段分离 + 量化-写入 cache 融合
  算子) 设计思路


─────────────── 项目 1: W8A8 GEMM (你已完成的) ───────────────
基于 Triton 实现 W8A8 GEMM kernel, 在 RTX 5090 上达到 2.17x cuBLAS fp16. 
集成到 Qwen3-0.6B, PPL 1.032x. 通过 nsys/ncu 完成完整 profile, 验证 sm_120 
上 Triton 编译路径瓶颈.

─────────────── 项目 2: INT8 KV Cache (即将做的) ───────────────
[面向国产芯片无 FP8 场景] 设计 INT8 KV cache 量化方案, KV cache 分布分析 → 
量化维度选取 → PyTorch PageAttention 算子实现 → Qwen 模型动态量化集成. 
精度损失 X%, decode 阶段显存节省 ~50%.

─────────────── 组合价值 ───────────────
两个项目覆盖 "weight + activation + cache" 完整 W8A8C8 量化栈, 
对应业界 SmoothQuant + KIVI 两条技术路线.