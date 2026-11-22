# verify/ —— 验收装置

> 状态：**设计稿**
> 更新日期：2026-11-21
>
> 适用范围：对 IR 的 bit-exact 验收装置；读者为实现者。
> 相关：参考 [circuit](../circuit/README.md)、基线 [BASELINES](../docs/measure/BASELINES.md)。

---


> **ggml 版对 IR 的 `evaluate` 逐位一致，不是「差不多」。**

## 一、验收三层

| 层 | 验什么 | 怎么验 | 门槛 |
|---|---|---|---|
| **L1 单层** | 一层 chip 的前向 | ggml 输出 vs `ir.evaluate` 的对应 dim | 逐位相同 |
| **L2 电路链** | 13 层串起来 | ggml 跑一串 token vs `ref.py` 的 golden trace | 逐位相同 |
| **L3 完整推理** | chip + Qwen 46 层 | ggml 输出 logits vs 参考 | 逐位相同 |

## 二、已有的黄金参考（可直接复用）

| 资源 | 在哪 | 用途 |
|---|---|---|
| **`ir.evaluate`** | `circuit/q35/ir.py` | **L1/L2 的验收标准**（numpy float64，精确） |
| **`ref.py`** | `circuit/rv/ref.py` | golden trace 生成（Python 参考机） |
| **`test_rv_ref.py`** | QwenPU 仓库（130 个测试） | L2 的现成验收 |
| **`test_rv_core.py`** | QwenPU 仓库（94 个测试） | L2 的 IR 对比 |
| **`corpus()`** | `circuit/rv/corpus.py` | overlay 的精确性测量 |

**关键**：`ir.evaluate` 不依赖 torch，所以它可以验收 ggml 实现——**不会出现「用被验收的东西验收自己」**。

## 三、验收判据（必须写死）

| 项 | 判据 |
|---|---|
| **位精确** | 逐元素 `a == b`（不是 `abs(a-b) < eps`） |
| **范围** | L1：单层所有输出 dim；L2：整串 token 的所有 dim；L3：logits |
| **样本量** | L2 至少 4,600,000 个值（参照 Triton 版的验证规模） |
| **失败处理** | 报**第一个**不同的 (token, dim) 和两边的值 |

**为什么是逐位而不是容差**：电路的 pinned-constant 契约要求位精确恒等。**一个 ULP 的差就可能让电路读错一条线。**

## 四、还没建的部分

这个目录目前只有说明。**验收装置要在 M1–M3 阶段建**：

| 阶段 | 建什么 |
|---|---|
| M1 | 状态布局映射的检查（`llama_memory_recurrent` 装下 chip 状态） |
| M2 | **L1 单层验收**（最关键——ggml 的 bf16 累加能否对上 IR） |
| M3 | L2 电路链验收（复用 `test_rv_ref.py` 的 golden trace） |
| M5 | L3 完整推理验收 |

## 五、一个诚实的提醒

**M2 是生死线。** 如果 ggml 的 bf16 累加无法与 `ir.evaluate` 的 float64 逐位一致，整个方案要重新评估。

但那不是「方向错了」——是**具体卡在某个算子的精度控制上**。到时候你知道该修哪个算子，而不是「跨框架整体不行」。
