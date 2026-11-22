# _reference/ —— 归档的 Triton 实现

> 状态：**归档**
> 更新日期：2026-11-21
>
> 适用范围：已归档的 Triton 实现，仅供性能基线与优化思路参考；不是生产路径。
> 相关：基线 [BASELINES](../docs/measure/BASELINES.md)。

---


> **不是生产路径。** 保留它是为了性能基线和优化思路，不是为了让别人 import 它。

## 一、为什么归档

全 llama.cpp 之后，Triton 执行器不再参与推理。但它有三个方面价值：

| 价值 | 用途 |
|---|---|
| **性能基线** | chip 步 204.5 μs 是目标线（ggml 版要打到这个数以下） |
| **优化思路** | head 拆 CTA、load 交织、per-layer KS 的**思路**可移植 |
| **反例** | 它证明了「Triton 能跑，但 22.87 GiB 装不进这台机器」 |

## 二、这里是什么

| 文件 | 做什么 | 关键数字 |
|---|---|---|
| `fusedexec.py` | `FreeRun3` —— chip 执行器（Triton） | 204.5 μs/token（B=1） |
| `tritonexec.py` | `Executor2` —— 上一代执行器 | — |
| `fastexec.py` | `FreeRun` —— CUDA graph 路径 | — |
| `rvbatch.py` | 批量解码 | B=8 → 841.1 μs |
| `chat.py` | chat 路径（chip + Qwen 串接） | — |

## 三、为什么不能直接用（三条证据）

### 1. 装不下

safetensors 22.87 GiB，8 GB 显存 / 31.6 GiB 内存。**完整模型跑不起来**——之前所有性能数字都是「chip 层单独」测的。

### 2. 数值差

Triton 与 ggml 是两套数学实现（归约顺序、中间精度、atomic 时机都不同）。电路对数值是**位级敏感**的，跨框架的微小差异会直接算错。

### 3. K-Llama v1 的教训

> 跨框架接力 = **27,825 行（全仓 21.5%）**，处置**抛弃**。不是技术不可行，是产品决策放弃的。

## 四、能复用什么

| 可复用 | 怎么用 |
|---|---|
| **head 拆 CTA + 跨 CTA 归并** | ggml 里可以用多个 `ggml_get_rows` + `ggml_argmax` 实现类似效果 |
| **load 交织**（X、Y 按 `lcm(2,3)=6` 交织成 8 B） | 内存布局思路，与实现无关 |
| **per-layer KS**（每层选最优拆分度） | 同样的调优逻辑 |
| **`tune()` 的教训** | **必须用 CUDA graph 计时**——eager 循环会把两次 launch 的版本误判 |

## 五、不要做什么

- **不要 import 它**——它是归档，不是库
- **不要拿它当参考实现**——验收标准是 `circuit/q35/ir.py` 的 `evaluate`（纯 Python）
- **不要试图桥接它**——那是 27,825 行的路

## 六、许可

同样来自 [QwenPU](https://github.com/divagr18/QwenPU)（上游）。本项目在它之上做了
chip token 连续性、head 拆 CTA、record 压缩等改动，这些改动的**结果**（204.5 μs、bit-exact）
已在 `docs/measurements/BASELINES.md` 里记录。
