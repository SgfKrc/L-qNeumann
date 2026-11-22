# engine/ —— L-qNeumann 推理引擎

> 状态：**设计稿（M0）**
> 更新日期：2026-11-21
>
> 适用范围：ggml 电路执行器的设计与路线；读者为实现者。
> 相关：立项 [PROPOSAL](../docs/design/PROPOSAL.md)、边界 [ENGINE-LIMITS](../docs/limits/ENGINE-LIMITS.md)、验收 [verify](../verify/README.md)。

---


> 选题 1 的主战场：**在 llama.cpp 里实现 chip 电路的执行器**，把 chip 步从 **204.5 μs** 压到 **100 μs 以内**。

## 一、一个已经用 27,825 行代码买来的教训

**不要跨框架。**

K-Llama v1 做过「GPU 段 PyTorch / CPU 段 llama.cpp」的跨框架接力。它的结局记录在
`docs/design/EXLLAMAV3-GPU段候选.md`：

> v1 的跨框架接力（GPU 段 PyTorch / CPU 段 llama.cpp）**不是技术不可行，是产品决策放弃的**。
> F12 多模态/跨框架/PyTorch = **27,825 行（21.5%）**，处置**抛弃**。

**为什么数值差消除不了**：Triton 和 ggml 是两套完全不同的数学实现——归约顺序、中间精度、
atomic 累加时机都不同。即使权重逐字节相同、都用 bf16+fp32 累加，两边的结果也会在最后几位不同。

**而 chip 电路对数值极其敏感**：pinned-constant 契约要求位精确恒等，bf16 ballast（2^40）
一旦被舍入到相邻值，电路就"看不见"那条线了。**跨框架的微小差异会直接毁掉电路的正确性**，
不是"精度差一点"，是"算错"。

**所以：引擎必须用单一实现——llama.cpp + ggml。**

## 一.5 但「单一实现」比「移植」更彻底：**放弃 PyTorch/Triton 生态**

这是第二个 K-Llama 的坑，比数值差更根本：

**单机性能上，safetensors 永远赢不过 ggml。**

| 维度 | safetensors + PyTorch | GGUF + ggml |
|---|---|---|
| **加载** | 全量进 RAM，torch 管张量 | **mmap**，按需缺页 |
| **显存** | 显式 `to(device)`，全量 | 按层 offload，可部分 |
| **内存峰值** | 权重 + activation + 碎片 | 权重 mmap + 计算缓冲 |
| **启动** | 秒级加载 22.87 GiB | 毫秒级 mmap |

**22.87 GiB 的 safetensors 在这台 8 GB 卡 / 31.6 GiB 内存的机器上根本跑不起来**——这不是优化问题，
是生态选择问题。

**所以决策是**：**全 llama.cpp。不保留 Triton，不保留 PyTorch，不保留 safetensors。**

| 环节 | 之前 | 之后 |
|---|---|---|
| 权重格式 | safetensors（22.87 GiB） | **GGUF（14.20 GiB）** |
| 执行器 | Triton（`fusedexec.py`） | **ggml 算子** |
| 状态管理 | torch tensor | **`llama_memory_recurrent`** |
| 验收标准 | 对 Triton 参考值 bit-exact | **对 IR 的 `evaluate`（纯 Python）bit-exact** |
| torchwright | 生产路径的一部分 | **验证工具**（证明状态可精确递推，代码不直接用） |

**注意验收标准的变化**：不再需要 Triton 做「参考实现」——因为 **IR 的 `ir.evaluate` 是纯 Python，
不依赖 torch**，它本来就是 QwenPU 的黄金参考。`test_rv_ref.py`（golden trace）和 `test_rv_core.py`
（IR 对比）都是这套体系里的，**可以直接用来验收 ggml 实现**。

**torchwright 的角色变化**：它的 P2a/P2b/P3 状态机是**为 PyTorch 生态做的**，全 llama.cpp 之后
代码不直接用。但它**验证了「DeltaNet 的矩阵状态能在 bf16 权重里精确递推」这个结论**——
这个结论对 llama.cpp 实现同样成立，而且它是「这件事可行」的构造性证据。**torchwright 从
生产路径降级为验证工具，不是浪费。**

## 二、引擎做什么

把现在 **Triton 实现的 chip 执行器**（`qwenpu/chipformer/rv/fusedexec.py`）**移植成 ggml 算子**，
让 chip 层和 Qwen 层在**同一个 llama.cpp 进程里、同一套数值实现下**跑完。

| 做 | 不做 |
|---|---|
| chip 层的 ggml 前向（one-hot write / gather / decay / ballast） | 跨框架混合（PyTorch ↔ llama.cpp） |
| 状态管理（复用 `llama_memory_recurrent`） | 重写 Qwen 层（用 llama.cpp 原生） |
| bit-exact 验证（对 IR / 对 Triton 参考实现） | 量化（那是 `scripts/` 的事） |
| 性能剖析（逐层时间、访存占比） | 训练（全程无训练） |

## 三、基于哪个 llama.cpp

| 选项 | 优点 | 缺点 | 建议 |
|---|---|---|---|
| **上游 llama.cpp** | 干净、活跃、数值路径无历史包袱 | 没有分段能力 | ★ **首选**——我们不需要分段（chip + Qwen 在同一份 GGUF 里，全量加载） |
| **K-Llama 的 fork** | 有 `layer_inp` 槽位、`split-stage` | 带了大量分段改造，且它的分工是为「多机」设计的 | 不选——它的能力我们用不上 |

**关键判断**：K-Llama 的 `layer_inp` 槽位是给「跨机器传 hidden」用的，而我们要的是
「同一进程内 chip 层 → Qwen 层」，**不需要跨进程，所以不需要分段基础设施**。

## 四、要移植什么（对照表）

现在 `fusedexec.py` 里的 6 个 kernel 变体，逐个对应到 ggml：

| Triton（现在） | 做什么 | ggml 对应 |
|---|---|---|
| `_ffn3` / `_ffn4` | MLP 前向（gate/up/down 三段） | `ggml_mul_mat` + `ggml_silu` + `ggml_mul` |
| `_head3` / `_head4` | head 的 one-hot 写 + top-2 读 | `ggml_get_rows` + `ggml_set_rows`（需确认上游有 set_rows） |
| `_head_scan` | 128 行分块扫描（拆 CTA 后） | 多个 `ggml_get_rows` + `ggml_argmax` |
| `_head_pick` | 跨块归并 argmax | `ggml_argmax` + 归并 |
| `_decode_read` | 从 code dim 取 bit | 需新写（`ggml` 没有直接对应） |
| `_book` | 记账（`r_i = X[i%2] + Y[(i-1)%3]`） | `ggml_add` |

**新增的部分**：
- **`ballast` 的 bf16 精确写入**（2^40 标记）——ggml 的 `ggml_set_rows` 是否保精度要验
- **`decay` 的 counter 语义**（只对计数器衰减）——可能 `llama_memory_recurrent` 已有
- **状态布局映射**：Triton 的 `(head, 128 行, 128 位)` vs `llama_memory_recurrent` 的布局

## 五、工程量与风险

| 阶段 | 内容 | 估计 |
|---|---|---|
| **M1. 状态复用** | 确认 `llama_memory_recurrent` 能装下 chip 的状态（**266** value head × 128 行） | 1–2 周 |
| **M2. 单层前向** | 在 ggml 里实现一层 chip 的前向，对 Triton 参考值 bit-exact | 2–3 周 |
| **M3. 13 层串起来** | 整条 chip 链在 llama.cpp 里跑通，输出与 IR 一致 | 2–3 周 |
| **M4. 性能** | 优化到 100 μs 以内 | 3–4 周 |
| **M5. 端到端** | chip + Qwen 一起跑（同一进程、同一份 GGUF） | 2 周 |

**总 3–4 个月。**

**风险**：

1. **精度**：ggml 的 bf16 累加是否与 Triton 的 fp32 累加一致——这是 M2 的核心验证点
2. **性能**：ggml 的 kernel 可能不如 Triton 快（Triton 是专门为这批 kernel 写的）
3. **状态布局**：`llama_memory_recurrent` 原本是给 GDN 设计的，chip 的 266 head 是否能装下
4. **`ggml_set_rows`**：one-hot key write 需要 scatter 语义，要看上游有没有、精度如何

## 六、为什么不能从零，也不能跨框架

| 路线 | 为什么不行 |
|---|---|
| **从零写引擎** | 要重走 P2a/P2b/P3/O3-2/O5 全部（GDN 数值等价、状态时序、bf16 契约、词表映射） |
| **跨框架（Triton + llama.cpp）** | v1 用 27,825 行证明过：数值差消除不了，电路会算错 |
| **Triton 一路到底** | 22.87 GiB 装不下，不能部署 |
| **★ llama.cpp 单一实现** | **唯一同时满足「能部署」和「bit-exact」的路** |

## 七、`fusedexec.py` 的定位变化

它**不再是参考实现**（验收标准换成了 IR 的 `evaluate`）。剩余价值：

- **性能基线**：204.5 μs 是目标线（ggml 版要打到这个数以下）
- **优化思路**：head 拆 CTA、load 交织、per-layer KS 的**思路**可移植（实现完全不同）
- **反例价值**：它证明了「Triton 能跑，但 22.87 GiB 装不进这台机器」——这正是放弃它的理由

**移植的正确性判据**：ggml 版对 **`ir.evaluate`** 逐位相同，不是"差不多"。

## 八、验收

| # | 判据 | 当前 |
|---|---|---|
| 1 | chip 单层前向对 Triton 参考值 bit-exact | 未做 |
| 2 | 13 层链在 llama.cpp 里跑通，输出与 IR 一致 | 未做 |
| 3 | 完整 46 层推理能跑（chip + Qwen 同进程） | **从未跑过** |
| 4 | chip 步 ≤ 150 μs | Triton 版 204.5 μs（参考） |

**第 3 条是真正要补的洞**——现有 chip 性能数字都是「chip 层单独」测的，完整推理从未在这台机器上跑过。
