# 方案与计划

> 状态：**现行**
> 更新日期：2026-11-23
>
> 适用范围：本项目的技术路线、阶段划分、验收判据与风险应对；读者为实现者与评审。
> 相关：立项 [PROPOSAL](PROPOSAL.md)、里程碑记录 [MILESTONES](../measure/MILESTONES.md)、
> 引擎设计 [engine/README](../../engine/README.md)、边界 [ENGINE-LIMITS](../limits/ENGINE-LIMITS.md)。

---

## 1. 方案

### 1.1 总路线

**在 llama.cpp 里实现 chip 电路的执行器，全栈单一数值实现。**

```
GGUF（一份文件）
└── llama.cpp 全量跑
    ├── blk.0-13   chip + kernel  ← 新写的 ggml 算子
    └── blk.14-45  Qwen 32 层     ← llama.cpp 原生
```

**没有 hidden 交换，没有框架边界，没有第二套数值实现。**

### 1.2 三个已定的决策

| 决策 | 内容 | 依据 |
|---|---|---|
| **不跨框架** | 不保留 Triton/PyTorch 执行路径 | 跨框架的数值差消除不了；K-Llama v1 用 27,825 行证明过 |
| **不保留 safetensors** | 权重只用 GGUF | 单机装载上 safetensors 赢不过 ggml（全量进 RAM vs mmap） |
| **验收对 IR** | 判据是 `circuit/chipformer/q35/ir.py` 的 `evaluate`（numpy float64） | 纯 Python，不依赖被验收的实现 |

### 1.3 组件职责

| 组件 | 职责 | 状态 |
|---|---|---|
| `circuit/` | 电路定义、黄金参考、权重生成 | 已迁移 |
| `engine/` | ggml 的 chip 执行器（llama.cpp fork 侧） | **M1–M4 已在 fork 里跑通**；`engine/` 只留说明 |
| `verify/` | 对 IR 的 bit-exact 验收 | 判据已固化（见 MILESTONES），脚本待整理 |
| `_reference/` | 归档的 Triton 实现（性能基线） | 已归档 |
| `tools/` | 量化、构建脚本 | 已迁移 |

### 1.4 技术路线

**从「一层」到「整条链」再到「端到端」，每步都以 bit-exact 为门槛。**

1. **先证状态能装下**（M1）——如果 `llama_memory_recurrent` 装不下 chip 的 266 head，后面全白做
2. **再证一层能算对**（M2）——这是唯一可能让方案崩掉的地方
3. **然后才是规模**（M3）和**速度**（M4）
4. **最后才是端到端**（M5）——因为它是前四步的自然结果，不是独立任务

## 2. 计划

| 阶段 | 周 | 任务 | 验收 | 依赖 |
|---|---|---|---|---|
| **M1** | 1–2 | 状态布局映射 | ✅ **已完成** | — |
| **M2** | 3–5 | 单层前向 + 数值对齐 | ✅ **已完成**：13 层全部 bit-exact，0/6965 | M1 |
| **M3** | 6–8 | 13 层链在 llama.cpp 里跑通 | ✅ **已完成**：c-crc32 程序 4165/4165 | M2 |
| **M4** | 9–12 | 性能：基线 + 两端裁剪 + 稀疏执行器 | ✅ **已完成**：dense 26.5 → **3.40 ms**（7.8×）；A+B 两端裁剪 −66%/−56% | M3 |
| **M5** | 13–14 | 端到端：chip + Qwen 同进程跑完整 46 层 | 输出 logits 与参考一致 | M3 |

**总 14 周（约 3.5 个月）。**

### 2.0 里程碑结论（细节见 [MILESTONES](../measure/MILESTONES.md)）

| 里程碑 | 一句话结论 | 判据 |
|---|---|---|
| **M1** | 容器直接用，不改分配代码 | 每层 524,288 = value head ≤32 × 128 × 128，一字不差 |
| **M2** | 13 层全部 bit-exact | **0 / 6965 dims**，3 条轨迹 |
| **M3** | 真实布局的偏差修到 0 | L0 **185 → 0/635**；程序 **4165/4165** |
| **M4** | dense 不可达 150 μs，必须换稀疏执行器 | A+B −66%/−56%；稀疏 kernel 落地 |
| **M4b（A+）** | `blk.13` 进稀疏 | chip 步 **5.11 → 3.40 ms**（−33.5%，相对 dense 7.8×）；bit-exact **0 差异** |
| **M4c（code_token）** | chip token 不走 lm_head | 窄头 **4165/4165**，argmax 与 fullhead **4166/4166** 逐位一致 |

### 2.1 为什么 M2 是生死线

M2 的难点在「**谁适应谁**」，而不在「实现一层前向」：

- `ir.evaluate` 是电路的规范（numpy float64）
- ggml 的数值行为由算子内部决定（归约顺序、累加精度、FMA）
- 两者在最后几位可能不同

**如果不同，不一定是让 ggml 服从 float64**——也可能是电路本身对舍入太敏感，需要改。

| 段 | 做什么 | 产出 |
|---|---|---|
| **M2a** | ✅ level 1 的 ggml(bf16) 与 `ir.evaluate` **0 / 6965 dims 差异** | 偏差表 |
| **M2b** | 偏差影响关键路径时，调整电路使其鲁棒（ballast 降位、值域收紧、2 的幂替代一般整数） | 改后的权重 |
| **M2c** | 验收：改后的电路在 ggml 下对 `ir.evaluate` 逐位一致 | 0 差异 |

**判断标准**：

- 改电路如果只是「调整数值设计让它更鲁棒」→ **改进**（对任何实现都更好，包括 Triton）
- 如果「接受 ggml 的误差、带误差跑」→ **错**（ballast 读不到，电路算错）

**取舍的顺序**：优先改 ggml 的配置（如果它有精度开关），其次改电路的数值设计（小改），最后才考虑接受非关键路径的偏差。

### 2.2 每阶段的产出物

| 阶段 | 代码 | 文档 | 测试 |
|---|---|---|---|
| M1 | 状态映射模块 | 布局说明 | 分配/读写检查 |
| M2 | 一层前向的 ggml 算子 | 算子对照表（Triton ↔ ggml） | 单层 bit-exact |
| M3 | 13 层链 | 执行流程 | golden trace |
| M4 | 性能优化 | 逐层剖析报告 | 性能回归 |
| M5 | 端到端 | 部署指南 | logits 对比 |

## 3. 风险与应对

| # | 风险 | 影响 | 应对 | 状态 |
|---|---|---|---|---|
| **R1** | ggml 的 bf16 累加无法与 `ir.evaluate` 的 float64 逐位一致 | M2 卡死 | M2a 先测偏差分布 | ✅ **未发生**（0/6965） |
| **R2** | `llama_memory_recurrent` 的布局装不下 chip 状态 | M1 卡死 | 自建 state buffer | ✅ **未发生**（1:1） |
| **R3** | ggml 的 kernel 性能不如 Triton | M4 达不到 150 μs | 移植 head 拆 CTA、load 交织 | ⚠️ **成立**：dense 26.5 ms；稀疏后 3.40 ms |
| **R4** | `ggml_set_rows` 不保 bf16 精度 | ballast 写入失真 | 手写 scatter kernel | ✅ **未发生** |
| **R5** | 完整模型在 8 GB 卡上的 offload 调度 | M5 跑不动 | 分段 + 各自全上卡 | ⚠️ **部分成立**：46 层全上卡 7679/8188 MiB（93.8%），能跑但 16.3 tok/s，余量仅 500 MiB |
| **R6** | 初始 recurrent state 是程序的一部分，零 state 跑出另一条轨迹 | 端到端复现不了 | 逐 token decode 前注入 state（`llama_set_ssm_state`） | ✅ **已定位**（见 MILESTONES M4c） |

## 4. 里程碑判据

| 里程碑 | 判据 | 不可退让 |
|---|---|---|
| **M2 通过** | 改后的电路在 ggml 下对 `ir.evaluate` 逐位一致（0 差异） | ✅ **是** |
| **M3 通过** | golden trace 全过（130 个测试） | ✅ **是** |
| **M4 达标** | ~~chip 步 ≤ 150 μs~~ → **dense 不可达**；现行地板 **3.40 ms**（稀疏），融合后可到 ~1 ms | 软目标，已按实测修正 |
| **M5 通过** | 46 层跑通，GPU 与 CPU 两条贪心链的 token 序列逐 token 相同 | ✅ **是**（12/12，16.3 tok/s；跨后端逐位 logits 不成立，见 MILESTONES M5） |

**位精确是硬门槛**（M2/M3/M5），**性能是软目标**（M4）。

**150 μs 这个数要连同口径读**：它是 Triton `FreeRun3` 单实现的数字（见 [BASELINES](../measure/BASELINES.md) §一），
判的是「电路本身能跑多快」；llama.cpp 路的 3.40 ms 含通用引擎的建图/调度/launch 税，
**两者不可直接比**。要把 3.40 ms 推向 1 ms 量级，只能融合 sublayer，不是稀疏存储能解决的。

## 5. 不做什么

| 不做 | 理由 |
|---|---|
| 跨框架（PyTorch ↔ llama.cpp） | 数值差消除不了；K-Llama v1 用 27,825 行证明过 |
| 新硬件设备（AES/SHA/zlib） | token 带宽 15–16 bit 是硬墙 |
| 流水线（IF/ID/EX/MEM/WB） | 分支率 26.5%，实测负收益 |
| persistent kernel | 实测 1.017×，阶段 2 会慢 14% |
| 训练 | **电路全程无训练**（QwenPU 的前提，电路是编译的）；**语言层可选 LoRA 适配**（不碰 0–13 层，见 [BRIDGE-LORA](BRIDGE-LORA.md)） |

## 6. 待评估的方向

| # | 方向 | 收益 | 状态 |
|---|---|---|---|
| 1 | **payload 15 → 16** | tpi −3~5%，地址效率 80.7 → 84.2% | 待评估 |
| 2 | **让「族」可读**（族 id → Qwen embedding） | Qwen 理解程序意图，可能减 text token | 待评估（见 [TOKEN-ENCODING](TOKEN-ENCODING.md) §6） |
| 3 | LoRA 训练桥（`bridge`） | 模型学会用电路结果 | 待评估 |
| 4 | 视觉塔 → 电路 | 看图算数 | 待评估（见 [VISION-TOWER](VISION-TOWER.md)） |
| **5** | **chip 层稀疏存储**（99.96% 为零，5.30 GiB → 8.30 MiB） | 让 chip 步离开「读权重」区 | ✅ **已完成**（M4） |
| **6** | **chip token embedding 程序化生成**（不存表） | 省 3.38 GiB 磁盘/RAM | ✅ **已完成** |
| **7** | **`blk.13` 进稀疏** | 见 #5 | ✅ **已完成**（M4b）：5.11 → 3.40 ms |
| **8** | **code_token：chip token 不走 lm_head** | 省 3.383 GiB（窄头替代 fullhead） | ✅ **已完成**（M4c）：4165/4165 |
| **9** | **融合 op**（`GGML_OP_CHIP_SUBLAYER`，每层 1–2 kernel） | 3.40 → ~0.55–0.65 ms | 下一步（**当前最大的单项**） |
| **10** | **抽薄引擎**（8,000–10,000 行） | 再拿掉 ~0.3 ms 建图税 | 4–8 周（**#9 之后评估**） |
| **11** | **不做：删通用性**（179 arch / 5,900 行 split / KV 家族） | **~0 ms**（它们不进热路径） | 不做 |

**薄引擎调研的结论**（详见工作区 `_k0v5/REPORT.md` 与本节）：

- **通用性税里只有两项是真成本**：融合白名单太窄、backend sched/建图；其余（179 arch / 408 tensor 名 / 5,900 行 split / KV 家族）全是死代码
- **M4 的「全宽 grid barrier」结论要修正**：对 dense 成立；对 batch=1 的稀疏 GEMV，归约可由每个 program 冗余算、barrier 可消——**所以「13 层融成 1 个」不能，「每层 1–2 个」能**
- **手搓 kernel 的真实形态不是「写更多 kernel」，是「写更少的 kernel + 每条配一条 bit-exact 证据链」**
- **3.40 ms 里已经没有被「稀疏化」能拿走的东西**：剩下是 14 层 × 约 15 节点/层的建图、调度与 launch

## 7. 下一步

| 顺序 | 任务 | 依据 |
|---|---|---|
| ✅ 1 | **消除 state 搬运的恒等拷贝**（`build_rs` 用别名代替 gather） | 已完成（`861e3d659`）：**1.74 → 1.59 ms（−8.6%）**，bit-exact。单序列下 14 层各搬 2 MB「自己→自己」，**每 token 56 MiB，是 chip 权重的 8 倍**；写侧早有 `ggml_cuda_try_gdn_cache_fusion`，读侧漏了。详见 [MILESTONES](../measure/MILESTONES.md) M4e |
| ~~2~~ | ~~按累加桶做行内并行~~ | **已试并回退**：graph 模式只值 −4%（噪声内），no-graph 值 −18%。graph 里 84 个 matvec 本就并发，单 kernel 的延迟尾巴被盖住，**生产路径是吞吐受限**。详见 [BASELINES](../measure/BASELINES.md) §一之二 |
| 3 | **融合 sublayer**（`GGML_OP_CHIP_SUBLAYER`）：335 → 约 60 个 kernel | 预计再降一半；GDN 要么移植（bit-exact 风险）要么保留原生 |
| ✅ 4 | **M5 端到端**：chip + Qwen 同进程跑完整 46 层，含 state 注入 | 已完成：46 层 GPU 跑通（7679/8188 MiB），GPU/CPU 两条贪心链的 token 序列 12/12 相同，**16.3 tok/s**。端到端分解见 [MILESTONES](../measure/MILESTONES.md) M5：**chip 段只占 5%** |
| 5 | 抽薄引擎（视 #3 的收益决定） | 再拿掉 ~0.3 ms 建图税 |

**已经量清楚的事**：decode 的 GPU busy 1.525 ms 里，稀疏 matvec 占 59%，GDN 只占 5%，state 搬运原占
13%（已消除）；CPU 只占约 0.1 ms（CUDA graph 生效）。三条截断诊断里，**不读 x 只值 6%、单累加器值
12%**（后者动的是局部内存，但 32 路累加器是 bit-exact 的硬约束，拿不到）——这两条是纯工作量，说明
稀疏 matvec 已经逼近「1.6 M 个非零 × 每非零成本」的下限；按桶并行把延迟摊开也只值 4%，因为 graph
里 kernel 本来就并发。**所以优化目标只剩 kernel 数与无效拷贝，不是核内并行度、不是带宽、不是 CPU、
不是访存 locality。**

### 7.1 待办：`output.weight` 与 `token_embd` 的重复（省 2.131 GiB，暂不做）

`k0v5` 的 GGUF 里 `token_embd.weight` 与 `output.weight` 都是 `[4608, 248320]` BF16、**逐字节相同**，
白占 2.131 GiB。已在工作区查清可行性（`_k0v5/REPORT_TIED.md`），结论：

- **`n_vocab` 来自 `vocab.n_tokens()`，与 `token_embd` 的行数无关**——所以「tied 会把词表缩成 248K」这个担心不成立
- 唯一代码障碍是 `qwen35.cpp:309-317` 在「chip_emb 存在且没有独立 output」时直接 `throw`（它的理由是「tied head 必须是全词表」，忽略了 k0v5 的 head 本来就只需 248,320 行）
- **陷阱**：input 层被钉死在 CPU。只删 `throw`、保留 `TENSOR_DUPLICATED` 的话两个 ctx 找不到同名 tensor，会各建一份 → **一个字节都不省，只省文件**。必须让 `output = tok_embd` 指针共享
- **放置权衡**：共享且留在 CPU → 省 2.13 GiB VRAM 但每 token 要做一次 248K×4608 的 CPU matvec；共享 + `-ot 'token_embd\.weight=CUDA0'` → 省 2.13 GiB host RAM，速度不变
- **代价**：`llama-quant.cpp` 用「没有 output.weight」判定 tied，会把 `token_embd` 按 output 类型量化，而 chip 表硬性要求 BF16 → **tied 化后不能再量化**

**为什么暂不做**：它省的是磁盘与 RAM，不是 chip 步；而 lm_head 放 CPU 还是 GPU 是真实权衡，且会关掉量化这条路。
