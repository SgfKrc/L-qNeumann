# 谱系：L-qNeumann 站在谁的肩膀上

> 状态：**现行**
> 更新日期：2026-11-23
>
> 适用范围：与 qwenputer / QwenPU / torchwright 的关系及本项目的全部改动；读者为 review 与上游回馈。
> 相关：立项 [PROPOSAL](PROPOSAL.md)、基线 [BASELINES](../measure/BASELINES.md)。

---


## 一图

```
qwenputer (divagr18)                 上游总项目：把一台计算机编译进 Qwen3.5 的权重
├── Qwenputer model                  HF/ModelScope 上的成品权重
├── Filesystem model                 文件系统版本
├── NPU model                        NPU 单元版本
├── QwenFX                           模型内的 coding agent
└── QwenPU (divagr18)                ★ 本项目的 fork 源
    │                                Qwen3.5-4B + 32 位 RISC-V CPU + VX 向量单元
    │
    └── [fork + 深度魔改]
        ├── qwenpu/                  overlay · kernel 插槽 · 记录压缩 · 访存优化
        ├── torchwright/             ★ 编译器，本项目扩展了它的状态机能力
        └── engine/                  ★ 本项目新写的推理引擎（QS-1）
```

## 三个上游各提供什么

### qwenputer —— 概念来源

它证明了「**一个 transformer 的权重里能装下一台计算机**」，而且真的跑起来了。这个构造性证明是
本项目所有工作的前提。我们**不直接改它**，因为它是成品发布（HF/ModelScope 上的权重），不是开发仓库。

### QwenPU —— 本项目的 fork 源

上游 QwenPU 是 qwenputer 的子项目，提供：

- **13 层 RV32IM 电路**（编译进权重，不是训练出来的）
- **VX 向量单元**（64 lane，8 个 op）
- **394K 电路 token 词表**（`FAMILY_SIZES`）
- **merge 工具**（把电路和 Qwen 拼成一份 safetensors）

**我们 fork 了它，并做了这些改动**（都是上游没有的）：

| # | 改动 | 文件 | 效果 | 状态 |
|---|---|---|---|---|
| 1 | **overlay 列复用** | `q35/overlay.py` | 电路 dim 复用 Qwen 的 1267 列；columns 5119→4374 | ✅ 完成 |
| 2 | **O1 瘦身** | merge `--hidden 4608 --ballast 18` | checkpoint 24.89→22.40 GiB | ✅ 完成 |
| 3 | **KERNEL 插槽** | `q35/kernel.py` | 第 13 层，80 列，1 token 走完 4 层 DAG | ✅ K0/K1 完成，K2–K4 判不做 |
| 4 | **SB/SH 记录压缩** | `rv/{spec,ref,core}.py` | 4 token→2 token，sieve −18.8% | ✅ 完成 |
| 5 | **chip token 连续性** | `rv/{fastexec,fusedexec,chat}.py` | 一次 `run(k)`，2.31× | ✅ 完成 |
| 6 | **head 拆 CTA + 跨 CTA 归并** | `rv/fusedexec.py` | chip 步 1.307×（口径见 [BASELINES](../measure/BASELINES.md)） | ✅ 完成 |
| 7 | **calc 清零精确化 + 延迟约分** | `env/calc/calc.c` | `4+3` 826→326（−60.5%） | ✅ 完成 |
| 8 | **混合量化** | `v1_*.py` + tensor-type 规则 | 14.20 GiB，chip 层逐字节未变 | ✅ 完成 |

### torchwright —— 本项目扩展了它的状态机

上游 torchwright 是**编译器**：把 Python 计算图变成 transformer 权重，**全程不训练**。默认产出
Phi-3 架构的 `transformers` checkpoint。

**本项目扩展的部分**（64 个测试）：

| 层 | 能力 | 文件 |
|---|---|---|
| **组件层** | `StatefulComponent` / `AccumulatorComponent` / `DeltaRuleWindow` / `GatedDeltaRuleComponent` | `compiler/components/stateful.py` |
| **图契约** | `attach_state` / `init_states` / `advance_states` | `compiler/transformer.py` |
| **ONNX 导出** | `state_in_i` / `state_out_i` / `read_i` 三族 IO + Loop prefill | `compiler/export.py` |
| **ONNX 读侧** | `OnnxPast.read` / `prefill_chunk` / 状态偏移校验 | `compiler/onnx_load.py` |

**为什么需要它**：QwenPU 的 DeltaNet 需要**跨 decode step 持有矩阵状态**（`(rows, cols)`）。上游
torchwright 只能产无状态计算图，所以我们加了整套状态机。P3 验证：对官方 `Qwen3_5RMSNormGated`
和 `torch_recurrent_gated_delta_rule`，state `max|diff| = 1.19e-7`。

## 为什么不叫「QwenPU v2」

因为**它已经不是 QwenPU 了**。上游 QwenPU 的核心主张是「RISC-V CPU + 向量单元」，本项目在此之上：

1. **换了执行模型**（chip token 连续 burst、head 拆 CTA、cross-CTA 归并）
2. **加了新设备**（KERNEL 插槽）
3. **改了数值通路**（overlay 复用列、SB/SH 2-token 记录）
4. **换了推理路径**（Triton 电路执行器 + 状态机，不是 HF 原样跑）

如果沿用「QwenPU」这个名字，会让人以为「还是那个 QwenPU，只是快了点」。**它不是——它是同一台机器
的一个新实现，带着自己的引擎。**

## 与 K-Llama 的关系：上游是 K-Llama 的 llama.cpp

**本项目 fork 的是 K-Llama 的分段改造分支**（`H:\llamacpp-c`，从 `rebase/onto-20261115` 切出本项目分支 `Lqn-Llama`），
**不是上游 llama.cpp**。

| 维度 | 说明 |
|---|---|
| **上游** | K-Llama 的 `rebase/onto-20261115` 分支（含 12 个分段改造提交） |
| **本项目分支** | `Lqn-Llama`（7 个提交：A+B / loader / k0v4 稀疏 store / `ssm_state` / 稀疏 kernel / A+ / code_token） |
| **为什么用 K-Llama** | 它保留 llama.cpp 的通用性（不吊死在千问上）；**我们不承担这个约束**——主板只跑 Qwen3.5-4B + chip 层 |
| **魔改边界** | 加 op（`GGML_OP_CHIP_SPARSE_MM`）与图替换，**不改 llama.cpp 的通用路径** |

### 为什么不从上游 llama.cpp 起

**K-Llama 保留 llama.cpp 的通用性是为了给自己留后路**（不吊死在千问一棵树上）。**我们不承担这个约束**，
所以可以：

1. **加专用 op**（`GGML_OP_CHIP_SPARSE_MM`，已做）
2. **改图构建**（6 处 matvec 替换，已做）
3. **以后加融合 op**（`GGML_OP_CHIP_SUBLAYER`，待评估）

**代价**：每次上游升级要重跑一遍 bit-exact 链（已列为风险）。

### 为什么引擎从 Triton 迁到 llama.cpp（v1 的结论作废）

v1 曾判断「引擎留在 Python + Triton，因为移植到 ggml 是另一次重写且有数值风险」。**M1–M3 证明这个判断错了**：

| 项 | v1 的担心 | 实测 |
|---|---|---|
| 移植成本 | 「另一次重写」 | M2 用 ctypes 直连 ggml，**0/6965 差异** |
| 数值风险 | 「GDN chunk kernel 算法不同」 | ggml bf16 与 IR 的 bf16 舍入**完全一致** |
| bit-exact | 未验证 | **c-crc32 4165/4165** |

**所以引擎现在是 llama.cpp 单实现**（见 [PLAN](PLAN.md) §2.0）。Triton 只作 `_reference/` 的性能基线。

## 未来主板计划（Qwen3.5 → Qwen4）

**Qwen3.5 与 Qwen4 之间没有重大架构变化**（都是 GDN + full attention 混合），所以换基座是**可预期的工程**，
不是重新立项。

| 项 | 现状 | 换基座时要做什么 |
|---|---|---|
| **基座** | Qwen3.5-4B | 换 Qwen4 同尺寸系列 |
| **chip 层** | 13 层（`core.build` 的产物） | **重新 merge**（`live_token_dims` 的 pinned 要按新 corpus 重算） |
| **llama.cpp** | `Lqn-Llama` 分支 | 可能要 rebase 到 K-Llama 的最新上游 |
| **稀疏 store** | 8.30 MiB（`blk.0-13`，84 矩阵） | 重算 nnz（`blk.13` 只有 4,203，比 0–12 稀 10 倍） |
| **读出** | `code_token`（21 列 → token id） | 表的来源（`chipformer.json` 的 `ranges`/`printable`）跟着 checkpoint 走 |
| **验收** | c-crc32 4165/4165 | 同一套判据可复用 |

**触发条件**：Qwen4 正式发布 + 其 GDN 参数（`linear_num_key_heads` / `linear_value_heads` / `conv_kernel_dim`）与 Qwen3.5 可比。

**不急。** M1–M4 已收尾（A+ 与 `code_token` 均已完成，见 [MILESTONES](../measure/MILESTONES.md)），
下一步是融合 sublayer 与 M5 端到端。
