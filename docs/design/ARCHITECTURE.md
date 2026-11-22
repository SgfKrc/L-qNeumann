# 架构：这台机器怎么组成

> 状态：**现行**
> 更新日期：2026-11-23
>
> 适用范围：这台机器的四维资源（列×层×时间×状态）与五个插槽；读者为维护或移植该机制的开发者。
> 相关：立项 [PROPOSAL](PROPOSAL.md)、谱系 [LINEAGE](LINEAGE.md)、实测 [BASELINES](../measure/BASELINES.md)。

---


> 详细版见工作区 `mainboard-architecture.md`（~340 行）。这里是给仓库读者看的摘要。

## 一、四维资源

这台机器的资源不是「显存/算力」，而是四个正交的维度：

| 维度 | 是什么 | 容量 | 当前用法 |
|---|---|---|---|
| **列** | 残差流的宽度 | 4608（可用 4374） | 电路 + Qwen 分时复用（overlay） |
| **层** | 一次前向的深度 | 46 | 13 电路 + 1 kernel + 32 Qwen |
| **时间** | token 流 | 2,782 token/s（现在约 11,000） | 一个 token = 一个时钟周期 |
| **状态** | DeltaNet 的 recurrent state | **266** value head × 128 行 | 寄存器/RAM/ROM/数字文件都在这里 |

**关键**：这四维是**绑死的**——列决定一个 token 能带多少信息，层决定一个 token 能算多深，时间决定吞吐，状态决定跨 token 能记住什么。

## 二、五个插槽

| 插槽 | 是什么 | 状态 |
|---|---|---|
| **CPU** | 13 层 RV32IM 电路 | 原生 |
| **VX** | 64-lane 向量单元，8 个 op | 原生 |
| **KERNEL** | 第 13 层，64×64 matvec，1 token 走 DAG | K0/K1 完成 |
| **KERNEL_CALL** | CPU 调 kernel 的 MMIO 接口 | K2–K4 判不做（需真实需求） |
| **VISION** | Medium ViT，输入设备（不是计算设备） | **q35 已接入；rv1 丢弃**（`rv/merge.py:239` 跳过 `model.visual.*`） |

## 三、三条精确性条件

电路能在 Qwen 的列上跑，靠 overlay 复用。但复用有三个条件，**靠 `corpus()` 测量**而不是构造性保证：

1. 电路的 dim 在 text token 上必须为 0（否则污染语言）
2. Qwen 的 dim 在 chip token 上必须为 0（否则污染电路）
3. 两者共用的列必须**分时**（不能同时活跃）

**RMSNorm 是全宽的**——任何跨 token 的动态列都会破坏 pinned-constant 位精确恒等。所以**状态必须层内私有，不能进残差**。

## 四、为什么它长得像 LLM

因为它**复用了 LLM 的封装**：

| 类比 | LLM | 这台机器 |
|---|---|---|
| 封装 | 46 层 transformer | 一样（标准 Qwen3.5 结构） |
| 词表 | 248K 文本 token | + 394K 电路 token + 8 kernel token |
| embedding | 词向量 | **设备寄存器**（含 2^40 ballast） |
| 输出 | 下一个 token | **设备回传结果** |
| 「学到的东西」 | 后 32 层 | 前 14 层是**编出来的**，不是学的 |

**所以它是一块主板**：Qwen 32 层是 CPU/GPU（通用计算），前 14 层是外设（计算器/向量单元/kernel 加速器），残差流是总线，token 流是时钟。

### 读出：chip token 的 id 就在 hidden 里，不用 lm_head

残差的 21 个 code dim（`code_dims` 2725..2745，值域 ±2⁻²⁰）**直接就是下一个 chip token 的 id**：
符号 = bit → 5 位 family + 16 位 payload → 查表。这条规则是参考实现 `chat.py` 的 `code_token`，
在 llama.cpp 里已落地（`llama_context::decode` 把该位置整行 logits 写成 one-hot）。

**这意味着 chip 区不需要那 3.383 GiB 的输出头**（`output.weight` 的 chip 行）。已验收：
窄头 + code_token 的 c-crc32 teacher forcing **4165/4165**，argmax 与 fullhead 的 `lm_head`
**4166/4166 逐位一致**——同一个 hidden 上的两条读出，数值路径完全不同。

**零 state 跑不出它**：chip 段的初始 recurrent state 是程序的一部分（c-crc32 的非零 30,788 个），
从零开始模型会认为程序执行了非法指令。详见 [MILESTONES](../measure/MILESTONES.md) M4c。

## 五、时间轴（一个 chip token 做什么）

```
token:    t
            │
CPU:      CALL ──→ 读结果（下一 token）
KERNEL:        exec（一次前向走完整个 DAG）
```

**13 层是一个 token 内的组合逻辑深度，不是 13 个时钟。** 取指（L1–L2）、译码（L3–L4）、执行（L5–L8）、访存（L9–L10）、写回（L11–L13）全在**同一个 token 内**串行完成。

所以「一条指令 = 1–4 token」（大部分是 1），不是「几十 token」。

## 六、一句话

> **它是一个 LLM，但它的前 1/3 不是神经网络，是电路；它的价值不在说人话，在它真的把一台计算机放进了权重里。**

## 视觉塔：旁路形态（已决）

**rv1 的 checkpoint 里没有视觉塔**——`rv/merge.py:239-240` 用 `if not k.startswith(SRC_TEXT): continue`
把 `model.visual.*` 整份丢弃，产物是 `Qwen3_5ForCausalLM`（622 个 key 全是 `model.layers.*`，0 个 visual key）。

**所以视觉能力用「旁路」形态提供**，不进 checkpoint：

```
图像 → 视觉塔(636 MiB, 冻结) → 读数头 → 文本 token（"123"）
     → prefill 阶段拼进 token 流（同 BIOS boot 机制）
     → 电路 P3 文本捕获 → TEXT_ADDR 缓冲 → calc 固件 → 精确结果
```

**为什么是旁路而不是残差**：

| 项 | 旁路 | 残差 |
|---|---|---|
| merge / 电路 / IR 改动 | **0 行** | merge ~30 + chat ~80 行 + **overlay corpus 重测** |
| 8 GB 可行性 | **130 ms/图**（747 MiB） | 百秒/图（576 token × 32 层） |
| 能力完整性 | 互补（读得准 + 算得准） | 重叠（LLM 本来就不擅长精确算术） |

**实测**（`H:/sgfd9/vision-tower-report.md`）：

| 任务 | 结果 |
|---|---|
| 视觉塔独立加载 | 333.5 M 参数 / 636 MiB / 297 tensors / 0 missing |
| 前向 | 256px 28.8 ms；768px 129.6 ms |
| 单数字 0–9，8 字体 | **跨字体 100%** |
| 3 位数 | **98.9%** |
| 增强训练后（噪声/模糊/旋转） | **100%** |
| **不增强训练** | **1.5px 模糊 → 0%**（关键脆性） |

**已知边界**：仓库里 0 处图像输入代码/测试；读数错误率 0–2.8% 对「验算」是致命的，需要拒读/置信度机制；
`pos_embed 48×48` 硬编码 → 768×768 上限。
