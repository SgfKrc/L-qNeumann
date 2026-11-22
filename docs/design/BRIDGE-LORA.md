# 桥的 LoRA：能加在哪、要什么、怎么验、值不值得做

> 状态：**调研报告（未实施）**
> 更新日期：2026-11-22
> 适用范围：`circuit/chipformer/rv/`（RV32 电路版，46 层 = 13 电路 + 1 kernel + 32 Qwen）的「电路 ↔ 语言」桥；
> 读者：决定要不要立项的人。
> 相关：[ARCHITECTURE](ARCHITECTURE.md)、[PLAN](PLAN.md)、[BASELINES](../measure/BASELINES.md)、[BUILD](../deploy/BUILD.md)、
> 桥本体 `circuit/chipformer/rv/bridge.py`、解码循环 `circuit/chipformer/rv/chat.py`、评测协议 `circuit/chipformer/q35/evals.py`

**结论先行**（四句）：

1. **LoRA 在技术上可行，且有一个干净的落点**：只加在 14–45 层；chip token 的计算图里没有任何 Qwen 权重，所以 LoRA **构造性地**碰不到 bit-exact（§1.3）。
2. **但「训练桥」大概率不是当前瓶颈**：仓库里已有 `pytool_inline`（把电路的回显换成 Python 逐 token 写出）这个对照条件。先量 `merged − pytool_inline`，就知道桥究竟丢了多少分。**这一步不训练，半天**（§3.1、§5-D0）。
3. **8 GB 能训，但很紧**：4bit QLoRA、`r=16`、seq ≤ 640、bs 1 + 梯度累积；实测 seq 512 峰值 **6.53 GiB / 316 tok/s**，seq 768 峰值 **7.56 GiB / 270 tok/s**，seq 1024 峰值 **8.58 GiB 溢出到共享内存（53 tok/s）**（§2.2）。
4. **有一个决策前提要先过**：`docs/design/PLAN.md:326` 仍把「训练」写在 §5「不做什么」里（"全程无训练（QwenPU 的前提）"），而 `PLAN.md:334` 同时把「LoRA 训练桥」登记为 §6 待评估方向 #3。两处并列 = 还没决定。本报告是给 #3 的评估，不是它的执行（§4-R1）。

---

## 一、LoRA 能加在哪（含 bit-exact 影响分析）

### 1.1 可加性表（已按代码核实）

| 层 | 可加 LoRA | 原因（证据） |
|---|---|---|
| 0–12 circuit | ❌ | 权重是 `chipformer.rv.merge.build_merged` 用 `q35.emit.emit_layers` 从电路 IR **逐 cell 发射**出来的；M2/M3 的验收就是这些 cell 与 `ir.evaluate` 逐位相等（`docs/measure/BASELINES.md`、`PLAN.md` §2.0.b/c）。浮点扰动即破坏 bit-exact。 |
| 13 kernel | ❌ | 同上，`kernel_layer(...)` 发射；GGUF 里 0–13 层是 bf16 且「170 tensor 逐字节未变」是量化验收项。 |
| **14–45 Qwen** | ✅ | 标准 Qwen3.5 层，`chat.py` 的 `_qwen_step` 只让 text token 经过它们（`chat.py:373-382`）。 |

原始 4B 的层结构（`H:/sgfd9/models/Qwen3.5-4B/config.json` + 实测模块名）：
32 层，`full_attention_interval=4` → **8 层 full attention（索引 3,7,…,31）+ 24 层 Gated DeltaNet**；
`hidden=2560`、`intermediate=9216`、`head_dim=256`、`partial_rotary_factor=0.25`（RoPE 只转 64 维，与 `mute.py:26 ROTARY=64` 对齐）、`vocab=248320`、`tie_word_embeddings=true`。

实测可被 peft 命中的线性模块共 **249 个**（248 + `lm_head`）：

```
24 × linear_attn.{in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj}   (DeltaNet 层)
 8 × self_attn.{q_proj, k_proj, v_proj, o_proj}                              (full attention 层，q_proj 含 gate 半区)
32 × mlp.{gate_proj, up_proj, down_proj}
 1 × lm_head (linear, tied)
```

**建议 target**：`mlp.*` + `o_proj`/`out_proj` + `qkv` 类，`r=16`（实测 32.46 M 可训参数 = 全模型的 1.32%）。
不建议 `lm_head`/`embed_tokens`：tied、且 merge 时 embedding 会被重写成「设备寄存器 + 2^40 ballast」，改它等于改设备语义。

### 1.2 用户问题的正面回答：LoRA 会不会通过 KV cache / recurrent state 影响 chip token？

**不会。而且是构造性的，不是经验性的。** 两条路径在 `chat.py` 里是分开的：

| 事实 | 证据 |
|---|---|
| chip token **只跑 0–13 层**，Qwen 层根本不被调用 | `chat.py:544-548`：`chip_in` 时 `nxt = self.code_token(...)`，走的是 `_chip_step`（`chat.py:363-371`，循环 `for L in range(self.n_chip)`），不调用 `_qwen_step` |
| chip 的 cache 与 Qwen 的 cache 是两个对象 | `_chip_cache()`（`chat.py:317-330`）返回独立的 `DynamicCache`；Qwen 用 `out.past_key_values`（`chat.py:460`）。`hf.inject(c, ccache)` 只写 chip 层的 `recurrent_states[0]`（`hf.py:61-71`） |
| chip token **不占 Qwen 的 RoPE 位置** | `chat.py:478` `pos = len(ids)`，只有 text token 分支 `pos += 1`（`chat.py:558`）；chip 分支不碰 `pos` |
| Qwen 的 K/V 里没有 chip 位置的条目 | 同上：Qwen 层从未见过 chip token 的 hidden，自然不写 cache |

所以：**LoRA（只改 14–45 层权重）→ 只改 text token 的 K/V 与 Qwen 的 recurrent state → 只影响后续 text token 的解码。** 这是 LoRA 的目的，不是污染。**信息流方向是单向的**：chip→Qwen（exit 写 `mu/tau/code` 到残差），Qwen 从不回流到 chip 层。

> 顺带一个设计红利：chip token 不占位置，所以**电路 free-run 多长都不吃 Qwen 的上下文**；吃上下文的只有回显出来的文本 token。

### 1.3 bit-exact 的三条不变量（这才是「不该影响」的证明）

| # | 不变量 | 为什么成立 |
|---|---|---|
| **I1** | 0–13 层权重逐字节不变 → bit-exact 不变 | LoRA 只挂 14–45 层；merge 是「替换 Qwen 层权重」，chip 层那张发射表不动 |
| **I2** | chip 状态机不读 Qwen 权重 → LoRA 进不去 chip | chip 层的输入是 chip embedding + chip 层自己的 recurrent/conv state（`chat.py:333-338`）；`fast.py` 的 chip 视图 `compact(..., n_chip=13)` 只含 13 个真层 + 1 个全零 dummy 层（`chat.py:298-307`） |
| **I3** | 若将来启用 mute，LoRA 也不会破它——**只要 adapter 按 merge 的补齐规则扩展** | `mute()` 写的是 `in_proj_a/b/z.weight[:, ballast]` 与 `q_proj` 的 gate 行、`k_proj[:, ballast]`（`mute.py:76-90`）。ballast/mu 列是 merge 后**才出现**的第 `d0=2560` 维之后的新列（`merge.py:245-250` 用 `cat([w, zeros])` 扩展）。在原始 4B 上训的 adapter，扩展时按同样的 `[前 2560 列, 0]` 补齐 → **ΔW 在 mu 列恒为 0** → mute 精确保持。 |

有一条**必须先验证的等价性**（I3 的细节）：merge 把 hidden 从 2560 扩到 4608，同时把 norm gain 乘了 `scale=sqrt(2560/4608)`（`merge.py:232` 定义、`merge.py:249-251` 应用）。推导一下：扩展后 `rms' = rms·scale`，`(1+w') = (1+w)·scale`，两者相消，所以 **merge 后 Qwen 层在 `d0` 列上的归一化输入与原始 4B 逐位相同**，后 2048 列恒为 0。因此 adapter 只要补零扩展，效果就与原始 4B+LoRA 等价。这条要实测：同一个 prompt 上 `base+adapter` 与 `merged(带adapter)` 的 logits 差应 ≈ merge 自身的数值噪声。

### 1.4 唯一的真实耦合：mute 路径（当前 build 没开，且 rv 路径不需要）

- 当前部署的两个 checkpoint（`H:/sgfd9/models/q4-rv1-k0v2`、`k0v3`）的 `chipformer.json` 里 **`"muted": false`**——因为 `fit_anti_sink` 要把 7.4 GiB 模型全量上 8 GB 卡（`docs/deploy/BUILD.md:57-63`、`README.md:78`）。
- mute 的语义是「chip token 上 Qwen 层输出恰为 0」（`mute.py:1-17`）：`a=b=z=-1024` 让 DeltaNet 的 `g=0, β=0, output=0`；full attention 的 output gate `sigmoid(-1024)=0`。**但 chip 位置仍是 full attention 的 key（value≈0），靠 anti-sink key 防止它们吸走注意力。**
- 因此：**若某天启用 mute，并且走「chip token 也过 Qwen 层」的路径**（q35 的 `BatchDecoder(skip_qwen=False)`、或直接 `model.generate()` 喂含 chip token 的 prompt），那么任何作用在 `in_proj_a/b/z`、`q_proj`、`k_proj` 上的 ΔW 都会让 mute 失效，chip token 就会真的进 K/V 和状态。
- rv 路径（`chat.py`）本来就不让 chip token 过 Qwen 层 → **不需要 mute，也不存在这个耦合**。结论：**LoRA 与 mute 的关系是「二选一或按 I3 处理」；当前 rv 部署属于前者，最省事。**

---

## 二、训练桥需要什么

### 2.1 数据：教什么，和「训练数据里到底长什么样」

**训练数据里不应该出现任何 chip token。** 这是由桥的定义决定的：Qwen 在推理时看到的就是「去掉 chip token 之后的文本流」——`Session.run()` 把每条 BYTE 记录回显成 Qwen token（`bridge.py:156-181`），`qwen_text()` 再滤掉 chip 侧（`bridge.py:183-185`）。所以训练样本就是**标准 tool-use SFT 序列**，长度 500–700 token：

```
[system: 工具 schema + nudge]      ← chat template，与推理时同一份
[user: 问题]
[assistant: <tool_call>…<function=calc>…<parameter=expression>…a*b…</parameter>…]
[assistant(continue): "\n<tool_response>\n 结果 \n</tool_response>\n"]
[assistant: 最终答案]
```

**一个必须遵守的 token 化约束（容易踩）**：电路的结果是 `putchar` 逐字节打印的，回显时每字节一个 token（`bridge.py:66, 168-171`：`self.T.printable[p]`，即字节级 token），**不是 BPE 合并后的 token**。所以数据生成**不能**用 `tok.encode("123456789")`（那会合并成少得多的 token），必须用 `bridge.Text` 的 `printable` 表逐字节拼。否则训练/推理的 `tool_response` 分布不一致——这是最隐蔽的坑。
（仓库其实已经知道这件事：`evals.py` 的 `pytool_inline` 条件就是"a Python executor writes exactly what the circuit would print, same token stream as merged, minus the chip tokens"。）

**数据的四类**（都从 `ref.Machine` + `Session` 的**真实回显**生成，不手写）：

| 类 | 内容 | 目的 |
|---|---|---|
| 正向 | 正常算术题，工具结果正确，答案直接引用 | 学会「结果即事实」 |
| 冲突 | 大数乘/除，模型先验算错，工具结果与先验相反 | 学会「放弃心算」——这是桥的核心 |
| 边界 | 分数 `p/q`、不整除的 `10 位小数...`、负数、0、`gcd`、`**`、超 62 位 | 学会正确读格式 |
| 负向 | 超出 calc 能力（`error (too complex)` 一类）、非法表达式 | 学会报告失败而不是编造 |

生成器 = `elf.program("tests/data/rv/calc-bridge.elf")` + `BR.park` + `BR.Session.qwen(ids)`，拿 `session.runs[-1]`（就是 Qwen 该看到的 echo token 序列）。**每条样本必须用 `chat.replay()`（`chat.py:580-626`）过一遍**：circuit-owned token 全部等于 IR 预测，才允许入集。

### 2.2 规模：8 GB 显存能训多少（实测，不是估算）

探针（仓库外，`H:/sgfd9/_lora_probe*.py`）在这台机器上跑出来的：

| 项 | 实测 |
|---|---|
| 模型 | `Qwen3_5ForCausalLM` 直读原始 4B（safetensors 5.33+3.99 = **9.12 GB**，bf16 → 8 GB 卡**装不下**，必须 4bit） |
| nf4 加载（double quant，compute=bf16） | **2.90 GiB**，10–18 s |
| + LoRA r16 / α32，12 类 target | **4.21 GiB**，可训 **32.46 M = 1.32%** |
| seq 384，bs 1，gc on | 峰值 **6.01 GiB**，323 tok/s |
| **seq 512** | 峰值 **6.53 GiB**，**316 tok/s** ← 推荐 |
| seq 768 | 峰值 **7.56 GiB**，270 tok/s |
| seq 1024 | 峰值 **8.58 GiB** → 溢出到 WDDM 共享内存，**53–57 tok/s**（不可用） |
| seq 2048 | 峰值 **12.69 GiB**，62 tok/s（不可用） |

由此：

- **可用配置**：4bit QLoRA + gradient checkpointing（`use_reentrant=False`）+ `bs=1` + `grad_accum=16` + **seq ≤ 640**。有效 batch ≈ 8k token/update。
- **一个 epoch 的成本**（3000 条 × ~650 token ≈ 1.95 M token）：约 **1.7 小时/epoch**（按 300 tok/s）。3 epoch ≈ **5 小时**；8000 条 × 3 epoch ≈ **14 小时**（过夜）。
- 数据量建议（行为 SFT 的经验量级，要用学习曲线定，不要拍死）：**500（冒烟）→ 3000（主）→ 8000（若未饱和）**。
- **不要**在 8 GB 上碰：全参、bf16 训练、seq ≥ 1024、`r ≥ 64`、同时开 `gradient_checkpointing` 与 `use_cache`（会警告并自动关）。

### 2.3 框架：本机有什么、缺什么

| 组件 | 版本（实测） | 用途 |
|---|---|---|
| torch | 2.13.0+cu126 | ✅ |
| transformers | 5.17.0（`Qwen3_5ForCausalLM` 存在且可载原始 4B） | ✅ |
| peft | 0.19.1 | ✅ `LoraConfig/get_peft_model/prepare_model_for_kbit_training` 全通 |
| bitsandbytes | 0.49.2 | ✅ nf4 加载/LoRA 挂载正常 |
| accelerate / datasets | 1.14.0 / 5.0.0 | ✅ |
| `flash-linear-attention`（fla） | 0.6.0 装了但**导不进来** | ❌ 见下 |
| trl / LLaMA-Factory | 未装 | 不需要（原生 torch 训练循环足够） |

**fla 的两层坑（实测）**：
1. **GBK locale**：`import fla` 报 `UnicodeDecodeError: 'gbk' codec can't decode byte 0x94`。设 `PYTHONUTF8=1` 可过。
2. 过了第一关后，transformers 会走 Triton 内核编译；本机 Triton 在 Windows 上编译**极慢/卡死**（一次 300 s 超时未完成），而且 DSH 会话的 `TEMP` 指向不存在目录时直接 `FileNotFoundError`（`tmp*.ptx` 写不进去）。
→ **结论：按参考实现（`chunk_gated_delta_rule` 的 PyTorch fallback）训**。上面 316 tok/s 就是参考实现下的数字。transformers 会警告"much slower"，但**正确**。

**环境坑（与仓库既有记录合并）**：
- `from ortools.sat.python import cp_model` 必须在 `import torch` **之前**（`docs/deploy/BUILD.md:19`）——训练脚本同样适用。
- Windows 上跑任何 Triton 相关的东西，先把 `TMP/TEMP` 指到一个**存在**的目录。
- merge 阶段要 22.87 GiB 的 CPU 内存（`BASELINES.md`）；本机 32 GiB RAM、空闲 ~25.8 GiB，够，但 `merge.py` 已经做了分片落盘以避开峰值。

---

## 三、目标形态：怎么定义「桥可用」，怎么量

### 3.1 桥的损失现在就能量出来（**不用训练**）

`evals.py` 已经定义好了对照条件（`evals.py:8-25`），把桥的损失做成一个**配对差值**：

| 条件 | 是什么 | 在桥的评估里扮演 |
|---|---|---|
| `base` / `base_think` | 原始 4B，无工具 | 下界（自己心算） |
| **`merged`** | 46 层，电路跑每次调用 | **被测对象** |
| **`pytool_inline`** | base + 工具，Python 执行器写出**与电路完全相同**的文本 | **文本上界**（模型侧已无 chip 机制，只剩"用不用结果"） |
| `pytool` | 标准 agent loop（外部计算器） | 外部计算器上界 |
| `oraclec_nudge` | 在 `=` 后注入正确结果 | 绝对上界（注入正确性的天花板） |
| `merged_plain` | 无 BIOS 无工具，应等于 `base` | **保真检查**（LoRA 后不退化） |

**桥可用性 = Acc(`merged`) ÷ Acc(`pytool_inline`)**。如果这个比值已经 ≈ 0.95+，训练桥是浪费；差距在哪一项再针对那一项。

### 3.2 桥的四项可测子能力（把损失拆开）

```
Acc_merged ≈ P(发起调用) × P(参数正确) × P(采纳结果) × P(基于结果给出正确答案)
```

| 子能力 | 指标 | 数据从哪来 |
|---|---|---|
| 调用率 | `tool-call rate` | evals 已有（`summary` 里就有） |
| 参数正确率 | `call correctness` | evals 已有 |
| **采纳率** | 最终答案里的数字是否落在 `tool_response` 的数字集合里；`tool_response` 之后是否又写出**不同**的数（重算/自相矛盾） | 需新增一个小检查（答案数字溯源） |
| 组合正确率 | 最终答案 exact（有理数精确比较，`arith_bench.strict_pred` 已有） | evals 已有 |

**注意一个口径缺陷**：`pytool_inline` 的 token 序列与电路**逐字节回显未必逐 token 相同**（BPE 合并 vs 单字节）。它作为"上界"是近似的，报告数字时要写明。

### 3.3 这是「指令微调」还是「对齐」

- **是 SFT（指令/行为微调）**：训练目标是「给定已出现的结果，后续 token 的分布」，标签来自真实轨迹。
- 不是 RLHF/偏好对齐。**但它调的是一个"信任谁"的偏置**（自己的先验 vs 外部权威），这更接近**校准/顺从性（compliance）**问题。
- 只有当 SFT 出现「该信的时候不信 / 不该信的时候也信」（过度依赖）时，才需要进入 DPO/拒绝采样那一层。**别一上来就做对齐。**

---

## 四、风险

| # | 风险 | 影响 | 应对 / 止损 |
|---|---|---|---|
| **R1** | **与项目前提冲突**：`PLAN.md:326` 仍把「训练」列在 §5 不做什么（"全程无训练（QwenPU 的前提）"），而 `PLAN.md:334` 又把「LoRA 训练桥」登记为 §6 待评估方向 #3 | 立项层面，不是技术层面：两处并列等于「还没有决定」 | 本报告就是给 §6 #3 的评估；先做 §3.1 的无训练测量，若桥损失小，§5 的前提不用动 |
| **R2** | **Q4_K 量化吃掉 ΔW** | LoRA 训好了，量化后等于没训 | LoRA 的 ΔW 是低秩小幅修改；Q4_K 每 32 元素一个 scale，可能吞掉。**必须**用「量化后」而非「merge 后」的数字验收，并与未训练模型对比 |
| **R3** | 8 GB 上限 | seq ≤ 640、r ≤ 16、不能全参 | 实测表（§2.2）；退路：租 24 GB 卡跑一次（成本极低），或 rank 8 + seq 384 |
| **R4** | **训练/推理 token 化不一致** | 训练白做 | `tool_response` 必须用 `bridge.Text.printable` 逐字节构造；每条样本过 `chat.replay()` |
| **R5** | 训练时把 chip token 混进 `input_ids` | 训练经过全 46 层（含 chip 层），与推理路径完全不同，还可能与 mute 打架 | 数据里禁止出现 `id ≥ 248320`；训练前 grep 一遍 |
| **R6** | **过度依赖电路** | 工具不可用/超限时退化到"编造" | 负样本 + `merged_plain` 对照；指标：工具报错时的正确"放弃率" |
| **R7** | 文本能力漂移 | KL(base‖merged) 上升，日常对话退化 | `textcheck.py` 的 KL 与噪声底对比，硬门槛：KL ≈ 噪声底量级 |
| **R8** | 合成数据的表面记忆 | 只学会"抄 tool_response 里的数字"，换格式就崩 | 数据格式覆盖（分数/小数/负数/巨数）+ held-out 家族 + 用 `dmi_*`/`nupa_*` 之外的分布外集合 |
| **R9** | mute 不共存（若将来启用） | chip token 污染 Qwen 的 K/V 与状态 | 按 I3 补零扩展 adapter；或明确「启用 mute 时不训 LoRA」 |
| **R10** | 训练成本被低估 | 一晚上只跑了半个 epoch | 先跑 50 step 测吞吐，再算总时长（§2.2 的 300 tok/s 是 seq≥512 的实测） |

---

## 五、计划（如果真要做）

**门禁原则：每一步都有「不做下去」的判据。** D0 通过才做 D1。

| 步 | 做什么 | 产出 | 工时 | 门（不过就停） |
|---|---|---|---|---|
| **D0** | **先量桥的损失，不训练**。把 `evals.py` 的 `base / merged / pytool_inline / oraclec_nudge` 接到 rv 的 `RvChat.generate`；跑 `arith`（约 200 题）+ `gsm8k`（约 300 题，配对） | `dist/evals/bridge-loss.json`：四项子能力 + McNemar 配对差 | **0.5 天** | 若 `Acc(merged)/Acc(pytool_inline) ≥ 0.95` → **不训**，改做提示/解码器规则（`nudge`/inline 已存在） |
| **D1** | 数据生成器：用 `ref.Machine + BR.Session` 出真实回显；四类样本；逐字节 token；每条过 `chat.replay` | `dist/bridge/data/{train,heldout}.jsonl`（500 / 3000 / 8000 三档） | **1.0 天** | 回显 token 与电路 `--check` 100% 一致；否则先修数据 |
| **D2** | 训练管线：QLoRA(r16) + gc + seq640 + bs1×acc16；50-step 吞吐标定；500 条冒烟 | 可复现的训练脚本 + 冒烟 adapter | **1.0 天** | 实测吞吐 ≥ 250 tok/s、显存峰值 ≤ 7.0 GiB、loss 下降 |
| **D3** | 主训练 + 学习曲线（500 → 1500 → 3000 → 8000，同一 held-out 上评） | adapter + 曲线 | 工程 0.5 天 + **机器 1–2 晚**（每个规模 1–2 epoch） | 曲线在 3000 处仍未饱和才继续加数据 |
| **D4** | 合并与回归：`merge_and_unload` → 存 bf16 4B → `rv.merge`（带 adapter 的 Qwen 层、chip 层不动）→ 重量化 Q4_K → **bit-exact 回归** | 新 checkpoint + 回归报告 | **0.5 天** | `test_rv_ref`、`test_rv_fusedexec`、`chat --check`（wrong_predictions=0，owner_mismatch=0）任一项红 → 回滚 |
| **D5** | 全量评估与决策：D0 的同一套条件重跑；`textcheck` KL；D0 的四项子能力 | 决策报告 | **0.5 天** | 只有「子能力里至少一项显著改善且其他不退化」才保留 adapter |

**合计：约 4.0–4.5 人日 + 1–2 晚 GPU。** 全部硬件已就位（torch 2.13 / peft 0.19 / bnb 0.49 / transformers 5.17），不需要装任何新框架。

**D4 的捷径（让 merge 零改动）**：不要动 `merge.py`。用 `peft` 的 `merge_and_unload()` 在 CPU 上把 adapter 融回 bf16 的原始 4B（~9 GiB RAM，本机 32 GiB 够），存成 `H:/sgfd9/models/Qwen3.5-4B-bridge`，然后 `python -m chipformer.rv.merge` 以它为 `src` 重新构建 46 层。这样 `[前 2560, 0]` 的补齐、norm 的 `scale`、mute 的 mu 列全部沿用既有规则，I1/I3 自动成立。

**如果只能做一件事**：做 **D0**。半天，不碰训练，直接回答「桥到底丢了什么」。这也是本报告唯一无条件推荐的动作。

---

## 六、与 PLAN §6 其他方向的关系（别把它当唯一解）

`PLAN.md:330-337` 的待评估方向里，有三条会和「训 LoRA」抢预算，**必须先横向比**：

| PLAN §6 | 是什么 | 与 Bridge-LoRA 的关系 |
|---|---|---|
| **#2 让「族」可读**（族 id → Qwen embedding） | 不改权重，改 embedding，让 Qwen **直接读懂 chip token 的语义** | **互补/竞争**。它省的是「文字回显」的 token 开销，不改「信不信结果」的偏置；LoRA 改的是偏置。两者可以叠，也可能 #2 单独就把桥损失降下来 |
| **#5 chip 层稀疏存储** | 性能（省 5.30 GiB） | 无关，但**同时做会撞显存/适配期**：稀疏化那段别碰 14–45 层 |
| **#6 chip token embedding 程序化生成** | 省磁盘/RAM | 无关；但它动 embedding 表，**adapter 不能挂 embedding**（本报告已排除） |

另外提醒一条**更便宜的替代路线**（本报告 §3.1 的 D0 就能验证它够不够）：`evals.py` 里已有 `nudge` / `NUDGE2` / `nudge3` / inline 模式——这些都是**提示工程 + 解码器规则**版本的桥。如果 D0 显示桥损失主要来自「模型不写计算式 / 格式不匹配」，改 prompt 与解码约束比训 LoRA 便宜两个数量级。**先证伪提示工程，再上训练。**

---

## 附：本报告的实测来源

全部探针在仓库外（`H:/sgfd9/`），不污染 git：

| 文件 | 内容 |
|---|---|
| `_lora_probe1.py` | nf4 加载可行性、模块名清单、显存 2.90 GiB |
| `_lora_probe2.py` | QLoRA 挂载 + seq 1024/2048 的步时与峰值（8.33–12.69 GiB） |
| `_lora_probe3.py` | fla 导入失败诊断 + seq 384/512/768 的峰值与吞吐 |
| `_lora_probe4.py` | `PYTHONUTF8=1` 下 fla 启用尝试（Triton 编译超时，未完成） |

复现：`cd H:/sgfd9 && python -u _lora_probe3.py`（注意先 ortools 后 torch、`TMP` 指到存在的目录）。
