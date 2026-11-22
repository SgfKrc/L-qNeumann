# 里程碑记录：M1–M4 与两条收尾

> 状态：**现行**
> 更新日期：2026-11-23
>
> 适用范围：每个里程碑的**结论与判据**，按时间正序。过程细节在工作区的 `_m2a/`、`_m3a/`、`_m4ab/`、`_k0v5/` 报告里，此处只留可复核的结论。
> 相关：计划 [PLAN](../design/PLAN.md)、实测 [BASELINES](BASELINES.md)、边界 [ENGINE-LIMITS](../limits/ENGINE-LIMITS.md)。

---

## M1：容器直接用，不改分配代码

每层 `n_embd_s = ssm_d_state × ssm_d_inner = 128 × 4096 = 524,288`（2 MiB F32），
而 chip 每层 value head ≤32 × 128 行 × 128 列 = **524,288**——**一字不差**。

原因：merge 时把 chip 层前置成 `linear_attention`，与 Qwen 层共用同一份 GGUF 元数据
（`linear_num_key_heads=16`、`linear_num_value_heads=32`、`linear_key_head_dim=128`、
`linear_conv_kernel_dim=4`）。

| # | 结论 | 依据 |
|---|---|---|
| 1 | **容量够**（每层 2 MiB，38 个 recurrent 层合计 ≈76 MiB） | `llama-memory-recurrent.cpp` 的 `n_embd_s()` |
| 2 | **布局 1:1**（转置存储后，「行」= 一个 key 索引的 128 个 value） | `ggml-cpu/ops.cpp` 的 `s_out[j*S_v+i] = S[i][j]` |
| 3 | **conv 窗口现成**（4-tap = 3 列） | `llama-graph.cpp` 的 `build_conv_state_store` |

**要自己写的只有两件**：chip 层的图构建（M2）、decay（`ggml_mul` + one-hot 覆盖）。
**不自建 buffer**——那 ≈ 重写 500–800 行 memory 语义 + 1.5–2 周，且从此 `seq_rm/seq_cp/state_read/write/offload` 自己背。

## M2：13 层全部 bit-exact

**端到端 0 / 6965 dims 差异**（3 条轨迹，每层独立 ggml 上下文、residual 经 numpy 传递）：

| 轨迹 | tok | bad | max\|d\| |
|---|---:|---:|---:|
| call | 128 | **0/6965** | **0** |
| datarom | 19 | **0/6965** | **0** |
| signext | 41 | **0/6965** | **0** |

- **conv window 跨 token**：4 个连续 token 全 0，回看深度逐级展开 `{0}` → `{0,1}` → `{0,1,2}` → `{0,1,2,3}`
- **decay 三值**：ggml `exp(g)` 取值集合 = `{0.0, 0.499999999, 1.0}`，三值全中；401 个「清零后 1–5 token 内读出」事件全 0 差异
- **L2/L10 被补上**：`div` 汇编激活 L2；`datarom` 激活 L10（`drom*`，2048/2048 满）
- **L6/L11 只测到 FFN**：head 恒 0 时两边都是 0，**不计入已验证**（属于 M3/M5 范围）

**一个真 bug**：`Qwen3_5RMSNormGated` 是**标准 `x·w`**（weight=ones → 增益 1），不是 `(1+w)`。
按 `(1+w)` 会多乘 2，制造 22 个假差异；改正后 22 → 0。三种 norm 的语义：

| norm | 形式 | emit 权重 | 增益 |
|---|---|---|---|
| `input_layernorm` / `post_attention_layernorm` | `x·(1+w)/rms` | 0 | 1 |
| **`linear_attn.norm`** | **`x·w/rms`** | **ones** | **1（不是 2）** |
| `final norm` | `x·(1+w)/rms` | −1 | 0 |

**M2 不需要改电路，也不需要改 ggml 配置。** M2b 的预案（ballast 降位、值域收紧）不启用。

## M3：真实布局下的 185 个偏差修到 0，程序 4165/4165

### 布局（slot 复用是唯一变量）

M3 走真实 `llama_context`（零改源码，用 `llama_set_embeddings_layer_inp`），`ggml vs llama` 的
identical slots **635/635、max|d| 0**，但逐位一致没拿到（L0 185/635）。分离实验：

| 变体 | 布局 | L0 bad |
|---|---|---:|
| **A**（M2 基线） | hidden 7168，无 slots | **0/641** |
| **B**（真实） | 4608，1812 复用，overlay | **185/635** |
| **C**（= B，overlay 不写进图） | 同上 | **185/635** |
| **D**（= B，每 dim 独占） | 18432，3995 独占 | **0/641** |
| **B20**（ballast 2^40→2^20） | 同 B | **185/635** |

**A = D = 0、B = C、B20 = B** → 唯一变量是 **slot 复用**；overlay、ballast、hidden、m/off 全部无辜。

**根因**：L0 非零的 token 特征（`isR`/`isGO`/`isNT`/`w.b*`）被让给后面 dim 时没有 handover
（`alloc.clears` 只有 level 8/10 两个键）。被污染的槽位改了 `attn_norm` 的分母，阈值敏感的比较器
（`n>k`）和半衰计数器（0.5/0.25）翻转。

**修复**：`emit.py` 新增 `live_token_dims()` + `allocate(pinned=)`，`merge.py` 用 corpus 的 token
集合算 pinned（**284** 个）。

| 项 | 结果 |
|---|---|
| **L0（真实布局）** | **185 → 0 / 635，max\|d\| = 0** |
| **13 层逐层** | **全部 0 bad**（6445 个 IR dim） |
| `n_slots` | 1812 → **1935**（+123） |
| `kernel_base` | 4391 → **4514**（< 4608 ✓） |
| M2 的 0/6965 | **没破** |

### 行为（308 ulp 可接受）

| 程序 | 可验证位置 | 命中 | 最小 margin |
|---|---:|---:|---:|
| call | 474 | **474/474** | 173.7 |
| **c-crc32** | 4165 | **4165/4165** | **184.0** |

与 qemu golden 三段论闭环：`c-crc32` exit 低 8 位 **57**、输出 `crc32=414fa339`，
模型 token 流 == `ref.Machine` 的 token 流 ⇒ 程序输出 == golden。

**机制**：`merge.py` 的 `norm[d0:] = -1.0` 只给 21 个 code 位 + tau 设 gain，第 12 层之后
**live 只有 22 列、exact 差 0 个 cell**——不是「差得小」，是逐位相等。

**308 ulp 不会让电路算错**：13,905,920 个 gate 输入，`b != 0` 占 **93.73%**（K=128 缩放后实测
放大到 4.9e-04 → **1024× 余量**）；`b == 0` 占 6.27%，唯一没有 bf16 量子的位置。
**「final 730 bad / max 15」是测量假阳性**：比了全部 3208 个 slot，而 exit 层会清 `c.keep` 之外的 dim；
拆分后 **live 22 列 bad 0**，730 全是 dead 列。

### relay 段：两个遗留都不是本项目或 K-Llama 的缺陷

| 遗留 | 裁定 |
|---|---|
| 1. relay 段 ulp 级不等价 | **不存在（误判）**；「bf16 vs f32 走不同 Q4_K kernel」不成立 |
| 2. 含 attention 的 relay 段非确定 | **探针缺陷**；根因在 `llama-batch.cpp` 的 `llama_batch_compat::init` |

**遗留 2 的根因**是 upstream 的一行：embedding 批次的 M-RoPE 位置是 **section-major 4 段**，
`llama_batch_init` 只给 1 段，兼容层照读 4 段 → **越界读堆**（qwen3.5 是 `LLAMA_ROPE_TYPE_IMROPE`
→ `n_pos_per_embd == 4`）。纯净上游 `b94959a64` 原样复现；K-Llama 的 `split_expand_pos()`
反而早修好了真实链路。**修好后三段 relay 全部 bit-equal**（只改探针，llama.cpp 一行没动）：
`range 14:18` 5/5、`range 0:18` 19/19、`range 14:46` 33/33 层，`logits` 642496/642496。

**一个措辞纠正**：`chip_dims_offset=2560` 是 **hidden 偏移**（`merge.py`），所以 `hidden[0:2559]`
是文本区、`[2560:4607]` 是电路区。差异落在文本区是因为**电路区 2048 维不被 Qwen 写**。

## M4：150 μs 在 dense 下不可达，必须换稀疏执行器

### 14.19 GiB 的真实账本

| 块 | GiB | 占 |
|---|---:|---:|
| chip 13 层（bf16） | 4.92 | 34.7% |
| kernel 1 层 | 0.38 | 2.7% |
| Qwen 32 层（Q4_K） | 3.37 | 23.7% |
| **token_embd / lm_head（bf16 tied）** | **5.52** | **38.9%** |

**embedding 是最大的一块**（642,496 行 × 4608 维 × 2 B = 5.52 GiB），比 chip 和 Qwen 都大。

**150 μs 的数学**：13 个 chip 层光「把 4.92 GiB 权重读一遍」就要 GPU 理论 20.6 ms / **实测 32.2 ms**（164 GB/s）
/ CPU 138 ms。**150 μs 需要 35.2 TB/s 有效带宽**，是实测的 **214 倍**。它只在稀疏执行下有意义
（nnz ≈ 1.45M MAC → 23 MB → 155 GB/s，落在带宽内）。

### 两个必须记住的事实

1. **`-ngl N` 是 upstream 语义：从尾部 offload**（`i_gpu_start = n_layer + 1 - N`）。
   `-ngl 14` **不是** chip 上卡，而是 layer 33–45 + 输出层上卡 → tied lm_head 拖上 GPU → 必崩。
   **chip 上卡只能用 `-ot 'blk\.(?:[0-9]|1[0-3])\..*=CUDA0'`。**
2. **瓶颈是带宽（100%）**：batch 1→4 总时间完全不变（37.8→39.7 ms）= 读一遍权重。
   launch 不显著（CUDA graph 已把 ~1000 次 launch 压成一次，固定开销 ~0.3 ms/前向）。

**46 层 ≈ 900–1000 kernel/前向**（13 chip 层 ≈ 260–290，32 Qwen 层每层 ~20）。融合瓶颈 =
**全宽 grid barrier**（rms_norm/l2_norm 跨 CTA 归约、ssm_conv 跨 lane、ssm_scan 顺序依赖）——
这正是 Triton 那次 24→1 慢 14% 的同一物理约束。**极端乐观的叠加仍是 150 μs 的 870 倍。**

### M4 的基本盘（dense，已达成）

| 配置 | CPU | GPU |
|---|---:|---:|
| 46 层 + 全量 tied 头 | **380 ms** | — |
| 46 层无头 | 253 ms | — |
| **chip 段 0:14（无头）** | **128 ms**（−66%） | **26.5 ms** |
| text 段 + 248K 头 | **168 ms**（−56%） | **28.6 ms** |

**显存**：chip 段 **5468 MiB**、text 段 **6597 MiB**（`n_batch ≤ 32`），各自都装得进 8 GB。
**A+B 的 bit-exact 三组全过**：chip 段 vs 全模型 15 × 147456 全等；tied vs split 248320/248320 全等；
A+B 后的 dll 跑 M3 配置 14 层 × 589824 float 全等。

**A+B 里的一个真 bug**：`build_inp_embd` 用 `tok_embd != nullptr` 判断「有 token 路径」，但范围不从 0
开始时 loader 只把该 tensor 标成 lazy、对象还在 → `get_rows` 用**未初始化的 `inp->tokens`** 越界读。
实测同一命令跑两次 logits 不同（argmax 69811 vs 148466）。改成 `tok_embd && hparams.needs_tok_embd()`。

### M4b（A+）：`blk.13` 进稀疏，chip 步 5.11 → 3.40 ms

`blk.13` 的 nnz = 4,203 / 203,194,368 = **0.0021%**，比 `blk.0–12`（0.0215%）还稀 10 倍。
store 从 13 层 78 矩阵扩到 **14 层 84 矩阵**（1,615,758 nnz，**8.30 MiB**），84 个 dense 矩阵
（5.30 GiB）全部删掉。

| 配置（单 token decode，同构建同机器） | chip 步 |
|---|---:|
| dense GPU，14 层（`k0v5a`） | 26.48 ms |
| sparse GPU，13 层 store（`blk.13` 仍 dense） | **5.11 ms** |
| **sparse GPU，14 层 store（`blk.13` 稀疏）** | **3.40 ms** |

**同轮对照 −1.71 ms（−33.5%）**。反算带宽 387.1 MiB / 1.71 ms ≈ **226 GB/s**，与 RTX 4060 Laptop
的 ~256 GB/s 同量级——这一项确实是权重带宽。**相对 dense 是 7.8×。**

**没到 ~1.05 ms 的原因**：3.46 ms 那个旧数字是 13 层 store 上测的，本轮同配置复现到 5.11 ms
（GPU 时钟状态不同，相隔一天），所以不能拿「3.46 − 2.36」推剩余。3.40 ms 里 `blk.13` 的 387 MiB
已经不在，剩下是 14 层 × 每层 6 个稀疏 matvec + norm/conv/GDN ≈ 200 多个节点的**建图/调度/launch**，
不是权重带宽。要再往下必须融合 sublayer（`GGML_OP_CHIP_SUBLAYER`），稀疏存储已榨干。

**bit-exact 0 差异**（uint32 逐位）：同 dll，A=`k0v5a`+`LLAMA_CHIP_SPARSE=0`、B=同文件+`=1`、
C=`k0v5`（84 个 dense 已删）+`=1`。32-token batch（tinyBLAS 树）与单 token decode（32-lane vec_dot 树）
各 **48 文件 / 7,572,928 float32 → A vs B、A vs C 全 0 差异**；与历史 13 层 dense dump 也 0 差异。
c-crc32（fullhead）：logits bit-equal **642496/642496**、argmax **4166/4166**、teacher forcing
**4165/4165**、margin **min 184 / median 552**。

### M4c（code_token）：chip token 不走 lm_head

参考规则（`circuit/chipformer/rv/chat.py`）：

```python
code = hc[code_dims]                  # 21 列 = NFAM(5) + NPAY(16)
bits = (code.float() > 0)             # 符号就是 bit，KAPPA = 2^-20
fid  = sum(b << i for i, b in enumerate(bits[:NFAM]))
p    = sum(b << i for i, b in enumerate(bits[NFAM:]))
t    = core.decode_code(fid, p, text) # 查表
return t - TEXT_TOKEN if is_text(t) else v0 + t
```

落地在 `llama_context::decode`：列与表全部来自已有的 `llama_chip_emb` blob（`code_cols[21]`、
`ranges[15]`、`ctrls[8]`、`nfam/npay/v0`），只有 text-echo 家族 "Q" 的 260 个 printable id
从 checkpoint 的 `chipformer.json` 生成成 static 表（tokenizer 属性，不是权重）。
hidden 取 **源层 13 的输入**（= 第 12 层输出 = `layer_inp[13]` = 参考实现的 `hc`）。
**把该位置整行 logits 写成 one-hot**，所以所有读 `llama_get_logits()` 的调用方不用新 API。
开关 `LLAMA_CHIP_CODE_TOKEN=0/1/force`，只对 batch=1 + 无头段生效。

**关键发现：解释不了的不是「口径」，是初始 recurrent state。**

| 项 | 结果 |
|---|---|
| HF 的 `hc` vs llama.cpp `layer_inp[13]`（21 列） | **max\|d\| = 0**，符号 32/32 相同 |
| 从**零 GDN state** 跑 32 个 chip token | 412227 / 478212 这类码，**0/32** |
| 同一序列 batch=1 / 32 / 128 预填 | 三者 argmax 都是 **478212**（同一个错） |

c-crc32 的 teacher forcing 有**初始 recurrent state**（13 层 × 524,288 float，非零 30,788）。
零 state 跑出来的是另一条轨迹（模型认为程序执行了非法指令 → `FLT(0)` = 478212），与 hidden 读取方式无关。

**注入初始 state 后**：窄头 `k0v5-q4km-mixed.gguf` + chip 段 0:14 + 逐 token decode 4166 步 →
**teacher forcing 4165/4165**，且 argmax 与 fullhead 的 `lm_head` argmax **4166/4166 逐位一致**
（两个不同文件、不同头、不同数值路径）。开关对比 mode=1 63.4 s vs mode=0 65.8 s，**开销在噪声内**。

**窄头能替代 fullhead**：省 **3.383 GiB**（`output.weight` 的 chip 区，394,176 行 × 4608 × 2 B）。
生产形态用 `k0v5-q4km-mixed.gguf`（7.663 GiB），fullhead（11.046 GiB）只留作全量头口径的对照。

**限制**：只对 batch=1 生效（prefill 口径仍由 lm_head 回答）；printable 表在代码里，换 tokenizer 要重生成；
`QWEN35_CHIP_LAYERS = 13` 仍是硬编码（`chipformer.json` 的 `chip_layers`，GGUF 与 blob 都不带）。
