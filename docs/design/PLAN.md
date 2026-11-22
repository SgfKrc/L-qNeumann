# 方案与计划

> 状态：**现行**
> 更新日期：2026-11-21
>
> 适用范围：本项目的技术路线、阶段划分、验收判据与风险应对；读者为实现者与评审。
> 相关：立项 [PROPOSAL](PROPOSAL.md)、引擎设计 [engine/README](../../engine/README.md)、边界 [ENGINE-LIMITS](../limits/ENGINE-LIMITS.md)。

---

## 1. 方案

### 1.1 总路线

**在 llama.cpp 里实现 chip 电路的执行器，全栈单一数值实现。**

```
GGUF（14.20 GiB，一份文件）
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
| **验收对 IR** | 判据是 `circuit/q35/ir.py` 的 `evaluate`（numpy float64） | 纯 Python，不依赖被验收的实现 |

### 1.3 组件职责

| 组件 | 职责 | 状态 |
|---|---|---|
| `circuit/` | 电路定义、黄金参考、权重生成 | 已迁移（34 文件） |
| `engine/` | ggml 的 chip 执行器 | **待实现** |
| `verify/` | 对 IR 的 bit-exact 验收 | 待建 |
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
| **M1** | 1–2 | 状态布局映射 | ✅ **已完成**：结论见 §2.0 | — |
| **M2** | 3–5 | **单层前向 + 数值对齐** | ✅ **已完成（2026-11-22）**：13 层全部 bit-exact | M1 |
| **M3** | 6–8 | 13 层链：整条 chip 链在 llama.cpp 里跑通 | ✅ **已完成（2026-11-22）**：程序 4165/4165，308 ulp 可接受 | M2 |
| **M4** | 9–12 | 性能：基线 + offload + 两端裁剪 | ⚠️ **部分完成（2026-11-22）**：A+B 已落地（−66%/−56%），150 μs 在 dense 下不可达（需 C 稀疏存储），见 §2.0.e/§2.0.f | M3 |
| **M5** | 13–14 | 端到端：chip + Qwen 同进程跑完整 46 层 | 输出 logits 与参考一致 | M3 |

**总 14 周（约 3.5 个月）。**

### 2.0.f A+B 完成：两端裁剪（2026-11-22）

**A — chip 段不建 lm_head**（`--split-layer-range 0:14`）：`needs_output(0,14,46)==false` → loader 不读
`output.weight`，图里 `needs_logits=false` 整段不建。

**B — text 段只算 248K 行**：新 GGUF `k0v3-q4km-mixed-split.gguf`（16.336 GiB = 14.205 + 2.13），
新增 `output.weight` = `token_embd` 前 248320 行；`n_vocab` 保持 642496（chip token 要查表）。

**实测**（单 token decode，median of 20，8 线程 CPU / mmap）：

| 配置 | CPU | GPU |
|---|---:|---:|
| 46 层 + 全量 tied 头（现状） | **380 ms** | — |
| 46 层无头 | 253 ms | — |
| **chip 段 0:14（无头）** | **128 ms**（−66%） | **25.66 ms** |
| text 段 + 全量 tied 头 | 248 ms | — |
| **text 段 + 248K 头** | **168 ms**（−56%） | **28.56 ms** |

**显存**：chip 段 **5468 MiB**、text 段 **6597 MiB**（`n_batch ≤ 32`），各自都装得进 8 GB。

**bit-exact 三组全过**：

| 组 | 内容 | 结果 |
|---|---|---|
| A | chip 段 vs 全模型，`layer_00..14` | **15 × 147456 全等，max\|d\| = 0** |
| B | tied vs split，`logits[:248320]` | **248320/248320 全等，max\|d\| = 0** |
| M3 契约 | A+B 后的 dll 跑 M3 配置 | **14 层 × 589824 float 全等** |

**为什么 101/95 ms 没达到**：254 ms 是「46 层、不请求 logits」的耗时，**不含 5.52 GiB 输出头**；
101/95 按字节线性折算，但实测各块带宽不同（chip 41 GB/s、Qwen 29 GB/s、头 42 GB/s）。
**相对改进与目标一致**（−66% / −56% vs 目标 −63% / −60%）。

**A 的修复**（三处「没人读的值不要写」）：chip 段 0..13 无 attention 层 → KV 消费者为 0 → sched 不分配
buffer，而 `set_input` 无条件写。修了 `mem_hybrid::set_input`（k/v_idxs、mask）与 `pos::set_input`。

**一个真正的 bug**：`build_inp_embd` 用 `tok_embd != nullptr` 判断「有 token 路径」，但范围不从 0 开始时
loader 只把该 tensor 标成 lazy、对象还在 → `get_rows` 用**未初始化的 `inp->tokens`** 越界读复用 buffer。
实测：text 段同一命令跑两次 logits 不同（argmax 69811 vs 148466）。改成 `tok_embd && hparams.needs_tok_embd()`。

**两个遗留问题：都不是本项目或 K-Llama 的缺陷，是探针 + llama.cpp legacy 接口。**

| 遗留 | 裁定 |
|---|---|
| 1. relay 段 ulp 级不等价 | **不存在（误判）**；「bf16 vs f32 走不同 Q4_K kernel」不成立 |
| 2. 含 attention 的 relay 段非确定 | **探针缺陷**；K-Llama 无责 |

根因是一行代码：`src/llama-batch.cpp:1576` 的 `llama_batch_compat::init`（**baseline b94959a64 与 HEAD
逐行相同，K-Llama 没碰**）——embedding 批次的 M-RoPE 位置是 **section-major 4 段**，`llama_batch_init`
只给 1 段，兼容层照读 4 段 → **越界读堆**。qwen3.5 是 `LLAMA_ROPE_TYPE_IMROPE` → `n_pos_per_embd == 4`。

**纯净上游对照组**（`git worktree add H:/llamacpp-up b94959a64`，无任何分段代码）：

| 配置 | 结果 |
|---|---|
| upstream + embd + **1 段 pos** | **两次不同**（`layer_18` 起、31/32 行、列 [0,2559]、max 2.87e-1） |
| upstream + embd + **4 段 pos** | **两次逐位相同**，且与 token 路径 **46 层 + 642496 logits 全 bit-equal** |

**所以遗留 2 在 upstream 上原样复现，遗留 1 的因果链整体不成立。**

**K-Llama 反而早修好了真实链路**：`split_expand_pos()`（`llama-context.cpp:2763`）在两处 hop 都按同一规则
展开 pos；12 个分段提交没有动 `llama_batch_compat` / `ggml_backend_sched` / `llama_memory_hybrid`。

**修好后三段 relay 全部 bit-equal**（只改探针，llama.cpp 一行没动）：

| 配置 | 结果 |
|---|---|
| `range 14:18` embd（5 层） | 5/5 层 147456/147456 bit-equal |
| `range 0:18` embd（19 层） | 19/19 层 bit-equal |
| `range 14:46` embd（33 层） | 33 层全 bit-equal；`logits` 642496/642496 bit-equal |

**一个措辞纠正**：`chip_dims_offset=2560` 是 **hidden 偏移**（`merge.py:254`），所以 `hidden[0:2559]` 是
**文本区**、`[2560:4607]` 是**电路区**。差异落在文本区，是因为**电路区 2048 维不被 Qwen 写**——
不是电路层在 L14 前有特殊处理。

**上线前唯一还必须做的**：真实 split chain 端到端（`tests/split-smoke` / `tools/split-stage`），
本次证明的是单元口径下 relay == 全模型。

### 2.0.e M4 结论：150 μs 在 dense 下不可达，必须换稀疏执行器

**14.19 GiB 的真实账本**（避免漏算）：

| 块 | GiB | 占 |
|---|---:|---:|
| chip 13 层（bf16） | 4.92 | 34.7% |
| kernel 1 层 | 0.38 | 2.7% |
| Qwen 32 层（Q4_K） | 3.37 | 23.7% |
| **token_embd / lm_head（bf16 tied）** | **5.52** | **38.9%** |

**embedding 是最大的一块**（642,496 行 × 4608 维 × 2 bytes = 5.52 GiB），比 chip 和 Qwen 都大。

**150 μs 的数学**：13 个 chip 层光「把 4.92 GiB 权重读一遍」就要——

| 设备 | 时间 |
|---|---:|
| GPU 理论带宽（256 GB/s） | 20.6 ms |
| **GPU 实测（164 GB/s 有效）** | **32.2 ms** |
| CPU（41 GB/s） | 138 ms |

**150 μs 需要 35.2 TB/s 有效带宽**，是实测的 **214 倍**。

**它只在稀疏执行下有意义**：nnz 1.45M MAC ≈ 23 MB → 需要 155 GB/s，落在带宽内。

**实测基线**（`llama-bench`，ms/token，每变体独立进程）：

| 配置 | ms/token |
|---|---:|
| 纯 CPU `-ngl 0` | **376** |
| **chip 0–13 上卡** | **278** |
| Qwen 14–45 上卡 | 338 |
| chip + Qwen 都上卡（8.67 GiB） | 478（超显存回退，比全 CPU 还慢） |
| 全上卡 `-ngl 46` | 卡死 |

**两个必须记住的事实**：

1. **`-ngl N` 是 upstream 语义：从尾部 offload**（`i_gpu_start = n_layer + 1 - N`）。
   `-ngl 14` **不是** chip 上卡，而是 layer 33–45 + 输出层上卡 → tied lm_head 拖上 GPU → 必崩。
   **chip 上卡只能用 `-ot 'blk\.(?:[0-9]|1[0-3])\..*=CUDA0'`。**
2. **瓶颈是带宽（100%）**：batch 1→4 总时间完全不变（37.8→39.7 ms）= 读一遍权重。
   launch 不显著（CUDA graph 已把 ~1000 次 launch 压成一次，固定开销 ~0.3 ms/前向）。

**46 层 ≈ 900–1000 kernel/前向**（13 chip 层 ≈ 260–290，32 Qwen 层每层 ~20）。
融合瓶颈 = **全宽 grid barrier**（rms_norm/l2_norm 跨 CTA 归约、ssm_conv 跨 lane、ssm_scan 顺序依赖）
——这正是 Triton 那次 24→1 慢 14% 的同一物理约束。

**极端乐观的叠加**：278 → 修 graph 税 250 → 加 lm_head 量化 162 → 加 text 跳过 chip 130 ms。
**仍是 150 μs 的 870 倍。**

**所以 M4 的真实结论是**：offload 能让 46 层跑得动（B1 已证明），但**不能让 chip 步快**。
**要快，必须回到稀疏执行器**（chip 段 32 ms → 0.2–0.3 ms）。

### 2.0.d M3 行为验证：程序 4165/4165，308 ulp 可接受

**结论：308 个 float32 ulp 不会让电路算错，不需要修。**

| 程序 | 可验证位置 | 命中 | 最小 margin |
|---|---:|---:|---:|
| call | 474 | **474/474** | 173.7 |
| **c-crc32** | 4165 | **4165/4165** | **184.0** |

与仓库自带 qemu golden 三段论闭环：`c-crc32` exit 低 8 位 **57**、`c-crc32p` 输出 `crc32=414fa339`，
模型走出来的 token 流 == `ref.Machine` 的 token 流 ⇒ 程序输出 == golden。最后位置预测的是 `HALT`。

**机制**：`merge.py` 的 `norm[d0:] = -1.0` 只给 21 个 code 位 + tau 设 gain，第 12 层之后
**live 只有 22 列、exact 差 0 个 cell**——不是「差得小」，是**逐位相等**。

**比较器余量**：13,905,920 个 gate 输入，`b != 0` 占 **93.73%**（最小 \|b\| = 1 → K=128 缩放后
bf16 半量子 0.5，实测放大到 4.9e-04 → **1024× 余量**）；`b == 0` 占 **6.27%**——唯一没有 bf16
量子的位置，`relu(0+ε)=ε` 而不是翻转。

**「final 730 bad / max 15」是测量假阳性**：`m3_compare.py` 比了全部 3208 个 slot，但 exit 层会把
不在 `c.keep` 的 dim 清零。拆分后 **live 22 列 bad 0**，730 全部是 dead 列（正好是 exit-clear 的
730 个 dim）。那 19 个带残差的死列对 final norm 的 RMS 影响 2.1e-14；即使真差 1 ulp，传到 logit
是 1e-13，而最小 margin 是 173.7/184.0——**差 15 个数量级**。

**308 的口径**：13 层累计值，去重后 **72 个不同 dim、3438 个 cell**，全是
`x.rs1[*] / x.rd[*] / pc[*] / uop=* / add.d* / mul.p* / div.* / drom.*`，没有一个是 11 层以上都错的。

### 2.0.c M3 结论：真实布局下的 185 个偏差修到 0

**M3 走了真实 `llama_context`（零改源码，用 `llama_set_embeddings_layer_inp`），生产路径零误差**：

```
ggml(GGUF weights)  bad 240/635
llama_context       bad 240/635
ggml vs llama identical slots: 635/635   max|d| 0
```

**但逐位一致没拿到**（L0 185/635）。分离实验定位到根因：

| 变体 | 布局 | L0 bad |
|---|---|---:|
| **A**（M2 基线） | hidden 7168，无 slots | **0/641** |
| **B**（真实） | 4608，1812 复用，overlay | **185/635** |
| **C**（= B，overlay 不写进图） | 同上 | **185/635** |
| **D**（= B，每 dim 独占） | 18432，3995 独占 | **0/641** |
| **B20**（ballast 2^40→2^20） | 同 B | **185/635** |

**A = D = 0、B = C、B20 = B** → 唯一变量是 **slot 复用**。overlay、ballast、hidden、m/off 全部无辜。

**具体**：L0 非零的 token 特征（`isR`/`isGO`/`isNT`/`w.b*`）被让给后面 dim 时没有 handover——
`alloc.clears` 只有 level 8 / 10 两个键，L0 一个都没有。被污染的槽位改了 `attn_norm` 的分母，
所有 head 的输入偏一点，阈值敏感的比较器（`n>k`）和半衰计数器（0.5/0.25）翻转。

**修复**：`emit.py` 新增 `live_token_dims()` + `allocate(pinned=)`，`merge.py` 用 corpus 的 token
集合算 pinned（**284** 个）。

**结果**：

| 项 | 结果 |
|---|---|
| **L0（真实布局）** | **185 → 0 / 635，max\|d\| = 0** |
| **13 层逐层** | **全部 0 bad**（6445 个 IR dim） |
| `n_slots` | 1812 → **1935**（+123） |
| `kernel_base` | 4391 → **4514**（< 4608 ✓） |
| M2 的 0/6965 | **没破** |
| `test_rv_overlay` | **6 passed** |

**M3 漏的 init**：`core.snapshot` 的初始 recurrent state 没喂给 llama_context，量化后贡献
**55–57 个 bad**（240−185）。真注入待重出 GGUF 后做。

**那 7 个不是 bf16 下限**：放宽到 corpus 的 284 个后一起消失，**精确 0**。

### 2.0.b M2 结论：13 层全部 bit-exact

**端到端 0 / 6965 dims 差异**（3 条轨迹，每层独立 ggml 上下文、residual 经 numpy 传递）：

| 轨迹 | tok | bad | max\|d\| |
|---|---:|---:|---:|
| call | 128 | **0/6965** | **0** |
| datarom | 19 | **0/6965** | **0** |
| signext | 41 | **0/6965** | **0** |

**conv window 跨 token**：4 个连续 token，max\|diff\| 全 0，回看深度逐级展开
`{0}` → `{0,1}` → `{0,1,2}` → `{0,1,2,3}`。

**decay 三值**：ggml 的 `exp(g)` 取值集合 = `{0.0, 0.499999999, 1.0}`，三值全中；
401 个「清零后 1–5 token 内发生读出」事件全部 0 差异。

**L2/L10 被补上**：`div` 汇编激活 L2（`quo*` + `vx.wrom*`），`datarom`（带 `.rodata` 的 DROM 读）
激活 L10（`drom*`，2048/2048 满）。

**L6/L11 只测到 FFN**：head 在全部 10 条轨迹下恒为 0（L11 需要 `bridge.Session` 的 tool-call
状态机，L6 需要 bundle/vx 协议）。恒 0 时两边都是 0，**不计入已验证**——它们的 head 属于
M3/M5 的端到端范围。

**M2a 的一个 bug 已修**：`Qwen3_5RMSNormGated` 是**标准 `x·w`**（weight=ones → 增益 1），
不是 `(1+w)`。按 `(1+w)` 会多乘 2，制造 22 个假差异；改正后 22 → 0。三种 norm 的语义：

| norm | 形式 | emit 权重 | 增益 |
|---|---|---|---|
| `input_layernorm` / `post_attention_layernorm` | `x·(1+w)/rms` | 0 | 1 |
| **`linear_attn.norm`** | **`x·w/rms`** | **ones** | **1（不是 2）** |
| `final norm` | `x·(1+w)/rms` | −1 | 0 |

### 2.0 M1 结论：容器直接用，不改分配代码

每层 `n_embd_s = ssm_d_state × ssm_d_inner = 128 × 4096 = 524,288`（2 MiB F32），
而 chip 每层 value head ≤32 × 128 行 × 128 列 = **524,288**——**一字不差**。

原因：merge 时把 chip 层前置成 `linear_attention`，与 Qwen 层共用同一份 GGUF 元数据
（`linear_num_key_heads=16`、`linear_num_value_heads=32`、`linear_key_head_dim=128`、
`linear_conv_kernel_dim=4`）。

三个结论：

| # | 结论 | 依据 |
|---|---|---|
| 1 | **容量够**（每层 2 MiB，38 个 recurrent 层合计 s ≈ 76 MiB） | `llama-memory-recurrent.cpp:103-104` 的 `n_embd_s()` |
| 2 | **布局 1:1**（转置存储后，"行" = 一个 key 索引的 128 个 value 值） | `ggml-cpu/ops.cpp:11238` 的 `s_out[j*S_v+i] = S[i][j]` |
| 3 | **conv 窗口现成**（4-tap = 3 列） | `llama-graph.cpp:4296` 的 `build_conv_state_store` |

**要自己写的只有两件**：chip 层的图构建（M2）、decay（`ggml_mul` + one-hot 覆盖）。

**不用自建 buffer**：自建 ≈ 重写 500–800 行 memory 语义 + 1.5–2 周，且从此所有
`seq_rm/seq_cp/state_read/write/offload` 自己背。

**M4 的已知欠账**：recurrent 的写回是**整块 cpy**（每 token 每层搬 2 MiB），而 chip 只改一行。
M4 必须改成「局部 `set_rows` 写 + 不清零整块」，否则带宽吃掉提速。

### 2.1 为什么 M2 是生死线

M2 的难点在「**谁适应谁**」，而不在「实现一层前向」：

- `ir.evaluate` 是电路的规范（numpy float64）
- ggml 的数值行为由算子内部决定（归约顺序、累加精度、FMA）
- 两者在最后几位可能不同

**如果不同，不一定是让 ggml 服从 float64**——也可能是电路本身对舍入太敏感，需要改。

所以 M2 拆成三段：

| 段 | 做什么 | 产出 |
|---|---|---|
| **M2a** | ✅ **已完成（2026-11-22）**：level 1 的 ggml(bf16) 与 `ir.evaluate` **0 / 6965 dims 差异** | 偏差表见下 |
| **M2b** | 偏差影响关键路径时，调整电路使其鲁棒（ballast 降位、值域收紧、2 的幂替代一般整数） | 改后的权重 |
| **M2c** | 验收：改后的电路在 ggml 下对 `ir.evaluate` 逐位一致 | 0 差异 |

**M2a 已通过（2026-11-22）**：level 1（13 key / 17 value head，真实活跃）的 ggml bf16 CPU 后端
与 `ir.evaluate` **0 / 6965 dims 差异，max|diff| = 0**。

| 对比 | 不同 dim / 6965 | max abs |
|---|---:|---:|
| **ggml bf16（CPU）** | **0** | **0** |
| ggml f32（CPU） | 7 | 1.19e-7 |
| HF bf16 prefill / decode | 0 / 0 | 0 |

- 唯一差异源是 **f32 路径不做 bf16 舍入**（7 个"常数 1"的 dim 差 1.19e-7）；
  **生产路径（GGUF 里 chip 层就是 bf16）是 bit-exact 的**
- 逐算子全部 0 差异，仅 `ggml_gated_delta_net` 4.2e-8（fp32 累加顺序），整层输出相对差 1.1e-19

**所以 M2 不需要改电路，也不需要改 ggml 配置。** M2b 的预案（ballast 降位、值域收紧）暂不启用。

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

| # | 风险 | 影响 | 应对 | 止损 |
|---|---|---|---|---|
| **R1** | ggml 的 bf16 累加无法与 `ir.evaluate` 的 float64 逐位一致 | **M2 卡死** | M2a 先测偏差分布；偏差只在非关键路径则可接受；影响 ballast 读取则 M2b 调整电路 | M2b 也解决不了时**重新评估方案** |
| **R2** | `llama_memory_recurrent` 的布局装不下 chip 状态 | M1 卡死 | 自建 state buffer（不依赖 recurrent 的布局） | 自建成本过高时重新评估 |
| **R3** | ggml 的 kernel 性能不如 Triton | M4 达不到 150 μs | 移植 head 拆 CTA、load 交织的思路 | 目标降到 200 μs（与 Triton 持平） |
| **R4** | `ggml_set_rows` 不保 bf16 精度 | ballast 写入失真 | 手写 scatter kernel | — |
| **R5** | 完整模型 14.20 GiB 在 8 GB 卡上的 offload 调度 | M5 跑不动 | 用 `-ot token_embd.weight=CPU` + 分层 | 减到只跑部分层验证 |

## 4. 里程碑判据

| 里程碑 | 判据 | 不可退让 |
|---|---|---|
| **M2 通过** | 改后的电路在 ggml 下对 `ir.evaluate` 逐位一致（0 差异） | ✅ **是** |
| **M3 通过** | golden trace 全过（130 个测试） | ✅ **是** |
| **M4 达标** | chip 步 ≤ 150 μs | 可放宽到 200 μs |
| **M5 通过** | 46 层跑通，logits 一致 | ✅ **是** |

**位精确是硬门槛**（M2/M3/M5），**性能是软目标**（M4）。

## 5. 不做什么

| 不做 | 理由 |
|---|---|
| 跨框架（PyTorch ↔ llama.cpp） | 数值差消除不了；K-Llama v1 用 27,825 行证明过 |
| 新硬件设备（AES/SHA/zlib） | token 带宽 15–16 bit 是硬墙 |
| 流水线（IF/ID/EX/MEM/WB） | 分支率 26.5%，实测负收益 |
| persistent kernel | 实测 1.017×，阶段 2 会慢 14% |
| 训练 | **电路全程无训练**（QwenPU 的前提，电路是编译的）；**语言层可选 LoRA 适配**（不碰 0–13 层，见 [BRIDGE-LORA](BRIDGE-LORA.md)） |

## 6. 待评估的方向

| # | 方向 | 收益 | 归属 |
|---|---|---|---|
| 1 | **payload 15 → 16** | tpi −3~5%，地址效率 80.7 → 84.2% | token 编码 |
| 2 | **让「族」可读**（族 id → Qwen embedding） | Qwen 理解程序意图，可能减 text token | **性能优化**（见 [TOKEN-ENCODING](TOKEN-ENCODING.md) §6） |
| 3 | LoRA 训练桥（`bridge`） | 模型学会用电路结果 | 语义增强 |
| 4 | 视觉塔 → 电路 | 看图算数 | 扩展 |
| **5** | **C：chip 层稀疏存储**（99.96% 为零，5.30 GiB → 5–10 MB） | **150 μs 的唯一路径** | **性能优化（M4）** |
| **6** | **D：chip token embedding 程序化生成**（不存表） | 省 3.38 GiB 磁盘/RAM | ✅ 已完成 |
| **7** | **A+：补 `blk.13` 进稀疏**（nnz **0.0021%**，比 `blk.0-12` 还稀 10–30 倍） | **3.46 → ~1.05 ms（3.4×）** | 1–2 天（最便宜的 69%） |
| **8** | **C：融合 op**（`GGML_OP_CHIP_SUBLAYER`，每层 1–2 kernel） | 1.05 → ~0.55–0.65 ms | 2–3 周 |
| **9** | **D：抽薄引擎**（8,000–10,000 行） | 0.55 → ~0.25–0.35 ms（拿掉 0.3 ms 建图税） | 4–8 周（**C 之后评估**） |
| **10** | **不做：删通用性**（179 arch / 5,900 行 split / KV 家族） | **~0 ms**（它们不进热路径） | 不做 |

**薄引擎调研的结论**（详见 `_k0v5/REPORT.md` 与本节）：

- 3.46 ms 的构成：**`blk.13` dense 带宽 2.36 ms（69%）**、13 层稀疏 0.42 ms、节点税 0.22 ms、建图 0.30 ms
- **通用性税里只有两项是真成本**：融合白名单太窄、backend sched/建图；其余（179 arch / 408 tensor 名 / 5,900 行 split / KV 家族）全是死代码
- **M4 的「全宽 grid barrier」结论要修正**：对 dense 成立；对 batch=1 的稀疏 GEMV，归约可由每个 program 冗余算、barrier 可消——**所以「13 层融成 1 个」不能，「每层 1–2 个」能**
- **手搓 kernel 的真实形态不是「写更多 kernel」，是「写更少的 kernel + 每条配一条 bit-exact 证据链」**

