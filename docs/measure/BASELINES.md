# 实测基线

> 状态：**现行**
> 更新日期：2026-11-23
>
> 适用范围：所有实测数字与口径；不同口径不可混用。
> 相关：边界 [ENGINE-LIMITS](../limits/ENGINE-LIMITS.md)、里程碑 [MILESTONES](MILESTONES.md)。

---


> 所有数字都可复现，口径注明来源。**不同口径不可混用。**

## 环境

| 项 | 值 |
|---|---|
| GPU | RTX 4060 Laptop **8 GB**（实际 free ~7.1 GB） |
| CPU | Intel Iris Xe（核显，未用） |
| 系统 | Windows + WSL(Ubuntu-22.04) |
| Python / torch | 3.12.10 / 2.13.0+cu126 |
| 磁盘 | H: / D: |

## 一、chip 步（**Triton v1 口径**：电路本身能跑多快）

**这一节量的是「纯 Triton 单实现」，不含通用推理引擎的开销。它给出电路自身的性能上限；
llama.cpp 路的数字在 §一之二，两节不可直接相减或相除。**

| 阶段 | μs/token | 倍数 | 口径 |
|---|---:|---:|---|
| 原始 FreeRun3（逐 token） | 830.6 | 1.00× | merged 真实电路，D=6914 |
| + O3-2 chip 段一次 `run(k)` | 359.4 | 2.31× | 同上 |
| + head 拆 CTA / 归并 / load 交织 | **204.5** | **4.06×**（累计） | 同上，B=1 |

| 口径 | baseline | 优化后 | 倍数 |
|---|---:|---:|---:|
| force3（3 次 load 上限） | 309.05 | **248.7** | 1.243× |
| 生产路径 B=1 | 267.4 | **204.5** | **1.307×** |
| 生产路径 B=8 | 1134.4 | **841.1** | 1.349× |

**bit-exact 验证**：B=1×200 步 460 万值、B=8×60 步 3680 万值、`_head4` 路径×100 步，
比较 resX/resY/keep/tok/11 个 S 表 + 11 个 lag 窗口，**全 0 差异**。

## 一之二、chip 步与显存（**llama.cpp 路**口径：含引擎建图/调度税）

单 token decode，`--split-layer-range 0:14`，median of 20，除非另注。**这张表里的数字不能与 §一
的 μs 互算**——它们量的不是同一件事。

| 配置 | chip 步 | 说明 |
|---|---:|---|
| dense GPU，14 层 | **26.48 ms** | 读 4.92 GiB 权重，纯带宽 |
| sparse GPU，13 层 store（`blk.13` 仍 dense） | **5.11 ms** | A+ 前（同轮对照） |
| **sparse GPU，14 层 store（A+ 后）** | **3.40 ms** | 同轮对照，**相对 dense 7.8×** |
| sparse GPU，14 层 store（次日再测） | **1.65 ms** | 同一配置换一天，GPU SM 时钟 2250/3105 MHz |
| sparse CPU 8 线程，14 层 | 27.35 ms | 无 GPU 时的退路 |

**⚠️ 这张表里 3.40 与 1.65 是同一配置换一天测的，差 2 倍——笔记本 GPU 的时钟/热状态造成的。**
**只有同轮对照可信**（5.11 → 3.40 是同一天同一轮）。引用 chip 步时必须同时给出「哪个配置、哪天、是否同轮」。

**A+ 的同轮对照 −1.71 ms（−33.5%）**。反算 blk.13 的带宽：387.1 MiB / 1.71 ms ≈ **226 GB/s**，
与 RTX 4060 Laptop 的 ~256 GB/s 同量级。3.40 ms 的剩余主体是 14 层 × 约 15 节点/层的**建图、调度与
launch**，不是带宽——继续稀疏化已无收益，下一步只能融合 sublayer。

**decode 的 GPU 时间实测分解**（Nsight Systems，单 token，chip 段 0:14，GPU busy **1.525 ms**）：

| 项 | 合计 | 实例 | 占 |
|---|---:|---:|---:|
| `k_chip_sparse_mm`（稀疏 matvec） | **0.903 ms** | 84 | **59%** |
| **`k_get_rows_float_vec`（SSM state 恒等搬运）** | **0.202** | 14 | **13%** |
| `rms_norm_f32` | 0.126 | 69 | 8% |
| `gated_delta_net_cuda` | 0.076 | 14 | 5% |
| 其余 | 0.22 | — | 15% |

**CPU 只占约 0.1 ms**（CUDA graph 已把 335 个 kernel 的 launch 压掉）。
**稀疏 matvec 的 903 μs 里，43 个短 kernel 只花 103 μs，41 个长尾花 800 μs**
（行 nnz 最长 979，warp 内工作量不均）。

**`concat` 的 GPU 时间**：非连续路径 `concat_non_cont` 22.59 → **11.06 μs/层**
（展平 block 几何后），但**只在 prefill 上出现**；decode 走 `concat_cont`（2.18 μs）。

**46 层的两端裁剪（A+B，单 token decode）**：

| 配置 | CPU | GPU |
|---|---:|---:|
| 46 层 + 全量 tied 头 | **380 ms** | — |
| 46 层无头 | 253 ms | — |
| **chip 段 0:14（无头）** | **128 ms**（−66%） | **25.66 ms** |
| **text 段 + 248K 行的输出头** | **168 ms**（−56%） | **28.56 ms** |

**显存**：`k0v5` 切段两段驻留 ≈ 18 MiB（chip，权重全在稀疏 store）+ 6597 MiB（text）
= **6615 MiB < 7106 MiB**，余量约 491 MiB（`n_batch ≤ 32`）。**不需要 offload**：两段各自全上卡，
段间只传 18 KB 的 hidden。

## 一之三、一句话的代价账

| 项 | 数字 |
|---|---|
| chip 层 dense 权重 | **5.30 GiB**（84 矩阵）→ 稀疏 **8.30 MiB**（619×） |
| chip token embedding | 394,164 行 → **826 B**（程序化，省 3.38 GiB） |
| 窄头替代 fullhead | 省 **3.383 GiB**（code_token 读出） |
| 生产 GGUF | `k0v5-q4km-mixed` **7.663 GiB**（fullhead 对照 11.046 GiB） |


## 二、calc 固件（C 层）

| expr | 官方 ELF (GCC13.2) | 新基线 (GCC10.2) | **最终** | vs 新基线 |
|---|---:|---:|---:|---:|
| 4+3 | 768 | 826 | **326** | **−60.5%** |
| 8*8 | 852 | 898 | **398** | −55.7% |
| 12/4 | 1832 | 1905 | **1215** | −36.2% |
| 1/3+1/6 | — | — | 2992 | −45% |
| 1/2+1/3+1/6 | — | — | 3855 | −59% |
| 3000 case 总 | — | — | 7,241,280 | −8.9% |

**正确性**：3000 随机 case **0 mismatch / 0 limited**；`test_rv_ref` 130 passed；
`test_rv_core -k calc` 6 passed。

**限量修复**：32 个原本报 `error (too complex)` 的 case 现在能算出正确值。

## 三、token 编码

| 指标 | 数字 | 说明 |
|---|---:|---|
| token/指令（11 程序加权） | **1.134** | 4 个 C 程序实测 1.378 |
| 理论下限 | **1.0** | 每条指令一个 commit token |
| SB/SH 记录 | **4 token → 2 token** | sieve 1.3928 → 1.1310（−18.8%） |
| 全局压缩 | **−14.1%** | 4 个 C 程序合计 |

## 四、checkpoint 与量化

| 项 | 数字 |
|---|---|
| 原始 4B | 8.70 GiB |
| 合并版（overlay + 瘦身 + kernel） | **22.87 GiB**，46 层 |
| `circuit.dims` | 6914 → **6966**（record-compress 后） |
| 量化版（混合 Q4_K + bf16，`k0v3`） | **14.205 GiB** |
| **`k0v5` 窄头（生产）** | **7.663 GiB**（84 个 chip dense 矩阵已删） |
| **`k0v5` fullhead（对照）** | **11.046 GiB** |
| **`k0v5-13L`（上一版，留对照）** | 8.041 / 11.424 GiB |
| **`k0v5a`（A/B 同文件对照：dense 全在 + 14 层 store）** | 12.961 GiB |
| chip + kernel 层 | **170 tensor 逐字节未变**（`k0v3` 口径） |
| chip token embedding | 394,164 行 → **826 B**（程序化） |
| offload 可用配置 | `-ngl N -ot "token_embd.weight=CPU"`；**chip 上卡必须用 `-ot 'blk\.(?:[0-9]|1[0-3])\..*=CUDA0'`** |

## 五、测试成本（用于压缩验证时间）

| 套件 | 耗时 | 用途 |
|---|---:|---|
| `test_rv_ref.py` | **14 秒** | golden trace，改动后必跑 |
| `3000 case` calc 回归 | **36 秒** | C 改动后必跑 |
| `test_rv_core.py -k calc` | **62 秒** | 电路改动后跑 calc 子集 |
| `test_rv_core.py`（全量） | **15 分钟** | 最后跑一次 |
| `test_rv_fusedexec.py` | 23 秒 | executor 改动后必跑 |

**策略**：小改动跑前两项（50 秒），全量只在交付前跑一次。

## 六、口径警告

- **§一的 204.5 μs 与 §一之二的 3.40 ms 不是同一个东西**：前者是 Triton 单实现的电路执行时间，
  后者是 llama.cpp 通用引擎跑同一段电路的端到端步进（含建图/调度/launch）。**不可相除、不可相减。**
- **chip 步的数字必须标「几层 store、dense 还是 sparse、同轮还是跨轮」**：跨轮比（如拿昨天的 3.46 ms
  比今天的 3.40 ms）会把 GPU 时钟状态差异算成人头——本仓库只采**同轮对照**。
- **官方 ELF（GCC 13.2）与新编译（GCC 10.2, rv32im）差 ~7%**——`-march=rv32im` 退档少了 `sh1add` 合成。收益必须用同口径基线报。
- **旧版 `q4-rv1-4608` 已删**（22.49 GiB），`q4-rv1-k0v2`/`k0v3` 是 22.87 GiB / 46 层的历史基线。
- **`359.4 ÷ 24 ≈ 15 μs/launch` 是错的**——15 μs 是单个 kernel 执行时长，launch 间隙只有 0.20 μs（全步 1.5%）。
- **`-ngl N` 是 upstream 语义：从尾部 offload**（`i_gpu_start = n_layer + 1 - N`）。`-ngl 14` 是
  layer 33–45 + 输出层上卡，不是 chip 上卡。
- **零 recurrent state 跑不出正确轨迹**：c-crc32 的 teacher forcing 依赖初始 state（13 层 × 524,288，
  非零 30,788）。零 state 下模型走的是另一条轨迹（`FLT(0)` = 478212），**与读数方式无关**。
