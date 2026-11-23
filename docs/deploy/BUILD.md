# 构建与运行

> 状态：**现行**
> 更新日期：2026-11-23
>
> 适用范围：构建与运行：riscv 工具链、merge、量化、测试口径；读者为复现本仓库的人。
> 相关：基线 [BASELINES](../measure/BASELINES.md)。

---


## 一、环境（已验证）

| 组件 | 版本 | 备注 |
|---|---|---|
| Windows | — | 主开发环境 |
| WSL | Ubuntu-22.04 | calc 固件编译用 |
| Python | 3.12.10 | |
| torch | 2.13.0+cu126 | **先 `from ortools.sat.python import cp_model` 再 `import torch`**（否则段错误） |
| onnx | 1.16.2 | 1.23 产 IR v14，onnxruntime 1.27 只支持 v13 |
| riscv 工具链 | gcc 10.2.0 / binutils 2.35.1 / picolibc 1.7.4 | 见下 |

## 二、riscv 工具链（calc 固件）

**不需要 root。** 在 WSL 里：

```bash
# 1. 下载并解包（免 root）
cd ~ && mkdir -p rvroot && cd rvroot
apt-get download gcc-riscv64-unknown-elf binutils-riscv64-unknown-elf \
               picolibc-riscv64-unknown-elf libc6-riscv64-cross 2>/dev/null
for d in *.deb; do dpkg -x "$d" ~/rvroot; done

# 2. 裸名 symlink（Debian 交叉 gcc 调裸名 as/ld，会命中 x86 工具）
mkdir -p ~/rvbin
for t in as ld ar nm objcopy objdump size strip ranlib readelf; do
  ln -sf ~/rvroot/usr/bin/riscv64-unknown-elf-$t ~/rvbin/$t
done

# 3. 编译
bash /mnt/h/sgfd9/wsl_build_calc.sh /mnt/h/sgfd9/L-qNeumann/out calc
```

**三个坑**（都在脚本里处理了）：

1. Debian 交叉 gcc 调**裸名 `as`/`ld`** → 必须让 `~/rvbin` 在 PATH 前，或用 `-B"$HOME/rvbin/"`
2. gcc 10.2 **没有 zba** → `-march=rv32im`（功能等价，少 `sh1add` 合成，基线因此比官方高 ~7%）
3. picolibc 1.7.4 的 lib 在 `lib/rv32im/ilp32/`，**没有 `lib/release/` 那一层**

## 三、合并 checkpoint

```bash
cd qwenpu
python -u -m chipformer.rv.merge H:/sgfd9/models/Qwen3.5-4B \
  --out H:/sgfd9/models/q4-rv1-k0v2 \
  --vx --overlay --hidden 4608 --ballast 18 \
  --kernel H:/sgfd9/k0_kernel.npz --no-mute
```

**两个必须的参数**：

- **`--kernel`**：不加只有 45 层、`blk.13` 是 Qwen 层，与「46 层 + kernel」矛盾
- **`--no-mute`**：`fit_anti_sink` 要把 7.4 GiB 模型全量上 8 GB 卡，装不下

**`--hidden 4608 --ballast 18` 必须成对**（单独给 `--hidden` 会 `TypeError`）。

## 四、量化（混合：chip + kernel bf16，Qwen Q4_K）

```bash
# 1. HF → bf16 GGUF（约 4 分钟）
python v1_convert.py H:/sgfd9/models/q4-rv1-k0v2 D:/sgfd9/gguf --no-mtp

# 2. 量化（规则文件排除 blk.0-13）
/h/llamacpp-c/build-eval/bin/llama-quantize.exe \
  D:/sgfd9/gguf/k0v2-bf16.gguf D:/sgfd9/gguf/k0v2-q4km-mixed.gguf Q4_K \
  --tensor-type-file k0v2_chip_kernel_bf16.txt

# 3. 校验（必做——磁盘满时量化器静默截断且 exit 0）
python v1_verify.py meta D:/sgfd9/gguf/k0v2-q4km-mixed.gguf
python v1_verify.py cmp-gguf --layers 0-13 D:/sgfd9/gguf/k0v2-q4km-mixed.gguf
```

**tensor-type 规则**（`k0v2_chip_kernel_bf16.txt`）：

```
blk\.(?:[0-9]|1[0-3])\.=bf16     # chip 0-12 + kernel 13
token_embd\.weight=bf16          # 含 2^40，压不动
```

## 五、运行

```bash
# 纯 CPU
llama-cli -m D:/sgfd9/gguf/k0v2-q4km-mixed.gguf -ngl 0 -p "2+2=" -n 4

# GPU 部分 offload（token_embd 必须留 CPU，否则 CUDA 越界）
llama-cli -m D:/sgfd9/gguf/k0v2-q4km-mixed.gguf -ngl 4 \
  -ot "token_embd.weight=CPU" -p "2+2=" -n 4
```

## 六、测试

```bash
cd qwenpu
python -m pytest tests/test_rv_ref.py -q              # 14 秒，必跑
python -m pytest tests/test_rv_core.py -q -k calc     # 62 秒
python -m pytest tests/test_rv_core.py -q             # 15 分钟，交付前跑一次
python -m pytest tests/test_rv_fusedexec.py -q        # 23 秒
```

**注意**：`test_rv_fusedexec.py:18` 的 `CKPT` 指 `H:/sgfd9/models/q4-rv1-k0v2`。删掉那个 checkpoint 会让 3 个 merged 用例变成 skip（不是失败）。

## 七、llama.cpp 侧（`Lqn-Llama` 分支）

**fork 的是 K-Llama 的分段改造分支**（`rebase/onto-20261115`），不是上游 llama.cpp。

```bash
cd H:/llamacpp-c            # 已 checkout Lqn-Llama
cmake --build build-split   # CPU 验证用（bit-exact 全部在这个构建上）
cmake --build build-cuda    # GPU（RTX 4060 Laptop）
```

**三个必须知道的坑**：

1. **`build-split` 的 ninja 依赖记录基本是空的**（`ninja -t deps` 多数对象 deps=0）。**改任何 `.h`
   之后必须 `--clean-first`**，否则会产出 `.obj`/`.dll` 结构体布局不一致的构建，运行期直接崩。
2. **`qwen35.cpp` 与 `llama-context.cpp` 都在 `unity_9` 里**。改 `.cpp` 后只 `--target llama` 会
   「BUILD OK」但 dll 没变——**要删 `unity_9_cxx.cxx.obj` 再编**。
3. **`-ngl N` 是 upstream 语义（从尾部 offload）**。chip 层上卡只能用
   `-ot 'blk\.(?:[0-9]|1[0-3])\..*=CUDA0'`。

**跑 chip 段**：

```bash
# 生产形态（窄头 + 稀疏 store + code_token）
PATH="/h/llamacpp-c/build-split/bin:$PATH" LLAMA_CHIP_SPARSE=1 \
  llama-cli -m D:/sgfd9/gguf/k0v5-q4km-mixed.gguf --split-layer-range 0:14 ...

# 稀疏开关（默认开；关掉走 dense 路径，需要 k0v5a 那种 dense 全在的文件）
LLAMA_CHIP_SPARSE=0

# code_token 开关（默认 1：只在没有输出头时启用；force：有头也用；0：关闭）
LLAMA_CHIP_CODE_TOKEN=0|1|force
```

**注意**：`code_token` 只在 **batch=1 且该段没有自己的输出头**时生效；prefill 口径下仍由 lm_head 回答。
**初始 recurrent state 是程序的一部分**——逐 token 复现 c-crc32 时必须先注入（`llama_set_ssm_state`），
零 state 会走另一条轨迹。

## 六、构建陷阱：改了任何 `.h` 都必须全量重编

两个 build 目录（`build-cuda` 与 `build-split`）的 ninja 依赖记录是空的（`ninja -t deps` 多数对象
`deps=0`，尽管 `rules.ninja` 里有 `deps = msvc`）。后果：**改头文件不会触发重编**，增量构建会产出
结构体布局不一致的混合 ABI。它不报错、不警告，运行期才崩。

两次实测：

- 改 `llama-hparams.h` 之后，vocab-only 加载 SIGSEGV。
- 2026-11-23 在 `models.h` 里给 `llama_model_qwen35` 加了 5 个 `chip_sparse_*` 指针成员。`build-cuda`
  的 12 个 unity 在 11:20–11:21 重编了，但 `llama-model.cpp.obj` **没有**重编（ninja 报 `no work to
  do`），链接出的 `llama.dll` 里**该对象小 40 字节**。于是标准 Qwen3.5 模型（文件里没有 chip store，
  `n_chip == 0`，五个成员从未赋值）在 `load_arch_tensors()` 里解引用对象尾之外的野指针，**exit 139，
  崩在 `cls.output.bias` 之后**；而 k0v5 因为五个成员都被赋了值、写完立刻读回同一批地址，**反而
  自洽不崩**——**只测 k0v5 永远发现不了这个问题**。

规矩三条：

1. 改过任何 `.h`：`ninja -t clean` 之后**两个 build 目录都全量重编**。
2. 编完**核对 `bin/*.dll` 的 mtime 比 `src/` 与 `ggml/src/` 下所有源文件都新**。不核对就等于在测
   旧 dll 或者坏 dll。
3. **不要**跑不带目标的 `ninja -C build-cuda`：它会先跑 UI 资源步骤，断网时停在 `[2/103]` 并静默
   失败。用显式目标，例如 `ninja -C build-cuda bin/llama.dll`。
