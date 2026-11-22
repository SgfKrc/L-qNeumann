# 构建与运行

> 状态：**现行**
> 更新日期：2026-11-21
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
