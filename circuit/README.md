# circuit/ —— 电路定义与黄金参考

> 状态：**现行**
> 更新日期：2026-11-21
>
> 适用范围：电路定义与黄金参考：纯 Python + numpy，不依赖 torch；读者为 engine 实现者与验收者。
> 相关：机制 [ARCHITECTURE](../docs/design/ARCHITECTURE.md)、验收 [verify](../verify/README.md)。

---


> **纯 Python + numpy，不依赖 torch。** 这是「电路长什么样」的唯一事实源。

## 一、这里是什么

`circuit/` 保存**电路的静态定义**和**黄金参考实现**，不包含任何执行器。

```
circuit/
└── chipformer/          完整包（86 文件：55 纯 Python + 31 需 torch）
    ├── rv/              RV32IM 电路
    │   ├── spec.py      token 族（FAMILY_SIZES）与 trace 格式
    │   ├── core.py      13 层电路的 IR 定义（哪些 head、哪些 unit）
    │   ├── ref.py       参考实现（Python 版的电路前向）
    │   ├── corpus.py    overlay 的精确性测量（三条条件靠测量，不是构造）
    │   ├── calc.py      calc 固件的参考运行
    │   ├── elf.py       ELF 加载
    │   └── vx.py / vxcirc.py  向量单元的参考 + 电路
    ├── q35/             Qwen3.5 侧
    │   ├── ir.py        IR 与 evaluate（numpy float64，黄金参考）
    │   ├── emit.py      把 IR 变成权重（需 torch）
    │   ├── kernel.py    kernel 插槽（K0/K1，需 torch）
    │   ├── overlay.py   列复用
    │   └── merge.py     合并成 checkpoint（需 torch）
    └── dsl/             表达式图（纯 Python）
```

**验收路径不碰 torch**：`ir.evaluate` 是 numpy float64，`ref.py` 是 Python 参考机。
需要 torch 的只有**权重生成**（`emit` / `merge` / `kernel`），它们不参与验收。

**三个口径**（实测 `core.build`，`2026-11-22`）：

| `vx` | 层数 | key head | value head |
|---|---:|---:|---:|
| **`True`（生产配置）** | **13** | **142** | **266** |
| `False` | 11 | 104 | 197 |

## 二、为什么是「纯 Python」

**因为它是验收标准，不能自己依赖被验收的东西。**

`ir.evaluate` 是 **numpy float64** 的精确实现：

```python
def evaluate(g: Graph, tokens: list[int], ...) -> np.ndarray:
    """Residual stream after every node: (n_tokens, n_dims), exact in float64."""
```

- 它**不依赖 torch**（torch 只在可选的 `kernel_check` 里用，且是函数内 import）
- 它算的是**精确浮点**，不是 bf16
- 所以它是**独立的第三方参考**

**ggml 实现的验收标准就是它**：ggml 跑出来的结果，必须与 `ir.evaluate` 逐位一致。

## 三、和 engine/ 的关系

```
circuit/          定义电路、算黄金参考、生成权重
    │
    ├── ir.evaluate ──→ 验收标准（engine 要对齐它）
    │
    └── emit/merge ──→ 权重 ──→ GGUF ──→ engine/（ggml 执行器）
```

**circuit/ 是「图纸 + 标准答案」，engine/ 是「按图纸施工」。**

## 四、这里没有的东西

| 不在这里 | 在哪 | 为什么 |
|---|---|---|
| Triton 执行器（`fusedexec.py`） | `_reference/` | 归档，不是生产路径 |
| PyTorch 张量工具（`tiny.py`） | 不进仓库 | 全 llama.cpp 后不需要 |
| 量化脚本 | `tools/` | 一次性工具 |
| 测试 | `verify/` | 验收装置 |

## 五、许可与来源

这些文件来自 [QwenPU](https://github.com/divagr18/QwenPU)（上游，divagr18），本项目做了以下改动：

- `q35/kernel.py` —— 新增（K0/K1）
- `q35/overlay.py` —— 新增（列复用）
- `rv/spec.py`、`rv/ref.py`、`rv/core.py` —— SB/SH 记录压缩（4 token → 2 token）
- `rv/measure.py` —— 跟进 SBH 的 token 数变化

改动原则：**保持纯 Python**，不引入 torch 依赖。
