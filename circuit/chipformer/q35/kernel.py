"""Kernel slot: a matvec compiled to weights, run on kernel tokens.

The chip region owns the residual columns ``[off, off + m + 1 + n_slots)``; the kernel slot is the
block of columns right after it (:func:`slot_after_chip`), so a kernel never takes a column from
the circuit and the circuit never reads a kernel column. Two layouts are possible:

``shared=False`` (the default): 64 input columns, then 64 result columns, then ``TEMP`` scratch
    columns -- 144 columns. This is the layout torchwright compiles a 64x64 matvec into (its
    residual stream holds 64 input columns, one literal constant column and 64 output columns;
    the constant is the ballast here, so it costs no column).
``shared=True``: the results overwrite the input columns -- 80 columns. Every result unit then
    also subtracts the column's old value, one extra unit per lane; the subtraction is exact
    (the value is a bf16 residual value and the unit rides on the same gate).

Time reuse
----------
Every unit of the kernel layer is gated by the ``is_kernel`` token feature: ``silu(K * is_kernel)``
is exactly ``K`` on a kernel token and exactly ``0`` everywhere else, so a unit is the identity
``a * relu(1) = a`` there and dead elsewhere. Kernel weights therefore never act on a text, chip or
CPU token. In the other direction Qwen's weights cannot act on a kernel token either: the widen
pass (q35.merge) zero-pads every Qwen projection to ``hidden`` columns, so Qwen reads nothing from
a kernel column and writes nothing into one. This is why K0 puts the kernel in its own trailing
layer (option A of the design): no Qwen weight shares the layer, so nothing needs muting. Option B
(kernel units inside a Qwen layer) additionally needs the ``-KILL * is_kernel`` gate overlay.py
uses for text, because Qwen's own units there would otherwise also read the kernel columns.

What a kernel token is
----------------------
A real machine takes the 64 input values from MMIO (design section 4). K0 has no MMIO, so a
kernel token is a virtual token id ``KERNEL_BASE + i`` whose embedding row carries the i-th input
vector: :func:`residual_rows` builds those rows with ``emit.embedding_rows``. The token id is above
the chip vocabulary and above ``TEXT_TOKEN``, so it is nonzero in no chip feature except the
kernel's own.

Exactness
---------
A kernel does not meet the circuit's bf16-exact contract: its weights are arbitrary real numbers
rounded to bf16 values (``emit.check_bf16``), and its values are arbitrary real numbers, not the
integers the circuit's silu trick needs. :func:`run` therefore replays Qwen3.5's MLP sublayer in
bf16 and the caller compares against an exact reference; the error is a measured number, not an
assertion.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

from chipformer.dsl.graph import ONE, Expr
from chipformer.q35.emit import Layout, emit_layers, embedding_rows
from chipformer.q35.ir import FFN, Graph, Target

LANES = 64  # one residual column per matvec input and per result
TEMP = 16  # scratch columns for a multi-op kernel (K1 and later); unused by K0
KERNEL_BASE = 1 << 25  # first kernel token id: above the chip vocabulary and TEXT_TOKEN
KILL = 2**20  # what a stage's constant bias is pushed down by on a non-kernel token

# Qwen3.5-4B's layers: a kernel graph is built against the host's per-sublayer capacity.
T4B = Target(key_heads=16, value_heads=32, intermediate=9216)


@dataclass(frozen=True)
class Slot:
    """The residual columns a kernel owns, right after the chip region.

    ``base`` is the first input column; ``out_base`` the first result column. ``shared`` reuses
    the input columns as the result columns, which is what fits the design's 80-column budget.
    """

    base: int
    lanes: int = LANES
    temp: int = TEMP
    shared: bool = False

    @property
    def out_base(self) -> int:
        return self.base if self.shared else self.base + self.lanes

    @property
    def width(self) -> int:
        return self.lanes + (0 if self.shared else self.lanes) + self.temp

    @property
    def end(self) -> int:
        return self.base + self.width

    def columns(self) -> range:
        """The result columns."""
        return range(self.out_base, self.out_base + self.lanes)

    def temp_columns(self) -> range:
        """The scratch columns after the inputs (and results, when they are separate)."""
        first = self.out_base + self.lanes
        return range(first, first + self.temp)


def slot_after_chip(lay: Layout, n_slots: int, *, shared: bool = False, lanes: int = LANES,
                    temp: int = TEMP) -> Slot:
    """The kernel slot right after the chip region: ``off + m + 1 + n_slots``.

    ``n_slots`` is ``Alloc.n_slots`` of the circuit the kernel is merged next to. Raises
    ``ValueError`` when the slot would run past ``hidden``.
    """
    slot = Slot(lay.off + lay.m + 1 + n_slots, lanes, temp, shared)
    if slot.end > lay.hidden:
        raise ValueError(
            f"kernel slot needs {slot.width} columns after the chip region (columns "
            f"{slot.base}..{slot.end}), but hidden is {lay.hidden}: {lay.hidden - slot.base} free"
        )
    return slot


@dataclass(frozen=True)
class Kernel:
    """A kernel: its IR graph, the layout mapping it into the slot, and its input rows."""

    slot: Slot
    graph: Graph
    layout: Layout
    is_kernel: Expr
    inputs: list[Expr]
    outputs: list[Expr]
    n_tokens: int
    weight: np.ndarray  # the matvec actually emitted, after bf16 rounding
    token_base: int = KERNEL_BASE  # id of the first kernel token

    def tokens(self) -> list[int]:
        """The token id of each input row."""
        return [self.token_base + i for i in range(self.n_tokens)]

    def other_tokens(self) -> list[int]:
        """Token ids that are not kernel tokens (the gate must be dead on them)."""
        return [0, 7, 1 << 24, self.token_base + self.n_tokens]  # chip, chip, text, one past

    def unit_count(self) -> int:
        return sum(len(n.terms) for n in self.graph.nodes if isinstance(n, FFN))


def _round_bf16(w: np.ndarray) -> np.ndarray:
    """``w`` rounded to the nearest bf16 value per element (the only weights emit accepts)."""
    t = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)).to(torch.bfloat16)
    return t.to(torch.float64).numpy()


def build(W: np.ndarray, inputs: np.ndarray, slot: Slot, lay: Layout, *,
          token_base: int = KERNEL_BASE) -> Kernel:
    """The kernel's graph for the matvec ``y = W x``, and the layout that maps it into the slot.

    One FFN unit per result row: ``a`` is the row's linear form over the input lanes and the gate
    is the kernel-token feature, so the unit computes that row on a kernel token and 0 elsewhere.
    ``W`` is rounded to bf16 values (emit accepts nothing else); splitting a weight over several
    bf16 units does not help, because the bf16 rounding of the unit's output is the bigger term.
    With ``slot.shared`` the row's unit also subtracts its input column, so the result replaces the
    input instead of adding to it (this is exact: both units ride on the same gate).

    ``token_base`` is the id of the first input row's token: above Qwen's vocabulary by default,
    and inside it when a merged checkpoint carries the input rows in its padding (rv.merge).
    """
    n = slot.lanes
    if W.shape != (n, n):
        raise ValueError(f"kernel weight must be ({n}, {n}), got {W.shape}")
    if inputs.shape != (len(inputs), n):
        raise ValueError(f"kernel inputs must be (n_tokens, {n}), got {inputs.shape}")
    w = _round_bf16(W)
    g = Graph(target=T4B)
    n_tokens = len(inputs)

    def is_kernel_value(t: int) -> float:
        return 1.0 if token_base <= t < token_base + n_tokens else 0.0

    is_kernel = g.token_feature("is_kernel", is_kernel_value, full=True)

    def lane(j: int):
        def lane_value(t: int) -> float:
            i = t - token_base
            return 0.0 if not 0 <= i < n_tokens else float(inputs[i, j])

        return lane_value

    ins = [g.token_feature(f"k.in[{j}]", lane(j), full=True) for j in range(n)]
    outs: list[Expr] = []
    for i in range(n):
        a = Expr({d: float(w[i, j]) for j, e in enumerate(ins) for d in e.terms if w[i, j]})
        terms: list[tuple[int, Expr, Expr]] = [(1, a, is_kernel)]
        if slot.shared:  # the result overwrites the input column
            terms.append((-1, ins[i], is_kernel))
        outs.append(g.ffn(terms, f"k.out[{i}]"))

    kernel: dict[int, int] = {}
    for j, e in enumerate(ins):
        for d in e.terms:
            kernel[d] = j
    for i, e in enumerate(outs):
        for d in e.terms:
            kernel[d] = i if slot.shared else slot.lanes + i
    # the gate feature needs its own kernel column too: on a column of the chip region the circuit
    # would overwrite it, so it lives in the first scratch column
    temp_offset = (slot.out_base - slot.base) + slot.lanes
    for d in is_kernel.terms:
        kernel[d] = temp_offset
    for d, col in kernel.items():
        if not 0 <= slot.base + col < lay.hidden:
            raise ValueError(f"kernel dim {g.dims[d].name} lands on column {slot.base + col}, "
                             f"outside [0, {lay.hidden})")
    layout = replace(lay, kernel=kernel, kernel_base=slot.base)
    return Kernel(slot, g, layout, is_kernel, ins, outs, n_tokens, w, token_base)


def residual_rows(k: Kernel | Dag, tokens: list[int]) -> torch.Tensor:
    """The residual rows of ``tokens``: ballast plus that token's kernel input lanes."""
    return embedding_rows(k.graph, k.layout, tokens, k.layout.hidden)


def emit(k: Kernel | Dag, tc: dict, first: int, prefix: str = "") -> dict:
    """The kernel layer's Qwen3.5 weights, named ``{prefix}layers.{first}.*``.

    The layer's attention and every unneeded MLP slot are zero, so the whole sublayer is the
    kernel's units; ``emit_layers`` checks that every weight is a bf16 value, so a weight that is
    not one fails here rather than at load time.
    """
    return emit_layers(k.graph, k.layout, tc, first, prefix)


def sublayer(layer: dict, first: int, prefix: str = "") -> dict[str, torch.Tensor]:
    """One emitted layer's weights, keyed by their local name (``mlp.gate_proj.weight`` ...)."""
    p = f"{prefix}layers.{first}."
    return {name[len(p) :]: t for name, t in layer.items() if name.startswith(p)}


def run(sub: dict[str, torch.Tensor], lay: Layout, res: torch.Tensor) -> torch.Tensor:
    """Qwen3.5's MLP sublayer on residual rows ``res``, bf16 as the HF kernels compute it.

    ``sub`` is one layer out of :func:`emit` (:func:`sublayer`). The input is normally ``emit.Layout
    .scale`` times the residual: on a chip or kernel token the RMSNorm divides by the ballast
    alone, so its output is exactly ``res * c``. Returns the whole post-sublayer residual; the
    kernel's results are at ``slot.columns()``.
    """
    bf = torch.bfloat16
    x = (res.to(torch.float32) * lay.scale).to(bf)
    gate = x @ sub["mlp.gate_proj.weight"].T.to(bf)
    up = x @ sub["mlp.up_proj.weight"].T.to(bf)
    h = torch.nn.functional.silu(gate) * up
    mlp = h @ sub["mlp.down_proj.weight"].T.to(bf)
    return res + mlp.to(torch.float32)


def results(k: Kernel | Dag, res: torch.Tensor) -> np.ndarray:
    """The kernel's 64 result values of each residual row: ``(n_tokens, lanes)``."""
    return res[:, k.slot.out_base : k.slot.out_base + k.slot.lanes].detach().numpy()


# --- a DAG of stages --------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage:
    """One ``relu(W h + b)`` stage of a kernel DAG: a compiled FFN layer's math.

    The lanes are the stage's outputs: lane ``i`` is ``relu(W[i] . h + b[i])``. This is exactly the
    unit shape the emitted MLP has (``coef * a * relu(b)`` with ``a = ONE``), so a stage costs one
    Qwen layer and 64 units (plus 64 that clear the columns it overwrites, in pipeline mode).
    """

    weight: np.ndarray  # (lanes, lanes)
    bias: np.ndarray  # (lanes,)


@dataclass(frozen=True)
class Dag:
    """A kernel that walks several stages, one Qwen layer each (:func:`build_dag`).

    ``tokens_per_result`` is 1 in pipeline mode -- one kernel token walks the whole DAG in one
    forward pass, because the residual stream carries a stage's output into the next layer -- and
    ``depth`` in per-token mode, where each token computes one stage and the caller hands the
    result back as the next token's input (the MMIO round trip of design section 4).
    """

    slot: Slot
    graph: Graph
    layout: Layout
    is_kernel: Expr
    inputs: list[Expr]
    outputs: list[Expr]
    n_tokens: int
    token_base: int
    stages: list[Stage]
    tokens_per_result: int

    @property
    def depth(self) -> int:
        return len(self.stages)

    def tokens(self) -> list[int]:
        """The token id of each input row."""
        return [self.token_base + i for i in range(self.n_tokens)]

    def other_tokens(self) -> list[int]:
        """Token ids that are not kernel tokens (the gate must be dead on them)."""
        return [0, 7, 1 << 24, self.token_base + self.n_tokens]

    def unit_count(self) -> int:
        return sum(len(n.terms) for n in self.graph.nodes if isinstance(n, FFN))


def _lane_expr(weight: np.ndarray, bias: float, prev: list[Expr]) -> Expr:
    """``W[i] . h + b[i]`` as one linear form over the previous stage's lanes."""
    out = Expr({ONE: float(bias)})
    for j, e in enumerate(prev):
        if weight[j]:
            for d in e.terms:
                out = out + Expr({d: float(weight[j])})
    return out


def build_dag(stages: list[Stage], inputs: np.ndarray, slot: Slot, lay: Layout, *,
              token_base: int = KERNEL_BASE) -> Dag:
    """A kernel that walks every stage in **one forward pass**, one Qwen layer per stage.

    Stage ``L``'s units read the residual columns stage ``L - 1`` wrote (the token features for
    ``L = 0``) and write the same columns back, after subtracting what was there: with ``slot.shared``
    every stage reuses the lane columns, so a DAG of any depth still costs ``lanes + scratch``
    columns. The subtraction is the exact a-form unit (``-x * relu(ONE)``), so it works on the raw
    input too, not just on the non-negative stage outputs.

    Exactness: a lane's gate is ``K (W h + b)``, and ``silu(K n) == K relu(n)`` exactly only for
    integers ``|n| >= 1`` (or 0). Integer weights, integer biases and non-negative integer stage
    outputs stay in that regime; arbitrary reals do not, which is the boundary K0 measured.
    """
    n = slot.lanes
    if not stages:
        raise ValueError("a DAG needs at least one stage")
    for i, st in enumerate(stages):
        if st.weight.shape != (n, n):
            raise ValueError(f"stage {i}: weight must be ({n}, {n}), got {st.weight.shape}")
        if st.bias.shape != (n,):
            raise ValueError(f"stage {i}: bias must be ({n},), got {st.bias.shape}")
    if inputs.shape != (len(inputs), n):
        raise ValueError(f"kernel inputs must be (n_tokens, {n}), got {inputs.shape}")
    g = Graph(target=T4B)
    n_tokens = len(inputs)

    def is_kernel_value(t: int) -> float:
        return 1.0 if token_base <= t < token_base + n_tokens else 0.0

    is_kernel = g.token_feature("is_kernel", is_kernel_value, full=True)

    def lane(j: int):
        def lane_value(t: int) -> float:
            i = t - token_base
            return 0.0 if not 0 <= i < n_tokens else float(inputs[i, j])

        return lane_value

    ins = [g.token_feature(f"k.in[{j}]", lane(j), full=True) for j in range(n)]
    kernel: dict[int, int] = {}
    for j, e in enumerate(ins):
        for d in e.terms:
            kernel[d] = j
    prev = ins
    outs: list[Expr] = ins
    for stage_i, st in enumerate(stages):
        w, b = _round_bf16(st.weight), _round_bf16(st.bias)
        outs = []
        for i in range(n):
            # the lane's gate is ``W h + b[i]``; the constant part is not carried by ``h``, so it
            # needs the kernel switch too -- otherwise it would open the unit on every token
            gate = _lane_expr(w[i], float(b[i]), prev) - KILL + KILL * is_kernel
            terms: list[tuple[int, Expr, Expr]] = [
                (1, Expr({ONE: 1}), gate),
                (-1, prev[i], is_kernel),  # the lane column is reused: subtract its old value
            ]
            outs.append(g.ffn(terms, f"k.s{stage_i}[{i}]"))
        for i, e in enumerate(outs):
            for d in e.terms:
                kernel[d] = i
        prev = outs
    temp_offset = (slot.out_base - slot.base) + slot.lanes
    for d in is_kernel.terms:
        kernel[d] = temp_offset
    for d, col in kernel.items():
        if not 0 <= slot.base + col < lay.hidden:
            raise ValueError(f"kernel dim {g.dims[d].name} lands on column {slot.base + col}, "
                             f"outside [0, {lay.hidden})")
    if g.n_layers() != len(stages):
        raise ValueError(f"a {len(stages)}-stage DAG landed in {g.n_layers()} layers")
    layout = replace(lay, kernel=kernel, kernel_base=slot.base)
    return Dag(slot, g, layout, is_kernel, ins, outs, n_tokens, token_base,
               [Stage(_round_bf16(st.weight), _round_bf16(st.bias)) for st in stages], 1)


def build_dag_per_token(stages: list[Stage], inputs: np.ndarray, slot: Slot, lay: Layout, *,
                        token_base: int = KERNEL_BASE) -> Dag:
    """A kernel that computes **one stage per token**: token ``p`` runs stage ``p``.

    Stage ``L``'s units are gated by a per-stage phase feature (``t == token_base + L``) instead of
    the kernel-token feature, and every stage reads the lane columns of its own token (the caller
    hands the previous stage's result back as the next token's input row). Needs ``slot.shared``
    off: the input and the result are different tokens, so they cannot share a column.

    This is the design's "depth N takes N tokens" mode; :func:`build_dag` is the one that does not.
    """
    n = slot.lanes
    depth = len(stages)
    if depth == 0:
        raise ValueError("a DAG needs at least one stage")
    if slot.shared:
        raise ValueError("per-token stages need separate input and result columns")
    if inputs.shape != (depth, n):
        raise ValueError(f"per-token inputs must be ({depth}, {n}), got {inputs.shape}")
    g = Graph(target=T4B)

    def phase_value(t: int, stage: int) -> float:
        return 1.0 if t == token_base + stage else 0.0

    phases = [g.token_feature(f"phase[{L}]", lambda t, L=L: phase_value(t, L), full=True)
              for L in range(depth)]

    def lane(j: int):
        def lane_value(t: int) -> float:
            i = t - token_base
            return 0.0 if not 0 <= i < depth else float(inputs[i, j])

        return lane_value

    ins = [g.token_feature(f"k.in[{j}]", lane(j), full=True) for j in range(n)]
    kernel: dict[int, int] = {}
    for j, e in enumerate(ins):
        for d in e.terms:
            kernel[d] = j
    outputs: list[Expr] = []
    for stage_i, st in enumerate(stages):
        w, b = _round_bf16(st.weight), _round_bf16(st.bias)
        outs = []
        for i in range(n):
            # the constant bias needs the phase switch too, exactly like the pipeline gate does
            gate = _lane_expr(w[i], float(b[i]), ins) - KILL + KILL * phases[stage_i]
            terms: list[tuple[int, Expr, Expr]] = [(1, Expr({ONE: 1}), gate)]
            # push the stage onto its own Qwen layer: every stage reads the same token features,
            # so nothing but min_level separates them
            outs.append(g.ffn(terms, f"k.t{stage_i}[{i}]", min_level=2 * stage_i))
        for i, e in enumerate(outs):
            for d in e.terms:
                kernel[d] = n + i
        if stage_i == depth - 1:
            outputs = outs
    # the phase features go into the scratch columns after the result columns: anything inside the
    # result span would be read back as part of the result
    for L, e in enumerate(phases):
        for d in e.terms:
            kernel[d] = 2 * n + L
    for d, col in kernel.items():
        if not 0 <= slot.base + col < lay.hidden:
            raise ValueError(f"kernel dim {g.dims[d].name} lands on column {slot.base + col}, "
                             f"outside [0, {lay.hidden})")
    if g.n_layers() != depth:
        raise ValueError(f"a {depth}-stage DAG landed in {g.n_layers()} layers")
    layout = replace(lay, kernel=kernel, kernel_base=slot.base)
    return Dag(slot, g, layout, phases[0], ins, outputs, depth, token_base,
               [Stage(_round_bf16(st.weight), _round_bf16(st.bias)) for st in stages], depth)


def dag_reference(stages: list[Stage], x: np.ndarray) -> np.ndarray:
    """The stages' exact math on ``x`` (float64): ``h <- relu(W h + b)``."""
    h = np.asarray(x, dtype=np.float64)
    for st in stages:
        h = np.maximum(h @ st.weight.T + st.bias, 0.0)
    return h


def dag_pipeline(dag: Dag, layers: dict, first: int, prefix: str, tokens: list[int],
                 res: torch.Tensor) -> torch.Tensor:
    """Run every stage's layer on one residual, in order (one kernel token, one forward pass)."""
    for i in range(dag.depth):
        sub = sublayer(layers, first + i, prefix)
        res = run(sub, dag.layout, res)
    return res


def token_row(k: Kernel | Dag, token: int, lanes: np.ndarray | None = None) -> torch.Tensor:
    """The residual row of ``token``: ballast, its token features, and ``lanes`` in the input columns.

    ``lanes`` defaults to what ``build`` was given. The per-token DAG mode overrides it, because
    each step's input is the previous step's result rather than a row fixed at build time.
    """
    row = torch.zeros(k.layout.hidden, dtype=torch.float32)
    stage = k.layout
    row[stage.off:stage.off + stage.m] = float(stage.ballast)
    for d in k.graph.token_fns:
        row[stage.dim(d)] = float(k.graph.feature_value(d, token))
    if lanes is not None:
        row[k.slot.base:k.slot.base + k.slot.lanes] = torch.as_tensor(
            np.asarray(lanes, dtype=np.float32)
        )
    return row
