"""Overlay: circuit dims that live only on chip tokens share Qwen's dims in a merged checkpoint.

The chip layers run first, and while they do every text token carries its Qwen embedding in dims
[0, D0); on chip tokens those dims are 0. So a circuit dim that is 0 at every text position can
use a Qwen dim as its residual slot (emit.Layout.over): a chip token sees the circuit's value
there, a text token sees q, a Qwen embedding value. That stays exact when

1. nothing writes into an overlaid slot at a text position, so Qwen's input is bit-identical:
   every unit and head column that writes an overlaid dim is switched off on text tokens;
2. whatever reads an overlaid slot at a text position is switched off there, or cannot matter:
   - an FFN unit: its gate gets -KILL * is_text, so silu is exactly 0 whatever q is;
   - a head that writes no key on text tokens: its key channels that read the overlay get the
     same switch, so the key stays 0 (no state write), and every output gate z of a column
     that is 0 on text does too (a query of q then reads nothing out);
3. nothing that matters on text tokens reads an overlaid dim at all: units and head columns that
   are nonzero there, keys written there (the bridge's capture of a tool call), decays, and
   channels with conv lag taps (which carry a text token's projection into the next token).

The handovers and the exit clear of overlaid slots are gated by mu (chip tokens) instead of ONE
(emit.emit_layers), so a text token keeps q.

What is 0 or switched off on text tokens is measured on the traces the caller passes in
(``rv.corpus`` for the RV32 machine: text, tool calls, runs and their echoes). The measurement is
not a proof: a trace the corpus does not cover can show an overlaid dim carrying something that
matters on text, and then the overlay is not exact for that input. ``check_noisy`` is the check
on a set of traces, not a guarantee for every input.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

import numpy as np

from chipformer.dsl.graph import ONE, Expr
from chipformer.q35 import bridge as B
from chipformer.q35 import ir
from chipformer.q35.ir import FFN, K, Chan, Graph, Head, conv, ev

KILL = 2**20  # -KILL * is_text switches a gate, key or output gate off on text tokens


@dataclass
class Plan:
    dims: set[int] = field(default_factory=set)  # dims that may use Qwen's dims
    terms: set[tuple[int, int]] = field(default_factory=set)  # (node, term): FFN units
    keys: set[tuple[int, int]] = field(default_factory=set)  # (node, key channel)
    gates: set[tuple[int, int, int]] = field(default_factory=set)  # (node, vhead, column): z
    q_bound: float = 0.0  # the largest |q| every switched-off value still absorbs

    def summary(self) -> str:
        return (
            f"overlay: {len(self.dims)} dims, {len(self.terms)} units, {len(self.keys)} key "
            f"channels and {len(self.gates)} output gates switched off on text, |q| <= "
            f"{self.q_bound:.4g}"
        )


# --- text positions ---------------------------------------------------------------------------


def feature(g: Graph, name: str) -> Expr:
    """A token feature the circuit builders set on the graph (``is_text``, ``mu``)."""
    e = getattr(g, name, None)
    assert isinstance(e, Expr), f"the overlay needs a graph with a {name!r} feature"
    return e


def text_dim(g: Graph) -> int:
    """The IR dim of the graph's ``is_text`` token feature."""
    return next(iter(feature(g, "is_text").terms))


def text_mask(g: Graph, tokens: list[int]) -> np.ndarray:
    """1 where the token is one of Qwen's text tokens (``is_text`` on)."""
    d = text_dim(g)
    return np.array([bool(g.feature_value(d, t)) for t in tokens])


# --- measuring and planning -------------------------------------------------------------------


def _reads(e: Expr) -> set[int]:
    return {d for d in e.terms if d != ONE}


def _vh_reads(vh) -> set[int]:
    """What a value head's write gate and decay read: they act at text positions too."""
    return _reads(vh.a) | (set() if vh.beta is None else _reads(vh.beta))


@dataclass
class Activity:
    """What is nonzero at text positions, over a corpus."""

    zero: np.ndarray  # dim -> 0 at every text position
    terms: dict = field(default_factory=dict)  # (node, term) -> the unit fires somewhere
    gate_max: dict = field(default_factory=dict)  # (node, term) -> max b
    keys: dict = field(default_factory=dict)  # node -> a key written at some text position
    key_max: dict = field(default_factory=dict)  # (node, channel) -> max conv output
    cols: dict = field(default_factory=dict)  # (node, column) -> output nonzero somewhere
    z_max: dict = field(default_factory=dict)  # (node, column) -> max z


def activity(
    g: Graph, traces: list[list[int]], prepare: Callable[[int], None] | None = None
) -> Activity:
    """What is nonzero at text positions over ``traces``; ``prepare(i)`` runs before trace i (a
    machine's state goes into the graph's head states there: rv.corpus + core.snapshot)."""
    act = Activity(np.ones(len(g.dims), bool))

    def upd(d, k, v, f=max):
        d[k] = f(d[k], v) if k in d else v

    for i, toks in enumerate(traces):
        if prepare is not None:
            prepare(i)
        txt = text_mask(g, toks)
        m = ir.evaluate(g, toks, check=False)
        act.zero &= ~(m[txt] != 0).any(0)
        for i, n in enumerate(g.nodes):
            if isinstance(n, FFN):
                for j, (coef, a, b) in enumerate(n.terms):
                    bv = ev(b, m)[txt]
                    v = float(coef) * ev(a, m)[txt] * np.maximum(bv, 0)
                    upd(act.terms, (i, j), bool((v != 0).any()), lambda x, y: x or y)
                    upd(act.gate_max, (i, j), float(bv.max(initial=-np.inf)))
                continue
            wrote = False
            for ch_i, ch in n.k.items():
                x = conv(ch, m, False)[txt]
                wrote |= bool((x > 0).any())
                upd(act.key_max, (i, ch_i), float(ev(ch.expr, m)[txt].max(initial=-np.inf)))
            upd(act.keys, i, wrote, lambda x, y: x or y)
            for x_, vh in enumerate(n.vheads()):
                for col, d, _ in vh.out:
                    # (a dim several heads add into is judged by its total)
                    upd(act.cols, (i, x_, col), bool((m[txt, d] != 0).any()), lambda x, y: x or y)
                    if col in vh.z:
                        upd(
                            act.z_max,
                            (i, x_, col),
                            float(ev(vh.z[col], m)[txt].max(initial=-np.inf)),
                        )
    return act


def needed(g: Graph, act: Activity, keep: set[int]) -> set[int]:
    """Dims whose value at text positions matters: the outputs the lm_head reads, what heads
    write into their state there (decays always; keys and values of heads that write keys on
    text), what conv lag taps carry to the next token, and everything those depend on."""
    need = set(keep) | _reads(feature(g, "is_text")) | _reads(feature(g, "mu"))
    for i, n in enumerate(g.nodes):
        if isinstance(n, Head):
            vhs = n.vheads()
            vchans = [ch for vh in vhs for ch in vh.v.values()]
            need |= set().union(*(_vh_reads(vh) for vh in vhs))
            chans = list(n.q.values()) + list(n.k.values()) + vchans
            need |= {d for ch in chans if any(ch.taps[1:]) for d in _reads(ch.expr)}
            if act.keys.get(i):
                need |= {d for ch in list(n.k.values()) + vchans for d in _reads(ch.expr)}
    for n in sorted(g.nodes, key=lambda n: -n.level):  # reads come from lower levels
        if isinstance(n, FFN):
            if n.out in need:
                need |= {d for _, a, b in n.terms for d in _reads(a) | _reads(b)}
            continue
        for vh in n.vheads():
            for col, d, _ in vh.out:
                if d in need:
                    need |= {x for ch in n.q.values() for x in _reads(ch.expr)}
                    need |= _reads(vh.z.get(col, Expr({})))
    return need


def plan(g: Graph, act: Activity, keep: set[int]) -> Plan:
    """The overlay dims and the switch-offs they need.

    A unit or head column may be switched off on text tokens when its output does not matter
    there (``needed``) or is 0 there anyway (``activity``). A dim is overlaid when it is not
    needed on text or is 0 there, every unit and column writing it or reading it may be switched
    off, and no decay, conv lag tap or key of a head writing keys on text reads it.
    """
    need = needed(g, act, keep)
    fixed = set(keep) | _reads(feature(g, "is_text")) | _reads(feature(g, "mu"))
    ok = {d for d in range(1, len(g.dims)) if d not in fixed and (d not in need or act.zero[d])}
    killable: dict[tuple, bool] = {}  # (node, term) / (node, ("col", c))
    for i, n in enumerate(g.nodes):
        if isinstance(n, FFN):
            for j in range(len(n.terms)):
                killable[i, j] = n.out not in need or not act.terms.get((i, j), False)
        else:
            for x_, vh in enumerate(n.vheads()):
                for col, d, _ in vh.out:
                    killable[i, ("col", x_, col)] = d not in need or not act.cols.get(
                        (i, x_, col), False
                    )
    for i, n in enumerate(g.nodes):
        if isinstance(n, FFN):
            for j, (_, a, b) in enumerate(n.terms):
                if not killable[i, j]:
                    ok -= _reads(a) | _reads(b) | {n.out}
            continue
        vhs = n.vheads()
        vchans = [ch for vh in vhs for ch in vh.v.values()]
        for vh in vhs:
            ok -= _vh_reads(vh)
        for ch in list(n.q.values()) + list(n.k.values()) + vchans:
            if any(ch.taps[1:]):
                ok -= _reads(ch.expr)
        if act.keys.get(i):
            ok -= {d for ch in list(n.k.values()) + vchans for d in _reads(ch.expr)}
        cols = [(x_, col, d) for x_, vh in enumerate(vhs) for col, d, _ in vh.out]
        if not all(killable[i, ("col", x_, col)] for x_, col, _ in cols):
            ok -= {d for ch in n.q.values() for d in _reads(ch.expr)}
        for x_, col, d in cols:
            if not killable[i, ("col", x_, col)]:
                ok -= _reads(vhs[x_].z.get(col, Expr({}))) | {d}
    p = Plan(ok)
    bounds = [np.inf]

    def bound(top, s, margin):
        if s:
            bounds.append((KILL - max(top, 0) - margin) / s)

    def noise(e):
        return sum(abs(float(c)) for d, c in e.terms.items() if d in ok)

    for i, n in enumerate(g.nodes):
        if isinstance(n, FFN):
            for j, (_, a, b) in enumerate(n.terms):
                if n.out in ok or (_reads(a) | _reads(b)) & ok:
                    p.terms.add((i, j))
                    bound(act.gate_max.get((i, j), 0), noise(b), 104 / K)
            continue
        if not act.keys.get(i):
            for ch_i, ch in n.k.items():
                if _reads(ch.expr) & ok:
                    assert not any(ch.taps[1:]), f"{n.name}: key channel with lag taps"
                    p.keys.add((i, ch_i))
                    bound(act.key_max.get((i, ch_i), 0), noise(ch.expr), 104 / ch.taps[0])
        noisy = any(_reads(ch.expr) & ok for ch in n.q.values())
        for x_, vh in enumerate(n.vheads()):
            for col, d, _ in vh.out:
                if col in vh.z and (noisy or d in ok or _reads(vh.z[col]) & ok):
                    p.gates.add((i, x_, col))
                    bound(act.z_max.get((i, x_, col), 0), noise(vh.z[col]), 104)
    p.q_bound = float(min(bounds))
    return p


def _vz(n, x_: int) -> dict:
    """The output gates of value head x_ of head n (the first value head's are n.z)."""
    return n.z if x_ == 0 else n.vx[x_ - 1].z


# --- the CHIP-8 corpus (the RV32 machine's is rv/corpus.py) ------------------------------------

FORMS = {
    "add": [[B.PLUS]],
    "sub": [[B.MINUS], [B.MINUS2]],
    "mul": [[B.TIMES], [B.BSL, B.TIMESW, B.SPACE]],
    "div": [[B.DIV]],
}
LEADS = [
    [B.SPACE],
    [B.DOLLAR],
    [B.SDOLLAR, B.SPACE],
    [B.DDOLLAR],
    [B.SDDOLLAR],
    [B.NL],
    [B.WS],
    [B.LPAREN],
    [B.OTHER],
    [B.EQ, B.SPACE],
]


def _digits(rng: random.Random, n: int, commas: bool) -> list[int]:
    s = str(rng.randrange(10 ** (n - 1), 10**n))
    out = []
    for i, ch in enumerate(s):
        if commas and i and (len(s) - i) % 3 == 0:
            out.append(B.COMMA)
        out.append(B.HEX[int(ch)])
    return out


def _expr(rng: random.Random) -> list[int]:
    op = rng.choice(list(FORMS))
    commas = rng.random() < 0.3
    a = _digits(rng, rng.randint(1, 7), commas)
    b = _digits(rng, rng.randint(1, 5), commas)
    gap = [B.SPACE] if rng.random() < 0.5 else []
    return rng.choice(LEADS) + a + rng.choice(FORMS[op]) + gap + b + [B.EQ]


def _number(rng: random.Random, n: int) -> str:
    return str(rng.randrange(10 ** (n - 1), 10**n))


def _tool(rng: random.Random, bm) -> None:
    """A tool call: one of the four library routines the bridge answers."""
    op = rng.choice(list(FORMS))
    bm.tool_call(op, _number(rng, rng.randint(1, 6)), _number(rng, rng.randint(1, 4)))


def corpus(seed: int = 7, docs: int = 16) -> list[list[int]]:
    """Token sequences of mixed CHIP-8 sessions: Qwen text, inline expressions, tool calls and
    prints (the CHIP-8 circuit's bridge; the RV32 machine's corpus is rv/corpus.py)."""
    from chipformer.q35 import bridge as B
    from chipformer.q35.lib import bcd

    rng = random.Random(seed)
    text = [t for t in range(B.TEXT_BASE, B.TEXT_BASE + B.N_TEXT) if t != B.TC_CLOSE]
    common = [B.HEX[d] for d in range(10)] * 3 + [B.SPACE] * 6 + [B.OTHER] * 6 + text
    out = []
    for _ in range(docs):
        bm = B.BridgeMachine(bcd.ROM, bcd.QUIRKS)
        bm.text(rng.randint(1, 4))
        for _ in range(rng.randint(3, 8)):
            r = rng.random()
            if r < 0.5:
                bm.qwen(_expr(rng) + [rng.choice(common) for _ in range(rng.randint(1, 6))])
            elif r < 0.75:
                bm.qwen([rng.choice(common) for _ in range(rng.randint(5, 25))])
            else:
                _tool(rng, bm)
                while not bm.halted:
                    bm.step()
                bm.text(rng.randint(1, 4))
        out.append(bm.tokens)
    return out


def apply(g: Graph, p: Plan) -> None:
    """Write the switch-offs into the graph (in place); overlaid token features read 0 on text
    (the text rows carry Qwen's embedding there)."""
    off = KILL * feature(g, "is_text")
    for i, j in p.terms:
        n = g.nodes[i]
        assert isinstance(n, FFN)
        coef, a, b = n.terms[j]
        n.terms[j] = (coef, a, b - off)
    for i, ch_i in p.keys:
        h = g.nodes[i]
        assert isinstance(h, Head)
        ch = h.k[ch_i]
        h.k[ch_i] = Chan(ch.expr - off, ch.taps)
    for i, x_, col in p.gates:
        z = _vz(g.nodes[i], x_)
        z[col] = z[col] - off
    td = text_dim(g)
    for d in p.dims & set(g.token_fns):
        f = cast(Callable[[int], float], g.token_fns[d])
        g.token_fns[d] = lambda t, f=f, td=td: 0 if g.feature_value(td, t) else f(t)


def build(
    g: Graph, keep: set[int], traces: list[list[int]], prepare: Callable[[int], None] | None = None
) -> Plan:
    """Measure, plan and apply the overlay of ``g`` (in place)."""
    p = plan(g, activity(g, traces, prepare), keep)
    apply(g, p)
    return p


# --- checking -----------------------------------------------------------------------------------


def evaluate_noisy(
    g: Graph, p: Plan, tokens: list[int], noise: dict[int, np.ndarray]
) -> np.ndarray:
    """``ir.evaluate`` with the overlay dims holding ``noise`` (Qwen's embedding values) at text
    positions. Raises unless every write into them there is exactly 0 and every switched-off
    gate, key and output gate is dead (silu exactly 0) despite the noise."""
    n = len(tokens)
    txt = text_mask(g, tokens)
    toks = np.asarray(tokens, dtype=np.int64)
    m = np.zeros((n, len(g.dims)))
    chip = toks >= 0
    m[:, ONE] = chip
    uniq, inv = np.unique(toks, return_inverse=True)
    for d, fn in g.token_fns.items():
        f = cast(Callable[[int], float], fn)
        m[:, d] = np.array([float(f(int(u))) for u in uniq])[inv]
    over = sorted(d for d in p.dims if d in noise)
    for d in over:
        ir.require(m[txt, d] == 0, f"overlay token feature {g.dims[d].name} on text", m[txt, d])
        m[txt, d] = noise[d][txt]
    for i, node in enumerate(g.nodes):
        if isinstance(node, FFN):
            for j, (_, _, b) in enumerate(node.terms):
                if (i, j) in p.terms:
                    gate = K * ev(b, m)[txt]
                    ir.require(ir.dead(gate), f"{node.name}: unit not dead", gate)
            vals = np.zeros(n)
            for coef, a, b in node.terms:
                vals += float(coef) * ev(a, m) * np.maximum(ev(b, m), 0.0)
            m[:, node.out] = (m[:, node.out] + vals) if node.into else vals
            if node.out in noise:
                ir.require(
                    vals[txt] == 0, f"{node.name}: write into an overlay dim on text", vals[txt]
                )
                m[txt, node.out] = noise[node.out][txt]
            continue
        for ch_i, ch in node.k.items():
            if (i, ch_i) in p.keys:
                x = ch.taps[0] * ev(ch.expr, m)[txt]
                ir.require(ir.dead(x), f"{node.name}: key channel not dead", x)
        vhs = node.vheads()
        for x_, vh in enumerate(vhs):
            for col, d, _ in vh.out:
                if (i, x_, col) in p.gates:
                    zz = ev(vh.z[col], m)[txt]
                    ir.require(ir.dead(zz), f"{node.name}: output gate not dead", zz)
        outs = {d for vh in vhs for _, d, _ in vh.out}
        # an overlaid dim several heads add into holds the noise at text positions: take it out
        # before the head adds (its share there must be 0) and put it back after
        held = {d: m[txt, d].copy() for d in outs if d in noise}
        for d in held:
            if any(vh.into for vh in vhs):
                m[txt, d] = 0
        ir.eval_head(node, m, chip, False)
        for d in outs:
            if d in noise:
                ir.require(
                    m[txt, d] == 0, f"{node.name}: write into an overlay dim on text", m[txt, d]
                )
                m[txt, d] = held[d]
    return m


def check_noisy(
    g: Graph, p: Plan, tokens: list[int], rng: np.random.Generator, scale: float = 1.0
) -> None:
    """The overlay leaves the circuit exact: with random Qwen values in the overlaid dims at text
    positions, every dim everywhere else equals the clean evaluation."""
    clean = ir.evaluate(g, tokens)
    txt = text_mask(g, tokens)
    noise = {d: ir.bf16(rng.normal(0, scale, len(tokens))) for d in p.dims}
    m = evaluate_noisy(g, p, tokens, noise)
    mask = np.ones(m.shape, bool)
    mask[np.ix_(txt, sorted(p.dims))] = False
    bad = np.nonzero(mask & (m != clean))
    if len(bad[0]):
        t, d = int(bad[0][0]), int(bad[1][0])
        raise ir.ExactnessError(
            f"overlay changes {g.dims[d].name} at position {t}: {m[t, d]!r} vs {clean[t, d]!r}"
        )
