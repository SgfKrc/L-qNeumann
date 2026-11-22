"""Executor v3: Executor2's kernels without the apply launches, and the token decode on the GPU.

Each launch is the only sync. The residual at sublayer i is r_i = X[i % 2] + Y[(i - 1) % 3]; the
sublayer adds its outputs into Y[i % 3] (atomics, as before), and its programs also do the
bookkeeping that the apply kernel did, each for its own slice of the dims:

  X[(i + 1) % 2] = r_i              (the next sublayer's base: r_(i+1) = that + Y[i % 3])
  Y[(i + 1) % 3] = 0                (last read by sublayer i - 1; written next by sublayer i + 1)
  the previous head sublayer's lag history moves on, its input r_(i-1) = X[i % 2] becoming lag 1
                                    (only sublayer i - 1 reads that history, and not now)

The one thing this executor does differently from Executor2 is the layout of those five buffers.
`r_i` needs element d of one X slot and element d of one Y slot, at the same random index, twice
per lane -- and a lane that gathers is the whole cost of a step (``gather-latency-report.md``:
34.2 M random 4-byte loads, 85% of 323 us). So the slots are stored interleaved, in the six pairs
the rotation actually visits: `R[k]` holds, element by element, `X[k % 2]` next to
`Y[(k - 1) % 3]`, and lane d of sublayer i = k % 6 reads the pair with one 8-byte load. That is
three loads per lane (pair, index, weight) instead of four, with the same arithmetic:
`_pair` adds the two halves in the order Executor2's two loads were added.

The six pairs are needed because the two rotations have different periods: X cycles through two
slots, Y through three, so a given (X slot, Y slot) pairing comes back every lcm(2, 3) = 6
sublayers. ``_decode_embed`` writes the next token's embedding into the X half of ``R[0]`` and
clears the Y halves, so a free run is still one graph replay.

A head sublayer is the other half of the same problem: it reads its 128 rows to find the k and q
argmaxes, and with grid = V (11 to 32 programs) it fills a third of the card. ``_head_scan``
splits those rows over SPLIT programs per head and leaves the four row vectors plus a local
argmax in scratch; ``_head_pick`` merges the local argmaxes (strictly larger wins, so a tie keeps
the lowest row as the single ``argmax`` did) and does the S-table work once, in order. The
``tl.debug_barrier`` that used to order the S-table write against the QHOT reads is now a kernel
boundary. Both bodies are chosen per sublayer by ``tune``.

After the last sublayer one program decodes the output code into the next token (decode_read),
then every program writes that token's embedding and zeroes the Y buffers (decode_embed): 24
launches a step (35 with the split head) instead of Executor2's 2 per sublayer plus FreeRun's
torch ops. Same arithmetic as Executor2 (fp32, every value exact), so the same tokens.

    fr = FreeRun3(c); core.snapshot(c, m); fr.load(0, S.tok("GO")); toks = fr.run(n)
"""

import time

import numpy as np
import torch
import triton
import triton.language as tl

OWN = tl.constexpr(1024)  # dims per bookkeeping chunk
PAIRS = 6   # lcm(2, 3): the distinct (X slot, Y slot) pairings one sublayer rotation visits
SPLIT = 8   # rows per program in the split head scan: NC = 128 / SPLIT, V * SPLIT programs


@triton.jit
def _pair(rb, i, m):
    """x_i + y_i out of one interleaved 8-byte read (see the module doc).

    ``rb`` is the same buffer viewed as int64, so ``rb[i]`` is the pair (X[i % 2][i],
    Y[(i - 1) % 3][i]) and one gather replaces two. The halves are added in the order of
    Executor2's two separate loads, so the sum is the same value bit for bit.
    """
    v = tl.load(rb + i, mask=m, other=0).to(tl.uint64)
    return ((v & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
            + (v >> 32).to(tl.uint32).to(tl.float32, bitcast=True))


@triton.jit
def _book(cur, nxt, yz, hs, D, pid, nprog, SHIFT: tl.constexpr):
    """This program's slice of the bookkeeping (see the module doc).

    ``cur`` is this sublayer's pair: its X half is X[i % 2] = r_(i - 1), its Y half is the
    output buffer the sublayer read. ``nxt`` is the next pair, whose X half takes r_i and whose Y
    half the sublayer's programs just accumulated into; ``yz`` is the pair after it, whose Y half
    is the buffer to clear.
    """
    ch = tl.cdiv(D, nprog)
    lo = pid * ch
    for off in range(0, ch, OWN):
        i = lo + off + tl.arange(0, OWN)
        m = (i < D) & (i < lo + ch)
        base = tl.load(cur + 2 * i, mask=m, other=0.0)
        tl.store(nxt + 2 * i, base + tl.load(cur + 2 * i + 1, mask=m, other=0.0), mask=m)
        tl.store(yz + 2 * i + 1, tl.zeros([OWN], dtype=tl.float32), mask=m)
        if SHIFT:
            h0 = tl.load(hs + i, mask=m, other=0.0)
            h1 = tl.load(hs + D + i, mask=m, other=0.0)
            tl.store(hs + 2 * D + i, h1, mask=m)
            tl.store(hs + D + i, h0, mask=m)
            tl.store(hs + i, base, mask=m)


@triton.jit
def _ffn3(rb, cur, nxt, yz, hs, D, U, rpa, ixa, wa, rpb, ixb, wb, coef, outd, act,
          BLOCK: tl.constexpr, KA: tl.constexpr, KB: tl.constexpr, SHIFT: tl.constexpr):
    pid = tl.program_id(0)
    bat = tl.program_id(1)
    if tl.load(act + bat) == 0:  # a parked row: no state of its own may move
        return
    nprog = tl.num_programs(0)
    rw = rb + bat * D
    u = pid * BLOCK + tl.arange(0, BLOCK)
    m = u < U
    a0 = tl.load(rpa + u, mask=m, other=0)
    na = tl.load(rpa + u + 1, mask=m, other=0) - a0
    ka = tl.arange(0, KA)
    ma = m[:, None] & (ka[None, :] < na[:, None])
    ia = tl.load(ixa + a0[:, None] + ka[None, :], mask=ma, other=0)
    av = tl.sum(tl.load(wa + a0[:, None] + ka[None, :], mask=ma, other=0.0) * _pair(rw, ia, ma),
                axis=1)
    b0 = tl.load(rpb + u, mask=m, other=0)
    nb = tl.load(rpb + u + 1, mask=m, other=0) - b0
    kb = tl.arange(0, KB)
    mb = m[:, None] & (kb[None, :] < nb[:, None])
    ib = tl.load(ixb + b0[:, None] + kb[None, :], mask=mb, other=0)
    bv = tl.sum(tl.load(wb + b0[:, None] + kb[None, :], mask=mb, other=0.0) * _pair(rw, ib, mb),
                axis=1)
    val = tl.load(coef + u, mask=m, other=0.0) * av * tl.maximum(bv.to(tl.bfloat16).to(tl.float32), 0.0)
    od = tl.load(outd + u, mask=m, other=0)
    tl.atomic_add(nxt + bat * 2 * D + 2 * od + 1, val, mask=m)
    _book(cur + bat * 2 * D, nxt + bat * 2 * D, yz + bat * 2 * D, hs + bat * 3 * D, D, pid,
          nprog, SHIFT)


@triton.jit
def _ffn4(rb, cur, nxt, yz, hs, D, U, rpa, ixa, wa, rpb, ixb, wb, coef, outd, bka, bkb, act,
          BLOCK: tl.constexpr, KC: tl.constexpr, SHIFT: tl.constexpr):
    """_ffn3 over units sorted by row length: each block loops over KC-wide chunks of its rows,
    only as far as its own longest row (bka / bkb), instead of the sublayer's longest padded."""
    pid = tl.program_id(0)
    bat = tl.program_id(1)
    if tl.load(act + bat) == 0:
        return
    nprog = tl.num_programs(0)
    rw = rb + bat * D
    u = pid * BLOCK + tl.arange(0, BLOCK)
    m = u < U
    kc = tl.arange(0, KC)
    a0 = tl.load(rpa + u, mask=m, other=0)
    na = tl.load(rpa + u + 1, mask=m, other=0) - a0
    av = tl.zeros([BLOCK], dtype=tl.float32)
    for k0 in range(0, tl.load(bka + pid), KC):
        mk = m[:, None] & ((k0 + kc)[None, :] < na[:, None])
        ia = tl.load(ixa + a0[:, None] + k0 + kc[None, :], mask=mk, other=0)
        av += tl.sum(tl.load(wa + a0[:, None] + k0 + kc[None, :], mask=mk, other=0.0)
                     * _pair(rw, ia, mk), axis=1)
    b0 = tl.load(rpb + u, mask=m, other=0)
    nb = tl.load(rpb + u + 1, mask=m, other=0) - b0
    bv = tl.zeros([BLOCK], dtype=tl.float32)
    for k0 in range(0, tl.load(bkb + pid), KC):
        mk = m[:, None] & ((k0 + kc)[None, :] < nb[:, None])
        ib = tl.load(ixb + b0[:, None] + k0 + kc[None, :], mask=mk, other=0)
        bv += tl.sum(tl.load(wb + b0[:, None] + k0 + kc[None, :], mask=mk, other=0.0)
                     * _pair(rw, ib, mk), axis=1)
    val = tl.load(coef + u, mask=m, other=0.0) * av * tl.maximum(bv.to(tl.bfloat16).to(tl.float32), 0.0)
    od = tl.load(outd + u, mask=m, other=0)
    tl.atomic_add(nxt + bat * 2 * D + 2 * od + 1, val, mask=m)
    _book(cur + bat * 2 * D, nxt + bat * 2 * D, yz + bat * 2 * D, hs + bat * 3 * D, D, pid,
          nprog, SHIFT)


@triton.jit
def _lag(rb, hb, i, m, D):
    """The lagged residual: index lag * D + d (lag 0: r_i, lags 1..3: this sublayer's history)."""
    lo = i < D
    return tl.where(lo, _pair(rb, i, m & lo), tl.load(hb + i - D, mask=m & (i >= D), other=0.0))


@triton.jit
def _rows3(rb, hb, D, rp, ix, w, base, ch, KP: tl.constexpr):
    r0 = tl.load(rp + base + ch)
    n = tl.load(rp + base + ch + 1) - r0
    kk = tl.arange(0, KP)
    mk = kk[None, :] < n[:, None]
    i = tl.load(ix + r0[:, None] + kk[None, :], mask=mk, other=0)
    acc = tl.sum(tl.load(w + r0[:, None] + kk[None, :], mask=mk, other=0.0)
                 * _lag(rb, hb, i, mk, D), axis=1)
    return tl.maximum(acc, 0.0)


@triton.jit
def _scalar3(rb, hb, D, rp, ix, w, vh):
    lane = tl.arange(0, 32)
    s0 = tl.load(rp + vh)
    n = tl.load(rp + vh + 1) - s0
    mk = lane < n
    i = tl.load(ix + s0 + lane, mask=mk, other=0)
    return tl.sum(tl.load(w + s0 + lane, mask=mk, other=0.0) * _lag(rb, hb, i, mk, D), axis=0)


@triton.jit
def _head3(rb, cur, nxt, yz, hs, h_ptr, S_ptr, D, V,
           qrp, qix, qw, krp, kix, kw, vrp, vix, vw, zrp, zix, zw,
           arp, aix, aw, brp, bix, bw, bconst, odim, ow, khead, nrows, act,
           KQ: tl.constexpr, KK: tl.constexpr, KV: tl.constexpr, KZ: tl.constexpr,
           QHOT: tl.constexpr, SHIFT: tl.constexpr):
    vh = tl.program_id(0)
    bat = tl.program_id(1)
    if tl.load(act + bat) == 0:
        return
    nprog = tl.num_programs(0)
    ch = tl.arange(0, 128)
    rw = rb + bat * D
    hb = h_ptr + bat * 3 * D
    h = tl.load(khead + vh)
    k = _rows3(rw, hb, D, krp, kix, kw, h * 128, ch, KK)
    v = _rows3(rw, hb, D, vrp, vix, vw, vh * 128, ch, KV)
    z = _rows3(rw, hb, D, zrp, zix, zw, vh * 128, ch, KZ)
    q = _rows3(rw, hb, D, qrp, qix, qw, h * 128, ch, KQ)
    bv = _scalar3(rw, hb, D, brp, bix, bw, vh) + tl.load(bconst + vh)
    Sb = S_ptr + (bat * V + vh) * 16384
    nr = tl.load(nrows + vh)
    if nr > 0:
        av = _scalar3(rw, hb, D, arp, aix, aw, vh)
        fac = tl.where(av == 0.0, 1.0, tl.where(av == 110.0, 0.5, 0.0))
        if fac != 1.0:
            for r in range(nr):
                tl.store(Sb + r * 128 + ch, tl.load(Sb + r * 128 + ch) * fac)
    kmax = tl.max(k, axis=0)
    krow = tl.argmax(k, axis=0)
    if (kmax > 0.0) & (bv == 1.0):
        tl.store(Sb + krow * 128 + ch, v)
    tl.debug_barrier()
    o = tl.zeros([128], dtype=tl.float32)
    for _ in range(QHOT):
        qm = tl.max(q, axis=0)
        qi = tl.argmax(q, axis=0)
        if qm > 0.0:
            o += tl.load(Sb + qi * 128 + ch)
        q = tl.where(ch == qi, 0.0, q)
    ref = tl.sum(tl.where(ch < 2, o * o, 0.0), axis=0)
    norm = tl.math.sqrt_rn(ref / 128.0)
    out = tl.where(ref > 0.0, tl.math.div_rn(o, tl.where(ref > 0.0, norm, 1.0)), 0.0)
    wv = tl.load(ow + vh * 128 + ch)
    dd = tl.load(odim + vh * 128 + ch)
    tl.atomic_add(nxt + bat * 2 * D + 2 * dd + 1, out * z * wv, mask=wv != 0.0)
    _book(cur + bat * 2 * D, nxt + bat * 2 * D, yz + bat * 2 * D, hs + bat * 3 * D, D, vh,
          nprog, SHIFT)


@triton.jit
def _head_scan(rb, h_ptr, D, VMAX,
               qrp, qix, qw, krp, kix, kw, vrp, vix, vw, zrp, zix, zw,
               khead, act, kms, krs, qt, zt, vt,
               KQ: tl.constexpr, KK: tl.constexpr, KV: tl.constexpr, KZ: tl.constexpr,
               KS: tl.constexpr, NC: tl.constexpr):
    """Stage 1 of the split head: this program's NC rows of one value head.

    The head's work is the same 128-row scan for q, k, v and z, but its grid is V -- 11 to 32
    programs, so most of the card idles and the gathers of one head wait on one SM's latency.
    Splitting the rows over KS programs puts them on KS SMs (``gather-latency-report.md``:
    sub 2 drops 53.4 -> 25.8 us at KS = 4 with the same lanes). The four vectors go to scratch;
    only the local argmax travels on, so stage 2 has the same amount of S-table work as before.
    """
    pid = tl.program_id(0)
    bat = tl.program_id(1)
    if tl.load(act + bat) == 0:
        return
    vh, s = pid // KS, pid % KS
    rw = rb + bat * D
    hb = h_ptr + bat * 3 * D
    ch = s * NC + tl.arange(0, NC)
    h = tl.load(khead + vh)
    k = _rows3(rw, hb, D, krp, kix, kw, h * 128, ch, KK)
    v = _rows3(rw, hb, D, vrp, vix, vw, vh * 128, ch, KV)
    z = _rows3(rw, hb, D, zrp, zix, zw, vh * 128, ch, KZ)
    q = _rows3(rw, hb, D, qrp, qix, qw, h * 128, ch, KQ)
    o = (bat * VMAX + vh) * 128 + ch
    tl.store(qt + o, q)
    tl.store(zt + o, z)
    tl.store(vt + o, v)
    tl.store(kms + (bat * VMAX + vh) * KS + s, tl.max(k, axis=0))
    tl.store(krs + (bat * VMAX + vh) * KS + s, s * NC + tl.argmax(k, axis=0))


@triton.jit
def _head_pick(rb, cur, nxt, yz, hs, h_ptr, S_ptr, D, V, VMAX,
               brp, bix, bw, arp, aix, aw, bconst, odim, ow, nrows, act,
               kms, krs, qt, zt, vt,
               QHOT: tl.constexpr, SHIFT: tl.constexpr, KS: tl.constexpr):
    """Stage 2 of the split head: the KS local argmaxes merged into the head's, then everything
    that happens once and in order -- the S-table rescale, the winning row's v, the QHOT reads of
    S, the norm, the output and the bookkeeping.

    The merge keeps the strictly larger value, so a tie keeps the lowest row, exactly as one
    ``argmax`` over all 128 rows did. It costs one program per head and a launch; the S-table
    ordering that ``tl.debug_barrier`` used to give is now a kernel boundary.
    """
    vh = tl.program_id(0)
    bat = tl.program_id(1)
    if tl.load(act + bat) == 0:
        return
    nprog = tl.num_programs(0)
    ch = tl.arange(0, 128)
    rw = rb + bat * D
    hb = h_ptr + bat * 3 * D
    o = (bat * VMAX + vh) * 128 + ch
    q = tl.load(qt + o)
    z = tl.load(zt + o)
    v = tl.load(vt + o)
    kmax = tl.load(kms + (bat * VMAX + vh) * KS)
    krow = tl.load(krs + (bat * VMAX + vh) * KS)
    for s in range(1, KS):
        km = tl.load(kms + (bat * VMAX + vh) * KS + s)
        kr = tl.load(krs + (bat * VMAX + vh) * KS + s)
        win = km > kmax
        kmax = tl.where(win, km, kmax)
        krow = tl.where(win, kr, krow)
    bv = _scalar3(rw, hb, D, brp, bix, bw, vh) + tl.load(bconst + vh)
    Sb = S_ptr + (bat * V + vh) * 16384
    nr = tl.load(nrows + vh)
    if nr > 0:
        av = _scalar3(rw, hb, D, arp, aix, aw, vh)
        fac = tl.where(av == 0.0, 1.0, tl.where(av == 110.0, 0.5, 0.0))
        if fac != 1.0:
            for r in range(nr):
                tl.store(Sb + r * 128 + ch, tl.load(Sb + r * 128 + ch) * fac)
    if (kmax > 0.0) & (bv == 1.0):
        tl.store(Sb + krow * 128 + ch, v)
    tl.debug_barrier()
    acc = tl.zeros([128], dtype=tl.float32)
    for _ in range(QHOT):
        qm = tl.max(q, axis=0)
        qi = tl.argmax(q, axis=0)
        if qm > 0.0:
            acc += tl.load(Sb + qi * 128 + ch)
        q = tl.where(ch == qi, 0.0, q)
    ref = tl.sum(tl.where(ch < 2, acc * acc, 0.0), axis=0)
    norm = tl.math.sqrt_rn(ref / 128.0)
    out = tl.where(ref > 0.0, tl.math.div_rn(acc, tl.where(ref > 0.0, norm, 1.0)), 0.0)
    wv = tl.load(ow + vh * 128 + ch)
    dd = tl.load(odim + vh * 128 + ch)
    tl.atomic_add(nxt + bat * 2 * D + 2 * dd + 1, out * z * wv, mask=wv != 0.0)
    _book(cur + bat * 2 * D, nxt + bat * 2 * D, yz + bat * 2 * D, hs + bat * 3 * D, D, vh,
          nprog, SHIFT)


@triton.jit
def _rows4(rb, hb, D, rp, ix, w, ids, ent, ch, NRC: tl.constexpr, KP: tl.constexpr):
    """_rows3 over only the entity's non-empty rows (ids: their row numbers, -1 padding), placed
    into the 128-wide vector by a compare-and-sum."""
    c = tl.arange(0, NRC)
    base = ent * NRC
    r0 = tl.load(rp + base + c)
    n = tl.load(rp + base + c + 1) - r0
    kk = tl.arange(0, KP)
    mk = kk[None, :] < n[:, None]
    i = tl.load(ix + r0[:, None] + kk[None, :], mask=mk, other=0)
    acc = tl.maximum(tl.sum(tl.load(w + r0[:, None] + kk[None, :], mask=mk, other=0.0)
                            * _lag(rb, hb, i, mk, D), axis=1), 0.0)
    rid = tl.load(ids + base + c)
    return tl.sum(tl.where(ch[:, None] == rid[None, :], acc[None, :], 0.0), axis=1)


@triton.jit
def _head4(rb, cur, nxt, yz, hs, h_ptr, S_ptr, D, V,
           qrp, qix, qw, qid, krp, kix, kw, kid, vrp, vix, vw, vid, zrp, zix, zw, zid,
           arp, aix, aw, brp, bix, bw, bconst, odim, ow, khead, nrows, act,
           KQ: tl.constexpr, KK: tl.constexpr, KV: tl.constexpr, KZ: tl.constexpr,
           NQ: tl.constexpr, NK: tl.constexpr, NV: tl.constexpr, NZ: tl.constexpr,
           QHOT: tl.constexpr, SHIFT: tl.constexpr):
    vh = tl.program_id(0)
    bat = tl.program_id(1)
    if tl.load(act + bat) == 0:
        return
    nprog = tl.num_programs(0)
    ch = tl.arange(0, 128)
    rw = rb + bat * D
    hb = h_ptr + bat * 3 * D
    h = tl.load(khead + vh)
    k = _rows4(rw, hb, D, krp, kix, kw, kid, h, ch, NK, KK)
    v = _rows4(rw, hb, D, vrp, vix, vw, vid, vh, ch, NV, KV)
    z = _rows4(rw, hb, D, zrp, zix, zw, zid, vh, ch, NZ, KZ)
    q = _rows4(rw, hb, D, qrp, qix, qw, qid, h, ch, NQ, KQ)
    bv = _scalar3(rw, hb, D, brp, bix, bw, vh) + tl.load(bconst + vh)
    Sb = S_ptr + (bat * V + vh) * 16384
    nr = tl.load(nrows + vh)
    if nr > 0:
        av = _scalar3(rw, hb, D, arp, aix, aw, vh)
        fac = tl.where(av == 0.0, 1.0, tl.where(av == 110.0, 0.5, 0.0))
        if fac != 1.0:
            for r in range(nr):
                tl.store(Sb + r * 128 + ch, tl.load(Sb + r * 128 + ch) * fac)
    kmax = tl.max(k, axis=0)
    krow = tl.argmax(k, axis=0)
    if (kmax > 0.0) & (bv == 1.0):
        tl.store(Sb + krow * 128 + ch, v)
    tl.debug_barrier()
    o = tl.zeros([128], dtype=tl.float32)
    for _ in range(QHOT):
        qm = tl.max(q, axis=0)
        qi = tl.argmax(q, axis=0)
        if qm > 0.0:
            o += tl.load(Sb + qi * 128 + ch)
        q = tl.where(ch == qi, 0.0, q)
    ref = tl.sum(tl.where(ch < 2, o * o, 0.0), axis=0)
    norm = tl.math.sqrt_rn(ref / 128.0)
    out = tl.where(ref > 0.0, tl.math.div_rn(o, tl.where(ref > 0.0, norm, 1.0)), 0.0)
    wv = tl.load(ow + vh * 128 + ch)
    dd = tl.load(odim + vh * 128 + ch)
    tl.atomic_add(nxt + bat * 2 * D + 2 * dd + 1, out * z * wv, mask=wv != 0.0)
    _book(cur + bat * 2 * D, nxt + bat * 2 * D, yz + bat * 2 * D, hs + bat * 3 * D, D, vh,
          nprog, SHIFT)


@triton.jit
def _decode_read(rb, D, fam_idx, pay_idx, ids, tok_out, fid_out, p_out, pay_out, own_out,
                 NF: tl.constexpr, NP: tl.constexpr, NFAM: tl.constexpr):
    """One program per row: the output code (family and payload bits) into the next token, and
    whether the circuit owns the position (any code dim non-zero: the exit zeroes them all
    where it does not). A set bit is a positive code dim, which holds for the standalone
    machine's 0/1 select dims and for the merged model's KAPPA * (2 bit - 1) exit gains."""
    bat = tl.program_id(0)
    j = tl.arange(0, 32)
    w2 = tl.exp2(j.to(tl.float32))
    mf = j < NF
    rf = _pair(rb + bat * D, tl.load(fam_idx + j, mask=mf, other=0), mf)
    fb = (rf > 0).to(tl.float32)
    fid = tl.sum(tl.where(mf, fb * w2, 0.0), axis=0).to(tl.int32)
    fid = tl.minimum(tl.maximum(fid, 0), NFAM - 1)
    mp = j < NP
    rp = _pair(rb + bat * D, tl.load(pay_idx + j, mask=mp, other=0), mp)
    pb = (rp > 0).to(tl.float32)
    p = tl.sum(tl.where(mp, pb * w2, 0.0), axis=0).to(tl.int64)
    own = tl.sum((mf & (rf != 0.0)).to(tl.int32)) + tl.sum((mp & (rp != 0.0)).to(tl.int32))
    tl.store(tok_out + bat, tl.load(ids + fid) + p)
    tl.store(fid_out + bat, fid)
    tl.store(p_out + bat, p)
    tl.store(pay_out + bat * 32 + j, pb, mask=mp)
    tl.store(own_out + bat, (own > 0).to(tl.int32))


@triton.jit
def _keep_read(rb, D, keep_idx, keep_out, N, BLOCK: tl.constexpr):
    """One row: the kept dims of the final residual (the output code, tau, mu) into keep_out
    (rv/chat.py's hc, read between the last sublayer and the decode's embeds)."""
    bat = tl.program_id(0)
    for off in range(0, N, BLOCK):
        lane = off + tl.arange(0, BLOCK)
        m = lane < N
        i = tl.load(keep_idx + lane, mask=m, other=0)
        tl.store(keep_out + bat * N + lane, _pair(rb + bat * D, i, m), mask=m)


@triton.jit
def _decode_embed(rt, kstride, D, fid_in, p_in, pay_in, small, row0, table, base, bits, NT,
                  NP: tl.constexpr, NPAIR: tl.constexpr):
    """The next token's embedding into the X half of pair 0 (small families from the table, big
    ones linear in the payload bits), and every Y half zeroed.

    Every pair's, not one per Y slot: a Y slot appears in two of the six pairs, and only the copy
    the rotation reaches is kept clear. Both have to start the next step at zero, just as
    Executor2's three Y buffers did.
    """
    pid = tl.program_id(0)
    bat = tl.program_id(1)
    i = pid * 1024 + tl.arange(0, 1024)
    m = i < D
    fid = tl.load(fid_in + bat)
    if tl.load(small + fid) != 0:
        row = tl.minimum(tl.maximum(tl.load(row0 + fid) + tl.load(p_in + bat), 0), NT - 1)
        x = tl.load(table + row * D + i, mask=m, other=0.0)
    else:
        x = tl.load(base + fid * D + i, mask=m, other=0.0)
        for b in range(NP):
            x += tl.load(pay_in + bat * 32 + b) * tl.load(bits + (fid * NP + b) * D + i, mask=m, other=0.0)
    o, z = bat * 2 * D + 2 * i, tl.zeros([1024], dtype=tl.float32)
    tl.store(rt + o, x, mask=m)
    for k in range(NPAIR):
        tl.store(rt + k * kstride + o + 1, z, mask=m)


class FreeRun3:
    """fastexec.FreeRun's interface on Executor3: generation on the GPU, one graph per step.

    ``keep`` are the circuit dims of the final residual read into ``self.keep`` every step, for
    rv/chat.py's hc. Every step decodes the output code into the next token and embeds it there
    anyway (that is what makes a free run one graph replay), so the two paths share one graph
    and one state: ``step(emb)`` feeds an embedding from the outside, ``run_owned`` lets each
    step embed the token it decoded, and ``keep`` is read either way.
    """

    def __init__(self, c, batch: int = 1, device="cuda", keep: list[int] | None = None):
        from chipformer.rv import fastexec

        self.c, self.B, self.dev = c, batch, device
        # the tables of the v2 FreeRun (token embeddings, decode maps), and Executor2's arrays
        fr2 = fastexec.FreeRun(c, batch, device, compile_=False, v2=True)
        self.fr2, self.ex = fr2, fr2.ex
        ex, D = self.ex, self.ex.D
        self.D = D
        # the six interleaved residual pairs (see the module doc); R[k] is int64-viewable so a
        # lane reads its (X, Y) pair with one 8-byte load
        self.Rt = torch.zeros(PAIRS, batch, 2 * D, device=device)
        self.R = [self.Rt[k] for k in range(PAIRS)]
        self.R64 = [r.view(torch.int64) for r in self.R]
        self.L = len(ex.steps)
        assert ex.steps[-1]["kind"] == "ffn", "the last sublayer is the exit MLP"
        self.dummy = torch.zeros(batch, 3, D, device=device)
        # the rows this step moves: a parked row keeps every buffer of its own where it was
        self.act = torch.ones(batch, dtype=torch.int32, device=device)
        self.keep_idx = (None if keep is None
                         else torch.tensor(keep, dtype=torch.int32, device=device))
        self.NK = 0 if keep is None else len(keep)
        self.keep = None if keep is None else torch.zeros(batch, self.NK, device=device)
        self.fam_idx = fr2.fam_idx.to(torch.int32)
        self.pay_idx = fr2.pay_idx.to(torch.int32)
        self.NF, self.NP = len(self.fam_idx), len(self.pay_idx)
        assert self.NF <= 32 and self.NP <= 32
        self.tok = torch.zeros(batch, dtype=torch.long, device=device)
        self.own = torch.zeros(batch, dtype=torch.int32, device=device)
        self.fid = torch.zeros(batch, dtype=torch.int32, device=device)
        self.p = torch.zeros(batch, dtype=torch.long, device=device)
        self.pay = torch.zeros(batch, 32, dtype=torch.float32, device=device)
        self.small = fr2.small.to(torch.int32)
        self.NP_bits = fr2.bits.shape[1]
        self.bits = fr2.bits.contiguous()
        self.sorted, self.compact = {}, {}
        for i, st in enumerate(ex.steps):
            if st["kind"] == "ffn":
                self.sorted[i] = self._sort_units(st)
            else:
                self.compact[i] = {nm: self._compact(st[nm]) for nm in ("q", "k", "v", "z")}
        # scrub for the split head: the four row vectors of each value head, and the KS local
        # argmaxes stage 1 leaves for stage 2 to merge
        self.vmax = max(st["V"] for st in ex.steps if st["kind"] == "head")
        nh = batch * self.vmax
        self.qt, self.zt, self.vt = (torch.zeros(nh * 128, device=device) for _ in range(3))
        self.kms = torch.zeros(nh * SPLIT, device=device)
        self.krs = torch.zeros(nh * SPLIT, dtype=torch.int32, device=device)
        self.use4, self.head_kind = {}, {}
        self.graph = None

    def _compact(self, csr3):
        """Rows grouped by entity (128 each: a key head's q / k, a value head's v / z), only the
        non-empty ones, padded to the sublayer's most: (rp, ix, w, ids, NRC)."""
        rp, ix, w = (t.cpu().numpy() for t in csr3)
        n = np.diff(rp)
        ents = len(n) // 128
        keep = [[r for r in range(128) if n[e * 128 + r]] for e in range(ents)]
        nrc = max(2, 1 << max(max((len(k) for k in keep), default=1) - 1, 1).bit_length())
        rows, ids = [], []
        for e, kr in enumerate(keep):
            for j in range(nrc):
                if j < len(kr):
                    r = e * 128 + kr[j]
                    rows.append((ix[rp[r]:rp[r + 1]], w[rp[r]:rp[r + 1]]))
                    ids.append(kr[j])
                else:
                    rows.append((np.zeros(0, np.int32), np.zeros(0, np.float32)))
                    ids.append(-1)
        nrp = np.concatenate([[0], np.cumsum([len(r[0]) for r in rows])]).astype(np.int32)
        nix = np.concatenate([r[0] for r in rows] + [np.zeros(1, np.int32)]).astype(np.int32)
        nw = np.concatenate([r[1] for r in rows] + [np.zeros(1, np.float32)]).astype(np.float32)
        t = lambda a: torch.tensor(a, device=self.dev)
        return [t(nrp), t(nix), t(nw), t(np.asarray(ids, np.int32))], nrc

    def _sort_units(self, st):
        """The sublayer's units by b-row length (longest first), with each block's longest rows."""
        dev, BL = self.dev, self.ex.BLOCK
        rpa, ixa, wa = (t.cpu().numpy() for t in st["a"])
        rpb, ixb, wb = (t.cpu().numpy() for t in st["b"])
        na, nb = np.diff(rpa), np.diff(rpb)
        order = np.lexsort((-na, -nb))

        def csr(rp, ix, w):
            rows = [(ix[rp[u]:rp[u + 1]], w[rp[u]:rp[u + 1]]) for u in order]
            nrp = np.concatenate([[0], np.cumsum([len(r[0]) for r in rows])]).astype(np.int32)
            nix = np.concatenate([r[0] for r in rows] + [np.zeros(1, np.int32)]).astype(np.int32)
            nw = np.concatenate([r[1] for r in rows] + [np.zeros(1, np.float32)]).astype(np.float32)
            return [torch.tensor(a, device=dev) for a in (nrp, nix, nw)]

        U = st["U"]
        nblk = -(-U // BL)
        bka = [int(na[order[b * BL:(b + 1) * BL]].max()) for b in range(nblk)]
        bkb = [int(nb[order[b * BL:(b + 1) * BL]].max()) for b in range(nblk)]
        t32 = lambda a: torch.tensor(np.asarray(a, np.int32), device=dev)
        return {"a": csr(rpa, ixa, wa), "b": csr(rpb, ixb, wb),
                "coef": st["coef"][torch.tensor(order, device=dev)].contiguous(),
                "out": st["out"][torch.tensor(order, device=dev)].contiguous(),
                "bka": t32(bka), "bkb": t32(bkb)}

    def load(self, row: int, first_token: int):
        self.ex.load_init(row)
        self.ex.reset([row])
        self.R[0][row, 0::2] = self.ex.embed([first_token])[0]
        for r in self.R:
            r[row, 1::2].zero_()

    @property
    def X(self) -> list[torch.Tensor]:
        """The two X slots as the next step's sublayers will read them: a strided view of the X
        half of the pair that sublayer 0 and sublayer 1 start from (``X[i][b, d]`` is element d
        of the b-th row's residual base).

        Between steps this is where a batch executor drops one row's embedding
        (``rvbatch`` does ``fr.X[0][b].copy_(emb)``); inside a step the pairs are the state and
        this view is not the place to look. Writing ``X[1]`` has no effect -- sublayer 0's own
        bookkeeping overwrites that half before sublayer 1 reads it.
        """
        return [self.R[k][:, 0::2] for k in (0, 1)]

    def embed(self, toks: list[int]) -> torch.Tensor:
        """Executor input rows for chip-vocabulary token ids (the circuit's own embedding)."""
        return self.ex.embed(toks)

    def zero_state(self, rows: list[int] | torch.Tensor | None = None) -> None:
        """The head states back to zero, with their lag windows (the circuit from nothing): a
        prefill before the snapshot, as rv/chat.py starts a conversation. ``rows`` limits it to
        those rows of a shared executor (the others keep their conversation's state)."""
        for st in self.ex.steps:
            if st["kind"] != "head":
                continue
            if rows is None:
                st["S"].zero_()
                st["hist"].zero_()
            else:
                st["S"][rows].zero_()
                st["hist"][rows].zero_()

    def init_rows(self) -> list[np.ndarray]:
        """The head states ``load_init`` would install, from the circuit's current ``init``: one
        array per head sublayer. Read it right after ``core.snapshot`` and hand it to
        ``load_init`` later, so a shared executor's rows can carry different machines."""
        return self.ex.init_rows()

    def load_init(self, row: int = 0, rows: list[np.ndarray] | None = None) -> None:
        """The head states from the graph's tables: the snapshot core.snapshot wrote (or the
        ``init_rows`` read earlier), into ``row``."""
        self.ex.load_init(row, rows)

    def set_input(self, emb: torch.Tensor) -> None:
        """The next step's embedding without stepping: the outside-driven input of the keep
        path, and the first step of a free run after a token it did not decode itself."""
        self.R[0][:, 0::2].copy_(emb)

    def save_state(self) -> list[torch.Tensor]:
        """Every recurrent buffer (the six residual pairs, each head's state and lag windows):
        the rollback point for a run that went past a position the circuit does not own."""
        heads = [st for st in self.ex.steps if st["kind"] == "head"]
        return ([t.clone() for t in self.R]
                + [st["S"].clone() for st in heads] + [st["hist"].clone() for st in heads])

    def restore_state(self, saved: list[torch.Tensor]) -> None:
        """Put a ``save_state`` snapshot back."""
        heads = [st for st in self.ex.steps if st["kind"] == "head"]
        bufs = self.R + [st["S"] for st in heads] + [st["hist"] for st in heads]
        for b, s0 in zip(bufs, saved):
            b.copy_(s0)

    @torch.no_grad()
    def step(self, emb: torch.Tensor | None = None,
             act: torch.Tensor | None = None) -> None:
        """One step of the keep path: ``emb`` ([B, D], Executor2's input rows) becomes the next
        embedding and the graph replays; the kept dims are in ``keep`` afterwards. Without
        ``emb`` the decoded token's embedding stands, i.e. a free run one step at a time.

        ``act`` ([B], int32) parks the rows it leaves 0: their residual, head states and lag
        windows do not move, so a batch executor can step the live conversations alone while
        the rest wait (rvbatch.RvBatch). Omitted, every row steps."""
        assert self.keep_idx is not None, "a free-running instance steps with run()"
        if self.graph is None:
            self.capture()  # it leaves act all-live, so the mask is set after it
        if act is not None:
            self.act.copy_(act)
        if emb is not None:
            self.set_input(emb)
        self.graph.replay()

    def _step(self):
        ex, D, B = self.ex, self.D, self.B
        prev_hist = None  # the history the previous sublayer (a head) left to move on
        for i, st in enumerate(ex.steps):
            k = i % PAIRS
            rb, cur, nxt = self.R64[k], self.R[k], self.R[(k + 1) % PAIRS]
            yz = self.R[(k + 2) % PAIRS]
            shift = prev_hist is not None
            hs = prev_hist if shift else self.dummy
            if st["kind"] == "ffn":
                self._ffn(i, st, rb, cur, nxt, yz, hs, shift, self.use4.get(i, False))
                prev_hist = None
            else:
                self._head(i, st, rb, cur, nxt, yz, hs, shift, self.head_kind.get(i, 0))
                prev_hist = st["hist"]
        rb = self.R64[self.L % PAIRS]
        if self.keep_idx is not None:  # rv/chat.py also reads the kept dims every step
            _keep_read[(B,)](rb, D, self.keep_idx, self.keep, self.NK, BLOCK=128)
        fr2 = self.fr2
        _decode_read[(B,)](rb, D, self.fam_idx, self.pay_idx, fr2.ids, self.tok, self.fid,
                           self.p, self.pay, self.own, NF=self.NF, NP=self.NP,
                           NFAM=len(fr2.ids))
        _decode_embed[(triton.cdiv(D, 1024), B)](
            self.Rt, B * 2 * D, D, self.fid, self.p, self.pay, self.small, fr2.row0,
            fr2.table, fr2.base, self.bits, len(fr2.table), NP=self.NP_bits, NPAIR=PAIRS)

    def _ffn(self, i, st, rb, cur, nxt, yz, hs, shift, sorted_):
        ex, D, B = self.ex, self.D, self.B
        grid = (triton.cdiv(st["U"], ex.BLOCK), B)
        if sorted_:
            so = self.sorted[i]
            _ffn4[grid](rb, cur, nxt, yz, hs, D, st["U"], *so["a"], *so["b"], so["coef"],
                        so["out"], so["bka"], so["bkb"], self.act, BLOCK=ex.BLOCK, KC=16,
                        SHIFT=shift)
        else:
            _ffn3[grid](rb, cur, nxt, yz, hs, D, st["U"], *st["a"], *st["b"], st["coef"],
                        st["out"], self.act, BLOCK=ex.BLOCK, KA=st["KA"], KB=st["KB"],
                        SHIFT=shift)

    def _head(self, i, st, rb, cur, nxt, yz, hs, shift, kind: int) -> None:
        """One head sublayer, by whichever of its three bodies ``tune`` found fastest here.

        ``kind`` is 0 for the single-program scan (_head3), 1 for the split scan (_head_scan +
        _head_pick, the default when the card is idle), 2 for _head4's non-empty-rows scan.
        """
        D, B = self.D, self.B
        if kind == 2:
            cq = self.compact[i]
            _head4[(st["V"], B)](
                rb, cur, nxt, yz, hs, st["hist"], st["S"], D, st["V"], *cq["q"][0], *cq["k"][0],
                *cq["v"][0], *cq["z"][0], *st["a"], *st["b"], st["bconst"], st["odim"], st["ow"],
                st["khead"], st["nrows"], self.act, KQ=st["KQ"], KK=st["KK"], KV=st["KV"],
                KZ=st["KZ"], NQ=cq["q"][1], NK=cq["k"][1], NV=cq["v"][1], NZ=cq["z"][1],
                QHOT=st["QHOT"], SHIFT=shift)
        elif kind == 0:
            _head3[(st["V"], B)](
                rb, cur, nxt, yz, hs, st["hist"], st["S"], D, st["V"], *st["q"], *st["k"],
                *st["v"], *st["z"], *st["a"], *st["b"], st["bconst"], st["odim"], st["ow"],
                st["khead"], st["nrows"], self.act, KQ=st["KQ"], KK=st["KK"], KV=st["KV"],
                KZ=st["KZ"], QHOT=st["QHOT"], SHIFT=shift)
        else:
            _head_scan[(st["V"] * SPLIT, B)](
                rb, st["hist"], D, self.vmax, *st["q"], *st["k"], *st["v"], *st["z"],
                st["khead"], self.act, self.kms, self.krs, self.qt, self.zt, self.vt,
                KQ=st["KQ"], KK=st["KK"], KV=st["KV"], KZ=st["KZ"], KS=SPLIT, NC=128 // SPLIT)
            _head_pick[(st["V"], B)](
                rb, cur, nxt, yz, hs, st["hist"], st["S"], D, st["V"], self.vmax,
                *st["b"], *st["a"], st["bconst"], st["odim"], st["ow"], st["nrows"],
                self.act, self.kms, self.krs, self.qt, self.zt, self.vt,
                QHOT=st["QHOT"], SHIFT=shift, KS=SPLIT)

    @staticmethod
    def _launch_us(fn, reps: int = 30) -> float:
        """Device time of one call of ``fn``, measured under its own graph.

        A graph is not a convenience here: the split head is two launches, and an eager timing
        loop would charge it both launches' host overhead (~5-10 us each) while a step pays only
        the ~0.2 us gap between them -- which is enough to make the single-launch variant look
        faster than it is and pick the wrong one.
        """
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(reps):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1) / reps

    def tune(self, reps: int = 30):
        """Per sublayer, the faster of the two bodies this GPU has for it (they give the same
        values): _ffn3 by padded rows against _ffn4 by sorted rows, and the split head scan
        against _head4's non-empty rows."""
        r0 = torch.zeros_like(self.R[0])
        scratch = [r0.view(torch.int64), r0, torch.zeros_like(r0), torch.zeros_like(r0),
                   torch.zeros_like(self.dummy)]
        pinned = dict(self.head_kind)  # a body set from outside wins over the measurement
        self.use4 = {}
        for i, st in enumerate(self.ex.steps):
            if st["kind"] != "ffn":
                continue
            best = [self._launch_us(lambda v=v: self._ffn(i, st, *scratch, False, v), reps)
                    for v in (False, True)]
            self.use4[i] = best[1] < best[0]
        self.head_kind = {}
        for i, st in enumerate(self.ex.steps):
            if st["kind"] != "head":
                continue
            S0, H0 = st["S"].clone(), st["hist"].clone()
            best = []
            for kd in range(3):
                st["S"].copy_(S0)
                st["hist"].copy_(H0)
                best.append(self._launch_us(
                    lambda kd=kd: self._head(i, st, *scratch, False, kd), reps))
            st["S"].copy_(S0)
            st["hist"].copy_(H0)
            self.head_kind[i] = best.index(min(best))
        self.head_kind.update(pinned)

    def capture(self):
        if not self.use4:
            self.tune()
        self.act.fill_(1)  # the graph is captured with every row live; step() sets the mask
        saved = self.save_state()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._step()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._step()
        self.restore_state(saved)
        self.graph = g

    def run_owned(self, n_steps: int,
                  act: torch.Tensor | None = None) -> tuple[np.ndarray, np.ndarray]:
        """[B, n_steps] decoded tokens and whether the circuit owned each position (its output
        code non-zero; where it does not, the exit zeroes every code dim). A free run's chip
        burst is the prefix it owns, so this is what rv/chat.py reads a chunk at a time instead
        of one token per synchronise."""
        if self.graph is None:
            self.capture()  # it leaves act all-live, so the mask is set after it
        if act is not None:
            self.act.copy_(act)
        toks = torch.zeros(self.B, n_steps, dtype=torch.long, device=self.dev)
        own = torch.zeros(self.B, n_steps, dtype=torch.int32, device=self.dev)
        for t in range(n_steps):
            self.graph.replay()
            toks[:, t].copy_(self.tok)
            own[:, t].copy_(self.own)
        torch.cuda.synchronize()
        return toks.cpu().numpy(), own.cpu().numpy()

    def run(self, n_steps: int) -> np.ndarray:
        """[B, n_steps] generated tokens (the free-running path of rv/storyrun.py)."""
        return self.run_owned(n_steps)[0]


def timed3(c, toks: list[int], steps: int = 500, batch: int = 1) -> float:
    """Seconds per step of B rows with FreeRun3, the embeddings fed from the outside (the
    rv/chat.py path), under its CUDA graph; the replays' kept dims are checked against the IR
    first. The counterpart of ``fastexec.timed2``'s Executor2 row."""
    from chipformer.q35 import ir

    fr = FreeRun3(c, batch, "cuda", keep=c.keep)
    for b in range(batch):
        fr.load(b, toks[0])
    emb = fr.embed(toks[:steps + 20])
    heads = [st for st in fr.ex.steps if st["kind"] == "head"]
    states = [st["S"].clone() for st in heads]
    keep = list(c.keep)

    def restore():
        for st, s0 in zip(heads, states):
            st["S"].copy_(s0)
        fr.ex.reset()

    fr.capture()
    n = min(300, len(emb))
    want = ir.evaluate(c, toks[:n], check=False)[:, keep]
    restore()
    for t in range(n):
        fr.step(emb[t].expand(batch, -1))
        got = fr.keep.double().cpu().numpy()
        assert (got == want[t][None]).all(), f"graph replay differs at {t}"
    restore()
    for t in range(20):
        fr.step(emb[t].expand(batch, -1))
    torch.cuda.synchronize()
    t0 = time.time()
    for t in range(20, 20 + steps):
        fr.step(emb[t].expand(batch, -1))
    torch.cuda.synchronize()
    return (time.time() - t0) / steps
