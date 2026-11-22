"""Merge the RV32 machine and its text bridge into Qwen3.5-4B (layout "rv1").

    python -m chipformer.rv.merge SRC --out dist/q4-rv1 [--no-mute] [--vx] [--overlay]

The result is a text-only stock ``Qwen3_5ForCausalLM``:

- the chip layers come first (the circuit of ``core.build(text=...)``), then Qwen's layers,
  zero-padded to the new hidden size (q35.merge.widen's rules: Qwen never reads or writes chip
  dims; the full-residual norms are rescaled so its normalized inputs stay the same);
- every Qwen vocabulary row carries the ballast and its text features in the chip dims (is_text,
  tau, its first 16 UTF-8 bytes, the ``<tool_call>`` / ``</tool_call>`` flags, and for the 260
  printable tokens the output code of their echo); the RV32 tokens are appended after Qwen's
  vocabulary as special tokens;
- the tied output head reads only the output code and tau among the chip dims (q35.merge's
  final-norm gains), and Qwen's layers are muted on chip tokens (q35.mute, on mu).

With ``overlay`` the circuit dims that only chip tokens need live in Qwen's dims instead of a
chip region of their own (q35/overlay.py, measured on ``rv.corpus``), which makes the checkpoint
narrower: chip tokens see the circuit's value there, text tokens keep Qwen's embedding.

With ``kernel`` a compiled 64-lane matvec (q35/kernel.py) gets one more layer right after the
circuit's: Qwen's own layers move up by one, and the kernel's virtual tokens sit in the vocabulary
padding right after the RV32 tokens, so the checkpoint can be handed a kernel token id directly.
The kernel's weights are gated by an ``is_kernel`` token feature, so they act on a kernel token and
are exactly dead everywhere else.

``chipformer.json`` records the layout for the decoder (rv/chat.py).
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from chipformer.q35.merge import (IM_START, IN_COLS, OUT_ROWS, TAU_GAIN, WIDE_NORMS,
                                  pick_hidden)
from chipformer.rv import bridge as BR
from chipformer.rv import core
from chipformer.rv import spec as S

SRC_TEXT = "model.language_model."
DST = "model."


@dataclass(frozen=True)
class KernelPlan:
    """A compiled matvec to emit as the layer right after the circuit's (q35/kernel.py).

    ``weight`` is the ``(lanes, lanes)`` matrix, ``inputs`` the ``(n_tokens, lanes)`` rows of the
    kernel's virtual tokens. The rows go into the vocabulary padding right after the RV32 tokens,
    so the merged checkpoint can be handed a kernel token id and nothing else. ``shared`` reuses
    the input columns as the result columns (80 columns instead of 144).
    """

    weight: np.ndarray
    inputs: np.ndarray
    shared: bool = True


def kernel_layer(lay, n_slots: int, tc: dict, first: int, token_base: int, plan: KernelPlan):
    """The kernel's IR graph and the weights of its layer (``first``), keyed like a Qwen layer."""
    from chipformer.q35 import kernel as K

    slot = K.slot_after_chip(lay, n_slots, shared=plan.shared)
    k = K.build(plan.weight, plan.inputs, slot, lay, token_base=token_base)
    return k, K.emit(k, tc, first, DST)


def kernel_rows(k) -> torch.Tensor:
    """The embedding rows of the kernel's virtual tokens: ballast plus their input lanes."""
    from chipformer.q35 import kernel as K

    return K.residual_rows(k, k.tokens())


def shard_keys(keys: list[str], sizes: dict[str, int], limit: int = 2**31) -> list[list[str]]:
    """The greedy shard split of a merged checkpoint: sorted keys, 2 GiB per shard.

    The checkpoint is written in two passes -- the layer weights, then the embedding on its own
    last shard -- because holding both is ~28 GiB at 4B's sizes and the merge has to fit next to a
    7 GiB fp32 chip-token matrix. The weight names, ``config.json`` and ``chipformer.json`` are the
    same either way; only the shard each embedding row lands in changes.
    """
    groups: list[list[str]] = []
    current: list[str] = []
    size = 0
    for key in keys:
        current.append(key)
        size += sizes[key]
        if size > limit:
            groups.append(current)
            current, size = [], 0
    groups.append(current)
    return groups


def chip_token_names() -> list[str]:
    names = []
    for t in range(S.VOCAB):
        fam, p = S.family(t)
        names.append(f"<|rv:{fam}|>" if fam in S.CONTROLS else f"<|rv:{fam}:{p}|>")
    return names


def extend_tokenizer(src: Path, dst: Path, first_id: int) -> None:
    import shutil

    from tokenizers import AddedToken, Tokenizer

    tok = Tokenizer.from_file(str(src / "tokenizer.json"))
    n = tok.get_vocab_size(with_added_tokens=True)
    pad = [AddedToken(f"<|reserved_{i}|>", special=True) for i in range(first_id - n)]
    chip = [AddedToken(s, special=True) for s in chip_token_names()]
    tok.add_special_tokens(pad + chip)
    assert tok.token_to_id(chip_token_names()[0]) == first_id
    dst.mkdir(parents=True, exist_ok=True)
    tok.save(str(dst / "tokenizer.json"))
    for f in ("chat_template.jinja", "LICENSE"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)
    # tokenizer.json holds every added token; listing them in the config too makes it tens of
    # MB, too big for the hubs to parse
    tc = json.loads((src / "tokenizer_config.json").read_text(encoding="utf-8"))
    tc.pop("added_tokens_decoder", None)
    (dst / "tokenizer_config.json").write_text(json.dumps(tc, indent=2, ensure_ascii=False),
                                               encoding="utf-8")


def text_rows(c, lay, T: BR.Text, base_emb: torch.Tensor, v0: int) -> torch.Tensor:
    """Qwen's rows: the Qwen part, plus the chip features of each text token (a few special
    rows exactly, every other row = the plain text row plus its byte bits)."""
    from chipformer.q35.emit import embedding_rows

    h = lay.hidden
    d0 = base_emb.shape[1]
    byte_dims = [next(iter(c.tb[k][i].terms)) for k in range(S.TEXT_ROW) for i in range(8)]
    assert not set(byte_dims) & set(lay.over or {}), \
        "a captured byte feature cannot live in Qwen's dims (a text row carries Qwen's input)"
    full = torch.zeros(v0, h, dtype=torch.bfloat16)
    full[:, :d0] = base_emb[:v0, :d0]
    special = sorted({T.TCO, T.TCC, *T.printable})
    q_plain = next(q for q in range(T.n) if q not in set(special))
    plain = embedding_rows(c, lay, [BR.text_token(q_plain)], h)[0]  # (bytes zeroed below)
    byte_cols = [[lay.dim(next(iter(c.tb[k][i].terms))) for i in range(8)]
                 for k in range(S.TEXT_ROW)]
    for k in range(S.TEXT_ROW):
        for i in range(8):
            plain[byte_cols[k][i]] = 0.0
    full[:, d0:] = plain[d0:].to(torch.bfloat16)
    by = np.zeros((v0, S.TEXT_ROW), dtype=np.uint8)
    for q in range(min(v0, T.n)):
        b = T.row_bytes(q)
        by[q] = np.frombuffer(b, dtype=np.uint8)
    for k in range(S.TEXT_ROW):
        for i in range(8):
            full[:, byte_cols[k][i]] = torch.tensor((by[:, k] >> i) & 1, dtype=torch.bfloat16)
    rows = embedding_rows(c, lay, [BR.text_token(q) for q in special], h)
    for q, r in zip(special, rows):
        full[q, d0:] = r[d0:].to(torch.bfloat16)
    return full


def build_merged(src, dst, mute_qwen: bool = True, hidden: int | None = None,
                 m: int | None = None, vx: bool = False, overlay: bool = False,
                 kernel: KernelPlan | None = None) -> dict:
    from safetensors.torch import save_file

    from chipformer.q35.emit import Layout, allocate, check_bf16, emit_layers, live_token_dims
    from chipformer.q35.ir import Target
    from chipformer.q35.tiny import code_gain
    from chipformer.rv.tiny import embedding_matrix

    src, dst = Path(src), Path(dst)
    cfg = json.loads((src / "config.json").read_text())
    tc0 = cfg["text_config"]
    d0, v0, inter = tc0["hidden_size"], tc0["vocab_size"], tc0["intermediate_size"]
    T = BR.Text(src / "tokenizer.json")
    c = core.build(Target(tc0["linear_num_key_heads"], tc0["linear_num_value_heads"], inter), T, vx=vx)
    keep = set(c.keep)
    plan = None
    pinned = None
    if overlay:  # circuit dims that only chip tokens need move into Qwen's dims (q35/overlay.py)
        from chipformer.q35 import overlay as ov
        from chipformer.rv import corpus as CO

        docs = CO.corpus(T)
        plan = ov.build(c, keep, [d[0] for d in docs],
                        lambda i: core.snapshot(c, docs[i][1]))
        print(plan.summary(), flush=True)
        # token features are live from level 0: their slots must not be handed over
        pinned = live_token_dims(c, sorted({t for toks, _m in docs for t in toks}))
        print(f"pinned token-feature dims: {len(pinned)}", flush=True)
    alloc = allocate(c, None, keep, inter, overlay=plan.dims if plan else None, overlay_cap=d0,
                     pinned=pinned)
    if alloc.over:  # every overlay slot is one of Qwen's dims
        assert len(alloc.final_over) <= d0, (len(alloc.final_over), d0)
    if hidden is None:
        hidden, m = pick_hidden(d0, 1 + alloc.n_slots)
    lay = Layout(hidden=hidden, off=d0, m=m, slots=alloc.slots, over=alloc.over)
    assert d0 + m + 1 + alloc.n_slots <= hidden, (alloc.n_slots, hidden)
    assert not set(alloc.over) & keep, "the lm-head's dims cannot be overlaid"
    lay.scale  # noqa: B018 - asserts m * B^2 / hidden is a power of 4
    n_chip = c.n_layers()
    n0 = tc0["num_hidden_layers"]
    n_kernel = 1 if kernel is not None else 0
    qwen_first = n_chip + n_kernel  # Qwen's own layers move up by the kernel's layer
    token_base = v0 + S.VOCAB  # the kernel's tokens sit in the vocabulary padding
    n_kernel_tokens = 0 if kernel is None else len(kernel.inputs)
    vocab = -(-(token_base + n_kernel_tokens) // 64) * 64
    print(f"circuit {c.stats()}, slots {alloc.n_slots}, hidden {hidden} (m={m}), "
          f"{n_chip} chip layers, columns {d0 + m + 1 + alloc.n_slots}, "
          f"kernel {'layer ' + str(n_chip) if kernel else 'none'}, vocab {vocab}", flush=True)

    tc = {k: v for k, v in tc0.items() if k != "full_attention_interval"}
    tc.update(hidden_size=hidden, num_hidden_layers=qwen_first + n0, mtp_num_hidden_layers=0,
              layer_types=["linear_attention"] * qwen_first + tc0["layer_types"],
              vocab_size=vocab, tie_word_embeddings=True,
              architectures=["Qwen3_5ForCausalLM"], model_type="qwen3_5_text")

    sinks = None
    if mute_qwen:
        from chipformer.q35.mute import fit_anti_sink

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        sinks = fit_anti_sink(src, dev)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Qwen's weights, widened and renamed one tensor at a time (keeps peak memory low)
    scale = (d0 / hidden) ** 0.5
    sd = {}
    base_emb = None
    for f in sorted(src.glob("*.safetensors")):
        with safe_open(f, "pt") as fh:
            for k in fh.keys():  # noqa: SIM118
                if not k.startswith(SRC_TEXT):
                    continue  # vision tower, mtp
                w = fh.get_tensor(k)
                if k == SRC_TEXT + "embed_tokens.weight":
                    base_emb = w
                    continue
                if IN_COLS.search(k):
                    w = torch.cat([w, w.new_zeros(*w.shape[:-1], hidden - d0)], -1)
                elif OUT_ROWS.search(k):
                    w = torch.cat([w, w.new_zeros(hidden - d0, *w.shape[1:])], 0)
                elif WIDE_NORMS.search(k):
                    g = (1.0 + w.float()) * scale - 1.0
                    w = torch.cat([g.to(w.dtype), w.new_zeros(hidden - d0)])
                k2 = DST + k[len(SRC_TEXT):]
                k2 = re.sub(r"^(model\.layers\.)(\d+)\.", lambda mt: f"{mt[1]}{int(mt[2]) + qwen_first}.",
                            k2)
                sd[k2] = w
    assert base_emb is not None
    if mute_qwen:
        from chipformer.q35.mute import mute

        mute(sd, tc, n0, lay.dim(next(iter(c.mu.terms))), hidden**0.5, prefix=DST, sinks=sinks,
             first=qwen_first)

    # the chip layers (fp32 while emitted, bf16 stored)
    layers = emit_layers(c, lay, tc, 0, DST, alloc, keep, chip=c.mu)
    for k in list(layers):
        sd[k] = layers.pop(k).to(torch.bfloat16)

    # the kernel's layer, right after the circuit's
    kern = None
    if kernel is not None:
        kern, klayer = kernel_layer(lay, alloc.n_slots, tc, n_chip, token_base, kernel)
        for k in list(klayer):
            sd[k] = klayer.pop(k).to(torch.bfloat16)

    # embeddings are built after the layer weights are on disk (see the save block below): the two
    # together are ~28 GiB, more than this machine has, and the fp32 chip-token matrix alone is 7.

    code = [next(iter(e.terms)) for e in c.code]
    tau = next(iter(c.tau.terms))
    norm = sd[DST + "norm.weight"].float()
    norm[d0:] = -1.0  # chip dims are invisible to the lm_head ...
    for d in code:  # ... except the output code and tau
        norm[lay.dim(d)] = code_gain(lay)
    norm[lay.dim(tau)] = TAU_GAIN
    check_bf16("final norm", norm[d0:])
    sd[DST + "norm.weight"] = norm.to(torch.bfloat16)

    dst.mkdir(parents=True, exist_ok=True)
    embed_key = DST + "embed_tokens.weight"
    sizes = {k: t.numel() * 2 for k, t in sd.items()}
    groups = shard_keys(sorted(sd), sizes)
    total_shards = len(groups) + 1  # the embedding is its own last shard
    index = {"metadata": {"total_size": 0}, "weight_map": {}}
    for i, keys in enumerate(groups):
        shard = {k: sd[k].contiguous() for k in keys}
        name = f"model-{i + 1:05d}-of-{total_shards:05d}.safetensors"
        save_file(shard, str(dst / name), metadata={"format": "pt"})
        index["weight_map"].update(dict.fromkeys(keys, name))
        index["metadata"]["total_size"] += sum(sizes[k] for k in keys)
        del shard
    del sd  # ~18 GiB back before the embedding matrix is built

    emb = torch.zeros(vocab, hidden, dtype=torch.bfloat16)
    emb[:v0] = text_rows(c, lay, T, base_emb, v0)
    del base_emb
    chip = embedding_matrix(c, lay, S.VOCAB)
    emb[v0:v0 + S.VOCAB] = chip[:S.VOCAB].to(torch.bfloat16)
    del chip
    if kern is not None:
        emb[token_base:token_base + kern.n_tokens] = kernel_rows(kern).to(torch.bfloat16)
    check_bf16("embedding", emb[: min(vocab, 300000), d0:].float())
    name = f"model-{total_shards:05d}-of-{total_shards:05d}.safetensors"
    save_file({embed_key: emb}, str(dst / name), metadata={"format": "pt"})
    index["weight_map"][embed_key] = name
    index["metadata"]["total_size"] += emb.numel() * 2
    del emb
    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    (dst / "config.json").write_text(json.dumps(tc, indent=2))
    extend_tokenizer(src, dst, v0)
    gen = {"eos_token_id": [248046, tc0.get("eos_token_id", 248044)],
           "pad_token_id": tc0.get("eos_token_id", 248044), "bad_words_ids": [[IM_START]],
           "do_sample": False}
    (dst / "generation_config.json").write_text(json.dumps(gen, indent=2))
    meta = {"layout": "rv1", "chip_token_offset": v0, "chip_vocab": S.VOCAB,
            "chip_layers": n_chip, "qwen_layers_from": qwen_first, "hidden": hidden,
            "chip_dims_offset": d0, "ballast_dims": m, "circuit": c.stats(),
            "slots": alloc.n_slots, "muted": mute_qwen,
            "code_dims": [lay.dim(d) for d in code], "tau_dim": lay.dim(tau),
            "mu_dim": lay.dim(next(iter(c.mu.terms))), "nfam": core.NFAM, "npay": core.NPAY,
            "tool_call": [T.TCO, T.TCC], "printable": T.printable, "vx": vx,
            "overlay": {"dims": len(plan.dims), "slots": len(alloc.final_over)}
            if plan else None,
            "kernel": None if kern is None else {
                "layer": n_chip, "layers_total": tc["num_hidden_layers"],
                "token_base": token_base, "tokens": kern.n_tokens,
                "base": kern.slot.base, "end": kern.slot.end, "width": kern.slot.width,
                "lanes": kern.slot.lanes, "scratch": kern.slot.temp,
                "shared": kern.slot.shared, "units": kern.unit_count(),
                "input_columns": [kern.slot.base, kern.slot.lanes],
                "output_columns": [kern.slot.out_base, kern.slot.lanes],
                "gate_column": kern.layout.dim(next(iter(kern.is_kernel.terms)))}}
    (dst / "chipformer.json").write_text(json.dumps(meta, indent=1))
    return meta


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Merge the RV32 machine into Qwen3.5 (layout rv1)")
    ap.add_argument("src")
    ap.add_argument("--out", default="dist/q4-rv1")
    ap.add_argument("--no-mute", action="store_true")
    ap.add_argument("--hidden", type=int)
    ap.add_argument("--ballast", type=int)
    ap.add_argument("--vx", action="store_true", help="with the VX vector unit (QwenPU)")
    ap.add_argument("--overlay", action="store_true",
                    help="put circuit dims that only chip tokens need into Qwen's dims")
    ap.add_argument("--kernel", type=Path,
                    help=".npz holding a compiled matvec ('W': lanes x lanes, 'X': n_tokens x "
                         "lanes) to emit as the layer right after the circuit's")
    args = ap.parse_args()
    kernel = None
    if args.kernel is not None:
        z = np.load(args.kernel)
        kernel = KernelPlan(z["W"], z["X"])
    meta = build_merged(args.src, args.out, not args.no_mute, args.hidden, args.ballast, args.vx,
                        args.overlay, kernel)
    keys = ("hidden", "chip_layers", "slots", "circuit")
    print(json.dumps({**{k: meta[k] for k in keys}, "kernel": meta["kernel"] and {
        "layer": meta["kernel"]["layer"], "base": meta["kernel"]["base"],
        "width": meta["kernel"]["width"]}}))


if __name__ == "__main__":
    main()
