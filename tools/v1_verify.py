#!/usr/bin/env python3
"""V1 mixed-quantization checks: GGUF structure, byte-level tensor diffs, and HF fidelity.

Every verdict printed here comes from reading the files involved; nothing is inferred from the
conversion log.  The three checks that matter for V1:

  meta          structure: block count, recurrent-layer mask, tokenizer size, per-tensor types
  cmp-gguf      byte-identical tensors between two GGUFs (quantized output vs. the bf16 source)
  hf-fidelity   GGUF payload vs. the original safetensors checkpoint

`hf-fidelity` re-derives the expected GGUF payload from the safetensors checkpoint with an
independent implementation of the Qwen3.5 projection transforms:

  - linear-attention V heads: grouped-by-K order -> tiled order (row and column variants)
  - in_proj_qkv + in_proj_z  -> attn_qkv_gate,  in_proj_b + in_proj_a -> ssm_beta_alpha
  - A_log -> -exp(A_log),  dt_bias -> ssm_dt.bias
  - every ``*norm.weight`` except ``linear_attn.norm`` gets the +1 offset the converter applies
  - dtype: 1D/norm/conv tensors land as F32, dense matrices as BF16 (rounded from the f32 path)

Import order: ortools must precede torch on this machine (torch 2.13 + ortools 9 native ABI
clash; see e1-direction-b-assessment.md:358).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from ortools.sat.python import cp_model  # noqa: F401  (must precede torch on this box)

import numpy as np
import gguf

# GGUF dtype for each tensor family, mirrored from the converter's output (see convert log):
#   1D tensors, *_norm.weight, ssm_conv1d            -> F32
#   everything else (dense matrices, token_embd)     -> BF16
def expected_dtype(gguf_name: str, n_dims: int) -> str:
    if n_dims <= 1 or gguf_name.endswith("_norm.weight") or "conv1d" in gguf_name:
        return "F32"
    return "BF16"


# ---------------------------------------------------------------------------------------------
# GGUF helpers
# ---------------------------------------------------------------------------------------------

class GGUFFile:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.reader = gguf.GGUFReader(str(path))
        self.by_name = {t.name: t for t in self.reader.tensors}

    def raw_bytes(self, name: str) -> np.ndarray:
        return np.asarray(self.by_name[name].data).reshape(-1).view(np.uint8)

    def scalar(self, key: str):
        field = self.reader.fields.get(key)
        if field is None:
            return None
        if field.types[0] == gguf.GGUFValueType.STRING:
            return bytes(field.parts[-1]).decode("utf-8", "replace")
        if field.types[0] == gguf.GGUFValueType.ARRAY:
            # parts: [key_len, key_bytes, ARRAY, value_type, count, value...]
            return field.parts[-1] if len(field.parts) == 6 else np.concatenate(
                [np.asarray(p).ravel() for p in field.parts[5:]]).tolist()
        return np.asarray(field.parts[-1]).ravel()[0].item()

    def array_count(self, key: str) -> int | None:
        field = self.reader.fields.get(key)
        if field is None or field.types[0] != gguf.GGUFValueType.ARRAY:
            return None
        return int(np.asarray(field.parts[4]).ravel()[0])


def tensor_type_names(g: GGUFFile) -> dict[str, str]:
    return {name: t.tensor_type.name for name, t in g.by_name.items()}


# ---------------------------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------------------------

def cmd_meta(args) -> int:
    g = GGUFFile(Path(args.gguf))
    keys = [
        "general.architecture", "general.file_type", "general.quantization_version",
        "qwen35.block_count", "qwen35.embedding_length", "qwen35.feed_forward_length",
        "qwen35.context_length", "qwen35.attention.head_count", "qwen35.attention.head_count_kv",
        "qwen35.attention.key_length", "qwen35.attention.value_length",
        "qwen35.full_attention_interval", "qwen35.rope.dimension_sections", "qwen35.rope.freq_base",
        "qwen35.ssm.conv_kernel", "qwen35.ssm.state_size", "qwen35.ssm.group_count",
        "qwen35.ssm.time_step_rank", "qwen35.ssm.inner_size",
        "tokenizer.ggml.model", "tokenizer.ggml.pre",
    ]
    meta = {k: g.scalar(k) for k in keys}

    meta["tokenizer.ggml.n_tokens"] = g.array_count("tokenizer.ggml.tokens")
    rec = g.scalar("qwen35.attention.recurrent_layers")
    if rec is not None:
        rec = [int(v) for v in rec]
        meta["recurrent_layers_true_ids"] = [i for i, v in enumerate(rec) if v != 0]
        meta["n_recurrent_layers"] = sum(1 for v in rec if v != 0)

    types = tensor_type_names(g)
    by_type: dict[str, int] = {}
    for t in types.values():
        by_type[t] = by_type.get(t, 0) + 1

    report = {
        "file": str(g.path),
        "size_bytes": g.path.stat().st_size,
        "n_tensors": len(g.reader.tensors),
        "tensor_types": by_type,
        "meta": meta,
        "global_tensors": {n: {"type": types[n], "shape": [int(d) for d in g.by_name[n].shape]}
                           for n in types if not n.startswith("blk.")},
        "chip_layer_tensor_types": {
            f"blk.{i}": {n: types[n] for n in sorted(types) if n.startswith(f"blk.{i}.")}
            for i in (0, 12)
        },
        "last_layer_tensor_types": {
            f"blk.{i}": {n: types[n] for n in sorted(types) if n.startswith(f"blk.{i}.")}
            for i in (44,)
        },
    }
    print(json.dumps(report, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


def _layer_of(name: str) -> int | None:
    m = re.match(r"blk\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def cmd_cmp_gguf(args) -> int:
    a = GGUFFile(Path(args.a))
    b = GGUFFile(Path(args.b))

    check_layers: set[int] | None = None
    if args.layers:
        check_layers = set()
        for part in args.layers.split(","):
            if "-" in part:
                lo, hi = part.split("-")
                check_layers.update(range(int(lo), int(hi) + 1))
            else:
                check_layers.add(int(part))

    def selected(name: str) -> bool:
        if check_layers is None:
            return True
        layer = _layer_of(name)
        # global tensors (token_embd, output_norm) are always part of the comparison
        return layer is None or layer in check_layers

    names_a = {n for n in a.by_name if selected(n)}
    names_b = {n for n in b.by_name if selected(n)}
    only_a = sorted(names_a - names_b)
    only_b = sorted(names_b - names_a)

    mismatched_type: list[str] = []
    mismatched_shape: list[str] = []
    mismatched_data: list[str] = []
    identical: list[str] = []
    total_bytes = 0

    for name in sorted(names_a & names_b):
        ta, tb = a.by_name[name], b.by_name[name]
        if ta.tensor_type != tb.tensor_type:
            mismatched_type.append(f"{name}: {ta.tensor_type.name} vs {tb.tensor_type.name}")
            continue
        if [int(x) for x in ta.shape] != [int(x) for x in tb.shape]:
            mismatched_shape.append(f"{name}: {list(ta.shape)} vs {list(tb.shape)}")
            continue
        ba, bb = a.raw_bytes(name), b.raw_bytes(name)
        total_bytes += int(ba.nbytes)
        if ba.nbytes != bb.nbytes:
            mismatched_shape.append(f"{name}: {ba.nbytes} vs {bb.nbytes} bytes")
        elif np.array_equal(ba, bb):
            identical.append(name)
        else:
            mismatched_data.append(name)
    print("=== gguf/gguf tensor comparison ===")
    print(f"a: {a.path}  ({len(a.by_name)} tensors)")
    print(f"b: {b.path}  ({len(b.by_name)} tensors)")
    print(f"compared: {len(names_a & names_b)} tensors, {total_bytes/2**20:.2f} MiB payload")
    for label, items in (("identical", identical), ("type mismatch", mismatched_type),
                         ("shape mismatch", mismatched_shape), ("data mismatch", mismatched_data),
                         ("only in a", only_a), ("only in b", only_b)):
        print(f"{label:14s}: {len(items)}")
        for line in items[:40]:
            print(f"  {line}")
        if len(items) > 40:
            print(f"  ... {len(items) - 40} more")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "identical": identical, "type_mismatch": mismatched_type,
            "shape_mismatch": mismatched_shape, "data_mismatch": mismatched_data,
            "only_a": only_a, "only_b": only_b,
        }, indent=2), encoding="utf-8")

    ok = not (mismatched_type or mismatched_shape or mismatched_data or only_a or only_b)
    print("RESULT:", "PASS - byte-identical" if ok else "FAIL")
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------------
# HF fidelity
# ---------------------------------------------------------------------------------------------

def _bf16_bytes(x: np.ndarray) -> np.ndarray:
    """f32 -> the exact bf16 byte payload gguf would store (2 bytes per element)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    quiet = np.where((u & 0x7FFFFFFF) > 0x7F800000, (u & np.uint64(0xFFFF0000)) | np.uint64(64 << 16), u)
    r = ((quiet + 0x7FFF + ((quiet >> 16) & 1)) >> 16).astype(np.uint16)
    return r.reshape(-1).view(np.uint8)


def _reorder_v_heads(x: np.ndarray, dim: int, n_k: int, n_v_per_k: int, head_dim: int) -> np.ndarray:
    """Grouped-by-K -> tiled head order; mirrors conversion/qwen.py:_reorder_v_heads."""
    idx = np.arange(n_k * n_v_per_k * head_dim).reshape(n_k, n_v_per_k, head_dim)
    perm = np.swapaxes(idx, 0, 1).reshape(-1)
    if dim == 0:
        return np.take(x, perm, axis=0)
    if dim == 1:
        return np.take(x, perm, axis=1)
    raise NotImplementedError(f"dim={dim}")


class HFModel:
    """Minimal reader over the checkpoint: config + per-tensor lazy access."""

    def __init__(self, hf_dir: Path) -> None:
        from safetensors import safe_open

        self.dir = hf_dir
        self.config = json.loads((hf_dir / "config.json").read_text(encoding="utf-8"))
        self.index = json.loads((hf_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
        self.weight_map: dict[str, str] = self.index["weight_map"]
        self._safe_open = safe_open

        self.n_k = self.config["linear_num_key_heads"]
        self.n_v = self.config["linear_num_value_heads"]
        self.n_v_per_k = self.n_v // self.n_k
        self.head_k = self.config["linear_key_head_dim"]
        self.head_v = self.config["linear_value_head_dim"]
        self.layer_types = self.config["layer_types"]
        self.n_layers = self.config["num_hidden_layers"]

    def names(self) -> list[str]:
        return sorted(self.weight_map)

    def get_f32(self, name: str) -> np.ndarray:
        """Read one safetensors tensor as float32 (bf16 -> f32 is exact)."""
        shard = self.weight_map[name]
        with self._safe_open(str(self.dir / shard), framework="pt") as f:
            t = f.get_tensor(name)
        return t.float().numpy()

    def _is_linear(self, layer: int) -> bool:
        return self.layer_types[layer] == "linear_attention"

    def _merge_partner(self, hf_name: str) -> str | None:
        """The other half of a tensor the converter fuses (qkv<->z, beta<->alpha)."""
        if hf_name.endswith(".linear_attn.in_proj_qkv.weight"):
            return hf_name.removesuffix("in_proj_qkv.weight") + "in_proj_z.weight"
        if hf_name.endswith(".linear_attn.in_proj_z.weight"):
            return hf_name.removesuffix("in_proj_z.weight") + "in_proj_qkv.weight"
        if hf_name.endswith(".linear_attn.in_proj_a.weight"):
            return hf_name.removesuffix("in_proj_a.weight") + "in_proj_b.weight"
        if hf_name.endswith(".linear_attn.in_proj_b.weight"):
            return hf_name.removesuffix("in_proj_b.weight") + "in_proj_a.weight"
        return None

    def expected(self, hf_name: str) -> tuple[str, np.ndarray, str, bool] | None:
        """Return (gguf_name, expected f32 payload, dtype tag, is_primary).

        is_primary is False for the half of a fused pair that is consumed by its partner, so the
        caller can skip it.  The payload is laid out C-contiguous the way GGUF stores it.
        """
        if hf_name == "model.embed_tokens.weight":
            return ("token_embd.weight", self.get_f32(hf_name), "BF16", True)
        if hf_name == "model.norm.weight":
            return ("output_norm.weight", self.get_f32(hf_name) + 1.0, "F32", True)

        m = re.match(r"model\.layers\.(\d+)\.(.+)$", hf_name)
        if not m:
            return None
        layer, suffix = int(m.group(1)), m.group(2)
        prefix = f"blk.{layer}."

        # ---- linear-attention layers (chip layers and Qwen's linear layers) ----
        if suffix == "linear_attn.in_proj_qkv.weight":
            w = self.get_f32(hf_name)
            z = self.get_f32(self._merge_partner(hf_name))
            q_dim = self.head_k * self.n_k
            q, k = w[:q_dim], w[q_dim:2 * q_dim]
            v = _reorder_v_heads(w[2 * q_dim:], 0, self.n_k, self.n_v_per_k, self.head_v)
            z = _reorder_v_heads(z, 0, self.n_k, self.n_v_per_k, self.head_v)
            return (prefix + "attn_qkv_gate.weight", np.concatenate([q, k, v, z], axis=0), "BF16", True)
        if suffix == "linear_attn.in_proj_z.weight":
            return (prefix + "attn_qkv_gate.weight", None, "BF16", False)  # consumed by qkv
        if suffix == "linear_attn.in_proj_b.weight":
            return (prefix + "ssm_beta_alpha.weight", None, "BF16", False)  # consumed by alpha
        if suffix == "linear_attn.in_proj_a.weight":
            a = _reorder_v_heads(self.get_f32(hf_name), 0, self.n_k, self.n_v_per_k, 1)
            b = _reorder_v_heads(self.get_f32(self._merge_partner(hf_name)), 0, self.n_k, self.n_v_per_k, 1)
            return (prefix + "ssm_beta_alpha.weight", np.concatenate([b, a], axis=0), "BF16", True)
        if suffix == "linear_attn.out_proj.weight":
            return (prefix + "ssm_out.weight",
                    _reorder_v_heads(self.get_f32(hf_name), 1, self.n_k, self.n_v_per_k, self.head_v),
                    "BF16", True)
        if suffix == "linear_attn.A_log":
            a = _reorder_v_heads(self.get_f32(hf_name).reshape(-1, 1), 0, self.n_k, self.n_v_per_k, 1)
            return (prefix + "ssm_a", -np.exp(a.reshape(-1)), "F32", True)
        if suffix == "linear_attn.dt_bias":
            d = _reorder_v_heads(self.get_f32(hf_name).reshape(-1, 1), 0, self.n_k, self.n_v_per_k, 1)
            return (prefix + "ssm_dt.bias", d.reshape(-1), "F32", True)
        if suffix == "linear_attn.conv1d.weight":
            conv = self.get_f32(hf_name).squeeze()          # [channels, kernel]
            qk = self.head_k * self.n_k * 2
            v = _reorder_v_heads(conv[qk:], 0, self.n_k, self.n_v_per_k, self.head_v)
            return (prefix + "ssm_conv1d.weight", np.concatenate([conv[:qk], v], axis=0), "F32", True)
        if suffix == "linear_attn.norm.weight":
            return (prefix + "ssm_norm.weight", self.get_f32(hf_name), "F32", True)

        # ---- tensors shared by linear and full-attention layers ----
        plain = {
            "input_layernorm.weight": "attn_norm.weight",
            "post_attention_layernorm.weight": "post_attention_norm.weight",
            "pre_feedforward_layernorm.weight": "ffn_norm.weight",
            "self_attn.q_norm.weight": "attn_q_norm.weight",
            "self_attn.k_norm.weight": "attn_k_norm.weight",
            "mlp.gate_proj.weight": "ffn_gate.weight",
            "mlp.up_proj.weight": "ffn_up.weight",
            "mlp.down_proj.weight": "ffn_down.weight",
        }
        if suffix in plain:
            value = self.get_f32(hf_name)
            if suffix.endswith("norm.weight"):
                value = value + 1.0
            return (prefix + plain[suffix], value, None, True)

        full = {
            "self_attn.q_proj.weight": "attn_q.weight",
            "self_attn.k_proj.weight": "attn_k.weight",
            "self_attn.v_proj.weight": "attn_v.weight",
            "self_attn.o_proj.weight": "attn_output.weight",
        }
        if suffix in full:
            return (prefix + full[suffix], self.get_f32(hf_name), "BF16", True)

        raise KeyError(f"unhandled hf tensor: {hf_name}")


def _payload_bytes(expected_f32: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "F32":
        return np.ascontiguousarray(expected_f32, dtype=np.float32).view(np.uint8).reshape(-1)
    if dtype == "BF16":
        return _bf16_bytes(expected_f32)
    raise ValueError(dtype)


def cmd_hf_fidelity(args) -> int:
    hf = HFModel(Path(args.hf_dir))
    g = GGUFFile(Path(args.gguf))

    layers: set[int] | None = None
    if args.layers:
        layers = set()
        for part in args.layers.split(","):
            if "-" in part:
                lo, hi = part.split("-")
                layers.update(range(int(lo), int(hi) + 1))
            else:
                layers.add(int(part))

    checked = 0
    mismatched: list[str] = []
    missing: list[str] = []
    unhandled: list[str] = []
    token_embd = None

    for hf_name in hf.names():
        if args.token_rows:
            if hf_name != "model.embed_tokens.weight":
                continue
        elif hf_name == "model.embed_tokens.weight":
            continue
        m = re.match(r"model\.layers\.(\d+)\.", hf_name)
        if m and layers is not None and int(m.group(1)) not in layers:
            continue

        try:
            spec = hf.expected(hf_name)
        except KeyError as exc:
            unhandled.append(str(exc))
            continue
        if spec is None or not spec[3]:
            continue
        gguf_name, expected_f32, dtype, _ = spec

        if hf_name == "model.embed_tokens.weight":
            # compare in row blocks so the 6.6 GiB embedding table never has to sit in RAM twice
            gt = g.by_name.get(gguf_name)
            if gt is None:
                missing.append(f"{hf_name}: {gguf_name}")
                continue
            if gt.tensor_type.name != "BF16":
                mismatched.append(f"{gguf_name}: type {gt.tensor_type.name}, expected BF16")
                continue
            rows, cols = expected_f32.shape
            row_bytes = cols * 2  # bf16
            flat = np.asarray(gt.data).reshape(-1).view(np.uint8)
            chunk = 16384
            bad_rows = []
            for start in range(0, rows, chunk):
                stop = min(start + chunk, rows)
                want = _payload_bytes(expected_f32[start:stop], "BF16").reshape(-1, row_bytes)
                got = np.asarray(flat[start * row_bytes:stop * row_bytes]).reshape(-1, row_bytes)
                if not np.array_equal(want, got):
                    bad = np.nonzero((want != got).any(axis=1))[0]
                    bad_rows.extend(int(start + r) for r in bad)
            token_embd = {
                "shape": [rows, cols],
                "equal": not bad_rows,
                "bad_rows": bad_rows[:40],
                "n_bad_rows": len(bad_rows),
            }
            continue

        gt = g.by_name.get(gguf_name)
        if gt is None:
            missing.append(f"{hf_name}: {gguf_name}")
            continue
        want_type = dtype or expected_dtype(gguf_name, expected_f32.ndim)
        if gt.tensor_type.name != want_type:
            mismatched.append(f"{gguf_name}: type {gt.tensor_type.name}, expected {want_type}")
            continue
        want = _payload_bytes(expected_f32, want_type)
        got = g.raw_bytes(gguf_name)
        if want.nbytes != got.nbytes:
            mismatched.append(f"{gguf_name}: {want.nbytes} bytes expected, {got.nbytes} stored")
        elif np.array_equal(want, got):
            checked += 1
        else:
            n_diff = int(np.count_nonzero(want != got))
            mismatched.append(f"{gguf_name}: {n_diff} differing bytes")

    print("=== hf -> gguf fidelity ===")
    print(f"hf   : {hf.dir}")
    print(f"gguf : {g.path}")
    print(f"tensors verified byte-equal : {checked}")
    print(f"tensors mismatched          : {len(mismatched)}")
    for line in mismatched[:40]:
        print(f"  {line}")
    if missing:
        print(f"missing in gguf             : {len(missing)}")
        for line in missing[:20]:
            print(f"  {line}")
    if unhandled:
        print(f"unhandled hf tensors        : {len(unhandled)}")
        for line in unhandled[:20]:
            print(f"  {line}")
    if token_embd and token_embd["equal"]:
        print(f"token_embd                  : all {token_embd['shape'][0]} rows equal (BF16 bytes)")
    elif token_embd:
        print(f"token_embd                  : {token_embd['n_bad_rows']} bad rows, first {token_embd['bad_rows'][:10]}")

    ok = not (mismatched or missing or unhandled) and (token_embd is None or token_embd["equal"])
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("meta", help="print gguf structure and key metadata")
    s.add_argument("gguf")
    s.add_argument("--json")
    s.set_defaults(func=cmd_meta)

    s = sub.add_parser("cmp-gguf", help="byte-compare two gguf files, optionally limited to layers")
    s.add_argument("a")
    s.add_argument("b")
    s.add_argument("--layers", help="layer ids / ranges, e.g. 0-12 or 0,5,7")
    s.add_argument("--json")
    s.set_defaults(func=cmd_cmp_gguf)

    s = sub.add_parser("hf-fidelity", help="rebuild each gguf payload from safetensors and compare bytes")
    s.add_argument("hf_dir")
    s.add_argument("gguf")
    s.add_argument("--layers", help="layer ids / ranges, e.g. 0-12; default: all")
    s.add_argument("--token-rows", action="store_true", help="only check the token embedding table")
    s.set_defaults(func=cmd_hf_fidelity)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
