#!/usr/bin/env python3
"""V1 feasibility probe: HF checkpoint -> GGUF (bf16), with this box's import-order workaround.

On this machine `import torch` before `import ortools` segfaults (torch 2.13 + ortools 9 native
ABI clash; see e1-direction-b-assessment.md:358). The converter imports torch at module top, so
ortools is preloaded here before runpy executes the real script.

Usage:
    python v1_convert.py <hf_dir> <out_gguf> [extra convert_hf_to_gguf.py args...]
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

from ortools.sat.python import cp_model  # noqa: F401  (must precede torch on this box)

CONVERTER = Path("H:/llamacpp-c/convert_hf_to_gguf.py")
# runpy does not add the script's directory to sys.path; the converter imports its sibling
# `conversion` package, so make it importable first.
sys.path.insert(0, str(CONVERTER.parent))


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    hf_dir, out_gguf, *extra = sys.argv[1:]
    sys.argv = [str(CONVERTER), hf_dir, "--outtype", "bf16", "--outfile", out_gguf, *extra]
    runpy.run_path(str(CONVERTER), run_name="__main__")


if __name__ == "__main__":
    main()
