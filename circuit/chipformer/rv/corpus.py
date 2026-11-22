"""Mixed token streams for measuring the overlay (rv.corpus -> q35.overlay.activity).

Each document is one session with the RV32 machine: Qwen text, a tool call whose text the bridge
captures, the run that follows it (the machine's records and the echo of every putchar), and more
text after it. What the circuit does on text tokens depends on the state they start from, so the
documents put text after plain text, after a captured call, after a run of each record family
(plain RV32IM, the digit file of calc, the vector unit, a fault) and in the middle of a run.

The corpus is a measurement of the circuit's text behavior, not a proof: a session that looks
nothing like these (a record family the corpus never runs, a longer or stranger text) can still
show an overlaid dim carrying something that matters. ``overlay.check_noisy`` on a wider set of
traces is how far that can be pushed.
"""

import copy
import random

from chipformer.rv import bridge as BR
from chipformer.rv import fuzz, ref
from chipformer.rv.asm import assemble

TEXTS = [
    "Hello, world! The answer is (roughly) 42.\n",
    " Sure. gcd(1071, 462) = 21, and 2**10 = 1024.\n",
    "naïve café — 東京 🙂 1234567890\n",
    "  indented\n\tcode()  # comment\n",
    "The quick brown fox jumps over the lazy dog.\n",
    "\n\n\n",
    "x = (a + b) * c / d % e\n",
    "0123456789 abcdef ABCDEF 0x1F\n",
]
# short expressions: a calc run is long (the digit file works a digit at a time) and the corpus
# is evaluated in full, so one run of it is enough to see the record family
EXPRS = ["1+2", "12*34", "999-1000", "8//3", "17%5"]
EXIT = "\nli a7, 93\necall"
FAULT = "\n.word 0x00000001"  # after the parking exit: the GO runs an illegal word
# one instruction of every record family: M (the 3-token value records), DIV (QS/QD), the Zba
# ops, every load and store width (ST0/ST1 for SW, V/F1 keep records for SB/SH), a jump, and a
# 63-digit digit-file row (which reaches N_MAX and so an R refresh)
COVER = (
    """
addi s3, s3, 1
"""
    + "addi s3, s3, 1\n" * 68
    + """
li a0, 0x7FFFFFFF
li a1, 0x80000000
add t0, a0, a1
mul t1, a0, a0
div t2, a0, a1
divu t3, a0, a1
rem t4, a0, a1
remu t5, a0, a1
sh1add t0, a0, a1
sh2add t1, a0, a1
sh3add t2, a0, a1
la s1, buf
sw a0, 0(s1)
sh a0, 4(s1)
sb a0, 6(s1)
sw a0, 64(s1)
sh a0, 66(s1)
sb a0, 68(s1)
lw s2, 0(s1)
lh s3, 4(s1)
lhu s4, 4(s1)
lb s5, 6(s1)
lbu s6, 6(s1)
slli t0, a0, 31
srai t1, a0, 1
srli t2, a0, 1
slt t3, a0, a1
sltu t4, a0, a1
xor t5, a0, a1
andi t6, a0, 0x7FF
ori s7, a0, 0x123
lui s8, 0xABCDE
auipc s9, 1
li t0, 1056712
li t1, 0
li t2, 0
bmac t1, t2, t0
beq a0, a0, ok
li s10, 1
ok: li s10, 2
li a0, 65
li a7, 1
ecall
li a7, 93
ecall
.data
buf: .space 128
"""
)


def _go_program(work: str) -> str:
    """A program that parks at its first instruction, so the GO after ``</tool_call>`` runs
    ``work`` (which ends with an exit): what a session with the machine looks like."""
    return "li a7, 93\necall\n" + work


def _vx_program(seed: int) -> str:
    """A short vector unit program (source): VPUT a register, VQ it out, VGET the results."""
    rng = random.Random(seed)
    lines = []
    for _ in range(3):
        reg, pair = rng.randrange(16), rng.randrange(32)
        lines += [
            f"li t0, {reg * 32 + pair}",
            f"li t1, {rng.randint(-30000, 30000)}",
            f"li t2, {rng.randint(-30000, 30000)}",
            "vput t0, t1, t2",
        ]
    cfg = 0 | 0 << 4 | 64 << 10 | 0 << 17 | 8 << 19
    lines += [
        f"li t0, {cfg}",
        "li t1, 0",
        "li t2, 0",
        "vq t0, t1, t2",
        "li t0, 64",
        "vred t0, zero, zero",
        "li t1, 1",
        "vget s1, t1, zero",
    ]
    return "\n".join(lines) + EXIT


def _tool_call(expr: str) -> str:
    return (
        "<tool_call>\n<function=calc>\n<parameter=expression>\n"
        f"{expr}\n</parameter>\n</function>\n</tool_call>"
    )


def _doc(T: BR.Text, prog, blocks, park: bool = True) -> tuple[list[int], ref.Machine]:
    """(tokens, the machine the stream starts from).

    ``blocks``: ("text", str) writes Qwen text, ("call", str) a tool call (a run follows).
    ``park`` runs the machine to its first exit first (a firmware the host injects parked, like
    calc); otherwise the first GO runs the program from its first instruction (a whole program as
    one run). The returned machine is the state before the stream: ``core.snapshot`` it into the
    graph, or the circuit reads an empty ROM."""
    m = ref.Machine.from_program(prog)
    if park:
        BR.park(m)
    init = copy.deepcopy(m)  # the state the stream starts from (the Session runs the machine on)
    ses = BR.Session(T, m)
    for kind, body in blocks:
        ses.qwen(T.encode(_tool_call(body) if kind == "call" else body))
    return ses.toks, init


def corpus(
    text: BR.Text | None = None, seed: int = 7, docs: int = 8
) -> list[tuple[list[int], ref.Machine]]:
    """(tokens, starting machine) of mixed sessions: Qwen text, tool calls, runs and echoes.

    The machine is what ``core.snapshot(graph, machine)`` needs: a run's ROM, RAM, registers and
    counters live there, not in the token stream."""
    T = text if text is not None else BR.Text(BR.default_tokenizer())
    rng = random.Random(seed)
    calc = "tests/data/rv/calc-bridge.elf"
    from chipformer.rv import elf

    # ("two": a second capture after a run needs a firmware that parks again; the calc one is
    # long, so the corpus keeps one capture per document and two documents of the same family)
    kinds = ["text", "calc", "cover", "fuzz", "vx", "fault", "fuzz", "text"]
    out = []
    for k in range(docs):
        kind = kinds[k % len(kinds)]
        pre = rng.choice(TEXTS)
        if kind == "text":
            blocks = [("text", pre + rng.choice(TEXTS))]
            prog = elf.program(calc)
        elif kind == "calc":
            blocks = [
                ("text", pre[:30]),
                ("call", rng.choice(EXPRS)),
                ("text", rng.choice(TEXTS)[:30]),
            ]
            prog = elf.program(calc)
        elif kind == "cover":
            blocks = [("text", pre[:30]), ("call", "1+2"), ("text", rng.choice(TEXTS)[:30])]
            prog = assemble(COVER)
        elif kind == "fuzz":
            blocks = [
                ("text", pre[:30]),
                ("call", rng.choice(EXPRS)),
                ("text", rng.choice(TEXTS)[:30]),
            ]
            prog = assemble(fuzz.gen_program(rng.randrange(1 << 30), block=6, iters=1))
        elif kind == "vx":
            blocks = [("text", pre[:30]), ("call", "1+2"), ("text", rng.choice(TEXTS)[:30])]
            prog = assemble(_vx_program(rng.randrange(1 << 30)))
        elif kind == "two":  # two runs back to back (a second capture after a run)
            blocks = [
                ("text", pre[:20]),
                ("call", "1+2"),
                ("call", "1+2"),
                ("text", rng.choice(TEXTS)[:20]),
            ]
            prog = assemble(fuzz.gen_program(rng.randrange(1 << 30), block=4, iters=1))
        else:  # fault
            blocks = [("text", pre[:30]), ("call", "1+2"), ("text", rng.choice(TEXTS)[:30])]
            prog = assemble(_go_program(FAULT))
        out.append(_doc(T, prog, blocks, park=kind in ("calc", "fault")))
    return out
