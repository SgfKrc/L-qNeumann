"""Batched rv1 decoding: N conversations at once on one GPU.

    rb = RvBatch(chat, batch=8, max_len=8192)          # chat: RvChat(path, executor=True)
    rb.run(jobs, budget, max_chip, done)                 # jobs: iterable of Job; done(job, result)

Every conversation (a row) keeps its own circuit state -- its residual, its head states and
lag windows -- inside one batch-B fusedexec.FreeRun3 shared by all rows, so a turn's circuit
steps for every live row go out as one replay instead of B batch-1 replays. Its text tokens
go through Qwen's layers in one batched step for all live rows, on a cache with per-row
positions (q35.batch._Rows: full attention writes row b's key at its own slot and the mask
admits slots <= pos[b]; DeltaNet rows are independent). Qwen's prompt prefill is alone (batch
1, as generate does) and copied into its row; the circuit's prefill steps are the batch's,
with the rows still drawing their prompt parked through the executor's activity mask. A
finished row is refilled from the queue.

Each row is a generator (``_row``) with RvChat.generate's executor-path logic, that asks the
run loop for the two things it cannot do alone: a circuit step of its embedding (``("emb",
row)``, answered with the kept dims) and Qwen's argmaxes (``("qwen", hc, pos)``). The loop
steps every row that has an embedding pending in one replay, parked rows keep their state
(the executor's activity mask), and only when every live row waits on Qwen does it take one
batched Qwen step -- so a row's circuit run never has to match another's length. Only Qwen's
GEMMs see the batch (M = live rows instead of 1), which can change their bf16 rounding; the
circuit is exact either way.
"""

import time
from dataclasses import dataclass, field

import torch

from chipformer.rv import bridge as BR
from chipformer.rv import core
from chipformer.rv import spec as S


@dataclass
class Job:
    key: object
    ids: list[int]
    machine: object  # a parked ref.Machine
    lock: object = None  # a BR.Session on a copy (history = the prompt), or None
    extra: dict = field(default_factory=dict)


class RvBatch:
    def __init__(self, chat, batch: int = 8, max_len: int = 8192) -> None:
        from chipformer.rv.chat import ExecRow
        from chipformer.rv.fusedexec import FreeRun3

        assert chat.executor and not chat.verify
        self.chat, self.B, self.max_len = chat, batch, max_len
        self.dev = chat.dev
        # one executor for every row: each keeps its own conversation in its own row. The
        # chat's own row becomes row 0 of it, so its embedding table is the batch's (a pure
        # function) and nothing builds a second batch-1 executor's head states.
        ex = self.ex = FreeRun3(chat.c, batch, chat.dev, keep=chat.c.keep)
        chat.erow = ExecRow(chat, row=0, batch=batch, ex=ex)
        self.rows = [chat.erow] + [ExecRow(chat, row=b, batch=batch, ex=ex)
                                   for b in range(1, batch)]
        self.pos = torch.zeros(batch, dtype=torch.long, device=self.dev)
        self.slots = torch.arange(max_len, device=self.dev)
        self.cache = None  # _Rows, built from the first prefill's cache (its shapes)
        cfg = chat.model.config
        self.kinds = list(cfg.layer_types)

    # --- Qwen's batched step -----------------------------------------------------------------
    @torch.no_grad()
    def _qwen(self, H: torch.Tensor):
        """Qwen's layers on one text token per row (residuals H [B, hidden] after the chip
        layers, RoPE positions / KV slots self.pos); returns the argmaxes (all, Qwen's)."""
        chat = self.chat
        h = H[:, None, :]
        p4 = self.pos.view(1, -1, 1).expand(4, -1, 1)
        pe = chat.lm.rotary_emb(h, p4[1:])
        mask = (self.slots[None, :] <= self.pos[:, None])[:, None, None, :]
        for i in range(chat.n_chip, len(chat.lm.layers)):
            full = self.kinds[i] == "full_attention"
            h = chat.lm.layers[i](h, position_embeddings=pe,
                                  attention_mask=mask if full else None,
                                  position_ids=p4[0], past_key_values=self.cache.cache)
        logits = chat.model.lm_head(chat.lm.norm(h))[:, -1]
        both = torch.stack([logits.argmax(-1), logits[:, : chat.v0].argmax(-1)], 1)
        return both.tolist()

    @torch.no_grad()
    def _prefill(self, b: int, ids: list[int]) -> int:
        from chipformer.q35.batch import _Rows

        chat = self.chat
        out = chat.model(input_ids=torch.tensor([ids], device=self.dev), use_cache=True)
        small = out.past_key_values
        if self.cache is None:
            self.cache = _Rows(chat.model.config, small, self.B, self.max_len, self.pos)
        self.cache.load_row(b, small, len(ids))
        return int(torch.argmax(out.logits[0, -1][: chat.v0]))

    # --- one conversation --------------------------------------------------------------------
    def _row(self, b: int, job: Job, budget: int, max_chip: int, check_head: bool):
        """RvChat.generate's executor path as a generator of requests: ``("emb", row)`` asks
        the batch for one circuit step of this row (the kept dims are sent back, fp32 on the
        CPU, the code first), ``("qwen", hc, pos)`` asks for Qwen's two argmaxes. Returns
        generate's result dict, whose ``chip_seconds`` is the time this row spent waiting on
        the circuit -- one replay serves every live row, so it is that wait, not a share."""
        chat, erow, lock = self.chat, self.rows[b], job.lock
        ids = job.ids
        nxt = self._prefill(b, ids)
        core.snapshot(chat.c, job.machine)
        # the circuit holds one machine's tables, so this row's are read out now and go in
        # after the prompt (whose steps overwrite the head states the snapshot describes)
        init = erow.begin()
        embs = chat._exec_emb(ids)
        for t in range(len(ids)):
            yield ("emb", embs[t])
        erow.install(init)
        pos = len(ids)
        gen, n_text, n_chip_tok = [], 0, 0
        t0 = time.time()
        mismatch = 0
        t_chip = 0.0
        expect, lock_error = [], None
        while n_text < budget and n_chip_tok < max_chip:
            gen.append(nxt)
            if lock is not None:
                st = nxt - chat.v0 if nxt >= chat.v0 else BR.text_token(nxt)
                if expect:
                    e = expect.pop(0)
                    if st != e:
                        lock_error = {"at": len(gen) - 1, "got": S.name(st) if st < S.VOCAB
                                      else chat.tok.decode([nxt]),
                                      "want": S.name(e) if e < S.VOCAB
                                      else chat.tok.decode([e - S.TEXT_TOKEN])}
                        break
                elif st < S.VOCAB:
                    lock_error = {"at": len(gen) - 1, "got": S.name(st), "want": "text"}
                    break
                else:
                    k = len(lock.toks)
                    try:
                        lock.qwen([nxt])
                    except RuntimeError as e:  # the reference run did not halt
                        lock_error = {"at": len(gen) - 1, "got": str(e), "want": "a halt"}
                        break
                    expect = list(lock.toks[k + 1:])
            if nxt in chat.eos:
                break
            ts = time.time()
            keep = yield ("emb", chat._exec_row(nxt))
            if nxt >= chat.v0:  # a chip token
                n_chip_tok += 1
                nxt = chat.code_token(keep[: len(chat.code_dims)])
                t_chip += time.time() - ts
                if nxt is None:
                    raise RuntimeError("the circuit gave up a chip position")
            else:
                n_text += 1
                if pos >= self.max_len:
                    lock_error = {"at": len(gen) - 1, "got": "no KV slot left", "want": "text"}
                    break
                hc = chat._exec_hc(nxt, keep)
                ct = chat.circuit_token(hc)
                q, qv = yield ("qwen", hc, pos)
                pos += 1
                if ct is not None and check_head and q != ct:
                    mismatch += 1
                nxt = ct if ct is not None else qv
        return {"tokens": gen, "text_tokens": n_text, "chip_tokens": n_chip_tok,
                "seconds": time.time() - t0, "head_mismatch": mismatch,
                "lock_error": lock_error, "verify_mismatch": None,
                "chip_seconds": round(t_chip, 2), "text_seconds": None}

    # --- the queue ---------------------------------------------------------------------------
    @torch.no_grad()
    def run(self, jobs, budget: int, max_chip: int, done, check_head: bool = True) -> dict:
        """Decode every job; done(job, result) as each finishes. Returns timing stats.

        A turn of the loop either replays the circuit once for every row with an embedding
        pending (the rest parked, their state untouched) or -- once no row has one -- takes one
        batched Qwen step for the rows waiting on it."""
        queue = iter(jobs)
        live: list = [None] * self.B  # (generator, job, pending request)
        hidden = self.chat.lm.embed_tokens.weight.shape[1]
        H = torch.zeros(self.B, hidden, dtype=torch.bfloat16, device=self.dev)
        E = torch.zeros(self.B, self.ex.D, dtype=torch.float32, device=self.dev)
        act = torch.zeros(self.B, dtype=torch.int32, device=self.dev)
        stats = {"steps": 0, "row_steps": 0, "chip_steps": 0, "jobs": 0, "t_step": 0.0,
                 "t_chip": 0.0}

        def advance(b, gen, job, send):
            """Run row b until its next request; refill it when it finishes."""
            while True:
                try:
                    req = gen.send(send) if send is not None else next(gen)
                    live[b] = (gen, job, req)
                    return
                except StopIteration as fin:
                    done(job, fin.value)
                    stats["jobs"] += 1
                    nj = next(queue, None)
                    if nj is None:
                        live[b] = None
                        return
                    gen, job, send = self._row(b, nj, budget, max_chip, check_head), nj, None

        for b in range(self.B):
            j = next(queue, None)
            if j is None:
                break
            advance(b, self._row(b, j, budget, max_chip, check_head), j, None)
        while any(x is not None for x in live):
            emb_rows = [b for b in range(self.B)
                        if live[b] is not None and live[b][2][0] == "emb"]
            if emb_rows:  # one replay moves every row that has a token to push
                act.zero_()
                for b in emb_rows:
                    E[b].copy_(live[b][2][1])
                    act[b] = 1
                ts = time.time()
                self.ex.step(E, act)
                keep = self.ex.keep.cpu()  # one sync for the whole batch
                stats["chip_steps"] += 1
                stats["row_steps"] += len(emb_rows)
                stats["t_chip"] += time.time() - ts
                for b in emb_rows:
                    gen, job, _ = live[b]
                    advance(b, gen, job, keep[b])
                continue
            # no row has a token to push: each waits on Qwen, so take one batched step
            wait = [b for b in range(self.B) if live[b] is not None]
            for b in wait:
                hc, p = live[b][2][1], live[b][2][2]
                H[b].copy_(hc)
                self.pos[b] = p
            ts = time.time()
            res = self._qwen(H)
            stats["t_step"] += time.time() - ts
            stats["steps"] += 1
            for b in wait:
                gen, job, _ = live[b]
                advance(b, gen, job, tuple(res[b]))
        stats["t_chip"] = round(stats["t_chip"], 2)
        return stats
