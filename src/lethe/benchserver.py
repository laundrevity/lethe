"""Benchmark through llama-server: the number llama-bench can't give you (MTP / FastMTP, at context depth)."""

from __future__ import annotations

import time
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from . import server, spec as S
from .chat import TurnStats, stream_chat, tokenize

con = Console()

PROMPTS = {
    "prose": "Write a detailed, 400-word essay on how unified memory architectures change the economics of running large language models on personal computers.",
    "reasoning": "A train leaves city A at 9:00 travelling 80 km/h toward city B, 300 km away. Another leaves B at 9:30 travelling 100 km/h toward A. When and where do they meet? Show your working step by step, then give the answer.",
    "code": "Write a Python function that parses the header of a GGUF file (magic, version, tensor count, and all key-value metadata) without loading tensor data. Include a docstring and type hints.",
}
FILLER = (
    "The archive room was quiet except for the hum of the dehumidifier. Rows of grey boxes lined the shelves, each labelled "
    "in a careful hand with a year and a district. Nobody had opened most of them in decades, and the paper inside had "
    "yellowed at the edges into the colour of weak tea. She pulled the third box from the left and set it on the table. "
)


@dataclass
class Row:
    config: str
    depth: int
    pp_tps: float
    tg_tps: float
    accept: float | None
    n_gen: int


def _filler_of_tokens(url: str, api_key: str | None, n: int) -> str:
    if n <= 0:
        return ""
    reps = max(1, n // 60)
    text = FILLER * reps
    while tokenize(url, text, api_key) < n:
        text += FILLER * max(1, reps // 4)
    return text


def run_server_bench(lm: S.LocalModel, *, depths: list[int], kv: str, launch) -> None:
    has_draft = any("fastmtp" in p.name.lower() for p in lm.path.parent.glob("*.gguf"))
    configs = [("no MTP", dict(mtp=False, draft=False)), ("embedded MTP d3", dict(mtp=True, draft=False))]
    if has_draft:
        configs.append(("FastMTP d3", dict(mtp=True, draft=True)))
    ctx = max(depths) + 8192
    rows: list[Row] = []
    con.print(f"[bold]server benchmark[/] {lm.spec} · configs: {', '.join(c for c, _ in configs)} · depths {depths} · greedy, thinking off, 256 tokens")
    for name, opts in configs:
        rec = launch(lm, ctx=ctx, kv=kv, draft_n=3, vision=False, thinking=False, effort=None, port=server.DEFAULT_PORT,
                     no_mmap=False, log=False, api_key=True, extra=[], **opts)
        url, key = rec["url"], rec.get("api_key")
        for d in depths:
            prefix = _filler_of_tokens(url, key, d)
            stats: list[TurnStats] = []
            for pname, ptext in PROMPTS.items():
                msgs = [{"role": "user", "content": (prefix + "\n\n---\n\nIgnore the text above. " if prefix else "") + ptext}]
                t0 = time.time()
                with con.status(f"{name} · depth {d:,} · {pname}…"):
                    for ev in stream_chat(url, msgs, n_ctx=ctx, sampling={"temperature": 0.0}, max_tokens=256, chat_template_kwargs={"enable_thinking": False}, api_key=key):
                        if isinstance(ev, TurnStats):
                            stats.append(ev)
                con.print(f"[dim]  {name} · d{d:,} · {pname}: {stats[-1].line(markup=False)} · {time.time() - t0:.1f}s[/]")
            # pp from the first prompt (the others hit the prefix cache); tg averaged over all
            pp = stats[0].prompt_tps
            gen_n = sum(s.predicted_n for s in stats)
            gen_ms = sum(s.predicted_ms for s in stats)
            dn = sum(s.draft_n for s in stats)
            da = sum(s.draft_n_accepted for s in stats)
            rows.append(Row(name, d, pp, gen_n / (gen_ms / 1000) if gen_ms else 0.0, (da / dn) if dn else None, gen_n))
    server.stop()
    t = Table(title=f"{lm.spec} — llama-server, {kv} KV", expand=False)
    t.add_column("config")
    t.add_column("depth", justify="right")
    t.add_column("pp tok/s", justify="right")
    t.add_column("tg tok/s", justify="right")
    t.add_column("accept", justify="right")
    t.add_column("vs no-MTP", justify="right")
    base = {r.depth: r.tg_tps for r in rows if r.config == "no MTP"}
    for r in rows:
        ratio = (r.tg_tps / base[r.depth]) if base.get(r.depth) else None
        t.add_row(r.config, f"{r.depth:,}", f"{r.pp_tps:,.0f}", f"{r.tg_tps:.1f}", f"{r.accept:.0%}" if r.accept is not None else "—", f"{ratio:.2f}×" if ratio else "—")
    con.print(t)
