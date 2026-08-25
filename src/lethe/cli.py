"""lethe — logless local llama.cpp runner with explicit throughput and context."""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import config, hub, server, spec as S, vault
from .ggufmeta import ModelShape, parse_local, quant_from_filename, shape_from_kv
from .paths import llama_bin, llama_dir, models_dir
from .profiles import profile_for
from .sysinfo import GiB, fmt_bytes, llama_processes, machine, wired_bytes

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich")
con = Console()
err = Console(stderr=True)

KV_BYTES = {"f16": 2.0, "bf16": 2.0, "q8_0": 1.0625, "q4_0": 0.5625}
CTX_COLUMNS = [32_768, 131_072]
MODEL_HELP = "Model: owner/repo:QUANT (canonical), a .gguf path, or a unique substring of a local file name"


# ----------------------------------------------------------------------------- helpers


def _shape(path: Path) -> ModelShape:
    return shape_from_kv(parse_local(path), path.name)


def _fit_row(shape: ModelShape | None, size: int, budget: int, kvb: float) -> dict:
    if not shape or not shape.n_layer:
        return {"kv_tok": "?", "ctx_cols": ["?"] * len(CTX_COLUMNS), "native": "?", "max_ctx": "?"}
    per = shape.kv_bytes_per_token(kvb)
    cols = []
    for c in CTX_COLUMNS:
        tot = size + per * c
        cols.append(("[green]" if tot <= budget else "[red]") + fmt_bytes(tot) + "[/]")
    nat = size + per * shape.n_ctx_train
    max_ctx = min(shape.n_ctx_train, shape.max_ctx_for(budget, size, kvb))
    return {
        "kv_tok": f"{per / 1024:.0f} KiB",
        "ctx_cols": cols,
        "native": ("[green]" if nat <= budget else "[red]") + fmt_bytes(nat) + "[/]",
        "max_ctx": f"{max_ctx:,}" if max_ctx > 0 else "[red]0[/]",
    }


def _ctx_headers(t: Table, kvb_name: str):
    t.add_column("size", justify="right")
    t.add_column("quant")
    t.add_column("native ctx", justify="right")
    t.add_column(f"KV/tok ({kvb_name})", justify="right")
    for c in CTX_COLUMNS:
        t.add_column(f"mem @{c // 1024}K", justify="right")
    t.add_column("mem @native", justify="right")
    t.add_column("max ctx here", justify="right")


def _budget_line(kv: str) -> int:
    m = machine()
    con.print(
        f"[dim]{m.chip} · RAM {fmt_bytes(m.total_ram)} · GPU-wired limit {fmt_bytes(m.gpu_wired_limit)} · "
        f"model+KV budget {fmt_bytes(m.usable_for_model)} · KV cache {kv}[/]"
    )
    _in_use_line()
    return m.usable_for_model


def _in_use_line() -> None:
    """Tell the user what other model processes are already holding, so 'fits' is honest."""
    ours = {r["pid"] for r in server.running_servers()}
    procs = [p for p in llama_processes() if p.pid in ours or p.rss > 256 * 1024 * 1024]
    if procs:
        bits = [f"pid {p.pid} {fmt_bytes(p.rss)} ({'lethe' if p.pid in ours else Path(p.cmd.split()[0]).parent.parent.parent.name})" for p in procs]
        con.print(f"[dim]in use now: {', '.join(bits)} · wired total {fmt_bytes(wired_bytes())}[/]")


def _resolve_or_offer(model: str, yes: bool = False) -> S.LocalModel:
    """Resolve a model spec; for owner/repo:QUANT that isn't downloaded, offer to pull it."""
    try:
        return S.resolve(model)
    except S.ModelNotFound as e:
        ms = e.spec
        if not (ms.is_hub and ms.quant):
            err.print(f"[red]{e}[/]. See [bold]lethe models[/] or use owner/repo:QUANT.")
            raise typer.Exit(1)
        with con.status(f"looking up {ms.repo}…"):
            ggufs = hub.list_repo_ggufs(ms.repo)
        hits = [g for g in hub.match_quant(ggufs, ms.quant) if not S.is_sidecar(g.base)]
        if len(hits) != 1:
            err.print(f"[red]{ms} — {'no' if not hits else 'several'} matching file(s) on the Hub[/]: " + ", ".join(g.base for g in ggufs if not S.is_sidecar(g.base)))
            raise typer.Exit(1)
        g = hits[0]
        con.print(f"{ms} is not downloaded: {fmt_bytes(g.size)} → {hub.local_dir_for(ms.repo)}")
        if not yes and not (sys.stdin.isatty() and typer.confirm("download it now?", default=False)):
            raise typer.Exit(1)
        _pull(ms.repo, g, ggufs)
        return S.resolve(model)


def _pull(repo: str, g: hub.HubGguf, ggufs: list[hub.HubGguf], sidecars: bool = True) -> None:
    want = list(g.files)
    if sidecars:
        for s in ggufs:
            if S.is_sidecar(s.base) and s.size < 3 * GiB:
                want += s.files
        from huggingface_hub import HfApi

        for s in HfApi().model_info(repo).siblings or []:
            if s.rfilename.endswith((".patch", "SHA256SUMS", "MANIFEST.json")):
                want.append(s.rfilename)
    want = list(dict.fromkeys(want))
    total = sum(x.size for x in ggufs for f in x.files if f in want)
    con.print(f"downloading {len(want)} files (~{fmt_bytes(total)}) → {hub.local_dir_for(repo)}")
    for f in want:
        con.print(f"  {f}")
    hub.download(repo, want)
    con.print("[green]done[/]")


def _find_sidecar(lm: S.LocalModel, kind: str) -> Path | None:
    for p in lm.path.parent.glob("*.gguf"):
        n = p.name.lower()
        if kind == "draft" and "fastmtp" in n:
            return p
        if kind == "mmproj" and n.startswith("mmproj"):
            return p
    return None


def _plan_ctx(shape: ModelShape, size: int, ctx: int, kv: str, draft_size: int) -> int:
    """Pick/validate a context size against what is free *now* and print the memory plan."""
    m = machine()
    kvb = KV_BYTES[kv]
    per = shape.kv_bytes_per_token(kvb)
    fixed = size + draft_size
    capacity = m.usable_for_model
    available = m.available_now()
    budget = min(capacity, available)
    max_fit = shape.max_ctx_for(budget, fixed, kvb)
    if ctx == 0:
        ctx = shape.n_ctx_train
        if ctx > max_fit:
            ctx = max(4096, (max_fit // 4096) * 4096)
            con.print(f"[yellow]native context {shape.n_ctx_train:,} does not fit in what's free; using {ctx:,}[/]")
    kv_total = per * ctx
    total = fixed + kv_total
    color = "green" if total <= budget else "red"
    con.print(
        f"context [bold]{ctx:,}[/] (native max {shape.n_ctx_train:,}) · "
        f"model {fmt_bytes(size)}"
        + (f" + draft {fmt_bytes(draft_size)}" if draft_size else "")
        + f" + KV {fmt_bytes(kv_total)} ({kv}, {per / 1024:.0f} KiB/tok) = [{color}]{fmt_bytes(total)}[/]"
        + f" of {fmt_bytes(available)} free now ({fmt_bytes(capacity)} capacity)"
    )
    if available < capacity - 2 * GiB:
        _in_use_line()
    if total > budget:
        con.print(f"[red]that exceeds what is free[/] — stop other models, lower --ctx (max ≈ {max_fit:,}), or use --kv q8_0")
    return ctx


def _launch(
    lm: S.LocalModel, *, ctx: int, kv: str, mtp: bool, draft: bool, draft_n: int, vision: bool, thinking: bool,
    effort: str | None, port: int, no_mmap: bool, log: bool, api_key: bool, extra: list[str], foreground: bool = False,
) -> dict:
    shape = _shape(lm.path)
    draft_path = _find_sidecar(lm, "draft") if (mtp and draft) else None
    draft_size = draft_path.stat().st_size if draft_path else 0
    if server.running_servers():
        con.print("stopping previous server…")
        server.stop()
    ctx = _plan_ctx(shape, lm.size, ctx, kv, draft_size)
    if mtp and not shape.has_mtp and not draft_path:
        con.print("[yellow]model has no MTP head; disabling speculative decoding[/]")
        mtp = False
    mmproj = _find_sidecar(lm, "mmproj") if vision else None
    if vision and not mmproj:
        con.print("[yellow]no mmproj next to the model; vision off[/]")
    cfg = server.ServeConfig(
        model=lm.path, n_ctx=ctx, draft=draft_path, mtp=mtp, draft_n_max=draft_n, mmproj=mmproj, kv_type=kv,
        thinking=thinking, reasoning_effort=effort, port=port, no_mmap=no_mmap, log=log,
        api_key=server.new_api_key() if api_key else None, extra=list(extra),
    )
    con.print(
        f"speculative: {'FastMTP draft ' + draft_path.name if draft_path else ('embedded MTP' if mtp else 'off')}"
        + (f" (depth {draft_n})" if mtp else "")
        + f" · thinking {'on' + (' ' + effort if effort else '') if thinking else 'off'}"
        + (" · vision" if mmproj else "")
    )
    if foreground:
        con.print("[dim]$ " + " ".join(cfg.argv()) + "[/]")
        raise typer.Exit(subprocess.call(cfg.argv()))
    t0 = time.time()
    with con.status("loading model…") as st:
        rec = server.start(cfg, on_wait=lambda s: st.update(f"loading model… {s:.0f}s"))
    p = server.props(rec["url"], cfg.api_key)
    con.print(
        f"[green]ready[/] in {time.time() - t0:.1f}s at {rec['url']} · n_ctx {p.get('default_generation_settings', {}).get('n_ctx', '?'):,} · "
        + (f"[yellow]logging to {server.log_path()}[/]" if log else "[dim]traceless: no log, no web UI, no slots endpoint[/]")
        + (f"\n[dim]api key required for other clients: {cfg.api_key} (also: lethe key)[/]" if cfg.api_key else "\n[yellow]no api key: any local process or web page can use this server[/]")
    )
    return rec


# ----------------------------------------------------------------------------- commands


@app.command()
def doctor():
    """Verify the Metal build, memory limits, and that nothing is on disk that shouldn't be."""
    m = machine()
    con.print(f"[bold]{m.chip}[/] — {m.ncpu} CPU cores, RAM {fmt_bytes(m.total_ram)}")
    con.print(f"GPU-wired limit: {fmt_bytes(m.gpu_wired_limit)} (raise with: sudo sysctl iogpu.wired_limit_mb=<MB>)")
    con.print(f"Model+KV budget: {fmt_bytes(m.usable_for_model)} · free now {fmt_bytes(m.available_now())}")
    _in_use_line()
    ok = True
    for name in ("llama-server", "llama-cli", "llama-bench"):
        b = llama_bin(name)
        if b.exists():
            con.print(f"[green]✓[/] {b.relative_to(llama_dir().parent.parent)}")
        else:
            con.print(f"[red]✗[/] missing {name} — run scripts/build-llama.sh")
            ok = False
    if ok:
        out = subprocess.run([str(llama_bin("llama-server")), "--version"], capture_output=True, text=True)
        ver = (out.stderr or out.stdout).strip().splitlines()
        con.print(f"  {ver[0] if ver else out.returncode}")
        if out.returncode != 0:
            con.print("[red]✗[/] llama-server does not run (moved repo? rebuild: scripts/build-llama.sh --clean)")
            ok = False
        if llama_bin("libggml-metal.dylib").exists():
            con.print("[green]✓[/] Metal backend linked (libggml-metal)")
        else:
            con.print("[red]✗[/] Metal backend NOT linked — rebuild with scripts/build-llama.sh")
            ok = False
        dev = subprocess.run([str(llama_bin("llama-server")), "--list-devices"], capture_output=True, text=True)
        txt = (dev.stdout + dev.stderr).strip()
        if txt:
            con.print("  " + "\n  ".join(txt.splitlines()[:4]))
    head = subprocess.run(["git", "-C", str(llama_dir()), "log", "-1", "--format=%h %s"], capture_output=True, text=True).stdout.strip()
    con.print(f"llama.cpp: {head}")
    con.print(f"local models: {len(S.local_models())} in {models_dir()}" + (f" · default {config.default_model()}" if config.default_model() else ""))
    rec = server.running_server()
    if rec:
        con.print(
            f"[green]server running[/] pid {rec['pid']} at {rec['url']} — {Path(rec['model']).name}"
            + (" [yellow](logging ON)[/]" if rec["logging"] else " [dim](traceless)[/]")
            + ("" if rec["api_key"] else " [yellow](no api key)[/]")
        )
    if server.log_path().exists():
        con.print(f"[yellow]![/] {server.log_path()} exists (from a --log run) — remove with: lethe traces --delete")
    else:
        con.print("[green]✓[/] no logs on disk")
    n = len(vault.list_entries())
    if n:
        con.print(f"[dim]{n} encrypted conversation(s) in {vault.vault_dir()} (lethe conversations)[/]")
    raise typer.Exit(0 if ok else 1)


@app.command()
def browse(
    repo: str = typer.Argument(..., help="HF repo id (owner/repo, a trailing :QUANT is ignored)"),
    kv: str = typer.Option("f16", help="KV cache type used for the memory columns: f16|q8_0|q4_0"),
    each: bool = typer.Option(False, help="Parse every file's header (default: one per repo, shared shape)"),
):
    """List a Hub repo's GGUFs with size, quant, native context, KV cost and what fits on this machine."""
    repo = S.ModelSpec.parse(repo).repo or repo
    budget = _budget_line(kv)
    kvb = KV_BYTES[kv]
    with con.status(f"listing {repo}…"):
        ggufs = hub.list_repo_ggufs(repo)
    if not ggufs:
        err.print("[red]no .gguf files in that repo[/]")
        raise typer.Exit(1)
    shared: ModelShape | None = None
    with con.status("reading GGUF headers…"):
        for g in ggufs:
            if S.is_sidecar(g.base):
                continue
            if each or shared is None:
                hub.fetch_shape(g)
                if g.shape and g.shape.n_layer and shared is None:
                    shared = g.shape
            if g.shape is None and shared is not None:
                g.shape = replace(shared, quant=quant_from_filename(g.base) or shared.quant)
    local = {lm.spec for lm in S.local_models()}
    t = Table(title=repo, expand=False, pad_edge=False)
    t.add_column("spec (owner/repo:QUANT)")
    _ctx_headers(t, kv)
    for g in ggufs:
        s = g.shape
        fit = _fit_row(s, g.size, budget, kvb)
        if S.is_sidecar(g.base):
            t.add_row(f"[dim]{g.base}[/]", fmt_bytes(g.size), "[dim]sidecar[/]", "", "", *[""] * len(CTX_COLUMNS), "", "")
            continue
        quant = (s.quant if s and s.quant else quant_from_filename(g.base)) or "?"
        spec_str = f"{repo}:{quant}"
        t.add_row(
            spec_str + (" [green]✓ local[/]" if spec_str in local else "") + (f" [dim]({len(g.files)} shards)[/]" if len(g.files) > 1 else ""),
            fmt_bytes(g.size), quant,
            f"{s.n_ctx_train:,}" if s and s.n_ctx_train else "?",
            fit["kv_tok"], *fit["ctx_cols"], fit["native"], fit["max_ctx"],
        )
    con.print(t)
    if shared:
        con.print(
            f"[dim]{shared.name}: {shared.n_layer} layers, {shared.n_attn_layers} with KV cache"
            f"{' (hybrid: interval ' + str(shared.extra.get('full_attention_interval')) + ')' if shared.n_attn_layers != shared.n_layer else ''}, "
            f"{shared.n_head_kv} KV heads × {shared.key_len}+{shared.val_len}"
            f"{', MTP head' if shared.has_mtp else ''}"
            f"{', MoE ' + str(shared.extra.get('expert_used_count')) + '/' + str(shared.extra.get('expert_count')) if shared.extra.get('expert_count') else ''}"
            f" · sampler profile: {profile_for(shared).name}[/]"
        )
    con.print("[dim]mem = file size + KV cache; the real footprint adds ~1–3 GiB of compute buffers. Green fits the budget.[/]")


@app.command()
def pull(
    model: str = typer.Argument(..., help="owner/repo:QUANT (or owner/repo with --quant / --file)"),
    quant: Optional[str] = typer.Option(None, "--quant", "-q", help="Quant to fetch if not given in the spec"),
    file: list[str] = typer.Option([], "--file", "-f", help="Exact filename(s) to fetch (repeatable)"),
    sidecars: bool = typer.Option(True, help="Also fetch mmproj / FastMTP draft / patch files"),
):
    """Download GGUF(s) from the Hub into ./models/<owner>--<repo>/."""
    ms = S.ModelSpec.parse(model)
    if not ms.is_hub:
        err.print("[red]give a Hub spec: owner/repo:QUANT[/]")
        raise typer.Exit(1)
    quant = (quant or ms.quant or "").upper() or None
    with con.status(f"listing {ms.repo}…"):
        ggufs = hub.list_repo_ggufs(ms.repo)
    if file:
        hub.download(ms.repo, list(file))
        con.print("[green]done[/]")
        return
    if not quant:
        err.print("[red]which quant?[/] " + ", ".join(f"{ms.repo}:{quant_from_filename(g.base) or g.base}" for g in ggufs if not S.is_sidecar(g.base)))
        raise typer.Exit(1)
    hits = [g for g in hub.match_quant(ggufs, quant) if not S.is_sidecar(g.base)]
    if len(hits) != 1:
        err.print(f"[red]{'no' if not hits else 'several'} files match {quant}[/]: " + ", ".join(g.base for g in (hits or ggufs)))
        raise typer.Exit(1)
    _pull(ms.repo, hits[0], ggufs, sidecars=sidecars)


@app.command()
def models(kv: str = typer.Option("f16", help="KV cache type for memory columns: f16|q8_0|q4_0")):
    """List local models (by canonical spec) with native context, KV cost and max context that fits."""
    budget = _budget_line(kv)
    kvb = KV_BYTES[kv]
    rows = S.local_models(include_sidecars=True)
    if not rows:
        con.print(f"no models in {models_dir()} — try [bold]lethe pull owner/repo:QUANT[/]")
        return
    default = config.default_model()
    t = Table(expand=False, pad_edge=False)
    t.add_column("spec (owner/repo:QUANT)")
    _ctx_headers(t, kv)
    for lm in rows:
        if lm.is_sidecar:
            t.add_row(f"[dim]{lm.path.relative_to(models_dir())}[/]", fmt_bytes(lm.size), "[dim]sidecar[/]", "", "", *[""] * len(CTX_COLUMNS), "", "")
            continue
        s = _shape(lm.path)
        fit = _fit_row(s, lm.size, budget, kvb)
        t.add_row(
            lm.spec + (" [green]★ default[/]" if lm.spec == default else "") + (f" [dim]({len(lm.files)} shards)[/]" if len(lm.files) > 1 else ""),
            fmt_bytes(lm.size), lm.quant or "-", f"{s.n_ctx_train:,}",
            fit["kv_tok"], *fit["ctx_cols"], fit["native"], fit["max_ctx"],
        )
    con.print(t)


@app.command()
def default(model: Optional[str] = typer.Argument(None, help=MODEL_HELP + " — omit to show the current default")):
    """Show or set the default model (stored in lethe.toml — configuration only, never content)."""
    if model is None:
        cur = config.default_model()
        con.print(cur or "[dim]no default set — lethe default owner/repo:QUANT[/]")
        return
    lm = _resolve_or_offer(model)
    config.set_default_model(lm.spec)
    con.print(f"default model: [bold]{lm.spec}[/] ({config.config_path()})")


@app.command()
def serve(
    model: Optional[str] = typer.Argument(None, help=MODEL_HELP + " (default: lethe.toml)"),
    ctx: int = typer.Option(0, "--ctx", "-c", help="Context tokens (0 = native max if it fits)"),
    kv: str = typer.Option("f16", help="KV cache type: f16|q8_0|q4_0"),
    mtp: bool = typer.Option(True, help="MTP speculative decoding (embedded head)"),
    draft: bool = typer.Option(True, help="Use a FastMTP draft sidecar if one sits next to the model"),
    draft_n: int = typer.Option(3, help="Draft depth (--spec-draft-n-max)"),
    vision: bool = typer.Option(False, help="Load the mmproj vision projector if present"),
    thinking: bool = typer.Option(True, help="Thinking/reasoning mode on"),
    effort: Optional[str] = typer.Option(None, help="Reasoning effort: low|medium|xhigh (Qwen3.8)"),
    port: int = typer.Option(server.DEFAULT_PORT),
    api_key: bool = typer.Option(True, help="Require a per-launch random API key (protects the server from other local software)"),
    no_mmap: bool = typer.Option(False, help="Read the whole model into memory instead of mmap"),
    foreground: bool = typer.Option(False, "--fg", help="Run in the foreground with llama-server's output on the terminal"),
    log: bool = typer.Option(False, "--log", help="OPT-IN: keep llama-server's log in .lethe/server.log (off by default: traceless)"),
    yes: bool = typer.Option(False, "-y", "--yes", help="Download the model without asking if it isn't local"),
    extra: list[str] = typer.Option([], "--arg", help="Extra llama-server args (repeatable)"),
):
    """Launch llama-server for a model with Metal, flash-attn and MTP, and print the memory plan."""
    model = model or config.default_model()
    if not model:
        err.print("[red]no model given and no default set[/] — lethe serve owner/repo:QUANT, or lethe default …")
        raise typer.Exit(1)
    lm = _resolve_or_offer(model, yes)
    _launch(lm, ctx=ctx, kv=kv, mtp=mtp, draft=draft, draft_n=draft_n, vision=vision, thinking=thinking, effort=effort,
            port=port, no_mmap=no_mmap, log=log, api_key=api_key, extra=extra, foreground=foreground)


@app.command()
def stop():
    """Stop the running llama-server."""
    n = server.stop()
    con.print(f"[green]stopped {n}[/]" if n else "no server running")


@app.command()
def key():
    """Print the running server's API key (for pointing other OpenAI-compatible clients at it)."""
    rec = server.running_server()
    if not rec:
        err.print("no server running")
        raise typer.Exit(1)
    con.print(rec["api_key"] or "[dim](server started without an api key)[/]")


@app.command()
def chat(
    model: Optional[str] = typer.Argument(None, help=MODEL_HELP + " (default: running server, else lethe.toml)"),
    ctx: int = typer.Option(0, "--ctx", "-c", help="Context tokens (0 = native max if it fits)"),
    kv: str = typer.Option("f16", help="KV cache type: f16|q8_0|q4_0"),
    thinking: bool = typer.Option(True, help="Thinking mode (server-side default + sampler preset)"),
    show_thinking: bool = typer.Option(True, help="Stream the model's reasoning (dimmed)"),
    vision: bool = typer.Option(False, help="Load the vision projector so /image works"),
    system: Optional[str] = typer.Option(None, help="System prompt"),
    file: list[Path] = typer.Option([], "--file", "-f", help="Attach text/PDF file(s) to the first message (repeatable)"),
    max_tokens: Optional[int] = typer.Option(None, help="Cap generation length per turn"),
    temp: Optional[float] = typer.Option(None, help="Override sampler temperature"),
    top_p: Optional[float] = typer.Option(None, help="Override top_p"),
    top_k: Optional[int] = typer.Option(None, help="Override top_k"),
    min_p: Optional[float] = typer.Option(None, help="Override min_p"),
    save: bool = typer.Option(False, "--save", help="OPT-IN: save this conversation encrypted (asks for a passphrase)"),
    resume: Optional[str] = typer.Option(None, "--resume", help="Resume a saved conversation by id (see lethe conversations)"),
    port: int = typer.Option(server.DEFAULT_PORT),
    yes: bool = typer.Option(False, "-y", "--yes", help="Download the model without asking if it isn't local"),
):
    """Interactive chat. Starts the server if needed. Prints throughput + context stats after every turn."""
    for f in file:
        if not f.is_file():
            err.print(f"[red]no such file: {f}[/]")
            raise typer.Exit(1)
    resumed: vault.Conversation | None = None
    passphrase: str | None = None
    if resume:
        from .repl import ask_passphrase

        passphrase = ask_passphrase(confirm=False)
        if not passphrase:
            raise typer.Exit(1)
        try:
            resumed = vault.load(resume, passphrase)
        except (vault.WrongPassphrase, FileNotFoundError, ValueError) as e:
            err.print(f"[red]{e}[/]")
            raise typer.Exit(1)
        model = model or resumed.model
        save = True
    elif save:
        from .repl import ask_passphrase

        passphrase = ask_passphrase(confirm=True)
        if not passphrase:
            raise typer.Exit(1)

    rec = server.running_server(port)
    lm: S.LocalModel | None = None
    if model:
        lm = _resolve_or_offer(model, yes)
        if not rec or Path(rec["model"]) != lm.path or (vision and "--mmproj" not in rec["argv"]):
            rec = _launch(lm, ctx=ctx, kv=kv, mtp=True, draft=True, draft_n=3, vision=vision, thinking=thinking, effort=None,
                          port=port, no_mmap=False, log=False, api_key=True, extra=[])
    elif not rec:
        dflt = config.default_model()
        if not dflt:
            err.print("[red]no server running, no model given, no default set[/] — lethe chat owner/repo:QUANT")
            raise typer.Exit(1)
        lm = _resolve_or_offer(dflt, yes)
        rec = _launch(lm, ctx=ctx, kv=kv, mtp=True, draft=True, draft_n=3, vision=vision, thinking=thinking, effort=None,
                      port=port, no_mmap=False, log=False, api_key=True, extra=[])
    if lm is None:
        lm = S.resolve(rec["model"])
    api_key = rec.get("api_key")
    p = server.props(rec["url"], api_key)
    n_ctx = int(p.get("default_generation_settings", {}).get("n_ctx", 0))
    has_vision = bool((p.get("modalities") or {}).get("vision"))
    overrides = {k: v for k, v in {"temperature": temp, "top_p": top_p, "top_k": top_k, "min_p": min_p}.items() if v is not None}

    from .repl import ReplConfig, run

    run(ReplConfig(
        url=rec["url"], api_key=api_key, n_ctx=n_ctx, model_spec=lm.spec, profile=profile_for(_shape(lm.path)),
        thinking=thinking, show_thinking=show_thinking, system=system, max_tokens=max_tokens,
        sampling_overrides=overrides, attachments=list(file), vision=has_vision,
        save=save, resume=resumed, passphrase=passphrase,
    ))


@app.command()
def conversations(
    delete: Optional[str] = typer.Option(None, "--delete", help="Delete a saved conversation by id (overwritten, then removed)"),
    unlock: bool = typer.Option(False, "--unlock", help="Ask for the passphrase and show titles/models (never shown otherwise)"),
):
    """List encrypted saved conversations (ids, sizes, dates — titles only with --unlock)."""
    if delete:
        con.print("[green]deleted[/]" if vault.delete(delete) else f"no conversation {delete}")
        return
    entries = vault.list_entries()
    if not entries:
        con.print("[dim]no saved conversations (saving is opt-in: lethe chat --save)[/]")
        return
    passphrase = None
    if unlock:
        from .repl import ask_passphrase

        passphrase = ask_passphrase(confirm=False)
    t = Table(expand=False, pad_edge=False)
    t.add_column("id")
    t.add_column("updated")
    t.add_column("size", justify="right")
    if passphrase:
        t.add_column("model")
        t.add_column("turns", justify="right")
        t.add_column("title")
    for e in entries:
        row = [e.id, time.strftime("%Y-%m-%d %H:%M", time.localtime(e.mtime)), fmt_bytes(e.size)]
        if passphrase:
            try:
                c = vault.load(e.id, passphrase)
                row += [c.model, str(c.turns), c.title]
            except vault.WrongPassphrase:
                row += ["[red]wrong passphrase[/]", "", ""]
        t.add_row(*row)
    con.print(t)
    con.print("[dim]resume: lethe chat --resume <id> · delete: lethe conversations --delete <id>[/]")


@app.command()
def bench(
    model: Optional[str] = typer.Argument(None, help=MODEL_HELP + " (default: lethe.toml)"),
    prompt: str = typer.Option("2048", "-p", help="llama-bench: prompt lengths (comma list)"),
    gen: str = typer.Option("128", "-n", help="llama-bench: generation lengths (comma list)"),
    depth: str = typer.Option("0", "-d", help="Context depths to test at, e.g. 0,32768"),
    kv: str = typer.Option("f16", help="KV cache type"),
    server_mode: bool = typer.Option(False, "--server", help="Benchmark through llama-server instead: measures real MTP/FastMTP speed and acceptance"),
    extra: list[str] = typer.Option([], "--arg", help="Extra llama-bench args"),
):
    """Throughput benchmark. Default: llama-bench (no speculative decoding = the floor). --server: with MTP."""
    model = model or config.default_model()
    if not model:
        err.print("[red]no model given and no default set[/]")
        raise typer.Exit(1)
    lm = _resolve_or_offer(model)
    depths = [int(x) for x in depth.split(",") if x.strip()]
    if server_mode:
        from .benchserver import run_server_bench

        run_server_bench(lm, depths=depths, kv=kv, launch=_launch)
        return
    argv = [
        str(llama_bin("llama-bench")), "-m", str(lm.path), "-ngl", "99", "-fa", "1",
        "-p", prompt, "-n", gen, "-d", depth, "-ctk", kv, "-ctv", kv, *extra,
    ]
    con.print("[dim]$ " + " ".join(argv) + "[/]")
    con.print("[dim]note: llama-bench cannot use MTP; its tg is the floor. Use --server for the real number.[/]")
    raise typer.Exit(subprocess.call(argv))


@app.command()
def log(lines: int = typer.Option(60, "-n")):
    """Tail the server log (only exists if you started the server with --log)."""
    p = server.log_path()
    if not p.exists():
        con.print("[dim]no log — logging is off by default. Opt in with[/] lethe serve <model> --log")
        return
    con.print("\n".join(p.read_text(errors="replace").splitlines()[-lines:]))


@app.command()
def traces(delete: bool = typer.Option(False, "--delete", help="Delete what is found (saved conversations included!)")):
    """List (and optionally delete) anything lethe may have written to disk beyond model files."""
    found: list[tuple[Path, str]] = []
    if server.log_path().exists():
        found.append((server.log_path(), "llama-server log (opt-in --log)"))
    for e in vault.list_entries():
        found.append((e.path, "encrypted saved conversation (opt-in --save)"))
    if config.config_path().exists():
        found.append((config.config_path(), "configuration (default model) — no content"))
    for p in models_dir().glob("*/.cache"):
        if p.is_dir():
            found.append((p, "huggingface_hub download bookkeeping (which files/revisions were fetched — no content)"))
    if not found:
        con.print("[green]clean[/] — nothing on disk except model files")
        return
    for p, what in found:
        con.print(f"{'[red]deleting[/]' if delete else '[yellow]found[/]'} {p} — {what}")
        if delete:
            import shutil

            shutil.rmtree(p) if p.is_dir() else p.unlink()


if __name__ == "__main__":
    app()
