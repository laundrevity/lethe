"""lethe — logless local llama.cpp runner with explicit throughput and context."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Iterator, Optional

import typer
from rich.console import Console
from rich.table import Table

from . import config, hub, server, spec as S, tools as T, vault
from .attachments import is_image, is_video
from .ggufmeta import ModelShape, parse_local, quant_from_filename, shape_from_kv
from .paths import llama_bin, llama_dir, models_dir
from .profiles import describe, profile_for
from .shell import ShellMode
from .sysinfo import GiB, fmt_bytes, llama_processes, machine, wired_bytes

app = typer.Typer(add_completion=False, rich_markup_mode="rich", help="Bare `lethe …` (no command) runs `lethe chat …` — e.g. `lethe --web` chats with internet tools. See `lethe chat --help` for all chat flags.")
con = Console()
err = Console(stderr=True)

KV_BYTES = {"f16": 2.0, "bf16": 2.0, "q8_0": 1.0625, "q4_0": 0.5625}
CTX_COLUMNS = [32_768, 131_072]
MODEL_HELP = "Model: owner/repo:QUANT (canonical), a .gguf path, or a unique substring of a local file name"
SERVE_KEY_ENV = "LETHE_API_KEY"
SERVE_KEY_MIN = 16
SAMPLER_FLAGS = {"temperature": "--temp", "top_p": "--top-p", "top_k": "--top-k", "min_p": "--min-p",
                 "presence_penalty": "--presence-penalty", "repeat_penalty": "--repeat-penalty"}


# ----------------------------------------------------------------------------- helpers


def _resume_args(context: typer.Context) -> list[str]:
    """Keep explicit launch options for the terminal hint, never in config or the vault.

    Saved messages already contain the system prompt and attachments. Saving and downloads require
    a fresh opt-in; the resume id is supplied separately. Use parsed values so aliases, enums and
    negative boolean flags round-trip without copying raw argv or exposing prompt text.
    """
    omitted = {"system", "file", "save", "resume", "yes"}
    args = []
    for param in context.command.params:
        source = context.get_parameter_source(param.name)
        if param.name in omitted or source is None or source.name != "COMMANDLINE":
            continue
        value = context.params[param.name]
        if param.name == "model":
            args.append(str(value))
        elif isinstance(value, bool):
            flags = param.opts if value else param.secondary_opts
            args.extend(flags[:1])
        else:
            if param.name == "shell_dir":
                value = Path(value).expanduser().resolve()
            args.extend([param.opts[0], str(getattr(value, "value", value))])
    return args


def _default_system(lm: S.LocalModel, shape: ModelShape, net_tools: bool = False) -> str:
    """Ground the model in its actual situation: what it is, whose machine this is, and that nobody is listening."""
    m = machine()
    prof = profile_for(shape)
    ident = shape.name or str(lm.spec)
    bits = [b for b in (f"{shape.params_b:g}B parameters" if shape.params_b else "", shape.quant or "") if b]
    ident += f" ({', '.join(bits)})" if bits else ""
    lineage = f", based on the {prof.name} model family by {prof.developer}" if prof.developer else ""
    owner = str(lm.spec).split("/")[0]
    tuned = f", packaged by {owner}" if prof.developer and owner.lower() not in prof.developer.lower() else ""
    return (
        f"You are {ident}{lineage}{tuned}. You are running entirely locally on the user's own computer — "
        f"an {m.chip} with {m.total_ram // GiB} GB of unified memory — via llama.cpp under lethe, a private, logless runner.\n"
        + time.strftime("Current date and time at launch: %A %Y-%m-%d %H:%M %Z.\n")
        + "Your training data ends well before this date; assume the world has moved on in ways you do not yet know. "
        "This is a real private session with a real person — not a test, an evaluation, or roleplay — so treat surprising "
        "post-cutoff information (unfamiliar events, products, model names) as genuine news, not as fiction or a scenario.\n"
        + "This conversation happens in a vacuum: inference is on-device, and nothing is "
        + (T.VACUUM_CLAIM if not net_tools else "")
        + T.PRIVACY_NOTE
        + (T.SYSTEM_NOTE if net_tools else "")
    )


def _shape(path: Path) -> ModelShape:
    return shape_from_kv(parse_local(path), path.name)


def _fit_row(shape: ModelShape | None, size: int, budget: int, kvb: float) -> dict:
    if not shape or not shape.n_layer:
        return {"kv_tok": "?", "ctx_cols": ["?"] * len(CTX_COLUMNS), "native": "?", "max_ctx": "?"}
    per = shape.kv_bytes_per_token(kvb)
    cols = []
    for c in CTX_COLUMNS:
        tot = size + shape.kv_bytes(c, kvb)
        cols.append(("[green]" if tot <= budget else "[red]") + fmt_bytes(tot) + "[/]")
    nat = size + shape.kv_bytes(shape.n_ctx_train, kvb)
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
    procs = [p for p in llama_processes() if p.rss > 256 * 1024 * 1024]
    if procs:
        bits = [f"pid {p.pid} {fmt_bytes(p.rss)} ({Path(p.cmd.split()[0]).parent.parent.parent.name})" for p in procs]
        suffix = f" · wired total {fmt_bytes(wired_bytes())}" if sys.platform == "darwin" else " (process RSS; GPU allocations may be additional)"
        con.print(f"[dim]in use now: {', '.join(bits)}{suffix}[/]")


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
    total = sum(x.size for x in ggufs if any(f in want for f in x.files))  # x.size covers all shards of x
    con.print(f"downloading {len(want)} files (~{fmt_bytes(total)}) → {hub.local_dir_for(repo)}")
    for f in want:
        con.print(f"  {f}")
    hub.download(repo, want)
    con.print("[green]done[/]")


def _find_sidecar(lm: S.LocalModel, kind: str) -> Path | None:
    directories = [lm.path.parent]
    # Sharded quants often live in a subdirectory, with a shared projector at
    # the repo root. Never search another model's directory for a projector.
    if kind == "mmproj" and lm.repo:
        root = models_dir() / S.local_dir_name(lm.repo)
        if lm.path.is_relative_to(root):
            while directories[-1] != root:
                directories.append(directories[-1].parent)
    for directory in directories:
        candidates = sorted(directory.glob("*.gguf"), key=lambda p: (not p.stem.lower().endswith("-f16"), p.name))
        for p in candidates:
            n = p.name.lower()
            if kind == "draft" and ("fastmtp" in n or n.startswith("mtp-")):
                return p
            if kind == "mmproj" and n.startswith("mmproj"):
                return p
    return None


def _plan_ctx(shape: ModelShape, size: int, ctx: int, kv: str, draft_size: int, mmproj_size: int = 0) -> int:
    """Pick/validate a context size against what is free *now* and print the memory plan."""
    m = machine()
    kvb = KV_BYTES[kv]
    per = shape.kv_bytes_per_token(kvb)
    fixed = size + draft_size + mmproj_size
    capacity = m.usable_for_model
    available = m.available_now()
    budget = min(capacity, available)
    max_fit = shape.max_ctx_for(budget, fixed, kvb)
    if ctx == 0:
        ctx = shape.n_ctx_train
        if ctx > max_fit:
            ctx = max(4096, (max_fit // 4096) * 4096)
            con.print(f"[yellow]native context {shape.n_ctx_train:,} does not fit in what's free; using {ctx:,}[/]")
    kv_total = shape.kv_bytes(ctx, kvb)
    total = fixed + kv_total
    color = "green" if total <= budget else "red"
    con.print(
        f"context [bold]{ctx:,}[/] (native max {shape.n_ctx_train:,}) · "
        f"model {fmt_bytes(size)}"
        + (f" + draft {fmt_bytes(draft_size)}" if draft_size else "")
        + (f" + vision {fmt_bytes(mmproj_size)}" if mmproj_size else "")
        + f" + KV {fmt_bytes(kv_total)} ({kv}, {per / 1024:.0f} KiB/tok"
        + (f" + {fmt_bytes(shape.kv_fixed_bytes(kvb))} in {shape.n_swa_layers} sliding-window layers" if shape.swa_window else "")
        + f") = [{color}]{fmt_bytes(total)}[/]"
        + f" of {fmt_bytes(available)} free now ({fmt_bytes(capacity)} capacity)"
    )
    if available < capacity - 2 * GiB:
        _in_use_line()
    if total > budget:
        con.print(f"[red]that exceeds what is free[/] — stop other models, lower --ctx (max ≈ {max_fit:,}), or use --kv q8_0")
    return ctx


@contextmanager
def _start(
    lm: S.LocalModel, *, ctx: int, kv: str, vision: bool, thinking: bool, effort: str | None,
    mtp: bool = True, draft: bool = True, debug: bool = False,
    port: int = 0, parallel: int = 1, api_key: str | None = None, extra: list[str] | None = None,
) -> Iterator[server.Session]:
    """A server for this run only: `with _start(...) as srv:` kills it on the way out.

    `port`, `parallel`, `api_key` and `extra` exist for `lethe serve`; every other caller gets a free
    port, one slot and a fresh random key.
    """
    shape = _shape(lm.path)
    mmproj = _find_sidecar(lm, "mmproj") if vision else None
    if vision and not mmproj:
        raise typer.BadParameter("no mmproj found beside the model or in its model repository", param_hint="--vision")
    draft_path = _find_sidecar(lm, "draft") if (mtp and draft) else None
    draft_size = draft_path.stat().st_size if draft_path else 0
    ctx = _plan_ctx(shape, lm.size, ctx, kv, draft_size, mmproj.stat().st_size if mmproj else 0)
    if mtp and not shape.has_mtp and not draft_path:
        con.print("[yellow]model has no MTP head; disabling speculative decoding[/]")
        mtp = False
    cfg = server.ServeConfig(
        model=lm.path, n_ctx=ctx, draft=draft_path, mtp=mtp, mmproj=mmproj, kv_type=kv,
        thinking=thinking, reasoning_effort=effort, debug=debug, api_key=api_key or server.new_api_key(),
        port=port, parallel=parallel, extra=list(extra or []),
    )
    con.print(
        f"speculative: {'FastMTP draft ' + draft_path.name if draft_path else ('embedded MTP' if mtp else 'off')}"
        + (f" (depth {cfg.draft_n_max})" if mtp else "")
        + f" · thinking {'on' + (' ' + effort if effort else '') if thinking else 'off'}"
        + (" · vision" if mmproj else "")
    )
    status = con.status("loading model…")
    status.start()
    t0 = time.time()
    try:
        with server.Session(cfg, on_wait=lambda s: status.update(f"loading model… {s:.0f}s")) as session:
            status.stop()
            con.print(
                f"[green]ready[/] in {time.time() - t0:.1f}s · "
                + ("[yellow]llama-server output on the terminal[/]" if debug else "[dim]traceless: no log, no web UI, no slots endpoint; the model is unloaded when you quit[/]")
            )
            yield session
    finally:
        status.stop()


# ----------------------------------------------------------------------------- commands


@app.command()
def doctor():
    """Verify the GPU build, memory limits, and that nothing is on disk that shouldn't be."""
    m = machine()
    con.print(f"[bold]{m.chip}[/] — {m.ncpu} CPU cores, RAM {fmt_bytes(m.total_ram)}")
    hint = " (raise with: sudo sysctl iogpu.wired_limit_mb=<MB>)" if m.is_apple_silicon else ""
    con.print(f"GPU memory limit: {fmt_bytes(m.gpu_wired_limit)}{hint}")
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
        if m.is_apple_silicon and llama_bin("libggml-metal.dylib").exists():
            con.print("[green]✓[/] Metal backend linked (libggml-metal)")
        elif not m.is_apple_silicon and m.gpu_wired_limit > 0:
            con.print("[green]✓[/] GPU backend reports usable memory")
        else:
            con.print("[red]✗[/] No GPU memory detected — check drivers and selected backend")
            ok = False
        dev = subprocess.run([str(llama_bin("llama-server")), "--list-devices"], capture_output=True, text=True)
        txt = (dev.stdout + dev.stderr).strip()
        if txt:
            con.print("  " + "\n  ".join(txt.splitlines()[:4]))
    head = subprocess.run(["git", "-C", str(llama_dir()), "log", "-1", "--format=%h %s"], capture_output=True, text=True).stdout.strip()
    con.print(f"llama.cpp: {head}")
    con.print(f"local models: {len(S.local_models())} in {models_dir()}" + (f" · default {config.default_model()}" if config.default_model() else ""))
    if server.log_path().exists():
        con.print(f"[yellow]![/] {server.log_path()} exists (from an old --log run) — remove with: lethe traces --delete")
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
            f"[dim]{shared.name}: {shared.n_layer} layers, {shared.n_attn_layers} with growing KV"
            + (f" + {shared.n_swa_layers} sliding-window ({shared.swa_window} tok)" if shared.swa_window else "")
            + f"{' (hybrid: interval ' + str(shared.extra.get('full_attention_interval')) + ')' if not shared.swa_window and shared.n_attn_layers != shared.n_layer else ''}, "
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
def chat(
    cli_context: typer.Context,
    model: Optional[str] = typer.Argument(None, help=MODEL_HELP + " (default: lethe.toml)"),
    ctx: int = typer.Option(0, "--ctx", "-c", help="Context tokens (0 = native max if it fits)"),
    kv: str = typer.Option("f16", help="KV cache type: f16|q8_0|q4_0"),
    thinking: bool = typer.Option(True, help="Thinking mode (server-side default + sampler preset)"),
    show_thinking: bool = typer.Option(True, help="Stream the model's reasoning (dimmed)"),
    vision: bool = typer.Option(False, help="Load the vision projector for images and sampled video frames; enable the view_image tool unless --no-tools"),
    video_frames: int = typer.Option(16, min=1, max=128, help="Maximum sampled frames per video (up to 1 frame/second; no audio)"),
    system: Optional[str] = typer.Option(None, help="System prompt"),
    file: list[Path] = typer.Option([], "--file", "-f", help="Attach text/PDF/image/video to the first message (repeatable; media needs --vision)"),
    max_tokens: Optional[int] = typer.Option(None, help="Cap generation length per turn"),
    temp: Optional[float] = typer.Option(None, help="Override sampler temperature"),
    top_p: Optional[float] = typer.Option(None, help="Override top_p"),
    top_k: Optional[int] = typer.Option(None, help="Override top_k"),
    min_p: Optional[float] = typer.Option(None, help="Override min_p"),
    no_system: bool = typer.Option(False, "--no-system", help="No system prompt at all (default: a short one grounding the model: local, private, this hardware)"),
    web: bool = typer.Option(False, "--web", help="Shorthand for --tools web,fetch: let the model search the web and read pages"),
    tools: Optional[str] = typer.Option("shell", "--tools", help="Model tools (default: shell, Python/Bash, SSH, and file tools). Bare --tools = shell; or name them: web,fetch,shell. Composes with --web"),
    no_tools: bool = typer.Option(False, "--no-tools", help="Disable all model tools for this invocation"),
    shell: bool = typer.Option(False, "--shell", help="Enable the shell tool (enabled by default); commands require approval unless --auto-approve all"),
    shell_dir: Optional[Path] = typer.Option(None, "--shell-dir", help="Shell workspace or, in full mode, starting directory (default: current directory); requires --shell or --tools shell"),
    shell_mode: ShellMode = typer.Option(ShellMode.FULL, "--shell-mode", help="full (default): run as you, with writes, network, SSH, inherited PATH/environment and credentials. read-only: offline macOS sandbox"),
    searx: Optional[str] = typer.Option(None, "--searx", help="Back web search with this SearXNG instance URL instead of the ddgs metasearch"),
    no_instructions: bool = typer.Option(False, "--no-instructions", help="Do not load LETHE.md instruction files (global: the lethe repo's LETHE.md; project: nearest LETHE.md or AGENTS.md from the launch directory up)"),
    tools_confirm: bool = typer.Option(False, "--tools-confirm", help="Ask y/n before every tool call (shell commands always ask)"),
    auto_approve: Optional[str] = typer.Option(None, "--auto-approve", help="OPT-IN: skip the y/n prompt. files = read_file/view_image/write_file/edit_file run unasked (shell still asks); all = shell commands too. Calls and diffs are still printed"),
    save: bool = typer.Option(False, "--save", help="OPT-IN: save this conversation encrypted (asks for a passphrase; with --resume, keeps saving to that id)"),
    resume: Optional[str] = typer.Option(None, "--resume", help="Continue a saved conversation by id (see lethe conversations). Reads only: new turns are saved only with --save or /save"),
    effort: Optional[str] = typer.Option(None, "--effort", help="Reasoning effort: low|medium|xhigh (Qwen3.8; the template's default is xhigh)"),
    debug_server: bool = typer.Option(False, "--debug-server", help="Show llama-server's own output on the terminal (nothing is written to disk)"),
    yes: bool = typer.Option(False, "-y", "--yes", help="Download the model without asking if it isn't local"),
):
    """Interactive chat. Loads the model for this session and unloads it when you quit."""
    for f in file:
        if not f.is_file():
            err.print(f"[red]no such file: {f}[/]")
            raise typer.Exit(1)
        if (is_image(f) or is_video(f)) and not vision:
            err.print("image/video attachments require --vision", style="red")
            raise typer.Exit(1)
    if no_tools and (web or shell or tools not in (None, "shell")):
        raise typer.BadParameter("--no-tools cannot be combined with tool-enabling options")
    tool_names = [] if no_tools else [t.strip() for t in (tools or "").split(",") if t.strip()]
    if web:
        tool_names.append("web")
    if shell:
        tool_names.append("shell")
    if shell_dir is not None and "shell" not in tool_names:
        err.print("--shell-dir requires --tools or --shell", style="red")
        raise typer.Exit(1)
    if auto_approve is not None and auto_approve not in T.AUTO_APPROVE_SCOPES:
        err.print("--auto-approve takes files or all", style="red")
        raise typer.Exit(1)
    if auto_approve and tools_confirm:
        err.print("--auto-approve and --tools-confirm contradict each other", style="red")
        raise typer.Exit(1)
    tool_registry: dict = {}
    if tool_names:
        try:
            tool_registry = T.build_registry(
                tool_names, searx=searx, shell_dir=shell_dir,
                shell_mode=shell_mode if "shell" in tool_names else ShellMode.READ_ONLY,
            )
        except KeyError as e:
            err.print(f"unknown tool {e} — available: web, fetch, shell", style="red", markup=False)
            raise typer.Exit(1)
        except (OSError, ValueError, RuntimeError) as e:
            err.print(str(e), style="red", markup=False)
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
        # Resuming only reads. Saving stays opt-in: --save (same id and passphrase) or /save in the chat.
    elif save:
        from .repl import ask_passphrase

        passphrase = ask_passphrase(confirm=True)
        if not passphrase:
            raise typer.Exit(1)

    model = model or (resumed.model if resumed else None) or config.default_model()
    if not model:
        err.print("[red]no model given and no default set[/] — lethe owner/repo:QUANT, or lethe default …")
        raise typer.Exit(1)
    lm = _resolve_or_offer(model, yes)
    overrides = {k: v for k, v in {"temperature": temp, "top_p": top_p, "top_k": top_k, "min_p": min_p}.items() if v is not None}

    from .repl import ReplConfig, run

    with _start(lm, ctx=ctx, kv=kv, vision=vision, thinking=thinking, effort=effort, debug=debug_server) as srv:
        p = server.props(srv.url, srv.api_key)
        n_ctx = int(p.get("default_generation_settings", {}).get("n_ctx", 0))
        has_vision = bool((p.get("modalities") or {}).get("vision"))
        if has_vision and not no_tools:
            tool_registry["view_image"] = T.build_image_tool(
                shell_dir if shell_dir is not None else Path.cwd(), confined=shell_mode != ShellMode.FULL,
            )
        sh = _shape(lm.path)
        if system is None and not no_system:
            system = T.sync_system_text(_default_system(lm, sh), tool_registry)
        if not no_instructions and not no_system:
            from . import instructions

            block, loaded = instructions.load(shell_dir if shell_dir is not None else Path.cwd())
            if system is not None:
                system = instructions.apply(system, block)
            if resumed:  # a saved conversation carries the instructions it was started with; use today's
                for m in resumed.messages:
                    if m["role"] == "system" and isinstance(m["content"], str):
                        m["content"] = instructions.apply(m["content"], block)
            if loaded:
                err.print("instructions: " + " · ".join(f"{p} ({layer}, ~{n // 4:,} tok)" for layer, p, n in loaded),
                          style="dim", markup=False, highlight=False)
        run(ReplConfig(
            url=srv.url, api_key=srv.api_key, n_ctx=n_ctx, model_spec=lm.spec, profile=profile_for(sh),
            thinking=thinking, show_thinking=show_thinking, system=system, max_tokens=max_tokens,
            sampling_overrides=overrides, attachments=list(file), vision=has_vision, video_frames=video_frames,
            save=save, resume=resumed, passphrase=passphrase,
            tools=tool_registry, tools_confirm=tools_confirm, searx=searx, auto_approve=auto_approve,
            shell_dir=(shell_dir if shell_dir is not None else Path.cwd()).expanduser().resolve(),
            resume_args=_resume_args(cli_context),
        ))


def _sampler_flags(sampling: dict) -> list[str]:
    """The profile's sampler as llama-server defaults, for requests that do not set their own."""
    out: list[str] = []
    for key, flag in SAMPLER_FLAGS.items():
        if key in sampling:
            out += [flag, str(sampling[key])]
    return out


def _port_free(port: int) -> bool:
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


@app.command()
def serve(
    model: Optional[str] = typer.Argument(None, help=MODEL_HELP + " (default: lethe.toml)"),
    port: int = typer.Option(..., "--port", min=1024, max=65535, help="Port on 127.0.0.1 (required: the programs you serve need a stable address)"),
    parallel: int = typer.Option(1, "--parallel", min=1, max=64, help="Requests served at once; they share the context"),
    ctx: int = typer.Option(0, "--ctx", "-c", help="Context tokens for all slots together (0 = native max if it fits)"),
    kv: str = typer.Option("f16", help="KV cache type: f16|q8_0|q4_0"),
    thinking: bool = typer.Option(True, help="Thinking mode by default (a request may override it with chat_template_kwargs)"),
    effort: Optional[str] = typer.Option(None, "--effort", help="Reasoning effort: low|medium|xhigh (Qwen3.8; the template's default is xhigh)"),
    debug_server: bool = typer.Option(False, "--debug-server", help="Show llama-server's own output on the terminal (nothing is written to disk)"),
    yes: bool = typer.Option(False, "-y", "--yes", help="Download the model without asking if it isn't local"),
):
    """Serve the model to other programs on this machine until you press Ctrl-C.

    Foreground only: the model is loaded when this command starts and unloaded when it ends, like a chat. Always bound to 127.0.0.1, OpenAI-compatible at /v1. Requests must send the key you put in LETHE_API_KEY as `Authorization: Bearer …`; lethe never prints, stores, or takes it as an argument. Nothing is logged.
    """
    key = os.environ.get(SERVE_KEY_ENV, "")
    if len(key) < SERVE_KEY_MIN:
        err.print(f"[red]set {SERVE_KEY_ENV} to a secret of at least {SERVE_KEY_MIN} characters[/] — "
                  "clients send it as 'Authorization: Bearer …'; lethe never prints or stores it")
        raise typer.Exit(1)
    if kv not in KV_BYTES:
        raise typer.BadParameter(f"choose one of {', '.join(KV_BYTES)}", param_hint="--kv")
    if not _port_free(port):
        err.print(f"[red]port {port} is already in use on 127.0.0.1[/] — choose another; lethe never touches servers it did not start")
        raise typer.Exit(1)
    model = model or config.default_model()
    if not model:
        err.print("[red]no model given and no default set[/] — lethe serve owner/repo:QUANT --port …, or lethe default …")
        raise typer.Exit(1)
    lm = _resolve_or_offer(model, yes)
    sampling = profile_for(_shape(lm.path)).sampling(thinking)
    try:
        with _start(lm, ctx=ctx, kv=kv, vision=False, thinking=thinking, effort=effort, debug=debug_server,
                    port=port, parallel=parallel, api_key=key, extra=_sampler_flags(sampling)) as srv:
            p = server.props(srv.url, srv.api_key)
            n_ctx = int(p.get("default_generation_settings", {}).get("n_ctx", 0))
            con.print(
                f"serving [bold]{lm.spec}[/] at [bold]{srv.url}/v1[/] (OpenAI-compatible) · "
                f"{parallel} slot{'s' if parallel != 1 else ''} · context per request {n_ctx:,} · "
                f"thinking {'on' if thinking else 'off'} · default sampler {describe(sampling)} · key from ${SERVE_KEY_ENV}"
            )
            con.print("[dim]foreground only: Ctrl-C (or closing this terminal) stops the server and unloads the model[/]")
            code = srv.wait()
    except KeyboardInterrupt:
        con.print("[dim]stopped; the model is unloaded[/]")
        return
    err.print(f"[red]llama-server exited on its own with {code}[/]; run with --debug-server to see its output")
    raise typer.Exit(1)


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
        con.print("[dim]no saved conversations (saving is opt-in: lethe --save)[/]")
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
    con.print("[dim]resume: lethe --resume <id> · delete: lethe conversations --delete <id>[/]")


@app.command()
def bench(
    model: Optional[str] = typer.Argument(None, help=MODEL_HELP + " (default: lethe.toml)"),
    prompt: str = typer.Option("2048", "-p", help="llama-bench: prompt lengths (comma list)"),
    gen: str = typer.Option("128", "-n", help="llama-bench: generation lengths (comma list)"),
    depth: str = typer.Option("0", "-d", help="Context depths to test at, e.g. 0,32768"),
    kv: str = typer.Option("f16", help="KV cache type"),
    server_mode: bool = typer.Option(False, "--server", help="Benchmark through llama-server instead: measures real MTP/FastMTP speed and acceptance"),
    ctx: int = typer.Option(0, min=0, help="Server benchmark context allocation (0: largest depth + 8192)"),
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

        if ctx and ctx < max(depths) + 8192:
            raise typer.BadParameter("--ctx must be at least the largest depth + 8192")
        run_server_bench(lm, depths=depths, kv=kv, launch=_start, ctx=ctx)
        return
    argv = [
        str(llama_bin("llama-bench")), "-m", str(lm.path), "-ngl", "99", "-fa", "1",
        "-p", prompt, "-n", gen, "-d", depth, "-ctk", kv, "-ctv", kv, *extra,
    ]
    con.print("[dim]$ " + " ".join(argv) + "[/]")
    con.print("[dim]note: llama-bench cannot use MTP; its tg is the floor. Use --server for the real number.[/]")
    raise typer.Exit(subprocess.call(argv))


@app.command()
def traces(delete: bool = typer.Option(False, "--delete", help="Delete what is found (saved conversations included!)")):
    """List (and optionally delete) anything lethe may have written to disk beyond model files."""
    found: list[tuple[Path, str]] = []
    if server.log_path().exists():
        found.append((server.log_path(), "llama-server log from an old --log run"))
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


def _normalize_argv(argv: list[str]) -> list[str]:
    """Bare `lethe [chat args…]` means `lethe chat …` — chat is what you run 95% of the time.

    Bare `--tools` (alias `--tool`) means the local shell tool, so `--web --tools` is internet + shell;
    `--tools web,fetch,shell` still picks tools by name.
    """
    commands = typer.main.get_command(app).commands  # type: ignore[union-attr]
    if not argv or (argv[0] not in commands and argv[0] != "--help"):
        argv = ["chat", *argv]
    out: list[str] = []
    for i, a in enumerate(argv):
        out.append("--tools" if a == "--tool" else a)
        if a in ("--tools", "--tool"):
            nxt = argv[i + 1] if i + 1 < len(argv) else ""
            if not nxt or not set(nxt.split(",")) <= {"web", "fetch", "shell"}:
                out.append("shell")
    return out


def main() -> None:
    app(_normalize_argv(sys.argv[1:]))


if __name__ == "__main__":
    main()
