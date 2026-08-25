"""lethe — local llama.cpp runner with explicit throughput and context."""

from __future__ import annotations

import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import hub, server
from .chat import INSTRUCT_SAMPLING, THINKING_SAMPLING, Delta, TurnStats, stream_chat
from .ggufmeta import ModelShape, parse_local, quant_from_filename, shape_from_kv
from .paths import find_ggufs, llama_bin, llama_dir, models_dir
from .sysinfo import GiB, fmt_bytes, machine

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode="rich")
con = Console()
err = Console(stderr=True)

KV_BYTES = {"f16": 2.0, "bf16": 2.0, "q8_0": 1.0625, "q4_0": 0.5625}
CTX_COLUMNS = [32_768, 131_072]


# ----------------------------------------------------------------------------- helpers


def _sidecar(name: str) -> bool:
    n = name.lower()
    return n.startswith("mmproj") or "fastmtp" in n or n.startswith("imatrix") or "-draft" in n


def _local_models() -> list[tuple[Path, list[Path], int]]:
    """Group local GGUFs into (first_shard, all_shards, total_size)."""
    groups: dict[Path, list[Path]] = {}
    for p in find_ggufs():
        m = hub.SHARD_RE.match(p.name)
        key = p.with_name(m["base"] + ".gguf") if m else p
        groups.setdefault(key, []).append(p)
    out = []
    for key, files in groups.items():
        files.sort()
        out.append((files[0], files, sum(f.stat().st_size for f in files)))
    return sorted(out, key=lambda t: t[2])


def _resolve_model(spec: str) -> Path:
    p = Path(spec).expanduser()
    if p.is_file():
        return p
    cands = [(f, fs, sz) for f, fs, sz in _local_models() if not _sidecar(f.name)]
    hits = [c for c in cands if spec.lower() in str(c[0].relative_to(models_dir())).lower()]
    if len(hits) == 1:
        return hits[0][0]
    if not hits:
        err.print(f"[red]no local model matches '{spec}'[/]. Try [bold]lethe models[/].")
        raise typer.Exit(1)
    err.print(f"[red]'{spec}' is ambiguous:[/]")
    for h in hits:
        err.print(f"  {h[0].relative_to(models_dir())}")
    raise typer.Exit(1)


def _find_draft(model: Path) -> Path | None:
    for p in model.parent.glob("*.gguf"):
        if "fastmtp" in p.name.lower():
            return p
    return None


def _find_mmproj(model: Path) -> Path | None:
    for p in model.parent.glob("mmproj*.gguf"):
        return p
    return None


def _shape(path: Path) -> ModelShape:
    return shape_from_kv(parse_local(path), path.name)


def _fit_row(shape: ModelShape | None, size: int, budget: int, kvb: float) -> dict:
    if not shape or not shape.n_layer:
        return {"kv_tok": "?", "ctx_cols": ["?"] * len(CTX_COLUMNS), "native": "?", "max_ctx": "?", "fits": "?"}
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
        "fits": "[green]yes[/]" if size < budget else "[red]no[/]",
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
    return m.usable_for_model


# ----------------------------------------------------------------------------- commands


@app.command()
def doctor():
    """Verify the Metal build, memory limits, and local state."""
    m = machine()
    con.print(f"[bold]{m.chip}[/] — {m.ncpu} CPU cores, RAM {fmt_bytes(m.total_ram)}")
    con.print(f"GPU-wired limit: {fmt_bytes(m.gpu_wired_limit)} (raise with: sudo sysctl iogpu.wired_limit_mb=<MB>)")
    con.print(f"Model+KV budget: {fmt_bytes(m.usable_for_model)}")
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
        con.print(f"  {(out.stderr or out.stdout).strip().splitlines()[0]}")
        otool = subprocess.run(["otool", "-L", str(llama_bin("llama-server"))], capture_output=True, text=True).stdout
        metal_lib = llama_bin("libggml-metal.dylib")
        if "libggml-metal" in otool or metal_lib.exists():
            con.print("[green]✓[/] Metal backend linked (libggml-metal)")
        else:
            con.print("[red]✗[/] Metal backend NOT linked — rebuild with scripts/build-llama.sh")
            ok = False
        # ask the binary what devices it sees
        dev = subprocess.run([str(llama_bin("llama-server")), "--list-devices"], capture_output=True, text=True)
        txt = (dev.stdout + dev.stderr).strip()
        if txt:
            con.print("  " + "\n  ".join(txt.splitlines()[:6]))
            if "Metal" not in txt and "MTL" not in txt:
                con.print("[yellow]![/] no Metal device reported")
    try:
        head = subprocess.run(["git", "-C", str(llama_dir()), "log", "-1", "--format=%h %s"], capture_output=True, text=True).stdout.strip()
        con.print(f"llama.cpp: {head}")
    except Exception:
        pass
    n = len([x for x in _local_models() if not _sidecar(x[0].name)])
    con.print(f"local models: {n} in {models_dir()}")
    rec = server.running_server()
    if rec:
        con.print(
            f"[green]server running[/] pid {rec['pid']} at {rec['url']} — {Path(rec['model']).name}"
            + (" [yellow](logging ON)[/]" if rec["logging"] else " [dim](traceless)[/]")
        )
    if server.log_path().exists():
        con.print(f"[yellow]![/] {server.log_path()} exists (from a --log run) — remove with: lethe traces --delete")
    else:
        con.print("[green]✓[/] no logs on disk")
    raise typer.Exit(0 if ok else 1)


@app.command()
def browse(
    repo: str = typer.Argument(..., help="HF repo id, e.g. HauhauCS/Qwen3.8-27B-...-GGUF"),
    kv: str = typer.Option("f16", help="KV cache type used for the memory columns: f16|q8_0|q4_0"),
    each: bool = typer.Option(False, help="Parse every file's header (default: one per repo, shared shape)"),
):
    """List a Hub repo's GGUFs with size, quant, native context, KV cost and what fits on this machine."""
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
            if _sidecar(g.base):
                continue
            if each or shared is None:
                hub.fetch_shape(g)
                if g.shape and g.shape.n_layer and shared is None:
                    shared = g.shape
            if g.shape is None and shared is not None:
                g.shape = replace(shared, quant=quant_from_filename(g.base) or shared.quant)
    t = Table(title=repo, expand=False, pad_edge=False)
    t.add_column("file")
    _ctx_headers(t, kv)
    for g in ggufs:
        s = g.shape
        fit = _fit_row(s, g.size, budget, kvb)
        quant = (s.quant if s and s.quant else "") or "-"
        if _sidecar(g.base):
            t.add_row(g.base, fmt_bytes(g.size), "[dim]sidecar[/]", "", "", *[""] * len(CTX_COLUMNS), "", "")
            continue
        t.add_row(
            g.base + (f" [dim]({len(g.files)} shards)[/]" if len(g.files) > 1 else ""),
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
            f"{', MoE ' + str(shared.extra.get('expert_used_count')) + '/' + str(shared.extra.get('expert_count')) if shared.extra.get('expert_count') else ''}[/]"
        )
    con.print("[dim]mem = file size + KV cache; the real footprint adds ~1–3 GiB of compute buffers. Green fits the budget.[/]")


@app.command()
def pull(
    repo: str = typer.Argument(..., help="HF repo id"),
    quant: Optional[str] = typer.Option(None, "--quant", "-q", help="Quant to fetch, e.g. Q8_K_P, Q4_K_M, IQ4_XS"),
    file: list[str] = typer.Option([], "--file", "-f", help="Exact filename(s) to fetch (repeatable)"),
    sidecars: bool = typer.Option(True, help="Also fetch mmproj / FastMTP draft / patch files"),
):
    """Download GGUF(s) from the Hub into ./models/<owner>--<repo>/."""
    with con.status(f"listing {repo}…"):
        ggufs = hub.list_repo_ggufs(repo)
    want: list[str] = list(file)
    if quant:
        hits = [g for g in hub.match_quant(ggufs, quant) if not _sidecar(g.base)]
        if not hits:
            err.print(f"[red]no file matches quant {quant}[/]. Available: " + ", ".join(g.base for g in ggufs))
            raise typer.Exit(1)
        if len(hits) > 1:
            err.print("[red]quant is ambiguous:[/] " + ", ".join(g.base for g in hits))
            raise typer.Exit(1)
        want += hits[0].files
    if not want:
        err.print("[red]give --quant or --file[/]")
        raise typer.Exit(1)
    if sidecars:
        for g in ggufs:
            if _sidecar(g.base) and g.size < 3 * GiB:
                want += g.files
        from huggingface_hub import HfApi

        for s in HfApi().model_info(repo).siblings or []:
            if s.rfilename.endswith((".patch", "SHA256SUMS", "MANIFEST.json")):
                want.append(s.rfilename)
    want = list(dict.fromkeys(want))
    total = sum(g.size for g in ggufs for f in g.files if f in want)
    con.print(f"downloading {len(want)} files (~{fmt_bytes(total)}) → {hub.local_dir_for(repo)}")
    for f in want:
        con.print(f"  {f}")
    paths = hub.download(repo, want)
    con.print(f"[green]done[/] — {len(paths)} files")


@app.command()
def models(kv: str = typer.Option("f16", help="KV cache type for memory columns: f16|q8_0|q4_0")):
    """List local models with native context, KV cost and max context that fits."""
    budget = _budget_line(kv)
    kvb = KV_BYTES[kv]
    rows = _local_models()
    if not rows:
        con.print(f"no models in {models_dir()} — try [bold]lethe pull <repo> -q <quant>[/]")
        return
    t = Table(expand=False, pad_edge=False)
    t.add_column("model (models/…)")
    _ctx_headers(t, kv)
    for first, files, size in rows:
        rel = str(first.relative_to(models_dir()))
        if _sidecar(first.name):
            t.add_row(rel, fmt_bytes(size), "[dim]sidecar[/]", "", "", *[""] * len(CTX_COLUMNS), "", "")
            continue
        s = _shape(first)
        fit = _fit_row(s, size, budget, kvb)
        t.add_row(
            rel + (f" [dim]({len(files)} shards)[/]" if len(files) > 1 else ""),
            fmt_bytes(size), s.quant or "-", f"{s.n_ctx_train:,}",
            fit["kv_tok"], *fit["ctx_cols"], fit["native"], fit["max_ctx"],
        )
    con.print(t)


def _plan_ctx(shape: ModelShape, size: int, ctx: int, kv: str, draft_size: int) -> int:
    """Pick/validate a context size and print the memory plan."""
    m = machine()
    kvb = KV_BYTES[kv]
    per = shape.kv_bytes_per_token(kvb)
    fixed = size + draft_size
    max_fit = shape.max_ctx_for(m.usable_for_model, fixed, kvb)
    if ctx == 0:
        ctx = shape.n_ctx_train
        if ctx > max_fit:
            ctx = max(4096, (max_fit // 4096) * 4096)
            con.print(f"[yellow]native context {shape.n_ctx_train:,} does not fit; using {ctx:,}[/]")
    kv_total = per * ctx
    total = fixed + kv_total
    color = "green" if total <= m.usable_for_model else "red"
    con.print(
        f"context [bold]{ctx:,}[/] (native max {shape.n_ctx_train:,}) · "
        f"model {fmt_bytes(size)}"
        + (f" + draft {fmt_bytes(draft_size)}" if draft_size else "")
        + f" + KV {fmt_bytes(kv_total)} ({kv}, {per / 1024:.0f} KiB/tok) = [{color}]{fmt_bytes(total)}[/] of {fmt_bytes(m.usable_for_model)} budget"
    )
    if total > m.usable_for_model:
        con.print(f"[red]that exceeds the GPU-wired budget[/] — lower --ctx (max ≈ {max_fit:,}) or use --kv q8_0")
    return ctx


@app.command()
def serve(
    model: str = typer.Argument(..., help="Local model: path, or substring of models/… (see `lethe models`)"),
    ctx: int = typer.Option(0, "--ctx", "-c", help="Context tokens (0 = native max if it fits)"),
    kv: str = typer.Option("f16", help="KV cache type: f16|q8_0|q4_0"),
    mtp: bool = typer.Option(True, help="MTP speculative decoding (embedded head)"),
    draft: bool = typer.Option(True, help="Use a FastMTP draft sidecar if one sits next to the model"),
    draft_n: int = typer.Option(3, help="Draft depth (--spec-draft-n-max)"),
    vision: bool = typer.Option(False, help="Load the mmproj vision projector if present"),
    thinking: bool = typer.Option(True, help="Thinking/reasoning mode on"),
    effort: Optional[str] = typer.Option(None, help="Reasoning effort: low|medium|xhigh"),
    port: int = typer.Option(server.DEFAULT_PORT),
    no_mmap: bool = typer.Option(False, help="Read the whole model into memory instead of mmap"),
    foreground: bool = typer.Option(False, "--fg", help="Run in the foreground with llama-server's output on the terminal"),
    log: bool = typer.Option(False, "--log", help="OPT-IN: keep llama-server's log in .lethe/server.log (off by default: traceless)"),
    extra: list[str] = typer.Option([], "--arg", help="Extra llama-server args (repeatable)"),
):
    """Launch llama-server for a local model with Metal, flash-attn and MTP, and print the memory plan."""
    path = _resolve_model(model)
    shape = _shape(path)
    size = sum(f.stat().st_size for f in path.parent.glob(path.name.replace(".gguf", "*.gguf")) if f.is_file())
    draft_path = _find_draft(path) if (mtp and draft) else None
    draft_size = draft_path.stat().st_size if draft_path else 0
    ctx = _plan_ctx(shape, size, ctx, kv, draft_size)
    if mtp and not shape.has_mtp and not draft_path:
        con.print("[yellow]model has no MTP head; disabling speculative decoding[/]")
        mtp = False
    cfg = server.ServeConfig(
        model=path, n_ctx=ctx, draft=draft_path, mtp=mtp, draft_n_max=draft_n,
        mmproj=_find_mmproj(path) if vision else None, kv_type=kv, thinking=thinking,
        reasoning_effort=effort, port=port, no_mmap=no_mmap, log=log, extra=list(extra),
    )
    con.print(
        f"speculative: {'FastMTP draft ' + draft_path.name if draft_path else ('embedded MTP' if mtp else 'off')}"
        + (f" (depth {draft_n})" if mtp else "")
        + f" · thinking {'on' + (' ' + effort if effort else '') if thinking else 'off'}"
    )
    if server.running_servers():
        con.print("stopping previous server…")
        server.stop()
    if foreground:
        con.print("[dim]$ " + " ".join(cfg.argv()) + "[/]")
        raise typer.Exit(subprocess.call(cfg.argv()))
    t0 = time.time()
    with con.status("loading model…") as st:
        rec = server.start(cfg, on_wait=lambda s: st.update(f"loading model… {s:.0f}s"))
    p = server.props(rec["url"])
    con.print(
        f"[green]ready[/] in {time.time() - t0:.1f}s at {rec['url']} · n_ctx {p.get('default_generation_settings', {}).get('n_ctx', '?'):,} · "
        + (f"[yellow]logging to {server.log_path()}[/]" if log else "[dim]traceless: no log, no web UI, no slots endpoint[/]")
    )


@app.command()
def stop():
    """Stop the running llama-server."""
    n = server.stop()
    con.print(f"[green]stopped {n}[/]" if n else "no server running")


def _make_reader():
    """Return a read(prompt) callable.

    On a TTY: prompt_toolkit — bracketed paste makes a multi-line paste ONE message (Enter sends),
    Alt+Enter inserts a newline, Up/Down recall earlier inputs from an in-memory history only.
    Mouse support stays off so the terminal's native text selection/copy keeps working.
    Piped stdin: plain input(), one message per line.
    """
    import atexit
    import sys

    if not sys.stdin.isatty():
        return lambda prompt: input(prompt)

    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import ANSI
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys

    # Ctrl+Enter (and Shift+Enter) only exist as distinct keys under the kitty keyboard protocol
    # (iTerm2 3.5+, Ghostty, kitty, WezTerm…) or xterm's modifyOtherKeys. Map every known encoding
    # onto ControlJ, which we bind to "insert newline". Plain Enter (\r) still sends.
    ANSI_SEQUENCES.update({
        "\x1b[13;5u": Keys.ControlJ,      # kitty: Ctrl+Enter
        "\x1b[13;2u": Keys.ControlJ,      # kitty: Shift+Enter
        "\x1b[27;5;13~": Keys.ControlJ,   # xterm modifyOtherKeys: Ctrl+Enter
        "\x1b[27;2;13~": Keys.ControlJ,   # xterm modifyOtherKeys: Shift+Enter
        "\x1b[27u": Keys.Escape,          # kitty: bare Esc while the protocol is on
    })
    kb = KeyBindings()

    @kb.add("c-j")  # Ctrl+Enter / Ctrl+J / raw \n
    @kb.add("escape", "enter")  # Alt/Option+Enter
    def _newline(event):
        event.current_buffer.insert_text("\n")

    @kb.add("enter")
    def _enter(event):
        buf = event.current_buffer
        if buf.document.text_before_cursor.endswith("\\"):  # trailing backslash: newline (Claude Code style)
            buf.delete_before_cursor(1)
            buf.insert_text("\n")
        else:
            buf.validate_and_handle()

    # Ask the terminal for kitty "disambiguate" mode so Ctrl+Enter is distinguishable; terminals
    # that don't support it ignore the sequence. Popped again at exit.
    sys.stdout.write("\x1b[>1u")
    sys.stdout.flush()
    atexit.register(lambda: (sys.stdout.write("\x1b[<u"), sys.stdout.flush()))

    session = PromptSession(history=InMemoryHistory(), key_bindings=kb, mouse_support=False, enable_history_search=False)

    def read(prompt: str) -> str:
        return session.prompt(ANSI(prompt), prompt_continuation="   … ")

    return read


def _pbcopy(text: str) -> bool:
    try:
        subprocess.run(["pbcopy"], input=text.encode(), check=True)
        return True
    except Exception:
        return False


@app.command()
def chat(
    model: Optional[str] = typer.Argument(None, help="Local model (default: whatever server is running)"),
    ctx: int = typer.Option(0, "--ctx", "-c", help="Context tokens (0 = native max if it fits)"),
    kv: str = typer.Option("f16", help="KV cache type: f16|q8_0|q4_0"),
    thinking: bool = typer.Option(True, help="Thinking mode (server-side default + sampler preset)"),
    show_thinking: bool = typer.Option(True, help="Stream the model's reasoning (dimmed)"),
    system: Optional[str] = typer.Option(None, help="System prompt"),
    max_tokens: Optional[int] = typer.Option(None, help="Cap generation length per turn"),
    port: int = typer.Option(server.DEFAULT_PORT),
):
    """Interactive chat. Starts the server if needed. Prints throughput + context stats after every turn."""
    rec = server.running_server()
    if model is not None:
        path = _resolve_model(model)
        if not rec or Path(rec["model"]) != path:
            serve(
                model=model, ctx=ctx, kv=kv, mtp=True, draft=True, draft_n=3, vision=False, thinking=thinking,
                effort=None, port=port, no_mmap=False, foreground=False, log=False, extra=[],
            )
            rec = server.running_server()
    if not rec:
        err.print("[red]no server running[/] — pass a model name, e.g. [bold]lethe chat q8_k_p[/]")
        raise typer.Exit(1)
    url = rec["url"]
    p = server.props(url)
    n_ctx = int(p.get("default_generation_settings", {}).get("n_ctx", 0))
    con.print(f"[bold]{Path(rec['model']).name}[/] · ctx {n_ctx:,} · {url}")
    con.print(
        "[dim]/quit  /reset  /wipe  /copy  /think on|off  /stats — Ctrl-C interrupts a reply · "
        "paste multi-line text freely · Enter sends, Ctrl+Enter (or \\\\+Enter) adds a line · nothing is written to disk[/]"
    )

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    sampling = THINKING_SAMPLING if thinking else INSTRUCT_SAMPLING
    kwargs = None if thinking else {"enable_thinking": False}
    last: TurnStats | None = None
    last_reply = ""
    read = _make_reader()
    while True:
        try:
            user = read("\x1b[1;36myou ›\x1b[0m ")
        except (EOFError, KeyboardInterrupt):
            con.print()
            break
        if not user.strip():
            continue
        if user.startswith("/"):
            cmd, _, arg = user[1:].partition(" ")
            if cmd in ("quit", "exit", "q"):
                break
            if cmd == "reset":
                messages = messages[:1] if system else []
                con.print("[dim]conversation cleared[/]")
            elif cmd == "wipe":
                # clear the conversation AND the terminal screen + scrollback (CSI 3J)
                messages = messages[:1] if system else []
                con.file.write("\x1b[2J\x1b[3J\x1b[H")
                con.file.flush()
                con.print("[dim]conversation and terminal scrollback cleared[/]")
            elif cmd == "think":
                thinking = arg.strip() != "off"
                sampling = THINKING_SAMPLING if thinking else INSTRUCT_SAMPLING
                kwargs = None if thinking else {"enable_thinking": False}
                con.print(f"[dim]thinking {'on' if thinking else 'off'}[/]")
            elif cmd == "stats":
                con.print(last.line() if last else "[dim]no turns yet[/]")
            elif cmd == "copy":
                # copies the last reply (without reasoning) to the macOS clipboard. Note: clipboard
                # managers and Universal Clipboard can carry this off the machine — user-initiated only.
                if not last_reply:
                    con.print("[dim]nothing to copy yet[/]")
                elif _pbcopy(last_reply):
                    con.print(f"[dim]copied {len(last_reply):,} chars to the clipboard[/]")
                else:
                    con.print("[red]pbcopy failed[/]")
            else:
                con.print("[dim]unknown command[/]")
            continue

        messages.append({"role": "user", "content": user})
        reply, reasoning = [], []
        in_reasoning = False
        t0 = time.time()
        try:
            for ev in stream_chat(url, messages, n_ctx=n_ctx, sampling=sampling, max_tokens=max_tokens, chat_template_kwargs=kwargs):
                if isinstance(ev, TurnStats):
                    last = ev
                    continue
                if ev.reasoning:
                    reasoning.append(ev.reasoning)
                    if show_thinking:
                        if not in_reasoning:
                            con.file.write("\x1b[2;3mthinking…\x1b[23m ")
                            in_reasoning = True
                        con.file.write(ev.reasoning)
                if ev.content:
                    if in_reasoning:
                        con.file.write("\x1b[0m\n")
                        in_reasoning = False
                    reply.append(ev.content)
                    con.file.write(ev.content)
                con.file.flush()
        except KeyboardInterrupt:
            con.print("\n[yellow]interrupted[/]")
        except Exception as e:  # noqa: BLE001
            con.print(f"\n[red]{e}[/]")
            messages.pop()
            continue
        if in_reasoning:
            con.file.write("\x1b[0m\n")
        con.print()
        text = "".join(reply)
        last_reply = text
        messages.append({"role": "assistant", "content": text})
        wall = time.time() - t0
        if last:
            con.print(f"[dim]{last.line()} · wall {wall:.1f}s[/]")
        con.print()


@app.command()
def bench(
    model: str = typer.Argument(..., help="Local model: path or substring"),
    prompt: str = typer.Option("2048", "-p", help="Prompt lengths to test (comma list)"),
    gen: str = typer.Option("128", "-n", help="Generation lengths (comma list)"),
    depth: str = typer.Option("0", "-d", help="Context depths to test at, e.g. 0,32768,131072"),
    kv: str = typer.Option("f16", help="KV cache type"),
    extra: list[str] = typer.Option([], "--arg", help="Extra llama-bench args"),
):
    """Run llama-bench: prompt-processing and generation tok/s, optionally at context depth."""
    path = _resolve_model(model)
    argv = [
        str(llama_bin("llama-bench")), "-m", str(path), "-ngl", "99", "-fa", "1",
        "-p", prompt, "-n", gen, "-d", depth, "-ctk", kv, "-ctv", kv, *extra,
    ]
    con.print("[dim]$ " + " ".join(argv) + "[/]")
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
def traces(delete: bool = typer.Option(False, "--delete", help="Delete what is found")):
    """List (and optionally delete) anything lethe may have written to disk beyond model files."""
    found = [p for p in [server.log_path(), server.state_dir() / "server.json"] if p.exists()]
    found += [p for p in models_dir().glob("*/.cache") if p.is_dir()]
    if not found:
        con.print("[green]clean[/] — nothing on disk except model files")
        return
    for p in found:
        what = "llama-server log (opt-in)" if p.name == "server.log" else \
            "huggingface_hub download bookkeeping (which files/revisions were fetched — no content)" if p.name == ".cache" else "state"
        con.print(f"{'[red]deleting[/]' if delete else '[yellow]found[/]'} {p} — {what}")
        if delete:
            import shutil

            shutil.rmtree(p) if p.is_dir() else p.unlink()


if __name__ == "__main__":
    app()
