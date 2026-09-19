"""The interactive chat loop. Writes nothing to disk unless the user turns on --save."""

from __future__ import annotations

import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.prompt import Confirm

from . import tools as T, vault
from .attachments import UnsupportedFile, approx_tokens, file_block, image_data_url, is_image
from .chat import Delta, TurnStats, complete, stream_chat
from .profiles import Profile, describe

con = Console()
MAX_TOOL_ROUNDS = 8
MAX_TOOL_CALLS = 16
_DISPLAY_ESCAPES = {
    code: f"\\x{code:02x}" for code in [*range(32), *range(127, 160)] if code not in (9, 10)
}
_DISPLAY_ESCAPES.update({code: f"\\u{code:04x}" for code in [*range(0x202A, 0x202F), *range(0x2066, 0x206A)]})


def _terminal_safe(text: str) -> str:
    """Keep model/file text from controlling the terminal around approval prompts.

    Applied per chunk, so even split escape sequences stay inert. The original
    text remains in the conversation; only its terminal presentation is escaped.
    """
    return text.translate(_DISPLAY_ESCAPES)

HELP = """\
/quit          leave (Ctrl-D too)
/reset         clear the conversation (server KV cache is reused only for shared prefixes)
/wipe          clear the conversation AND the terminal scrollback
/copy          copy the last reply (without reasoning) to the clipboard
/file PATH     attach a text/PDF file to your next message (also: lethe -f PATH)
/image PATH    attach an image to your next message (needs `serve --vision`)
/think on|off  toggle reasoning mode (switches the sampler preset too)
/compact       summarize the conversation so far with the model and continue from the summary
/save          start saving this conversation, encrypted, to .lethe/conversations/ (asks for a passphrase)
/stats         last turn's throughput/context line
/web [off]     enable internet tools mid-chat (search + page fetch; every request is printed) — or launch with --web
/shell [off]   show shell status or disable it (enable at launch with --shell --shell-dir PATH)
Enter sends · Ctrl+Enter / Ctrl+J / \\+Enter add a line · paste multi-line freely · Ctrl-C interrupts a reply"""


@dataclass
class ReplConfig:
    url: str
    api_key: str | None
    n_ctx: int
    model_spec: str
    profile: Profile
    thinking: bool = True
    show_thinking: bool = True
    system: str | None = None
    max_tokens: int | None = None
    sampling_overrides: dict = field(default_factory=dict)
    attachments: list[Path] = field(default_factory=list)
    vision: bool = False
    save: bool = False
    resume: vault.Conversation | None = None
    passphrase: str | None = None
    tools: dict = field(default_factory=dict)  # name -> tools.Tool; empty = no model tools
    tools_confirm: bool = False
    searx: str | None = None  # so /web can build the registry mid-session with the launch-time backend


def _make_reader():
    """read(prompt) on a TTY via prompt_toolkit (bracketed paste, Ctrl+Enter newline, in-memory history);
    plain input() when stdin is piped."""
    import atexit

    if not sys.stdin.isatty():
        return lambda prompt: input(prompt)

    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import ANSI
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys

    ANSI_SEQUENCES.update({
        "\x1b[13;5u": Keys.ControlJ,      # kitty: Ctrl+Enter
        "\x1b[13;2u": Keys.ControlJ,      # kitty: Shift+Enter
        "\x1b[27;5;13~": Keys.ControlJ,   # xterm modifyOtherKeys: Ctrl+Enter
        "\x1b[27;2;13~": Keys.ControlJ,   # xterm modifyOtherKeys: Shift+Enter
        "\x1b[27u": Keys.Escape,          # kitty: bare Esc while the protocol is on
    })
    kb = KeyBindings()

    @kb.add("c-j")
    @kb.add("escape", "enter")
    def _newline(event):
        event.current_buffer.insert_text("\n")

    @kb.add("enter")
    def _enter(event):
        buf = event.current_buffer
        if buf.document.text_before_cursor.endswith("\\"):
            buf.delete_before_cursor(1)
            buf.insert_text("\n")
        else:
            buf.validate_and_handle()

    sys.stdout.write("\x1b[>1u")
    sys.stdout.flush()
    atexit.register(lambda: (sys.stdout.write("\x1b[<u"), sys.stdout.flush()))
    session = PromptSession(history=InMemoryHistory(), key_bindings=kb, mouse_support=False, enable_history_search=False)
    return lambda prompt: session.prompt(ANSI(prompt), prompt_continuation="   … ")


def ask_passphrase(confirm: bool) -> str | None:
    import getpass

    try:
        while True:
            p1 = getpass.getpass("passphrase (not stored anywhere): ")
            if not p1:
                con.print("[yellow]empty passphrase — not saving[/]")
                return None
            if not confirm:
                return p1
            p2 = getpass.getpass("again: ")
            if p1 == p2:
                return p1
            con.print("[red]they differ, try again[/]")
    except (EOFError, KeyboardInterrupt):
        con.print()
        return None


def _pbcopy(text: str) -> bool:
    try:
        subprocess.run(["pbcopy"], input=text.encode(), check=True)
        return True
    except Exception:
        return False


def _user_content(text: str, files: list[Path], images: list[Path]) -> str | list:
    blocks = []
    for f in files:
        blocks.append(file_block(f))
    body = ("\n\n".join(blocks) + "\n\n" + text) if blocks else text
    if not images:
        return body
    parts = [{"type": "text", "text": body}]
    for im in images:
        parts.append({"type": "image_url", "image_url": {"url": image_data_url(im)}})
    return parts


def _sync_tool_prompt(messages: list[dict], cfg: ReplConfig) -> None:
    for message in messages:
        if message["role"] == "system" and isinstance(message["content"], str):
            message["content"] = T.sync_system_text(message["content"], cfg.tools)


def _invoke_tool(cfg: ReplConfig, name: str, arguments: str) -> str:
    tool = cfg.tools.get(name)
    if tool is None:
        return f"unknown or disabled tool {name}"
    try:
        args = tool.parse_arguments(arguments)
    except (ValueError, TypeError) as e:
        return f"invalid tool arguments: {e}"
    # Approval must cover the entire command. Escape control characters and turn
    # off Rich markup so model text cannot hide or restyle what is being approved.
    if tool.confirmation_context:
        con.print(tool.confirmation_context, style="dim", markup=False, highlight=False)
    con.print("⚙ " + T.describe_call(name, arguments, full=True), style="cyan", markup=False, highlight=False)
    if cfg.tools_confirm or tool.requires_confirmation:
        try:
            approved = Confirm.ask("  run this tool?", default=False)
        except EOFError:
            approved = False
        if not approved:
            return "the user declined this request"
    try:
        return tool.fn(**args)
    except Exception as e:  # noqa: BLE001 — give the model the error, never bypass a failed sandbox
        return f"error: {e}"


def run(cfg: ReplConfig) -> None:
    conv = cfg.resume or vault.Conversation.new(cfg.model_spec)
    messages: list[dict] = list(conv.messages)
    if cfg.system and not any(m["role"] == "system" for m in messages):
        messages.insert(0, {"role": "system", "content": cfg.system})
    sysmsg = next((m for m in messages if m["role"] == "system"), None)
    if cfg.resume and sysmsg and isinstance(sysmsg["content"], str):
        # a saved conversation carries its original system prompt; don't let it resurrect a stale "today"
        sysmsg["content"] = re.sub(
            r"(?m)^Current date and time at launch: .*$",
            time.strftime("Current date and time at launch: %A %Y-%m-%d %H:%M %Z."),
            sysmsg["content"],
        )
    _sync_tool_prompt(messages, cfg)
    thinking = cfg.thinking
    sampling = {**cfg.profile.sampling(thinking), **cfg.sampling_overrides}
    kwargs = None if (thinking or not cfg.profile.supports_thinking) else cfg.profile.disable_thinking_kwargs
    saving = cfg.save
    passphrase = cfg.passphrase
    pending_files: list[Path] = list(cfg.attachments)
    pending_images: list[Path] = []
    last: TurnStats | None = None
    last_reply = ""

    con.print(f"[bold]{cfg.model_spec}[/] · ctx {cfg.n_ctx:,} · {cfg.url}")
    con.print(
        f"[dim]sampler: {cfg.profile.name} {'thinking' if thinking and cfg.profile.supports_thinking else 'instruct'} preset ({describe(sampling)})"
        + (f" · {len(pending_files)} file(s) attached to the first message" if pending_files else "")
        + (f" · resumed {conv.id}: {conv.turns} turns" if cfg.resume else "")
        + (" · [yellow]saving encrypted[/]" if saving else "")
        + (f" · [yellow]tools ON: {', '.join(cfg.tools)}[/]" if cfg.tools else "")
        + " · /help for commands[/]"
    )
    if "run_shell" in cfg.tools:
        con.print("shell: read-only, network blocked, confirmation required for every command", style="yellow")
    if cfg.resume:
        for m in messages:
            if m["role"] == "user":
                c = m["content"]
                if not isinstance(c, str):
                    texts = [p.get("text", "") for p in c if p.get("type") == "text"]
                    n_img = sum(1 for p in c if p.get("type") == "image_url")
                    c = "\n".join(t for t in texts if t) + (f" [{n_img} image(s)]" if n_img else "")
                sys.stdout.write(f"\x1b[1;36myou ›\x1b[0m {_terminal_safe(c)}\n")
            elif m["role"] == "assistant" and isinstance(m["content"], str):
                sys.stdout.write(f"{_terminal_safe(m['content'])}\n\n")
        sys.stdout.flush()

    def persist():
        nonlocal passphrase, saving
        if not saving:
            return
        conv.messages = messages
        try:
            vault.save(conv, passphrase)
        except Exception as e:  # noqa: BLE001
            con.print(f"[red]save failed: {e}[/]")

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
            arg = arg.strip()
            if cmd in ("quit", "exit", "q"):
                break
            elif cmd == "help":
                con.print(HELP)
            elif cmd == "reset":
                messages = [m for m in messages if m["role"] == "system"]
                last, last_reply = None, ""
                con.print("[dim]conversation cleared[/]")
            elif cmd == "wipe":
                messages = [m for m in messages if m["role"] == "system"]
                last, last_reply = None, ""
                sys.stdout.write("\x1b[2J\x1b[3J\x1b[H")
                sys.stdout.flush()
                con.print("[dim]conversation and terminal scrollback cleared[/]")
            elif cmd == "think":
                thinking = arg != "off"
                sampling = {**cfg.profile.sampling(thinking), **cfg.sampling_overrides}
                kwargs = None if (thinking or not cfg.profile.supports_thinking) else cfg.profile.disable_thinking_kwargs
                con.print(f"[dim]thinking {'on' if thinking else 'off'} · sampler {describe(sampling)}[/]")
            elif cmd == "stats":
                con.print(last.line() if last else "[dim]no turns yet[/]")
            elif cmd == "copy":
                if not last_reply:
                    con.print("[dim]nothing to copy yet[/]")
                elif _pbcopy(last_reply):
                    con.print(f"[dim]copied {len(last_reply):,} chars to the clipboard[/]")
                else:
                    con.print("[red]pbcopy failed[/]")
            elif cmd in ("file", "image"):
                p = Path(arg).expanduser()
                if not p.is_file():
                    con.print(f"[red]no such file: {arg}[/]")
                elif cmd == "image" or is_image(p):
                    if not cfg.vision:
                        con.print("[red]server has no vision projector — start with `lethe serve … --vision`[/]")
                    else:
                        pending_images.append(p)
                        con.print(f"[dim]image {p.name} attached to your next message[/]")
                else:
                    try:
                        n = approx_tokens(file_block(p))
                    except UnsupportedFile as e:
                        con.print(f"[red]{e}[/]")
                        continue
                    pending_files.append(p)
                    con.print(f"[dim]{p.name} attached to your next message (~{n:,} tokens)[/]")
            elif cmd == "save":
                if saving:
                    con.print(f"[dim]already saving as {conv.id}[/]")
                else:
                    passphrase = ask_passphrase(confirm=True)
                    if passphrase:
                        saving = True
                        persist()
                        con.print(f"[yellow]saving encrypted as {conv.id}[/] → resume with: lethe --resume {conv.id}")
            elif cmd == "compact":
                if not any(m["role"] == "user" for m in messages):
                    con.print("[dim]nothing to compact[/]")
                    continue
                with con.status("summarizing…"):
                    summary, _ = complete(
                        cfg.url,
                        [m for m in messages if m["role"] != "system"]
                        + [{"role": "user", "content": "Summarize everything in this conversation so far — facts, decisions, open questions, and any code or data that must be carried forward — as a compact briefing for continuing the conversation. Output only the briefing."}],
                        sampling={"temperature": 0.3},
                        max_tokens=4096,
                        chat_template_kwargs=cfg.profile.disable_thinking_kwargs if cfg.profile.supports_thinking else None,
                        api_key=cfg.api_key,
                    )
                sys_msgs = [m for m in messages if m["role"] == "system"]
                messages = sys_msgs + [
                    {"role": "user", "content": "Here is a briefing of our conversation so far; continue from it.\n\n" + summary.strip()},
                    {"role": "assistant", "content": "Understood — continuing from that briefing."},
                ]
                con.print(f"[dim]compacted to ~{approx_tokens(summary):,} tokens of summary[/]")
                persist()
            elif cmd == "web":
                if arg == "off":
                    for name in T.WEB_TOOL_NAMES:
                        cfg.tools.pop(name, None)
                    con.print("[dim]internet tools off[/]")
                else:
                    cfg.tools.update(T.build_registry(["web"], searx=cfg.searx))
                    con.print("[yellow]internet tools ON: web_search, fetch_url — model-composed queries/URLs will leave this machine; every request is printed[/]")
                _sync_tool_prompt(messages, cfg)
            elif cmd == "shell":
                if arg == "off":
                    cfg.tools.pop("run_shell", None)
                    _sync_tool_prompt(messages, cfg)
                    con.print("[dim]shell tool off[/]")
                elif "run_shell" in cfg.tools:
                    con.print(cfg.tools["run_shell"].description, markup=False)
                else:
                    con.print("[dim]shell tool off — enable at launch with --shell --shell-dir PATH[/]")
            else:
                con.print("[dim]unknown command — /help[/]")
            continue

        try:
            content = _user_content(user, pending_files, pending_images)
        except UnsupportedFile as e:
            con.print(f"[red]{e}[/]")
            continue
        pending_files, pending_images = [], []
        turn_start = len(messages)
        messages.append({"role": "user", "content": content})
        tool_specs = [t.spec() for t in cfg.tools.values()] or None
        round_text: list[str] = []
        in_reasoning, t0, rounds, calls_used = False, time.time(), 0, 0
        try:
            while True:
                round_text, pending_calls = [], None
                for ev in stream_chat(cfg.url, messages, n_ctx=cfg.n_ctx, sampling=sampling, max_tokens=cfg.max_tokens, chat_template_kwargs=kwargs, api_key=cfg.api_key, tools=tool_specs):
                    if isinstance(ev, TurnStats):
                        last = ev
                        continue
                    if ev.tool_calls:
                        pending_calls = ev.tool_calls
                        continue
                    if ev.reasoning and cfg.show_thinking:
                        if not in_reasoning:
                            sys.stdout.write("\x1b[2;3mthinking…\x1b[23m ")
                            in_reasoning = True
                        sys.stdout.write(_terminal_safe(ev.reasoning))
                    if ev.content:
                        if in_reasoning:
                            sys.stdout.write("\x1b[0m\n")
                            in_reasoning = False
                        round_text.append(ev.content)
                        sys.stdout.write(_terminal_safe(ev.content))
                    sys.stdout.flush()
                if in_reasoning:
                    sys.stdout.write("\x1b[0m\n")
                    in_reasoning = False
                if not pending_calls:
                    break
                messages.append({"role": "assistant", "content": "".join(round_text) or None, "tool_calls": pending_calls})
                if tool_specs is None:
                    for call in pending_calls:
                        messages.append({"role": "tool", "tool_call_id": call["id"], "name": call["function"]["name"],
                                         "content": "tool execution disabled or tool-call limit reached"})
                    round_text = ["Stopped: the model requested tools after they were disabled or the tool-call limit was reached."]
                    con.print(round_text[0], style="yellow")
                    break  # a model ignoring withdrawn tools must not create an infinite loop
                rounds += 1
                for call in pending_calls:
                    name, arguments = call["function"]["name"], call["function"]["arguments"]
                    if calls_used >= MAX_TOOL_CALLS:
                        result = "tool-call limit reached: answer with what you have"
                    else:
                        calls_used += 1
                        result = _invoke_tool(cfg, name, arguments)
                    con.print(f"[dim]  → {len(result):,} chars[/]")
                    messages.append({"role": "tool", "tool_call_id": call["id"], "name": name, "content": result})
                if rounds >= MAX_TOOL_ROUNDS or calls_used >= MAX_TOOL_CALLS:
                    con.print("[yellow]tool-call limit reached; requesting a final answer[/]")
                    tool_specs = None
        except KeyboardInterrupt:
            con.print("\n[yellow]interrupted[/]")
            del messages[turn_start:]
            continue
        except Exception as e:  # noqa: BLE001
            con.print(f"\n[red]{e}[/]")
            del messages[turn_start:]
            continue
        con.print()
        last_reply = "".join(round_text)
        messages.append({"role": "assistant", "content": last_reply})
        if last:
            con.print(f"[dim]{last.line()} · wall {time.time() - t0:.1f}s[/]")
            if last.ctx_frac >= 0.8:
                con.print(f"[yellow]context {last.ctx_frac:.0%} full — /compact to summarize, or /reset[/]")
            if last.finish_reason == "length" and not cfg.max_tokens:
                con.print("[yellow]reply was cut off by the context limit — /compact[/]")
        persist()
        con.print()
    if saving:
        con.print(f"[dim]saved {conv.id} · resume with: lethe --resume {conv.id}[/]")
