"""The interactive chat loop. Writes nothing to disk unless the user turns on --save."""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

from . import vault
from .attachments import UnsupportedFile, approx_tokens, file_block, image_data_url, is_image
from .chat import Delta, TurnStats, complete, stream_chat
from .profiles import Profile, describe

con = Console()

HELP = """\
/quit          leave (Ctrl-D too)
/reset         clear the conversation (server KV cache is reused only for shared prefixes)
/wipe          clear the conversation AND the terminal scrollback
/copy          copy the last reply (without reasoning) to the clipboard
/file PATH     attach a text/PDF file to your next message (also: lethe chat -f PATH)
/image PATH    attach an image to your next message (needs `serve --vision`)
/think on|off  toggle reasoning mode (switches the sampler preset too)
/compact       summarize the conversation so far with the model and continue from the summary
/save          start saving this conversation, encrypted, to .lethe/conversations/ (asks for a passphrase)
/stats         last turn's throughput/context line
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


def run(cfg: ReplConfig) -> None:
    conv = cfg.resume or vault.Conversation.new(cfg.model_spec)
    messages: list[dict] = list(conv.messages)
    if cfg.system and not any(m["role"] == "system" for m in messages):
        messages.insert(0, {"role": "system", "content": cfg.system})
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
        + " · /help for commands[/]"
    )
    if cfg.resume:
        for m in messages[-2:]:
            if m["role"] in ("user", "assistant"):
                c = m["content"] if isinstance(m["content"], str) else "[multimodal]"
                con.print(f"[dim]{m['role']} › {c[:200]}{'…' if len(c) > 200 else ''}[/]")

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
                        con.print(f"[yellow]saving encrypted as {conv.id}[/] → resume with: lethe chat --resume {conv.id}")
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
            else:
                con.print("[dim]unknown command — /help[/]")
            continue

        try:
            content = _user_content(user, pending_files, pending_images)
        except UnsupportedFile as e:
            con.print(f"[red]{e}[/]")
            continue
        pending_files, pending_images = [], []
        messages.append({"role": "user", "content": content})
        reply, in_reasoning, t0 = [], False, time.time()
        try:
            for ev in stream_chat(cfg.url, messages, n_ctx=cfg.n_ctx, sampling=sampling, max_tokens=cfg.max_tokens, chat_template_kwargs=kwargs, api_key=cfg.api_key):
                if isinstance(ev, TurnStats):
                    last = ev
                    continue
                if ev.reasoning and cfg.show_thinking:
                    if not in_reasoning:
                        sys.stdout.write("\x1b[2;3mthinking…\x1b[23m ")
                        in_reasoning = True
                    sys.stdout.write(ev.reasoning)
                if ev.content:
                    if in_reasoning:
                        sys.stdout.write("\x1b[0m\n")
                        in_reasoning = False
                    reply.append(ev.content)
                    sys.stdout.write(ev.content)
                sys.stdout.flush()
        except KeyboardInterrupt:
            con.print("\n[yellow]interrupted[/]")
        except Exception as e:  # noqa: BLE001
            con.print(f"\n[red]{e}[/]")
            messages.pop()
            continue
        if in_reasoning:
            sys.stdout.write("\x1b[0m\n")
        con.print()
        last_reply = "".join(reply)
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
        con.print(f"[dim]saved {conv.id} · resume with: lethe chat --resume {conv.id}[/]")
