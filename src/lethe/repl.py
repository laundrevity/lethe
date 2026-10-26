"""The interactive chat loop. Conversation storage is opt-in; full shell commands may write files."""

from __future__ import annotations

import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

from . import tools as T, vault
from .attachments import UnsupportedFile, approx_tokens, file_block, image_data_url, is_image, is_video
from .chat import Delta, TurnStats, complete, stream_chat
from .profiles import Profile, describe
from .terminal import StatusLine, ToolCallStream, notify, terminal_safe as _terminal_safe

con = Console()
#MAX_TOOL_ROUNDS = 8
#MAX_TOOL_CALLS = 16

# TODO: add option to disable tool call limits via CLI e.g. lethe.toml
MAX_TOOL_ROUNDS = 10**9
MAX_TOOL_CALLS = 10**9

HELP = """\
/quit          leave (Ctrl-D too)
/reset         clear the conversation (server KV cache is reused only for shared prefixes)
/wipe          clear the conversation AND the terminal scrollback
/copy          copy the last reply (without reasoning) to the clipboard
/file PATH     attach text/PDF/image/video to your next message (also: lethe -f PATH)
/image PATH    attach an image to your next message (needs `lethe --vision`)
/video PATH    attach sampled video frames, without audio (needs --vision and FFmpeg)
               up to 1 frame/s, spread across the clip; --video-frames sets the cap (default 16)
/think on|off  toggle reasoning mode (switches the sampler preset too)
/compact       summarize the conversation so far with the model and continue from the summary
/save          start saving this conversation, encrypted, to .lethe/conversations/ (asks for a passphrase;
               after --resume it keeps the same id and passphrase — resuming alone never writes)
/stats         last turn's throughput/context line
/web [off]     enable internet tools mid-chat (search + page fetch; every request is printed) — or launch with --web
/shell [full|read-only|off] [PATH]  show, enable, change, or disable shell access mid-chat
               (brings read_file; full also brings write_file + edit_file, approved as diffs)
/tools [full|off] [PATH]  show tools, enable web + full shell access, or disable all tools
Tool approval: y runs the call · n declines · p sends the model a note instead of running it (p your note, or type it on the next line)
Enter sends · Option+Enter, Ctrl+J, Shift+Enter, Ctrl+Enter or a trailing \\ add a line (Apple Terminal cannot tell Shift/Ctrl+Enter
from Enter: use Option+Enter with "Use Option as Meta Key" on, or Ctrl+J) · paste multi-line freely
Ctrl-C stops a reply, keeping what it thought and wrote so far · the bell rings when a tool call waits for you and when a reply ends
The bottom line shows context used (≈ until the server reports it), the phase, speed and elapsed time while a reply is produced"""

# Prefixed to the next prompt after Ctrl-C, so the model knows why its last message stops short.
INTERRUPT_NOTE = (
    "[Your previous response was cut short before it finished: the user interrupted it. "
    "Your reasoning and output up to that point are kept above.]"
)
# Stands in for a reply when Ctrl-C arrived before any reasoning or text.
NO_OUTPUT = "[Stopped before any response.]"


class ToolInterrupted(KeyboardInterrupt):
    """Ctrl-C arrived while a tool was running, so the call may have partly happened."""


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
    video_frames: int = 16
    save: bool = False
    resume: vault.Conversation | None = None
    passphrase: str | None = None
    tools: dict = field(default_factory=dict)  # name -> tools.Tool; empty = no model tools
    tools_confirm: bool = False
    searx: str | None = None  # so /web can build the registry mid-session with the launch-time backend
    shell_dir: Path = field(default_factory=Path.cwd)  # remembered across mode changes and /shell off
    auto_approve: str | None = None  # --auto-approve files|all; calls are still printed, just not asked about
    resume_args: list[str] = field(default_factory=list)  # launch options for the printed hint; RAM only

    def resume_command(self, conversation_id: str) -> str:
        return shlex.join(["lethe", "--resume", conversation_id, *self.resume_args])


def key_sequences() -> dict:
    """Key encodings prompt_toolkit does not know, mapped onto the keys it does.

    Terminals speaking the kitty keyboard protocol (Ghostty, kitty, WezTerm, iTerm2 3.5+) report modified
    keys as `CSI code;modifiers u` while the protocol is on — including Ctrl+letter, so the editing keys
    need their names back. Every modified Enter (Shift, Ctrl, Option) becomes a newline. Also covered:
    xterm's modifyOtherKeys form of a modified Enter and the keypad Enter. Apple Terminal supports none of
    these; it sends Shift+Enter and Ctrl+Enter as a plain Enter, and Option+Enter as ESC CR when the
    profile uses Option as Meta — that form is a key binding below.
    """
    import string

    from prompt_toolkit.keys import Keys

    keys = {
        "\x1b[27u": Keys.Escape, "\x1b[13u": Keys.ControlM, "\x1b[9u": Keys.ControlI, "\x1b[9;2u": Keys.BackTab,
        "\x1b[127u": Keys.ControlH, "\x1bOM": Keys.ControlM,
    }
    for mod in (2, 3, 4, 5, 6, 7, 8):  # shift, alt, alt+shift, ctrl, ctrl+shift, ctrl+alt, all three
        keys[f"\x1b[13;{mod}u"] = keys[f"\x1b[27;{mod};13~"] = Keys.ControlJ
        keys[f"\x1b[127;{mod}u"] = Keys.ControlH
    for c in string.ascii_lowercase:
        keys[f"\x1b[{ord(c)};5u"] = keys[f"\x1b[{ord(c)};6u"] = getattr(Keys, f"Control{c.upper()}")
    return keys


def _make_reader():
    """read(prompt) on a TTY via prompt_toolkit (bracketed paste, modified-Enter newlines, in-memory history);
    plain input() when stdin is piped."""
    if not sys.stdin.isatty():
        return lambda prompt: input(prompt)

    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import ANSI
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
    from prompt_toolkit.key_binding import KeyBindings

    ANSI_SEQUENCES.update(key_sequences())
    kb = KeyBindings()

    @kb.add("c-j")
    @kb.add("escape", "enter")  # Option/Alt+Enter as ESC CR (Apple Terminal with Option as Meta, iTerm2, VS Code)
    @kb.add("escape", "c-j")    # ESC LF, which some terminal key mappings send instead
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

    session = PromptSession(history=InMemoryHistory(), key_bindings=kb, mouse_support=False, enable_history_search=False)

    def read(prompt: str) -> str:
        # The kitty keyboard protocol is on only while the line editor runs. Left on, it would also turn
        # the Ctrl-C that stops a streaming reply into an escape sequence the tty does not treat as SIGINT.
        sys.stdout.write("\x1b[>1u")
        sys.stdout.flush()
        try:
            return session.prompt(ANSI(prompt), prompt_continuation="   … ")
        finally:
            sys.stdout.write("\x1b[<u")
            sys.stdout.flush()

    return read


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


def _user_content(
    text: str, files: list[Path], images: list[Path], videos: list[Path] | None = None, *, video_frames: int = 16,
) -> str | list:
    blocks = []
    for f in files:
        blocks.append(file_block(f))
    body = ("\n\n".join(blocks) + "\n\n" + text) if blocks else text
    if not images and not videos:
        return body
    parts = [{"type": "text", "text": body}]
    for im in images:
        parts.append({"type": "image_url", "image_url": {"url": image_data_url(im)}})
    if videos:
        from .video import video_parts

        for video in videos:
            parts.extend(video_parts(video, video_frames))
    return parts


def _with_note(content: str | list, note: str) -> str | list:
    """Put a note before the user's text, whether it is plain or a list of parts."""
    if isinstance(content, str):
        return f"{note}\n\n{content}"
    parts = [dict(part) for part in content]
    for part in parts:
        if part.get("type") == "text":
            part["text"] = f"{note}\n\n{part['text']}"
            break
    else:
        parts.insert(0, {"type": "text", "text": note})
    return parts


def _assistant_message(text: list[str], reasoning: list[str], calls: list[dict] | None = None) -> dict:
    """Keep the full streamed round for model context and opt-in encrypted saves."""
    message = {"role": "assistant", "content": "".join(text)}
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    if calls:
        message["content"] = message["content"] or None
        message["tool_calls"] = calls
    return message


APPROVAL_HINT = "  y = run it · n = decline · p = not yet: send the model a note first (p your note)"
# Tool result when the user answers `p` with a note instead of approving the call.
NOTE_RESULT = (
    'not run: the user wants to clarify first. User: "{note}". '
    "Follow this; request the call again if it is still appropriate."
)


def ask_approval() -> bool | str:
    """The question before a tool runs: True = run it, False = declined, a str = the user's note for the model.

    `p` ("perhaps") hands the model a note instead of a verdict: nothing runs, the model reads the note and
    may ask again. The note can follow on the same line (`p use ninja instead`) or be typed on the next.
    Enter alone declines, as does end of input.
    """
    while True:
        answer = con.input("  [bold]run this tool?[/] [dim]\\[y/n/p][/] ").strip()
        verdict, _, note = answer.partition(" ")
        verdict, note = verdict.lower(), note.strip()
        if verdict in ("", "n", "no"):
            return False
        if verdict in ("y", "yes") and not note:
            return True
        if verdict in ("p", "perhaps"):
            note = note or con.input("  [dim]note for the model:[/] ").strip()
            return note or False
        con.print(APPROVAL_HINT, style="dim", markup=False)


def _print_tool_call(name: str, arguments: str) -> None:
    con.print("⚙ " + T.describe_call(name, arguments, full=True), style="cyan", markup=False, highlight=False)


def _print_tool_result(result: str | list[dict]) -> None:
    if isinstance(result, str):
        con.print(f"[dim]  → {len(result):,} chars[/]")
    else:
        chars = sum(len(p.get("text", "")) for p in result if p.get("type") == "text")
        images = sum(p.get("type") == "image_url" for p in result)
        con.print(f"[dim]  → {chars:,} chars · {images} image(s)[/]")


def _replay_history(messages: list[dict], *, show_thinking: bool) -> None:
    """Render saved messages only; never invoke tools or rebuild previews from files."""
    pending: dict[str, tuple[str, str]] = {}
    for message in messages:
        role = message["role"]
        if role == "tool":
            # Pair each call with its result, as in a live turn with multiple calls.
            function = pending.pop(message.get("tool_call_id"), None)
            if function:
                _print_tool_call(*function)
            _print_tool_result(message.get("content") or "")
            continue
        # Older/incomplete histories may contain calls without a corresponding reply.
        for function in pending.values():
            _print_tool_call(*function)
        pending.clear()
        content = message.get("content")
        if role == "user":
            if not isinstance(content, str):
                parts = content or []
                texts = [p.get("text", "") for p in parts if p.get("type") == "text"]
                n_img = sum(1 for p in parts if p.get("type") == "image_url")
                content = "\n".join(t for t in texts if t) + (f" [{n_img} image(s)]" if n_img else "")
            sys.stdout.write(f"\x1b[1;36myou ›\x1b[0m {_terminal_safe(content)}\n")
        elif role == "assistant":
            reasoning = message.get("reasoning_content")
            if show_thinking and isinstance(reasoning, str) and reasoning:
                sys.stdout.write(f"\x1b[2;3mthinking…\x1b[23m {_terminal_safe(reasoning)}\x1b[0m\n")
            if isinstance(content, str) and content:
                sys.stdout.write(f"{_terminal_safe(content)}\n")
            for call in message.get("tool_calls") or []:
                function = call["function"]
                pending[call["id"]] = (function["name"], function.get("arguments", ""))
            if not pending:
                sys.stdout.write("\n")
    for function in pending.values():
        _print_tool_call(*function)
    sys.stdout.flush()


def _sync_tool_prompt(messages: list[dict], cfg: ReplConfig) -> None:
    for message in messages:
        if message["role"] == "system" and isinstance(message["content"], str):
            message["content"] = T.sync_system_text(message["content"], cfg.tools)


def _enable_shell(cfg: ReplConfig, mode: T.ShellMode, directory: Path | None = None, *, web: bool = False) -> None:
    """Build and probe the replacement before changing any active tools."""
    root = (directory if directory is not None else cfg.shell_dir).expanduser().resolve()
    registry = T.build_registry(
        ["shell", "web"] if web else ["shell"], searx=cfg.searx, shell_dir=root, shell_mode=mode,
    )
    if cfg.vision:
        registry["view_image"] = T.build_image_tool(root, confined=mode != T.ShellMode.FULL)
    for name in T.SHELL_TOOL_NAMES:  # a switch to read-only must drop write_file/edit_file
        cfg.tools.pop(name, None)
    cfg.tools.update(registry)
    cfg.shell_dir = root


def _approval_line(cfg: ReplConfig) -> str:
    if cfg.auto_approve == "all":
        return "AUTO-APPROVE all: shell commands and file changes run without asking"
    if cfg.auto_approve == "files":
        return "AUTO-APPROVE files: file reads, writes and edits run without asking; shell commands still ask"
    return "confirmation required for every shell command and file tool call"


def _tool_command(cmd: str, arg: str, messages: list[dict], cfg: ReplConfig) -> None:
    usage = (
        "usage: /shell [full|read-only|off] [PATH]"
        if cmd == "shell" else "usage: /tools [full|off] [PATH]"
    )
    try:
        parts = shlex.split(arg)
        if not parts:
            if cmd == "tools":
                con.print("tools: " + (", ".join(cfg.tools) or "off"), markup=False)
            if "run_shell" in cfg.tools:
                con.print(cfg.tools["run_shell"].description, markup=False, highlight=False)
            else:
                con.print("shell tool off — /shell read-only, /shell full, or /tools full to enable", style="dim")
            return
        if parts == ["off"]:
            if cmd == "tools":
                cfg.tools.clear()
            else:
                for name in T.SHELL_TOOL_NAMES:
                    cfg.tools.pop(name, None)
            _sync_tool_prompt(messages, cfg)
            con.print("all tools off" if cmd == "tools" else "shell tool off", style="dim")
            return
        modes = {"full": T.ShellMode.FULL}
        if cmd == "shell":
            modes["read-only"] = T.ShellMode.READ_ONLY
        if parts[0] not in modes or len(parts) > 2:
            con.print(usage + " (quote paths with spaces)", style="yellow", markup=False)
            return
        directory = Path(parts[1]) if len(parts) == 2 else None
        _enable_shell(cfg, modes[parts[0]], directory, web=cmd == "tools")
    except (OSError, ValueError, RuntimeError) as e:
        con.print("tools unchanged: " + _terminal_safe(str(e)), style="red", markup=False)
        return
    except KeyboardInterrupt:
        con.print("tool change cancelled; tools unchanged", style="dim")
        return
    _sync_tool_prompt(messages, cfg)
    if cmd == "tools":
        con.print("web tools ON: web_search, fetch_url", style="yellow")
    con.print(cfg.tools["run_shell"].confirmation_context, style="yellow", markup=False, highlight=False)
    con.print("file tools: " + ", ".join(n for n in cfg.tools if n in T.FILE_TOOL_NAMES), style="yellow")
    con.print(_approval_line(cfg), style="red" if cfg.auto_approve else "dim")


def _invoke_tool(cfg: ReplConfig, name: str, arguments: str) -> str | list[dict]:
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
    if tool.preview is not None:
        # File changes are approved as a diff, not as a JSON blob of escaped newlines.
        try:
            header, _, body = tool.preview(**args).partition("\n")
        except Exception as e:  # noqa: BLE001 — e.g. old_string not found: tell the model, don't bother the user
            return f"error: {e}"
        con.print("⚙ " + ascii(name) + " " + _terminal_safe(header), style="cyan", markup=False, highlight=False)
        for line in body.split("\n"):
            style = "green" if line.startswith("+") else "red" if line.startswith("-") else "dim"
            con.print(_terminal_safe(line), style=style, markup=False, highlight=False, soft_wrap=True)
    else:
        _print_tool_call(name, arguments)
    auto = cfg.auto_approve == "all" or (cfg.auto_approve == "files" and name in T.FILE_TOOL_NAMES)
    if (cfg.tools_confirm or tool.requires_confirmation) and not auto:
        notify("lethe: a tool call needs your approval")
        try:
            decision = ask_approval()
        except EOFError:
            decision = False
        if decision is False:
            return "the user declined this request"
        if decision is not True:
            return NOTE_RESULT.format(note=decision)
    try:
        return tool.fn(**args)
    except KeyboardInterrupt as e:
        raise ToolInterrupted from e
    except Exception as e:  # noqa: BLE001 — give the model the error, never bypass a failed sandbox
        return f"error: {e}"


def _approx_messages(messages: list[dict]) -> int:
    """Rough token count of messages, for the status line until the server reports the real position."""
    n = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            n += approx_tokens(content)
        elif isinstance(content, list):
            n += sum(approx_tokens(part.get("text", "")) for part in content if part.get("type") == "text")
        if isinstance(m.get("reasoning_content"), str):
            n += approx_tokens(m["reasoning_content"])
        for call in m.get("tool_calls") or []:
            n += approx_tokens(call["function"].get("arguments", ""))
        n += 4  # role and turn markers
    return n


def _status_text(phase: str, progress: TurnStats | None, ctx_est: int, n_ctx: int, t0: float) -> str:
    """The pinned line during a turn: context position (≈ until the server reports it), phase, speed, time."""
    used = progress.ctx_used if progress else ctx_est
    ctx = f"ctx {'' if progress else '≈'}{used:,}"
    if n_ctx:
        ctx += f"/{n_ctx:,} ({used / n_ctx:.0%})"
    if progress and progress.predicted_n:
        phase = f"{phase} {progress.predicted_n:,} tok @ {progress.gen_tps:.1f} tok/s"
    return f"{ctx} · {phase} · {time.time() - t0:.0f}s"


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
    interrupted = bool(getattr(cfg.resume, "interrupted", False))  # the last turn stopped early
    thinking = cfg.thinking
    sampling = {**cfg.profile.sampling(thinking), **cfg.sampling_overrides}
    kwargs = None if (thinking or not cfg.profile.supports_thinking) else cfg.profile.disable_thinking_kwargs
    saving = cfg.save
    passphrase = cfg.passphrase
    pending_files = [p for p in cfg.attachments if not is_image(p) and not is_video(p)]
    pending_images = [p for p in cfg.attachments if is_image(p)]
    pending_videos = [p for p in cfg.attachments if is_video(p)]
    if (pending_images or pending_videos) and not cfg.vision:
        con.print("server has no vision projector — start with `lethe … --vision`", style="red")
        return
    last: TurnStats | None = None
    last_reply = ""

    con.print(f"[bold]{cfg.model_spec}[/] · ctx {cfg.n_ctx:,} · {cfg.url}")
    con.print(
        f"[dim]sampler: {cfg.profile.name} {'thinking' if thinking and cfg.profile.supports_thinking else 'instruct'} preset ({describe(sampling)})"
        + (f" · {len(cfg.attachments)} file(s) attached to the first message" if cfg.attachments else "")
        + (f" · resumed {conv.id}: {conv.turns} turns" if cfg.resume else "")
        + (" · [yellow]saving encrypted[/]" if saving else " · not saving (/save keeps this session's turns)" if cfg.resume else "")
        + (f" · [yellow]tools ON: {', '.join(cfg.tools)}[/]" if cfg.tools else "")
        + " · /help for commands[/]"
    )
    if "run_shell" in cfg.tools:
        scope = (
            "full access, files writable, network allowed, inherited environment"
            if cfg.tools["run_shell"].shell_mode == T.ShellMode.FULL else "read-only, network blocked"
        )
        con.print(f"shell: {scope}, confirmation required for every command", style="yellow")
    if cfg.auto_approve:
        con.print(_approval_line(cfg), style="bold red")
    if cfg.resume:
        _replay_history(messages, show_thinking=cfg.show_thinking)

    def persist():
        nonlocal passphrase, saving
        if not saving:
            return
        conv.messages = messages
        conv.interrupted = interrupted
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
                con.print(HELP, markup=False)
            elif cmd == "reset":
                messages = [m for m in messages if m["role"] == "system"]
                pending_files, pending_images, pending_videos = [], [], []
                last, last_reply = None, ""
                con.print("[dim]conversation cleared[/]")
            elif cmd == "wipe":
                messages = [m for m in messages if m["role"] == "system"]
                pending_files, pending_images, pending_videos = [], [], []
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
            elif cmd in ("file", "image", "video"):
                # Accept a literal path with spaces or a quoted/terminal-dropped path.
                p = Path(arg).expanduser()
                if not p.is_file():
                    try:
                        words = shlex.split(arg)
                        if len(words) == 1:
                            p = Path(words[0]).expanduser()
                    except ValueError:
                        pass
                if not p.is_file():
                    con.print(_terminal_safe(f"no such file: {arg}"), style="red", markup=False)
                elif cmd in ("image", "video") or is_image(p) or is_video(p):
                    if not cfg.vision:
                        con.print("[red]server has no vision projector — start with `lethe … --vision`[/]")
                    elif cmd == "video" or (cmd == "file" and is_video(p)):
                        pending_videos.append(p)
                        con.print(_terminal_safe(
                            f"video {p.name} attached to your next message (up to {cfg.video_frames} sampled frames; no audio)"
                        ), style="dim", markup=False)
                    elif not is_image(p):
                        con.print("unsupported image extension; use /video for video files", style="red")
                    else:
                        pending_images.append(p)
                        con.print(_terminal_safe(f"image {p.name} attached to your next message"), style="dim", markup=False)
                else:
                    try:
                        n = approx_tokens(file_block(p))
                    except (UnsupportedFile, OSError) as e:
                        con.print(_terminal_safe(str(e)), style="red", markup=False)
                        continue
                    pending_files.append(p)
                    con.print(f"[dim]{p.name} attached to your next message (~{n:,} tokens)[/]")
            elif cmd == "save":
                if saving:
                    con.print(f"[dim]already saving as {conv.id}[/]")
                else:
                    if not passphrase:  # a resumed conversation keeps the passphrase it was opened with
                        passphrase = ask_passphrase(confirm=True)
                    if passphrase:
                        saving = True
                        persist()
                        con.print(_terminal_safe(f"saving encrypted as {conv.id} → resume with: {cfg.resume_command(conv.id)}"),
                                  style="yellow", markup=False, highlight=False, soft_wrap=True)
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
                last = None  # the context position is unknown until the next turn reports it
                con.print(f"[dim]compacted to ~{approx_tokens(summary):,} tokens of summary[/]")
                persist()
            elif cmd == "web":
                if arg == "off":
                    for name in T.WEB_TOOL_NAMES:
                        cfg.tools.pop(name, None)
                    con.print("[dim]web search/fetch tools off[/]")
                else:
                    cfg.tools.update(T.build_registry(["web"], searx=cfg.searx))
                    con.print("[yellow]internet tools ON: web_search, fetch_url — model-composed queries/URLs will leave this machine; every request is printed[/]")
                _sync_tool_prompt(messages, cfg)
            elif cmd in ("shell", "tools"):
                _tool_command(cmd, arg, messages, cfg)
            else:
                con.print("[dim]unknown command — /help[/]")
            continue

        try:
            if pending_videos:
                with con.status("sampling video frames…"):
                    content = _user_content(user, pending_files, pending_images, pending_videos, video_frames=cfg.video_frames)
            else:
                content = _user_content(user, pending_files, pending_images)
        except (UnsupportedFile, OSError) as e:
            con.print(_terminal_safe(str(e)), style="red", markup=False)
            continue
        except KeyboardInterrupt:
            con.print("[dim]attachment preparation stopped; attachments kept (/reset clears them)[/]")
            continue
        pending_files, pending_images, pending_videos = [], [], []
        if interrupted:
            content = _with_note(content, INTERRUPT_NOTE)
        turn_start = len(messages)
        messages.append({"role": "user", "content": content})
        tool_specs = [t.spec() for t in cfg.tools.values()] or None
        round_text: list[str] = []
        round_reasoning: list[str] = []
        awaiting: list[dict] = []  # the model's tool calls that have no reply yet
        in_reasoning, t0, rounds, calls_used = False, time.time(), 0, 0
        # Context position for the status line: the server's running timings once a round streams; until
        # then the last turn's exact number (or a character-count guess) plus what has been added since.
        ctx_est = (last.ctx_used if last else _approx_messages(messages[:-1])) + _approx_messages(messages[-1:])
        progress: TurnStats | None = None
        status = StatusLine()

        def show(phase: str, *, force: bool = False) -> None:
            status.update(_status_text(phase, progress, ctx_est, cfg.n_ctx, t0),
                          sgr="\x1b[2;3m" if in_reasoning else "\x1b[0m", force=force)

        def emit(text: str) -> None:  # streamed text; the status line tracks the cursor column
            sys.stdout.write(text)
            status.wrote(text)

        try:
            with status:
                while True:
                    round_text, round_reasoning, pending_calls, progress, phase = [], [], None, None, "prompt"
                    show(phase, force=True)
                    with ToolCallStream(status) as call_stream:
                        for ev in stream_chat(cfg.url, messages, n_ctx=cfg.n_ctx, sampling=sampling, max_tokens=cfg.max_tokens, chat_template_kwargs=kwargs, api_key=cfg.api_key, tools=tool_specs):
                            if isinstance(ev, TurnStats):
                                last = ev
                                continue
                            if ev.progress:
                                phase = f"prompt {ev.progress.get('processed', 0):,}/{ev.progress.get('total', 0):,} tok"
                            if ev.timings:
                                progress = TurnStats.from_timings(ev.timings, cfg.n_ctx, None)
                            if ev.reasoning or ev.content:
                                call_stream.finish()
                            if ev.reasoning:
                                phase = "thinking"
                                round_reasoning.append(ev.reasoning)
                            if ev.reasoning and cfg.show_thinking:
                                if not in_reasoning:
                                    sys.stdout.write("\x1b[2;3mthinking…\x1b[23m ")
                                    status.wrote("thinking… ")
                                    in_reasoning = True
                                emit(_terminal_safe(ev.reasoning))
                            if ev.content or ev.tool_call_deltas:
                                if in_reasoning:
                                    sys.stdout.write("\x1b[0m")
                                    emit("\n")
                                    in_reasoning = False
                            if ev.content:
                                phase = "writing"
                                round_text.append(ev.content)
                                emit(_terminal_safe(ev.content))
                            if ev.tool_call_deltas:
                                phase = "tool call"
                                call_stream.feed(ev.tool_call_deltas)
                            if ev.tool_calls:
                                phase = "tool call"
                                pending_calls = ev.tool_calls
                                call_stream.finish()
                            sys.stdout.flush()
                            show(phase)
                    if in_reasoning:
                        sys.stdout.write("\x1b[0m")
                        emit("\n")
                        in_reasoning = False
                    if not pending_calls:
                        break
                    messages.append(_assistant_message(round_text, round_reasoning, pending_calls))
                    awaiting = list(pending_calls)
                    if tool_specs is None:
                        for call in pending_calls:
                            messages.append({"role": "tool", "tool_call_id": call["id"], "name": call["function"]["name"],
                                             "content": "tool execution disabled or tool-call limit reached"})
                        awaiting.clear()
                        round_text = ["Stopped: the model requested tools after they were disabled or the tool-call limit was reached."]
                        round_reasoning = []  # this synthetic reply has no reasoning of its own
                        con.print(round_text[0], style="yellow")
                        break  # a model ignoring withdrawn tools must not create an infinite loop
                    rounds += 1
                    ctx_est = progress.ctx_used if progress else ctx_est  # exact at the end of the round…
                    progress = None  # …then tool results are added by estimate until the next round reports
                    for call in pending_calls:
                        name, arguments = call["function"]["name"], call["function"]["arguments"]
                        show(f"tool {name}", force=True)
                        if calls_used >= MAX_TOOL_CALLS:
                            result = "tool-call limit reached: answer with what you have"
                        else:
                            calls_used += 1
                            result = _invoke_tool(cfg, name, arguments)
                        _print_tool_result(result)
                        status.wrote("\n")  # printed lines end at the left margin
                        messages.append({"role": "tool", "tool_call_id": call["id"], "name": name, "content": result})
                        ctx_est += _approx_messages(messages[-1:])
                        awaiting.remove(call)
                    if rounds >= MAX_TOOL_ROUNDS or calls_used >= MAX_TOOL_CALLS:
                        con.print("[yellow]tool-call limit reached; requesting a final answer[/]")
                        status.wrote("\n")
                        tool_specs = None
        except KeyboardInterrupt as e:
            # Keep the turn as far as it got, so the next prompt can build on it. A thinking
            # loop is the usual reason to press Ctrl-C, and losing the thread each time
            # would make the model unusable.
            if in_reasoning:
                sys.stdout.write("\x1b[0m\n")
                sys.stdout.flush()
            if awaiting:
                # Every recorded tool call needs a reply for the history to stay valid.
                note = (
                    "interrupted by the user; the call may have partially run"
                    if isinstance(e, ToolInterrupted)
                    else "not run: the user stopped the turn instead of approving this call"
                )
                for call in awaiting:
                    messages.append({"role": "tool", "tool_call_id": call["id"], "name": call["function"]["name"], "content": note})
                    note = "not run: interrupted by the user"
            elif round_text or round_reasoning:
                messages.append(_assistant_message(round_text, round_reasoning))
            if messages[-1]["role"] != "assistant":  # keep roles alternating for strict templates
                messages.append({"role": "assistant", "content": NO_OUTPUT})
            interrupted = True
            last_reply = "".join(round_text)
            con.print(f"\n[yellow]interrupted after {time.time() - t0:.1f}s · thinking and output so far are kept in the conversation[/]")
            persist()
            continue
        except Exception as e:  # noqa: BLE001
            con.print(f"\n[red]{e}[/]")
            notify("lethe: the reply failed")
            del messages[turn_start:]
            continue
        con.print()
        last_reply = "".join(round_text)
        messages.append(_assistant_message(round_text, round_reasoning))
        interrupted = False
        if last:
            con.print(f"[dim]{last.line()} · wall {time.time() - t0:.1f}s[/]")
            if last.ctx_frac >= 0.8:
                con.print(f"[yellow]context {last.ctx_frac:.0%} full — /compact to summarize, or /reset[/]")
            if last.finish_reason == "length" and not cfg.max_tokens:
                con.print("[yellow]reply was cut off by the context limit — /compact[/]")
        notify("lethe: reply finished")
        persist()
        con.print()
    if saving:
        con.print(_terminal_safe(f"saved {conv.id} · resume with: {cfg.resume_command(conv.id)}"),
                  style="dim", markup=False, highlight=False, soft_wrap=True)
    elif cfg.resume:
        new_turns = sum(1 for m in messages if m["role"] == "user") - conv.turns
        if new_turns > 0:
            con.print(f"[dim]{conv.id} is unchanged on disk: this session's {new_turns} new turn(s) were not saved "
                      "(next time, --resume with --save, or /save during the chat)[/]")
