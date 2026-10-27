"""The status line, notifications and key encodings touch only the terminal, and only when there is one."""

import io
import json
import os

import pytest
from rich.console import Console

from lethe import repl, terminal


class FakeTTY(io.StringIO):
    def isatty(self):
        return True

    def fileno(self):
        return 99


def test_status_line_reserves_the_bottom_row_and_gives_it_back(monkeypatch):
    monkeypatch.setattr(terminal.os, "get_terminal_size", lambda fd: os.terminal_size((80, 24)))
    out = FakeTTY()
    with terminal.StatusLine(out) as status:
        assert status.active
        assert out.getvalue() == "\x1bD\x1b[A\x1b7\x1b[1;23r\x1b8"  # scroll region above a free last row
        out.seek(0), out.truncate()
        status.update("ctx 1/2", force=True)
        assert out.getvalue() == "\x1b7\x1b[24;1H\x1b[0;2mctx 1/2\x1b[0m\x1b[K\x1b8\x1b[0m"
        out.seek(0), out.truncate()
        status.update("x" * 200, sgr="\x1b[2;3m", force=True)
        drawn = out.getvalue()
        assert "x" * 79 + "\x1b[0m" in drawn and "x" * 80 not in drawn  # never wraps on the last row
        assert drawn.endswith("\x1b8\x1b[2;3m")  # the caller's attributes come back
    assert out.getvalue().endswith("\x1b7\x1b[24;1H\x1b[2K\x1b[r\x1b8")
    assert not status.active


def test_status_line_waits_while_the_cursor_sits_at_the_right_margin(monkeypatch):
    monkeypatch.setattr(terminal.os, "get_terminal_size", lambda fd: os.terminal_size((80, 24)))
    out = FakeTTY()
    with terminal.StatusLine(out) as status:
        status.wrote("thinking… " + "a" * 70)  # exactly 80 cells: Apple Terminal would lose the pending wrap
        assert status.at_margin
        out.seek(0), out.truncate()
        status.update("busy", force=True)
        assert out.getvalue() == ""
        status.wrote("b")
        assert not status.at_margin
        status.update("busy", force=True)
        assert "busy" in out.getvalue()
        status.wrote("日本")  # wide characters take two cells
        assert status.col == 85
        status.wrote("\tx\n")
        assert status.col == 0
        status.wrote("é")  # a combining mark takes none
        assert status.col == 1


def test_status_line_throttles_and_follows_a_resize(monkeypatch):
    size = [os.terminal_size((80, 24))]
    monkeypatch.setattr(terminal.os, "get_terminal_size", lambda fd: size[0])
    clock = [100.0]
    monkeypatch.setattr(terminal.time, "monotonic", lambda: clock[0])
    out = FakeTTY()
    with terminal.StatusLine(out) as status:
        status.update("one", force=True)
        out.seek(0), out.truncate()
        status.update("two")  # too soon
        assert out.getvalue() == "" and status.text == "two"
        clock[0] += 1
        status.update("three")
        assert "three" in out.getvalue()
        size[0] = os.terminal_size((100, 30))
        out.seek(0), out.truncate()
        status.update("four", force=True)
        assert out.getvalue() == "\x1b7\x1b[r\x1b8" + "\x1bD\x1b[A\x1b7\x1b[1;29r\x1b8" + "\x1b7\x1b[30;1H\x1b[0;2mfour\x1b[0m\x1b[K\x1b8\x1b[0m"
        assert (status.rows, status.cols) == (30, 100)


def test_status_line_is_inert_without_a_terminal_or_room(monkeypatch):
    out = io.StringIO()
    with terminal.StatusLine(out) as status:
        status.update("anything", force=True)
    assert out.getvalue() == "" and not status.active
    monkeypatch.setattr(terminal.os, "get_terminal_size", lambda fd: os.terminal_size((80, 3)))
    out = FakeTTY()
    with terminal.StatusLine(out) as status:
        status.update("anything", force=True)
    assert out.getvalue() == "" and not status.active


def test_tool_stream_renders_split_names_and_arguments_without_waiting():
    out = io.StringIO()
    status = terminal.StatusLine(out)
    out.write("Assistant text")
    status.wrote("Assistant text")
    with terminal.ToolCallStream(status) as stream:
        stream.feed([{"index": 0, "function": {"name": "run_"}}])
        assert out.getvalue() == "Assistant text\n⚙ tool call 1: run_"
        stream.feed([{"index": 0, "function": {"name": "shell", "arguments": '{"command":"head'}}])
        assert out.getvalue().endswith('run_shell\n{"command":"head')
        stream.feed([{"index": 0, "function": {"arguments": ' tail"}'}}])
        assert out.getvalue().endswith('{"command":"head tail"}')
    assert out.getvalue().endswith('tail"}\n') and status.col == 0


def test_tool_stream_labels_interleaved_calls_and_escapes_controls_on_abort():
    out = io.StringIO()
    status = terminal.StatusLine(out)
    with pytest.raises(KeyboardInterrupt), terminal.ToolCallStream(status) as stream:
        stream.feed([{"index": 1, "function": {"name": "run_shell", "arguments": "first\x1b"}}])
        stream.feed([{"index": 0, "function": {"name": "read_file", "arguments": "second"}}])
        stream.feed([{"index": 1, "function": {"arguments": "[2J\r\u202e[bold]tail"}}])
        raise KeyboardInterrupt
    shown = out.getvalue()
    assert shown == (
        "⚙ tool call 2: run_shell\nfirst\\x1b\n"
        "⚙ tool call 1: read_file\nsecond\n"
        "⚙ tool call 2: run_shell\n[2J\\x0d\\u202e[bold]tail\n"
    )
    assert status.col == 0


def test_tool_stream_tracks_wrapping_for_status_redraws(monkeypatch):
    monkeypatch.setattr(terminal.os, "get_terminal_size", lambda fd: os.terminal_size((80, 24)))
    out = FakeTTY()
    with terminal.StatusLine(out) as status, terminal.ToolCallStream(status) as stream:
        stream.feed([{"function": {"name": "write_file", "arguments": "x" * 80}}])
        before = out.getvalue()
        status.update("tool call", force=True)
        assert status.at_margin and out.getvalue() == before
        stream.feed([{"function": {"arguments": "y"}}])
        status.update("tool call", force=True)
        assert not status.at_margin and "\x1b[24;1H" in out.getvalue()[len(before):]


def _finished_call(name="run_shell", arguments='{"command":"pwd"}', call_id="first"):
    return {"id": call_id, "function": {"name": name, "arguments": arguments}}


def test_stream_receipt_matches_exact_complete_name_arguments_and_id():
    out = io.StringIO()
    with terminal.ToolCallStream(terminal.StatusLine(out)) as stream:
        stream.feed([{"index": 2, "id": "first", "function": {"name": "run_"}}])
        stream.feed([{"index": 2, "function": {"name": "shell", "arguments": '{"command":'}}])
        stream.feed([{"index": 2, "function": {"arguments": '"pwd"}'}}])
    assert stream.shown_call(_finished_call()) == 3
    assert stream.shown_call(_finished_call(name="other")) is None
    assert stream.shown_call(_finished_call(arguments='{"command":"pwd; hidden"}')) is None
    assert stream.shown_call(_finished_call(call_id="different")) is None
    # The receipt is exact, not a semantic JSON comparison.
    assert stream.shown_call(_finished_call(arguments='{"command": "pwd"}')) is None


def test_stream_receipt_uses_default_ids_and_refuses_unseen_or_ambiguous_calls():
    with terminal.ToolCallStream(terminal.StatusLine(io.StringIO())) as stream:
        assert stream.shown_call(_finished_call(call_id="call_0")) is None
        stream.feed([{"function": {"name": "run_shell", "arguments": '{"command":"pwd"}'}}])
        stream.feed([{"index": 4, "function": {"name": "read_file"}}])
    assert stream.shown_call(_finished_call(call_id="call_0")) == 1
    assert stream.shown_call({"function": _finished_call()["function"]}) == 1
    assert stream.shown_call(_finished_call(name="read_file", arguments="", call_id="call_4")) is None
    stream.feed([{"index": 4, "id": "call_0", "function": {"arguments": "{}"}}])
    assert stream.shown_call(_finished_call(call_id="call_0")) is None


def test_stream_receipt_requires_one_contiguous_block():
    with terminal.ToolCallStream(terminal.StatusLine(io.StringIO())) as stream:
        stream.feed([{"id": "first", "function": {"name": "run_shell", "arguments": '{"command":'}}])
        stream.feed([{"index": 1, "id": "second", "function": {"name": "read_file", "arguments": '{"path":"a"}'}}])
        stream.feed([{"function": {"arguments": '"pwd"}'}}])
    assert stream.shown_call(_finished_call()) is None
    assert stream.shown_call(_finished_call("read_file", '{"path":"a"}', "second")) == 2
    with terminal.ToolCallStream(terminal.StatusLine(io.StringIO())) as stream:
        stream.feed([{"id": "first", "function": {"name": "run_shell", "arguments": '{"command":'}}])
        stream.finish()  # Assistant text may appear before this call resumes.
        stream.feed([{"function": {"arguments": '"pwd"}'}}])
    assert stream.shown_call(_finished_call()) is None


def test_stream_receipt_refuses_a_name_that_finishes_after_arguments():
    args = '{"host":"oswald","python":"print(1)"}'
    with terminal.ToolCallStream(terminal.StatusLine(io.StringIO())) as stream:
        stream.feed([{"id": "first", "function": {"name": "run_", "arguments": args[:9]}}])
        stream.feed([{"function": {"name": "ssh", "arguments": args[9:]}}])
    assert stream.shown_call(_finished_call("run_ssh", args)) is None


def test_ssh_stream_receipt_waits_for_pending_decoded_characters():
    args = '{"python":"\\ud83d'
    with terminal.ToolCallStream(terminal.StatusLine(io.StringIO())) as stream:
        stream.feed([{"id": "first", "function": {"name": "run_ssh", "arguments": args}}])
        assert stream.shown_call(_finished_call("run_ssh", args)) is None
        stream.feed([{"function": {"arguments": '\\ude00"}'}}])
    assert stream.shown_call(_finished_call("run_ssh", args + '\\ude00"}')) == 1


def test_stream_color_survives_redraws_and_never_changes_column_accounting(monkeypatch):
    monkeypatch.setattr(terminal.os, "get_terminal_size", lambda fd: os.terminal_size((80, 24)))
    out = FakeTTY()
    with terminal.StatusLine(out) as status, terminal.ToolCallStream(status, color=True) as stream:
        stream.feed([{"function": {"name": "run_"}}])
        assert "\x1b[36m⚙ tool call 1: run_\x1b[0m" in out.getvalue()
        assert status.col == len("⚙ tool call 1: run_")
        status.update("tool call", force=True)
        stream.feed([{"function": {"name": "shell", "arguments": "x" * 80}}])
        assert "\x1b[36mshell\x1b[0m" in out.getvalue()
        assert "\x1b[36m" + "x" * 80 + "\x1b[0m" in out.getvalue()
        assert status.col == 80 and status.at_margin
        before = out.getvalue()
        status.update("still busy", force=True)
        assert out.getvalue() == before
        stream.feed([{"function": {"arguments": "\x1b[2J"}}])
        assert out.getvalue().endswith("\x1b[36m\\x1b[2J\x1b[0m")


def test_stream_color_is_reset_on_interrupt_and_omitted_for_pipes(monkeypatch):
    out = FakeTTY()
    with pytest.raises(KeyboardInterrupt), terminal.ToolCallStream(terminal.StatusLine(out), color=True) as stream:
        stream.feed([{"function": {"name": "run_shell", "arguments": "part"}}])
        raise KeyboardInterrupt
    assert out.getvalue().endswith("\x1b[0m")
    out.write("next prompt")
    assert out.getvalue().endswith("\x1b[0mnext prompt")
    out = io.StringIO()
    with terminal.ToolCallStream(terminal.StatusLine(out), color=True) as stream:
        stream.feed([{"function": {"name": "run_shell", "arguments": "{}"}}])
    assert "\x1b" not in out.getvalue()
    for variable, value in [("NO_COLOR", "1"), ("TERM", "dumb")]:
        with monkeypatch.context() as env:
            env.setenv(variable, value)
            out = FakeTTY()
            with terminal.ToolCallStream(terminal.StatusLine(out)) as stream:
                stream.feed([{"function": {"name": "run_shell", "arguments": "{}"}}])
            assert "\x1b" not in out.getvalue()


def test_ssh_code_stream_decodes_every_possible_escape_chunk_boundary():
    source = 'import json\nprint("line\\n", "café 😀")\n\t# \x1b[2J\r\u202e'
    arguments = json.dumps({"host": "oswald", "python": source}, separators=(",", ":"))
    expected = '{"host":"oswald",\n"python":"' + terminal.terminal_safe(source) + '"}\n'
    for split in range(len(arguments) + 1):
        out = io.StringIO()
        with terminal.ToolCallStream(terminal.StatusLine(out)) as stream:
            stream.feed([{"function": {"name": "run_"}}])
            stream.feed([{"function": {"name": "ssh", "arguments": arguments[:split]}}])
            stream.feed([{"function": {"arguments": arguments[split:]}}])
        assert out.getvalue() == "⚙ tool call 1: run_ssh\n" + expected, split
        assert "\x1b" not in out.getvalue() and "\r" not in out.getvalue()


def test_ssh_bash_command_array_is_readable_while_still_streaming():
    out = io.StringIO()
    with terminal.ToolCallStream(terminal.StatusLine(out)) as stream:
        stream.feed([{"function": {"name": "run_ssh", "arguments": '{"bash":["cd project",'}}])
        assert out.getvalue().endswith('{"bash":["cd project",\n')
        stream.feed([{"function": {"arguments": '"printf \\"one,two\\n\\""],"host":"oswald"}'}}])
    assert out.getvalue().endswith('{"bash":["cd project",\n"printf "one,two\n""],\n"host":"oswald"}\n')


def test_ssh_preview_preserves_escape_state_across_interleaved_calls():
    out = io.StringIO()
    with terminal.ToolCallStream(terminal.StatusLine(out)) as stream:
        stream.feed([{"index": 0, "function": {"name": "run_ssh", "arguments": '{"python":"emoji: \\ud83d'}}])
        stream.feed([{"index": 1, "function": {"name": "run_shell", "arguments": '{"command":"printf \\n"}'}}])
        stream.feed([{"index": 0, "function": {"arguments": '\\ude00\\nnext"}'}}])
    shown = out.getvalue()
    assert shown == (
        '⚙ tool call 1: run_ssh\n{"python":"emoji: \n'
        '⚙ tool call 2: run_shell\n{"command":"printf \\n"}\n'
        '⚙ tool call 1: run_ssh\n😀\nnext"}\n'
    )


@pytest.mark.parametrize("source", ['{"python":"incomplete' + "\\", r'{"python":"incomplete\u12', r'{"python":"\ud83d'])
def test_ssh_preview_shows_incomplete_escape_on_interrupt(source):
    out = io.StringIO()
    with pytest.raises(KeyboardInterrupt), terminal.ToolCallStream(terminal.StatusLine(out)) as stream:
        for i, character in enumerate(source):
            stream.feed([{"function": {"name": "run_ssh" if i == 0 else "", "arguments": character}}])
        raise KeyboardInterrupt
    assert out.getvalue() == "⚙ tool call 1: run_ssh\n" + source + "\n"


def test_ssh_preview_keeps_invalid_escapes_visible_without_crashing():
    source = r'{"python":"\ud800x\udc00\q\u0x"}'
    out = io.StringIO()
    with terminal.ToolCallStream(terminal.StatusLine(out)) as stream:
        stream.feed([{"function": {"name": "run_ssh", "arguments": source}}])
    assert out.getvalue().endswith(source + "\n")


def test_notify_rings_the_bell_and_sends_no_content(monkeypatch):
    monkeypatch.delenv("KITTY_WINDOW_ID", raising=False)
    monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
    out = FakeTTY()
    terminal.notify("lethe: reply finished", out)
    assert out.getvalue() == "\a"  # Apple Terminal has nothing beyond the bell
    monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
    out = FakeTTY()
    terminal.notify("m", out)
    assert out.getvalue() == "\a\x1b]9;m\x1b\\"
    monkeypatch.setenv("KITTY_WINDOW_ID", "1")
    out = FakeTTY()
    terminal.notify("m", out)
    assert out.getvalue() == "\a\x1b]99;i=1:d=0;m\x1b\\"
    out = io.StringIO()
    terminal.notify("m", out)
    assert out.getvalue() == ""


def test_modified_enter_is_a_newline_and_editing_keys_survive_the_kitty_protocol():
    from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
    from prompt_toolkit.input.vt100_parser import Vt100Parser

    ANSI_SEQUENCES.update(repl.key_sequences())

    def keys(sequence):
        got = []
        parser = Vt100Parser(lambda press: got.append(press.key))
        parser.feed(sequence)
        parser.flush()
        return [k.value if hasattr(k, "value") else k for k in got]

    assert keys("\r") == ["c-m"] and keys("\n") == ["c-j"]
    for sequence in ("\x1b[13;2u", "\x1b[13;5u", "\x1b[13;3u", "\x1b[27;2;13~", "\x1b[27;5;13~", "\x1b[27;3;13~"):
        assert keys(sequence) == ["c-j"], sequence
    assert keys("\x1b\r") == ["escape", "c-m"] and keys("\x1b\n") == ["escape", "c-j"]
    assert keys("\x1b[99;5u") == ["c-c"] and keys("\x1b[100;5u") == ["c-d"] and keys("\x1b[106;5u") == ["c-j"]
    assert keys("\x1b[97;5u") == ["c-a"] and keys("\x1b[27u") == ["escape"] and keys("\x1bOM") == ["c-m"]
    assert keys("\x1b[13u") == ["c-m"] and keys("\x1b[9;2u") == ["s-tab"]


def test_approval_prompt_understands_y_n_and_p_with_a_note(monkeypatch):
    output = io.StringIO()
    monkeypatch.setattr(repl, "con", Console(file=output, color_system=None, width=200))
    answers = iter(["maybe", "y", "", "n", "p use ninja instead", "p", "  be careful  ", "P", "", "YES", "y please", "no"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    assert repl.ask_approval() is True  # "maybe" is explained, then y
    assert "y = run it" in output.getvalue() and "[y/n/p]" in output.getvalue()
    assert repl.ask_approval() is False  # Enter alone declines
    assert repl.ask_approval() is False
    assert repl.ask_approval() == "use ninja instead"
    assert repl.ask_approval() == "be careful"  # p, then the note on the next line
    assert repl.ask_approval() is False  # p with nothing to say
    assert repl.ask_approval() is True
    assert repl.ask_approval() is False  # "y please" is explained; then no
