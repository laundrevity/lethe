"""Saved chats retain reasoning and replay tool activity without executing it again."""

import copy
import io
import json

import pytest
from rich.console import Console

from lethe import repl, tools, vault
from lethe.profiles import Profile


def call(call_id, command):
    return {"id": call_id, "type": "function", "function": {
        "name": "run_shell", "arguments": json.dumps({"command": command}),
    }}


def drive(monkeypatch, inputs, **kwargs):
    output = io.StringIO()
    lines = iter([*inputs, "/quit"])
    monkeypatch.setattr(repl.sys, "stdout", output)
    monkeypatch.setattr(repl, "con", Console(file=output, color_system=None, width=1000))
    monkeypatch.setattr(repl, "_make_reader", lambda: lambda prompt: next(lines))
    repl.run(repl.ReplConfig("http://localhost", None, 4096, "test-model", Profile("test", {}, {}), **kwargs))
    return output.getvalue()


@pytest.mark.parametrize("show_thinking", [True, False])
def test_resume_replays_reasoning_and_tool_results_in_order_without_running_tools(monkeypatch, show_thinking):
    first, second = call("c1", "first command"), call("c2", "second command")
    conv = vault.Conversation.new("test-model")
    conv.messages = [
        {"role": "system", "content": "private system prompt"},
        {"role": "user", "content": "look at the files"},
        {"role": "assistant", "content": None, "reasoning_content": "Need two commands.", "tool_calls": [first, second]},
        {"role": "tool", "tool_call_id": "c1", "name": "run_shell", "content": "first result"},
        {"role": "tool", "tool_call_id": "c2", "name": "run_shell", "content": "second result"},
        {"role": "assistant", "content": "Here is the answer.", "reasoning_content": "I have the results."},
        {"role": "user", "content": "go on"},
        {"role": "assistant", "content": "", "reasoning_content": "Unfinished thought"},
    ]
    original = copy.deepcopy(conv.messages)

    def forbidden(*args, **kwargs):
        pytest.fail("replaying history must not request approval, execute tools, or contact the model")

    monkeypatch.setattr(repl, "_invoke_tool", forbidden)
    monkeypatch.setattr(repl, "stream_chat", forbidden)
    monkeypatch.setattr(repl, "ask_approval", forbidden)
    shown = drive(monkeypatch, [], resume=conv, show_thinking=show_thinking)
    markers = ["look at the files"]
    if show_thinking:
        markers.append("Need two commands.")
    markers += ["first command", "→ 12 chars", "second command", "→ 13 chars"]
    if show_thinking:
        markers.append("I have the results.")
    markers += ["Here is the answer.", "go on"]
    if show_thinking:
        markers.append("Unfinished thought")
    positions = [shown.index(marker) for marker in markers]
    assert positions == sorted(positions)
    assert shown.count("thinking…") == (3 if show_thinking else 0)
    if not show_thinking:
        assert all(m["reasoning_content"] not in shown for m in conv.messages if "reasoning_content" in m)
    assert "private system prompt" not in shown
    assert conv.messages == original


@pytest.mark.parametrize("save_at_start", [True, False])
def test_completed_reasoning_survives_encrypted_save_and_resume_even_when_hidden(monkeypatch, tmp_path, save_at_start):
    monkeypatch.setenv("LETHE_ROOT", str(tmp_path))
    monkeypatch.setattr(vault, "SCRYPT_N", 2**12)
    monkeypatch.setattr(repl, "ask_passphrase", lambda confirm: "pw")
    tool_call = call("c1", "saved command")
    requests = []

    def stream(url, messages, **kwargs):
        requests.append(copy.deepcopy(messages))
        if len(requests) == 1:
            yield repl.Delta(reasoning="First ")
            yield repl.Delta(reasoning="thought.")
            yield repl.Delta(content="Checking now.")
            yield repl.Delta(tool_calls=[tool_call])
        else:
            yield repl.Delta(reasoning="Final ")
            yield repl.Delta(reasoning="thought.")
            yield repl.Delta(content="Saved answer.")
        yield repl.TurnStats()

    monkeypatch.setattr(repl, "stream_chat", stream)
    registry = {"run_shell": tools.Tool("run_shell", "test", {
        "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"],
    }, lambda command: "saved result")}
    inputs = ["question"] + ([] if save_at_start else ["/save"])
    shown = drive(monkeypatch, inputs, save=save_at_start, passphrase="pw", tools=registry, show_thinking=False)
    assert "First thought." not in shown and "Final thought." not in shown
    entry, = vault.list_entries()
    conv = vault.load(entry.id, "pw")
    assistant = [m for m in conv.messages if m["role"] == "assistant"]
    assert [m["reasoning_content"] for m in assistant] == ["First thought.", "Final thought."]
    assert assistant[0]["tool_calls"] == [tool_call]
    assert conv.messages[2]["content"] == "saved result"
    assert requests[1][1]["reasoning_content"] == "First thought."
    raw = entry.path.read_bytes()
    assert all(secret not in raw for secret in (b"First thought.", b"Final thought.", b"saved command", b"saved result"))
    assert list(entry.path.parent.iterdir()) == [entry.path]
    replayed = drive(monkeypatch, [], resume=conv)
    for text in ("First thought.", "Final thought.", "saved command", "Saved answer."):
        assert text in replayed
    assert len(requests) == 2


def test_resume_supports_older_messages_and_escapes_terminal_controls(monkeypatch):
    command = "x" * 200 + "[conceal] tail\x1b[2J"
    conv = vault.Conversation.new("test-model")
    conv.messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "[bold]question\x1b[2J"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,not-for-display"}},
        ]},
        {"role": "assistant", "tool_calls": [call("c1", command)]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "assistant", "content": "[red]answer\r\u202e", "reasoning_content": "[bold]thought\x1b]52;clipboard\x07"},
        {"role": "user", "content": "plain question"},
        {"role": "assistant", "content": "plain answer"},
    ]
    shown = drive(monkeypatch, [], resume=conv)
    assert "[bold]question\\x1b[2J [1 image(s)]" in shown
    assert "not-for-display" not in shown
    assert "x" * 200 + "[conceal] tail\\u001b[2J" in shown
    assert "[bold]thought\\x1b]52;clipboard\\x07" in shown
    assert "[red]answer\\x0d\\u202e" in shown
    assert "plain question" in shown and "plain answer" in shown
    assert "\x1b[2J" not in shown and "\x1b]52" not in shown


def test_replay_keeps_unanswered_calls_and_handles_reused_ids(monkeypatch):
    conv = vault.Conversation.new("test-model")
    conv.messages = [
        {"role": "assistant", "content": None, "tool_calls": [call("c1", "unanswered command")]},
        {"role": "user", "content": "try again"},
        {"role": "assistant", "content": None, "tool_calls": [call("c1", "retry command")]},
        {"role": "tool", "tool_call_id": "c1", "content": "retried"},
        {"role": "assistant", "content": None, "tool_calls": [call("c1", "last unanswered command")]},
    ]
    shown = drive(monkeypatch, [], resume=conv)
    markers = ["unanswered command", "try again", "retry command", "→ 7 chars", "last unanswered command"]
    positions = [shown.index(marker) for marker in markers]
    assert positions == sorted(positions)
    assert shown.count("⚙") == 3 and shown.count("→") == 1


def test_tool_limit_stop_does_not_duplicate_reasoning_in_the_saved_history(monkeypatch):
    saved = []
    monkeypatch.setattr(vault, "save", lambda conv, passphrase: saved.append(copy.deepcopy(conv.messages)))
    monkeypatch.setattr(repl, "stream_chat", lambda *a, **kw: iter([
        repl.Delta(reasoning="Want a disabled tool."),
        repl.Delta(tool_calls=[call("c1", "disabled command")]),
        repl.TurnStats(),
    ]))
    drive(monkeypatch, ["question"], save=True, passphrase="pw")
    messages, = saved
    assistant = [m for m in messages if m["role"] == "assistant"]
    assert assistant[0]["reasoning_content"] == "Want a disabled tool."
    assert assistant[1]["content"].startswith("Stopped:")
    assert "reasoning_content" not in assistant[1]


def test_resume_alone_never_writes_and_slash_save_reuses_the_passphrase(monkeypatch, tmp_path):
    monkeypatch.setenv("LETHE_ROOT", str(tmp_path))
    monkeypatch.setattr(vault, "SCRYPT_N", 2**12)
    conv = vault.Conversation.new("test-model")
    conv.messages = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "before"}]
    vault.save(conv, "pw")
    stored = vault.path_for(conv.id).read_bytes()
    monkeypatch.setattr(repl, "stream_chat", lambda *a, **kw: iter([repl.Delta(content="later"), repl.TurnStats()]))
    monkeypatch.setattr(repl, "ask_passphrase", lambda confirm: pytest.fail("the passphrase from --resume must be reused"))
    shown = drive(monkeypatch, ["new question"], resume=vault.load(conv.id, "pw"), passphrase="pw")
    assert vault.path_for(conv.id).read_bytes() == stored
    assert "not saving" in shown and "1 new turn(s) were not saved" in shown
    shown = drive(monkeypatch, ["new question", "/save"], resume=vault.load(conv.id, "pw"), passphrase="pw")
    assert f"saving encrypted as {conv.id}" in shown and "were not saved" not in shown
    assert vault.load(conv.id, "pw").turns == 2
