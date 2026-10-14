"""Ctrl-C keeps the turn as far as it got; the next prompt tells the model it was cut short."""

import io

import pytest
from rich.console import Console

from lethe import repl, tools
from lethe.profiles import Profile


def make_config(**kwargs):
    return repl.ReplConfig("http://localhost", None, 4096, "test-model", Profile("test", {}, {}), **kwargs)


def make_tool(fn, confirm=False):
    return tools.Tool(
        name="run_shell", description="shell", fn=fn, requires_confirmation=confirm,
        parameters={"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    )


def call(number):
    return {"id": f"c{number}", "type": "function", "function": {"name": "run_shell", "arguments": '{"command":"ls"}'}}


def drive(monkeypatch, inputs, stream, **config):
    """Run the chat with scripted input and a scripted model; return the requests the model saw."""
    seen = []
    lines = iter([*inputs, "/quit"])
    monkeypatch.setattr(repl, "_make_reader", lambda: lambda prompt: next(lines))
    monkeypatch.setattr(repl, "con", Console(file=io.StringIO(), color_system=None, width=200))

    def recording_stream(url, messages, **kwargs):
        seen.append([dict(m) for m in messages])
        yield from stream(len(seen))

    monkeypatch.setattr(repl, "stream_chat", recording_stream)
    repl.run(make_config(**config))
    return seen


def roles(messages):
    return [m["role"] for m in messages]


def test_interrupt_mid_thinking_keeps_the_reasoning_for_the_next_turn(monkeypatch):
    def stream(request):
        if request == 1:
            yield repl.Delta(reasoning="The code is 4417, ")
            yield repl.Delta(reasoning="so")
            raise KeyboardInterrupt
        yield repl.Delta(content="4417")
        yield repl.TurnStats()

    seen = drive(monkeypatch, ["remember 4417", "what was it?"], stream)
    assert len(seen) == 2
    kept = seen[1]
    assert roles(kept) == ["user", "assistant", "user"]
    assert kept[0]["content"] == "remember 4417"
    assert kept[1] == {"role": "assistant", "content": "", "reasoning_content": "The code is 4417, so"}
    # The next prompt explains the cut-off message; the one after a finished turn does not.
    assert kept[2]["content"].startswith(repl.INTERRUPT_NOTE) and kept[2]["content"].endswith("\n\nwhat was it?")

    seen = drive(monkeypatch, ["one", "two"], lambda request: iter([repl.Delta(content="done"), repl.TurnStats()]))
    assert seen[1][-1]["content"] == "two"


def test_interrupt_before_any_output_or_during_a_tool_leaves_valid_history(monkeypatch):
    def stream(request):
        if request == 1:
            raise KeyboardInterrupt
        yield repl.Delta(content="ok")
        yield repl.TurnStats()

    seen = drive(monkeypatch, ["hello", "again"], stream)
    assert roles(seen[1]) == ["user", "assistant", "user"]
    assert seen[1][1]["content"] == repl.NO_OUTPUT

    # Ctrl-C while the second of two tool calls runs: the first keeps its result, the
    # running one and the rest get a reply saying so, and the turn ends on an assistant message.
    runs = []

    def slow(**kwargs):
        runs.append(kwargs)
        if len(runs) == 2:
            raise KeyboardInterrupt
        return "first result"

    def stream(request):
        if request == 1:
            yield repl.Delta(reasoning="Need three commands.")
            yield repl.Delta(tool_calls=[call(1), call(2), call(3)])
            yield repl.TurnStats()
        else:
            yield repl.Delta(content="ok")
            yield repl.TurnStats()

    seen = drive(monkeypatch, ["run things", "and?"], stream, tools={"run_shell": make_tool(slow)}, auto_approve="all")
    kept = seen[1]
    assert roles(kept) == ["user", "assistant", "tool", "tool", "tool", "assistant", "user"]
    assert kept[1]["reasoning_content"] == "Need three commands."
    assert [m["content"] for m in kept[2:5]] == [
        "first result",
        "interrupted by the user; the call may have partially run",
        "not run: interrupted by the user",
    ]
    assert [m["tool_call_id"] for m in kept[2:5]] == ["c1", "c2", "c3"]
    assert len(runs) == 2
    assert kept[6]["content"].startswith(repl.INTERRUPT_NOTE)


def test_ctrl_c_at_the_approval_prompt_stops_the_turn_without_running_the_call(monkeypatch):
    def refuse(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(repl, "ask_approval", refuse)
    runs = []

    def stream(request):
        if request == 1:
            yield repl.Delta(content="Let me look.")
            yield repl.Delta(tool_calls=[call(1)])
            yield repl.TurnStats()
        else:
            yield repl.Delta(content="ok")
            yield repl.TurnStats()

    seen = drive(monkeypatch, ["look", "ok"], stream, tools={"run_shell": make_tool(lambda **kw: runs.append(kw) or "ran", confirm=True)})
    kept = seen[1]
    assert roles(kept) == ["user", "assistant", "tool", "assistant", "user"]
    assert kept[1]["content"] == "Let me look." and kept[1]["tool_calls"] == [call(1)]
    assert kept[2]["content"] == "not run: the user stopped the turn instead of approving this call"
    assert not runs


def test_note_goes_into_the_first_text_part_of_a_multipart_prompt():
    parts = [{"type": "image_url", "image_url": {"url": "data:x"}}, {"type": "text", "text": "hi"}]
    noted = repl._with_note(parts, "[note]")
    assert noted[1]["text"] == "[note]\n\nhi" and noted[0] == parts[0] and parts[1]["text"] == "hi"
    assert repl._with_note([{"type": "image_url", "image_url": {"url": "data:x"}}], "[note]")[0] == {"type": "text", "text": "[note]"}


@pytest.mark.parametrize("saved", [True, False])
def test_the_flag_survives_a_saved_conversation(monkeypatch, tmp_path, saved):
    from lethe import vault

    conversation = vault.Conversation.new("test-model")
    conversation.messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "", "reasoning_content": "hm"}]
    conversation.interrupted = saved
    monkeypatch.setattr(vault, "vault_dir", lambda: tmp_path)
    vault.save(conversation, "pw")
    assert vault.load(conversation.id, "pw").interrupted is saved

    seen = drive(monkeypatch, ["go on"], lambda request: iter([repl.Delta(content="ok"), repl.TurnStats()]), resume=vault.load(conversation.id, "pw"))
    assert seen[0][-1]["content"].startswith(repl.INTERRUPT_NOTE) is saved
