"""Tool previews reach the terminal while the HTTP response is still arriving, without executing."""

import copy
import io
import json

import httpx
import pytest
from rich.console import Console

from lethe import chat, repl, tools
from lethe.profiles import Profile


@pytest.mark.parametrize("ending", ["complete", "length", "disconnect", "interrupt"])
def test_tool_arguments_stream_before_approval_and_only_complete_calls_run(monkeypatch, ending):
    output = io.StringIO()
    monkeypatch.setattr(repl.sys, "stdout", output)
    monkeypatch.setattr(repl, "con", Console(file=output, color_system=None, width=1000))
    inputs = iter(["test question", "/quit"])
    monkeypatch.setattr(repl, "_make_reader", lambda: lambda prompt: next(inputs))
    saved, approved, executed, requests = [], [], [], []
    monkeypatch.setattr(repl.vault, "save", lambda conv, pw: saved.append(copy.deepcopy(conv.messages)))
    command = "printf " + "x" * 2000 + " tail"
    arguments = json.dumps({"command": command})
    finished = False

    def approve():
        assert finished  # syntax-complete JSON alone must never trigger approval
        assert arguments in output.getvalue()
        approved.append(True)
        return True

    monkeypatch.setattr(repl, "ask_approval", approve)

    def chunk(delta, finish=None):
        return ("data: " + json.dumps({"choices": [{"delta": delta, "finish_reason": finish}]}) + "\n\n").encode()

    class ResponseStream(httpx.SyncByteStream):
        def __iter__(self):
            nonlocal finished
            yield chunk({"reasoning_content": "Need a command."})
            yield chunk({"content": "Checking now."})
            yield chunk({"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "run_"}}]})
            assert "Checking now.\n⚙ tool call 1 (streaming): run_" in output.getvalue()
            yield chunk({"tool_calls": [{"index": 0, "function": {"name": "shell", "arguments": arguments[:1000]}}]})
            assert arguments[:1000] in output.getvalue() and " tail" not in output.getvalue()
            assert not approved and not executed and not saved
            if ending == "interrupt":
                raise KeyboardInterrupt
            yield chunk({"tool_calls": [{"index": 0, "function": {"arguments": arguments[1000:]}}]})
            assert arguments in output.getvalue()
            assert not approved and not executed and not saved
            yield chunk({}, "length" if ending == "length" else "tool_calls")
            assert not approved and not executed
            if ending != "disconnect":
                finished = True
                yield b"data: [DONE]\n\n"

    def respond(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(200, stream=ResponseStream())
        return httpx.Response(200, content=chunk({"content": "Done."}, "stop") + b"data: [DONE]\n\n")

    client_type = httpx.Client
    monkeypatch.setattr(chat.httpx, "Client", lambda **kwargs: client_type(transport=httpx.MockTransport(respond), **kwargs))
    registry = {"run_shell": tools.Tool("run_shell", "test", {
        "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"],
    }, lambda **args: executed.append(args) or "result", requires_confirmation=True)}
    repl.run(repl.ReplConfig("http://localhost", None, 4096, "test-model", Profile("test", {}, {}),
                             tools=registry, save=True, passphrase="pw"))
    if ending == "complete":
        assert approved == [True] and executed == [{"command": command}]
        assert len(requests) == 2
        message = saved[0][1]
        assert message["content"] == "Checking now." and message["reasoning_content"] == "Need a command."
        assert message["tool_calls"] == [{"id": "c1", "type": "function", "function": {
            "name": "run_shell", "arguments": arguments,
        }}]
        assert requests[1]["messages"][-1]["role"] == "tool"
    else:
        assert not approved and not executed and len(requests) == 1
        if ending == "interrupt":
            assert saved[0][1] == {"role": "assistant", "content": "Checking now.", "reasoning_content": "Need a command."}
            assert "interrupted" in output.getvalue()
        else:
            assert not saved and "incomplete tool-call response" in output.getvalue()
