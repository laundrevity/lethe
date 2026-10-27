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
@pytest.mark.parametrize("tool_name", ["run_shell", "run_ssh"])
def test_tool_arguments_stream_before_approval_and_only_complete_calls_run(monkeypatch, ending, tool_name):
    output = io.StringIO()
    monkeypatch.setattr(repl.sys, "stdout", output)
    monkeypatch.setattr(repl, "con", Console(file=output, color_system=None, width=1000))
    inputs = iter(["test question", "/quit"])
    monkeypatch.setattr(repl, "_make_reader", lambda: lambda prompt: next(inputs))
    saved, approved, executed, requests = [], [], [], []
    monkeypatch.setattr(repl.vault, "save", lambda conv, pw: saved.append(copy.deepcopy(conv.messages)))
    command = "printf " + "x" * 2000 + " tail"
    source = 'import json\nprint(json.dumps({"message": "' + "x" * 2000 + '", "path": r"a\\b", "emoji": "😀"}))\n# tail'
    payload = {"command": command} if tool_name == "run_shell" else {"host": "oswald", "python": source}
    arguments = json.dumps(payload)
    finished = False

    def approve():
        assert finished  # syntax-complete JSON alone must never trigger approval
        assert (arguments if tool_name == "run_shell" else source) in output.getvalue()
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
            assert "Checking now.\n⚙ tool call 1: run_" in output.getvalue()
            if tool_name == "run_ssh":
                split = arguments.index("\\n") + 1  # JSON escape itself crosses two SSE events
                yield chunk({"tool_calls": [{"index": 0, "function": {"name": "ssh", "arguments": arguments[:split]}}]})
                assert "import json" in output.getvalue() and "print(json" not in output.getvalue()
                yield chunk({"tool_calls": [{"index": 0, "function": {"arguments": arguments[split:1000]}}]})
                assert 'import json\nprint(json.dumps({"message": "' in output.getvalue()
            else:
                yield chunk({"tool_calls": [{"index": 0, "function": {"name": "shell", "arguments": arguments[:1000]}}]})
                assert arguments[:1000] in output.getvalue()
            assert " tail" not in output.getvalue()
            assert not approved and not executed and not saved
            if ending == "interrupt":
                raise KeyboardInterrupt
            yield chunk({"tool_calls": [{"index": 0, "function": {"arguments": arguments[1000:]}}]})
            assert (arguments if tool_name == "run_shell" else source) in output.getvalue()
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
    registry = {tool_name: tools.Tool(tool_name, "test", {
        "type": "object", "properties": {key: {"type": "string"} for key in payload}, "required": list(payload),
    }, lambda **args: executed.append(args) or "result", requires_confirmation=True,
        preview=(lambda host, python: f"{host} (Python)\n{python}") if tool_name == "run_ssh" else None)}
    repl.run(repl.ReplConfig("http://localhost", None, 4096, "test-model", Profile("test", {}, {}),
                             tools=registry, save=True, passphrase="pw"))
    if ending == "complete":
        assert approved == [True] and executed == [payload]
        shown = output.getvalue()
        assert shown.count(arguments if tool_name == "run_shell" else source) == 1
        assert "⚙ '" + tool_name + "'" not in shown
        assert len(requests) == 2
        message = saved[0][1]
        assert message["content"] == "Checking now." and message["reasoning_content"] == "Need a command."
        assert message["tool_calls"] == [{"id": "c1", "type": "function", "function": {
            "name": tool_name, "arguments": arguments,
        }}]
        assert requests[1]["messages"][-1]["role"] == "tool"
        assert requests[1]["messages"][-2]["tool_calls"][0]["function"]["arguments"] == arguments
    else:
        assert not approved and not executed and len(requests) == 1
        if ending == "interrupt":
            assert saved[0][1] == {"role": "assistant", "content": "Checking now.", "reasoning_content": "Need a command."}
            assert "interrupted" in output.getvalue()
        else:
            assert not saved and "incomplete tool-call response" in output.getvalue()
