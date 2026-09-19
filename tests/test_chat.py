import json

import httpx
import pytest

from lethe import chat


def install_stream(monkeypatch, chunks, done=True):
    payload = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
    if done:
        payload += "data: [DONE]\n\n"
    client_type = httpx.Client
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=payload))
    monkeypatch.setattr(chat.httpx, "Client", lambda **kwargs: client_type(transport=transport, **kwargs))


def call_chunks(finish="tool_calls"):
    return [
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "abc", "function": {"name": "run_shell", "arguments": '{"command":'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"ls"}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": finish}]},
    ]


def test_stream_assembles_complete_tool_calls(monkeypatch):
    install_stream(monkeypatch, call_chunks())
    events = list(chat.stream_chat("http://localhost", [], n_ctx=100))
    assert events[0].tool_calls == [{"id": "abc", "type": "function", "function": {"name": "run_shell", "arguments": '{"command":"ls"}'}}]
    assert events[-1].finish_reason == "tool_calls"


@pytest.mark.parametrize("finish,done", [("tool_calls", False), ("length", True), (None, True)])
def test_incomplete_tool_calls_are_never_emitted(monkeypatch, finish, done):
    install_stream(monkeypatch, call_chunks(finish), done)
    with pytest.raises(RuntimeError, match="incomplete tool-call response"):
        list(chat.stream_chat("http://localhost", [], n_ctx=100))
