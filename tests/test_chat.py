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


def test_stream_reports_prompt_progress_and_running_timings(monkeypatch):
    requests = []
    chunks = [
        {"choices": [{"delta": {}}], "prompt_progress": {"total": 100, "cache": 0, "processed": 40, "time_ms": 10}},
        {"choices": [{"delta": {"content": "Hi"}}], "timings": {"prompt_n": 100, "cache_n": 20, "predicted_n": 1, "predicted_ms": 50.0}},
        {"choices": [{"delta": {}, "finish_reason": "stop"}], "timings": {"prompt_n": 100, "cache_n": 20, "predicted_n": 2, "predicted_ms": 100.0}},
    ]
    payload = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    client_type = httpx.Client

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, text=payload)

    monkeypatch.setattr(chat.httpx, "Client", lambda **kwargs: client_type(transport=httpx.MockTransport(respond), **kwargs))
    events = list(chat.stream_chat("http://localhost", [], n_ctx=1000))
    assert requests[0]["timings_per_token"] is True and requests[0]["return_progress"] is True
    assert events[0].progress == {"total": 100, "cache": 0, "processed": 40, "time_ms": 10} and events[0].content == ""
    assert events[1].content == "Hi" and events[1].timings["predicted_n"] == 1
    assert events[2].content == "" and events[2].timings["predicted_n"] == 2
    stats = events[-1]
    assert isinstance(stats, chat.TurnStats) and stats.ctx_used == 122 and stats.finish_reason == "stop"
    assert chat.TurnStats.from_timings(events[1].timings, 1000, None).ctx_used == 121
