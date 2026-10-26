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
    assert events[0].tool_call_deltas == call_chunks()[0]["choices"][0]["delta"]["tool_calls"]
    assert events[1].tool_call_deltas == call_chunks()[1]["choices"][0]["delta"]["tool_calls"]
    assert events[0].tool_calls is None and events[1].tool_calls is None
    assert events[2].tool_calls == [{"id": "abc", "type": "function", "function": {"name": "run_shell", "arguments": '{"command":"ls"}'}}]
    assert events[-1].finish_reason == "tool_calls"


@pytest.mark.parametrize("finish,done", [("tool_calls", False), ("length", True), (None, True)])
def test_incomplete_tool_calls_only_emit_display_fragments(monkeypatch, finish, done):
    install_stream(monkeypatch, call_chunks(finish), done)
    events = []
    with pytest.raises(RuntimeError, match="incomplete tool-call response"):
        for event in chat.stream_chat("http://localhost", [], n_ctx=100):
            events.append(event)
    assert len(events) == 2 and all(e.tool_calls is None and e.tool_call_deltas for e in events)


def test_interleaved_tool_fragments_keep_their_indices_and_assemble_in_order(monkeypatch):
    fragments = [
        [{"index": 1, "id": "second", "function": {"name": "read_", "arguments": '{"path":'}}],
        [{"index": 0, "id": "first", "function": {"name": "run_", "arguments": '{"command":'}}],
        [{"index": 1, "function": {"name": "file", "arguments": '"README.md"}'}}],
        [{"index": 0, "function": {"name": "shell", "arguments": '"pwd"}'}}],
    ]
    chunks = [{"choices": [{"delta": {"tool_calls": f, "content": "Working." if i == 0 else ""}}],
               "timings": {"predicted_n": i + 1}} for i, f in enumerate(fragments)]
    install_stream(monkeypatch, chunks + [{"choices": [{"delta": {}, "finish_reason": "stop"}]}])
    events = list(chat.stream_chat("http://localhost", [], n_ctx=100))
    assert [e.tool_call_deltas for e in events[:4]] == fragments
    assert events[0].content == "Working."
    assert [e.timings["predicted_n"] for e in events[:4]] == [1, 2, 3, 4]
    assert events[-2].tool_calls == [
        {"id": "first", "type": "function", "function": {"name": "run_shell", "arguments": '{"command":"pwd"}'}},
        {"id": "second", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"README.md"}'}},
    ]


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
