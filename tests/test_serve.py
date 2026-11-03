"""`lethe serve`: foreground only, loopback only, key from the environment, never printed."""

import socket
import sys
import textwrap
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from lethe import cli, server

KEY = "s3cret-key-for-tests-0123456789"


def invoke(args, env=None):
    return CliRunner().invoke(cli.app, ["serve", *args], env=env if env is not None else {"LETHE_API_KEY": KEY})


def free_port():
    return server.free_port()


class FakeSession:
    def __init__(self, port, key, exit_code=None):
        self.url = server.base_url(port)
        self.api_key = key
        self.exit_code = exit_code

    def wait(self):
        if self.exit_code is None:
            raise KeyboardInterrupt
        return self.exit_code


@pytest.fixture
def fake_model(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_resolve_or_offer", lambda value, yes=False: SimpleNamespace(path=tmp_path / "m.gguf", spec="owner/repo:Q8", size=1))
    monkeypatch.setattr(cli, "_shape", lambda path: SimpleNamespace(arch="qwen35", name="Qwen3.8-27B"))
    monkeypatch.setattr(cli.server, "props", lambda *a: {"default_generation_settings": {"n_ctx": 16384}})


def fake_start(calls, exit_code=None):
    @contextmanager
    def start(lm, **kw):
        calls.append(kw)
        yield FakeSession(kw["port"], kw["api_key"], exit_code)
    return start


@pytest.mark.parametrize("env", [{}, {"LETHE_API_KEY": ""}, {"LETHE_API_KEY": "short"}])
def test_refuses_without_a_strong_key(monkeypatch, env):
    monkeypatch.setattr(cli, "_start", lambda *a, **kw: pytest.fail("model started"))
    monkeypatch.delenv("LETHE_API_KEY", raising=False)
    result = invoke(["--port", str(free_port())], env=env)
    assert result.exit_code == 1
    assert "LETHE_API_KEY" in result.output


def test_port_is_required(monkeypatch):
    monkeypatch.setattr(cli, "_start", lambda *a, **kw: pytest.fail("model started"))
    result = invoke([])
    assert result.exit_code == 2


def test_refuses_a_port_in_use_and_touches_nothing(monkeypatch):
    monkeypatch.setattr(cli, "_start", lambda *a, **kw: pytest.fail("model started"))
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        result = invoke(["--port", str(port)])
    assert result.exit_code == 1
    assert f"port {port} is already in use" in result.output


def test_serves_on_loopback_with_the_users_key_and_never_prints_it(monkeypatch, fake_model):
    calls = []
    monkeypatch.setattr(cli, "_start", fake_start(calls))
    port = free_port()
    result = invoke(["owner/repo:Q8", "--port", str(port), "--parallel", "4", "--ctx", "65536", "--no-thinking"])
    assert result.exit_code == 0, result.output
    (kw,) = calls
    assert kw["port"] == port and kw["parallel"] == 4 and kw["ctx"] == 65536
    assert kw["api_key"] == KEY and kw["thinking"] is False and kw["vision"] is False
    # The Qwen profile's non-thinking sampler becomes the server's default.
    assert kw["extra"] == ["--temp", "0.7", "--top-p", "0.8", "--top-k", "20", "--min-p", "0.0",
                           "--presence-penalty", "1.5", "--repeat-penalty", "1.0"]
    assert f"http://127.0.0.1:{port}/v1" in result.output
    assert "stopped; the model is unloaded" in result.output
    assert KEY not in result.output


def test_server_dying_on_its_own_is_an_error(monkeypatch, fake_model):
    monkeypatch.setattr(cli, "_start", fake_start([], exit_code=3))
    result = invoke(["owner/repo:Q8", "--port", str(free_port())])
    assert result.exit_code == 1
    assert "exited on its own with 3" in result.output


def test_serve_argv_keeps_the_traceless_flags():
    cfg = server.ServeConfig(model=Path("/m.gguf"), n_ctx=65536, port=8091, parallel=4, api_key=KEY,
                             thinking=False, extra=cli._sampler_flags({"temperature": 0.7}))
    argv = cfg.argv()
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--port") + 1] == "8091"
    assert argv[argv.index("--parallel") + 1] == "4"
    assert argv[argv.index("--api-key") + 1] == KEY
    for flag in ("--no-webui", "--no-slots", "--log-disable"):
        assert flag in argv
    assert "--metrics" not in argv and "--slot-save-path" not in argv
    assert argv[-2:] == ["--temp", "0.7"]


# A stand-in llama-server that is healthy for a moment and then exits with 7.
SHORT_LIVED = textwrap.dedent("""
    import sys, time
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Health(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
        def log_message(self, *args):
            pass

    port = int(sys.argv[sys.argv.index("--port") + 1])
    srv = HTTPServer(("127.0.0.1", port), Health)
    srv.timeout = 0.1
    end = time.time() + 1.5
    while time.time() < end:
        srv.handle_request()
    sys.exit(7)
""")


def test_session_wait_returns_the_servers_exit_code(tmp_path):
    script = tmp_path / "short-lived-server.py"
    script.write_text(SHORT_LIVED)
    cfg = server.ServeConfig(model=tmp_path / "m.gguf", n_ctx=0, port=server.free_port())
    cfg.argv = lambda: [sys.executable, str(script), "--port", str(cfg.port)]
    with server.Session(cfg) as session:
        assert session.wait(poll=0.05) == 7
