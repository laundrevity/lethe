"""The model server lives exactly as long as the lethe run that started it."""

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from lethe import server

# A stand-in for llama-server: answers /health on the port it is given and otherwise sleeps.
FAKE = textwrap.dedent("""
    import sys
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Health(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
        def log_message(self, *args):
            pass

    port = int(sys.argv[sys.argv.index("--port") + 1])
    HTTPServer(("127.0.0.1", port), Health).serve_forever()
""")


def fake_config(tmp_path, exit_code=None):
    script = tmp_path / "fake-llama-server.py"
    script.write_text(f"import sys; sys.exit({exit_code})" if exit_code is not None else FAKE)
    # The real argv is built from the vendored binary; the test substitutes its own server.
    # (A fixed port, so Session keeps this config object and its substitute argv.)
    cfg = server.ServeConfig(model=tmp_path / "model.gguf", n_ctx=0, port=server.free_port())
    cfg.argv = lambda: [sys.executable, str(script), "--port", str(cfg.port)]
    return cfg


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def test_a_free_port_is_picked_when_none_is_given():
    cfg = server.ServeConfig(model=Path("model.gguf"), n_ctx=0)
    session = server.Session(cfg)
    assert session.cfg.port > 0 and session.url == f"http://127.0.0.1:{session.cfg.port}"
    assert cfg.port == 0  # the caller's config is left alone


def test_session_starts_on_a_free_port_and_is_killed_on_exit(tmp_path):
    cfg = fake_config(tmp_path)
    waits = []
    with server.Session(cfg, on_wait=waits.append) as session:
        assert session.url == f"http://127.0.0.1:{cfg.port}"
        assert server.is_healthy(session.url)
        pid = session.proc.pid
        assert alive(pid)
    assert not alive(pid)
    assert not server.is_healthy(session.url)
    assert session.proc is None


def test_session_is_killed_by_an_exception_and_by_a_failed_start(tmp_path):
    cfg = fake_config(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        with server.Session(cfg) as session:
            pid = session.proc.pid
            raise KeyboardInterrupt
    assert not alive(pid)

    with pytest.raises(RuntimeError, match="exited with 3"):
        with server.Session(fake_config(tmp_path, exit_code=3)):
            pytest.fail("should not have come up")


def test_a_terminated_lethe_takes_its_server_with_it(tmp_path):
    cfg = fake_config(tmp_path)
    driver = tmp_path / "driver.py"
    driver.write_text(textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {str(Path(server.__file__).resolve().parents[1])!r})
        from lethe import server
        cfg = server.ServeConfig(model={str(tmp_path / "model.gguf")!r}, n_ctx=0, port=server.free_port())
        cfg.argv = lambda: [sys.executable, {str(tmp_path / "fake-llama-server.py")!r}, "--port", str(cfg.port)]
        with server.Session(cfg) as session:
            print(session.proc.pid, flush=True)
            time.sleep(60)
    """))
    driver_proc = subprocess.Popen([sys.executable, str(driver)], stdout=subprocess.PIPE, text=True)
    child_pid = int(driver_proc.stdout.readline())
    assert alive(child_pid)
    driver_proc.send_signal(signal.SIGTERM)
    assert driver_proc.wait(timeout=20) == 128 + signal.SIGTERM
    for _ in range(40):
        if not alive(child_pid):
            break
        time.sleep(0.25)
    assert not alive(child_pid)
