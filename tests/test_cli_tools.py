from types import SimpleNamespace

from typer.testing import CliRunner

from lethe import cli, repl, tools
from lethe.shell import SandboxUnavailable


def test_shell_directory_requires_explicit_opt_in(tmp_path):
    result = CliRunner().invoke(cli.app, ["chat", "--shell-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "--shell-dir requires --tools or --shell" in result.output


def test_sandbox_failure_prevents_server_launch(monkeypatch, tmp_path):
    def unavailable(*args, **kwargs):
        raise SandboxUnavailable("sandbox unavailable")
    monkeypatch.setattr(tools, "build_registry", unavailable)
    monkeypatch.setattr(cli.server, "running_server", lambda *a: (_ for _ in ()).throw(AssertionError("server touched")))
    result = CliRunner().invoke(cli.app, ["chat", "--shell", "--shell-dir", str(tmp_path)])
    assert result.exit_code == 1 and "sandbox unavailable" in result.output


def test_shell_and_web_flags_compose(monkeypatch, tmp_path):
    captured = []
    registry = {"run_shell": object(), "web_search": object(), "fetch_url": object()}
    def build(names, searx=None, **kwargs):
        captured.append((names, kwargs))
        return registry
    monkeypatch.setattr(tools, "build_registry", build)
    monkeypatch.setattr(cli.server, "running_server", lambda port: {"model": "model.gguf", "url": "http://localhost", "api_key": None})
    monkeypatch.setattr(cli.server, "props", lambda *args: {"default_generation_settings": {"n_ctx": 4096}})
    monkeypatch.setattr(cli.S, "resolve", lambda value: SimpleNamespace(path=tmp_path / "model.gguf", spec="model"))
    monkeypatch.setattr(cli, "_shape", lambda path: SimpleNamespace(arch="generic", name="test"))
    monkeypatch.setattr(cli, "_default_system", lambda *args: "custom")
    configs = []
    monkeypatch.setattr(repl, "run", configs.append)
    result = CliRunner().invoke(cli.app, ["chat", "--tools", "shell", "--web", "--shell-dir", str(tmp_path)])
    assert result.exit_code == 0, result.exception
    assert captured == [(["shell", "web"], {"shell_dir": tmp_path})]
    assert configs[0].tools == registry


def test_bare_tools_flag_means_shell_and_composes_with_web():
    n = cli._normalize_argv
    assert n(["--web", "--tools"]) == ["chat", "--web", "--tools", "shell"]
    assert n(["--tool", "--web"]) == ["chat", "--tools", "shell", "--web"]
    assert n(["--tools", "q8_k_p"]) == ["chat", "--tools", "shell", "q8_k_p"]
    assert n(["--tools", "--shell-dir", "src"]) == ["chat", "--tools", "shell", "--shell-dir", "src"]
    assert n(["chat", "--tools", "web,fetch"]) == ["chat", "--tools", "web,fetch"]
    assert n(["models"]) == ["models"]
