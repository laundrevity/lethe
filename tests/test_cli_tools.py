from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from lethe import cli, repl, tools
from lethe.shell import SandboxUnavailable, ShellMode


def test_shell_directory_requires_explicit_opt_in(tmp_path):
    result = CliRunner().invoke(cli.app, ["chat", "--shell-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "--shell-dir requires --tools or --shell" in result.output


@pytest.mark.parametrize("flags", [[], ["--web"], ["--tools", "web"]])
def test_full_mode_requires_shell_enabled(monkeypatch, flags):
    monkeypatch.setattr(cli, "_start", lambda *a, **kw: pytest.fail("model started"))
    monkeypatch.setattr(tools, "build_registry", lambda *a, **kw: pytest.fail("tools built"))
    result = CliRunner().invoke(cli.app, ["chat", *flags, "--shell-mode", "full"])
    assert result.exit_code == 1
    assert "--shell-mode full requires --tools or --shell" in result.output


def test_invalid_shell_mode_fails_before_starting(monkeypatch):
    monkeypatch.setattr(tools, "build_registry", lambda *a, **kw: pytest.fail("tools built"))
    result = CliRunner().invoke(cli.app, ["chat", "--shell", "--shell-mode", "invalid"])
    assert result.exit_code == 2 and "Invalid value" in result.output


def test_sandbox_failure_prevents_server_launch(monkeypatch, tmp_path):
    def unavailable(*args, **kwargs):
        raise SandboxUnavailable("sandbox unavailable")
    monkeypatch.setattr(tools, "build_registry", unavailable)
    monkeypatch.setattr(cli, "_start", lambda *a, **kw: pytest.fail("model started"))
    result = CliRunner().invoke(cli.app, ["chat", "--shell", "--shell-dir", str(tmp_path)])
    assert result.exit_code == 1 and "sandbox unavailable" in result.output


@pytest.mark.parametrize("mode", list(ShellMode))
def test_shell_and_web_flags_compose(monkeypatch, tmp_path, mode):
    captured = []
    registry = {"run_shell": object(), "web_search": object(), "fetch_url": object()}
    def build(names, searx=None, **kwargs):
        captured.append((names, kwargs))
        return registry
    monkeypatch.setattr(tools, "build_registry", build)
    FakeSession.stopped = 0
    started = []
    monkeypatch.setattr(cli, "_start", lambda lm, **kw: started.append(kw) or FakeSession())
    monkeypatch.setattr(cli.server, "props", lambda *args: {"default_generation_settings": {"n_ctx": 4096}})
    monkeypatch.setattr(cli, "_resolve_or_offer", lambda value, yes=False: SimpleNamespace(path=tmp_path / "model.gguf", spec="model", size=1))
    monkeypatch.setattr(cli.config, "default_model", lambda: "owner/repo:Q8")
    monkeypatch.setattr(cli, "_shape", lambda path: SimpleNamespace(arch="generic", name="test"))
    monkeypatch.setattr(cli, "_default_system", lambda *args: "custom")
    configs = []
    monkeypatch.setattr(repl, "run", configs.append)
    args = ["--tools", "--web", "--shell-dir", str(tmp_path)]
    if mode == ShellMode.FULL:
        args += ["--shell-mode", "full"]
    result = CliRunner().invoke(cli.app, cli._normalize_argv(args))
    assert result.exit_code == 0, result.exception
    assert captured == [(["shell", "web"], {"shell_dir": tmp_path, "shell_mode": mode})]
    assert configs[0].tools == registry
    assert configs[0].shell_dir == tmp_path.resolve()
    # The model is loaded for this run and unloaded when the chat returns, whatever happens.
    assert started == [{"ctx": 0, "kv": "f16", "vision": False, "thinking": True, "effort": None, "debug": False}]
    assert configs[0].url == "http://127.0.0.1:1234" and configs[0].api_key == "k"
    assert FakeSession.stopped == 1


class FakeSession:
    stopped = 0
    url, api_key = "http://127.0.0.1:1234", "k"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        FakeSession.stopped += 1


def test_the_model_is_unloaded_even_when_the_chat_fails(monkeypatch, tmp_path):
    FakeSession.stopped = 0
    monkeypatch.setattr(cli, "_start", lambda lm, **kw: FakeSession())
    monkeypatch.setattr(cli.server, "props", lambda *args: {"default_generation_settings": {"n_ctx": 4096}})
    monkeypatch.setattr(cli, "_resolve_or_offer", lambda value, yes=False: SimpleNamespace(path=tmp_path / "model.gguf", spec="model", size=1))
    monkeypatch.setattr(cli, "_shape", lambda path: SimpleNamespace(arch="generic", name="test"))
    monkeypatch.setattr(cli, "_default_system", lambda *args: "custom")
    monkeypatch.setattr(repl, "run", lambda cfg: (_ for _ in ()).throw(RuntimeError("chat blew up")))
    result = CliRunner().invoke(cli.app, ["chat", "owner/repo:Q8"])
    assert isinstance(result.exception, RuntimeError)
    assert FakeSession.stopped == 1


def test_there_is_no_background_server_command():
    for gone in ("serve", "stop", "key", "log"):
        result = CliRunner().invoke(cli.app, [gone])
        assert result.exit_code == 2, gone
    assert "serve" not in cli._normalize_argv(["serve"])[:1] or cli._normalize_argv(["serve"]) == ["chat", "serve"]


def test_bare_tools_flag_means_shell_and_composes_with_web():
    n = cli._normalize_argv
    assert n(["--web", "--tools"]) == ["chat", "--web", "--tools", "shell"]
    assert n(["--tool", "--web"]) == ["chat", "--tools", "shell", "--web"]
    assert n(["--tools", "q8_k_p"]) == ["chat", "--tools", "shell", "q8_k_p"]
    assert n(["--tools", "--shell-dir", "src"]) == ["chat", "--tools", "shell", "--shell-dir", "src"]
    assert n(["--web", "--tools", "--shell-mode", "full"]) == ["chat", "--web", "--tools", "shell", "--shell-mode", "full"]
    assert n(["chat", "--tools", "web,fetch"]) == ["chat", "--tools", "web,fetch"]
    assert n(["models"]) == ["models"]


@pytest.mark.parametrize("flags,expect_save", [([], False), (["--save"], True)])
def test_resume_only_reads_unless_saving_is_asked_for(monkeypatch, tmp_path, flags, expect_save):
    from lethe import vault

    conv = vault.Conversation.new("owner/repo:Q8")
    monkeypatch.setattr(repl, "ask_passphrase", lambda confirm: "pw")
    monkeypatch.setattr(cli.vault, "load", lambda conv_id, passphrase: conv if (conv_id, passphrase) == (conv.id, "pw") else pytest.fail("wrong lookup"))
    monkeypatch.setattr(cli, "_start", lambda lm, **kw: FakeSession())
    monkeypatch.setattr(cli.server, "props", lambda *args: {"default_generation_settings": {"n_ctx": 4096}})
    monkeypatch.setattr(cli, "_resolve_or_offer", lambda value, yes=False: SimpleNamespace(path=tmp_path / "model.gguf", spec="owner/repo:Q8", size=1))
    monkeypatch.setattr(cli, "_shape", lambda path: SimpleNamespace(arch="generic", name="test"))
    monkeypatch.setattr(cli, "_default_system", lambda *args: "custom")
    configs = []
    monkeypatch.setattr(repl, "run", configs.append)
    result = CliRunner().invoke(cli.app, ["chat", "--resume", conv.id, "--no-instructions", *flags])
    assert result.exit_code == 0, result.output
    assert configs[0].resume is conv and configs[0].passphrase == "pw"
    assert configs[0].save is expect_save
