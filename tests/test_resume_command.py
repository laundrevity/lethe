"""The printed resume command carries launch options without persisting them or replaying inputs."""

import io
import shlex
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from rich.console import Console
from typer.testing import CliRunner

from lethe import cli, instructions, repl, vault
from lethe.profiles import Profile
from lethe.shell import ShellSandbox


@pytest.fixture
def launch(monkeypatch, tmp_path):
    configs, starts = [], []
    conv = vault.Conversation.new("owner/repo:Q8")

    def start(*args, **kwargs):
        starts.append(kwargs)
        return nullcontext(SimpleNamespace(url="http://localhost", api_key="private-key"))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_start", start)
    monkeypatch.setattr(cli.server, "props", lambda *a: {
        "default_generation_settings": {"n_ctx": 4096}, "modalities": {"vision": starts[-1]["vision"]},
    })
    monkeypatch.setattr(cli, "_resolve_or_offer", lambda value, yes=False: SimpleNamespace(path=tmp_path / "model.gguf", spec=value))
    monkeypatch.setattr(cli.config, "default_model", lambda: conv.model)
    monkeypatch.setattr(cli, "_shape", lambda p: SimpleNamespace(arch="generic", name="test"))
    monkeypatch.setattr(cli, "_default_system", lambda *a: "default system")
    monkeypatch.setattr(instructions, "load", lambda *a: ("", []))
    monkeypatch.setattr(ShellSandbox, "check_available", lambda self: None)
    monkeypatch.setattr(repl, "ask_passphrase", lambda confirm: "private-passphrase")
    monkeypatch.setattr(vault, "load", lambda *a: conv)
    monkeypatch.setattr(repl, "run", configs.append)

    def invoke(args):
        result = CliRunner().invoke(cli.app, cli._normalize_argv(args))
        assert result.exit_code == 0, result.exception or result.output
        return configs[-1], starts[-1]

    return invoke


@pytest.mark.parametrize("flags", [
    ["--web", "--tools", "--shell-mode", "read-only", "--tools-confirm", "-c", "8192", "--kv", "q8_0",
     "--no-thinking", "--no-show-thinking", "--vision", "--video-frames", "8", "--max-tokens", "512",
     "--temp", "0", "--top-p", "0.8", "--top-k", "20", "--min-p", "0.1", "--effort", "low", "--debug-server"],
    ["--tools", "fetch", "--no-vision", "--no-instructions", "--auto-approve", "files"],
    ["--no-tools", "--no-system"],
    ["--tools", "", "--vision", "--shell-mode", "read-only"],
    ["--shell", "--shell-mode", "full", "--auto-approve", "all", "--thinking", "--show-thinking"],
])
def test_resume_command_round_trips_launch_settings(launch, flags):
    cfg, started = launch(flags)
    argv = shlex.split(cfg.resume_command("abc123"))
    assert argv[:3] == ["lethe", "--resume", "abc123"]
    resumed, restarted = launch(argv[1:])
    assert restarted == started
    for name in ("thinking", "show_thinking", "vision", "video_frames", "max_tokens", "sampling_overrides",
                 "tools_confirm", "auto_approve", "searx", "shell_dir"):
        assert getattr(resumed, name) == getattr(cfg, name), name
    assert set(resumed.tools) == set(cfg.tools)
    assert resumed.resume_args == cfg.resume_args
    assert not resumed.save


def test_resume_command_quotes_paths_urls_and_explicit_model_and_omits_one_shot_inputs(launch, tmp_path):
    workspace = tmp_path / "[red] user's $(nope) `project`"
    workspace.mkdir()
    attachment = tmp_path / "private-input.txt"
    attachment.write_text("synthetic attachment")
    searx = "https://search.example/?key='value'&q=$(nope)"
    model = "owner/other-repo:Q4"
    cfg, started = launch([model, "--web", "--shell-dir", workspace.name, "--searx", searx,
                           "--system", "private prompt", "-f", str(attachment), "--save", "--yes"])
    command = cfg.resume_command("abc123")
    argv = shlex.split(command)
    assert argv == ["lethe", "--resume", "abc123", model, "--web", "--shell-dir", str(workspace), "--searx", searx]
    assert all(secret not in command for secret in ("private prompt", "private-input", "private-passphrase", "private-key"))
    resumed, restarted = launch(argv[1:])
    assert restarted == started and resumed.model_spec == model
    assert resumed.shell_dir == workspace and resumed.searx == searx
    assert not resumed.attachments and not resumed.save


def test_plain_launch_does_not_add_flags_or_implicitly_opt_in_to_saving(launch):
    cfg, _ = launch([])
    assert cfg.resume_command("abc123") == "lethe --resume abc123"


@pytest.mark.parametrize("save_at_start", [False, True])
def test_save_and_exit_print_the_quoted_resume_hint_without_rich_markup(monkeypatch, tmp_path, save_at_start):
    out = io.StringIO()
    monkeypatch.setattr(repl, "con", Console(file=out, width=40, color_system=None))
    inputs = iter(["/quit"] if save_at_start else ["/save", "/quit"])
    monkeypatch.setattr(repl, "_make_reader", lambda: lambda prompt: next(inputs))
    monkeypatch.setattr(vault, "save", lambda *a: None)
    cfg = repl.ReplConfig("http://localhost", "private-key", 4096, "test-model", Profile("test", {}, {}),
                         save=save_at_start, passphrase="private-passphrase",
                         resume_args=["--web", "--shell-dir", str(tmp_path / "my [red] project")])
    repl.run(cfg)
    hints = [line.split("resume with: ")[1] for line in out.getvalue().splitlines() if "resume with: " in line]
    assert len(hints) == (1 if save_at_start else 2)
    for hint in hints:
        argv = shlex.split(hint)
        assert argv[:2] == ["lethe", "--resume"] and argv[3:] == cfg.resume_args
    assert "private-passphrase" not in out.getvalue() and "private-key" not in out.getvalue()
