import io
import json

import pytest
from rich.console import Console

from lethe import repl, tools
from lethe.files import MAX_READ_CHARS, FileTools, apply_edit
from lethe.profiles import Profile
from lethe.shell import ShellMode, ShellSandbox


def make_config(**kwargs):
    return repl.ReplConfig("http://localhost", None, 4096, "test-model", Profile("test", {}, {}), **kwargs)


def test_read_file_numbers_lines_and_pages(tmp_path):
    ft = FileTools(tmp_path.resolve())
    (tmp_path / "a.txt").write_text("one\ntwo\nthree\n")
    assert ft.read_file("a.txt") == "1\tone\n2\ttwo\n3\tthree"
    assert ft.read_file("a.txt", offset=2, limit=1) == "2\ttwo\n[lines 2-2 of 3; call read_file with offset=3 to continue]"
    assert "past the end" in ft.read_file("a.txt", offset=9)
    (tmp_path / "big.txt").write_text("".join(f"line {i}\n" for i in range(5000)))
    first = ft.read_file("big.txt")
    assert len(first) <= MAX_READ_CHARS + 100
    nxt = int(first.rsplit("offset=", 1)[1].split()[0])
    assert ft.read_file("big.txt", offset=nxt).startswith(f"{nxt}\tline {nxt - 1}\n")
    (tmp_path / "min.js").write_text("x" * 50_000)
    assert "line truncated" in ft.read_file("min.js")
    (tmp_path / "bin").write_bytes(b"\xff\xfe\x00")
    with pytest.raises(ValueError, match="not UTF-8"):
        ft.read_file("bin")


def test_confined_read_cannot_leave_the_workspace(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (tmp_path / "secret").write_text("s")
    (root / "link").symlink_to(tmp_path / "secret")
    ft = FileTools(root.resolve())
    for path in ("../secret", str(tmp_path / "secret"), "link", "~/.ssh/id_rsa"):
        with pytest.raises(ValueError, match="within the shell workspace"):
            ft.read_file(path)
    assert FileTools(root.resolve(), confined=False).read_file("../secret") == "1\ts"


def test_apply_edit_requires_one_exact_match_and_explains_failures():
    text = "def a():\n    return 1\n\ndef b():\n    return 1\n"
    assert apply_edit(text, "def a():\n    return 1", "def a():\n    return 2") == (text.replace("return 1", "return 2", 1), 1)
    assert apply_edit(text, "return 1", "return 3", True)[1] == 2
    with pytest.raises(ValueError, match=r"matches 2 places \(starting at lines 2, 5\)"):
        apply_edit(text, "return 1", "return 3")
    with pytest.raises(ValueError, match=r"lines 1-2 match apart from\s+whitespace(.|\n)*    return 1"):
        apply_edit(text, "def a():\n  return 1", "x")
    with pytest.raises(ValueError, match="line-number prefixes"):
        apply_edit(text, "1\tdef a():\n2\t    return 1", "x")
    with pytest.raises(ValueError, match="not found in the file"):
        apply_edit(text, "nope", "x")
    with pytest.raises(ValueError, match="must not be empty"):
        apply_edit(text, "", "x")
    with pytest.raises(ValueError, match="identical"):
        apply_edit(text, "def", "def")
    assert apply_edit("a\r\nb\r\nc\r\n", "a\nb", "a\nB")[0] == "a\r\nB\r\nc\r\n"


def test_write_and_edit_round_trip_awkward_content(tmp_path):
    ft = FileTools(tmp_path.resolve(), confined=False)
    content = "#!/bin/sh\ncat <<'EOF'\n$HOME `x` \"q\" 'q' \\n é\nEOF\n"
    assert ft.write_file("deep/dir/s.sh", content) == "created deep/dir/s.sh (4 lines)"
    assert (tmp_path / "deep/dir/s.sh").read_bytes() == content.encode()
    with pytest.raises(ValueError, match="already has exactly this content"):
        ft.write_file("deep/dir/s.sh", content)
    assert ft.edit_file("deep/dir/s.sh", "$HOME `x`", "$PWD") == "edited deep/dir/s.sh: 1 replacement"
    assert "$PWD \"q\"" in (tmp_path / "deep/dir/s.sh").read_text()
    assert ft.write_file("deep/dir/s.sh", "new\n").startswith("overwrote")
    with pytest.raises(ValueError, match="no such file"):
        ft.edit_file("missing.py", "a", "b")
    with pytest.raises(ValueError, match="directory"):
        ft.write_file("deep", "x")


def test_read_only_registry_has_no_write_tools_and_mode_switch_drops_them(tmp_path, monkeypatch):
    monkeypatch.setattr(ShellSandbox, "check_available", lambda self: None)
    cfg = make_config(shell_dir=tmp_path, tools=tools.build_registry(["shell"], shell_dir=tmp_path, shell_mode=ShellMode.FULL))
    assert set(cfg.tools) == tools.SHELL_TOOL_NAMES
    assert all(t.requires_confirmation for t in cfg.tools.values())
    messages = []
    repl._tool_command("shell", "read-only", messages, cfg)
    assert set(cfg.tools) == {"run_shell", "read_file"}
    repl._tool_command("shell", "off", messages, cfg)
    assert cfg.tools == {}


def test_edit_is_approved_as_a_diff_and_bad_edits_never_prompt(tmp_path, monkeypatch):
    cfg = make_config(tools=tools.build_registry(["shell"], shell_dir=tmp_path, shell_mode=ShellMode.FULL))
    target = tmp_path / "m.py"
    target.write_text("a = 1\nb = 2\n")
    output = io.StringIO()
    monkeypatch.setattr(repl, "con", Console(file=output, color_system=None, width=200))
    monkeypatch.setattr(repl, "ask_approval", lambda *a, **kw: pytest.fail("prompted for an impossible edit"))
    bad = json.dumps({"path": "m.py", "old_string": "c = 3", "new_string": "c = 4"})
    assert repl._invoke_tool(cfg, "edit_file", bad).startswith("error: old_string not found")

    answers = iter([False, True, True])
    monkeypatch.setattr(repl, "ask_approval", lambda *a, **kw: next(answers))
    good = json.dumps({"path": "m.py", "old_string": "b = 2", "new_string": "b = \x1b[2J3"})
    assert "declined" in repl._invoke_tool(cfg, "edit_file", good)
    assert target.read_text() == "a = 1\nb = 2\n"
    shown = output.getvalue()
    assert "-b = 2" in shown and "+b = \\x1b[2J3" in shown and "\x1b" not in shown
    assert str(target.resolve()) in shown and "old_string" not in shown
    assert repl._invoke_tool(cfg, "edit_file", good) == "edited m.py: 1 replacement"
    assert target.read_text() == "a = 1\nb = \x1b[2J3\n"
    new = json.dumps({"path": "n/new.txt", "content": "hi\n"})
    assert repl._invoke_tool(cfg, "write_file", new) == "created n/new.txt (1 lines)"
    assert "+hi" in output.getvalue()
    assert "replace_all must be a boolean" in repl._invoke_tool(
        cfg, "edit_file", json.dumps({"path": "m.py", "old_string": "a", "new_string": "b", "replace_all": "true"}))


def test_auto_approve_scopes(tmp_path, monkeypatch):
    registry = tools.build_registry(["shell"], shell_dir=tmp_path, shell_mode=ShellMode.FULL)
    monkeypatch.setattr(repl, "ask_approval", lambda *a, **kw: False)
    write = json.dumps({"path": "a.txt", "content": "x\n"})
    shell = json.dumps({"command": "printf ran"})
    cfg = make_config(tools=registry)
    assert "declined" in repl._invoke_tool(cfg, "write_file", write)
    cfg = make_config(tools=registry, auto_approve="files")
    assert repl._invoke_tool(cfg, "write_file", write).startswith("created")
    assert "declined" in repl._invoke_tool(cfg, "run_shell", shell)
    cfg = make_config(tools=registry, auto_approve="all")
    assert json.loads(repl._invoke_tool(cfg, "run_shell", shell))["stdout"] == "ran"
