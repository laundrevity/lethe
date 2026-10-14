from pathlib import Path

from lethe import instructions, tools


def test_layers_precedence_and_search_bounds(tmp_path, monkeypatch):
    root = tmp_path / "lethe"
    repo = tmp_path / "work" / "repo"
    (repo / "src" / "pkg").mkdir(parents=True)
    (repo / ".git").mkdir()
    root.mkdir()
    monkeypatch.setenv("LETHE_ROOT", str(root))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    assert instructions.load(repo) == ("", [])

    (root / "LETHE.md").write_text("ask before guessing\n")
    (tmp_path / "work" / "LETHE.md").write_text("outside the repository\n")
    block, loaded = instructions.load(repo / "src" / "pkg")
    assert [layer for layer, _, _ in loaded] == ["global"] and "outside the repository" not in block

    (repo / "AGENTS.md").write_text("use tabs\n")
    block, loaded = instructions.load(repo / "src" / "pkg")
    assert [(layer, p.name) for layer, p, _ in loaded] == [("global", "LETHE.md"), ("project", "AGENTS.md")]
    assert block.index("ask before guessing") < block.index("use tabs")
    (repo / "src" / "LETHE.md").write_text("use spaces\n")
    block, loaded = instructions.load(repo / "src" / "pkg")
    assert loaded[1][1] == repo / "src" / "LETHE.md" and "use tabs" not in block

    # launched inside the lethe repo itself: the global file is not loaded twice
    assert [layer for layer, _, _ in instructions.load(root)[1]] == ["global"]


def test_block_survives_tool_sync_and_is_replaced_on_reapply():
    base = "This conversation happens in a vacuum: inference is on-device, and nothing is " + tools.VACUUM_CLAIM + tools.PRIVACY_NOTE
    block = instructions.MARKER + "old rules"
    synced = tools.sync_system_text(base + block, {"web_search": object()})
    assert synced.endswith(block) and tools.SYSTEM_NOTE in synced
    assert tools.sync_system_text(synced, {"web_search": object()}) == synced
    off = tools.sync_system_text(synced, {})
    assert off.endswith(block) and tools.SYSTEM_NOTE not in off
    assert instructions.apply(synced, instructions.MARKER + "new rules").endswith("new rules")
    assert "old rules" not in instructions.apply(synced, "")
