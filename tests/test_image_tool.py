"""Image tool approval, local paths, and multimodal tool replies."""

import base64
import copy
import io
import json
from pathlib import Path

import pytest
from rich.console import Console

from lethe import attachments, repl, tools, vault, video
from lethe.files import FileTools
from lethe.instructions import MARKER
from lethe.profiles import Profile
from lethe.shell import ShellMode, ShellSandbox


def config(root, **kwargs):
    return repl.ReplConfig(
        "http://localhost", None, 4096, "test", Profile("test", {}, {}),
        vision=True, shell_dir=root,
        tools={"view_image": tools.build_image_tool(root, confined=True)}, **kwargs,
    )


@pytest.mark.parametrize("suffix", [".PNG", ".jpg", ".gif", ".bmp", ".webp"])
def test_image_result_reuses_attachment_loader_without_text_truncation(tmp_path, monkeypatch, suffix):
    photo = tmp_path / ("a photo" + suffix)
    data = b"synthetic image bytes" * 2000  # exceeds the text tool result limit
    photo.write_bytes(data)
    monkeypatch.setattr(video, "image_png", lambda p: data if p == photo else pytest.fail("wrong path"))
    parts = FileTools(tmp_path).view_image(photo.name)
    assert parts[1] == repl._user_content("describe", [], [photo])[1]
    assert base64.b64decode(parts[1]["image_url"]["url"].split(",", 1)[1]) == data
    assert parts[0]["text"] == f"Image: {photo.name}"
    assert list(tmp_path.iterdir()) == [photo]


def test_image_paths_use_workspace_and_reject_symlink_escapes(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    photo = tmp_path / "photo.png"
    photo.write_bytes(b"synthetic PNG")
    (root / "outside.png").symlink_to(photo)
    monkeypatch.chdir(tmp_path)  # relative paths must use the tool root, not cwd
    home_path = "~/" + str(photo.relative_to(Path.home(), walk_up=True))
    confined = tools.build_image_tool(root, confined=True)
    for path in ("../photo.png", str(photo), "outside.png", home_path):
        with pytest.raises(ValueError, match="within the shell workspace"):
            confined.fn(path)
    full = tools.build_image_tool(root, confined=False)
    for path in ("../photo.png", str(photo), "outside.png", home_path):
        assert full.fn(path)[1]["image_url"]["url"] == attachments.image_data_url(photo)
    (root / "notes.txt").write_text("notes")
    for path in ("missing.png", ".", "notes.txt", "https://example.com/photo.png"):
        with pytest.raises(ValueError):
            full.fn(path)


@pytest.mark.parametrize("auto_approve", [None, "files", "all"])
def test_image_is_not_read_before_approval_and_decline_or_note_never_reads(tmp_path, monkeypatch, auto_approve):
    photo = tmp_path / "photo.png"
    photo.touch()
    reads = []
    monkeypatch.setattr(attachments, "image_data_url", lambda p: reads.append(p) or "data:image/png;base64,fixture")
    decisions = iter([False, "use another image", True])
    monkeypatch.setattr(repl, "ask_approval", lambda: next(decisions))
    cfg = config(tmp_path, auto_approve=auto_approve)
    args = json.dumps({"path": photo.name})
    if auto_approve is None:
        assert "declined" in repl._invoke_tool(cfg, "view_image", args)
        assert "use another image" in repl._invoke_tool(cfg, "view_image", args)
        assert reads == []
    else:
        monkeypatch.setattr(repl, "ask_approval", lambda: pytest.fail("auto-approved image prompted"))
    assert repl._invoke_tool(cfg, "view_image", args)[1]["type"] == "image_url"
    assert reads == [photo]


def test_image_tool_tracks_mode_directory_and_disable_commands(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    photo = tmp_path / "photo.png"
    photo.write_bytes(b"synthetic PNG")
    monkeypatch.setattr(ShellSandbox, "check_available", lambda self: None)
    cfg = config(tmp_path)
    base = "This conversation happens in a vacuum: inference is on-device, and nothing is " + tools.VACUUM_CLAIM + tools.PRIVACY_NOTE
    instructions = MARKER + "Use view_image only when asked."
    messages = [{"role": "system", "content": base + instructions}]
    repl._tool_command("shell", f'read-only "{root}"', messages, cfg)
    with pytest.raises(ValueError, match="within the shell workspace"):
        cfg.tools["view_image"].fn("../photo.png")
    repl._tool_command("shell", "off", messages, cfg)
    assert set(cfg.tools) == {"view_image"}
    assert tools.IMAGE_TOOL_NOTE in messages[0]["content"]
    assert messages[0]["content"].endswith(instructions)
    repl._tool_command("tools", "off", messages, cfg)
    assert cfg.tools == {} and tools.IMAGE_TOOL_NOTE not in messages[0]["content"]
    repl._tool_command("tools", "full", messages, cfg)
    assert cfg.tools["view_image"].fn("../photo.png")[1]["type"] == "image_url"
    assert cfg.shell_dir == root and messages[0]["content"].endswith(instructions)
    assert tools.sync_system_text(messages[0]["content"], cfg.tools) == messages[0]["content"]


@pytest.mark.parametrize("interrupt", [False, True])
def test_image_and_text_results_reach_next_turn_and_replay_without_rereading(tmp_path, monkeypatch, interrupt):
    photo = tmp_path / "photo.png"
    photo.write_bytes(b"synthetic PNG")
    cfg = config(tmp_path, auto_approve="files", save=True, passphrase="pw")
    cfg.tools["read_file"] = tools.build_registry(["shell"], shell_dir=tmp_path, shell_mode=ShellMode.FULL)["read_file"]
    (tmp_path / "note.txt").write_text("a note")
    calls = [
        {"id": "image", "type": "function", "function": {"name": "view_image", "arguments": '{"path":"photo.png"}'}},
        {"id": "text", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"note.txt"}'}},
    ]
    output = io.StringIO()
    inputs = iter(["describe", "followup", "/quit"])
    seen, saved = [], []
    monkeypatch.setattr(repl, "con", Console(file=output, color_system=None, width=200))
    monkeypatch.setattr(repl, "_make_reader", lambda: lambda prompt: next(inputs))
    monkeypatch.setattr(vault, "save", lambda conv, pw: saved.append(conv.to_bytes()))

    def stream(url, messages, **kwargs):
        seen.append(copy.deepcopy(messages))
        if len(seen) == 1:
            yield repl.Delta(tool_calls=calls)
        elif len(seen) == 2 and interrupt:
            raise KeyboardInterrupt
        else:
            yield repl.Delta(content="answer")
        yield repl.TurnStats()

    monkeypatch.setattr(repl, "stream_chat", stream)
    repl.run(cfg)
    assert len(seen) == 3
    image_reply, text_reply = seen[1][-2:]
    assert image_reply == {"role": "tool", "tool_call_id": "image", "name": "view_image", "content": FileTools(tmp_path).view_image("photo.png")}
    assert text_reply["tool_call_id"] == "text" and text_reply["content"] == "1\ta note"
    assert image_reply in seen[2] and text_reply in seen[2]
    assert seen[2][-1]["content"].startswith(repl.INTERRUPT_NOTE) is interrupt
    # Saved image bytes round-trip as JSON; replay must not open the source again.
    conv = vault.Conversation.from_bytes(saved[-1])
    assert image_reply in conv.messages
    monkeypatch.setattr(attachments, "image_data_url", lambda p: pytest.fail("replay read the image"))
    repl._replay_history(conv.messages, show_thinking=True)
    shown = output.getvalue()
    assert "1 image(s)" in shown and "data:image" not in shown
    assert base64.b64encode(b"synthetic PNG").decode() not in shown
    assert repl._approx_messages([image_reply]) < 30
