"""Media routing, bounded sampling, and the no-temporary-media decoder contract."""

import base64
import io
import json
import subprocess
from types import SimpleNamespace

import pytest
from rich.console import Console
from typer.testing import CliRunner

from lethe import attachments, cli, repl, spec, video
from lethe.profiles import Profile


def test_projector_above_shards_prefers_f16_and_stays_in_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("LETHE_ROOT", str(tmp_path))
    root = tmp_path / "models" / "unsloth--Model"
    quant = root / "UD-IQ4_XS"
    quant.mkdir(parents=True)
    model = quant / "model-IQ4_XS.gguf"
    model.touch()
    for name in ("mmproj-BF16.gguf", "mmproj-F16.gguf"):
        (root / name).touch()
    lm = spec.resolve("unsloth/Model")
    assert cli._find_sidecar(lm, "mmproj") == root / "mmproj-F16.gguf"
    (quant / "mmproj-local.gguf").touch()
    assert cli._find_sidecar(lm, "mmproj") == quant / "mmproj-local.gguf"
    external = spec.LocalModel(model, [model], 0, None, "IQ4_XS")
    (quant / "mmproj-local.gguf").unlink()
    assert cli._find_sidecar(external, "mmproj") is None
    (root / "mmproj-BF16.gguf").unlink()
    (root / "mmproj-F16.gguf").unlink()
    (root.parent / "mmproj-unrelated.gguf").touch()
    assert cli._find_sidecar(lm, "mmproj") is None


def test_missing_projector_fails_before_loading_model(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_shape", lambda p: object())
    monkeypatch.setattr(cli, "_plan_ctx", lambda *a: pytest.fail("planning before checking projector"))
    lm = spec.LocalModel(tmp_path / "model.gguf", [], 1, None, None)
    with pytest.raises(cli.typer.BadParameter, match="no mmproj"):
        with cli._start(lm, ctx=4096, kv="f16", vision=True, thinking=False, effort=None):
            pytest.fail("server started")


@pytest.mark.parametrize("suffix", [".PNG", ".mp4", ".mov"])
def test_cli_media_requires_vision_before_model_load(tmp_path, monkeypatch, suffix):
    path = tmp_path / ("input" + suffix)
    path.touch()
    monkeypatch.setattr(cli, "_start", lambda *a, **kw: pytest.fail("model loaded"))
    result = CliRunner().invoke(cli.app, ["chat", "-f", str(path)])
    assert result.exit_code == 1 and "require --vision" in result.output


@pytest.mark.parametrize("duration,cap,times", [(120, 4, [0, 30, 60, 90]), (2.5, 16, [0, 1, 2]), (0.2, 16, [0])])
def test_video_frames_cover_clip_with_bounded_sampling(monkeypatch, tmp_path, duration, cap, times):
    monkeypatch.setattr(video, "_run", lambda *a: json.dumps({"streams": [{}], "format": {"duration": str(duration)}}).encode())
    seen = []
    monkeypatch.setattr(video, "image_png", lambda path, timestamp: seen.append(timestamp) or b"PNG fixture")
    parts = video.video_parts(tmp_path / "clip.mp4", cap)
    assert seen == times
    assert "No audio" in parts[0]["text"] and "may be missed" in parts[0]["text"]
    images = [p for p in parts if p["type"] == "image_url"]
    assert len(images) == len(times)
    assert all(base64.b64decode(p["image_url"]["url"].split(",")[1]) == b"PNG fixture" for p in images)
    assert all(p["type"] != "input_video" for p in parts)


@pytest.mark.parametrize("metadata", [{"streams": []}, {"streams": [{"duration": "NaN"}]}, {"streams": [{"duration": "-1"}]}, {"streams": [{"duration": "inf"}]}])
def test_unknown_video_duration_fails_without_decoding(monkeypatch, tmp_path, metadata):
    monkeypatch.setattr(video, "_run", lambda *a: json.dumps(metadata).encode())
    monkeypatch.setattr(video, "image_png", lambda *a, **kw: pytest.fail("unbounded decode"))
    with pytest.raises(attachments.UnsupportedFile, match="duration"):
        video.video_parts(tmp_path / "clip.mp4")


def test_decoder_reads_local_file_outputs_only_to_pipe_and_drops_report_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FFREPORT", "file=secret.log")
    monkeypatch.setattr(video.shutil, "which", lambda name: "/bin/" + name)
    calls = []
    png = b"\x89PNG\r\n\x1a\nfixture"

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(stdout=png)

    monkeypatch.setattr(video.subprocess, "run", run)
    path = tmp_path / "clip with spaces.mp4"
    assert video.image_png(path, timestamp=3) == png
    argv, kwargs = calls[0]
    assert argv[argv.index("-i") + 1] == str(path.resolve())
    assert argv[argv.index("-protocol_whitelist") + 1] == "file,pipe"
    assert argv[-1] == "pipe:1" and not any("cache:" in a for a in argv)
    assert kwargs["stdout"] == subprocess.PIPE and kwargs["stdin"] == subprocess.DEVNULL
    assert "FFREPORT" not in kwargs["env"] and kwargs["timeout"] == 30
    assert not kwargs.get("shell")
    assert not list(tmp_path.iterdir())


def test_webp_is_converted_before_sending_to_llama(monkeypatch, tmp_path):
    monkeypatch.setattr(video, "image_png", lambda path: b"PNG fixture")
    url = attachments.image_data_url(tmp_path / "image.webp")
    assert url == "data:image/png;base64," + base64.b64encode(b"PNG fixture").decode()


def test_missing_ffmpeg_and_timeout_are_useful_attachment_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(video.shutil, "which", lambda name: None)
    with pytest.raises(attachments.UnsupportedFile, match="brew install ffmpeg"):
        video.image_png(tmp_path / "clip.mp4")
    monkeypatch.setattr(video.shutil, "which", lambda name: "/bin/" + name)

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("ffmpeg", 30)

    monkeypatch.setattr(video.subprocess, "run", timeout)
    with pytest.raises(attachments.UnsupportedFile, match="timed out"):
        video.image_png(tmp_path / "clip.mp4")


def drive(monkeypatch, lines, **kwargs):
    output = io.StringIO()
    reader = iter([*lines, "/quit"])
    seen = []
    monkeypatch.setattr(repl, "con", Console(file=output, width=200))
    monkeypatch.setattr(repl, "_make_reader", lambda: lambda prompt: next(reader))

    def stream(url, messages, **kwargs):
        seen.append([dict(m) for m in messages])
        yield repl.Delta(content="ok")
        yield repl.TurnStats()

    monkeypatch.setattr(repl, "stream_chat", stream)
    repl.run(repl.ReplConfig("http://localhost", None, 4096, "test", Profile("test", {}, {}), **kwargs))
    return seen, output.getvalue()


def test_repl_routes_initial_and_quoted_media_attachments(monkeypatch, tmp_path):
    photo, clip, notes = [tmp_path / name for name in ("a photo.png", "a clip.mp4", "note.txt")]
    photo.write_bytes(b"PNG fixture")
    clip.touch()
    notes.write_text("notes")
    monkeypatch.setattr(video, "video_parts", lambda path, cap: [{"type": "text", "text": f"video cap={cap}"}])
    seen, output = drive(monkeypatch, [f'/video "{clip}"', "describe", "followup"], attachments=[photo, notes], vision=True, video_frames=4)
    content = seen[0][-1]["content"]
    assert "notes" in content[0]["text"] and "describe" in content[0]["text"]
    assert content[1]["type"] == "image_url" and content[2]["text"] == "video cap=4"
    assert seen[1][-1]["content"] == "followup"
    assert "4 sampled frames" in output


@pytest.mark.parametrize("command", ["/reset", "/wipe"])
def test_reset_clears_pending_media(monkeypatch, tmp_path, command):
    clip = tmp_path / "clip.mp4"
    clip.touch()
    monkeypatch.setattr(video, "video_parts", lambda *a: pytest.fail("cleared attachment read"))
    seen, _ = drive(monkeypatch, [command, "hello"], attachments=[clip], vision=True)
    assert seen[0][-1]["content"] == "hello"


def test_repl_video_requires_vision_and_decode_failure_keeps_attachment(monkeypatch, tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.touch()
    seen, output = drive(monkeypatch, [f"/video {clip}", "hello"])
    assert seen[0][-1]["content"] == "hello" and "--vision" in output
    calls = []

    def decode(path, cap):
        calls.append(path)
        if len(calls) == 1:
            raise attachments.UnsupportedFile("decode failed")
        return [{"type": "text", "text": "decoded video"}]

    monkeypatch.setattr(video, "video_parts", decode)
    seen, output = drive(monkeypatch, [f"/file {clip}", "first", "retry"], vision=True)
    assert calls == [clip, clip] and len(seen) == 1
    assert seen[0][-1]["content"][-1]["text"] == "decoded video"
    assert "decode failed" in output
