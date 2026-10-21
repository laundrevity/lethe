"""Opt-in image/video smoke test; the server is unloaded at the end.

    uv run pytest -q -s -p no:cacheprovider tests/test_live_media.py --run-live-media

Only fixed synthetic fixtures are written under pytest's temporary directory.
Decoded frames, requests, and model replies stay in memory; no user media is read.
"""

import struct
import zlib

import pytest

from lethe import cli, repl, server, spec, video
from lethe.chat import Delta, TurnStats, stream_chat
from lethe.profiles import profile_for


def _two_squares():
    """A 384x256 PNG: red square on the left, blue square on the right."""
    rows = []
    for y in range(256):
        row = bytearray(b"\0")
        for x in range(384):
            color = (255, 255, 255)
            if 64 <= y < 192:
                if 32 <= x < 160:
                    color = (255, 0, 0)
                elif 224 <= x < 352:
                    color = (0, 0, 255)
            row.extend(color)
        rows.append(row)

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 384, 256, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b"")


def test_live_image_and_video(request, tmp_path):
    if not request.config.getoption("--run-live-media"):
        pytest.skip("pass --run-live-media to load the model")
    if request.config.getoption("capture") != "no":
        pytest.fail("use -s for live tests so replies are not captured in temporary files")

    photo = tmp_path / "image.png"
    clip = tmp_path / "clip.mp4"
    photo.write_bytes(_two_squares())
    # A six-second test source; FFmpeg output is piped into Python, then stored
    # as an explicit synthetic fixture. The real attachment decoder writes nothing.
    clip.write_bytes(video._run("ffmpeg", [
        "-nostdin", "-f", "lavfi", "-i", "color=c=red:s=320x240:r=1:d=2",
        "-f", "lavfi", "-i", "color=c=lime:s=320x240:r=1:d=2",
        "-f", "lavfi", "-i", "color=c=blue:s=320x240:r=1:d=2",
        "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]", "-map", "[v]",
        "-c:v", "mpeg4", "-an", "-movflags", "frag_keyframe+empty_moov", "-f", "mp4", "pipe:1",
    ], clip))
    image_content = repl._user_content("What two colors are the squares? Reply only the colors in left-to-right order.", [], [photo])
    video_content = repl._user_content("In chronological order, what three colors fill the screen in this video? Reply only the three colors.", [], [], [clip])
    assert sum(p["type"] == "image_url" for p in video_content) == 6

    model = spec.resolve(request.config.getoption("--media-model"))
    profile = profile_for(cli._shape(model.path))
    with cli._start(model, ctx=16384, kv="f16", vision=True, thinking=False, effort=None) as srv:
        proc = srv.proc
        assert server.props(srv.url, srv.api_key)["modalities"]["vision"]
        for label, content, colors in (
            ("Image", image_content, ["red", "blue"]),
            ("Video", video_content, ["red", "green", "blue"]),
        ):
            answer, stats = "", None
            for event in stream_chat(
                srv.url, [{"role": "user", "content": content}], n_ctx=16384,
                sampling=profile.sampling(False), max_tokens=128,
                chat_template_kwargs=profile.disable_thinking_kwargs, api_key=srv.api_key,
            ):
                if isinstance(event, Delta):
                    answer += event.content
                elif isinstance(event, TurnStats):
                    stats = event
            print(f"{label}: {answer.strip()}", flush=True)
            normalized = answer.lower().replace("lime", "green")
            positions = [normalized.find(color) for color in colors]
            assert all(pos >= 0 for pos in positions) and positions == sorted(positions), f"{label}: wrong colors/order"
            assert stats and stats.finish_reason == "stop"
    assert proc.poll() is not None, "test server was not unloaded"
    print("Image and video smoke tests passed; the model was unloaded.")
