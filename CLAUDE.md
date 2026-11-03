# lethe

Local LLM runner for Apple Silicon: a vendored, Metal-built `llama.cpp` driven by a small Python CLI.
Target machine: M5 Max, 128 GB unified memory, 40 GPU cores, macOS 26. See README.md for usage.

## Prime directive: logless and traceless by default

The reason this project exists (vs. Claude/ChatGPT/Codex) is that nobody but the user ever sees the
model's inputs and outputs. Every feature must preserve that:

- Conversation content never touches disk. No server log (`--log-disable`, output → /dev/null;
  `chat --debug-server` shows it on the terminal instead), no `--slot-save-path`, no pid/state files,
  no REPL history. Never add a "save chat" default; if you add persistence, it must be explicit
  opt-in per invocation and be listed by `lethe traces`.
- llama-server always runs with `--no-webui --no-slots`, bound to 127.0.0.1, no `--metrics`, and a
  per-launch random `--api-key` (recovered from the process table by `chat`/`key`; it stops other local
  software and browser pages from using the model — a same-user process could read it from `ps`, that's
  accepted). `lethe serve` uses the user's `LETHE_API_KEY` instead (see below).
- Saved conversations (`vault.py`) are the ONE content-persistence feature: opt-in per invocation
  (`chat --save`, `/save`; `--resume` only reads — nothing is written unless `--save` is given too or `/save`
  is typed, which continue the same id with the same passphrase), AES-256-GCM with a scrypt-derived key, passphrase held in memory
  only, random filenames, all metadata inside the ciphertext, plaintext never written (not even a temp
  file), delete = overwrite then unlink. `lethe.toml` holds configuration only (default model).
- Model tools (`tools.py`, `shell.py`, `files.py`) are opt-in per invocation (`--web`/`--tools`/`--shell`, `/web`). Web tools are
  the one way anything leaves the machine: every request is printed first, nothing is cached, fetches are guarded
  against private addresses and pinned with `curl --resolve`. The shell is read-only, offline Seatbelt
  (`sandbox-exec`), always asks y/n, and has no unsandboxed fallback — never add one. The only unsandboxed shell is the
  explicit opt-in `--shell-mode full` / `/shell full` / `/tools full` (a separate `FullShell` runner that runs as the
  user, with writes, network and inherited environment; still confirms every command; a sandbox failure never
  switches to it). The shell brings file tools with it (`files.py`): `read_file` in both modes (workspace-confined
  when read-only), `write_file`/`edit_file` (exact-string replace) in full mode only — never in read-only. They always
  ask y/n, writes are approved as a diff, and they keep no backups/temp files. The prompt (`repl.ask_approval`) also
  takes `p`: the call is NOT run and the user's note goes back to the model as the tool result — never execute on `p`.
  The only way past the y/n prompts is
  the per-invocation opt-in `--auto-approve files|all` (calls and diffs are still printed; never make it a default or
  a `lethe.toml` setting). The default system prompt's "nothing leaves this machine" claim is patched by
  `tools.sync_system_text` to match what is enabled.
- **Media attachments** (`attachments.py`, `video.py`) are explicit (`--vision` plus `/image`, `/video`, or `-f`).
  Video uses local FFmpeg/ffprobe to sample timestamped PNG frames through stdout pipes only (no audio,
  network protocols, FFREPORT, cache protocol, or extracted files). The server receives images, never
  `input_video`; native video decoding is disabled at build time to prevent its disk-backed cache.
  WebP is converted to PNG the same way. Frames are bounded by `--video-frames` (default 16) and 768px.
- **Terminal presentation lives in `terminal.py`.** `StatusLine` pins one line to the bottom row during a turn
  (DECSTBM scroll region; Apple Terminal keeps the scrollback for a top-anchored region but drops the pending-wrap
  flag across cursor save/restore — hence the column tracking and no redraw at the right margin). `notify()` is the
  bell plus OSC 9/99; its text must stay content-free (Notification Center retains it). The kitty keyboard protocol
  is pushed only while prompt_toolkit reads a line: left on, Ctrl-C during streaming is no longer SIGINT. Apple
  Terminal cannot report Shift+Enter at all; don't promise it there.
- **No background model server, ever.** `lethe` is to a local model what claude code is to Claude: the
  model is loaded when a run starts and unloaded when it ends. `server.Session` starts `llama-server` on a
  free loopback port and kills it on exit (`with` block, atexit, SIGTERM/SIGHUP handlers; only SIGKILL of
  lethe can leave one behind). There is no `stop`/`key` command, no daemon, no pid file and no process-table
  discovery; a second `lethe` is simply a second server. The user runs other llama-servers on this machine
  by hand (e.g. port 8090); never touch them.
- **`lethe serve` is the one fixed-port, fixed-key run** (approved by Conor 2026-11-03 so local programs,
  e.g. the earshot orchestrator, have a stable endpoint). It is a foreground command like `chat`: the same
  `server.Session`, so the model unloads when its terminal ends. Always bound to 127.0.0.1 (no host option),
  on the port the user names (`--port` is required; a port already in use is refused, never taken over),
  with the key read from `LETHE_API_KEY` (at least 16 characters; never generated, printed, stored, or taken
  as an argument), and the same traceless llama-server flags. It never daemonizes, detaches, or reattaches,
  and it logs no requests.
- Hugging Face: `HF_HUB_DISABLE_TELEMETRY=1` (set in `hub.py` before import); downloads are anonymous
  unless the repo is gated (then falls back to `HF_TOKEN`).
- Before adding any dependency or feature, ask: does it write, phone home, or cache content? If yes,
  it's opt-in and documented in README "Logless by default".

## Architecture (decisions, not descriptions)

- **llama.cpp is vendored as a git submodule at `vendor/llama.cpp`** and built by `scripts/build-llama.sh`.
  We do *not* use `llama-cpp-python` — it lags upstream and doesn't expose MTP speculative decoding,
  reasoning flags, or hybrid-attention models. The Python side talks to `llama-server` over HTTP
  (`/v1/chat/completions` streaming; the final chunk's `timings` gives prompt/gen tok/s and
  `draft_n`/`draft_n_accepted`; `/props` gives `n_ctx`).
- **Submodule pin = upstream master `27c7a769d` (2026-10-16) + one local commit** on branch
  `lethe-fastmtp` applying `patches/HauhauCS-FastMTP-llama.cpp.patch` (a `d2t` draft-vocab trim in
  `src/models/qwen35.cpp`). The build script applies/commits the patch automatically. Bumping upstream
  means re-running `git apply --check` on that patch; `qwen35.cpp` is the only file it touches.
- **Build flags are canonical in `scripts/build-llama.sh`.** Metal + Accelerate explicitly ON, Metal
  shader library embedded, `GGML_NATIVE`, Release, curl OFF, `MTMD_VIDEO=OFF` (native video uses disk cache), `CMAKE_BUILD_RPATH=@loader_path` (without it
  the binaries carry an absolute rpath and break when the repo is moved/renamed — exit -6 from dyld).
  Targets: `llama-server llama-cli llama-bench` (there is no `llama-gguf` target). `lethe doctor` verifies
  Metal is linked and lists devices.
- **Models live in `./models/<owner>--<repo>/`**, downloaded via `huggingface_hub` (`hf` CLI is part of
  that package), not the shared `~/.cache/huggingface` — files are 10–60 GB and we want them explicit.
  `models/` is gitignored. `HF_TOKEN` is set in the user's shell.
- **Models are identified as `owner/repo:QUANT`** (`spec.py`); local files are grouped by shard and
  mapped back to that form from `models/<owner>--<repo>/`. Sampler/thinking defaults come from
  `profiles.py`, keyed on GGUF architecture — add a profile there when adding a model family, don't
  hard-code sampler values in the REPL. `lethe.toml` (gitignored) holds the default model.
- **`scripts/bump-llama.sh`** moves the submodule to upstream master: it re-applies the FastMTP patch on a
  fresh `lethe-fastmtp` branch, stages the *upstream* commit as the superproject pin (not the patch
  commit), rebuilds and runs `doctor`. Update the pin note below when you use it.
- **`LETHE.md` is the user-facing instruction file for the *local model*** (`instructions.py`), not for coding agents
  working on this repo (that is this file). Global layer = `LETHE.md` at the repo root; project layer = nearest
  `LETHE.md`/`AGENTS.md` from the launch directory up to the repository root. The block is appended after everything
  else in the system prompt behind `instructions.MARKER`; `tools.sync_system_text` must keep leaving it untouched.
  Read-only: never write or cache these files.
- **Ctrl-C keeps the turn as far as it got** (`repl.run`): the reasoning and text streamed so far become
  an assistant message (`reasoning_content` included), finished tool rounds stay, tool calls that were cut off
  get a reply saying so, and the next prompt is prefixed with `INTERRUPT_NOTE`. Thinking loops are the default
  model's main failure mode; interrupting one must not throw the conversation away. A failed turn (server
  error) is still discarded.
- **Throughput and context are first-class.** Every chat turn prints prompt tok/s, gen tok/s, MTP
  acceptance, and context used/configured. `browse`/`models` show KV bytes/token and the max context
  that fits. KV math lives in `ggufmeta.py` (own ~100-line GGUF header parser; reads only the first
  16 MiB, works over HTTP Range for remote files — never downloads a model to inspect it).
- **Memory budget** = Metal's `recommendedMaxWorkingSetSize` as reported by `llama-server --list-devices`
  (107.5 GiB on this machine — higher than the 75%-of-RAM folk rule) minus 4 GiB headroom.
- Python is managed with **`uv`** (`uv run lethe ...`, `uv add ...`). Python 3.15.1, src layout, typer + rich.

## Apple Silicon notes

- Run with all layers on GPU (`-ngl all`). `sudo sysctl iogpu.wired_limit_mb=<MB>` raises the ceiling.
- Flash attention ON. Default F16 KV cache; `--kv q8_0` only under memory pressure.
- Hybrid models (Qwen3.5/3.8 Gated DeltaNet, `full_attention_interval=4`) only keep KV for 1 in 4
  layers: Qwen3.8-27B costs 64 KiB/token → 16 GiB at its full 262K context. Generation speed barely
  drops with context depth (16.3 → 15.6 tok/s at 32K in llama-bench).
- `llama-bench` does **not** do speculative decoding; its tg number is the floor. Server-side FastMTP
  gives 1.6–2.9× on top depending on acceptance.

## First model

`HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF`, quant `Q8_K_P` (31.5 GB), 262K native
context, embedded MTP head + 900 MB FastMTP draft sidecar (auto-detected when it sits next to
the model). Measured on this machine: pp2048 690 tok/s, tg 16 tok/s base, 26–47 tok/s with FastMTP,
RSS ≈ 48.6 GiB at full context. Sampler (thinking): temp 1.0, top_p 0.95, top_k 20, min_p 0.
Non-thinking: temp 0.7, top_p 0.8, presence_penalty 1.5, `enable_thinking=false`. Vision needs `--vision`
(mmproj not downloaded by default unless `pull` fetched it).
