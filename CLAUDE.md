# lethe

Local LLM runner for Apple Silicon: a vendored, Metal-built `llama.cpp` driven by a small Python CLI.
Target machine: M5 Max, 128 GB unified memory, 40 GPU cores, macOS 26. See README.md for usage.

## Prime directive: logless and traceless by default

The reason this project exists (vs. Claude/ChatGPT/Codex) is that nobody but the user ever sees the
model's inputs and outputs. Every feature must preserve that:

- Conversation content never touches disk. No server log (`--log-disable`, stdout → /dev/null), no
  `--slot-save-path`, no pid/state files, no REPL history. `serve --log` is the only opt-in, and
  `lethe traces --delete` cleans it up. Never add a "save chat" default; if you add persistence, it
  must be explicit opt-in per invocation and be listed by `lethe traces`.
- llama-server always runs with `--no-webui --no-slots`, bound to 127.0.0.1, no `--metrics`.
- Running servers are discovered from the process table (`server.running_servers()`), filtered to our
  own binary path — the user runs other llama-servers on this machine (e.g. port 8090); never touch them.
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
- **Submodule pin = upstream master `f280b26` (2026-08-24) + one local commit** on branch
  `lethe-fastmtp` applying `models/**/HauhauCS-FastMTP-llama.cpp.patch` (a `d2t` draft-vocab trim in
  `src/models/qwen35.cpp`). The build script applies/commits the patch automatically. Bumping upstream
  means re-running `git apply --check` on that patch; `qwen35.cpp` is the only file it touches.
- **Build flags are canonical in `scripts/build-llama.sh`.** Metal + Accelerate explicitly ON, Metal
  shader library embedded, `GGML_NATIVE`, Release, curl OFF, `CMAKE_BUILD_RPATH=@loader_path` (without it
  the binaries carry an absolute rpath and break when the repo is moved/renamed — exit -6 from dyld).
  Targets: `llama-server llama-cli llama-bench` (there is no `llama-gguf` target). `lethe doctor` verifies
  Metal is linked and lists devices.
- **Models live in `./models/<owner>--<repo>/`**, downloaded via `huggingface_hub` (`hf` CLI is part of
  that package), not the shared `~/.cache/huggingface` — files are 10–60 GB and we want them explicit.
  `models/` is gitignored. `HF_TOKEN` is set in the user's shell.
- **Throughput and context are first-class.** Every chat turn prints prompt tok/s, gen tok/s, MTP
  acceptance, and context used/configured. `browse`/`models` show KV bytes/token and the max context
  that fits. KV math lives in `ggufmeta.py` (own ~100-line GGUF header parser; reads only the first
  16 MiB, works over HTTP Range for remote files — never downloads a model to inspect it).
- **Memory budget** = Metal's `recommendedMaxWorkingSetSize` as reported by `llama-server --list-devices`
  (107.5 GiB on this machine — higher than the 75%-of-RAM folk rule) minus 4 GiB headroom.
- Python is managed with **`uv`** (`uv run lethe ...`, `uv add ...`). Python 3.12, src layout, typer + rich.

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
context, embedded MTP head + 900 MB FastMTP draft sidecar (auto-detected by `serve` when it sits next to
the model). Measured on this machine: pp2048 690 tok/s, tg 16 tok/s base, 26–47 tok/s with FastMTP,
RSS ≈ 48.6 GiB at full context. Sampler (thinking): temp 1.0, top_p 0.95, top_k 20, min_p 0.
Non-thinking: temp 0.7, top_p 0.8, presence_penalty 1.5, `enable_thinking=false`. Vision needs `--vision`
(mmproj not downloaded by default unless `pull` fetched it).
