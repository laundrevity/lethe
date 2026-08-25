# lethe

Run GGUF models from Hugging Face locally on Apple Silicon with a Metal-built `llama.cpp`,
and always know your **throughput** and **context window**.

```
$ uv run lethe models
Apple M5 Max · RAM 128.0 GiB · GPU-wired limit 107.5 GiB · model+KV budget 103.5 GiB · KV cache f16
┃ model                          ┃     size ┃ quant  ┃ native ctx ┃ KV/tok ┃ mem @32K ┃ mem @128K ┃ mem @native ┃ max ctx here ┃
│ …Aggressive-Q8_K_P.gguf        │ 29.3 GiB │ Q8_K_P │    262,144 │ 64 KiB │ 31.3 GiB │  37.3 GiB │    45.3 GiB │      262,144 │

$ uv run lethe chat q8_k_p
context 262,144 (native max 262,144) · model 29.3 GiB + draft 862 MiB + KV 16.0 GiB (f16, 64 KiB/tok) = 46.1 GiB of 103.5 GiB budget
speculative: FastMTP draft …FastMTP-32K.gguf (depth 3) · thinking on
ready in 3.6s at http://127.0.0.1:8080 · n_ctx 262,144
you › what is 17*23?
17 times 23 is 391.
prompt 68 tok @ 148 tok/s · gen 89 tok @ 46.7 tok/s · mtp accept 87% (65/75) · ctx 157/262,144 (0.1%) · wall 2.4s
```

## Logless by default

The point of running locally is that the conversation exists nowhere but in your head and, briefly, in RAM.
So the defaults are:

| surface | default | notes |
|---|---|---|
| llama-server log | **off** (`--log-disable`, stdout → `/dev/null`) | `serve --log` opts in to `.lethe/server.log` for debugging; `lethe traces --delete` removes it |
| web UI | **off** (`--no-webui`) | the stock UI stores chat history in the browser |
| `/slots` endpoint | **off** (`--no-slots`) | it exposes the live prompt text over HTTP |
| `/metrics` | off | |
| prompt/KV cache to disk (`--slot-save-path`) | never set | KV lives in GPU-wired memory only |
| pid / state files | none | running servers are found via the process table |
| network | `127.0.0.1` only | |
| `chat` REPL | writes nothing; no history file (Up-arrow recall is in-memory only) | `/wipe` clears the conversation **and** the terminal scrollback |
| clipboard | only on explicit `/copy` | puts the last reply on the macOS clipboard — clipboard managers and Universal Clipboard can persist or forward that, so it's never automatic |
| Hugging Face | telemetry disabled; downloads anonymous unless the repo is gated | what you download is still visible to HF, as with any download |

What's left on disk: the model files, and `models/*/.cache/` (huggingface_hub's bookkeeping of which
files/revisions were fetched — no content). `lethe traces` lists exactly this; `lethe doctor` warns if a
log exists. Outside lethe's control: your terminal's own scrollback and your shell history (which records
the `lethe chat …` command, not what you typed into it).

## Setup

```bash
git submodule update --init      # vendor/llama.cpp
scripts/build-llama.sh           # Metal + Accelerate, Release; applies model patches if present
uv sync
uv run lethe doctor            # confirms Metal is linked and reports the GPU memory budget
```

## Commands

| command | what it does |
|---|---|
| `lethe browse <hf-repo>` | every GGUF in a repo with size, quant, native context, KV bytes/token, memory at 32K / 128K / native context, and the max context that fits on *this* machine |
| `lethe pull <hf-repo> -q Q8_K_P` | download a quant (plus mmproj / draft sidecars) into `models/<owner>--<repo>/` |
| `lethe models` | the same table for what's on disk |
| `lethe serve <model> [--ctx N] [--kv q8_0] [--no-mtp] [--no-thinking] [--effort xhigh]` | launch `llama-server` with the memory plan printed; defaults to the native max context if it fits |
| `lethe chat <model>` | streaming REPL; after every turn: prompt tok/s, gen tok/s, MTP acceptance, context used/total. Paste multi-line text as one message; Enter sends, Ctrl+Enter adds a line (also Shift+Enter / Alt+Enter / Ctrl+J / trailing `\`); `/copy` puts the last reply on the clipboard |
| `lethe bench <model> -p 2048 -n 128 -d 0,32768` | `llama-bench` prompt/gen speeds, optionally at context depth (no speculative decoding — this is the floor) |
| `lethe stop` / `lethe log` | manage the background server (`log` only has content after `serve --log`) |
| `lethe traces [--delete]` | list/remove anything written to disk beyond model files |

`<model>` is a path or any unique substring of the file under `models/` (e.g. `q8_k_p`).
`lethe serve` leaves the server running in the background; `lethe chat` reuses it.
It is a stock `llama-server`, so anything that speaks the OpenAI API can hit `http://127.0.0.1:8080/v1`.

## How the memory columns are computed

`file size + KV cache`, where KV per token = `attention layers × KV heads × (K dim + V dim) × 2 bytes` (f16).
The header of each GGUF is read directly (only the first ~16 MiB, over HTTP for remote files), so hybrid
models such as Qwen3.5/3.8 (Gated DeltaNet + 1-in-4 attention layers) are costed correctly — their long
context is far cheaper than a dense model of the same size. The budget is what Metal reports as
`recommendedMaxWorkingSetSize` minus 4 GiB of headroom. Real usage adds 1–3 GiB of compute buffers.

## Reference numbers — M5 Max, 128 GB, Qwen3.8-27B Q8_K_P

| | pp2048 | tg128 |
|---|---:|---:|
| `llama-bench` (no MTP), depth 0 | 690 tok/s | 16.3 tok/s |
| `llama-bench` (no MTP), depth 32K | 348 tok/s | 15.6 tok/s |
| `llama-server` + FastMTP depth 3, chat | — | 26–47 tok/s (acceptance 33–87%) |

## Layout

```
scripts/build-llama.sh   canonical build (flags documented inside)
src/lethe/             cli.py · server.py · chat.py · hub.py · ggufmeta.py · sysinfo.py · paths.py
vendor/llama.cpp         submodule; the FastMTP patch is committed on local branch `lethe-fastmtp`
models/                  downloaded GGUFs (gitignored)
.lethe/                  empty unless you ran `serve --log` (gitignored)
```
