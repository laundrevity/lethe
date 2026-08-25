# lethe

Run GGUF models from Hugging Face locally on Apple Silicon with a Metal-built `llama.cpp` —
**logless and traceless by default**, and always knowing your **throughput** and **context window**.

```
$ uv run lethe models
Apple M5 Max · RAM 128.0 GiB · GPU-wired limit 107.5 GiB · model+KV budget 103.5 GiB · KV cache f16
in use now: pid 13326 11.8 GiB (llama.cpp) · wired total 4.8 GiB
┃ spec (owner/repo:QUANT)                                              ┃     size ┃ quant  ┃ native ctx ┃ KV/tok ┃ mem @32K ┃ mem @128K ┃ mem @native ┃ max ctx here ┃
│ HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF:Q8_K_P  │ 29.3 GiB │ Q8_K_P │    262,144 │ 64 KiB │ 31.3 GiB │  37.3 GiB │    45.3 GiB │      262,144 │

$ uv run lethe chat
context 262,144 (native max 262,144) · model 29.3 GiB + draft 862 MiB + KV 16.0 GiB (f16, 64 KiB/tok) = 46.1 GiB of 99.0 GiB free now
speculative: FastMTP draft …FastMTP-32K.gguf (depth 3) · thinking on
ready in 3.6s at http://127.0.0.1:8080 · n_ctx 262,144 · traceless: no log, no web UI, no slots endpoint
sampler: qwen3 thinking preset (temp 1.0 top_p 0.95 top_k 20 min_p 0.0) · /help for commands
you › /file paper.pdf
paper.pdf attached to your next message (~9,800 tokens)
you › what's the main result?
…
prompt 9,912 tok @ 655 tok/s · gen 212 tok @ 33.4 tok/s · mtp accept 52% (140/268) · ctx 10,124/262,144 (3.9%) · wall 21.7s
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
| API access | **per-launch random key** (`--api-key`) | other local software — including a web page doing `fetch("http://localhost:8080")` — can't use the model. `lethe key` prints it for clients you trust; `serve --no-api-key` disables |
| prompt/KV cache to disk (`--slot-save-path`) | never set | KV lives in GPU-wired memory only |
| pid / state files | none | running servers are found via the process table |
| network | `127.0.0.1` only | |
| `chat` REPL | writes nothing; no history file (Up-arrow recall is in-memory only) | `/wipe` clears the conversation **and** the terminal scrollback |
| saved conversations | **off**; `chat --save` / `/save` opt in **per conversation** | AES-256-GCM, key from your passphrase via scrypt, passphrase never stored; random filenames, title/model inside the ciphertext; `lethe conversations` lists ids only (`--unlock` to see titles), `--delete` overwrites then removes |
| clipboard | only on explicit `/copy` | clipboard managers and Universal Clipboard can persist or forward that, so it's never automatic |
| Hugging Face | telemetry disabled; downloads anonymous unless the repo is gated | what you download is still visible to HF, as with any download |
| `lethe.toml` | configuration only (default model) | never content |

`lethe traces` lists exactly what is on disk beyond model files: the opt-in log if any, encrypted
conversations if any, `lethe.toml`, and `models/*/.cache/` (huggingface_hub's bookkeeping of which
files were fetched — no content). `lethe doctor` warns if a log exists. Outside lethe's control: your
terminal's own scrollback (hence `/wipe`) and your shell history (which records the `lethe chat …`
command, not what you typed into it).

## Setup

```bash
git submodule update --init      # vendor/llama.cpp
scripts/build-llama.sh           # Metal + Accelerate, Release; applies model patches if present
uv sync
uv run lethe doctor              # confirms Metal is linked and reports the GPU memory budget
uv run lethe default HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF:Q8_K_P
uv run lethe chat                # downloads (with confirmation) on first use, then serves + chats
```

## Models are `owner/repo:QUANT`

Every command takes a model as `owner/repo:QUANT` (what `lethe browse` prints), a `.gguf` path, or any
unique substring of a local file name (`q8_k_p`). A model that isn't downloaded yet is offered for download
(`-y` to skip the question). `lethe default <spec>` records the one you use most in `lethe.toml`.

The sampler comes from a per-family profile (Qwen3.x, Gemma, MiniMax-M2, DeepSeek, GLM, generic) chosen
from the GGUF's architecture, following each model card's recommendations; override per flag
(`--temp`, `--top-p`, `--top-k`, `--min-p`). Multi-shard GGUFs are handled as one model.

## Commands

| command | what it does |
|---|---|
| `lethe browse owner/repo` | every GGUF in a repo with size, quant, native context, KV bytes/token, memory at 32K / 128K / native context, the max context that fits on *this* machine, and which are already local |
| `lethe pull owner/repo:QUANT` | download a quant (plus mmproj / draft sidecars) into `models/<owner>--<repo>/` |
| `lethe models` | the same table for what's on disk, plus what other model processes are holding right now |
| `lethe default [spec]` | show / set the default model |
| `lethe serve [spec] [--ctx N] [--kv q8_0] [--no-mtp] [--no-thinking] [--effort xhigh] [--vision]` | launch `llama-server` with the memory plan printed against what is *free now*; defaults to the native max context if it fits |
| `lethe chat [spec] [-f FILE …] [--save] [--resume ID]` | streaming REPL; after every turn: prompt tok/s, gen tok/s, MTP acceptance, context used/total (colored; warns at 80%) |
| `lethe conversations [--unlock] [--delete ID]` | encrypted saved conversations |
| `lethe bench [spec] [-d 0,32768]` | `llama-bench` prompt/gen speeds at context depth — **no speculative decoding, this is the floor** |
| `lethe bench [spec] --server [-d 0,32768]` | through `llama-server`: no-MTP vs embedded MTP vs FastMTP, real tok/s and acceptance |
| `lethe stop` / `lethe key` / `lethe log` | manage the background server (`log` only has content after `serve --log`) |
| `lethe traces [--delete]` | list/remove anything written to disk beyond model files |
| `lethe doctor` | Metal build, memory budget, what's in use, what's on disk |

In the REPL: `/file PATH` and `/image PATH` (with `--vision`) attach to the next message; `/compact`
summarizes the conversation with the model and continues from the summary; `/think on|off`; `/copy`;
`/wipe`; `/save`; `/help`. Enter sends; Ctrl+Enter, Shift+Enter, Alt+Enter, Ctrl+J or a trailing `\`
add a line; multi-line pastes arrive as one message. Apple Terminal sends Ctrl+Enter as plain Enter —
map Control-Return to `\012` in its Keyboard settings, or use the alternatives.

`lethe serve` leaves the server running in the background; `lethe chat` reuses it. It is a stock
`llama-server`, so anything that speaks the OpenAI API can use `http://127.0.0.1:8080/v1` with the key
from `lethe key`.

## How the memory columns are computed

`file size + KV cache`, where KV per token = `attention layers × KV heads × (K dim + V dim) × 2 bytes` (f16).
The header of each GGUF is read directly (only the first ~16 MiB, over HTTP for remote files), so hybrid
models such as Qwen3.5/3.8 (Gated DeltaNet + 1-in-4 attention layers) are costed correctly — their long
context is far cheaper than a dense model of the same size. Capacity is what Metal reports as
`recommendedMaxWorkingSetSize` minus 4 GiB of headroom; "free now" subtracts memory already wired by
other processes (e.g. another llama-server). Real usage adds 1–3 GiB of compute buffers.

## Reference numbers — M5 Max, 128 GB, Qwen3.8-27B Q8_K_P

| | pp2048 | tg128 |
|---|---:|---:|
| `llama-bench` (no MTP), depth 0 | 690 tok/s | 16.3 tok/s |
| `llama-bench` (no MTP), depth 32K | 348 tok/s | 15.6 tok/s |
| `llama-server` + FastMTP depth 3, chat | 650–700 tok/s on real prompts | 26–51 tok/s (acceptance 33–93%) |

`lethe bench --server` reproduces the last row for any model, with and without the draft.

## Layout

```
scripts/build-llama.sh   canonical build (flags documented inside)
scripts/bump-llama.sh    move to latest upstream llama.cpp, re-apply the FastMTP patch, rebuild, verify
src/lethe/               cli.py · repl.py · server.py · chat.py · spec.py · profiles.py · vault.py ·
                         attachments.py · benchserver.py · hub.py · ggufmeta.py · sysinfo.py · config.py · paths.py
tests/                   pytest (uv run pytest)
vendor/llama.cpp         submodule; the FastMTP patch is committed on local branch `lethe-fastmtp`
models/                  downloaded GGUFs (gitignored)
.lethe/                  empty unless you opted in to --log or --save (gitignored)
```
