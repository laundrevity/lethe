# lethe

Run GGUF models from Hugging Face locally on Apple Silicon with a Metal-built `llama.cpp` —
**logless and traceless by default**, and always knowing your **throughput** and **context window**.

```
$ uv run lethe models
Apple M5 Max · RAM 128.0 GiB · GPU-wired limit 107.5 GiB · model+KV budget 103.5 GiB · KV cache f16
in use now: pid 13326 11.8 GiB (llama.cpp) · wired total 4.8 GiB
┃ spec (owner/repo:QUANT)                                              ┃     size ┃ quant  ┃ native ctx ┃ KV/tok ┃ mem @32K ┃ mem @128K ┃ mem @native ┃ max ctx here ┃
│ HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF:Q8_K_P  │ 29.3 GiB │ Q8_K_P │    262,144 │ 64 KiB │ 31.3 GiB │  37.3 GiB │    45.3 GiB │      262,144 │

$ uv run lethe                    # bare `lethe` = `lethe chat`
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
| shell tool | **off**; `chat --tools [--shell-dir PATH]` opts in | macOS sandbox: read-only workspace + system runtime, no network, confirmation for every command; command text and captured output stay in memory unless you save the conversation |
| saved conversations | **off**; `chat --save` / `/save` opt in **per conversation** | AES-256-GCM, key from your passphrase via scrypt, passphrase never stored; random filenames, title/model inside the ciphertext; `lethe conversations` lists ids only (`--unlock` to see titles), `--delete` overwrites then removes |
| clipboard | only on explicit `/copy` | clipboard managers and Universal Clipboard can persist or forward that, so it's never automatic |
| Hugging Face | telemetry disabled; downloads anonymous unless the repo is gated | what you download is still visible to HF, as with any download |
| `lethe.toml` | configuration only (default model) | never content |

`lethe traces` lists exactly what is on disk beyond model files: the opt-in log if any, encrypted
conversations if any, `lethe.toml`, and `models/*/.cache/` (huggingface_hub's bookkeeping of which
files were fetched — no content). `lethe doctor` warns if a log exists. Outside lethe's control: your
terminal's own scrollback (hence `/wipe`) and your shell history (which records the `lethe chat …`
command, not what you typed into it).

By default `chat` sends a short system prompt telling the model what it is (name, family, developer),
what it runs on (chip, RAM), the current date and time (refreshed on `--resume`), that the
conversation is local-only and unobserved, and that post-cutoff reality is real — without that last
part, models tend to read genuine post-cutoff news as fiction or an evaluation scenario. `--system TEXT`
replaces it; `--no-system` sends none.

**Internet tools are the one opt-in exception to "nothing leaves the machine".** `chat --tools web,fetch`
gives the model `web_search` (keyless ddgs metasearch over Google/Brave/Bing/Mojeek/…, or your own
SearXNG via `--searx URL`) and `fetch_url` (page → markdown via trafilatura); `/web` in the REPL turns
them on mid-conversation and `/web off` withdraws them (the default system prompt is patched to match). The model composes the
queries and URLs from your conversation, so treat them as visible to the engines you hit. Every request
is printed before it is sent; `--tools-confirm` additionally asks y/n per request. Results are chunked
(~4K tokens; the model pages through long articles with an offset), never cached or written to disk,
and every URL and redirect hop is checked for private/LAN addresses, with the connection pinned to the
address that was checked (no DNS rebinding). The default system prompt stops claiming
network silence when internet tools are on.

## Sandboxed command-line tool

```bash
uv run lethe --tools                         # shell tool, reading the current directory
uv run lethe --tools --shell-dir ./src       # let the model inspect this directory instead
uv run lethe --web --tools                   # internet tools + shell
uv run lethe --tools web,fetch,shell         # or name tools explicitly (--shell is the same as bare --tools)
```

The model gets `run_shell(command, cwd=".", timeout_seconds=10)`. It can use macOS commands such as
`ls`, `find`, `grep`, `sed`, `awk`, and pipelines to explore files and compute answers. Before every
execution, Lethe shows the workspace and **entire command**, then asks for confirmation (default: no).
`--yes` only affects model downloads; shell approval is always required. `/shell` shows the scope;
`/shell off` removes the tool. `/web off` only removes internet tools.

Ask for an action when you want an actual command, for example:

```text
you › Use run_shell to execute exactly `printf lethe-shell-smoke-ok`, then report the actual stdout.
```

Lethe will show the proposed tool call and ask for approval. Enabling a tool lets the model choose
when to use it; a hypothetical question may receive an explanation without a call. With thinking
enabled, the model may reason before requesting the tool; Ctrl-C cancels that turn. Use `/think off`
or launch with `--no-thinking` to disable reasoning. `--no-show-thinking` only hides its display.
Bare `--tools` (or `--tool`, or `--shell`) enables the shell; `--web --tools` gives internet + shell.

This first version is **read-only**: even writes inside the selected directory are denied, including
temporary files, installs, edits, deletes, and caches. The directory defaults to the current working
directory; home and its parent directories are refused. Select the smallest useful directory: all
files inside it, including `.env` or other secrets, are readable. Symlinks do not grant access to files
outside the allowed paths. OS utilities, libraries, and limited system metadata are also readable.
Homebrew programs, virtual environments whose interpreters live outside the workspace, and build tools
that require scratch files may not work in this initial mode.

Each call starts a fresh noninteractive `/bin/sh` with a fixed `/usr/bin:/bin` PATH, no inherited
credentials/proxies/SSH agent, and no automatic shell startup files. The OS denies network access
(including localhost and Unix sockets), writes, and access to other host processes. Commands receive
a 10-second default wall-clock limit (maximum 30), hard CPU/file-size/file-descriptor limits, and a
16,000-byte combined output cap. Results include `stdout`, `stderr`, `exit_code`, `timed_out`, and
`truncated`; hitting the output cap stops the command. Normal child processes are killed with the
command on timeout, overflow, or Ctrl-C. There are at most 8 tool rounds and 16 calls per user turn.

Enforcement uses macOS `/usr/bin/sandbox-exec` (Seatbelt), with a real startup probe. Unsupported
platforms or sandbox setup errors disable the shell tool; there is no unsandboxed fallback. Apple
marks this interface deprecated, and it may stop working on a future macOS release. This is a
filesystem/network boundary, **not a VM or a complete resource-isolation system**: memory and aggregate
process count are not capped, and deliberately detached processes can escape process-group cleanup
(they still inherit the sandbox). Background daemons and hostile-code analysis are unsupported.
macOS itself may retain diagnostics; Lethe does not write command scripts, history, profiles, or output
files. Enabling `--web` separately still allows the model to send queries/URLs to the internet, so shell
read access and internet tools together can expose information from files in the chosen workspace.

Run the ordinary tests with `uv run pytest`. To verify the actual OS restrictions on macOS, run
`uv run pytest tests/test_shell.py --run-sandbox` from a normal terminal; nested sandboxes may prevent
the startup probe. These tests check allowed reads/pipelines and denied writes, outside reads,
symlink escapes, loopback/Unix sockets, host-process access, timeouts, and output flooding.

To check the model → shell → model path against an already-running server on port 8080, run
`uv run pytest -q -s tests/test_live_tools.py --run-live-tools`. This opt-in test uses the normal
thinking sampler, approves only the fixed `printf lethe-shell-smoke-ok` command in the read-only
sandbox, and checks the final answer. It neither starts a server nor saves the conversation.

## Setup

```bash
git submodule update --init      # vendor/llama.cpp
scripts/build-llama.sh           # Metal + Accelerate, Release; applies model patches if present
uv sync
uv run lethe doctor              # confirms Metal is linked and reports the GPU memory budget
uv run lethe default HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF:Q8_K_P
uv run lethe                     # downloads (with confirmation) on first use, then serves + chats
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
| `lethe [chat] [spec] [-f FILE …] [--save] [--resume ID] [--web] [--tools [--shell-dir PATH]] [--system TEXT] [--no-system]` (the command word is optional) | streaming REPL; after every turn: prompt tok/s, gen tok/s, MTP acceptance, context used/total (colored; warns at 80%) |
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

`lethe bench` (llama-bench, no speculative decoding):

| | pp2048 | tg128 |
|---|---:|---:|
| depth 0 | 690 tok/s | 16.3 tok/s |
| depth 32K | 348 tok/s | 15.6 tok/s |

`lethe bench --server -d 0` (llama-server, greedy, 256-token replies over prose/reasoning/code prompts):

| config | pp tok/s | tg tok/s | accept | vs no-MTP |
|---|---:|---:|---:|---:|
| no MTP | 746 | 16.8 | — | 1.00× |
| embedded MTP, depth 3 | 733 | 31.8 | 68% | 1.90× |
| FastMTP sidecar, depth 3 | 733 | 36.2 | 62% | 2.16× |

Acceptance (and therefore speed) depends on the text: 48% on free prose, 74% on code in the same run.

## Layout

```
scripts/build-llama.sh   canonical build (flags documented inside)
scripts/bump-llama.sh    move to latest upstream llama.cpp, re-apply the FastMTP patch, rebuild, verify
src/lethe/               cli.py · repl.py · server.py · chat.py · spec.py · profiles.py · vault.py ·
                         tools.py · shell.py · attachments.py · benchserver.py · hub.py · ggufmeta.py · sysinfo.py · config.py · paths.py
tests/                   pytest (uv run pytest)
vendor/llama.cpp         submodule; the FastMTP patch is committed on local branch `lethe-fastmtp`
models/                  downloaded GGUFs (gitignored)
.lethe/                  empty unless you opted in to --log or --save (gitignored)
```
