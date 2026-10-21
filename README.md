# lethe

Run GGUF models from Hugging Face locally on Apple Silicon (Metal) or Linux (Vulkan/ROCm) with `llama.cpp` —
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
ready in 3.6s · traceless: no log, no web UI, no slots endpoint; the model is unloaded when you quit
sampler: qwen3 thinking preset (temp 1.0 top_p 0.95 top_k 20 min_p 0.0) · /help for commands
you › /file paper.pdf
paper.pdf attached to your next message (~9,800 tokens)
you › what's the main result?
…
prompt 9,912 tok @ 655 tok/s · gen 212 tok @ 33.4 tok/s · mtp accept 52% (140/268) · ctx 10,124/262,144 (3.9%) · wall 21.7s
```

## Logless by default

Lethe keeps conversations in RAM unless you explicitly save them. Full-access shell and file tools
are enabled by default: approved commands may write files, send data, and leave their own logs.
Use `lethe --no-tools` for a session without model tools.
So the defaults are:

| surface | default | notes |
|---|---|---|
| llama-server log | **off** (`--log-disable`, output → `/dev/null`) | `chat --debug-server` shows the server's output on the terminal instead; nothing is written |
| web UI | **off** (`--no-webui`) | the stock UI stores chat history in the browser |
| `/slots` endpoint | **off** (`--no-slots`) | it exposes the live prompt text over HTTP |
| `/metrics` | off | |
| API access | **per-launch random key** (`--api-key`), on a free loopback port | other local software — including a web page doing `fetch("http://localhost:…")` — can't use the model |
| prompt/KV cache to disk (`--slot-save-path`) | never set | KV lives in GPU-wired memory only |
| model process | **lives only as long as the chat** | `lethe` starts `llama-server` when it starts and kills it when it exits, however it exits; there is no daemon, no pid file, nothing holding 30 GB between chats |
| model server network | `127.0.0.1` only | approved full-shell commands can use the network |
| `chat` REPL | writes nothing; no history file (Up-arrow recall is in-memory only) | `/wipe` clears the conversation **and** the terminal scrollback |
| terminal notifications | bell + a fixed phrase ("lethe: reply finished", "lethe: a tool call needs your approval") | Notification Center keeps notification text around, so no conversation content ever goes into one |
| shell and file tools | **on, full access** (equivalent to `--tools --shell-mode full`) | commands and file operations require approval; writes show a diff. `--no-tools` disables tools; `--shell-mode read-only` selects the offline macOS sandbox. Full mode allows writes, network, and inherited environment; command side effects can persist |
| saved conversations | **off**; `chat --save` / `/save` opt in **per invocation** — `--resume ID` only reads | AES-256-GCM, key from your passphrase via scrypt, passphrase never stored; random filenames, title/model inside the ciphertext; `lethe conversations` lists ids only (`--unlock` to see titles), `--delete` overwrites then removes |
| clipboard | only on explicit `/copy` | clipboard managers and Universal Clipboard can persist or forward that, so it's never automatic |
| Hugging Face | telemetry disabled; downloads anonymous unless the repo is gated | what you download is still visible to HF, as with any download |
| `lethe.toml` | configuration only (default model) | never content |

`lethe traces` lists exactly what is on disk beyond model files: the opt-in log if any, encrypted
conversations if any, `lethe.toml`, and `models/*/.cache/` (huggingface_hub's bookkeeping of which
files were fetched — no content). `lethe doctor` warns if a log exists. Outside lethe's control: your
terminal's own scrollback (hence `/wipe`) and your shell history (which records the `lethe chat …`
command, not what you typed into it).

Saved conversations include the model's thinking, tool calls and tool results inside the encrypted file.
`--resume ID` replays the saved thinking and tool calls (with result sizes) before continuing. Resuming
never writes: the file stays as it was unless you also pass `--save`, or type `/save` during the chat — both
keep saving to the same id with the passphrase you unlocked it with, and the exit line says when a session's
new turns were not kept.
`--no-show-thinking` hides thinking in both replay and live output without removing it from saved messages.
Older saves can replay their tool calls and any thinking they retained, but thinking discarded by earlier
versions cannot be recovered. Replaying a call never executes it again.

By default `chat` sends a short system prompt telling the model what it is (name, family, developer),
what it runs on (chip, RAM), the current date and time (refreshed on `--resume`), that the
conversation is local-only and unobserved, and that post-cutoff reality is real — without that last
part, models tend to read genuine post-cutoff news as fiction or an evaluation scenario. `--system TEXT`
replaces it; `--no-system` sends none.

**`LETHE.md` — your standing instructions.** Markdown files appended to the system prompt, the way `CLAUDE.md` and
`AGENTS.md` work for other agents. Two layers, both optional:

| Layer | File | Applies to |
|---|---|---|
| global | `LETHE.md` in the lethe repo root | every chat, from any directory |
| project | the nearest `LETHE.md` (else `AGENTS.md`) from the launch directory (or `--shell-dir`) upward, stopping at the repository root or your home directory | chats launched in that project |

Project instructions take precedence where the two conflict. The files loaded are listed at launch with their
approximate token cost; they are only read, never written, and are re-read on `--resume` (replacing the copy saved
with the conversation). `--no-instructions` skips them, as does `--no-system`; `--system TEXT` keeps them. A
project file is part of the system prompt, so read an unfamiliar repository's `AGENTS.md` before chatting inside it.

**Full-access shell tools are enabled by default; dedicated web search/fetch tools remain opt-in.** `chat --tools web,fetch`
gives the model `web_search` (keyless ddgs metasearch over Google/Brave/Bing/Mojeek/…, or your own
SearXNG via `--searx URL`) and `fetch_url` (page → markdown via trafilatura); `/web` in the REPL turns
them on mid-conversation and `/web off` withdraws them (the default system prompt is patched to match). The model composes the
queries and URLs from your conversation, so treat them as visible to the engines you hit. Every request
is printed before it is sent; `--tools-confirm` additionally asks y/n per request. Results are chunked
(~4K tokens; the model pages through long articles with an offset), never cached or written to disk,
and every URL and redirect hop is checked for private/LAN addresses, with the connection pinned to the
address that was checked (no DNS rebinding). The default system prompt stops claiming
network silence when internet tools or full-access shell mode are on. Full-access commands can also
write files or logs; Lethe's in-memory conversation handling does not prevent those command side effects.

## Command-line tool

```bash
uv run lethe                                # full shell + file tools, with approvals
uv run lethe --no-tools                     # no model tools
uv run lethe --shell-mode read-only         # offline macOS sandbox
uv run lethe --shell-dir ./src              # start shell commands in this directory
uv run lethe --web --tools                   # internet tools + shell
uv run lethe --web --tools --shell-mode full  # internet tools + full-access shell
uv run lethe --tools web,fetch,shell         # or name tools explicitly (--shell is the same as bare --tools)
```

The model gets `run_shell(command, cwd=".", timeout_seconds=...)`. Before every execution, Lethe
shows the mode, workspace or starting directory, and **entire command**, then asks for confirmation (default: no).
The answer is `y` to run it, `n` (or just Enter) to decline, or `p` — perhaps — to send the model a note instead
of a verdict: `p use ninja instead`, or `p` and then the note on the next line. Nothing runs on `p`; the model
gets the note as the call's result and may ask again. The terminal bell rings while a call waits for you.
`--yes` only affects model downloads; shell approval is always required. `/shell` shows the scope;
`/shell off` removes the tool. `/web off` only removes internet tools.

The shell brings file tools with it, so the model does not have to build files out of `cat` heredocs and `sed`:

| Tool | Modes | What it does |
|---|---|---|
| `read_file(path, offset=1, limit)` | read-only (inside the workspace only), full | UTF-8 text with line numbers, paged at 16,000 chars |
| `write_file(path, content)` | full only | create a file or replace all of it; parent directories are created |
| `edit_file(path, old_string, new_string, replace_all=false)` | full only | replace one exact, unique piece of text; failed matches tell the model how to fix the call |

Every file tool call asks for confirmation like a shell command. Writes and edits are shown as a **diff** of exactly
what will change; an edit that cannot apply is returned to the model without prompting you. Lethe keeps no backups,
temp files, or edit history. `/shell off` removes these tools too, and `/shell read-only` drops the two write tools.

`--auto-approve files` skips the y/n prompt for the three file tools (shell commands still ask);
`--auto-approve all` skips it for shell commands too. It is per launch, never a saved setting, and every call and
diff is still printed. With `all` in full mode the model runs arbitrary commands as you with nobody checking, and
with `--web` a fetched page can steer what it runs, so prefer `files` unless you are watching.

Any chat can enable tools later, including chats started with no tool flags. Enter these commands
at the `you ›` prompt; the conversation continues with the new tools on the next message:

| Chat command | Effect |
|---|---|
| `/web` | Enable web search and page fetching |
| `/shell full` | Enable or upgrade to full shell access |
| `/tools full` | Enable both web tools and full shell access |
| `/tools full "~/dev/my project"` | Enable both, starting the shell in the selected directory |
| `/shell read-only` | Switch to the read-only shell; preserve web tools |
| `/shell off` | Disable the shell; preserve web tools |
| `/tools off` | Disable all tools |

`/tools` shows current access. `/shell full` and `/shell read-only` also accept an optional directory;
otherwise they reuse the launch directory or the last directory you selected, even after switching
tools off. The built-in model instructions explain that the model may ask you to enable these tools
when needed. Typing the command grants access; shell executions still ask for approval individually.
Tool descriptions and the built-in system prompt update immediately. A failed mode change leaves
the existing tools and directory unchanged.

`--shell-mode full` is the default and selects a separate runner without
Lethe's sandbox, running as your current user. Full mode enables file edits, builds, installs, and
network access (including localhost) using your inherited PATH, environment, credentials, proxies,
and tool configuration. It works with Homebrew tools and virtual environments on that PATH.
`--shell-dir` sets the starting directory in full mode; commands can access other directories with
your account's normal permissions. `cwd` accepts relative paths, absolute paths, and `~`.
Each call starts a fresh noninteractive `/bin/sh`; shell aliases and functions from an interactive
shell are not loaded. Both modes still require confirmation for every command.

For example, `uv run lethe --tools --shell-mode full --shell-dir ~/dev/my-project` enables full
shell access starting in that project. Add `--web` for the dedicated search/fetch tools; shell
networking in full mode works independently of `--web`. `--tools web,fetch` selects only dedicated
web tools; `--no-tools` disables all tools at launch. In chat, use `/shell full`, `/shell read-only`, or `/tools full` to change
access without restarting. The model can ask for access; you grant it with a chat command.
A sandbox failure never switches modes automatically.

Ask for an action when you want an actual command, for example:

```text
you › Use run_shell to execute exactly `printf lethe-shell-smoke-ok`, then report the actual stdout.
```

Lethe will show the proposed tool call and ask for approval. Enabling a tool lets the model choose
when to use it; a hypothetical question may receive an explanation without a call. With thinking
enabled, the model may reason before requesting the tool. Use `/think off`
or launch with `--no-thinking` to disable reasoning. `--no-show-thinking` only hides its display.
Bare `--tools` (or `--tool`, or `--shell`) enables the shell; `--web --tools` gives internet + shell.

In **read-only mode**, even writes inside the selected directory are denied, including
temporary files, installs, edits, deletes, and caches. The directory defaults to the current working
directory; home and its parent directories are refused. Select the smallest useful directory: all
files inside it, including `.env` or other secrets, are readable. Symlinks do not grant access to files
outside the allowed paths. OS utilities, libraries, and limited system metadata are also readable.
Homebrew programs, virtual environments whose interpreters live outside the workspace, and build tools
that require scratch files may not work in this mode.

Read-only commands use a fixed `/usr/bin:/bin` PATH, no inherited
credentials/proxies/SSH agent, and no automatic shell startup files. The OS denies network access
(including localhost and Unix sockets), writes, and access to other host processes. Commands receive
a 10-second default wall-clock limit (maximum 30), hard CPU/file-size/file-descriptor limits, and a
16,000-byte combined output cap; hitting that cap stops the command in read-only mode.
Full mode defaults to a 120-second wall-clock limit (maximum 600) and uses normal resource limits.
It captures up to the same output cap, discarding excess output while the command continues until
exit or timeout, so verbose builds can finish. Results include `stdout`, `stderr`, `exit_code`,
`timed_out`, and `truncated`. Both modes clean up normal child processes when a call finishes,
times out, or is interrupted with Ctrl-C. There are at most 8 tool rounds and 16 calls per user turn.

Read-only enforcement uses macOS `/usr/bin/sandbox-exec` (Seatbelt), with a real startup probe. Unsupported
platforms or sandbox setup errors disable read-only mode; there is no automatic unsandboxed fallback. Apple
marks this interface deprecated, and it may stop working on a future macOS release. This is a
filesystem/network boundary, **not a VM or a complete resource-isolation system**: memory and aggregate
process count are not capped, and deliberately detached processes can escape process-group cleanup
(they still inherit the sandbox). Background daemons and hostile-code analysis are unsupported.
macOS itself may retain diagnostics; Lethe does not write command scripts, history, profiles, or output
files, but commands in full mode can write files and logs themselves. Enabling `--web` separately
still allows the model to send queries/URLs to the internet, so shell read access and internet tools
together can expose information from files in the chosen workspace.

Run the ordinary tests with `uv run pytest`. To verify the actual OS restrictions on macOS, run
`uv run pytest tests/test_shell.py --run-sandbox` from a normal terminal; nested sandboxes may prevent
the startup probe. These tests check allowed reads/pipelines and denied writes, outside reads,
symlink escapes, loopback/Unix sockets, host-process access, timeouts, and output flooding, plus
allowed localhost connections in full mode. Ordinary tests also exercise full-mode writes,
environment/PATH inheritance, confirmation, CLI selection, and enabling/changing tools mid-chat
without losing history. They also check failed mode changes preserve the previous access.

To check the model → shell → model path with the default model, run
`uv run pytest -q -s tests/test_live_tools.py --run-live-tools`. This opt-in test loads the model
for its own run, uses the normal thinking sampler, approves only the fixed `printf lethe-shell-smoke-ok`
command in the read-only sandbox, checks the final answer, and does not save the conversation.

## Setup

Requires CMake, a C++ compiler, and `uv`. On Apple Silicon macOS, install Xcode Command Line Tools.
On Linux, Vulkan needs the Vulkan development files, `glslc`, and SPIRV-Headers; ROCm needs HIP,
hipBLAS, and rocBLAS (the Strix Halo target is `gfx1151`).
Model files, local configuration, and encrypted conversations are kept out of Git;
set the default model and download it on each machine.

```bash
git clone --recurse-submodules https://github.com/laundrevity/lethe.git
cd lethe
uv sync
uv run lethe pull HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF:Q8_K_P
scripts/build-llama.sh           # Metal on macOS, ROCm on Linux; applies scripts/fastmtp.patch
uv run lethe doctor              # checks the selected GPU backend and memory budget
uv run lethe default HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF:Q8_K_P
uv run lethe                     # loads the model, chats, then unloads it on exit
```

Linux backend comparison (same upstream source and FastMTP patch):

```bash
scripts/build-llama.sh --backend=vulkan
scripts/build-llama.sh --backend=hip
LETHE_BACKEND=vulkan uv run lethe bench --server --ctx 262144 -d 0,32768
LETHE_BACKEND=hip uv run lethe bench --server --ctx 262144 -d 0,32768
LETHE_BACKEND=vulkan uv run lethe  # select Vulkan for this invocation
```

Linux defaults to ROCm/HIP; macOS defaults to Metal. The builds stay in separate directories.
`AMDGPU_TARGETS` overrides the HIP architecture; `JOBS` limits build parallelism.
The build also searches `.deps/` for optional project-local development dependencies.
If SPIRV-Headers is unavailable from the system package manager, it can be installed locally:

```bash
git clone https://github.com/KhronosGroup/SPIRV-Headers.git /tmp/lethe-spirv-headers
cmake -S /tmp/lethe-spirv-headers -B /tmp/lethe-spirv-headers/build -DCMAKE_INSTALL_PREFIX="$PWD/.deps"
cmake --install /tmp/lethe-spirv-headers/build
```

Read-only shell tools still require macOS Seatbelt and fail closed on Linux; full shell mode
is the default, with approval for every command unless you opt into auto-approval.

The upstream pin is `620d2225c` (2026-10-15), with the rebased FastMTP patch in
`scripts/fastmtp.patch`. `scripts/bump-llama.sh` checks that patch before moving to a newer pin.

For an existing clone, run `git submodule update --init` before building.
The `improvements` branch is also published; use `git switch improvements` to check it out.

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
| `lethe [chat] [spec] [-f FILE …] [--save] [--resume ID] [--web] [--tools [--shell-mode full] [--shell-dir PATH]] [--system TEXT] [--no-system] [--ctx N] [--kv q8_0] [--no-thinking] [--effort low] [--vision]` (the command word is optional) | loads the model for the session (memory plan printed against what is *free now*; native max context if it fits) and unloads it on exit; streaming REPL (Ctrl-C stops a reply but keeps what it thought and wrote so far; the next prompt tells the model it was cut short); while a reply is produced, a line pinned to the bottom of the terminal shows context used (≈ until the server reports the exact position), the phase (prompt processing with progress, thinking, writing, tool call), tokens and tok/s, and elapsed time; after every turn: prompt tok/s, gen tok/s, MTP acceptance, context used/total (colored; warns at 80%); the terminal bell rings when a reply finishes or a tool call waits for approval (iTerm2, Ghostty, WezTerm and kitty also get a desktop notification — a fixed phrase, never content) |
| `lethe conversations [--unlock] [--delete ID]` | encrypted saved conversations |
| `lethe bench [spec] [-d 0,32768]` | `llama-bench` prompt/gen speeds at context depth — **no speculative decoding, this is the floor** |
| `lethe bench [spec] --server [-d 0,32768]` | through `llama-server`: no-MTP vs embedded MTP vs FastMTP, real tok/s and acceptance |
| `lethe traces [--delete]` | list/remove anything written to disk beyond model files |
| `lethe doctor` | GPU build, memory budget, what's in use, what's on disk |

In the REPL: `/file PATH` and `/image PATH` (with `--vision`) attach to the next message; `/compact`
summarizes the conversation with the model and continues from the summary; `/think on|off`; `/copy`;
`/wipe`; `/save`; `/help`. Enter sends; Option+Enter, Ctrl+J, Shift+Enter, Ctrl+Enter or a trailing `\`
add a line; multi-line pastes arrive as one message. Shift+Enter and Ctrl+Enter need a terminal that reports
them — the kitty keyboard protocol (Ghostty, kitty, WezTerm, iTerm2 3.5+; lethe turns it on only while you
type a line) or xterm's modifyOtherKeys. **Apple Terminal sends both as a plain Enter** and offers no way
to change that (its key mappings cannot bind Return): use Option+Enter after enabling "Use Option as Meta
Key" (Settings → Profiles → Keyboard), or Ctrl+J, or remap Shift+Return to Ctrl+J with Karabiner-Elements.

There is no background server. Each `lethe` run starts its own `llama-server` on a free loopback
port with a random key and kills it on exit, so nothing holds the model's memory between chats.
With the model file in the OS cache that costs about three seconds per launch.

## How the memory columns are computed

`file size + KV cache`, where KV per token = `attention layers × KV heads × (K dim + V dim) × 2 bytes` (f16).
The header of each GGUF is read directly (only the first ~16 MiB, over HTTP for remote files), so hybrid
models such as Qwen3.5/3.8 (Gated DeltaNet + 1-in-4 attention layers) are costed correctly — their long
context is far cheaper than a dense model of the same size. Capacity is what Metal reports as
`recommendedMaxWorkingSetSize` minus 4 GiB of headroom; on Linux, the selected backend’s
reported GPU limit/free memory is capped by available host RAM and reserves 4 GiB. On macOS, "free now" subtracts memory already wired by
other processes (e.g. another llama-server). Real usage adds 1–3 GiB of compute buffers.

## Reference numbers — Ryzen AI Max+ 395, 128 GB, Qwen3.8-27B Q8_K_P

Measured 2026-10-15 on Radeon 8060S (`gfx1151`), Linux 7.2.8, Mesa/RADV 26.2.4,
ROCm packages 7.2.4, upstream `620d2225c` plus `scripts/fastmtp.patch`.
Both backends successfully allocate 262,144 context with F16 KV and generate with FastMTP
and thinking enabled. The model + draft + KV estimate is 46.1 GiB, excluding compute buffers.

Server benchmarks below allocate the full 262,144 context, with initially empty context,
greedy sampling, thinking off, and 256-token replies over the same prose/reasoning/code prompts.
Generation rates aggregate tokens divided by generation time; prefill is an uncached 2,669-token prompt.

| Backend | No MTP gen | Embedded MTP d3 gen | FastMTP d3 gen | FastMTP prefill | FastMTP acceptance |
|---|---:|---:|---:|---:|---:|
| ROCm/HIP | 7.2 tok/s | 15.8 tok/s | 18.3 tok/s | 359 tok/s | 62% |
| Vulkan/RADV | 7.3 tok/s | 15.7 tok/s | 17.9 tok/s | 355 tok/s | 62% |

With the same repeated-text prefix targeting 32K (actual prefill **42,605 tokens**),
FastMTP d3 measured:

| Backend | Long prefill | Generation after prefix | Acceptance |
|---|---:|---:|---:|
| ROCm/HIP | 288.6 tok/s | 16.29 tok/s | 62% |
| Vulkan/RADV | 268.3 tok/s | 15.81 tok/s | 64% |

ROCm was about 2–3% faster at FastMTP generation and 8% faster at the long prefill in this run.
The small generation difference is not a robust backend ranking; this is one suite per backend.
ROCm/HIP is the Linux default based on these measurements; bare `lethe` uses it.
Select Vulkan for a particular invocation with `LETHE_BACKEND=vulkan lethe`.
The full 262K context was allocated, but these throughput tests did not fill all 262K tokens.
Actual thinking-chat throughput varies with the prompt and draft acceptance.
The separate ROCm `llama-bench` baseline (3 repetitions) was pp2048 379.4 ± 0.7 tok/s,
tg128 7.21 ± 0.01 tok/s. No system power or memory-limit settings were changed for these tests.

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
src/lethe/               cli.py · repl.py · terminal.py · server.py · chat.py · spec.py · profiles.py · vault.py ·
                         tools.py · shell.py · files.py · instructions.py · attachments.py · benchserver.py · hub.py · ggufmeta.py · sysinfo.py · config.py · paths.py
tests/                   pytest (uv run pytest)
vendor/llama.cpp         submodule; the FastMTP patch is committed on local branch `lethe-fastmtp`
models/                  downloaded GGUFs (gitignored)
.lethe/                  empty unless you opted in to --save (gitignored)
```
