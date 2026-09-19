# Audit: model tools and sandboxed commands

Reviewed the Python CLI, tool registry, streaming chat loop, process launcher, file attachments, and
encrypted conversation storage. Existing uncommitted changes were preserved. This is a focused code
and integration review, not a penetration test of llama.cpp, third-party dependencies, or macOS.

## Findings addressed

| Finding | Change |
|---|---|
| Tool calls could be dispatched after a truncated SSE response, without a completed finish event. This is especially risky for executable commands. | Require a completed stream and a compatible finish reason before emitting accumulated calls. |
| Call previews were truncated at 160 characters and rendered with Rich markup; the user could approve a request without seeing its complete arguments. Non-object JSON also crashed the preview. | Validate arguments as an object before dispatch. Print complete escaped arguments with markup disabled. Shell commands always require explicit confirmation. Escape control characters in streamed/resumed model text so it cannot restyle the terminal around approval prompts. |
| The eight-round guard withdrew tools but could loop forever if a model kept requesting them; one round could contain arbitrarily many executable calls. | Hard limits of eight execution rounds and sixteen calls, followed by at most one final model response. |
| `/web` and `/web off` cleared the whole registry. Shell tools would disappear when toggling the web. | Add/remove internet tools by name, preserving shell state. `--web` now composes with `--tools shell`. |
| Resumed conversations could retain stale tool/privacy instructions. | Reconcile the built-in system prompt with currently enabled tools on resume and after toggles. Custom prompts are preserved. |
| `fetch_url` validated addresses from `getaddrinfo()` but let curl do an independent lookup, so a rebinding hostname could pass the check and then connect to a private address. | Every hop is pinned with `curl --resolve` to the address that was validated; the hostname is kept for TLS/Host. Proxy variables are still honoured (through a proxy the name resolves on the proxy's side, not against this machine's network). |

## Open findings

1. **Medium — vault IDs are not constrained to generated IDs.** `vault.path_for()` appends `.lethe`
   to the supplied ID without validating separators or canonical containment. `--resume` and
   `conversations --delete` can therefore address `.lethe` files outside the vault. The caller must
   supply the path, so this is a local input-safety issue, not a demonstrated remote exploit.
   Validate IDs against the eight-hex-character generated format and reject symlink escapes before
   reading, replacing, or deleting a file.
2. **Medium — model startup has no deadline.** `server.start()` polls indefinitely while the child
   stays alive but never becomes healthy. A startup deadline should stop only the process it spawned
   and report an actionable error. No model server was restarted during this audit.

## Shell boundary and verification

The new `shell.ShellSandbox` denies operations by default and explicitly permits reading the selected
canonical directory and a fixed macOS runtime. Network, filesystem mutations, host process inspection,
and host process signalling receive no grants. Path parameters are passed separately from the policy
source. Neither the model nor the tool schema can request a broader sandbox, writes, environment
inheritance, or an unsandboxed retry.

Commands are sent through stdin; policy and output are held in memory. Confirmation shows complete
escaped arguments and the workspace. Time and output limits stop a process group; inherited hard
CPU/file-size/file-descriptor limits provide additional bounds. Memory, total process count, and
deliberately detached descendants are not fully isolated. Read-only Seatbelt is an initial local
inspection tool, not a hostile-code execution service. See README for the supported scope.

Validation includes the existing suite, mocked model → confirmation → tool → model conversations,
CLI composition/failure behavior, incomplete streamed tool calls, and real macOS Seatbelt probes for
allowed reads/pipelines, denied writes/outside reads/symlinks/network/host-process access, timeouts,
and bounded output. Real sandbox tests require `--run-sandbox`; an unavailable sandbox fails these
tests rather than silently skipping enforcement checks.

Implementation references: the installed Apple `sandbox-exec(1)` manual and system sandbox profiles,
plus the current [Codex Seatbelt base policy](https://github.com/openai/codex/blob/main/codex-rs/sandboxing/src/seatbelt_base_policy.sbpl)
for inherited process restrictions. No additional dependency was introduced.
