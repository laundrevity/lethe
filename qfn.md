All set — the model's already registered. To run it:

```sh
lethe chat orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF:IQ4_XS
```

or make it your default so bare `lethe` uses it:

```sh
lethe default orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF:IQ4_XS
```

Useful extras from `chat --help`:

- `--vision` — loads the 866 MiB mmproj so `/image` works
- `--ctx 32768` (etc.) — cap context; default 0 = native max (262,144) if it fits
- `--web` — enable web/fetch tools

