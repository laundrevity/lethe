#!/usr/bin/env bash
# Canonical llama.cpp build for Apple Silicon (M5 Max). Run from anywhere.
#
#   scripts/build-llama.sh            # configure (if needed) + build
#   scripts/build-llama.sh --clean    # wipe build/ first
#   scripts/build-llama.sh --no-patch # skip the HauhauCS FastMTP patch
#
# Why these flags (see CLAUDE.md):
#   GGML_METAL=ON               GPU via Metal — the whole point on Apple Silicon
#   GGML_METAL_EMBED_LIBRARY=ON bake the .metal shaders into the binary; no runtime file lookup
#   GGML_ACCELERATE=ON          Apple Accelerate for the CPU-side BLAS bits
#   GGML_NATIVE=ON              -mcpu=native for this exact chip
#   LLAMA_CURL=OFF              we download with huggingface_hub, not llama-server
#   MTMD_VIDEO=OFF              native FFmpeg cache: writes temp files; lethe decodes frames in RAM
#   CMAKE_BUILD_RPATH=@loader_path  relative rpath so the build survives moving/renaming the repo
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/vendor/llama.cpp"
BUILD="$SRC/build"
JOBS="$(sysctl -n hw.ncpu)"
CLEAN=0; PATCH=1
for a in "$@"; do
  case "$a" in
    --clean) CLEAN=1 ;;
    --no-patch) PATCH=0 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

if [ ! -f "$SRC/CMakeLists.txt" ]; then
  echo "vendor/llama.cpp is empty — run: git submodule update --init" >&2
  exit 1
fi
if [ "$(uname -s)/$(uname -m)" != "Darwin/arm64" ]; then
  echo "this script is for Apple Silicon macOS only" >&2
  exit 1
fi

# --- optional model-specific patch (HauhauCS FastMTP draft-vocab trim) -------------------
# Applied as a commit on a local branch so the submodule stays clean and the build is recorded.
PATCH_FILE="$ROOT/patches/HauhauCS-FastMTP-llama.cpp.patch"
if [ "$PATCH" = 1 ] && [ -f "$PATCH_FILE" ]; then
  (
    cd "$SRC"
    if git apply --check -R "$PATCH_FILE" >/dev/null 2>&1; then
      echo "[patch] FastMTP patch already applied"
    elif git apply --check "$PATCH_FILE" >/dev/null 2>&1; then
      git switch -c lethe-fastmtp 2>/dev/null || git switch lethe-fastmtp
      git apply "$PATCH_FILE"
      git -c user.name=lethe -c user.email=lethe@localhost commit -qam "Apply HauhauCS FastMTP patch"
      echo "[patch] applied FastMTP patch on branch lethe-fastmtp"
    else
      echo "[patch] WARNING: FastMTP patch does not apply to $(git rev-parse --short HEAD); building unpatched" >&2
    fi
  )
fi

# --- configure ---------------------------------------------------------------------------
[ "$CLEAN" = 1 ] && rm -rf "$BUILD"
cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_BUILD_RPATH=@loader_path \
  -DGGML_METAL=ON \
  -DGGML_METAL_EMBED_LIBRARY=ON \
  -DGGML_ACCELERATE=ON \
  -DGGML_NATIVE=ON \
  -DLLAMA_CURL=OFF \
  -DMTMD_VIDEO=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_SERVER=ON

# --- build only what we use ---------------------------------------------------------------
cmake --build "$BUILD" --config Release -j"$JOBS" --target llama-server llama-cli llama-bench

echo
echo "built: $(git -C "$SRC" rev-parse --short HEAD) ($(git -C "$SRC" branch --show-current))"
ls -la "$BUILD/bin/llama-server" "$BUILD/bin/llama-cli" "$BUILD/bin/llama-bench"
echo "verify Metal with: uv run lethe doctor"
