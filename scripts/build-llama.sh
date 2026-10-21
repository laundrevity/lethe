#!/usr/bin/env bash
# Canonical llama.cpp build: Metal on macOS; Vulkan or ROCm/HIP on Linux.
#
#   scripts/build-llama.sh            # configure (if needed) + build
#   scripts/build-llama.sh --backend=vulkan # optional Vulkan (default: ROCm on Linux, Metal on macOS)
#   scripts/build-llama.sh --clean    # wipe build/ first
#   scripts/build-llama.sh --no-patch # skip the HauhauCS FastMTP patch
#
# Why these flags (see AGENTS.md):
#   GGML_METAL=ON               GPU via Metal — the whole point on Apple Silicon
#   GGML_METAL_EMBED_LIBRARY=ON bake the .metal shaders into the binary; no runtime file lookup
#   GGML_ACCELERATE=ON          Apple Accelerate for the CPU-side BLAS bits
#   GGML_NATIVE=ON              -mcpu=native for this exact chip
#   LLAMA_CURL=OFF              we download with huggingface_hub, not llama-server
#   CMAKE_BUILD_RPATH=@loader_path  relative rpath so the build survives moving/renaming the repo
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/vendor/llama.cpp"
BUILD="$SRC/build"
if [ "$(uname -s)" = Darwin ]; then
  JOBS="${JOBS:-$(sysctl -n hw.ncpu)}"
  BACKEND="${LETHE_BACKEND:-metal}"
else
  JOBS="${JOBS:-$(nproc)}"
  BACKEND="${LETHE_BACKEND:-hip}"
fi
CLEAN=0; PATCH=1
for a in "$@"; do
  case "$a" in
    --backend=*) BACKEND="${a#*=}" ;;
    --clean) CLEAN=1 ;;
    --no-patch) PATCH=0 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

if [ ! -f "$SRC/CMakeLists.txt" ]; then
  echo "vendor/llama.cpp is empty — run: git submodule update --init" >&2
  exit 1
fi
case "$BACKEND" in
  metal)
    [ "$(uname -s)/$(uname -m)" = Darwin/arm64 ] || { echo "Metal requires Apple Silicon" >&2; exit 1; }
    FLAGS=(-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_ACCELERATE=ON -DCMAKE_BUILD_RPATH=@loader_path)
    ;;
  vulkan)
    BUILD="$SRC/build-vulkan"
    if [ -d "$ROOT/.deps/include" ]; then
      export CPLUS_INCLUDE_PATH="$ROOT/.deps/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
    fi
    FLAGS=("-DCMAKE_PREFIX_PATH=$ROOT/.deps" -DGGML_METAL=OFF -DGGML_ACCELERATE=OFF -DGGML_VULKAN=ON -DGGML_HIP=OFF '-DCMAKE_BUILD_RPATH=$ORIGIN' -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON)
    ;;
  hip)
    BUILD="$SRC/build-hip"
    FLAGS=(-DGGML_METAL=OFF -DGGML_ACCELERATE=OFF -DGGML_VULKAN=OFF -DGGML_HIP=ON
      "-DCMAKE_HIP_ARCHITECTURES=${AMDGPU_TARGETS:-gfx1151}" "-DCMAKE_HIP_COMPILER=${ROCM_PATH:-/opt/rocm}/llvm/bin/clang++"
      "-DCMAKE_PREFIX_PATH=${ROCM_PATH:-/opt/rocm}" '-DCMAKE_BUILD_RPATH=$ORIGIN' -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON)
    ;;
  *) echo "unknown backend: $BACKEND (metal|vulkan|hip)" >&2; exit 2 ;;
esac

# --- optional model-specific patch (HauhauCS FastMTP draft-vocab trim) -------------------
# Applied as a commit on a local branch so the submodule stays clean and the build is recorded.
PATCH_FILE="$(ls "$ROOT"/models/*/HauhauCS-FastMTP-llama.cpp.patch 2>/dev/null | head -1 || true)"
# The checked-in patch is rebased for our upstream pin.
[ ! -f "$ROOT/scripts/fastmtp.patch" ] || PATCH_FILE="$ROOT/scripts/fastmtp.patch"
if [ "$PATCH" = 1 ] && [ -n "$PATCH_FILE" ]; then
  (
    cd "$SRC"
    if git apply --check -R "$PATCH_FILE" >/dev/null 2>&1; then
      echo "[patch] FastMTP patch already applied"
    elif git apply --check "$PATCH_FILE" >/dev/null 2>&1; then
      if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
        echo "[patch] submodule has local changes; commit or stash them first" >&2
        exit 1
      fi
      git switch -C lethe-fastmtp HEAD
      git apply "$PATCH_FILE"
      git -c user.name=lethe -c user.email=lethe@localhost commit -qam "Apply HauhauCS FastMTP patch"
      echo "[patch] applied FastMTP patch on branch lethe-fastmtp"
    else
      echo "[patch] FastMTP patch does not apply to $(git rev-parse --short HEAD); rebase it before building" >&2
      exit 1
    fi
  )
fi

# --- configure ---------------------------------------------------------------------------
[ "$CLEAN" = 1 ] && rm -rf "$BUILD"
cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  "${FLAGS[@]}" \
  -DGGML_NATIVE=ON \
  -DLLAMA_CURL=OFF \
  -DLLAMA_OPENSSL=OFF \
  -DLLAMA_BUILD_UI=OFF \
  -DLLAMA_USE_PREBUILT_UI=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_SERVER=ON

# --- build only what we use ---------------------------------------------------------------
cmake --build "$BUILD" --config Release -j"$JOBS" --target llama-server llama-cli llama-bench

echo
echo "built: $(git -C "$SRC" rev-parse --short HEAD) ($(git -C "$SRC" branch --show-current))"
ls -la "$BUILD/bin/llama-server" "$BUILD/bin/llama-cli" "$BUILD/bin/llama-bench"
echo "verify with: LETHE_BACKEND=$BACKEND uv run lethe doctor"
