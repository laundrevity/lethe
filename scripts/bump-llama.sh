#!/usr/bin/env bash
# Move vendor/llama.cpp to the latest upstream master, re-apply the FastMTP patch, rebuild, verify.
#
#   scripts/bump-llama.sh              # bump to origin/master
#   scripts/bump-llama.sh <commit>     # bump to a specific upstream commit
#
# The superproject records the *upstream* commit as the submodule pin; the patch lives as one extra
# commit on the local branch `lethe-fastmtp` (see CLAUDE.md). This script keeps that arrangement.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/vendor/llama.cpp"
PATCH_FILE="$ROOT/patches/HauhauCS-FastMTP-llama.cpp.patch"
TARGET="${1:-origin/master}"

cd "$SRC"
OLD="$(git rev-parse --short HEAD)"
git fetch -q origin master
NEW="$(git rev-parse --short "$TARGET")"
echo "[bump] $OLD -> $NEW ($(git log -1 --format=%cd --date=short "$TARGET"))"
echo "[bump] $(git rev-list --count "$OLD".."$TARGET" 2>/dev/null || echo '?') upstream commits"

if [ -f "$PATCH_FILE" ]; then
  # can the patch apply on the new base?
  if git diff --quiet "$TARGET" -- src/models/qwen35.cpp 2>/dev/null; then :; fi
  git stash -q 2>/dev/null || true
  git checkout -q --detach "$TARGET"
  if ! git apply --check "$PATCH_FILE" 2>/dev/null; then
    echo "[bump] FastMTP patch does NOT apply on $NEW — staying on $OLD" >&2
    git checkout -q lethe-fastmtp
    exit 1
  fi
  git checkout -q -B lethe-fastmtp "$TARGET"
  git apply "$PATCH_FILE"
  git -c user.name=lethe -c user.email=lethe@localhost commit -qam "Apply HauhauCS FastMTP patch"
  echo "[bump] patch re-applied on lethe-fastmtp"
else
  git checkout -q -B lethe-fastmtp "$TARGET"
fi

# record the upstream commit (not the patch commit) as the pin in the superproject
cd "$ROOT"
git update-index --cacheinfo "160000,$(git -C "$SRC" rev-parse "$TARGET"),vendor/llama.cpp"
echo "[bump] superproject pin staged: $(git -C "$SRC" rev-parse --short "$TARGET")"

"$ROOT/scripts/build-llama.sh"
uv run lethe doctor
echo
echo "[bump] done. Update the pin note in CLAUDE.md and commit: git commit -m 'Bump llama.cpp to $NEW'"
