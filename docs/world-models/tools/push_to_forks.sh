#!/usr/bin/env bash
# Push the per-repo review and ease-of-use files into each roboticarray fork.
#
# Prereqs: forks exist at github.com/roboticarray/<name> (see FORKING.md) and you can push to them.
# Usage:   docs/world-models/tools/push_to_forks.sh [name ...]     (default: every dir with a REVIEW.md)
# Env:     FORK_ORG (default roboticarray), BRANCH (default roboticarray), WORK (default /tmp/wm-forks), DRY_RUN=1
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK_ORG="${FORK_ORG:-roboticarray}"
BRANCH="${BRANCH:-roboticarray}"
WORK="${WORK:-/tmp/wm-forks}"
mkdir -p "$WORK"

if [ "$#" -gt 0 ]; then names=("$@"); else
  mapfile -t names < <(cd "$HERE" && for d in */; do [ -f "$d/REVIEW.md" ] && echo "${d%/}"; done)
fi

for name in "${names[@]}"; do
  src="$HERE/$name"
  url="https://github.com/$FORK_ORG/$name.git"
  dst="$WORK/$name"
  echo "== $name -> $url ($BRANCH)"
  if [ ! -d "$dst/.git" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 "$url" "$dst"
  fi
  git -C "$dst" fetch --depth 1 origin
  default="$(git -C "$dst" symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null | sed 's|origin/||' || echo main)"
  if git -C "$dst" ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
    git -C "$dst" fetch --depth 1 origin "$BRANCH" && git -C "$dst" checkout -B "$BRANCH" FETCH_HEAD
  else
    git -C "$dst" checkout -B "$BRANCH" "origin/$default"
  fi
  # Copy review + ease-of-use files. Never clobber an upstream Dockerfile: ours becomes Dockerfile.roboticarray.
  cp "$src/REVIEW.md" "$dst/REVIEW.md"
  cp "$src/CLAUDE.md" "$dst/CLAUDE.md"
  cp "$src/QUICKSTART.md" "$dst/QUICKSTART.md"
  [ -f "$src/DOCKERFILE_NOTES.md" ] && cp "$src/DOCKERFILE_NOTES.md" "$dst/DOCKERFILE_NOTES.md"
  if [ -f "$src/Dockerfile" ]; then
    if [ -f "$dst/Dockerfile" ]; then cp "$src/Dockerfile" "$dst/Dockerfile.roboticarray"; else cp "$src/Dockerfile" "$dst/Dockerfile"; fi
  fi
  git -C "$dst" add REVIEW.md CLAUDE.md QUICKSTART.md DOCKERFILE_NOTES.md Dockerfile Dockerfile.roboticarray 2>/dev/null || true
  if git -C "$dst" diff --cached --quiet; then echo "   nothing to commit"; continue; fi
  git -C "$dst" commit -q -m "Add roboticarray review and ease-of-use files

REVIEW.md scores the repo on usefulness, code quality, security, license,
extensibility, physics, applications and motion-control value. CLAUDE.md,
QUICKSTART.md and the Dockerfile make the repo runnable quickly."
  if [ "${DRY_RUN:-0}" = "1" ]; then echo "   DRY_RUN: would push $BRANCH"; else git -C "$dst" push -u origin "$BRANCH"; fi
done
