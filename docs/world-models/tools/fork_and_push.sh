#!/usr/bin/env bash
#
# Fork every reviewed world-model repo into an org, then push this directory's
# review and ease-of-use files into each fork on its own branch.
#
# Run this where `gh auth status` shows you as a member of the target org with
# permission to create repositories. It is safe to re-run: repos already forked
# are skipped, and pushes are idempotent.
#
#   docs/world-models/tools/fork_and_push.sh              # fork + push all 33
#   docs/world-models/tools/fork_and_push.sh --dry-run    # print, change nothing
#   docs/world-models/tools/fork_and_push.sh Vista opendw # just these
#
# Env: FORK_ORG (default roboticarray), BRANCH (default roboticarray),
#      WORK (default a temp dir), SKIP_FORK=1, SKIP_PUSH=1
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORK_ORG="${FORK_ORG:-roboticarray}"
BRANCH="${BRANCH:-roboticarray}"
WORK="${WORK:-$(mktemp -d -t wm-forks-XXXXXX)}"
DRY_RUN=0

args=()
for a in "$@"; do
  case "$a" in
    --dry-run|-n) DRY_RUN=1 ;;
    -h|--help) sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) args+=("$a") ;;
  esac
done

command -v gh  >/dev/null || { echo "error: gh CLI not found"; exit 1; }
command -v git >/dev/null || { echo "error: git not found"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "error: not logged in; run 'gh auth login'"; exit 1; }

# Build "upstream<TAB>dir" pairs from each review's scores.json.
mapfile -t PAIRS < <(
  for f in "$HERE"/*/scores.json; do
    python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["repo"]+"\t"+d["dir"])' "$f"
  done | sort
)

if [ "${#args[@]}" -gt 0 ]; then
  filtered=()
  for want in "${args[@]}"; do
    hit=""
    for p in "${PAIRS[@]}"; do [ "${p#*$'\t'}" = "$want" ] && hit="$p"; done
    [ -n "$hit" ] && filtered+=("$hit") || echo "warn: no review directory named '$want', skipping"
  done
  PAIRS=("${filtered[@]}")
fi

echo "org=$FORK_ORG branch=$BRANCH repos=${#PAIRS[@]} work=$WORK$([ $DRY_RUN = 1 ] && echo ' (dry run)')"
echo

forked=0 pushed=0 skipped=0 failed=()

for pair in "${PAIRS[@]}"; do
  upstream="${pair%$'\t'*}"
  name="${pair#*$'\t'}"
  src="$HERE/$name"
  fork="$FORK_ORG/$name"
  echo "== $upstream -> $fork"

  # --- fork -------------------------------------------------------------
  if [ "${SKIP_FORK:-0}" != "1" ]; then
    if gh repo view "$fork" >/dev/null 2>&1; then
      echo "   fork exists"
    elif [ $DRY_RUN = 1 ]; then
      echo "   would fork"
    else
      if gh repo fork "$upstream" --org "$FORK_ORG" --clone=false --default-branch-only >/dev/null 2>&1; then
        # Forking is asynchronous; wait for the repo to become readable.
        for _ in $(seq 1 30); do
          gh repo view "$fork" >/dev/null 2>&1 && break
          sleep 2
        done
        if gh repo view "$fork" >/dev/null 2>&1; then
          echo "   forked"; forked=$((forked+1))
        else
          echo "   FAILED: fork not visible after 60s"; failed+=("$name (fork timeout)"); continue
        fi
      else
        echo "   FAILED: gh repo fork"; failed+=("$name (fork)"); continue
      fi
    fi
  fi

  # --- push -------------------------------------------------------------
  [ "${SKIP_PUSH:-0}" = "1" ] && continue
  if [ $DRY_RUN = 1 ]; then
    echo "   would push $(ls "$src" | grep -v scores.json | tr '\n' ' ')to $BRANCH"
    continue
  fi

  dst="$WORK/$name"
  if [ ! -d "$dst/.git" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone --quiet --depth 1 "https://github.com/$fork.git" "$dst" 2>/dev/null \
      || { echo "   FAILED: clone"; failed+=("$name (clone)"); continue; }
  fi

  default="$(git -C "$dst" symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null | sed 's|origin/||' || echo main)"
  if git -C "$dst" ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
    git -C "$dst" fetch --quiet --depth 1 origin "$BRANCH" && git -C "$dst" checkout --quiet -B "$BRANCH" FETCH_HEAD
  else
    git -C "$dst" checkout --quiet -B "$BRANCH" "origin/$default"
  fi

  cp "$src/REVIEW.md" "$src/CLAUDE.md" "$src/QUICKSTART.md" "$dst/"
  [ -f "$src/DOCKERFILE_NOTES.md" ] && cp "$src/DOCKERFILE_NOTES.md" "$dst/"
  # Never clobber an upstream Dockerfile; ours lands beside it.
  if [ -f "$src/Dockerfile" ]; then
    if [ -f "$dst/Dockerfile" ]; then cp "$src/Dockerfile" "$dst/Dockerfile.roboticarray"
    else cp "$src/Dockerfile" "$dst/Dockerfile"; fi
  fi

  git -C "$dst" add REVIEW.md CLAUDE.md QUICKSTART.md DOCKERFILE_NOTES.md Dockerfile Dockerfile.roboticarray 2>/dev/null || true
  if git -C "$dst" diff --cached --quiet; then
    echo "   already up to date"; skipped=$((skipped+1)); continue
  fi

  git -C "$dst" commit --quiet -m "Add roboticarray review and ease-of-use files

REVIEW.md scores this repo on usefulness, code quality, security, license,
extensibility, physics, applications and motion-control value. CLAUDE.md,
QUICKSTART.md and the Dockerfile make it runnable quickly.

Upstream code is unmodified, so 'git fetch upstream && git merge' stays clean."

  if git -C "$dst" push --quiet -u origin "$BRANCH" 2>/dev/null; then
    echo "   pushed $BRANCH"; pushed=$((pushed+1))
  else
    echo "   FAILED: push"; failed+=("$name (push)")
  fi
done

echo
echo "forked=$forked pushed=$pushed unchanged=$skipped failed=${#failed[@]}"
if [ "${#failed[@]}" -gt 0 ]; then
  printf '  %s\n' "${failed[@]}"
  echo "Re-run the script to retry only what failed."
  exit 1
fi
echo "Review branches are at https://github.com/$FORK_ORG?q=&type=fork"
