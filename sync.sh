#!/usr/bin/env bash
# Push the local hyper tree to the build host (~/hyper/) and optionally run a command there.
# usage: ./sync.sh [remote command...]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SCRATCH="${SCRATCH:-/tmp/claude-1000/-home-user-Projects-ik-llama-claude/012b093d-ec42-4e05-9fdd-2741488ed2b8/scratchpad}"
tar -C "$HERE" --exclude=build --exclude=.git -czf "$SCRATCH/hyper.tgz" .
"$SCRATCH/venv/bin/python" "$SCRATCH/up.py" "$SCRATCH/hyper.tgz" /home/user/hyper/.upload.tgz > /dev/null
CMD="cd ~/hyper && tar -xzmf .upload.tgz && rm -f .upload.tgz"
if [ $# -gt 0 ]; then CMD="$CMD && $*"; fi
timeout "${SYNC_TIMEOUT:-1500}" "$SCRATCH/venv/bin/python" "$SCRATCH/r.py" "$CMD"
