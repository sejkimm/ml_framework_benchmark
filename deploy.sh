#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE' >&2
Usage: deploy.sh user@host /remote/path [local_path]

Required:
  user@host     SSH target (e.g. ubuntu@1.2.3.4)
  /remote/path  Remote directory to sync into

Optional:
  local_path    Local directory to deploy (default: current directory)

Env vars:
  DEPLOY_SSH_OPTS   Extra SSH options (e.g. "-i ~/.ssh/id_ed25519 -p 2222")
  DEPLOY_DELETE=1   Delete remote files not present locally
USAGE
}

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

TARGET=$1
REMOTE_DIR=$2
LOCAL_DIR=${3:-"$(pwd)"}
SSH_OPTS=${DEPLOY_SSH_OPTS:-}

if [[ ! -d "$LOCAL_DIR" ]]; then
  echo "Local path not found: $LOCAL_DIR" >&2
  exit 1
fi

RSYNC_ARGS=(
  -az
  --progress
  --exclude .git
  --exclude .venv
  --exclude __pycache__
  --exclude '*.pyc'
  --exclude .DS_Store
  -e "ssh $SSH_OPTS"
)

if [[ "${DEPLOY_DELETE:-0}" == "1" ]]; then
  RSYNC_ARGS+=(--delete)
fi

ssh $SSH_OPTS "$TARGET" "mkdir -p '$REMOTE_DIR'"

rsync "${RSYNC_ARGS[@]}" "$LOCAL_DIR/" "$TARGET:$REMOTE_DIR/"
