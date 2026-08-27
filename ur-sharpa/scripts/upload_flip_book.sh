#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE="${UPLOAD_SOURCE:-/media/yns/T9/0825_yns_flip_book/success}"
REMOTE_HOST="${UPLOAD_HOST:-BAAI}"
REMOTE_DIR="${UPLOAD_DESTINATION:-/share/project/yenaisheng/sharpa-deploy/datasets/postrain/ur-sharpa/raw_data/flip_book}"
JOBS="${UPLOAD_JOBS:-4}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Upload completed flip_book episodes with parallel, resumable rsync transfers.

Usage:
  ./scripts/upload_flip_book.sh [options]

Options:
  --source PATH        Local directory containing episode directories
  --host HOST          SSH host or alias (default: BAAI)
  --destination PATH   Absolute remote destination directory
  --jobs N             Parallel rsync connections (default: 4)
  --dry-run            Run local preflight checks without changing the server
  -h, --help           Show this help

The script uploads into a hidden staging directory, verifies each episode with
its checksums.sha256 on the server, and atomically renames it into place. A
rerun resumes partial files and skips already finalized episodes.
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

while (($#)); do
  case "$1" in
    --source)
      (($# >= 2)) || die "--source requires a value"
      SOURCE=$2
      shift 2
      ;;
    --host)
      (($# >= 2)) || die "--host requires a value"
      REMOTE_HOST=$2
      shift 2
      ;;
    --destination)
      (($# >= 2)) || die "--destination requires a value"
      REMOTE_DIR=$2
      shift 2
      ;;
    --jobs)
      (($# >= 2)) || die "--jobs requires a value"
      JOBS=$2
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

[[ -d "$SOURCE" ]] || die "source directory does not exist: $SOURCE"
[[ "$REMOTE_HOST" =~ ^[A-Za-z0-9_.@-]+$ ]] || die "unsafe SSH host: $REMOTE_HOST"
[[ "$REMOTE_DIR" =~ ^/[A-Za-z0-9._/-]+$ ]] || die "remote destination must be a simple absolute path"
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer"
((JOBS <= 16)) || die "--jobs must not exceed 16"

SOURCE=${SOURCE%/}
REMOTE_DIR=${REMOTE_DIR%/}
REMOTE_PARTIAL="$REMOTE_DIR/.upload-partial"

for command_name in find python3 rsync sha256sum sort ssh; do
  command -v "$command_name" >/dev/null || die "required command is missing: $command_name"
done

SSH_OPTIONS=(
  -T
  -o BatchMode=yes
  -o Compression=no
  -o ConnectTimeout=15
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=6
)
RSYNC_RSH="ssh -T -o BatchMode=yes -o Compression=no -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=6"

mapfile -d '' -t EPISODE_PATHS < <(
  find "$SOURCE" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z
)
((${#EPISODE_PATHS[@]} > 0)) || die "no episode directories found in $SOURCE"

EPISODE_NAMES=()
LOCAL_BYTES=0
LOCAL_FILES=0
for episode_path in "${EPISODE_PATHS[@]}"; do
  episode_name=${episode_path##*/}
  [[ "$episode_name" =~ ^[A-Za-z0-9._-]+$ ]] || die "unsafe episode name: $episode_name"
  [[ -f "$episode_path/COMPLETE" ]] || die "$episode_name is missing COMPLETE"
  [[ -f "$episode_path/manifest.json" ]] || die "$episode_name is missing manifest.json"
  [[ -f "$episode_path/checksums.sha256" ]] || die "$episode_name is missing checksums.sha256"
  awk '
    index($0, "  ") == 0 { exit 1 }
    {
      path = substr($0, index($0, "  ") + 2)
      if (path ~ /^\// || path ~ /(^|\/)\.\.($|\/)/) exit 1
    }
  ' "$episode_path/checksums.sha256" || die "$episode_name has an unsafe checksum list"
  EPISODE_NAMES+=("$episode_name")
done

python3 - "$SOURCE" "${EPISODE_NAMES[@]}" <<'PY' || die "one or more manifests failed validation"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
failed = False
for episode in sys.argv[2:]:
    try:
        manifest = json.loads((root / episode / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("state") != "complete" or manifest.get("quality", {}).get("ok") is not True:
            raise ValueError("state is not complete or quality.ok is not true")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"invalid manifest for {episode}: {exc}", file=sys.stderr)
        failed = True
if failed:
    raise SystemExit(1)
PY

read -r LOCAL_FILES LOCAL_BYTES < <(
  find "$SOURCE" -type f -printf '%s\n' |
    awk '{ files += 1; bytes += $1 } END { printf "%d %.0f\n", files, bytes }'
)

echo "Source:      $SOURCE"
echo "Destination: $REMOTE_HOST:$REMOTE_DIR"
echo "Episodes:    ${#EPISODE_NAMES[@]}"
echo "Files:       $LOCAL_FILES"
echo "Bytes:       $LOCAL_BYTES"
echo "Jobs:        $JOBS"

if ((DRY_RUN)); then
  echo "Dry run complete; the server was not changed."
  exit 0
fi

ssh "${SSH_OPTIONS[@]}" "$REMOTE_HOST" \
  "set -eu; command -v rsync >/dev/null; command -v sha256sum >/dev/null; mkdir -p -- '$REMOTE_DIR' '$REMOTE_PARTIAL'"

declare -A REMOTE_FINAL=()
REMOTE_FINAL_OUTPUT=$(
  ssh "${SSH_OPTIONS[@]}" "$REMOTE_HOST" \
    "find '$REMOTE_DIR' -mindepth 1 -maxdepth 1 -type d ! -name '.upload-partial' -printf '%f\\n'"
) || die "could not list finalized episodes on $REMOTE_HOST"
while IFS= read -r episode_name; do
  [[ -n "$episode_name" ]] && REMOTE_FINAL["$episode_name"]=1
done <<<"$REMOTE_FINAL_OUTPUT"

PENDING_NAMES=()
for episode_name in "${EPISODE_NAMES[@]}"; do
  if [[ -n "${REMOTE_FINAL[$episode_name]:-}" ]]; then
    echo "Already finalized, skipping: $episode_name"
  else
    PENDING_NAMES+=("$episode_name")
  fi
done

if ((${#PENDING_NAMES[@]} == 0)); then
  echo "Nothing to upload; every local episode is already finalized."
  exit 0
fi

WORKER_COUNT=$JOBS
if ((WORKER_COUNT > ${#PENDING_NAMES[@]})); then
  WORKER_COUNT=${#PENDING_NAMES[@]}
fi

TEMP_DIR=$(mktemp -d --tmpdir ur-sharpa-upload.XXXXXXXX)
cleanup() {
  if [[ -n "${TEMP_DIR:-}" && -d "$TEMP_DIR" && "$TEMP_DIR" == /tmp/ur-sharpa-upload.* ]]; then
    rm -rf -- "$TEMP_DIR"
  fi
}
trap cleanup EXIT

for ((worker = 0; worker < WORKER_COUNT; worker++)); do
  : >"$TEMP_DIR/bucket.$worker"
done
for ((index = 0; index < ${#PENDING_NAMES[@]}; index++)); do
  worker=$((index % WORKER_COUNT))
  printf '%s\n' "${PENDING_NAMES[$index]}" >>"$TEMP_DIR/bucket.$worker"
done

upload_bucket() {
  local worker=$1
  local bucket="$TEMP_DIR/bucket.$worker"
  local count
  count=$(wc -l <"$bucket")
  echo "[upload $((worker + 1))/$WORKER_COUNT] starting $count episodes"
  rsync \
    -a -r \
    --files-from="$bucket" \
    --partial \
    --append-verify \
    --protect-args \
    --human-readable \
    --info=stats2 \
    -e "$RSYNC_RSH" \
    "$SOURCE/" \
    "$REMOTE_HOST:$REMOTE_PARTIAL/"
  echo "[upload $((worker + 1))/$WORKER_COUNT] transfer complete"
}

echo "Uploading ${#PENDING_NAMES[@]} episodes with $WORKER_COUNT parallel connections..."
UPLOAD_PIDS=()
for ((worker = 0; worker < WORKER_COUNT; worker++)); do
  upload_bucket "$worker" &
  UPLOAD_PIDS+=("$!")
done

UPLOAD_FAILED=0
for pid in "${UPLOAD_PIDS[@]}"; do
  if ! wait "$pid"; then
    UPLOAD_FAILED=1
  fi
done
((UPLOAD_FAILED == 0)) || die "one or more rsync workers failed; rerun the script to resume"

verify_bucket() {
  local worker=$1
  local bucket="$TEMP_DIR/bucket.$worker"
  {
    cat <<'REMOTE_SCRIPT'
set -Eeuo pipefail
destination=$1
partial_root=$2
while IFS= read -r episode; do
  [[ -n "$episode" ]] || continue
  [[ -d "$partial_root/$episode" ]] || {
    echo "missing staged episode: $episode" >&2
    exit 1
  }
  cd "$partial_root/$episode"
  sha256sum -c --quiet checksums.sha256
  cd "$partial_root"
  [[ ! -e "$destination/$episode" ]] || {
    echo "final destination appeared during upload: $episode" >&2
    exit 1
  }
  mv -- "$episode" "$destination/$episode"
  echo "verified and finalized: $episode"
done <<'EPISODE_LIST'
REMOTE_SCRIPT
    cat "$bucket"
    echo 'EPISODE_LIST'
  } | ssh "${SSH_OPTIONS[@]}" "$REMOTE_HOST" bash -s -- "$REMOTE_DIR" "$REMOTE_PARTIAL"
}

echo "Verifying server checksums and atomically finalizing episodes..."
VERIFY_PIDS=()
for ((worker = 0; worker < WORKER_COUNT; worker++)); do
  verify_bucket "$worker" &
  VERIFY_PIDS+=("$!")
done

VERIFY_FAILED=0
for pid in "${VERIFY_PIDS[@]}"; do
  if ! wait "$pid"; then
    VERIFY_FAILED=1
  fi
done
((VERIFY_FAILED == 0)) || die "server checksum verification failed; staged data was not finalized"

REMOTE_STATS=$(
  ssh "${SSH_OPTIONS[@]}" "$REMOTE_HOST" \
    "episodes=0; files=0; bytes=0; for episode in '$REMOTE_DIR'/*; do [ -d \"\$episode\" ] || continue; episodes=\$((episodes + 1)); done; while IFS= read -r size; do files=\$((files + 1)); bytes=\$((bytes + size)); done < <(find '$REMOTE_DIR' -mindepth 2 -type f -printf '%s\\n'); printf '%d %d %d\\n' \"\$episodes\" \"\$files\" \"\$bytes\""
) || die "could not read final statistics from $REMOTE_HOST"
read -r REMOTE_EPISODES REMOTE_FILES REMOTE_BYTES <<<"$REMOTE_STATS"

echo "Upload complete."
echo "Remote episodes: $REMOTE_EPISODES"
echo "Remote files:    $REMOTE_FILES"
echo "Remote bytes:    $REMOTE_BYTES"
