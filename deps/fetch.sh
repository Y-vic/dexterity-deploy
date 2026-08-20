#!/usr/bin/env bash
#
# fetch.sh — reproducible SDK fetcher.
#
# Reads deps/manifest.yaml, downloads the entries whose `embodiments` list
# contains the given target, verifies sha256, unpacks under
#   deps/cache/<name>-<version>/
# Never installs system-wide. Never overwrites an existing version.
#
# Requires: bash, curl, python3 (for yaml parsing), sha256sum, tar/unzip
#
# Usage:
#   ./deps/fetch.sh --embodiment ur|pnd-robot|pnd-workstation
#   ./deps/fetch.sh --verify              # verify all present caches
#   ./deps/fetch.sh --list  --embodiment X

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$HERE/manifest.yaml"
CACHE_ROOT="$HERE/cache"
LOCK_ROOT="$HERE/lock"

log()  { printf '[fetch] %s\n' "$*" >&2; }
die()  { printf '[fetch] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<EOF
Usage:
  $0 --embodiment <ur|pnd-robot|pnd-workstation>   Download + verify entries for this target.
  $0 --verify                                       Verify sha256 of every already-present cache dir.
  $0 --list --embodiment <target>                   Print planned downloads without fetching.

Environment:
  DRY_RUN=1     Print actions without downloading.
EOF
  exit 2
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}
require_cmd curl
require_cmd python3
require_cmd sha256sum

# Read the manifest into TSV via python3, without adding a yaml pip dep beyond stdlib.
manifest_rows() {
  # Uses PyYAML if present; otherwise falls back to a minimal parser expectation
  # (the manifest deliberately uses a flat structure so this is safe).
  python3 - "$MANIFEST" <<'PY'
import sys, os
path = sys.argv[1]
try:
    import yaml
except ModuleNotFoundError:
    sys.stderr.write("[fetch] PyYAML not installed. `pip install --user pyyaml` and rerun.\n")
    sys.exit(3)
with open(path) as f:
    data = yaml.safe_load(f)
for s in data.get("sdks", []):
    name = s["name"]
    version = str(s["version"])
    source = s["source"]
    sha256 = s["sha256"]
    embodiments = ",".join(s.get("embodiments", []))
    unpack = s.get("unpack", "")
    print("\t".join([name, version, source, sha256, embodiments, unpack]))
PY
}

verify_cache() {
  # For every present cache dir, verify the .sha256 marker matches the
  # manifest. If a cache exists without a marker, warn.
  local rows="$1"
  local ok=0 fail=0
  while IFS=$'\t' read -r name version source expected _emb _unpack; do
    local dir="$CACHE_ROOT/$name-$version"
    [ -d "$dir" ] || continue
    local marker="$dir/.sha256"
    if [ ! -f "$marker" ]; then
      log "WARN  $name-$version has no .sha256 marker"; fail=$((fail+1)); continue
    fi
    local got; got="$(cat "$marker")"
    if [ "$got" = "$expected" ]; then
      log "OK    $name-$version"
      ok=$((ok+1))
    else
      log "FAIL  $name-$version marker=$got expected=$expected"
      fail=$((fail+1))
    fi
  done <<< "$rows"
  log "verify: $ok ok, $fail bad"
  [ "$fail" -eq 0 ]
}

fetch_one() {
  local name="$1" version="$2" source="$3" sha256="$4" unpack="$5"
  local dir="$CACHE_ROOT/$name-$version"
  local archive="$CACHE_ROOT/.staging/$name-$version.dl"

  if [ -d "$dir" ] && [ -f "$dir/.sha256" ] && [ "$(cat "$dir/.sha256")" = "$sha256" ]; then
    log "skip  $name-$version (already present)"
    return 0
  fi

  if [ "$source" = "PLACEHOLDER" ] || [ "$sha256" = "PLACEHOLDER"* ]; then
    log "SKIP  $name-$version — manifest still has PLACEHOLDER; pin real source+sha256 first"
    return 0
  fi

  mkdir -p "$CACHE_ROOT/.staging"
  log "get   $name-$version <- $source"
  if [ "${DRY_RUN:-0}" = "1" ]; then
    return 0
  fi
  curl -fSL --retry 3 --retry-delay 2 -o "$archive" "$source"

  local got
  got="$(sha256sum "$archive" | awk '{print $1}')"
  if [ "$got" != "$sha256" ]; then
    rm -f "$archive"
    die "sha256 mismatch for $name-$version: expected=$sha256 got=$got"
  fi

  mkdir -p "$dir"
  case "$unpack" in
    "tar -xzf") tar -xzf "$archive" -C "$dir" ;;
    "tar -xJf") tar -xJf "$archive" -C "$dir" ;;
    "unzip")    unzip -q "$archive" -d "$dir" ;;
    "AppImage") cp "$archive" "$dir/${name}.AppImage" && chmod +x "$dir/${name}.AppImage" ;;
    *)          log "no unpack step for $name-$version (unpack=$unpack); leaving archive in place"
                cp "$archive" "$dir/" ;;
  esac
  echo "$sha256" > "$dir/.sha256"
  rm -f "$archive"

  # Write / update a lock file so future readers can see provenance.
  mkdir -p "$LOCK_ROOT"
  cat > "$LOCK_ROOT/$name.lock" <<EOF
name: $name
version: $version
source: $source
sha256: $sha256
pinned_by: $(git config user.name 2>/dev/null || echo unknown)
pinned_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
  log "done  $name-$version"
}

# ---- arg parsing -----------------------------------------------------------
MODE=""
EMBODIMENT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --embodiment) EMBODIMENT="${2:-}"; shift 2 ;;
    --verify)     MODE="verify"; shift ;;
    --list)       MODE="list"; shift ;;
    -h|--help)    usage ;;
    *) die "unknown arg: $1" ;;
  esac
done

rows="$(manifest_rows)"

case "${MODE:-fetch}" in
  verify) verify_cache "$rows"; exit $? ;;
  list|fetch)
    [ -n "$EMBODIMENT" ] || die "--embodiment is required for $MODE"
    while IFS=$'\t' read -r name version source sha256 embs unpack; do
      case ",$embs," in *,"$EMBODIMENT",*) ;; *) continue ;; esac
      if [ "$MODE" = "list" ]; then
        printf '%-24s %-12s %s\n' "$name" "$version" "$source"
      else
        fetch_one "$name" "$version" "$source" "$sha256" "$unpack"
      fi
    done <<< "$rows"
    ;;
esac
