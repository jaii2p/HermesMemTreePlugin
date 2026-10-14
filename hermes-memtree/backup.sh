#!/usr/bin/env bash
# Back up the memory trees — so no server death ever costs a memory.
#
#   ./backup.sh                # make a backup of every tree found (default)
#   ./backup.sh list           # what's already saved
#   ./backup.sh verify <label> # check a saved backup's integrity + node count
#   ./backup.sh restore <label># put the newest saved tree back in place (asks first)
#
# What it covers:
#   - the local tree beside this package:  ./var  (memtree/tree.json + journal + mute/timer)
#   - a second tree, if you have one:      $MEMTREE_PEER_HOME
#
# Each run writes a timestamped, self-contained snapshot
#   backups/<label>/<label>-<ts>.tar.gz + SHA256SUMS + manifest.json
# and keeps only the newest $KEEP (default 20). Nothing is ever silently deleted
# beyond rotation, and "restore" always keeps the current tree aside first.
#
# Offsite: copy backups/ somewhere else (rsync, cloud, a second disk). A backup on
# the same dying machine is a hope, not a plan.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

BACKUP_ROOT="${BACKUP_ROOT:-$HERE/backups}"
KEEP="${KEEP:-20}"
TS="$(date +%Y%m%dT%H%M%S)"

# label -> tree directory (the dir that CONTAINS memtree/)
LOCAL_HOME="${MEMTREE_LOCAL_HOME:-$HERE/var}"
PEER_HOME="${MEMTREE_PEER_HOME:-$HOME/.hermes/memtree}"
declare -a LABELS=()
declare -a HOMES=()
[ -d "$LOCAL_HOME/memtree" ] && { LABELS+=(local); HOMES+=("$LOCAL_HOME"); }
[ -d "$PEER_HOME/memtree" ] && { LABELS+=(peer);  HOMES+=("$PEER_HOME");  }
[ ${#LABELS[@]} -ge 1 ] || { echo "no trees found to back up (looked in $LOCAL_HOME and $PEER_HOME)"; exit 1; }

node_count() { PYTHONPATH="$HERE" python3 - "$1" 2>/dev/null <<'PY'
import json,sys,os
p=os.path.join(sys.argv[1],"memtree","tree.json")
try:
    d=json.load(open(p))
except FileNotFoundError:
    print(0); raise SystemExit
b=d.get("branches",{})
n=sum(sum(len(lvl) for lvl in br.get("levels",[])) for br in b.values())
print(n)
PY
}

do_run() {
  for i in "${!LABELS[@]}"; do
    label="${LABELS[$i]}"; home="${HOMES[$i]}"
    dest="$BACKUP_ROOT/$label"; mkdir -p "$dest"
    tarball="$dest/$label-$TS.tar.gz"
    pushd "$(dirname "$home")" >/dev/null
    tar -czf "$tarball" "$(basename "$home")/memtree"
    popd >/dev/null
    nodes="$(node_count "$home")"
    sha="$(sha256sum "$tarball" | cut -d' ' -f1)"
    printf '{"label":"%s","ts":"%s","tree":"%s","memories":%s,"file":"%s","sha256":"%s"}\n' \
      "$label" "$TS" "$home/memtree" "$nodes" "$(basename "$tarball")" "$sha" \
      > "$dest/manifest.json"
    echo "  backed up '$label': $nodes memories -> $tarball (${sha:0:16}…)"
  done
  rotate
}

rotate() {
  for label in "${LABELS[@]}"; do
    dest="$BACKUP_ROOT/$label"
    # keep the KEEP newest tarballs, sorted by name (timestamps sort lexically)
    ls -1 "$dest"/$label-*.tar.gz 2>/dev/null | sort -r | tail -n +$((KEEP+1)) | while read -r old; do
      echo "  rotating out $old"
      rm -f -- "$old"
    done || true
  done
}

do_list() {
  if [ ! -d "$BACKUP_ROOT" ]; then echo "no backups yet — run ./backup.sh"; exit 0; fi
  echo "saved snapshots:"
  for label in "${LABELS[@]}"; do
    [ -f "$BACKUP_ROOT/$label/manifest.json" ] || continue
    m="$BACKUP_ROOT/$label/manifest.json"
    echo "  [$label]"
    # newest first per label
    ls -1t "$BACKUP_ROOT/$label"/$label-*.tar.gz 2>/dev/null | while read -r t; do
      printf '     %s  |  %s\n' "$(basename "$t")" "$(sha256sum "$t" | cut -c1-16)"
    done || true
  done
}

do_verify() {
  label="$1"
  match=0
  for home in "${HOMES[@]}"; do :; done
  for i in "${!LABELS[@]}"; do
    [ "${LABELS[$i]}" = "$label" ] && { home="${HOMES[$i]}"; match=1; break; }
  done
  [ "$match" = 1 ] || { echo "unknown label '$label' (known: ${LABELS[*]})"; exit 1; }
  newest="$(ls -1t "$BACKUP_ROOT/$label"/$label-*.tar.gz 2>/dev/null | head -1)"
  [ -n "$newest" ] || { echo "no backups for '$label'"; exit 1; }
  echo "verify: $newest"
  tar -tzf "$newest" >/dev/null || { echo "  FAIL: tarball is corrupt/incomplete"; exit 1; }
  echo "  tarball opens fine (stores: $(tar -tzf "$newest" | wc -l) entries)"
  T=$(mktemp -d); tar -xzf "$newest" -C "$T"
  mapfile -d '' tjs < <(find "$T" -name tree.json -print0)
  if [ ${#tjs[@]} -eq 0 ]; then
    echo "  note: no tree.json inside (was it empty?)"
  else
    for tj in "${tjs[@]}"; do
      c=$(PYTHONPATH="$HERE" python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
print(sum(sum(len(l) for l in br.get("levels", [])) for br in d.get("branches", {}).values()))' "$tj")
      echo "  restorable: $c memories ($tj)"
    done
  fi
  rm -rf "$T"
  echo "  sha256: $(sha256sum "$newest" | cut -d' ' -f1)"
}

do_restore() {
  label="$1"
  match=0; home=""
  for i in "${!LABELS[@]}"; do
    [ "${LABELS[$i]}" = "$label" ] && { home="${HOMES[$i]}"; match=1; break; }
  done
  [ "$match" = 1 ] || { echo "unknown label '$label' (known: ${LABELS[*]})"; exit 1; }
  newest="$(ls -1t "$BACKUP_ROOT/$label"/$label-*.tar.gz 2>/dev/null | head -1)"
  [ -n "$newest" ] || { echo "no backups for '$label'"; exit 1; }
  do_verify "$label"
  echo ""
  read -r -p "restore $(basename "$newest") into $home? this keeps the current tree aside [y/N] " a
  [ "$a" = "y" ] || { echo "aborted"; exit 0; }
  bk="$home/memtree.broken-$(date +%Y%m%dT%H%M%S)"
  [ -d "$home/memtree" ] && mv "$home/memtree" "$bk" && echo "  current tree moved aside -> $bk"
  mkdir -p "$home/memtree"
  tar -xzf "$newest" -C "$(dirname "$home")"
  echo "  restored -> $home/memtree"
}

mode="${1:-run}"
case "$mode" in
  run)        echo "backing up memory trees -> $BACKUP_ROOT"; do_run ;;
  list)       do_list ;;
  verify)     do_verify "${2:?verify needs a label (${LABELS[*]})}" ;;
  restore)    do_restore "${2:?restore needs a label (${LABELS[*]})}" ;;
  *) echo "usage: ./backup.sh [run|list|verify <label>|restore <label>]"; exit 2 ;;
esac
