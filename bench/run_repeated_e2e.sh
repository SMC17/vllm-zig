#!/usr/bin/env bash
# Repeat the bench-tinyllama-pool harness N times, collect per-run
# numbers, report median + p10/p90 over the runs. Used by BENCH.md.
#
# Each run loads the model fresh (~25s), runs (A) no-pool + (B) pool,
# both at v0.0.6 code on the same machine state. We capture the (B)
# decode-only ms/tok line from each run.
#
# Usage: run_repeated_e2e.sh N
# Output: stdout = a JSON line per run, then a summary block.

set -euo pipefail
N=${1:-5}
OUT_DIR=${OUT_DIR:-/tmp/vllm_zig_bench}
mkdir -p "$OUT_DIR"

BIN=$(find /home/user/vllm-zig/.zig-cache -name bench-tinyllama-pool -type f -newer /home/user/vllm-zig/src/model.zig 2>/dev/null | head -1)
if [ -z "$BIN" ]; then
  echo "no fresh bench-tinyllama-pool binary; run: zig build bench-tinyllama-pool" >&2
  exit 1
fi

declare -a A_DECODE B_DECODE A_FIRST B_FIRST A_TOTAL B_TOTAL

for i in $(seq 1 "$N"); do
  echo "=== run $i / $N — $(date -Iseconds) load=$(awk '{print $1}' /proc/loadavg) ===" >&2
  log="$OUT_DIR/run_$i.log"
  cd /home/user/vllm-zig && nice -n 19 "$BIN" 2>&1 | tee "$log" >&2
  a_decode=$(awk '/no pool/{flag=1} flag && /decode-only ms\/tok:/{print $NF; exit}' "$log")
  a_first=$(awk '/no pool/{flag=1} flag && /first-token \(prefill\+1\):/{print $(NF-1); exit}' "$log")
  a_total=$(awk '/no pool/{flag=1} flag && /^  total: /{print $2; exit}' "$log")
  b_decode=$(awk '/with persistent thread pool/{flag=1} flag && /decode-only ms\/tok:/{print $NF; exit}' "$log")
  b_first=$(awk '/with persistent thread pool/{flag=1} flag && /first-token \(prefill\+1\):/{print $(NF-1); exit}' "$log")
  b_total=$(awk '/with persistent thread pool/{flag=1} flag && /^  total: /{print $2; exit}' "$log")
  A_DECODE+=("${a_decode:-0}"); B_DECODE+=("${b_decode:-0}")
  A_FIRST+=("${a_first:-0}");   B_FIRST+=("${b_first:-0}")
  A_TOTAL+=("${a_total:-0}");   B_TOTAL+=("${b_total:-0}")
  echo "{\"run\":$i,\"a_decode_ms\":${a_decode:-0},\"b_decode_ms\":${b_decode:-0},\"a_first_ms\":${a_first:-0},\"b_first_ms\":${b_first:-0},\"a_total_ms\":${a_total:-0},\"b_total_ms\":${b_total:-0}}"
done

p() {
  # usage: p <pct> <values...>
  local pct=$1; shift
  printf '%s\n' "$@" | sort -n | awk -v n="$#" -v p="$pct" 'BEGIN{idx=int((p/100.0)*(n-1))+1} NR==idx{print; exit}'
}

summary() {
  local label=$1; shift
  local p10 p50 p90
  p10=$(p 10 "$@"); p50=$(p 50 "$@"); p90=$(p 90 "$@")
  printf "%-18s median=%s  p10=%s  p90=%s  raw=[%s]\n" "$label" "$p50" "$p10" "$p90" "$(echo "$@" | tr ' ' ',')"
}

echo
echo "=== summary over $N runs ==="
summary "A decode ms/tok"   "${A_DECODE[@]}"
summary "B decode ms/tok"   "${B_DECODE[@]}"
summary "A first-token ms"  "${A_FIRST[@]}"
summary "B first-token ms"  "${B_FIRST[@]}"
summary "A total ms"        "${A_TOTAL[@]}"
summary "B total ms"        "${B_TOTAL[@]}"
