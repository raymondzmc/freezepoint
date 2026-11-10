#!/usr/bin/env bash
# Run one cell: every config (one per arm) for every seed, then the analysis report over exactly those runs.
#
#   bash scripts/run_cell.sh configs/cn_frozen.yaml configs/cn_refine.yaml configs/cn_bestofn.yaml
#
# Environment: GEN (passed as --provider, e.g. fake), MODEL (passed as --model), SEEDS (default "0 1 2 3 4"),
# JOBS (runs at once, default 1), RUNS (runs directory, default runs), OUT (report directory, default $RUNS/report).
# If any run fails the script says which and where its log is, exits non-zero and writes no report. On INT (Ctrl-C) or
# TERM it sends TERM to every run it started and to everything those runs started, waits for them, exits non-zero and
# writes no report. Works with the macOS bash 3.2.
set -euo pipefail
set -m      # job control: each background run gets its own process group, so one signal reaches the whole run

[ "$#" -ge 1 ] || { echo "usage: bash scripts/run_cell.sh <config> [<config> ...]" >&2; exit 2; }

export MLFLOW_DISABLE_AGENT_HINT=1      # Qlib imports mlflow, which otherwise logs a line per process
SEEDS=${SEEDS:-"0 1 2 3 4"}
JOBS=${JOBS:-1}
RUNS=${RUNS:-runs}
OUT=${OUT:-$RUNS/report}
LOGS="$OUT/logs"
mkdir -p "$LOGS"

EXTRA=()
[ -z "${GEN:-}" ] || EXTRA+=(--provider "$GEN")
[ -z "${MODEL:-}" ] || EXTRA+=(--model "$MODEL")

# One run. Standard error goes to its log; the run id is the last line of its standard output.
run_one() {
    local cfg=$1 seed=$2 name=$3 out
    rm -f "$LOGS/$name.runid"
    out=$(uv run python -m loop.run "$cfg" --seed "$seed" --runs-dir "$RUNS" ${EXTRA[@]+"${EXTRA[@]}"} 2>"$LOGS/$name.log") || return 1
    printf '%s\n' "$out" | tail -n 1 >"$LOGS/$name.runid"
}

# There is no separate scoring pass: each run scores itself before it writes `done`, and running the same command
# again on a run that finished its arm but died while scoring only fills in the missing scores.
pids=(); names=(); labels=(); all_names=(); failed=0
reap() {    # wait for the oldest background run and note a failure
    local rc=0
    wait "${pids[0]}" || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "FAILED: ${labels[0]} (exit $rc); log: $LOGS/${names[0]}.log" >&2
        failed=1
    fi
    pids=(${pids[@]+"${pids[@]:1}"}); names=(${names[@]+"${names[@]:1}"}); labels=("${labels[@]:1}")
}

# On INT or TERM: end every run still going. A background job ignores SIGINT, and killing only the job's own shell
# would leave its `uv run python` behind, so TERM goes to each run's process group.
stop() {
    trap - INT TERM
    echo "caught SIG$1: stopping ${#pids[@]} run(s) still going" >&2
    for p in ${pids[@]+"${pids[@]}"}; do kill -TERM -- "-$p" 2>/dev/null || true; done
    for p in ${pids[@]+"${pids[@]}"}; do wait "$p" 2>/dev/null || true; done
    echo "stopped; no report written" >&2
    if [ "$1" = INT ]; then exit 130; else exit 143; fi
}
trap 'stop INT' INT
trap 'stop TERM' TERM

start=$SECONDS
i=0
for cfg in "$@"; do
    for s in $SEEDS; do
        i=$((i + 1))
        name=$(printf '%02d-%s-s%s' "$i" "$(basename "$cfg" .yaml)" "$s")
        all_names+=("$name")
        [ "${#pids[@]}" -lt "$JOBS" ] || reap
        run_one "$cfg" "$s" "$name" &
        pids+=($!); names+=("$name"); labels+=("config $cfg, seed $s")
    done
done
while [ "${#pids[@]}" -gt 0 ]; do reap; done

if [ "$failed" -ne 0 ]; then
    echo "at least one run failed; no report written" >&2
    exit 1
fi

dirs=()
for name in "${all_names[@]}"; do dirs+=("$RUNS/$(cat "$LOGS/$name.runid")"); done   # only this invocation's runs
uv run python -m analysis.report "${dirs[@]}" --out "$OUT" >/dev/null

printf '\nWall-clock time of the cell: %d seconds for %d runs.\n' "$((SECONDS - start))" "${#dirs[@]}" >>"$OUT/cost.md"
echo "$OUT/report.md"
