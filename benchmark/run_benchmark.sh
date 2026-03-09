#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ───────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
KAMSCAN="$PROJECT_DIR/scripts/kamscan.py"
DATAGEN="$SCRIPT_DIR/generate_data.py"
WORKDIR="$SCRIPT_DIR/bench_workdir"
RESULTS_CSV="$SCRIPT_DIR/benchmark_results.csv"

NUM_PATIENTS=80
TOP_TAGS=1000
SEED=42
ALL_TESTS="ttest pitest ziw wilcoxon variance"

# ── Mode selection ──────────────────────────────────────────────────────────
FULL=0
if [[ "${1:-}" == "--full" ]]; then
    FULL=1
fi

if [[ $FULL -eq 1 ]]; then
    DEFAULT_ROWS="10000 50000 200000 1000000"
    DEFAULT_PROCS="1 2 4 8"
    DEFAULT_CHUNKS="1000 5000 10000 50000"
else
    DEFAULT_ROWS="10000 50000"
    DEFAULT_PROCS="1 4"
    DEFAULT_CHUNKS="5000 10000"
fi

# Allow env var overrides
ROWS=(${BENCH_ROWS:-$DEFAULT_ROWS})
TESTS=(${BENCH_TESTS:-$ALL_TESTS})
PROCS=(${BENCH_PROCS:-$DEFAULT_PROCS})
CHUNKS=(${BENCH_CHUNKS:-$DEFAULT_CHUNKS})

# ── GNU time guard ──────────────────────────────────────────────────────────
if ! /usr/bin/time -v true 2>/dev/null; then
    echo "ERROR: /usr/bin/time -v is not available (GNU time required)." >&2
    echo "Install with: sudo apt-get install time" >&2
    exit 1
fi

# ── Utility: parse /usr/bin/time -v output ──────────────────────────────────
parse_elapsed() {
    # Parses "Elapsed (wall clock) time (h:mm:ss or m:ss): ..." from time -v output
    local logfile="$1"
    local raw
    raw=$(grep "Elapsed (wall clock)" "$logfile" | sed 's/.*): //')
    # Format: h:mm:ss or h:mm:ss.ss or m:ss.ss
    local parts
    IFS=':' read -ra parts <<< "$raw"
    if [[ ${#parts[@]} -eq 3 ]]; then
        echo "${parts[0]} * 3600 + ${parts[1]} * 60 + ${parts[2]}" | bc
    elif [[ ${#parts[@]} -eq 2 ]]; then
        echo "${parts[0]} * 60 + ${parts[1]}" | bc
    else
        echo "$raw"
    fi
}

parse_peak_rss() {
    local logfile="$1"
    grep "Maximum resident set size" "$logfile" | awk '{print $NF}'
}

# ── Data generation ─────────────────────────────────────────────────────────
echo "=== KamScan Benchmark ==="
echo "Rows:       ${ROWS[*]}"
echo "Tests:      ${TESTS[*]}"
echo "Processes:  ${PROCS[*]}"
echo "Chunks:     ${CHUNKS[*]}"
echo ""

mkdir -p "$WORKDIR"

for nr in "${ROWS[@]}"; do
    datadir="$WORKDIR/data_${nr}"
    if [[ -f "$datadir/matrix.txt" ]]; then
        echo "Data for $nr rows already exists, skipping generation."
    else
        echo "Generating synthetic data: $nr rows, $NUM_PATIENTS patients..."
        python3 "$DATAGEN" --num_rows "$nr" --num_patients "$NUM_PATIENTS" \
            --output_dir "$datadir" --seed "$SEED"
    fi
done

# ── CSV header ──────────────────────────────────────────────────────────────
echo "num_rows,test_type,processes,chunk_size,wall_clock_sec,peak_memory_kb,exit_code" > "$RESULTS_CSV"

# ── Benchmark runs ──────────────────────────────────────────────────────────
TOTAL=0
for _ in "${ROWS[@]}"; do for _ in "${TESTS[@]}"; do for _ in "${PROCS[@]}"; do for _ in "${CHUNKS[@]}"; do
    TOTAL=$((TOTAL + 1))
done; done; done; done

RUN=0
TIMELOG=$(mktemp)
trap 'rm -f "$TIMELOG"' EXIT

for nr in "${ROWS[@]}"; do
    datadir="$WORKDIR/data_${nr}"
    matrix="$datadir/matrix.txt"
    cond_dir="$datadir/conditions"
    cpm_file="$datadir/design_kmers_nb_per_patient"

    for test in "${TESTS[@]}"; do
        for proc in "${PROCS[@]}"; do
            for chunk in "${CHUNKS[@]}"; do
                RUN=$((RUN + 1))
                outdir="$WORKDIR/output_tmp"
                rm -rf "$outdir"
                mkdir -p "$outdir"

                echo -n "[$RUN/$TOTAL] rows=$nr test=$test procs=$proc chunk=$chunk ... "

                exit_code=0
                /usr/bin/time -v python3 "$KAMSCAN" \
                    -i "$matrix" \
                    -o "$outdir" \
                    -d "$cond_dir" \
                    -t "$TOP_TAGS" \
                    -c "$chunk" \
                    -p "$proc" \
                    -m "$cpm_file" \
                    --test_type "$test" \
                    2>"$TIMELOG" || exit_code=$?

                wall=$(parse_elapsed "$TIMELOG")
                mem=$(parse_peak_rss "$TIMELOG")

                echo "${wall}s, ${mem}KB, exit=$exit_code"
                echo "$nr,$test,$proc,$chunk,$wall,$mem,$exit_code" >> "$RESULTS_CSV"

                # Clean up output to save disk
                rm -rf "$outdir"
            done
        done
    done
done

echo ""
echo "Results written to $RESULTS_CSV"
echo ""

# ── Summary table ───────────────────────────────────────────────────────────
echo "=== Summary (wall_clock_sec: median / min / max) grouped by test_type x num_rows ==="
echo ""
printf "%-12s %-10s %10s %10s %10s %5s\n" "test_type" "num_rows" "median" "min" "max" "runs"
printf "%-12s %-10s %10s %10s %10s %5s\n" "-----------" "---------" "---------" "---------" "---------" "----"

# Collect unique (test, rows) pairs and compute stats
tail -n +2 "$RESULTS_CSV" | awk -F',' '{print $2, $1}' | sort -u | while read -r test nr; do
    times=$(tail -n +2 "$RESULTS_CSV" | awk -F',' -v t="$test" -v n="$nr" '$2==t && $1==n {print $5}')
    count=$(echo "$times" | wc -l | tr -d ' ')
    sorted=$(echo "$times" | sort -g)
    min_v=$(echo "$sorted" | head -1)
    max_v=$(echo "$sorted" | tail -1)
    mid=$((count / 2))
    if (( count % 2 == 1 )); then
        median=$(echo "$sorted" | sed -n "$((mid + 1))p")
    else
        v1=$(echo "$sorted" | sed -n "${mid}p")
        v2=$(echo "$sorted" | sed -n "$((mid + 1))p")
        median=$(echo "scale=2; ($v1 + $v2) / 2" | bc)
    fi
    printf "%-12s %-10s %10s %10s %10s %5s\n" "$test" "$nr" "$median" "$min_v" "$max_v" "$count"
done

echo ""
echo "Done."
