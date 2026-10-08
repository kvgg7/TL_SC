#!/usr/bin/env bash
# Submit one Slurm task per fixed a, then summarize after all tasks succeed.
# Start with: bash run_telomere_w1.sh submit

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Data selection: choose lab, natural or synthetic.
MODE="${MODE:-lab}"
DATA="${DATA:-data/Filtred_Results.csv}"
GROUP="${GROUP:-WT}"                 # lab: WT, RIF1, KU70, WT_rep1, ...
STRAIN="${STRAIN:-BGP_1a}"           # natural
EXTREMITY="${EXTREMITY:-chrI_Left}" # natural
LABEL="${LABEL:-WT_pooled}"
OUT_ROOT="${OUT_ROOT:-results_telomere_w1}"

# Model and optimization settings.
LMAX="${LMAX:-1400}"
A_MIN="${A_MIN:-2}"
A_MAX="${A_MAX:-20}"
BETA_MIN="${BETA_MIN:-0.0001}"
BETA_MAX="${BETA_MAX:-0.20}"
LS_MIN="${LS_MIN:-0}"
LS_MAX="${LS_MAX:-400}"
P_MIN="${P_MIN:-0.003}"
P_MAX="${P_MAX:-0.20}"
LS_ALPHA="${LS_ALPHA:-none}"         # none or 0.01, 0.05, 0.10
OUTLIER_FILTER="${OUTLIER_FILTER:-upper_iqr}" # none, upper_iqr, two_sided_iqr
IQR_K="${IQR_K:-2.0}"
OVERFLOW_DATA="${OVERFLOW_DATA:-error}" # error or drop; never clip at Lmax
MIN_N="${MIN_N:-1001}"
RESTARTS="${RESTARTS:-15}"
MAXITER="${MAXITER:-80}"
POPSIZE="${POPSIZE:-12}"
BOOTSTRAP="${BOOTSTRAP:-500}"
SOLVER_TOL="${SOLVER_TOL:-1e-9}"
SOLVER_MAXITER="${SOLVER_MAXITER:-50000}"
SEED="${SEED:-2026}"
KEEP_PARTS="${KEEP_PARTS:-0}"

# Synthetic example (used only when MODE=synthetic).
SYNTHETIC_N="${SYNTHETIC_N:-3000}"
TRUE_A="${TRUE_A:-4}"
TRUE_BETA="${TRUE_BETA:-0.045}"
TRUE_LS="${TRUE_LS:-90}"
TRUE_P="${TRUE_P:-0.026}"

# Cluster settings. Override with environment variables if needed.
PARTITION="${PARTITION:-long}"
WALLTIME="${WALLTIME:-12:00:00}"
MEMORY="${MEMORY:-8G}"
PARALLEL="${PARALLEL:-6}"
PYTHON="${PYTHON:-python}"
VENV="${VENV:-}"                     # optional virtual environment path

OUT_DIR="$OUT_ROOT/$LABEL"
STAGE="${1:-submit}"

if [[ "$STAGE" == "submit" ]]; then
    mkdir -p "$ROOT/logs"
    if [[ "$A_MAX" -lt "$A_MIN" ]]; then
        echo "A_MAX must be >= A_MIN" >&2
        exit 1
    fi
    if [[ -d "$OUT_DIR/_parts" ]]; then
        echo "Existing partial fits in $OUT_DIR/_parts; use a new LABEL or remove them." >&2
        exit 1
    fi
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "sbatch is not available. Run this on the Slurm login node." >&2
        exit 1
    fi

    n_tasks=$((A_MAX - A_MIN))
    job_id=$(sbatch --parsable --partition="$PARTITION" --time="$WALLTIME" \
        --cpus-per-task=1 --mem="$MEMORY" \
        --array="0-${n_tasks}%${PARALLEL}" \
        --output="$ROOT/logs/${LABEL}_fit_%A_%a.out" \
        --error="$ROOT/logs/${LABEL}_fit_%A_%a.err" \
        "$ROOT/run_telomere_w1.sh" fit)
    echo "Fixed-a fitting job: $job_id"

    summary_id=$(sbatch --parsable --partition="$PARTITION" --time="$WALLTIME" \
        --cpus-per-task=1 --mem="$MEMORY" --dependency="afterok:$job_id" \
        --output="$ROOT/logs/${LABEL}_summary_%j.out" \
        --error="$ROOT/logs/${LABEL}_summary_%j.err" \
        "$ROOT/run_telomere_w1.sh" summarize)
    echo "Summary job: $summary_id (starts when all fits succeed)"
    exit 0
fi

# Use the same Python environment for the fitting and summary jobs.
if [[ -n "$VENV" ]]; then
    source "$VENV/bin/activate"
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

ARGS=(
    --stage "$STAGE" --mode "$MODE" --out "$OUT_DIR"
    --Lmax "$LMAX" --a-min "$A_MIN" --a-max "$A_MAX"
    --beta-min "$BETA_MIN" --beta-max "$BETA_MAX"
    --Ls-min "$LS_MIN" --Ls-max "$LS_MAX"
    --p-min "$P_MIN" --p-max "$P_MAX"
    --outlier-filter "$OUTLIER_FILTER" --iqr-k "$IQR_K" --overflow-data "$OVERFLOW_DATA" --min-n "$MIN_N"
    --restarts "$RESTARTS" --cma-maxiter "$MAXITER" --cma-popsize "$POPSIZE"
    --bootstrap "$BOOTSTRAP" --tol "$SOLVER_TOL"
    --solver-maxiter "$SOLVER_MAXITER" --seed "$SEED"
)

case "$MODE" in
    lab)       ARGS+=(--data "$DATA" --group "$GROUP") ;;
    natural)   ARGS+=(--data "$DATA" --strain "$STRAIN" --extremity "$EXTREMITY") ;;
    synthetic) ARGS+=(--N "$SYNTHETIC_N" --true-a "$TRUE_A" --true-beta "$TRUE_BETA"
                       --true-Ls "$TRUE_LS" --true-p "$TRUE_P") ;;
    *) echo "Invalid MODE: $MODE" >&2; exit 1 ;;
esac

if [[ "$LS_ALPHA" != "none" ]]; then ARGS+=(--Ls-alpha "$LS_ALPHA"); fi
if [[ "$KEEP_PARTS" == "1" ]]; then ARGS+=(--keep-parts); fi

if [[ "$STAGE" == "fit" ]]; then
    : "${SLURM_ARRAY_TASK_ID:?Expected a Slurm array task}"
    ARGS+=(--fixed-a "$((A_MIN + SLURM_ARRAY_TASK_ID))")
elif [[ "$STAGE" != "summarize" ]]; then
    echo "Use: bash run_telomere_w1.sh submit" >&2
    exit 1
fi

"$PYTHON" -u "$ROOT/telomere_w1_pipeline.py" "${ARGS[@]}"
