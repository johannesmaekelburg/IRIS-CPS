#!/bin/bash
#SBATCH --job-name=multifidelity_cf
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=compute
#SBATCH --output=/storage/home/johannes_maekelburg/Causality-Inconsistency-Uncertainty/run_multifidelity_comparison_%j.log
#SBATCH --error=/storage/home/johannes_maekelburg/Causality-Inconsistency-Uncertainty/run_multifidelity_comparison_%j.err

###############################################################################
# Multi-Fidelity Counterfactual Evaluation and Figure Generation
#
# Runs counterfactual search with surrogate warm-start plus MFMC refinement
# and automatically generates comparison figures for:
#   - Phase 1 (surrogate) vs Phase 2 (MFMC-refined) final inconsistency
#   - Cost-quality tradeoffs across all three fidelity levels
#   - Sample complexity after warm-start vs cold-start
#
# Usage:
#   sbatch scripts/run_multifidelity_comparison.sh
#   sbatch --cpus-per-task=16 --mem=64G scripts/run_multifidelity_comparison.sh
#
###############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Script directory: $SCRIPT_DIR"
echo "Repo directory:   $REPO_DIR"

if [[ ! -f "$REPO_DIR/src/run_counterfactual_eval.py" ]]; then
    echo "ERROR: Could not find src/run_counterfactual_eval.py in $REPO_DIR" >&2
    echo "This script must be in the repo/scripts/ directory" >&2
    exit 1
fi

cd "$REPO_DIR"
echo "Changed to: $(pwd)"
echo ""

CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"
CONDA_INIT="$CONDA_BASE/etc/profile.d/conda.sh"
if [[ ! -f "$CONDA_INIT" ]]; then
    echo "ERROR: Conda init script not found at $CONDA_INIT" >&2
    exit 1
fi
source "$CONDA_INIT"

CONDA_ENV="${CONDA_ENV:-causal}"
conda activate "$CONDA_ENV"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export PYTHONUNBUFFERED=1

# Model and surrogate configuration
CHECKPOINT="${CHECKPOINT:-results/surrogate/asymmetric_v11b/best_model.pt}"
SURROGATE_DATA_ROOT="${SURROGATE_DATA_ROOT:-data/surrogate}"
SEARCH_MODE="${SEARCH_MODE:-multifidelity}"

# Evaluation parameters
SCENARIOS="${SCENARIOS:-convide}"
QUERIES="${QUERIES:-50}"
SEED="${SEED:-42}"
GAMMA="${GAMMA:-0.3}"
GAMMA_SURROGATE="${GAMMA_SURROGATE:-}"
MC_VERIFY="${MC_VERIFY:-500}"
MC_SCREEN="${MC_SCREEN:-200}"
MAX_ITER="${MAX_ITER:-500}"
LR="${LR:-0.01}"
LAMBDA_INIT="${LAMBDA_INIT:-10.0}"
CANDIDATE_STARTS="${CANDIDATE_STARTS:-3}"
RERANK_TOP_K="${RERANK_TOP_K:-4}"
MF_SPSA_ITER="${MF_SPSA_ITER:-40}"
MF_MC_PER_EVAL="${MF_MC_PER_EVAL:-50}"
MEASUREMENT_ROOT="${MEASUREMENT_ROOT:-data/measurements}"
MAX_PRE_INCONSISTENCY="${MAX_PRE_INCONSISTENCY:-}"
MAX_QUERY_INCONSISTENCY="${MAX_QUERY_INCONSISTENCY:-}"
DEVICE="${DEVICE:-auto}"
PARALLEL="${PARALLEL:-0}"
WORKERS="${WORKERS:-4}"

# Output paths
RESULTS_DIR_BASE="${RESULTS_DIR:-$REPO_DIR/results/counterfactual}"
FIGURES_BASE_DIR="${FIGURES_BASE_DIR:-$REPO_DIR/figures/counterfactual}"

if [[ ! "$RESULTS_DIR_BASE" = /* ]]; then
    RESULTS_DIR_BASE="$REPO_DIR/$RESULTS_DIR_BASE"
fi
if [[ ! "$FIGURES_BASE_DIR" = /* ]]; then
    FIGURES_BASE_DIR="$REPO_DIR/$FIGURES_BASE_DIR"
fi

RESULTS_DIR="$RESULTS_DIR_BASE"
if ! mkdir -p "$RESULTS_DIR" 2>/dev/null; then
    echo "WARNING: Could not create '$RESULTS_DIR' (permission denied)"
    echo "Falling back to /tmp"
    RESULTS_DIR="/tmp/counterfactual_${USER}_$$"
    mkdir -p "$RESULTS_DIR"
else
    echo "[ok] Created results directory: $RESULTS_DIR"
fi

if ! mkdir -p "$FIGURES_BASE_DIR" 2>/dev/null; then
    echo "WARNING: Could not create '$FIGURES_BASE_DIR' (permission denied)"
    FIGURES_BASE_DIR="/tmp/figures_${USER}_$$"
    mkdir -p "$FIGURES_BASE_DIR"
else
    echo "[ok] Created figures directory: $FIGURES_BASE_DIR"
fi

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
CKPT_TAG="$(basename "$(dirname "$CHECKPOINT_FULL")")"
RUN_ID="${TIMESTAMP}_${CKPT_TAG}"

CF_RESULTS_JSON="${RESULTS_DIR}/multifidelity_eval_${RUN_ID}.json"
FIGURES_DIR="${FIGURES_BASE_DIR}/multifidelity_${RUN_ID}"
mkdir -p "$FIGURES_DIR"
LOG_FILE="${RESULTS_DIR}/multifidelity_${RUN_ID}.log"
SUMMARY_FILE="${RESULTS_DIR}/multifidelity_summary_${RUN_ID}.txt"

CHECKPOINT_FULL="$CHECKPOINT"
if [[ ! "$CHECKPOINT_FULL" = /* ]]; then
    CHECKPOINT_FULL="$REPO_DIR/$CHECKPOINT"
fi

SURROGATE_DATA_ROOT_FULL="$SURROGATE_DATA_ROOT"
if [[ ! "$SURROGATE_DATA_ROOT_FULL" = /* ]]; then
    SURROGATE_DATA_ROOT_FULL="$REPO_DIR/$SURROGATE_DATA_ROOT"
fi

MEASUREMENT_ROOT_FULL="$MEASUREMENT_ROOT"
if [[ ! "$MEASUREMENT_ROOT_FULL" = /* ]]; then
    MEASUREMENT_ROOT_FULL="$REPO_DIR/$MEASUREMENT_ROOT"
fi

echo "###########################################################################"
echo "ENVIRONMENT AND OUTPUT PATHS"
echo "###########################################################################"
echo "Results JSON: $CF_RESULTS_JSON"
echo "Figures dir:  $FIGURES_DIR"
echo "Log file:     $LOG_FILE"
echo ""

CMD=(
  python "$REPO_DIR/src/run_counterfactual_eval.py"
  --checkpoint "$CHECKPOINT_FULL"
  --data-root "$SURROGATE_DATA_ROOT_FULL"
  --scenarios "$SCENARIOS"
  --queries "$QUERIES"
  --seed "$SEED"
  --gamma "$GAMMA"
  --mc-verify "$MC_VERIFY"
  --mc-screen "$MC_SCREEN"
  --max-iter "$MAX_ITER"
  --lr "$LR"
  --lambda-init "$LAMBDA_INIT"
  --search-mode "$SEARCH_MODE"
  --candidate-starts "$CANDIDATE_STARTS"
  --rerank-top-k "$RERANK_TOP_K"
  --mf-spsa-iter "$MF_SPSA_ITER"
  --mf-mc-per-eval "$MF_MC_PER_EVAL"
  --measurement-root "$MEASUREMENT_ROOT_FULL"
  --output "$CF_RESULTS_JSON"
  --device "$DEVICE"
)

if [[ -n "$GAMMA_SURROGATE" ]]; then
  CMD+=(--gamma-surrogate "$GAMMA_SURROGATE")
fi

if [[ -n "$MAX_PRE_INCONSISTENCY" ]]; then
  CMD+=(--max-pre-inconsistency "$MAX_PRE_INCONSISTENCY")
fi

if [[ -n "$MAX_QUERY_INCONSISTENCY" ]]; then
  CMD+=(--max-query-inconsistency "$MAX_QUERY_INCONSISTENCY")
fi

if [[ "$PARALLEL" == "1" ]]; then
  CMD+=(--parallel --workers "$WORKERS")
fi

echo "###########################################################################"
echo "Multi-Fidelity Counterfactual Search and Figure Generation"
echo "###########################################################################"
echo "Timestamp:         $TIMESTAMP"
echo "Run ID:            $RUN_ID"
echo "Checkpoint:        $CHECKPOINT"
echo "Scenarios:         $SCENARIOS ($QUERIES queries each, seed=$SEED)"
echo "Search mode:       $SEARCH_MODE"
echo "Phase 2 config:    $MF_SPSA_ITER SPSA iterations, $MF_MC_PER_EVAL MC samples/eval"
echo "MC verify budget:  $MC_VERIFY samples"
echo "Output JSON:       $CF_RESULTS_JSON"
echo "Figures directory: $FIGURES_DIR"
echo "Log file:          $LOG_FILE"
echo "###########################################################################"
echo ""
echo "Running command:"
printf '  %q' "${CMD[@]}"
printf '\n\n'

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"

echo ""
echo "Counterfactual evaluation completed."
echo "Results saved to: $CF_RESULTS_JSON"
echo ""

if [[ ! -f "$CF_RESULTS_JSON" ]]; then
  echo "ERROR: Counterfactual results JSON not found: $CF_RESULTS_JSON" >&2
  exit 1
fi

echo "###########################################################################"
echo "Generating Multi-Fidelity Comparison Figures"
echo "###########################################################################"
echo ""

echo "[1/1] Generating counterfactual figures..."
python "$REPO_DIR/src/analysis/generate_counterfactual_figures.py" \
  --results "$CF_RESULTS_JSON" \
  --output "$FIGURES_DIR" \
  2>&1 | tee -a "$LOG_FILE"

echo ""
echo "###########################################################################"
echo "Figure Generation Complete"
echo "###########################################################################"
echo "Output directory: $FIGURES_DIR"
ls -lh "$FIGURES_DIR"/*.pdf 2>/dev/null || echo "(No PDFs generated)"
ls -lh "$FIGURES_DIR"/*.png 2>/dev/null || echo "(No PNGs generated)"
echo ""

echo "###########################################################################"
echo "Generating Summary Report"
echo "###########################################################################"
echo ""

python <<PYTHON_SUMMARY
import json
import numpy as np

with open("$CF_RESULTS_JSON") as f:
    results = json.load(f)

per_scenario = results.get("per_scenario_summary", [])
agg = results.get("aggregate", {})

all_valid_counts = []
all_I_mc_deltas = []

for s_summary in per_scenario:
    n_valid = int(round(s_summary.get("n_queries", 0) * s_summary.get("validity_rate", 0.0)))
    all_valid_counts.append(n_valid)
    mc = s_summary.get("mean_I_mc_prime")
    star = s_summary.get("mean_I_surrogate_star")
    if mc is not None and star is not None:
        all_I_mc_deltas.append(star - mc)

with open("$SUMMARY_FILE", "w") as f:
    f.write("=" * 80 + "\n")
    f.write("Multi-Fidelity Counterfactual Search Summary\n")
    f.write("=" * 80 + "\n")
    f.write("Date:               $TIMESTAMP\n")
    f.write("Checkpoint:         $CHECKPOINT\n")
    f.write("Scenarios:          $SCENARIOS\n")
    f.write("Search mode:        $SEARCH_MODE\n")
    f.write("Queries per scenario: $QUERIES\n")
    f.write("Phase 2 config:     $MF_SPSA_ITER SPSA iters, $MF_MC_PER_EVAL MC/MFMC evals\n")
    f.write("\n")
    total_valid = sum(all_valid_counts)
    total_queries = sum(s.get("n_queries", 0) for s in per_scenario)
    validity_pct = 100.0 * total_valid / total_queries if total_queries > 0 else 0.0
    f.write(f"Valid fixes found:  {total_valid}/{total_queries} ({validity_pct:.1f}%)\n")
    f.write(f"  Mean per scenario: {np.mean(all_valid_counts):.2f}\n")
    if agg:
        f.write(f"  MC below 0.4:  {agg.get('mc_below_040_rate', 0)*100:.1f}%\n")
        f.write(f"  MC below 0.5:  {agg.get('mc_below_050_rate', 0)*100:.1f}%\n")
    f.write("\n")
    if all_I_mc_deltas:
        f.write("Inconsistency reduction I(theta*) - I_MC(theta') per scenario:\n")
        f.write(f"  Mean: {np.mean(all_I_mc_deltas):.4f}\n")
        f.write(f"  Std:  {np.std(all_I_mc_deltas):.4f}\n")
        f.write(f"  Min:  {np.min(all_I_mc_deltas):.4f}\n")
        f.write(f"  Max:  {np.max(all_I_mc_deltas):.4f}\n")
    f.write("\n")
    f.write("=" * 80 + "\n")
    f.write("Output files saved to: $FIGURES_DIR\n")

print(f"Summary written to: $SUMMARY_FILE")
PYTHON_SUMMARY

cat "$SUMMARY_FILE"

echo ""
echo "###########################################################################"
echo "SUCCESS: Multi-Fidelity Evaluation and Figures Complete"
echo "###########################################################################"
echo "Results directory: $RESULTS_DIR"
echo "Figures directory: $FIGURES_DIR"
echo "Summary:           $SUMMARY_FILE"
echo ""
