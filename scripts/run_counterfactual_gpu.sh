#!/bin/bash
#SBATCH --job-name=counterfactual_gpu
#SBATCH --partition=compute
#SBATCH --gres=gpu:nvidia:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/storage/home/johannes_maekelburg/Causality-Inconsistency-Uncertainty/counterfactual_%j.log
#SBATCH --error=/storage/home/johannes_maekelburg/Causality-Inconsistency-Uncertainty/counterfactual_%j.err

set -euo pipefail

REPO_DIR=/storage/home/johannes_maekelburg/Causality-Inconsistency-Uncertainty
cd "$REPO_DIR"

CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"
if [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "Conda init script not found at $CONDA_BASE/etc/profile.d/conda.sh" >&2
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate causal

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

# Override these with environment variables before calling sbatch if needed.
CHECKPOINT="${CHECKPOINT:-results/surrogate/asymmetric_v11b/best_model.pt}"
SCENARIOS="${SCENARIOS:-cps}"
QUERIES="${QUERIES:-20}"
GAMMA="${GAMMA:-0.3}"
GAMMA_SURROGATE="${GAMMA_SURROGATE:-}"
MC_VERIFY="${MC_VERIFY:-500}"
MC_SCREEN="${MC_SCREEN:-200}"
MAX_ITER="${MAX_ITER:-500}"
LR="${LR:-0.01}"
LAMBDA_INIT="${LAMBDA_INIT:-10.0}"
SEARCH_MODE="${SEARCH_MODE:-hybrid}"
CANDIDATE_STARTS="${CANDIDATE_STARTS:-3}"
RERANK_TOP_K="${RERANK_TOP_K:-4}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-auto}"
PARALLEL="${PARALLEL:-0}"
WORKERS="${WORKERS:-4}"
MEASUREMENT_ROOT="${MEASUREMENT_ROOT:-data/measurements}"
MAX_PRE_INCONSISTENCY="${MAX_PRE_INCONSISTENCY:-}"
MAX_QUERY_INCONSISTENCY="${MAX_QUERY_INCONSISTENCY:-}"
OUTPUT_PATH="${OUTPUT_PATH:-results/counterfactual/counterfactual_gpu_${SLURM_JOB_ID}.json}"

mkdir -p "$(dirname "$OUTPUT_PATH")"

CMD=(
  python src/run_counterfactual_eval.py
  --checkpoint "$CHECKPOINT"
  --scenarios "$SCENARIOS"
  --queries "$QUERIES"
  --gamma "$GAMMA"
  --mc-verify "$MC_VERIFY"
  --mc-screen "$MC_SCREEN"
  --max-iter "$MAX_ITER"
  --lr "$LR"
  --lambda-init "$LAMBDA_INIT"
  --search-mode "$SEARCH_MODE"
  --candidate-starts "$CANDIDATE_STARTS"
  --rerank-top-k "$RERANK_TOP_K"
  --seed "$SEED"
  --device "$DEVICE"
  --output "$OUTPUT_PATH"
)

if [[ -n "$GAMMA_SURROGATE" ]]; then
  CMD+=(--gamma-surrogate "$GAMMA_SURROGATE")
fi

if [[ -n "$MEASUREMENT_ROOT" ]]; then
  CMD+=(--measurement-root "$MEASUREMENT_ROOT")
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

echo "Running command:"
printf '  %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
