# Causal Inference: Uncertainty → Inconsistency

A framework for analyzing the causal relationship between uncertainty and inconsistency in cyber-physical systems using interventional experiments.

## Purpose

This project investigates whether and how uncertainty causes inconsistency in constrained polynomial zonotope propagation. Using Pearl's do-calculus and intervention types (widen, shrink, correlate), the framework quantifies causal effects via the global inconsistency metric I(θ). 

**Current Features:**
- 12 CONVIDE scenarios (4×2D, 4×3D, 4×4D) with engineering applications
- Global inconsistency metric I(θ) with Monte Carlo estimation
- Causal effects, local sensitivity, robustness margins
- Sobol variance-based sensitivity indices
- Multi-fidelity Monte Carlo (AABB low-fidelity + MC high-fidelity)
- Learned surrogate data generation pipeline (for GNN-based inconsistency prediction)

## Usage

### MATLAB: Generate CONVIDE Data
```matlab
% Navigate to examples folder
cd examples

% Generate CONVIDE scenarios with I_theta metric
generate_convide_examples
```

### Python: Sensitivity Analysis
```bash
# Analyze combined 2D+3D data (all 8 scenarios)
python src/sensitivity_analysis.py --data_dir data/convide_balanced --output_dir results/sensitivity --param param_value --threshold 0.5

# Analyze 2D only (scenarios 1-4)
python src/sensitivity_analysis.py --data_dir data/convide_with_I_theta --output_dir results/sensitivity_2d --param param_value
```

**Outputs:** 6 files including causal effects, local sensitivity, robustness margins, Sobol indices, surrogate model, and summary JSON.

**Available Datasets:**
- `convide_with_I_theta/`: 2D scenarios (1-4) - CAD drift, MBSE mismatch, etc.
- `convide_balanced/`: 2D+3D scenarios (1-8) - Complete engineering scenarios

### Learned Surrogate: Data Generation

The `src/learned_surrogate/` module generates training data for a GNN-based surrogate that predicts I(θ) directly from zonotope geometry, replacing expensive MC sampling at inference time.

**Pretraining data** (scenario-independent, teaches zonotope geometry):
- `volume` — random zonotopes → interval hull volume
- `containment` — random zonotopes + test points → in/out (LP-based)
- `pairwise_aabb` — random zonotope pairs → AABB Jaccard overlap
- `affine_map` — source + affine map → target (learns UPR propagation)

**Training data** (per-scenario, high-fidelity targets):
- Samples θ from parameter bounds, computes I(θ) via MC probability
- Also stores free AABB estimates for multi-fidelity training
- Stores post-intervention zonotope features for direct GNN input

```bash
cd src

# Generate everything for all 12 scenarios
python learned_surrogate/generate_data.py --tasks all --scenario all

# Just training data for scenario 4
python learned_surrogate/generate_data.py --tasks train --scenario 4

# Just specific pretraining objectives
python learned_surrogate/generate_data.py --tasks pretrain_volume pretrain_pairwise_aabb

# High-fidelity run for one scenario
python learned_surrogate/generate_data.py --tasks train --scenario 8 \
    --n_train 20000 --mc_samples 5000
```

Output structure:
```
data/surrogate/
  meta.json
  pretrain/
    volume.npz, containment.npz, pairwise_aabb.npz, affine_map.npz
  scenario_S01_CAD_Export_Drift/
    meta.json                     # scenario geometry (source, target, F, f)
    train_inconsistency.npz       # theta, I_mc, I_aabb, zonotope features
  ...
```

Use `--overwrite` to regenerate existing files; without it, existing outputs are skipped (resume-friendly).

### Learned Surrogate: Training

Train a GINE-based GNN to predict I(θ) from zonotope graph structure. Supports self-supervised pretraining on geometry tasks followed by supervised fine-tuning.

```bash
# 1. Generate training data (5k pretrain samples, 2k per scenario, 1k MC samples)
.venv/bin/python -m src.learned_surrogate.generate_data \
    --tasks all --scenario all \
    --n_pretrain 5000 --n_train 2000 --mc_samples 1000 \
    --output_dir data/surrogate

# 2. Train (pretrain + fine-tune) with default config
.venv/bin/python -m src.learned_surrogate.train \
    --config configs/surrogate_default.yaml

# Fine-tune only from a pretrained checkpoint
.venv/bin/python -m src.learned_surrogate.train \
    --config configs/surrogate_default.yaml \
    --finetune_only --checkpoint results/surrogate/pretrained_backbone.pt

# Pretrain only (no fine-tuning)
.venv/bin/python -m src.learned_surrogate.train \
    --config configs/surrogate_default.yaml --pretrain_only
```

All hyperparameters (pretraining tasks, train/test scenario splits, architecture, learning rates, etc.) are controlled via the YAML config — see `configs/surrogate_default.yaml` for all options.

Output:
```
results/surrogate/
  pretrained_backbone.pt   # backbone weights after pretraining
  best_model.pt            # best fine-tuned model (by val MSE)
  results.json             # final metrics + per-scenario test breakdown
```

## Requirements

**MATLAB (R2021a+)**:
- CORA Toolbox: https://tumcps.github.io/CORA/
- CPS-Uncertainty-Propagation-Framework: https://github.com/DE-TUM/CPS-Uncertainty-Propagation-Framework

**Python (3.8+)**:
```bash
pip install -r requirements.txt
```

Main packages: `numpy`, `scipy`, `matplotlib`, `pandas`, `seaborn`, `scikit-learn`


Core documentation in `docs/`:
- [QUICKSTART.md](docs/QUICKSTART.md) - Getting started guide with basic commands
- [METHODOLOGY.md](docs/METHODOLOGY.md) - Theoretical foundation and I(θ) metric
- [FRAMEWORK_OVERVIEW.md](docs/FRAMEWORK_OVERVIEW.md) - System architecture and data flow
- [CONVIDE_SCENARIOS.md](docs/CONVIDE_SCENARIOS.md) - 8 engineering scenarios explained
- [SENSITIVITY_ANALYSIS_PLOTS.md](docs/SENSITIVITY_ANALYSIS_PLOTS.md) - Complete guide to analysis outputstecture and theory
- [CONVIDE_SCENARIOS.md](docs/CONVIDE_SCENARIOS.md) - Engineering scenarios
- [QUICKSTART.md](docs/QUICKSTART.md) - Getting started guide
