# Neural Surrogate for Zonotope Inconsistency Estimation

A lightweight neural surrogate that estimates the inconsistency score
`I(theta)` between two constrained zonotopes without running Monte-Carlo
sampling.  Used to answer RQ1–RQ4 in the ICDM paper.

## Architectures

Several interchangeable architectures are provided in `models_v2.py`
(`MODEL_REGISTRY`); the deployed model is `product_transformer_exact`.
Select one or more with `--models` and they train and evaluate on the same
data split:

```bash
python -m surrogate.train_compare --models product_transformer_exact
python -m surrogate.train_compare --models product_transformer_exact set_transformer deepsets_v2 flat_mlp
```

All variants take the same per-dimension + global feature input, support
zonotope dimensions d in {2, 3, 4} (smaller dimensions are zero-padded with a
boolean mask), and output `I(theta)` in [0, 1].

## Requirements

```
pip install -r requirements.txt   # torch, numpy, scipy, matplotlib, SALib
```

Python 3.9+, PyTorch 2.x.  No GPU required (CPU inference ~2 µs/sample).

## Data

Raw training data (~8 GB) is hosted on figshare: https://figshare.com/s/75b399b32008c72925d5

Download and place under:
```
data/measurements_v6/          # ConVIDe scenarios   (12 JSON files)
data/measurements_cps_v6/      # CPS scenarios       (72 JSON files)
```

## Usage

### Full pipeline (train + evaluate + figures)

```bash
python -m surrogate.run_pipeline
```

### Individual stages

```bash
# Train only
python -m surrogate.train

# Evaluate against AABB and MFMC baselines
python -m surrogate.evaluate

# Generate all paper figures (RQ1-RQ4)
python -m surrogate.run_analysis
```

### Common options

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 20 | Training epochs |
| `--lr` | 1e-3 | Learning rate |
| `--max_samples` | None | Cap dataset size for quick tests |
| `--output` | `results/paper_figures` | Output directory |
| `--plots` | all | Comma-separated plot names or `q1`/`q2`/`q3`/`q4` |
| `--gamma` | 0.5 | Inconsistency threshold |

### Q4 consistency-recourse figures for specific scenarios

```bash
python -m surrogate.run_pipeline --skip_train --skip_evaluate \
    --plots q4 --cf_scenarios 35,49,11
```

Generates `results/paper_figures/Q4_counterfactual/q4_scenario_{id}.png/pdf`
for each requested scenario — a 2-panel figure with the minimal-recourse
landscape and all recourse trajectories coloured by dominant parameter.

### Skip training (use existing checkpoint)

```bash
python -m surrogate.run_pipeline --skip_train --plots q1,q2
```

## Output structure

```
results/paper_figures/
├── Q1_estimation/
│   ├── acc_scatter.png/pdf
│   ├── per_domain_table.png/pdf
│   ├── threshold_by_domain.png/pdf
│   ├── efficiency.png/pdf
│   └── pareto.png/pdf
├── Q2_surrogate/
│   ├── acc_by_dim.png/pdf
│   ├── accuracy_by_domain.png/pdf
│   ├── error_by_regime.png/pdf
│   ├── exploration_budget.png/pdf
│   └── consistency_rate.png/pdf
├── Q3_sensitivity/
│   ├── sensitivity_surrogate.png/pdf
│   ├── sensitivity_compare.png/pdf
│   ├── sobol_by_domain.png/pdf
│   └── response_surfaces.png/pdf
└── Q4_counterfactual/
    ├── q4_scenario_{id}.png/pdf   # one per scenario
    ├── q4_overview.png/pdf
    └── q4_fix_direction.png/pdf
```

## Checkpoint

Load any trained checkpoint with `load_checkpoint`, which reconstructs the
correct architecture from the saved `model_name`:

```python
from surrogate.models_v2 import load_checkpoint

model = load_checkpoint(
    "results/runs/joint_2d3d_exact/product_transformer_exact.pt")
# eval-mode model; call model(per_dim, mask, global_feats)
```

## Module overview

| File | Purpose |
|------|---------|
| `models_v2.py` | Surrogate architectures + `MODEL_REGISTRY`, `load_checkpoint` |
| `dataset_v2.py` | `ZonotopeDatasetV2` — reads `results_scenario_*.json` |
| `train_compare.py` | Train/compare architectures on a shared data split |
| `evaluate.py` | Evaluation vs AABB, MC, and MFMC baselines |
| `counterfactual.py` | Batched gradient-based consistency recourse |
| `repair_benchmark.py` | Consistency-recourse benchmark (surrogate vs. CMA-ES/FD) |
| `plot_repair.py` | Recourse figures (Pareto, trajectory) |
| `run_analysis.py` | All RQ1–RQ4 paper figures |
| `run_pipeline.py` | End-to-end entry point |
