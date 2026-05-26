# DeepSets Surrogate for Zonotope Inconsistency Estimation

A lightweight neural surrogate (~1 250 parameters) that estimates the
inconsistency score `I(theta)` between two constrained zonotopes without
running Monte-Carlo sampling.  Used to answer RQ1–RQ4 in the ICDM paper.

## Architecture

**DeepSets** over zonotope dimensions, followed by a global MLP:

```
For each dimension i:
    feats_i = [delta_c_i, ||G1_i||, ||G2_i||, cos(G1_i, G2_i)]  (normalised)
    e_i     = phi(feats_i)          # shared MLP  4 -> 16 -> 16

aggregate   = sum_i(e_i * mask_i)   # masked sum-pool
output      = sigmoid(rho([aggregate, global_feats]))  # MLP 26 -> 32 -> 1
```

`global_feats` (10-dim): log volume ratio, normalised centre distance,
8-class UPR-type one-hot encoding.

Supports zonotope dimensions d in {2, 3, 4}.  Smaller dimensions are
zero-padded with a boolean mask.

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

### Q4 counterfactual figures for specific scenarios

```bash
python -m surrogate.run_pipeline --skip_train --skip_evaluate \
    --plots q4 --cf_scenarios 35,49,11
```

Generates `results/paper_figures/Q4_counterfactual/q4_scenario_{id}.png/pdf`
for each requested scenario — a 2-panel figure with the minimal-repair
landscape and all repair trajectories coloured by dominant parameter.

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

`surrogate/model.pt` contains the pre-trained weights (8 KB).
Load it directly:

```python
import torch
from surrogate.model import DeepSetsZonotope

ckpt  = torch.load("surrogate/model.pt", map_location="cpu")
model = DeepSetsZonotope(**{k: ckpt["args"][k]
                            for k in ("phi_hidden", "rho_hidden")})
model.load_state_dict(ckpt["model_state"])
model.eval()
```

## Module overview

| File | Purpose |
|------|---------|
| `model.py` | `DeepSetsZonotope` network definition |
| `dataset.py` | `ZonotopeDataset` — reads `results_scenario_*.json` |
| `train.py` | Training loop with inductive scenario split |
| `evaluate.py` | Evaluation vs AABB and MFMC baselines |
| `counterfactual.py` | Batched gradient-based counterfactual search |
| `run_analysis.py` | All RQ1–RQ4 paper figures |
| `run_pipeline.py` | End-to-end entry point |
