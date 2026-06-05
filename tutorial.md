# Surrogate Model — What's New (Handoff Tutorial)

This covers the additions to the DeepSets-style inconsistency surrogate:
new input features, a new model architecture (`ProductSetTransformer`),
the new training/evaluation workflow, and how to load a saved model and
run it on your own data.

Everything lives in the `surrogate/` package. Environment is managed with
`uv` (run commands as `uv run python -m surrogate.<script>`).

---

## 1. What changed at a glance

- **New feature set (v2):** 5 per-dimension features + 5 global features
  (`surrogate/dataset_v2.py`). Two of the globals are new "support-function"
  features that capture *how* the offset relates to zonotope shape, not just
  center distance.
- **New model:** `ProductSetTransformer` — self-attention over dimensions
  with a multiplicative (product-of-experts) aggregation that matches how
  inconsistency actually combines across dimensions.
- **New training script:** `surrogate/train_compare.py` — trains/compares
  any subset of models, supports per-dimension training, synthetic-data
  augmentation, two-phase pretrain→finetune, and leave-one-scenario-out
  cross-validation. Saves a loadable checkpoint per run under
  `results/runs/<run_id>/`.
- **Synthetic data generator:** `surrogate/generate_synthetic.py` produces
  calibrated 2D/3D/4D scenarios in the same JSON format as the MATLAB data.

The original model (`surrogate/model.py`, `DeepSetsZonotope`) is unchanged
and still uses the old v1 features.

---

## 2. The v2 feature set

Each zonotope pair is described by per-dimension features (one token per
spatial dimension, up to 4) plus global features. All features are computed
in the propagated source frame (`y = F·x + f` applied to the source), so for
identity-UPR data the source is used as-is.

**Per-dimension features (5)** — `per_dim_v2`, shape `(d, 5)`:

| # | Name        | Meaning |
|---|-------------|---------|
| 0 | `delta_c`   | center offset / σ, where σ = ‖G1_row‖ + ‖G2_row‖ |
| 1 | `r1_norm`   | source half-width / σ |
| 2 | `r2_norm`   | target half-width / σ |
| 3 | `cos`       | cosine between source & target generator rows |
| 4 | `width_ratio` | source / target half-width |

**Global features (5)** — `global_v2`, shape `(5,)`:

| # | Name             | Meaning |
|---|------------------|---------|
| 0 | `log_vol_ratio`  | log(prod r1 / prod r2) |
| 1 | `norm_center_dist` | ‖c1−c2‖ / mean(σ) |
| 2 | `dim`            | dimensionality (2, 3, or 4) |
| 3 | `sep_ratio`      | ‖offset‖ / (src_support + tgt_support along offset) **(new)** |
| 4 | `off_over_tgt`   | ‖offset‖ / tgt_support along offset, clipped to 10 **(new)** |

The two new features (`sep_ratio`, `off_over_tgt`) use support functions:
the projection of each generator column onto the offset direction. They
encode whether the offset points along a "thin" or "thick" direction of the
target — which determines inconsistency even when centers are close. This
was the key addition that let the model rank 4D scenarios correctly.

Padding/masking: scenarios with d < 4 are zero-padded to 4 dimensions and a
boolean `mask` marks the real dimensions.

---

## 3. The ProductSetTransformer

`surrogate/models_v2.py` → `ProductSetTransformer` (~9.4k params).

**Why it exists:** inconsistency is a *product* across dimensions — a point
must lie inside the target in **all** dimensions simultaneously. A plain
sum/mean pooling (DeepSets) or additive attention head doesn't match that
multiplicative structure. This model builds the product rule into the
architecture.

**Forward pass:**

1. Project each dimension's 5 features to `d_model` (default 32).
2. Self-attention block(s) over the dimension tokens — lets dimensions
   interact (important because the `correlation_strength` intervention
   couples generator columns across dimensions).
3. A per-dimension head outputs one scalar = log-containment for that dim.
4. Masked **sum** of per-dim log-containments (= log of the product).
5. Add a global bias term (an MLP on the 5 global features).
6. `sigmoid` → predicted I(θ) ∈ [0, 1].

**Variants in the registry:**

- `product_transformer` — d_model=32, 2 heads, ff=64, 1 layer (~9.4k params)
- `product_transformer_large` — d_model=64, 4 heads, ff=128, 2 layers (~69.6k)

Other models also available for comparison: `deepsets_v2`, `flat_mlp`,
`siamese`, `set_transformer`.

---

## 4. Training

All training goes through `surrogate/train_compare.py`. Each run writes to
`results/runs/<run_id>/` (auto-timestamped if `--run_id` omitted) and saves:

- `<model>.pt` — loadable checkpoint
- `run_config.json` — all args + which scenarios were held out
- `results.json` — metrics
- scatter plots + learning curves

Per-dimension validation metrics (MSE/MAE/R²/ρ for each of 2D/3D/4D) print
after every model.

**Key finding that shapes the workflow:** the dimensions interfere in a
joint model — 2D dominates the gradient and prevents 3D/4D from learning
well. So we train **one model per dimensionality**. Routing at inference is
free because you always know the dimensionality of your input.

### Common commands

Compare models on all real data (joint):
```bash
uv run python -m surrogate.train_compare --models product_transformer --epochs 100
```

Train a single dimensionality (the recommended per-dim approach):
```bash
uv run python -m surrogate.train_compare --models product_transformer --dims 4 --epochs 150 --run_id model_4d
```

Add synthetic data (filtered to match `--dims`; used for training only):
```bash
uv run python -m surrogate.train_compare --models product_transformer --dims 4 --epochs 150 --synthetic_dir data/synthetic_v6
```

### Robust evaluation: leave-one-scenario-out CV

3D/4D have very few real scenarios, so a single random holdout is
unreliable (some scenarios are near-constant I≈1.0, which makes R²/ρ
meaningless). Use LOSO-CV — every scenario is held out exactly once:

```bash
uv run python -m surrogate.train_compare --models product_transformer --dims 4 --epochs 150 --loso
```

It reports per-fold MSE/MAE (with R²/ρ suppressed for near-constant
scenarios), pooled metrics across all held-out predictions, and saves
`loso_results.json`. Synthetic data can be added to every fold's training
set with `--synthetic_dir` (synth is never used for validation).

### Useful flags

| Flag | Effect |
|------|--------|
| `--models` | which models to train (space-separated) |
| `--dims` | restrict to dimensions, e.g. `--dims 4` |
| `--epochs` | training epochs (3D/4D may need ~80–150 to converge) |
| `--synthetic_dir` | add synthetic data (training only) |
| `--n_synth_2d/3d/4d` | cap synth scenarios per dimension |
| `--loso` | leave-one-scenario-out CV |
| `--pretrain_synth` | two-phase: pretrain on synth, fine-tune on real |
| `--weight_decay` | AdamW weight decay (0 = plain Adam) |
| `--run_id` | name the output directory |

Loss is Huber (δ=0.05); optimizer is AdamW; best checkpoint is selected by
balanced per-dimension MSE.

---

## 5. Loading a saved model and evaluating elsewhere

A checkpoint is a dict (`model_name`, `state_dict`, feature dims,
`val_metrics`, `feature_order`). Use the helper:

```python
from surrogate.models_v2 import load_checkpoint

model = load_checkpoint("results/runs/model_4d/product_transformer.pt")
# returns an eval-mode nn.Module
```

### Easiest path: reuse the dataset loader

If your data is in the same `results_scenario_*.json` format as the MATLAB
pipeline, let `ZonotopeDatasetV2` compute the features for you:

```python
import torch
from torch.utils.data import DataLoader
from surrogate.dataset_v2 import ZonotopeDatasetV2, collate_fn
from surrogate.models_v2 import load_checkpoint

model = load_checkpoint("results/runs/model_4d/product_transformer.pt")

ds = ZonotopeDatasetV2(files=["path/to/results_scenario_X.json"],
                       label_key="I_theta")
loader = DataLoader(ds, batch_size=2048, collate_fn=collate_fn)

preds, labels = [], []
with torch.no_grad():
    for batch in loader:
        p = model(batch["per_dim_v2"], batch["mask"], batch["global_v2"])
        preds.append(p); labels.append(batch["label"])
preds = torch.cat(preds).numpy()
labels = torch.cat(labels).numpy()
```

The model takes exactly three tensors:
`model(per_dim_v2, mask, global_feats)` with shapes
`(B, 4, 5)`, `(B, 4)`, `(B, 5)`.

### Computing features from raw zonotopes

If you only have raw `(c1, G1, c2, G2)` arrays, call the feature builder
directly (`surrogate/dataset_v2.py` → `_compute_features`). For
identity-UPR data pass `scales=ones(d)`, `offsets=zeros(d)`, and
`_upr_onehot("identity")`. It returns a dict whose `per_dim_v2` and
`global_v2` entries are the model inputs (then pad/mask to 4 dims — see
`collate_fn` for the exact layout).

---

## 6. Data format

Scenario files are `results_scenario_*.json` with an `experiments` list.
Each experiment has:

```json
{
  "post_state": {
    "uncertainty": {
      "source_center": [...], "source_generators": [[...], ...],
      "target_center": [...], "target_generators": [[...], ...]
    },
    "inconsistency": { "I_theta": 0.87, "jaccard_index": 0.05, ... }
  }
}
```

`I_theta` is the Monte-Carlo inconsistency label. Generators are `(d, p)`
matrices (d dimensions, p generator columns). Synthetic data
(`data/synthetic_v6/`) follows the same schema and stores the already-
propagated source so UPR defaults to identity.

---

## 7. Generating synthetic data (optional)

```bash
uv run python -m surrogate.generate_synthetic --n_2d 0 --n_3d 75 --n_4d 75 \
    --n_workers 8 --n_samples 2000 --mc_samples 500 --overwrite
```

The generator is calibrated to match the real Engineering geometry
(near-diagonal generators, realistic scales, identity-dominant UPR) and
balances the I(θ) distribution. Note: synthetic data has historically
helped 3D but is harder to make useful for 4D — evaluate with LOSO before
trusting it.
