# Estimating Inconsistency Response Surfaces under Uncertainty in Cyber-Physical System Development

Code accompanying the paper *"Estimating Inconsistency Response Surfaces under Uncertainty in Cyber-Physical System Development"*.

Zonotope-based reachability tools can fail silently when input uncertainty grows too large. This framework reformulates that failure as an intervention-response surface: using three intervention types (scale, center-shift, correlate) over the uncertainty geometry, it quantifies how uncertainty drives inconsistency in constrained zonotope propagation via the global inconsistency metric I(θ), learns a fast neural surrogate of the response surface, and uses it for sensitivity analysis and consistency recourse.

## Features

- CONVIDE scenarios (2D, 3D, 4D) covering automotive, aerospace, robotics, and other CPS domains
- Global inconsistency metric I(θ) with Monte Carlo and multi-fidelity (MFMC) estimation
- Sobol variance-based sensitivity analysis
- Neural surrogate for fast I(θ) prediction (several interchangeable architectures)
- Gradient-based consistency recourse: the minimal uncertainty intervention that restores consistency
- Pre-trained models for 2D/3D and 4D scenarios (no retraining needed for evaluation)

## Repository Structure

```
src/
  causal_engine.py              # Python reimplementation of the MATLAB experiment engine
  run_combined_figures.py       # Generate combined paper figures (2D/3D + 4D models)
  matlab/
    causal_experiment_engine_twostep.m   # Core MATLAB experiment engine
    global_inconsistency.m               # I(θ) computation
surrogate/                      # Neural surrogate: training, evaluation, recourse
examples/                       # MATLAB data generation scripts
data/
  measurements_v6/              # CONVIDE training measurements
  measurements_cps_v6/          # CPS domain training measurements
  measurements_cps_full/        # Full CPS measurements
  CPS-uncertainty-dataset/      # CPS uncertainty scenario definitions
  CPS-uncertainty-dataset-full/
  synthetic_v6/                 # Synthetic training data
  zonotopes/                    # Zonotope geometry data
product_transformer_2d3d/       # Pre-trained surrogate (2D + 3D scenarios)
product_transformer_4d_real/    # Pre-trained surrogate (4D scenarios)
results/
  combined/                     # Final paper tables and figures
  eval_2d3d/                    # 2D/3D model evaluation results
  eval_4d/                      # 4D model evaluation results
docs/                           # Extended documentation
tests/
```

## Data

All datasets are hosted on Figshare and are not tracked in this repository:

**[Datasets](https://figshare.com/s/75b399b32008c72925d5)**

Download and extract the archive into the `data/` directory before running any scripts. The expected folder structure is described in [FRAMEWORK_OVERVIEW.md](docs/FRAMEWORK_OVERVIEW.md).

## Usage

### Step 1 — MATLAB: Generate Data

From the MATLAB root, run all scenarios in one call:

```matlab
run_all_scenarios
```

Or generate individual scenario sets:

```matlab
generate_convide_twostep          % CONVIDE 2D / 3D / 4D scenarios
generate_cps_domains_twostep      % CPS engineering domain scenarios
generate_synthetic_v6_twostep     % Synthetic data
```

These scripts depend on:
- [CORA Toolbox](https://tumcps.github.io/CORA/)
- CPS-Uncertainty-Propagation-Framework

The core MATLAB engine lives in `src/matlab/` and is added to the path automatically by the example scripts.

### Step 2 — Python: Surrogate Model

**Option A — Use pre-trained models (recommended for evaluation)**

Pre-trained checkpoints are included in the repo:
- `product_transformer_2d3d/product_transformer_exact.pt` — 2D + 3D scenarios
- `product_transformer_4d_real/product_transformer_exact.pt` — 4D scenarios

Skip to Step 3 to generate figures directly from these checkpoints.

**Option B — Train from scratch**

```bash
# Install dependencies
pip install -r requirements.txt

# Train the deployed models
python -m surrogate.train_compare --models product_transformer_exact --dims 2 3 \
    --epochs 400 --synthetic_dir data/synthetic_v6 --n_synth_3d 10 \
    --holdout measurements_v6/results_scenario_7.json --run_id joint_2d3d_exact
python -m surrogate.train_compare --models product_transformer_exact --dims 4 \
    --epochs 250 --run_id final_4d_real_exact

# Evaluate against AABB, MC, and MFMC baselines
python -m surrogate.evaluate
```

Several surrogate architectures are interchangeable: pass one or more registry
names (defined in `surrogate/models_v2.py`, `MODEL_REGISTRY`) to `--models` to
train and compare them on the same data split, e.g.
`--models product_transformer_exact set_transformer deepsets_v2 flat_mlp`.
Checkpoints are written to `results/runs/<run_id>/<model>.pt`.

#### Architecture ablation

Ablating the components of the Product-Set Transformer on the 2D/3D split
(12 held-out scenarios incl. the 3D holdout, 200 epochs). The full model is the
most accurate and the most robust on the harder, data-scarce 3D split:

| Variant | Ablation vs. full model | Params | MSE | R² | ρ | R² (3D) |
|---|---|---:|---:|---:|---:|---:|
| `product_transformer_exact` | — (full: attention + product head) | 9,394 | **0.00014** | **0.9947** | **0.9963** | **0.9887** |
| `set_transformer` | standard pooled readout (no product head) | 10,609 | 0.00033 | 0.9875 | 0.9926 | 0.9300 |
| `deepsets_v2` | no attention + standard readout | 8,897 | 0.00037 | 0.9859 | 0.9910 | 0.9716 |
| `pst_exact_no_attn` | no self-attention (product head kept) | 850 | 0.00039 | 0.9849 | 0.9918 | 0.9793 |
| `flat_mlp` | no set structure (flattened MLP) | 7,937 | 0.00067 | 0.9742 | 0.9857 | 0.7812 |

Removing the set structure entirely (`flat_mlp`) collapses on 3D (R²=0.78),
and replacing the product/noisy-AND head with a standard pooled readout
(`set_transformer`) degrades 3D markedly (R²=0.93). Dropping only attention
(`pst_exact_no_attn`) retains strong accuracy at <1k parameters, indicating the
product head carries most of the architecture's inductive bias.

To reproduce this run:

```bash
python -m surrogate.train_compare \
    --models product_transformer_exact set_transformer deepsets_v2 pst_exact_no_attn flat_mlp \
    --dims 2 3 --epochs 200 --synthetic_dir data/synthetic_v6 --n_synth_3d 10 \
    --holdout measurements_v6/results_scenario_7.json --run_id ablations_2d3d
```

### Step 3 — Python: Generate Combined Figures

```bash
python src/run_combined_figures.py \
    --model-2d3d product_transformer_2d3d/product_transformer_exact.pt \
    --model-4d   product_transformer_4d_real/product_transformer_exact.pt \
    --scenarios-2d3d 69 32 36 3 25 42 1 57 62 51 58 7 \
    --scenarios-4d   9 202 216 232 251 256 291 296 304 315 317 \
    --output results/combined
```


### Step 4 — Python: Consistency Recourse

Given inconsistent configurations, search for the minimal uncertainty intervention
that restores consistency (I(θ) ≤ γ). The differentiable surrogate solves this by
projected gradient descent with zero expensive oracle calls; it is benchmarked
against derivative-free baselines (CMA-ES, finite-difference) driven by MC/MFMC
oracles.

```bash
# Run the recourse benchmark → writes results/repair_benchmark/records.json
python -m surrogate.repair_benchmark \
    --model_2d3d product_transformer_2d3d/product_transformer_exact.pt \
    --model_4d   product_transformer_4d_real/product_transformer_exact.pt \
    --out results/repair_benchmark

# Plot the recourse results (walltime vs. success rate and intervention distance)
python -m surrogate.plot_repair pareto_combined \
    --records results/repair_benchmark/records.json \
    --style dual --yaxis success --out results/repair_fig
```

Pass `--objective consistency` to instead drive I(θ) as low as possible
(unconstrained variant) rather than the minimal-distance recourse.

## Requirements

**MATLAB (R2021a+)**:
- [CORA Toolbox](https://tumcps.github.io/CORA/)
- CPS-Uncertainty-Propagation-Framework *(link anonymized for review)*

**Python (3.8+)**:
```bash
pip install -r requirements.txt
```

Main packages: `numpy`, `scipy`, `torch`, `matplotlib`, `pandas`, `scikit-learn`

## Documentation

- [QUICKSTART.md](docs/QUICKSTART.md) - Getting started guide
- [METHODOLOGY.md](docs/METHODOLOGY.md) - Theoretical foundation and I(θ) metric
- [FRAMEWORK_OVERVIEW.md](docs/FRAMEWORK_OVERVIEW.md) - System architecture and data flow
- [CONVIDE_SCENARIOS.md](docs/CONVIDE_SCENARIOS.md) - Engineering scenarios explained
