# Estimating Inconsistency Response Surfaces under Uncertainty in Cyber-Physical System Development

Code accompanying the paper *"Estimating Inconsistency Response Surfaces under Uncertainty in Cyber-Physical System Development"*.

Zonotope-based reachability tools can fail silently when input uncertainty grows too large. This framework maps that failure causally: using Pearl's do-calculus and three intervention types (widen, shrink, correlate), it quantifies how uncertainty causes inconsistency in constrained polynomial zonotope propagation via the global inconsistency metric I(θ), and learns a surrogate response surface over the parameter space.

## Features

- CONVIDE scenarios (2D, 3D, 4D) covering automotive, aerospace, robotics, and other CPS domains
- Global inconsistency metric I(θ) with Monte Carlo estimation
- Causal effects, local sensitivity, and robustness margins
- Sobol variance-based sensitivity indices
- DeepSets surrogate model for fast I(θ) prediction
- Pre-trained models for 2D/3D and 4D scenarios (no retraining needed for evaluation)

## Repository Structure

```
src/
  causal_engine.py              # Python reimplementation of the MATLAB experiment engine
  run_combined_figures.py       # Generate combined paper figures (2D/3D + 4D models)
  matlab/
    causal_experiment_engine_twostep.m   # Core MATLAB experiment engine
    global_inconsistency.m               # I(θ) computation
surrogate/                      # DeepSets surrogate model code
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

**[https://doi.org/10.6084/m9.figshare.32407989](https://doi.org/10.6084/m9.figshare.32407989)**

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
- CPS-Uncertainty-Propagation-Framework *(link anonymized for review)*

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

# Train on measurements_v6 + measurements_cps_v6
python -m surrogate.train

# Evaluate against AABB, MC, and MFMC baselines
python -m surrogate.evaluate

# Or run the full pipeline (train → evaluate → plot) in one command
python -m surrogate.run_pipeline
```

Training uses `data/measurements_v6/` and `data/measurements_cps_v6/` by default.
The trained checkpoint is saved to `surrogate/model.pt`.

### Step 3 — Python: Generate Combined Figures

```bash
python src/run_combined_figures.py \
    --model-2d3d product_transformer_2d3d/product_transformer_exact.pt \
    --model-4d   product_transformer_4d_real/product_transformer_exact.pt \
    --scenarios-2d3d 69 32 36 3 25 42 1 57 62 51 58 7 \
    --scenarios-4d   9 202 216 232 251 256 291 296 304 315 317 \
    --output results/combined
```

For a quick smoke test:

```bash
python src/run_combined_figures.py \
    --model-2d3d product_transformer_2d3d/product_transformer_exact.pt \
    --model-4d   product_transformer_4d_real/product_transformer_exact.pt \
    --max-samples 20000 --max-scenarios 5
```

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
