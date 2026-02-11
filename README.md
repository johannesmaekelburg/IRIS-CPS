# Causal Inference: Uncertainty → Inconsistency

A framework for analyzing the causal relationship between uncertainty and inconsistency in cyber-physical systems using interventional experiments.

## Purpose

This project investigates whether and how uncertainty causes inconsistency in constrained polynomial zonotope propagation. Using Pearl's do-calculus and intervention types (widen, shrink, correlate), the framework quantifies causal effects via the global inconsistency metric I(θ). 

**Current Features:**
- 8 CONVIDE scenarios (4×2D, 4×3D) with engineering applications
- Global inconsistency metric I(θ) with Monte Carlo estimation
- Causal effects, local sensitivity, robustness margins
- Sobol variance-based sensitivity indices
- Gaussian Process surrogate models for fast predictions

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
