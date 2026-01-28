# Causal Inference: Uncertainty → Inconsistency

A framework for analyzing the causal relationship between uncertainty and inconsistency in cyber-physical systems using interventional experiments.

## Purpose

This project investigates whether and how uncertainty causes inconsistency in constrained polynomial zonotope propagation. Using Pearl's do-calculus and five intervention types (widen, shrink, shift, rotate, correlate), the framework quantifies causal effects via the global inconsistency metric I(θ). Applications include robustness analysis, sensitivity analysis, and identifying critical parameters in engineering systems.

## Usage

```matlab
% Navigate to examples folder
cd examples

% Generate CONVIDE scenarios with I(θ) metric
generate_convide_examples

% Run sensitivity analysis (Python)
% python src/sensitivity_analysis.py --data_dir ./data/convide_with_I_theta --output_dir ./figures
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


## Documentation

Detailed documentation available in `docs/`:
- [FRAMEWORK_OVERVIEW.md](docs/FRAMEWORK_OVERVIEW.md) - Architecture and theory
- [CONVIDE_SCENARIOS.md](docs/CONVIDE_SCENARIOS.md) - Engineering scenarios
- [QUICKSTART.md](docs/QUICKSTART.md) - Getting started guide
