# Causal Inference: Uncertainty → Inconsistency

A framework for analyzing the causal relationship between uncertainty and inconsistency in cyber-physical systems using interventional experiments.

## Purpose

This project investigates whether and how uncertainty causes inconsistency in constrained polynomial zonotope propagation. Using Pearl's do-calculus and five intervention types (widen, shrink, shift, rotate, correlate), the framework quantifies causal effects via the global inconsistency metric I(θ). Applications include robustness analysis, sensitivity analysis, and identifying critical parameters in engineering systems.

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
# Analyze 2D data (4 scenarios)
python src/sensitivity_analysis.py --data_dir data/convide_with_I_theta --output_dir results/sensitivity_2d --param param_value

# Analyze 3D data (4 scenarios)
python src/sensitivity_analysis.py --data_dir data/convide_balanced --output_dir results/sensitivity_3d --param param_value
```

**Available Datasets:**
- `convide_with_I_theta/`: 2D scenarios (1-4) with I_theta metrics
- `convide_balanced/`: 3D scenarios (5-8) with I_theta metrics

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
