# Quick Start Guide

## Installation

### MATLAB Setup (5 minutes)

```matlab
% Navigate to folder
cd('/path/to/repo')

% Run setup
setup_causal_framework

% Test with example
example_causal_experiment
```

**Expected output:** 6-panel visualization showing uncertainty → inconsistency effects

### Python Setup (5 minutes)

```bash
# Install dependencies
pip install -r requirements.txt

# Test installation
python setup_and_test.py

# Run example (without MATLAB Engine)
python causal_orchestrator.py
```

## File Overview

| File | Purpose | Language |
|------|---------|----------|
| `causal_experiment_engine.m` | Core intervention engine | MATLAB |
| `causal_orchestrator.py` | Experiment orchestration | Python |
| `adaptive_selector.py` | Smart intervention selection | Python |
| `example_causal_experiment.m` | Standalone demo | MATLAB |
| `setup_causal_framework.m` | MATLAB setup | MATLAB |
| `setup_and_test.py` | Python setup/test | Python |

## Three Ways to Use

### 1. MATLAB: Generate Engineering Data

```matlab
% Navigate to examples folder
cd examples

% Generate Engineering scenarios with I_theta metric
generate_engineering_examples

% This creates:
% - data/engineering_with_I_theta/ (2D scenarios 1-4)
% - data/engineering_balanced/ (3D scenarios 5-8)
```

**Use when:** You need to generate experimental data with interventions

### 2. Python: Sensitivity Analysis (Primary Method)

```bash
# From project root (adjust paths if in src/)
cd Causality_Uncertainty_Inconsistency

# Analyze 2D data
python src/sensitivity_analysis.py --data_dir data/engineering_with_I_theta --output_dir results/sensitivity_2d --param param_value

# Analyze 3D data  
python src/sensitivity_analysis.py --data_dir data/engineering_balanced --output_dir results/sensitivity_3d --param param_value
```

**Outputs:**
- Sensitivity plots showing I_theta vs parameter values
- Statistical analysis of causal effects
- Delta metrics (delta_I_theta, delta_jaccard, delta_volume)

**Use when:** You have generated Engineering data and want to analyze causality

### 3. Data Inspection and Visualization

```python
import json
import matplotlib.pyplot as plt

# Load data
with open('data/engineering_with_I_theta/results_engineering_2d_scenario_1.json') as f:
    data = json.load(f)

# Extract metrics
param_values = [exp['param_value'] for exp in data['experiments']]
I_theta = [exp['pre_inconsistency']['I_theta'] for exp in data['experiments']]
jaccard = [exp['pre_inconsistency']['jaccard_index'] for exp in data['experiments']]

# Quick plot
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(param_values, I_theta, 'o-')
plt.xlabel('Parameter Value')
plt.ylabel('I_theta')
plt.title('Identity Causality')

plt.subplot(1, 2, 2)
plt.plot(param_values, jaccard, 's-')
plt.xlabel('Parameter Value')
plt.ylabel('Jaccard Index')
plt.title('Set Consistency')
plt.tight_layout()
plt.show()
```

**Use when:** You want to quickly inspect and visualize results

## Example: Studying I_theta Sensitivity

**Research Question:** How does uncertainty magnitude affect identity causality (I_theta)?

```bash
# Run sensitivity analysis on 2D scenario 1 (CAD Export Drift)
python src/sensitivity_analysis.py --data_dir data/engineering_with_I_theta --output_dir results/sensitivity_2d --param param_value
```

**Expected Finding:**
```
Param Value | I_theta  | Jaccard  | Interpretation
------------|----------|----------|----------------
0.5         | 0.98     | 0.17     | High inconsistency
1.0         | 0.95     | 0.25     | Moderate inconsistency  
1.5         | 0.90     | 0.35     | Improving consistency
2.0+        | 0.85     | 0.45     | Better consistency
```

**Causal Claim:** `do(increase_uncertainty) → ΔI_theta < 0` (uncertainty reduction improves identity consistency)

## Example: Comparing 2D vs 3D Causality

**Research Question:** Does dimensionality affect causal relationships?

```bash
# Analyze both datasets
python src/sensitivity_analysis.py --data_dir data/engineering_with_I_theta --output_dir results/sensitivity_2d --param param_value
python src/sensitivity_analysis.py --data_dir data/engineering_balanced --output_dir results/sensitivity_3d --param param_value

# Compare results
python -c "
import json
import numpy as np

# Load 2D results
with open('data/engineering_with_I_theta/results_engineering_2d_scenario_1.json') as f:
    data_2d = json.load(f)
I_theta_2d = [exp['pre_inconsistency']['I_theta'] for exp in data_2d['experiments']]

# Load 3D results
with open('data/engineering_balanced/results_engineering_3d_scenario_5.json') as f:
    data_3d = json.load(f)
I_theta_3d = [exp['pre_inconsistency']['I_theta'] for exp in data_3d['experiments']]

print(f'2D I_theta mean: {np.mean(I_theta_2d):.3f}')
print(f'3D I_theta mean: {np.mean(I_theta_3d):.3f}')
"
```

**Use case:** Understanding how geometric complexity affects causality propagation

**Expected Finding:**
- Linear slope ≈ 6.4 (misalignment amplifies volume 6.4× per unit error)
- Nonlinearity > 0.15 (quadratic amplification at high errors)

**Causal Claim:** `do(misalign by 10%) → E[volume] ≈ 3.2× baseline`

## Troubleshooting

### "CORA not found"
```matlab
% Add CORA to path
addpath(genpath('/path/to/CORA'));
savepath;
```

### "MATLAB Engine API not installed"
```bash
# Find MATLAB root
matlab -batch "disp(matlabroot)"

# Navigate to engine
cd "<matlabroot>/extern/engines/python"

# Install
python setup.py install
```

### "Empty set detection fails"
The framework uses multiple emptiness checks:
1. `representsa(Z, 'emptySet')`
2. Interval bounds check
3. Constraint satisfiability

If issues persist, increase tolerance in `isEmptySet.m`

## Output Files

After running experiments:

```
causal_inference_results/
├── data/
│   ├── results_20251216_143022.csv      # Full experimental data
│   └── adaptive_results.csv             # Adaptive campaign results
├── figures/
│   ├── widen_effect_*.png               # Causal pathway plots
│   └── information_gain_plot.png        # Learning efficiency
└── causal_report.md                     # Comprehensive analysis
```

## Next Steps

1. **Run example:** `example_causal_experiment` to verify installation
2. **Explore interventions:** Try all 10 intervention types
3. **Vary dimensions:** Test scalability with D=2,4,8,16
4. **Adaptive learning:** Use `adaptive_selector.py` for efficiency
5. **Custom scenarios:** Modify `create_baseline_scenario()` parameters

## Key Concepts

**Intervention (do() operator):** Explicit manipulation of system properties
- `do(widen)`: Multiply generator matrix by scale factor
- `do(misalign)`: Add error to transformation matrix

**Causal Effect:** Change in outcome due to intervention
- `Δ_emptiness = P(empty|do(widen=2.0)) - P(empty|do(widen=1.0))`

**Dose-Response:** Outcome as function of intervention strength
- Linear: `outcome = β₀ + β₁·dose`
- Nonlinear: Captured by Random Forest or GP

**Adaptive Selection:** Choose next experiment to maximize learning
- Uncertainty sampling: Pick where model is uncertain
- Expected improvement: Pick where outcome might exceed current best

## Citation

```bibtex
@software{causal_uncertainty_inconsistency,
  title = {Causal Inference Framework for Uncertainty-Inconsistency Analysis},
  year = {2025},
  url = {[anonymized for review]}
}
```

---

**Questions?** See `README.md` for comprehensive documentation.
