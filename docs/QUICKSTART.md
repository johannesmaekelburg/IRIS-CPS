# Quick Start Guide

## Installation

### MATLAB Setup (5 minutes)

```matlab
% Navigate to folder
cd('C:\Users\johan_rvnnln\OneDrive\Dokumente\MATLAB\MyCORA\Causality_Uncertainty_Inconsistency')

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

### 1. MATLAB Only (Quickest)

```matlab
% Create scenario
scenario = causal_experiment_engine.create_baseline_scenario(2, 1.0);

% Run intervention sweep
results = causal_experiment_engine.run_intervention_sweep(...
    scenario, 'widen', 'scale_factor', [0.5, 1.0, 1.5, 2.0, 3.0]);

% Results automatically plotted and saved
```

**Use when:** You want quick results with visualization

### 2. Python + MATLAB Engine (Most Powerful)

```python
from causal_orchestrator import CausalOrchestrator
import numpy as np

orch = CausalOrchestrator(use_engine=True)  # Requires matlab.engine

experiments = orch.design_experiment(
    intervention='widen',
    param_ranges={'scale_factor': np.linspace(0.5, 3.0, 20)},
    n_scenarios=10,
    dimensions=[2, 4, 8]
)

df = orch.run_experiment_batch(experiments)
orch.generate_causal_report(df)
```

**Use when:** You need large-scale experiments with statistical analysis

### 3. Adaptive Selection (Most Efficient)

```python
from adaptive_selector import AdaptiveSelector
from causal_orchestrator import CausalOrchestrator

orch = CausalOrchestrator(use_engine=True)
selector = AdaptiveSelector(orch)

df = selector.run_adaptive_campaign(
    n_experiments=50,
    strategy='uncertainty',
    exploration_ratio=0.3
)

selector.plot_information_gain()
```

**Use when:** You want to learn causal effects with minimal experiments

## Example: Studying Uncertainty → Emptiness

**Research Question:** At what uncertainty level do propagated sets become empty?

```matlab
% MATLAB approach
scenario = causal_experiment_engine.create_baseline_scenario(2, 1.0);
results = causal_experiment_engine.run_intervention_sweep(...
    scenario, 'widen', 'scale_factor', 0.5:0.1:3.0);
```

**Expected Finding:**
```
Scale Factor | Emptiness | Volume | Interpretation
-------------|-----------|--------|----------------
0.5-1.8      | 0         | grows  | Valid propagation
1.9-2.1      | 0→1       | →0     | Critical threshold
2.2+         | 1         | 0      | Always empty
```

**Causal Claim:** `do(widen by >2.0) → P(empty) = 1`

## Example: Studying Misalignment → Uncertainty

**Research Question:** How do mapping errors amplify uncertainty?

```python
# Python approach
orch = CausalOrchestrator(use_engine=True)

experiments = orch.design_experiment(
    intervention='misalign',
    param_ranges={'misalignment_strength': np.linspace(0, 0.5, 25)},
    n_scenarios=10,
    dimensions=[2, 4, 8]
)

df = orch.run_experiment_batch(experiments)

analysis = orch.analyze_causal_pathway(
    df,
    intervention='misalign',
    outcome_metric='effect_uncertainty_volume_ratio'
)

print(f"Linear slope: {analysis['linear_slope']:.3f}")
print(f"Nonlinearity: {analysis['nonlinearity']:.3f}")
```

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
  url = {https://github.com/DE-TUM/CPS-Uncertainty-Propagation-Framework}
}
```

---

**Questions?** See `README.md` for comprehensive documentation.
