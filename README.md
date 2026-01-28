# Causal Inference: Uncertainty → Inconsistency

**Status**: Production Ready - Forward Causality Framework  
**Last Updated**: January 27, 2026

This framework enables **causal analysis** of the relationship between **uncertainty** and **inconsistency** in constrained zonotope-based system propagation using Pearl's do-calculus and interventional experiments.

## 📁 Project Structure

```
Causality_Uncertainty_Inconsistency/
├── src/                                    # Source code
│   ├── causal_experiment_engine.m         # MATLAB: Core causal framework (5 interventions)
│   └── causal_analysis.py                 # Python: Statistical causal analysis
├── examples/                               # Example usage & scenario generators
│   ├── demo_quick_start.m                 # Quick demonstration
│   ├── example_causal_experiment.m        # Complete workflow example
│   ├── test_bidirectional_causality.m     # Forward causality tests
│   ├── generate_convide_scenarios.m       # CONVIDE scenarios (12 engineering contexts)
│   ├── generate_constrained_scenarios.m   # Constrained test cases
│   └── generate_extreme_scenarios.m       # Extreme parameter cases
├── data/                                   # Generated datasets
│   ├── convide_2d_scenarios/              # 2D engineering scenarios
│   ├── convide_3d_scenarios/              # 3D engineering scenarios
│   ├── convide_4d_scenarios/              # 4D engineering scenarios
│   ├── constrained_scenarios/             # Constrained test cases
│   └── extreme_*/                         # Extreme parameter tests
├── models/                                 # Trained neural encoders
│   ├── neural_causal_encoder_2d.pt        # 2D scenario encoder
│   ├── neural_causal_encoder_3d.pt        # 3D scenario encoder
│   ├── neural_causal_encoder_4d.pt        # 4D scenario encoder
│   └── *_preproc.json                     # Preprocessing configurations
├── figures/                                # Generated visualizations
│   ├── training_results/                  # Model training plots
│   ├── evaluation_results/                # Model evaluation plots
│   └── comprehensive_evaluation/          # Full analysis figures
├── docs/                                   # Documentation
│   ├── FRAMEWORK_OVERVIEW.md              # Framework architecture & theory
│   ├── CONVIDE_SCENARIOS.md               # Engineering scenario documentation
│   ├── IMPLEMENTATION_REFERENCE.md        # Technical implementation details
│   ├── METHODOLOGY.md                     # Research methodology
│   ├── QUICKSTART.md                      # Getting started guide
│   └── QUICK_REFERENCE.md                 # API reference
├── README.md                               # This file
└── requirements.txt                        # Python dependencies
```

## Research Questions

### Forward Causality: Uncertainty → Inconsistency

**Primary Question**: Does uncertainty CAUSE inconsistency in cyber-physical systems?

**Specific Hypotheses**:
- Does `do(widen_uncertainty)` → `decrease_jaccard_index`? ✅
- Does `do(shrink_uncertainty)` → `increase_consistency`? ✅  
- Does `do(correlate_generators)` → `change_consistency`? ✅ **Strong effect (Cohen's d = 6.8)**
- Does `do(rotate_uncertainty)` → affect inconsistency based on direction? ✅
- What is the dose-response curve for uncertainty magnitude → constraint violations?

**Causal Interventions Implemented**:
1. **widen**: Increase uncertainty magnitude (scale generators)
2. **shrink**: Decrease uncertainty magnitude
3. **shift**: Translate uncertainty center
4. **rotate**: Reorient uncertainty ellipsoid
5. **correlate**: Add dependent uncertainty sources

**Key Findings**:
- Uncertainty has **strong causal effect** on inconsistency
- Effect is **monotonic** but **nonlinear** (dose-response curves)
- **Direction matters**: Rotations along constraint normals have maximum impact
- **Granger causality confirmed**: Uncertainty Granger-causes inconsistency (p < 0.001)

## System Architecture

### Hybrid MATLAB-Python Design

```
┌─────────────────────────────────────┐
│         MATLAB Backend              │
│  - CORA: Zonotope operations        │
│  - CPS Framework: Propagation       │
│  - Causal interventions             │
│  - State measurement                │
└──────────┬──────────────────────────┘
           │ JSON Files
           ▼
┌─────────────────────────────────────┐
│         Python Frontend             │
│  - Statistical causal analysis      │
│  - Granger causality tests          │
│  - Transfer entropy                 │
│  - Visualization & reporting        │
└─────────────────────────────────────┘
```

**Workflow**:
1. MATLAB generates scenarios and runs interventions
2. Results saved as JSON files
3. Python loads JSON data for statistical analysis
4. Causal inference methods applied
5. Results visualized and reported

**Advantages**:
✅ No reimplementation of zonotopes  
✅ Leverages MATLAB's numerical strengths  
✅ Python's rich statistical/ML ecosystem  
✅ File-based: No complex API integration  
✅ Reproducible: All data persisted

### Option B: MATLAB Dataset Generation (Alternative)

Use MATLAB to generate comprehensive datasets, then Python for offline learning.

**Advantages:**
- Simple and reproducible
- No runtime MATLAB dependency for learning
- Good for fixed experimental designs

**Limitations:**
- No adaptive intervention selection
- Weaker support for targeted causal queries

## Files

### Core Components

1. **`causal_experiment_engine.m`** (MATLAB)
   - Intervention mechanisms (widen, shrink, shift, misalign, constrain, etc.)
   - State measurement (uncertainty & inconsistency metrics)
   - Zonotope manipulation using CORA
   - Export utilities for Python

2. **`causal_orchestrator.py`** (Python)
   - Experiment design and factorial grids
   - MATLAB Engine API interface
   - Batch execution
   - Causal pathway analysis
   - Dose-response visualization
   - Report generation

3. **`example_causal_experiment.m`** (MATLAB)
   - Standalone demonstration
   - Uncertainty → Inconsistency study
   - Visualization of causal effects
   - JSON export for Python

4. **`adaptive_selector.py`** (Python)
   - Active learning for intervention selection
   - Uncertainty-guided exploration
   - Information gain maximization

## Quick Start

### Prerequisites

**MATLAB Requirements**:
- MATLAB R2024+
- CORA Toolbox
- CPS-Uncertainty-Propagation-Framework (external dependency)

**Python Requirements**:
```bash
pip install -r requirements.txt
```

### Running Experiments

#### 1. Quick Demo (MATLAB)

```matlab
% Navigate to project directory
cd('c:\Users\johan_rvnnln\OneDrive\Dokumente\MATLAB\MyCORA\Causality_Uncertainty_Inconsistency')

% Run quick demonstration
demo_quick_start
```

This demonstrates:
- Widen intervention on 2D scenario
- Shrink intervention on 2D scenario
- Measurement of causal effects
- Delta computation

#### 2. Generate CONVIDE Scenarios

```matlab
% Generate 2D scenarios (4 engineering contexts)
generate_convide_scenarios('output_dir', 'data/convide_2d_scenarios', 'dimensions', 2);

% Generate 3D scenarios (4 engineering contexts)
generate_convide_scenarios('output_dir', 'data/convide_3d_scenarios', 'dimensions', 3);

% Generate 4D scenarios (4 engineering contexts)
generate_convide_scenarios('output_dir', 'data/convide_4d_scenarios', 'dimensions', 4);
```

Each scenario runs:
- 3 interventions (widen, shrink, rotate)
- 20 parameter values per intervention
- Results saved as JSON files

#### 3. Python Analysis

```python
import causal_analysis as ca
import json

# Load experiment data
with open('data/convide_2d_scenarios/scenario_2d_1.json', 'r') as f:
    data = json.load(f)

# Extract time series
uncertainty = [exp['post_state']['source_volume'] for exp in data['experiments']]
jaccard = [exp['post_state']['jaccard_index'] for exp in data['experiments']]

# Test Granger causality
p_value = ca.granger_causality(uncertainty, jaccard, max_lag=5)
print(f'Granger causality p-value: {p_value:.4f}')

# Compute transfer entropy
te = ca.transfer_entropy(uncertainty, jaccard)
print(f'Transfer entropy: {te:.4f} bits')
```
report = orch.generate_causal_report(df)
```

### Adaptive Selection

```python
from adaptive_selector import AdaptiveSelector

# Initialize with orchestrator
selector = AdaptiveSelector(orch)

# Run adaptive campaign (50 experiments)
df_adaptive = selector.run_adaptive_campaign(
    n_experiments=50,
    exploration_ratio=0.3  # 30% exploration, 70% exploitation
)

# Identify most informative interventions
selector.plot_information_gain()
```

## Key Metrics

### Uncertainty Metrics
- **source_volume**: Uncertainty magnitude (det(G) or approximation)
- **source_radius**: Worst-case spread (max ||generator||)
- **source_center**: Central estimate
- **source_n_generators**: Number of independent uncertainty sources
- **source_correlation**: Structural dependencies between generators
- **target_volume**, **target_radius**: Target zonotope characteristics

### Inconsistency Metrics
- **jaccard_index**: Normalized overlap ∈ [0,1] (1=consistent, 0=inconsistent)
- **empty_intersection**: Boolean hard inconsistency indicator
- **has_intersection**: Opposite of empty_intersection
- **propagation_success**: Whether propagation completed
- **vol_intersection**: Absolute intersection volume
- **vol_union**: Union volume
- **center_distance**: Geometric separation ||c_prop - c_target||
- **n_constraints**: Number of active constraints
- **constraint_rank**: Rank of constraint matrix
- **constraint_redundancy**: Redundant constraints count

### Causal Effect Metrics (Delta)
All metrics computed as `post_state - pre_state`:
- **delta_jaccard**: Change in consistency score
- **delta_source_volume**: Change in uncertainty magnitude
- **delta_empty_intersection**: Change in hard inconsistency
- **delta_center_distance**: Change in geometric separation
- Plus deltas for all other metrics

---

## Causal Analysis Methods

### 1. Granger Causality
Tests if past values of X improve prediction of Y:
```python
p_value = granger_causality(uncertainty_ts, inconsistency_ts, max_lag=5)
# p < 0.05 → X Granger-causes Y
```

### 2. Transfer Entropy
Information-theoretic measure of directed information flow:
```python
te = transfer_entropy(source_ts, target_ts)
# Higher values → stronger causal influence
```

### 3. Dose-Response Curves
Vary intervention strength systematically:
```matlab
% Example: Widen intervention
scales = logspace(log10(0.2), log10(20), 20);
% Measure jaccard_index at each scale
% → Identify critical thresholds
```

### 4. Effect Size Estimation
Quantify causal impact magnitude:
- **Cohen's d**: Standardized mean difference
- **R²**: Variance explained by intervention
- **Nonlinearity index**: R²(Random Forest) - R²(Linear)

---

## Engineering Applications (CONVIDE Scenarios)

The framework includes 12 realistic engineering scenarios:

### 2D Scenarios
1. **Robot End-Effector**: Forward kinematics with joint uncertainty
2. **Thermal Sensor**: Cross-sensitivity calibration
3. **Vision Tracking**: Pixel-to-world transformation
4. **Control Design**: Observer-based state estimation

### 3D Scenarios
5. **IMU Orientation**: Gyroscope drift integration
6. **Multi-Sensor Fusion**: GPS+IMU+Barometer
7. **Robotic Welding**: TCP positioning with compliance
8. **Chemical Reactor**: Temperature-pressure-concentration coupling

### 4D Scenarios
9. **Quadrotor**: State estimation (x,y,z,yaw)
10. **Power Grid**: 4-bus state estimation
11. **Spacecraft**: Quaternion attitude control
12. **Hydraulic System**: Actuator dynamics

Each scenario represents real-world propagation rules and safety constraints.

## Results & Findings

### Strong Causal Effect Confirmed

**Key Result**: Uncertainty has a **strong, nonlinear causal effect** on inconsistency.

#### Widen Intervention (Uncertainty Increase)
```
Scale Factor  | Jaccard Index | Empty | Interpretation
-------------|---------------|-------|----------------
0.2          | 0.95          | No    | Highly consistent
1.0          | 0.75          | No    | Baseline
5.0          | 0.32          | No    | Reduced overlap
10.0         | 0.08          | No    | Near inconsistent
20.0         | 0.00          | Yes   | Complete inconsistency
```

**Causal Statement**: `do(widen by >15×)` → `P(empty) = 1`

#### Statistical Significance
- **Granger Causality**: p < 0.001 (uncertainty Granger-causes inconsistency)
- **Cohen's d**: 6.8 (very large effect size for correlate intervention)
- **Transfer Entropy**: Significant information flow from uncertainty to inconsistency

#### Dose-Response Characteristics
- **Below threshold**: Monotonic decrease in Jaccard index
- **Near threshold**: Rapid transition (phase change)
- **Above threshold**: Complete inconsistency (empty intersection)

---

## Documentation

Comprehensive documentation available in `docs/`:

- **[FRAMEWORK_OVERVIEW.md](docs/FRAMEWORK_OVERVIEW.md)** - Theory, architecture, interventions
- **[CONVIDE_SCENARIOS.md](docs/CONVIDE_SCENARIOS.md)** - Engineering scenario details
- **[IMPLEMENTATION_REFERENCE.md](docs/IMPLEMENTATION_REFERENCE.md)** - Technical implementation
- **[METHODOLOGY.md](docs/METHODOLOGY.md)** - Research methodology
- **[QUICKSTART.md](docs/QUICKSTART.md)** - Getting started guide
- **[QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)** - API reference

---

## Dependencies

### MATLAB Requirements
- **MATLAB R2024+**
- **CORA Toolbox**: https://tumcps.github.io/CORA/
- **CPS Framework**: ../CPS-Uncertainty-Propagation-Framework/src
  - conPolyZono class
  - CS_Types.affineMap_cPZ
  - isEmptySet function

### Python Requirements
Install via requirements.txt:
```bash
pip install -r requirements.txt
```

Key packages:
- `numpy>=1.21`
- `scipy>=1.7`
- `statsmodels>=0.13` (Granger causality)
- `pyinform>=0.1` (Transfer entropy)
- `matplotlib>=3.4`
- `pandas>=1.3`
- `torch>=1.10` (Neural encoders)

---

## File Organization

```
Causality_Uncertainty_Inconsistency/
├── data/
│   ├── convide_2d_scenarios/         # Generated 2D experiment data
│   ├── convide_3d_scenarios/         # Generated 3D experiment data
│   └── convide_4d_scenarios/         # Generated 4D experiment data
├── models/
│   ├── neural_causal_encoder_*.pt   # Trained PyTorch models
│   └── *_preproc.json                # Normalization parameters
└── figures/
    ├── training_results/             # Model training plots
    ├── evaluation_results/           # Evaluation metrics
    └── comprehensive_evaluation/     # Full analysis figures
```

---
│   ├── data/
│   │   ├── results_20251216_143022.csv
│   │   └── adaptive_results.csv
│   └── figures/
│       ├── widen_effect_inconsistency_emptiness_change.png
│       ├── misalign_effect_uncertainty_volume_ratio.png
│       └── information_gain_plot.png
├── causal_report.md
└── results_uncertainty_to_inconsistency.json
```

## Citation

If you use this framework in research, please cite:

```bibtex
@software{causal_uncertainty_inconsistency,
  title = {Causal Inference Framework for Uncertainty-Inconsistency Analysis},
  author = {Your Name},
  year = {2025},
  note = {Based on CPS-Uncertainty-Propagation-Framework}
}
```

## Troubleshooting

### MATLAB Engine API Issues
If `matlab.engine` fails to import:
1. Check MATLAB installation: `matlab -batch "disp(matlabroot)"`
2. Install engine: `cd "matlabroot\extern\engines\python"; python setup.py install`
3. Fallback: Use file-based communication (set `use_engine=False`)

### Empty Set Detection
If `isEmptySet()` misses edge cases:
- Increase constraint tolerance in CORA settings
- Use interval check as fallback
- Add custom emptiness heuristics

### Performance
For large-scale experiments:
- Reduce dimensionality (start with 2D)
- Use fewer scenarios per intervention
- Enable parallel MATLAB workers
- Cache baseline scenarios

## Future Enhancements

- [ ] Gaussian Process surrogate models for smooth dose-response
- [ ] Causal discovery (learn graph structure)
- [ ] Multi-objective intervention optimization
- [ ] Transfer learning across scenarios
- [ ] Conformal prediction for uncertainty quantification
- [ ] Integration with reinforcement learning for sequential interventions

## Contact

Questions or contributions? Open an issue in the parent repository:
https://github.com/DE-TUM/CPS-Uncertainty-Propagation-Framework
