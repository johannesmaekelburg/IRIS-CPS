# Implementation Status and Architecture

**Last Updated**: January 27, 2026  
**Status**: Forward Causality - Production Ready

## System Architecture

### Hybrid MATLAB-Python Design

```
┌─────────────────────────────────────────────────────────┐
│                   MATLAB Backend                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │  causal_experiment_engine.m                     │   │
│  │  - run_intervention()                           │   │
│  │  - measure_state()                              │   │
│  │  - apply_intervention()                         │   │
│  │  - compute_delta()                              │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │  CORA Integration (CPS Framework)               │   │
│  │  - conPolyZono (uncertainty sets)               │   │
│  │  - CS_Types.affineMap_cPZ (propagation)        │   │
│  │  - isEmptySet (inconsistency detection)        │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────┘
                      │ JSON Exchange
                      ▼
┌─────────────────────────────────────────────────────────┐
│                   Python Frontend                       │
│  ┌─────────────────────────────────────────────────┐   │
│  │  causal_analysis.py                             │   │
│  │  - granger_causality()                          │   │
│  │  - transfer_entropy()                           │   │
│  │  - conditional_independence()                   │   │
│  │  - visualization()                              │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## Core Implementation

### 1. Causal Experiment Engine

**File**: `src/causal_experiment_engine.m`

#### Key Methods

##### `run_intervention(scenario, intervention, params)`
Executes a single causal intervention and measures effects.

```matlab
% Input:
%   scenario: struct with .source, .target, .mapping
%   intervention: 'widen'|'shrink'|'shift'|'rotate'|'correlate'
%   params: intervention-specific parameters
%
% Output:
%   result.pre_state: measurements before intervention
%   result.post_state: measurements after intervention
%   result.causal_effect: delta measurements
```

##### `measure_state(scenario)`
Computes all observables for causal analysis.

**Uncertainty Metrics**:
- `source_volume`: det(G) or volume estimate
- `source_radius`: max(||G_i||)
- `source_center`: center point
- `source_n_generators`: number of generators
- `source_correlation`: correlation coefficient
- `target_volume`: target zonotope volume
- `target_radius`: target zonotope radius

**Inconsistency Metrics**:
- `propagation_success`: boolean
- `propagated_volume`: vol(F*Z_source)
- `is_empty`: ~isEmptySet(intersection)
- `has_intersection`: logical indicator
- `empty_intersection`: inconsistency flag
- `vol_intersection`: volume of overlap
- `vol_union`: volume of union
- `jaccard_index`: vol_intersection / vol_union
- `center_distance`: ||c_propagated - c_target||
- `n_constraints`: number of constraints
- `constraint_rank`: rank of constraint matrix
- `constraint_redundancy`: redundant constraints count

##### `apply_intervention(scenario, type, params)`
Modifies scenario according to intervention type.

**Implementations**:
```matlab
case 'widen'
    G_new = G * params.scale_factor;
    
case 'shrink'
    G_new = G * params.scale_factor;
    
case 'shift'
    c_new = c + params.delta;
    
case 'rotate'
    R = rotation_matrix(params.angle, params.axis);
    G_new = R * G;
    
case 'correlate'
    g_new = sum(G, 2) * params.strength;
    G_new = [G, g_new];
```

##### `compute_delta(pre_state, post_state)`
Calculates causal effects with sensitivity checks.

```matlab
% For each metric:
delta.(metric) = post_state.(metric) - pre_state.(metric);

% Relative change (avoid division by zero):
if abs(pre_state.(metric)) > 1e-10
    relative_delta.(metric) = delta.(metric) / pre_state.(metric);
end
```

#### Helper Functions

##### `compute_volume(G)`
```matlab
if size(G,2) == size(G,1)
    vol = abs(det(G));
else
    vol = sqrt(det(G' * G));  % Approximate
end
```

##### `compute_radius(G)`
```matlab
radius = max(sqrt(sum(G.^2, 1)));
```

##### `compute_correlation(G)`
```matlab
if size(G,2) > 1
    C = corrcoef(G');
    corr_val = mean(abs(C(~eye(size(C)))));
else
    corr_val = 0;
end
```

---

## Scenario Generators

### 1. CONVIDE Scenarios

**File**: `examples/generate_convide_scenarios.m`

**Generates**: 12 engineering scenarios (2D: 1-4, 3D: 5-8, 4D: 9-12)

**Process**:
1. Define physical scenario (source, target, mapping)
2. Run 3 interventions (widen, shrink, rotate)
3. Each intervention: 20-point parameter sweep
4. Save results to JSON

```matlab
% Example output structure:
{
  "scenario_id": "convide_2d_1",
  "name": "Robot End-Effector Positioning",
  "dimension": 2,
  "experiments": [
    {"intervention": "widen", "params": {...}, "result": {...}},
    ...
  ]
}
```

### 2. Constrained Scenarios

**File**: `examples/generate_constrained_scenarios.m`

Tests behavior with varying constraint complexity.

### 3. Extreme Scenarios

**File**: `examples/generate_extreme_scenarios.m`

Edge cases: extreme uncertainty, near-singular mappings, high constraint count.

---

## Python Analysis

### File: `src/causal_analysis.py`

**Key Functions**:

```python
def granger_causality(x, y, max_lag=5):
    """Test if x Granger-causes y."""
    # Uses statsmodels VAR framework
    
def transfer_entropy(source, target):
    """Measure information transfer."""
    # Uses pyinform library
    
def conditional_independence(x, y, z):
    """Test X ⊥ Y | Z."""
    # Partial correlation approach
```

---

## Trained Models

### Neural Causal Encoders

**Location**: `models/`

**Model Types**:
- `neural_causal_encoder_2d.pt`: 2D scenarios encoder
- `neural_causal_encoder_3d.pt`: 3D scenarios encoder
- `neural_causal_encoder_4d.pt`: 4D scenarios encoder
- `neural_causal_encoder_constrained.pt`: Constrained scenarios
- `neural_causal_encoder_extreme.pt`: Extreme cases

**Preprocessing**: `*_preproc.json` files store normalization parameters

**Architecture** (typical):
```
Input: [uncertainty_metrics, inconsistency_metrics]
  ↓
Dense(128) → ReLU → Dropout(0.2)
  ↓
Dense(64) → ReLU → Dropout(0.2)
  ↓
Dense(32) → ReLU
  ↓
Output: [causal_effect_prediction]
```

---

## Data Formats

### Scenario JSON Structure

```json
{
  "id": "convide_2d_1",
  "name": "Robot End-Effector Positioning",
  "dimension": 2,
  "family": "forward",
  "source": {
    "center": [0, 0],
    "generators": [[1, 0], [0, 1]],
    "constraints": {"A": [], "b": []}
  },
  "target": {...},
  "mapping": {
    "F": [[1, 0], [0, 1]],
    "f": [0, 0]
  },
  "engineering_context": "...",
  "uncertainty_sources": ["Joint encoder noise", "Backlash"]
}
```

### Experiment Result JSON

```json
{
  "scenario_id": "convide_2d_1",
  "intervention": {
    "type": "widen",
    "params": {"scale_factor": 5.0},
    "timestamp": "2026-01-27T10:30:00"
  },
  "pre_state": {
    "source_volume": 1.0,
    "jaccard_index": 0.85,
    ...
  },
  "post_state": {
    "source_volume": 25.0,
    "jaccard_index": 0.42,
    ...
  },
  "causal_effect": {
    "delta_source_volume": 24.0,
    "delta_jaccard": -0.43,
    ...
  }
}
```

---

## Dependencies

### MATLAB Requirements

- **MATLAB R2024+**
- **CORA Toolbox**: Constrained reachability analysis
- **External Framework**: `../CPS-Uncertainty-Propagation-Framework/src`
  - `conPolyZono` class
  - `CS_Types.affineMap_cPZ`
  - `isEmptySet` function

### Python Requirements

From `requirements.txt`:
```
numpy>=1.21
scipy>=1.7
statsmodels>=0.13
pyinform>=0.1
matplotlib>=3.4
seaborn>=0.11
pandas>=1.3
torch>=1.10  # For neural encoders
```

---

## File Organization

```
Causality_Uncertainty_Inconsistency/
├── src/
│   ├── causal_experiment_engine.m    # Core MATLAB engine
│   └── causal_analysis.py            # Python analysis tools
├── examples/
│   ├── generate_convide_scenarios.m  # CONVIDE generator
│   ├── generate_constrained_scenarios.m
│   ├── generate_extreme_scenarios.m
│   ├── demo_quick_start.m            # Quick demo
│   └── example_causal_experiment.m   # Full example
├── models/
│   ├── neural_causal_encoder_*.pt    # Trained models
│   └── *_preproc.json                # Preprocessing configs
├── data/
│   ├── convide_2d_scenarios/         # 2D experiments
│   ├── convide_3d_scenarios/         # 3D experiments
│   ├── convide_4d_scenarios/         # 4D experiments
│   ├── constrained_scenarios/
│   └── extreme_*/
├── docs/
│   ├── FRAMEWORK_OVERVIEW.md         # This file
│   ├── CONVIDE_SCENARIOS.md
│   ├── METHODOLOGY.md
│   └── ...
└── figures/                          # Generated plots
```

---

## Testing Infrastructure

### Unit Tests

**Location**: `tests/`

**Files**:
- Test files removed (bidirectional causality no longer supported)
- Forward causality tests in `examples/test_bidirectional_causality.m` (renamed, tests forward only)

### Integration Tests

Run full pipeline:
```matlab
demo_quick_start  % Quick sanity check
example_causal_experiment  % Complete workflow
```

---

## Performance Characteristics

### Computation Time (typical)

- Single intervention: ~0.1-0.5s (depends on dimension)
- 20-point sweep: ~2-10s per scenario
- Full CONVIDE suite (720 experiments): ~10-30 minutes

### Memory Usage

- Typical scenario: <10 MB
- Full dataset: ~500 MB (all JSON files)
- Loaded models: ~50 MB (neural encoders)

---

## Known Limitations

1. **Propagation Method**: Only affine mappings currently supported
2. **Constraint Type**: Only linear constraints in cPZ
3. **Dimensionality**: Tested up to 4D (higher dimensions may have numerical issues)
4. **Intervention Types**: 5 types implemented (could expand)
5. **Framework Dependency**: Requires external CPS framework

---

## Future Extensions

Potential enhancements:
- [ ] Nonlinear propagation support
- [ ] Temporal dynamics (time-series causality)
- [ ] Adaptive intervention strategies
- [ ] Real-time causal monitoring
- [ ] GPU acceleration for large sweeps

---

## See Also

- [FRAMEWORK_OVERVIEW.md](FRAMEWORK_OVERVIEW.md) - Overall framework
- [CONVIDE_SCENARIOS.md](CONVIDE_SCENARIOS.md) - Scenario documentation
- [METHODOLOGY.md](METHODOLOGY.md) - Research methodology
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - API reference
