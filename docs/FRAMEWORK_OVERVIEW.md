# Causal Framework for Uncertainty-Inconsistency Analysis

**Last Updated**: January 27, 2026  
**Status**: Production - Forward Causality Only

## Research Question

**Does uncertainty CAUSE inconsistency in cyber-physical systems?**

This framework investigates the causal relationship between uncertainty in sensor measurements and inconsistency in constraint satisfaction using Pearl's causal hierarchy with interventional experiments.

## Causal Direction

**Forward Causality**: Uncertainty → Inconsistency

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│   Uncertainty   │ ─────→  │   Propagation    │ ─────→  │  Inconsistency  │
│   (Source)      │         │   (Mapping F)    │         │   (Constraint)  │
└─────────────────┘         └──────────────────┘         └─────────────────┘
        ↑                                                          ↓
        └─────────────── Interventions & Measurement ─────────────┘
```

We manipulate **uncertainty characteristics** and observe changes in **inconsistency metrics**.

---

## Framework Components

### 1. Uncertainty Representation

**Constrained Polynomial Zonotopes (cPZ)**:
```matlab
Z_source = conPolyZono(c, G, Grest, expMat, A, b, id)
```

**Uncertainty Metrics**:
- **Volume**: `det(G)` - Overall uncertainty magnitude
- **Radius**: `max(||G_i||)` - Worst-case directional spread  
- **Center**: `c` - Central estimate
- **Generators**: Number of independent uncertainty sources
- **Correlation**: `corr(G)` - Structural dependencies

### 2. Propagation Mechanism

**Affine Mapping**:
```matlab
Z_propagated = F * Z_source + f
```

Using `CS_Types.affineMap_cPZ` from CPS framework.

### 3. Inconsistency Detection

**Intersection Check**:
```matlab
Z_intersection = Z_propagated & Z_target
is_inconsistent = isEmptySet(Z_intersection)
```

**Inconsistency Metrics**:
- **Jaccard Index**: `|intersection| / |union|` ∈ [0,1]
  - 1.0 = perfect consistency
  - 0.0 = complete inconsistency
- **Empty Intersection**: Boolean indicator (hard constraint violation)
- **Center Distance**: `||c_propagated - c_target||`
- **Volume Ratio**: `vol(intersection) / vol(propagated)`

---

## Interventions (do-operators)

### Available Interventions

| Intervention | Operation | Parameter | Effect |
|-------------|-----------|-----------|--------|
| `do(widen)` | `G ← scale * G` | scale ∈ [0.2, 20] | Increase uncertainty |
| `do(shrink)` | `G ← scale * G` | scale ∈ [0.05, 5] | Decrease uncertainty |
| `do(shift)` | `c ← c + δ` | δ ∈ ℝⁿ | Translate center |
| `do(rotate)` | `G ← R·G` | R ∈ SO(n) | Reorient uncertainty |
| `do(correlate)` | `G ← [G, g_new]` | g_new = α∑G_i | Add dependencies |

### Parameter Sweeps

Each intervention uses **20-point parameter sweeps** to capture full causal effect:

```matlab
% Example: Widen intervention
scales = logspace(log10(0.2), log10(20), 20);
for i = 1:length(scales)
    params = struct('scale_factor', scales(i));
    result = causal_experiment_engine.run_intervention(scenario, 'widen', params);
end
```

---

## Experimental Workflow

### 1. Create Baseline Scenario

```matlab
scenario = struct();
scenario.source = conPolyZono(...);  % Uncertain measurement
scenario.target = conPolyZono(...);  % Existing constraint
scenario.mapping = struct('F', F, 'f', f);  % Propagation rule
```

### 2. Run Intervention

```matlab
result = causal_experiment_engine.run_intervention(scenario, 'widen', params);
```

### 3. Extract Causal Effects

```matlab
delta_jaccard = result.causal_effect.delta_jaccard;
delta_empty = result.causal_effect.delta_empty_intersection;
delta_volume = result.causal_effect.delta_source_volume;
```

### 4. Analyze Results

```python
# Python analysis
import causal_analysis
causal_analysis.granger_causality(uncertainty_ts, inconsistency_ts)
causal_analysis.information_flow(data)
```

---

## Key Features

✅ **Pearl's Do-Calculus**: Interventional (not observational) causality  
✅ **Controlled Experiments**: Systematic parameter sweeps  
✅ **Graded Metrics**: Jaccard index for sensitivity analysis  
✅ **Multi-dimensional**: Supports 2D, 3D, 4D scenarios  
✅ **Hybrid Architecture**: MATLAB (computation) + Python (analysis)  
✅ **Engineering Contexts**: CONVIDE scenarios from real applications  

---

## File Structure

```
src/causal_experiment_engine.m    - Core intervention & measurement engine
examples/generate_convide_scenarios.m - CONVIDE engineering scenarios
examples/demo_quick_start.m       - Quick demonstration
src/causal_analysis.py            - Python statistical analysis
models/                           - Trained neural encoders
data/convide_*_scenarios/         - Generated experiment data
```

---

## Example: Complete Experiment

```matlab
% 1. Load or create scenario
scenario = load('data/convide_2d_scenarios/scenario_2d_1.mat');

% 2. Run widen intervention sweep
scales = logspace(log10(0.2), log10(20), 20);
results = cell(20, 1);

for i = 1:20
    params = struct('scale_factor', scales(i));
    results{i} = causal_experiment_engine.run_intervention(...
        scenario, 'widen', params);
end

% 3. Extract time series
uncertainty = cellfun(@(r) r.post_state.source_volume, results);
jaccard = cellfun(@(r) r.post_state.jaccard_index, results);

% 4. Test causality
granger_pvalue = causal_analysis.granger_test(uncertainty, jaccard);
```

---

## Theoretical Foundation

### Pearl's Causal Hierarchy

1. **Association** (Observational): P(y|x) - "What if I observe X?"
2. **Intervention** (Experimental): P(y|do(x)) - "What if I SET X?" ← **We do this**
3. **Counterfactual**: P(y_x|x',y') - "What if X had been different?"

### Why Interventions Matter

**Observational**: Correlation between uncertainty and inconsistency  
**Interventional**: Does changing uncertainty CAUSE inconsistency changes?

Interventions **break confounding** by directly manipulating variables.

---

## Measurement Approach

### Pre-Intervention State
```matlab
pre = causal_experiment_engine.measure_state(scenario);
```

### Post-Intervention State
```matlab
scenario_modified = apply_intervention(scenario, 'widen', params);
post = causal_experiment_engine.measure_state(scenario_modified);
```

### Causal Effect (Delta)
```matlab
delta = compute_delta(pre, post);
% delta.delta_jaccard, delta.delta_source_volume, etc.
```

---

## References

- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*
- Althoff, M. (2015). *CORA: Constrained Reachability Analysis*
- Sharma, R. et al. (2023). *CONVIDE Engineering Scenarios*

---

## See Also

- [METHODOLOGY.md](METHODOLOGY.md) - Detailed research methodology
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Technical implementation
- [QUICKSTART.md](QUICKSTART.md) - Getting started guide
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - API reference
