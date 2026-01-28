# Causal Inference Methodology: Uncertainty-Inconsistency Relationship in CPS

## Research Question

**Does uncertainty CAUSE inconsistency in cyber-physical systems using constrained zonotope propagation?**

We investigate the bidirectional causal relationship between uncertainty and inconsistency using Pearl's causal hierarchy, moving beyond purely observational analyses to establish interventional causality. **We study causal effects under fixed propagation semantics (predefined UPRs), a geometric consistency definition (zonotope intersection), and early-phase static models without temporal dynamics.**

---

## Framework Architecture

### Hybrid MATLAB-Python Approach

```
┌─────────────────────────────┐
│  MATLAB Engine              │
│  - CORA Toolbox             │
│  - Zonotope Operations      │
│  - Constraint Propagation   │
│  - Causal Interventions     │
└──────────┬──────────────────┘
           │ JSON Exchange
           ▼
┌─────────────────────────────┐
│  Python Analysis            │
│  - Statistical Methods      │
│  - Information Theory       │
│  - Causal Inference         │
│  - Visualization            │
└─────────────────────────────┘
```

**Rationale:**
- MATLAB: Leverages existing CORA toolbox for rigorous zonotope mathematics
- Python: Provides rich ecosystem for causal inference (statsmodels, pyinform, scipy)
- Separation of concerns: Computation vs. Analysis

---

## Experimental Design

### 1. Baseline Scenarios

Multiple baseline scenarios created with varying **uncertainty levels**:
- Parameter: `uncertainty_level` ∈ {1.0, 2.5, 5.0, 7.5, 10.0}
- Controls generator matrix scaling: `G = eye(dim) * uncertainty_level`
- Ensures consistent baseline with overlapping source and target zonotopes

**Scenario Components:**
- `source` (Z_B): New measurement/state with uncertainty
- `target` (Z_old): Existing model/constraint
- `mapping` (F): Uncertainty Propagation Rule (UPR)
- **Consistency criterion**: Z_propagated ∩ Z_old ≠ ∅

### 2. Causal Interventions

**Three intervention types** designed to manipulate uncertainty structure:

#### a) Widen Intervention
- **Type**: Geometric scaling
- **Parameter**: `scale_factor` ∈ [0.2, 20.0] (20 points)
- **Operation**: `G_new = G * scale_factor`
- **Tests**: Effect of uncertainty magnitude on inconsistency

#### b) Shrink Intervention
- **Type**: Geometric reduction
- **Parameter**: `scale_factor` ∈ [0.05, 5.0] (20 points)
- **Operation**: `G_new = G * scale_factor`
- **Tests**: Inverse of widen (reduction effects)

#### c) Correlate Intervention
- **Type**: Structural dependency
- **Parameter**: `correlation_strength` ∈ [0.0, 0.999] (20 points)
- **Operation**: Adds dependent generator `new_gen = sum(G, 2) * correlation_strength`
- **Tests**: Effect of correlation structure on inconsistency
- **Note**: This intervention inherently changes both generator correlation *and* geometric volume; effects cannot be fully decoupled without normalization.

**Design rationale:**
- 20 data points per intervention (minimum for Granger causality)
- 5 uncertainty levels (scale-dependent effects)
- Total: 300 experiments (5 scenarios × 3 interventions × 20 points)

### 3. Outcome Metrics

For each experiment, we measure:

**Uncertainty Metrics:**
- Volume: `det(G)` or zonotope volume estimate (captures overall magnitude of uncertainty)
- Radius: Maximum generator norm (indicates worst-case directional spread)
- Number of generators (complexity measure for set representation)
- Generator correlation coefficient (detects structural dependencies)

**Inconsistency Metrics:**
- **Jaccard Index**: `|Z_propagated ∩ Z_old| / |Z_propagated ∪ Z_old|`
  - 1.0 = perfect consistency, 0.0 = complete inconsistency
  - Preferred over binary emptiness for graded sensitivity to partial overlaps
- Emptiness: Boolean indicator of intersection failure (hard constraint violation)
- Center distance: `||center(Z_propagated) - center(Z_old)||` (geometric separation independent of volume)
- Volume ratio: Relative size of intersection (captures degree of agreement)

---

## Causal Analysis Methods

### Causal Graph Structure

Our causal model is represented by the following directed acyclic graph (DAG):

```
Uncertainty (U) ───→ Inconsistency (K)
       ↑                     │
       └─────────────────────┘
          (optional feedback)

Intervention: do(U) → breaks incoming edge to U
```

**Interpretation:**
- Forward edge (U → K): Uncertainty propagates through fixed UPRs to produce inconsistency
- Optional backward edge (K → U): System may adjust uncertainty in response to detected inconsistency
- Intervention `do(U = u')`: Directly sets uncertainty, severing all incoming dependencies

### Pearl's Causal Hierarchy

Our methods span Pearl's three-level hierarchy:

**Level 1 (Associational)**: P(Y|X) - "Seeing" — Not addressed

**Level 2 (Observational)**: P(Y|X, context) - "Observing" — Granger causality and transfer entropy infer directed associations from observational sequences

**Level 3 (Interventional)**: P(Y|do(X)) - "Doing" — Our geometric interventional method directly manipulates zonotope structure and measures causal effects

**Primary method**: Interventional causality (Method 3) provides the gold standard. Granger and transfer entropy serve as diagnostic baselines to assess observational-interventional agreement.

---

### Method 3: Interventional Causality (Geometric — Primary)

**Approach**: Direct measurement via do-calculus

**Implementation:**
- Applies intervention: `do(intervention_type, parameter_value)`
- Measures inconsistency before and after
- **Causal effect**: `ACE = mean(inconsistency | do(X)) - mean(inconsistency | control)`

**Metrics:**
- **Average Causal Effect (ACE)**: Direct effect size
- **Cohen's d**: Standardized effect size
  - d < 0.2: negligible
  - d ∈ [0.2, 0.5): small
  - d ∈ [0.5, 0.8): medium  
  - d ≥ 0.8: large
- **p-value**: Statistical significance (t-test)

**Advantages:**
- **Gold standard** (Pearl's Level 3: interventional)
- Direct measurement of causal effects
- No confounding assumptions required
- Handles nonlinear relationships naturally
- Grounded in first-principles geometry (zonotope propagation)

---

### Method 1: Granger Causality (Statistical Baseline)Baseline)

**Approach**: VAR-based predictive causality test

**Implementation:**
- Constructs vector autoregression (VAR) model over intervention sequence
- **Note**: The "time" axis is intervention parameter order, not temporal evolution
- Tests if past uncertainty values improve prediction of inconsistency
- **Null hypothesis**: Uncertainty does NOT Granger-cause inconsistency

**Metric**: p-value from F-test (visualized as -log(p))
- Values > 1.3 indicate significance (p < 0.05)

**Limitations:**
- Requires sufficient observations (≥10 intervention points)
- Assumes stationarity in intervention effects
- **Observational method** (Pearl's Level 2) — insufficient alone for causal claims

### Method 2: Transfer Entropy (Information-Theoretic Baseline)

**Approach**: Directed information flow measurement

**Implementation:**
- Discretizes continuous values (5 bins) over intervention sequence
- Computes Shannon entropy of sequences
- Measures information transfer: TE(X→Y) and TE(Y→X)

**Metrics:**
- Net information flow: `TE(Uncertainty→Inconsistency) - TE(Inconsistency→Uncertainty)`
- Mutual information: Total shared information
- **Directionality**: Positive = uncertainty causes inconsistency

**Limitations:**
- Sensitive to discretization choices
- Requires sufficient data for entropy estimation
- **Observational method** (Pearl's Level 2) — insufficient alone for causal claims



---

---

## Comparative Analysis Strategy

### Research Contribution

**Hypothesis**: Interventional methods (Method 3) provide stronger and more reliable evidence for causality than observational methods (Methods 1-2).

**Comparison dimensions:**
1. **Effect size detection**: Which method identifies strongest causal signals?
2. **Consistency**: Do methods agree on intervention rankings?
3. **Statistical power**: Which provides most significant results?
4. **Interpretability**: Which offers clearest causal claims?

### Visualization Strategy

**Three-panel comparison:**
- Panel 1: Granger causality (-log p-value) — observational baseline
- Panel 2: Transfer entropy (net flow) — information-theoretic baseline
- Panel 3: Interventional causality (Cohen's d) — **primary causal evidence**

**Interpretation:**
- Agreement across methods → Strong causal claim
- Disagreement → Method-dependent effects or observational limitations
- Interventional > Observational → Validates Pearl's causal hierarchy

---

## From Interventional Causality to Representation Learning

### Purpose and Scope

Representation learning is **not used to discover causality**. Instead, it learns a surrogate model that predicts interventional causal effects (ACE, Cohen's d, dose–response parameters) computed via zonotope-based interventions.

### Pipeline

1. **Ground Truth Generation (MATLAB)**
   - Interventional causal effects are computed using geometric zonotope propagation
   - Each experiment yields causal effect sizes via do-calculus (Method 3)
   - These effects serve as **ground truth labels** for supervised learning

2. **Feature Extraction**
   - **Input features**: Scenario descriptors (dimension, generator count), propagation rule type (UPR class), baseline uncertainty metrics (volume, radius, correlation), intervention type, intervention strength
   - **Target labels**: Causal effect size (ACE, Cohen's d), optional dose–response severity curves

3. **Surrogate Model Training (Python)**
   - **Model class**: Lightweight feedforward neural network or gradient-boosted model
   - **Objective**: Predict causal effects for unseen scenario-intervention pairs
   - **Use case**: Fast causal effect estimation without running full MATLAB propagation

### Key Principles

- **Causality remains grounded in first-principles geometry**: All causal claims derive from zonotope propagation, not learned representations
- **Learning improves scalability and generalization**: The surrogate enables rapid exploration of intervention strategies across scenario spaces
- **Causal validity is not improved by learning**: The model is a predictive approximation, not a causal discovery tool

**Relation to Pearl's hierarchy**: The learned model operates at Level 3 (interventional) by predicting `P(Y | do(X))`, but only because it was trained on interventionally-derived labels. The neural network itself performs associational prediction (Level 1).

---

## Implementation Details

### Data Collection (`collect_training_data.m`)

1. Generate baseline scenarios with multiple uncertainty levels
2. For each scenario and intervention type:
   - Create parameter sweep (20 values)
   - Apply intervention via `apply_intervention()`
   - Measure pre/post uncertainty and inconsistency
   - Compute causal effect metrics
3. Export to JSON for Python analysis

### Causal Comparison (`compare_causal_methods.py`)

1. Load all experimental data from JSON files
2. For each intervention type:
   - Method 1: Run Granger causality test (statsmodels)
   - Method 2: Compute transfer entropy (pyinform)
   - Method 3: Calculate ACE and Cohen's d
3. Visualize comparative results
4. Generate summary report

---

## Scope and Limitations

### What This Framework Does

✓ Establishes causal effects of uncertainty on inconsistency via geometric interventions  
✓ Compares interventional vs. observational causal inference methods  
✓ Operates under fixed UPRs and geometric consistency definitions  
✓ Provides first-principles ground truth via zonotope propagation  

### What This Framework Does NOT Claim

✗ **Learning of UPRs**: Propagation rules are predefined, not inferred from data  
✗ **Real-world noise modeling**: Assumes idealized set-based uncertainty, not probabilistic distributions  
✗ **Temporal dynamics**: Studies static propagation scenarios, not time-evolving systems  
✗ **Causal discovery from observational data alone**: Requires interventional experiments  

This scoping ensures clear boundaries for causal claims and prevents overinterpretation of results.

---

## File Structure

```
Causality_Uncertainty_Inconsistency/
├── src/
│   ├── causal_experiment_engine.m      # MATLAB: Core framework
│   ├── collect_training_data.m         # MATLAB: Data generation
│   ├── compare_causal_methods.py       # Python: Method comparison
│   ├── causal_analysis.py              # Python: Dose-response
│   └── neural_causal_encoder.py        # Python: ML surrogate (optional)
├── data/
│   └── causal_inference/               # JSON experimental results
├── figures/                             # Generated visualizations
├── models/                              # Trained ML models (optional)
└── docs/                                # Documentation
```

---

## Key Contributions

1. **Interventional geometric causality**: First application of Pearl's do-calculus to constrained zonotope propagation
2. **Observational-interventional comparison**: Empirical validation that Level 3 methods outperform Level 2 baselines
3. **Hybrid rigor and scalability**: MATLAB for ground truth, Python for inference, learned surrogates for generalization
4. **Systematic intervention design**: 300 experiments spanning magnitude (widen/shrink) and structure (correlate) manipulations
5. **Scale-dependent causal mechanisms**: Multiple uncertainty levels reveal non-uniform causal effects

---

## Reproducibility

### Requirements
- MATLAB R2021a+ with CORA toolbox
- Python 3.8+ with packages: numpy, pandas, matplotlib, statsmodels, pyinform, scipy

### Execution
```bash
# 1. Generate data (MATLAB)
cd src
matlab -batch "collect_training_data"

# 2. Analyze causality (Python)
cd src
python compare_causal_methods.py
```

### Expected Runtime
- Data collection: ~30-60 minutes (300 experiments)
- Causal analysis: ~1-2 minutes
- Total: < 1 hour
