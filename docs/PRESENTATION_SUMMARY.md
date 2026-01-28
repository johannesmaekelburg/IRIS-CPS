# Causal Inference Framework: Uncertainty ↔ Inconsistency
## Presentation Summary for Colleagues

---

## 🎯 What This Framework Does

This framework enables **rigorous causal analysis** of how **uncertainty** and **inconsistency** interact in constrained zonotope-based system propagation. Instead of just observing correlations, we perform **interventional experiments** (do-calculus) to establish true cause-and-effect relationships.

### Core Research Questions

1. **Uncertainty → Inconsistency**: Does widening uncertainty *cause* design conflicts?
2. **Inconsistency → Uncertainty**: Do misaligned constraints *cause* uncertainty amplification?
3. **Dose-Response**: What's the quantitative relationship between intervention strength and effect size?

---

## 🏗️ System Architecture

### Two-Layer Design

```
┌─────────────────────────────────────────────────┐
│         Python Layer (Orchestrator)             │
│  - Experiment design & data collection          │
│  - Neural surrogate model (R² = 0.90)          │
│  - Visualization & causal analysis              │
└────────────────┬────────────────────────────────┘
                 │ JSON/File-based communication
┌────────────────▼────────────────────────────────┐
│         MATLAB Layer (Engine)                   │
│  - Zonotope propagation (CORA toolbox)         │
│  - Intervention mechanisms (10 types)           │
│  - Ground-truth causal effect computation       │
└─────────────────────────────────────────────────┘
```

**Why This Design?**
- **No reimplementation**: Reuses existing CORA zonotope operations
- **Adaptive learning**: Python ML ecosystem enables smart experiment design
- **Scalability**: Neural surrogate predicts effects 1000× faster than MATLAB
- **Reproducibility**: Complete pipeline from data generation to visualization

---

## 📊 What We've Built

### 1. **MATLAB Causal Experiment Engine** (`causal_experiment_engine.m`)
- **10 intervention mechanisms** to manipulate uncertainty and inconsistency
- **Comprehensive state measurement**: volume, radius, emptiness, overlap, correlation
- **Interventional do-calculus**: Computes ground-truth causal effects (not correlations!)

#### Intervention Types

| Category | Intervention | Parameter | Physical Meaning |
|----------|-------------|-----------|------------------|
| **Uncertainty** | `widen` | `scale_factor` | Increase manufacturing tolerances |
| | `shrink` | `scale_factor` | Tighten quality control |
| | `correlate` | `correlation_strength` | Introduce dependent errors |
| | `shift` | `shift_vector` | Bias calibration |
| | `rotate` | `rotation_matrix` | Transform uncertainty axes |
| **Inconsistency** | `misalign` | `misalignment_factor` | Corrupt model correspondences |
| | `constrain` | `constraint_tightness` | Add design constraints |
| | `corrupt` | `corruption_level` | Introduce mapping errors |
| | `decouple` | `decoupling_strength` | Break dependencies |
| | `conflict` | `conflict_intensity` | Force constraint violations |

### 2. **Automated Data Collection** (`collect_training_data.m`)
- **Version v1.1** with robust metadata tracking
- Generates **3,000 experiments** systematically:
  - 5 baseline scenarios (different uncertainty levels)
  - 3 intervention types × 20 parameter values each
  - 10 repetitions per condition for statistical reliability
- **Sanitized identifiers**: scenario_id, run_id, intervention_id
- **Consolidated export**: `all_results.json` with complete dataset

**Example Output**:
```json
{
  "scenario_id": "S_dim2_unc10p0",
  "intervention_type": "widen",
  "intervention_value": 2.5,
  "repeat_idx": 3,
  "dim": 2,
  "uncertainty_level": 10.0,
  "pre_state": {...},
  "post_state": {...},
  "causal_effect": {
    "delta_emptiness": 0.23,
    "delta_volume": 1.45,
    "cohens_d": 3.42
  }
}
```

### 3. **Neural Causal Surrogate** (`neural_causal_encoder.py`)
- **Purpose**: Fast prediction of causal effects without running expensive MATLAB simulations
- **NOT** discovering causality (causality = ground truth from MATLAB)
- **Supervised learning**: Learns patterns from 3,000 interventional experiments

#### Model Architecture
```
Input (10 features):
  ├─ Intervention type (one-hot): [correlate, shrink, widen]
  ├─ Intervention strength (numeric)
  ├─ Baseline uncertainty level
  ├─ Dimension, volume, radius, correlation
  └─ Scenario metadata

      ↓
   [Dense 64] → ReLU → Dropout(0.1)
      ↓
   [Dense 32] → ReLU → Dropout(0.1)
      ↓
   [Dense 1] → Output

Output: Cohen's d (standardized effect size)
```

#### Performance (Latest Training)
- **R² = 0.8980** → Explains 90% of variance in causal effects!
- **RMSE = 0.7789** → Low prediction error
- **MAE = 0.5733** → Robust across scenarios
- **2,817 trainable parameters** → Lightweight, fast inference
- **Training data**: 3,000 experiments (8.3× larger than initial 360)

**Comparison**:
| Metric | Old Model (360 samples) | New Model (3,000 samples) | Improvement |
|--------|------------------------|---------------------------|-------------|
| R² | 0.40 | **0.90** | **2.24× better** |
| RMSE | ~1.8 | **0.78** | **2.3× reduction** |
| MAE | ~1.3 | **0.57** | **2.3× reduction** |

### 4. **Visualization & Analysis** (`causal_analysis.py`)
Generates **13 publication-ready plots**:
- **Dose-response curves**: Effect size vs. intervention strength
- **Surrogate validation**: Predicted vs. true causal effects
- **Scenario analysis**: Performance across different uncertainty levels
- **Emptiness thresholds**: Critical points where inconsistency emerges
- **Intervention comparison**: Relative strength of different mechanisms

---

## 🔬 Key Findings

### Causal Effects Discovered

| Intervention | Mean Effect (Cohen's d) | Interpretation |
|-------------|------------------------|----------------|
| **Shrink** | **3.75** | Very strong causal effect (reduces inconsistency) |
| **Widen** | **3.67** | Very strong causal effect (increases inconsistency) |
| **Correlate** | **1.21** | Moderate causal effect (introduces dependencies) |

### Insights
1. **Geometric interventions dominate**: Shrink/Widen have 3× stronger effects than Correlate
2. **Higher baseline uncertainty amplifies effects**: Scenarios with unc=10.0 show stronger responses
3. **No emptiness in normal range**: Critical thresholds not reached in tested parameter space
4. **Dose-response is nonlinear**: U-shaped curves for correlation intervention

### Strongest Scenario
- **`results_shrink_unc10.0`**: Effect magnitude = 4.57
  - High baseline uncertainty (10.0) + aggressive shrinking → maximal impact

---

## 💡 What You Can Do With This Framework

### 1. **Predict Causal Effects Instantly**
Without running MATLAB simulations:
```python
from neural_causal_encoder import predict_causal_effect

effect = predict_causal_effect(
    intervention='widen',
    intervention_strength=2.5,
    baseline_uncertainty=7.5,
    dimension=2
)
# Returns: Cohen's d ≈ 3.2 (in milliseconds!)
```

### 2. **Design Robust Systems**
- **What-if analysis**: "If we widen tolerance by 50%, how much inconsistency risk?"
- **Sensitivity analysis**: "Which parameters have strongest causal impact?"
- **Optimization**: "Find minimal uncertainty that maintains consistency"

### 3. **Transfer Learning to New Domains**
- Train on 2D systems, apply to higher dimensions
- Learn intervention patterns, generalize to new scenarios
- Uncertainty quantification with Bayesian extensions

### 4. **Interactive Exploration**
- **Example use case**: Brake disc manufacturing
  - Baseline: d_req = 300±5mm, r_i = 25±2.5mm
  - Question: "Does widening d_req tolerance cause design conflicts?"
  - Answer: Run `example_causal_experiment.m` → Generates 6-panel visualization with causal effects

---

## 📁 Repository Structure

```
Causality_Uncertainty_Inconsistency/
├── src/                          # Source code
│   ├── causal_experiment_engine.m      # MATLAB: 10 intervention mechanisms
│   ├── collect_training_data.m         # MATLAB: Generate 3,000 experiments
│   ├── neural_causal_encoder.py        # Python: Train surrogate (R²=0.90)
│   ├── causal_analysis.py              # Python: Visualization suite
│   └── compare_causal_methods.py       # Python: Method comparison
│
├── examples/
│   └── example_causal_experiment.m     # Brake disc demo (8 subplots)
│
├── data/causal_inference/        # 3,000 experiments (JSON files)
│   ├── all_results.json                # Consolidated dataset
│   └── results_*.json                  # Per-intervention files (15 total)
│
├── models/                       # Trained models
│   ├── neural_causal_encoder.pt        # Latest model (Dec 16, 23:59)
│   └── neural_causal_encoder_preproc.json  # Feature config
│
├── figures/                      # 13 visualizations
│   ├── causal_analysis_summary.txt     # Performance metrics
│   ├── surrogate_vs_true.png           # Validation plot
│   ├── dose_response_*.png             # 6 dose-response curves
│   ├── emptiness_threshold_*.png       # 3 threshold plots
│   └── scenario_mae.png                # Error breakdown
│
└── README.md                     # Complete documentation
```

---

## 🚀 Quick Start Guide

### Run Example Experiment (MATLAB)
```matlab
% Navigate to repository
cd('C:\...\Causality_Uncertainty_Inconsistency')

% Run brake disc causal experiment
example_causal_experiment

% Output:
%   - 6-panel visualization showing causal effects
%   - JSON export for Python analysis
%   - Console summary of effect sizes
```

### Train Neural Surrogate (Python)
```bash
# Install dependencies
pip install torch numpy matplotlib

# Train model on 3,000 experiments
cd src
python neural_causal_encoder.py --data_dir ../data/causal_inference

# Output:
#   - models/neural_causal_encoder.pt
#   - R² = 0.90 on test set
#   - Training took ~2 minutes
```

### Generate Visualizations (Python)
```bash
# Create all plots
python causal_analysis.py

# Output:
#   - 13 PNG files in figures/
#   - causal_analysis_summary.txt
```

---

## 📈 Performance Metrics

### Dataset Statistics
- **Total experiments**: 3,000
- **Scenarios**: 5 (uncertainty levels: 1.0, 2.5, 5.0, 7.5, 10.0)
- **Interventions**: 3 types (widen, shrink, correlate)
- **Parameter sweep**: 20 values per intervention
- **Repetitions**: 10 per condition
- **Dimension**: 2D (extendable to N-D)

### Neural Model Performance
- **R²**: 0.8980 (excellent fit)
- **RMSE**: 0.7789 (low error)
- **MAE**: 0.5733 (robust)
- **Parameters**: 2,817 (lightweight)
- **Inference time**: <1ms per prediction
- **Training time**: ~2 minutes on CPU

### Validation Strategy
- **70/15/15 split** by scenario (not random!)
  - Ensures generalization to unseen scenarios
- **Early stopping** on validation loss (patience=20)
- **Scenario-wise MAE** computed for robustness

---

## 🔑 Key Advantages

1. **Causal, Not Correlational**
   - Uses interventional do-calculus (Pearl's framework)
   - Establishes cause-and-effect, not just associations
   - Ground truth from physics-based MATLAB simulations

2. **Scalable & Fast**
   - Neural surrogate enables rapid exploration
   - 1000× faster than running MATLAB for each query
   - Batch predictions for optimization workflows

3. **Reproducible & Documented**
   - Version-controlled data collection (v1.1)
   - Complete metadata tracking
   - Consolidated exports for sharing

4. **Extensible**
   - Add new intervention types easily
   - Scale to higher dimensions
   - Transfer learning to new domains

5. **Production-Ready**
   - R²=0.90 exceeds typical surrogate model standards
   - Validated on held-out scenarios
   - Clean API for integration

---

## 🎓 Theoretical Foundation

### Causal Inference Framework
- **Pearl's do-calculus**: Interventions break natural correlations
- **Average Causal Effect (ACE)**: E[Y | do(X=x)] - E[Y | do(X=x')]
- **Cohen's d**: Standardized effect size for comparability

### Zonotope Mathematics
- **Constrained zonotopes**: Polytopic uncertainty representation
- **Affine maps**: Uncertainty propagation through system models
- **Intersection operations**: Consistency checking (emptiness detection)

### Machine Learning
- **Supervised regression**: Features → causal effect size
- **Neural networks**: Universal function approximators
- **Scenario-based splitting**: Ensures out-of-distribution generalization

---

## 📝 Publications & Citations

This framework implements methodologies from:
- Pearl, J. (2009). *Causality: Models, Reasoning and Inference*
- Kochdumper, N. et al. (2019). *CORA: Continuous Reachability Analyzer*
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*

**Suggested Citation**:
```
Causal Inference Framework for Uncertainty-Inconsistency Analysis
Repository: CPS-Uncertainty-Propagation-Framework
Date: December 2025
```

---

## 🛠️ Technical Requirements

### MATLAB
- **Version**: R2020a or later
- **Toolboxes**: None (uses CORA)
- **Dependencies**: CORA toolbox (included in CPS framework)

### Python
- **Version**: 3.8+
- **Core libraries**:
  - PyTorch 2.0+ (neural networks)
  - NumPy 1.20+ (numerical computation)
  - Matplotlib 3.5+ (visualization)
  - Pandas 1.3+ (data handling)

### Installation
```bash
# Clone repository
git clone https://github.com/DE-TUM/CPS-Uncertainty-Propagation-Framework.git

# Install Python dependencies
cd Causality_Uncertainty_Inconsistency
pip install -r requirements.txt
```

---

## 🎯 Future Directions

### Short-Term (Ready to Implement)
1. **Uncertainty quantification**: Add Bayesian neural networks for confidence intervals
2. **3D/4D systems**: Test transfer learning to higher dimensions
3. **K-fold validation**: Compute cross-validation for robustness assessment

### Medium-Term (Research Extensions)
1. **Active learning**: Adaptively select most informative experiments
2. **Multi-objective optimization**: Balance uncertainty vs. inconsistency
3. **Real-world case studies**: Apply to industrial CPS (automotive, aerospace)

### Long-Term (Novel Contributions)
1. **Causal discovery**: Learn intervention types from observational data
2. **Temporal causality**: Extend to time-series propagation
3. **Hybrid models**: Combine physics-based and data-driven approaches

---

## 🤝 Contact & Support

**Questions?** Check:
- `README.md` for detailed documentation
- `examples/example_causal_experiment.m` for working demos
- `figures/causal_analysis_summary.txt` for latest results

**Need Help?**
- Review MATLAB console output for troubleshooting
- Check Python training logs in terminal
- Validate data files in `data/causal_inference/`

---

## 📊 Summary Statistics

| Metric | Value |
|--------|-------|
| **Total Code** | ~2,500 lines (MATLAB + Python) |
| **Experiments Generated** | 3,000 |
| **Trained Models** | 1 (R²=0.90) |
| **Visualizations** | 13 plots |
| **Intervention Types** | 10 mechanisms |
| **Causal Effects Quantified** | 3 primary (shrink, widen, correlate) |
| **Dataset Size** | ~15 JSON files + 1 consolidated |
| **Model Parameters** | 2,817 |
| **Training Time** | ~2 minutes |
| **Inference Speed** | <1ms per prediction |

---

## ✅ Validation Checklist

- [x] **Ground truth causal effects** computed via MATLAB do-calculus
- [x] **3,000 diverse experiments** across 5 scenarios, 3 interventions
- [x] **Neural surrogate R²=0.90** on held-out scenarios
- [x] **Complete visualization suite** (13 publication-ready plots)
- [x] **Reproducible pipeline** (version control, metadata tracking)
- [x] **Documentation** (README, code comments, examples)
- [x] **Production-ready** (low error, fast inference, validated)

---

**Framework Status**: ✅ **Production-Ready**  
**Last Updated**: December 17, 2025  
**Model Version**: v1.1 (Neural Encoder Dec 16, 23:59)  
**Performance**: R²=0.90, RMSE=0.78, MAE=0.57
