# Testing & Validation Guide

This guide explains how to test generalization and validate robustness of the neural causal encoder.

## 📋 Quick Reference

### 1. Generate 3D/4D Test Data (MATLAB)
```matlab
cd src
generate_3d_4d_test_data

% Creates:
%   - data/test_3d/  (3D scenarios, 270 experiments)
%   - data/test_4d/  (4D scenarios, 270 experiments)
```

### 2. Run 5-Fold Cross-Validation (Python)
```bash
cd src

# Standard 5-fold CV with 3 seeds
python neural_causal_encoder.py --mode cv --data_dir ../data/causal_inference

# More rigorous: 10-fold CV with 5 seeds
python neural_causal_encoder.py --mode cv --n_folds 10 --n_seeds 5

# Output:
#   - figures/cv_results.json (metrics for each fold)
#   - figures/cv_performance.png (visualization)
```

### 3. Test Generalization on 3D/4D Data (Python)
```bash
cd src

# Test on 3D data (2D-trained model)
python neural_causal_encoder.py \
    --mode evaluate \
    --data_dir ../data/test_3d \
    --model_path ../models/neural_causal_encoder.pt \
    --output_suffix _3d

# Test on 4D data
python neural_causal_encoder.py \
    --mode evaluate \
    --data_dir ../data/test_4d \
    --model_path ../models/neural_causal_encoder.pt \
    --output_suffix _4d

# Output:
#   - figures/test_results_3d.json
#   - figures/test_results_4d.json
```

---

## 📊 Detailed Instructions

### A. Generating Test Data for Different Dimensions

**Purpose**: Validate whether the 2D-trained model can generalize to higher dimensions (transfer learning test).

**Script**: `generate_3d_4d_test_data.m`

**Configuration**:
- **Dimensions**: 3D and 4D
- **Scenarios per dimension**: 3 (uncertainty levels: 2.0, 5.0, 8.0)
- **Interventions**: widen, shrink, correlate
- **Parameter values**: 10 per intervention (fewer than training for efficiency)
- **Repetitions**: 5 per condition (vs. 10 in training)
- **Total experiments**: ~270 per dimension

**Output Datasets**:
```
data/
├── causal_inference/        # Training data (2D, 3000 experiments)
├── test_3d/                 # Test data (3D, 270 experiments)
│   ├── all_results.json
│   ├── results_widen_unc*.json
│   ├── results_shrink_unc*.json
│   └── results_correlate_unc*.json
└── test_4d/                 # Test data (4D, 270 experiments)
    ├── all_results.json
    ├── results_widen_unc*.json
    ├── results_shrink_unc*.json
    └── results_correlate_unc*.json
```

**Key Features**:
- Different seed (123 vs. 42) ensures independence from training data
- Different uncertainty levels (2.0, 5.0, 8.0 vs. 1.0, 2.5, 5.0, 7.5, 10.0) tests interpolation/extrapolation
- Marked as `dataset_type: "test"` in metadata

---

### B. Cross-Validation for Robustness Assessment

**Purpose**: Validate that R²=0.90 is robust across different data splits, not just a lucky single split.

**Mode**: `--mode cv`

**How it Works**:
1. **Stratified scenario-based splitting**: Groups experiments by scenario, then splits scenarios (not individual experiments) into folds
   - Ensures each fold tests generalization to unseen scenarios
   - Preserves scenario distributions

2. **Multiple random seeds**: Repeats entire K-fold process with different seeds (default: 3)
   - Reduces variance from random splitting
   - Provides confidence intervals

3. **Per-fold training**: Trains separate model for each fold
   - Early stopping on validation loss
   - Same architecture and hyperparameters as main model

**Metrics Computed**:
- Mean, std, min, max, median across all folds
- 95% confidence intervals
- Robustness assessment (R² std < 0.05 = ROBUST)

**Example Output**:
```
CROSS-VALIDATION SUMMARY
======================================================================

R² Score:
  Mean ± Std:  0.8950 ± 0.0120
  95% CI:      [0.8830, 0.9070]
  Range:       [0.8750, 0.9180]
  Median:      0.8940

RMSE:
  Mean ± Std:  0.7850 ± 0.0340
  95% CI:      [0.7510, 0.8190]
  Range:       [0.7220, 0.8510]
  Median:      0.7805

MAE:
  Mean ± Std:  0.5780 ± 0.0260
  95% CI:      [0.5520, 0.6040]
  Range:       [0.5330, 0.6190]
  Median:      0.5755

======================================================================
  Model is ROBUST across folds
  (R² std = 0.0120, threshold = 0.05)
======================================================================
```

**Visualization**: `cv_performance.png` shows:
- R² and RMSE across folds (line plots)
- Distribution boxplots
- Summary statistics table

---

### C. Generalization Testing on New Data

**Purpose**: Evaluate pretrained model on completely unseen datasets (different dimensions, scenarios, interventions).

**Mode**: `--mode evaluate`

**Use Cases**:

1. **Higher dimensions** (3D, 4D):
   ```bash
   python neural_causal_encoder.py --mode evaluate --data_dir ../data/test_3d --output_suffix _3d
   ```
   - Tests if 2D-trained model transfers to 3D
   - Expects performance degradation (dimensionality mismatch in features)
   - Useful for understanding model limitations

2. **Different intervention ranges**:
   ```bash
   python neural_causal_encoder.py --mode evaluate --data_dir ../data/extreme_cases
   ```
   - Test extrapolation to extreme parameter values
   - Identify safe operating ranges

3. **New scenarios**:
   - Generate data with `generate_3d_4d_test_data.m` (modify to create new scenarios)
   - Test on scenarios not seen during training

**Key Difference from Training Mode**:
- **No retraining**: Uses existing model weights
- **Preprocessing config**: Loads training preprocessing (same feature scaling)
- **Full evaluation**: Computes R², RMSE, MAE on test set
- **Results saved**: JSON file with metrics for comparison

---

## 🎯 Expected Results & Interpretation

### Cross-Validation (2D Data)
| Metric | Expected Range | Interpretation |
|--------|----------------|----------------|
| **Mean R²** | 0.88 - 0.92 | Should be close to original 0.90 |
| **R² Std** | < 0.05 | Low variance = robust model |
| **95% CI width** | < 0.04 | Narrow CI = consistent performance |

**If R² std > 0.05**: Model is sensitive to data split → Consider:
- Larger dataset (more scenarios)
- Regularization (higher dropout)
- Ensemble methods

### Generalization (3D/4D Data)
| Dimension | Expected R² | Notes |
|-----------|-------------|-------|
| **2D** | 0.90 | Training dimension |
| **3D** | 0.70 - 0.85 | Partial transfer (some features mismatch) |
| **4D** | 0.60 - 0.80 | More degradation expected |

**Why degradation?**
- Input features include dimension-specific metrics (volume, radius)
- Higher dimensions → different zonotope geometry
- Model hasn't seen 3D/4D patterns during training

**Good performance (R² > 0.75) means**:
- Core causal patterns transfer across dimensions
- Intervention effects are dimension-invariant
- Model learned generalizable relationships

**Poor performance (R² < 0.60) suggests**:
- Need dimension-specific training data
- Feature engineering required for N-D cases
- Retraining on higher dimensions necessary

---

## 📁 Dataset Organization Best Practices

### Recommended Structure
```
data/
├── causal_inference/           # Main training dataset
│   ├── all_results.json        # 3,000 experiments (2D, 5 scenarios)
│   └── results_*.json          # Per-intervention files
│
├── test_2d_extended/           # Additional 2D test scenarios
│   └── all_results.json        # Different uncertainty levels
│
├── test_3d/                    # 3D generalization test
│   └── all_results.json        # 270 experiments
│
├── test_4d/                    # 4D generalization test
│   └── all_results.json        # 270 experiments
│
└── edge_cases/                 # Extreme parameter values
    └── all_results.json        # Boundary conditions
```

### Why Separate Datasets?

**Advantages**:
1. **Clean separation**: Training vs. validation vs. test
2. **Reproducibility**: Fixed test sets for comparisons
3. **Version control**: Track changes to specific datasets
4. **Flexibility**: Mix and match datasets for different experiments

**Example Workflows**:

```bash
# Train on 2D data
python neural_causal_encoder.py --data_dir ../data/causal_inference

# Validate robustness (same data, different splits)
python neural_causal_encoder.py --mode cv --data_dir ../data/causal_inference

# Test generalization (different data)
python neural_causal_encoder.py --mode evaluate --data_dir ../data/test_3d

# Test on edge cases
python neural_causal_encoder.py --mode evaluate --data_dir ../data/edge_cases
```

---

## 🛠️ Creating Custom Test Datasets

### Modify `generate_3d_4d_test_data.m`

**1. Change dimensions**:
```matlab
test_configs = {
    struct('dim', 5, 'output_dir', 'test_5d', 'n_scenarios', 3, 'n_param_values', 10);
    struct('dim', 8, 'output_dir', 'test_8d', 'n_scenarios', 3, 'n_param_values', 10);
};
```

**2. Change uncertainty levels**:
```matlab
test_uncertainty_levels = [1.0, 3.0, 6.0, 9.0];  % 4 scenarios instead of 3
```

**3. Change parameter ranges (extrapolation test)**:
```matlab
interventions_test = {
    struct('type', 'widen', 'param', 'scale_factor', ...
        'values', linspace(10.0, 50.0, 10));  % Extreme widening!
};
```

**4. Add new intervention types**:
```matlab
interventions_test = {
    struct('type', 'shift', 'param', 'shift_vector', ...
        'values', linspace(0, 10, 10));
    struct('type', 'rotate', 'param', 'rotation_angle', ...
        'values', linspace(0, pi/2, 10));
};
```

---

## 📊 Analyzing Results

### Compare Performance Across Datasets

```python
import json

# Load results
with open('../figures/cv_results.json') as f:
    cv_2d = json.load(f)

with open('../figures/test_results_3d.json') as f:
    test_3d = json.load(f)

with open('../figures/test_results_4d.json') as f:
    test_4d = json.load(f)

# Compare
print(f"2D (CV):  R² = {cv_2d['statistics']['r2']['mean']:.3f} ± {cv_2d['statistics']['r2']['std']:.3f}")
print(f"3D Test:  R² = {test_3d['metrics']['r2']:.3f}")
print(f"4D Test:  R² = {test_4d['metrics']['r2']:.3f}")

# Degradation
degradation_3d = (cv_2d['statistics']['r2']['mean'] - test_3d['metrics']['r2']) / cv_2d['statistics']['r2']['mean']
degradation_4d = (cv_2d['statistics']['r2']['mean'] - test_4d['metrics']['r2']) / cv_2d['statistics']['r2']['mean']

print(f"\nPerformance degradation:")
print(f"  3D: {degradation_3d*100:.1f}%")
print(f"  4D: {degradation_4d*100:.1f}%")
```

---

## ✅ Validation Checklist

Before presenting results:

- [ ] **Cross-validation completed** (5-fold, 3 seeds minimum)
  - [ ] R² mean > 0.85
  - [ ] R² std < 0.05
  - [ ] 95% CI width < 0.04

- [ ] **3D generalization tested**
  - [ ] Test data generated (`test_3d/` exists)
  - [ ] Evaluation run successfully
  - [ ] R² documented (expect 0.70-0.85)

- [ ] **4D generalization tested**
  - [ ] Test data generated (`test_4d/` exists)
  - [ ] Evaluation run successfully
  - [ ] R² documented (expect 0.60-0.80)

- [ ] **Results documented**
  - [ ] All JSON files saved in `figures/`
  - [ ] Visualizations generated (cv_performance.png, etc.)
  - [ ] Performance comparison table created

- [ ] **Interpretation prepared**
  - [ ] Robustness assessment (ROBUST vs. VARIABLE)
  - [ ] Generalization limits identified
  - [ ] Next steps planned (if needed)

---

## 🚀 Next Steps After Validation

### If Model is Robust (R² std < 0.05):
✅ **Production-ready!** Proceed to:
- Create inference API
- Build interactive dashboard
- Apply to real problems

### If Model Shows Degradation on 3D/4D (R² < 0.70):
**Option 1**: Retrain on mixed-dimension data
```bash
# Generate 3D training data (larger dataset)
# Modify collect_training_data.m to create 3D scenarios

# Train on combined 2D+3D data
python neural_causal_encoder.py --data_dir ../data/causal_inference_mixed
```

**Option 2**: Dimension-agnostic features
- Normalize features by dimension
- Use relative metrics (e.g., volume^(1/dim))
- Add dimension as input feature

**Option 3**: Ensemble approach
- Train separate models per dimension
- Route predictions based on input dimension

### If Cross-Validation Shows High Variance:
**Increase dataset**:
```matlab
% In collect_training_data.m
n_repeats = 20;  % More repetitions
uncertainty_levels = [1.0, 2.0, 3.0, ..., 10.0];  % More scenarios
```

**Add regularization**:
```bash
# Increase dropout
python neural_causal_encoder.py --hidden_dims 64 32 --dropout 0.2
```

**Try different architecture**:
```bash
# Deeper network
python neural_causal_encoder.py --hidden_dims 128 64 32
```

---

## 📚 References

- **Cross-validation**: Ensures generalization, provides confidence intervals
- **Transfer learning**: Tests if patterns learned in 2D apply to N-D
- **Dataset separation**: Training/validation/test prevents overfitting

**Key Papers**:
- Kohavi, R. (1995). "A study of cross-validation and bootstrap for accuracy estimation and model selection."
- Raschka, S. (2018). "Model evaluation, model selection, and algorithm selection in machine learning."
