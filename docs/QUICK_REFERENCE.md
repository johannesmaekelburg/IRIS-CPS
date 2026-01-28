# Quick Reference: Testing & Validation Commands

## 🎯 Three Simple Steps

### 1️⃣ Generate Test Data (MATLAB) - 5 minutes
```matlab
cd src
generate_3d_4d_test_data
```
**Creates**: `data/test_3d/` and `data/test_4d/` with 270 experiments each

---

### 2️⃣ Cross-Validation (Python) - 15 minutes
```bash
cd src
python neural_causal_encoder.py --mode cv
```
**Output**: 
- `figures/cv_results.json` (metrics)
- `figures/cv_performance.png` (visualization)

---

### 3️⃣ Test Generalization (Python) - 2 minutes
```bash
# 3D test
python neural_causal_encoder.py --mode evaluate --data_dir ../data/test_3d --output_suffix _3d

# 4D test
python neural_causal_encoder.py --mode evaluate --data_dir ../data/test_4d --output_suffix _4d
```
**Output**: `figures/test_results_3d.json` and `figures/test_results_4d.json`

---

## 📋 All Modes

| Command | What It Does | Time |
|---------|--------------|------|
| `--mode train` | Train new model on `--data_dir` | 2-3 min |
| `--mode cv` | K-fold cross-validation | 15-20 min |
| `--mode evaluate` | Test pretrained model on new data | 1-2 min |

---

## 🔧 Common Options

```bash
--data_dir PATH          # Which dataset to use
--mode {train,cv,evaluate}  # What to do
--n_folds N              # CV: number of folds (default: 5)
--n_seeds N              # CV: number of seeds (default: 3)
--output_suffix TEXT     # Add suffix to output files
--model_path PATH        # Evaluate: which model to test
```

---

## 📊 Check Results

```python
import json

# Cross-validation
with open('../figures/cv_results.json') as f:
    cv = json.load(f)
print(f"R² = {cv['statistics']['r2']['mean']:.3f} ± {cv['statistics']['r2']['std']:.3f}")

# Generalization
with open('../figures/test_results_3d.json') as f:
    test = json.load(f)
print(f"3D R² = {test['metrics']['r2']:.3f}")
```

---

## ✅ Success Criteria

- **Cross-Validation**: R² std < 0.05 → ROBUST
- **3D Generalization**: R² > 0.70 → Good transfer
- **4D Generalization**: R² > 0.60 → Acceptable transfer

---

For detailed instructions, see **`TESTING_GUIDE.md`**
