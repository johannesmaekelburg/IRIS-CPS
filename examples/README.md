# Examples Directory

This directory contains demonstration scripts and dataset generators for the bidirectional uncertainty-inconsistency causal inference framework.

## Quick Start

**New users start here:**
```matlab
demo_quick_start.m          % 30-second demo of forward & reverse causality
```

## Testing & Validation

```matlab
test_bidirectional_causality.m   % Comprehensive test of all intervention types
```

## Dataset Generation (Production)

Generate training data for neural surrogate models:

```matlab
generate_convide_scenarios.m              % ✅ v1.1: CONVIDE benchmark (12 scenarios, ~1,680 experiments, bidirectional)
generate_constrained_scenarios.m          % ✅ v2.1: Constrained scenarios (15 scenarios, ~30,000 experiments, bidirectional)  
generate_extreme_scenarios.m              % ✅ v2.1: Extreme interventions (9 scenarios, ~2,000 experiments, bidirectional)
```

**All generators now support 6 intervention types (3 forward + 3 reverse)**

## Deprecated / Legacy

These files are kept for backward compatibility but are superseded by the scripts above:

```matlab
example_causal_experiment.m          % OLD: Use demo_quick_start.m instead
collect_training_data.m              % OLD: Use generate_*.m scripts instead
generate_3d_4d_test_data.m           % OLD: Merged into main generators
```

---

## File Naming Convention

- `demo_*.m` - Interactive demonstrations for learning
- `test_*.m` - Validation and testing scripts
- `generate_*.m` - Production dataset generators

---

## Dataset Output Structure

```
../data/
├── convide_2d_scenarios/
├── convide_3d_scenarios/
├── convide_4d_scenarios/
├── constrained_2d_scenarios/
├── constrained_3d_scenarios/
├── constrained_4d_scenarios/
├── extreme_2d/
├── extreme_3d/
└── extreme_4d/
```

---

## Intervention Types

### Forward Direction (Uncertainty → Inconsistency)
- `widen` - Increase uncertainty by scaling generators
- `shrink` - Decrease uncertainty
- `correlate` - Add correlation between generators

### Reverse Direction (Inconsistency → Uncertainty) ✨ NEW
- `shift_target` - Move target center to create mismatch
- `scale_mapping` - Scale mapping matrix
- `break_correspondence` - Rotate mapping to break alignment

---

## Usage Examples

### Generate CONVIDE data:
```matlab
cd examples/
run('generate_convide_scenarios.m')
```

### Quick demo:
```matlab
run('demo_quick_start.m')
```

### Full validation:
```matlab
run('test_bidirectional_causality.m')
```

---

## Output Data Format

Each experiment generates a JSON record with:

```json
{
  "pre_state": {
    "uncertainty": {...},
    "inconsistency": {
      "jaccard_index": 0.85,
      "empty_intersection": false,
      "vol_intersection": 42.3,
      "vol_union": 49.8
    }
  },
  "post_state": {...},
  "causal_effect": {
    "inconsistency": {
      "delta_jaccard": -0.13,
      "delta_empty": 0
    },
    "uncertainty": {
      "volume_change": 13.3,
      "volume_change_pct": 10.6
    }
  }
}
```

---

**Version**: 1.1 (Bidirectional)  
**Last Updated**: December 2025
