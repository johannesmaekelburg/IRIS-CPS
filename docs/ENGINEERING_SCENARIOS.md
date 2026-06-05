# Engineering Scenarios - Engineering Applications

**Last Updated**: February 9, 2026  
**Status**: Production - 8 Scenarios Implemented (4 × 2D, 4 × 3D)

## Overview

**Engineering** (CONstrained VIrtual DEvelopment) scenarios provide physically realistic engineering contexts for testing causal uncertainty-inconsistency relationships. Each scenario represents a real cyber-physical system with:
- Authentic propagation rules (UPRs)
- Engineering-relevant constraints
- Realistic uncertainty characteristics

---

## Scenario Structure

### Common Format

```matlab
scenario = struct();
scenario.id = 'engineering_2d_1';
scenario.name = 'Robot End-Effector Positioning';
scenario.dimension = 2;
scenario.family = 'forward';
scenario.source = conPolyZono(...);      % Uncertain measurement
scenario.target = conPolyZono(...);      % Safety/quality constraint
scenario.mapping = struct('F', F, 'f', f);  % Physical propagation rule
scenario.engineering_context = '...';
scenario.constraint_source = '...';
```

---

## 2D Scenarios (1-4)

### Scenario 1: CAD Export Drift (Type A Causality)
**Context**: Measurement uncertainty affecting constraint satisfaction
**Engineering Application**: Design parameter uncertainty propagating through model transformations
**Uncertainty Source**: CAD export tolerances, numerical precision  
**Constraint**: Design specifications, safety margins  
**Engineering Goal**: Ensure exported models remain within specifications
**Causality Type**: Uncertainty-driven inconsistency

### Scenario 2: MBSE Version Mismatch (Type B)
**Context**: Model-Based Systems Engineering synchronization issues
**Engineering Application**: Multi-tool integration with version control
**Uncertainty Source**: Model version drift, tool interoperability  
**Constraint**: Interface specifications, requirement traceability  
**Engineering Goal**: Maintain consistency across MBSE toolchains
**Causality Type**: Structural inconsistency

### Scenario 3: Documentation Sync (Type B)
**Context**: Documentation-implementation consistency
**Engineering Application**: Requirements traceability in development lifecycle
**Uncertainty Source**: Manual updates, asynchronous changes  
**Constraint**: Specification compliance, audit requirements  
**Engineering Goal**: Ensure documentation matches implementation
**Causality Type**: Structural inconsistency

### Scenario 4: Control Design Conflict (Type C)
**Context**: Observer-controller co-design conflicts
**Engineering Application**: State estimation vs control objectives
**Uncertainty Source**: Model mismatch, estimation error  
**Constraint**: Stability margins, performance specifications  
**Engineering Goal**: Balance estimation accuracy with control robustness
**Causality Type**: Hybrid uncertainty-structural

---

## 3D Scenarios (5-8)

### Scenario 5: IMU Orientation Estimation
**Context**: 3-axis gyroscope integration with drift
```matlab
% Mapping: Rotation matrix (3D orientation)
F = rotationMatrix(roll, pitch, yaw)
```
**Uncertainty Source**: Gyro bias drift, integration error  
**Constraint**: Attitude limits (e.g., aircraft pitch/roll bounds)  
**Engineering Goal**: Maintain orientation estimate within safe envelope

### Scenario 6: Multi-Sensor Fusion (GPS + IMU + Barometer)
**Context**: Position-velocity-altitude fusion
```matlab
% Mapping: Sensor fusion transformation
F = [R_GPS,   0,     0;
     0,       R_IMU, 0;
     0,       0,     scale_baro]
```
**Uncertainty Source**: GPS multipath, IMU drift, pressure noise  
**Constraint**: Geofence boundaries  
**Engineering Goal**: Keep fused position estimate within allowed region

### Scenario 7: 3D Robotic Welding
**Context**: TCP (Tool Center Point) positioning
```matlab
% Mapping: Forward kinematics (3-DOF)
F = J(q) % Jacobian at configuration q
```
**Uncertainty Source**: Joint compliance, thermal expansion  
**Constraint**: Weld path tolerance zone  
**Engineering Goal**: Ensure weld accuracy despite mechanical uncertainties

### Scenario 8: Chemical Reactor Monitoring
**Context**: Temperature-pressure-concentration dynamics
```matlab
% Mapping: Thermodynamic coupling
F = [∂T/∂T, ∂T/∂P, ∂T/∂C;
     ∂P/∂T, ∂P/∂P, ∂P/∂C;
     ∂C/∂T, ∂C/∂P, ∂C/∂C]
```
**Uncertainty Source**: Sensor noise, reaction kinetics uncertainty  
**Constraint**: Safe operating envelope  
**Engineering Goal**: Stay within explosion/runaway boundaries

---

## 4D Scenarios (9-12)

### Scenario 9: Quadrotor State Estimation
**Context**: 4D state (x, y, z, yaw) from noisy sensors
```matlab
% Mapping: Sensor-to-state transformation
F = [R_GPS,  0;
     0,      compass_rotation]
```
**Uncertainty Source**: GPS noise, compass deviation, accelerometer bias  
**Constraint**: Collision avoidance zone  
**Engineering Goal**: Maintain safe separation from obstacles

### Scenario 10: Power Grid Monitoring
**Context**: 4-bus power system state estimation
```matlab
% Mapping: Power flow equations (linearized)
F = Y_bus % Admittance matrix
```
**Uncertainty Source**: PMU measurement noise, line parameter uncertainty  
**Constraint**: Voltage stability limits  
**Engineering Goal**: Ensure voltages within ±5% nominal

### Scenario 11: Spacecraft Attitude Control
**Context**: Quaternion-based orientation (4D unit sphere)
```matlab
% Mapping: Quaternion propagation
F = omega_to_quaternion_matrix(ω)
```
**Uncertainty Source**: Star tracker noise, gyro drift  
**Constraint**: Sun/Earth pointing constraints  
**Engineering Goal**: Maintain pointing accuracy for communication

### Scenario 12: Hydraulic System Dynamics
**Context**: 4-state hydraulic actuator (pressure, flow, position, velocity)
```matlab
% Mapping: State-space dynamics
F = exp(A*dt) % Discretized system matrix
```
**Uncertainty Source**: Leakage, fluid property variation  
**Constraint**: Actuator stroke limits, pressure limits  
**Engineering Goal**: Prevent hydraulic system damage from excursions

---

## Intervention Testing

Each scenario is tested with **3 interventions × 20-85 parameter values**:

### 1. Widen (Uncertainty Increase)
```matlab
% Varies by scenario, typically:
scales = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, ...];
% Tests: How does increasing uncertainty affect I_theta and inconsistency?
```

### 2. Shrink (Uncertainty Decrease)  
```matlab
scales = [0.1, 0.2, 0.3, ...];
% Tests: Does reducing uncertainty improve consistency?
```

### 3. Correlate (Structural Dependency)
```matlab
correlation_strength = [0.0, 0.1, 0.2, ..., 0.999];
% Tests: Effect of generator correlation on inconsistency
```

**Total Experiments**: 8 scenarios × 3 interventions × ~20-85 points = **~600+ experiments**

**Repeats**: 5 repeats per experiment for statistical robustness  
**Monte Carlo**: 300 samples per I_theta computation

---

## Data Generation

```matlab
% Generate all Engineering scenarios
generate_engineering_scenarios('output_dir', 'data/engineering_2d_scenarios', ...
                           'dimensions', 2);
generate_engineering_scenarios('output_dir', 'data/engineering_3d_scenarios', ...
                           'dimensions', 3);
generate_engineering_scenarios('output_dir', 'data/engineering_4d_scenarios', ...
                           'dimensions', 4);
```

**Output Format**: JSON files per scenario with complete intervention results

---

## Scenario Properties

### Metadata Fields

Each scenario includes:
- `id`: Unique identifier (e.g., 'engineering_2d_1')
- `name`: Human-readable description
- `dimension`: 2, 3, or 4
- `family`: 'forward' (all Engineering scenarios)
- `engineering_context`: Real-world application
- `constraint_source`: Physical/safety constraint rationale
- `uncertainty_sources`: List of physical uncertainty contributors

---

## Engineering Validity

✅ **Realistic Propagation Rules**: Based on actual physical laws  
✅ **Meaningful Constraints**: Derived from safety/quality requirements  
✅ **Authentic Uncertainty**: Modeled from sensor specs and system identification  
✅ **Actionable Results**: Causal insights applicable to real designs  

---

## Data Structure

```
data/
├── engineering_with_I_theta/
│   ├── results_engineering_2d_scenario_1.json    # CAD Export Drift
│   ├── results_engineering_2d_scenario_2.json    # MBSE Version Mismatch
│   ├── results_engineering_2d_scenario_3.json    # Documentation Sync
│   └── results_engineering_2d_scenario_4.json    # Control Design Conflict
└── engineering_balanced/
    ├── results_engineering_3d_scenario_5.json    # IMU orientation (3D)
    ├── results_engineering_3d_scenario_6.json    # Multi-sensor fusion (3D)
    ├── results_engineering_3d_scenario_7.json    # Robotic welding (3D)
    └── results_engineering_3d_scenario_8.json    # Chemical reactor (3D)
```

### JSON Format

```json
{
  "experiments": [
    {
      "intervention": "widen",
      "param_value": 0.5,
      "pre_uncertainty": {
        "source_volume": 25,
        "source_radius": 2.5,
        "source_center": [100, 50],
        "source_n_generators": 2,
        "source_correlation": 0
      },
      "post_uncertainty": { ... },
      "pre_inconsistency": {
        "I_theta": 0.98,
        "I_theta_se": 0.008,
        "I_theta_ci95_lower": 0.957,
        "I_theta_ci95_upper": 0.991,
        "jaccard_index": 0.17,
        "mc_probability": 0.027,
        "mc_num_samples": 300,
        "empty_intersection": false
      },
      "post_inconsistency": { ... },
      "causal_effect": {
        "delta_I_theta": -0.98,
        "delta_jaccard": -0.17,
        "delta_volume": -18.75
      }
    }
  ]
}
```

---

## Usage Example

### MATLAB: Generate Data
```matlab
% Generate Engineering scenarios
generate_engineering_examples
```

### Python: Sensitivity Analysis
```bash
# Analyze 2D data
python src/sensitivity_analysis.py --data_dir data/engineering_with_I_theta --output_dir results/sensitivity_2d --param param_value

# Analyze 3D data
python src/sensitivity_analysis.py --data_dir data/engineering_balanced --output_dir results/sensitivity_3d --param param_value
```

### Load and Inspect Results
```python
import json

# Load data
with open('data/engineering_with_I_theta/results_engineering_2d_scenario_1.json') as f:
    data = json.load(f)

# Extract I_theta values
I_theta_values = [exp['pre_inconsistency']['I_theta'] for exp in data['experiments']]
print(f'I_theta range: [{min(I_theta_values):.3f}, {max(I_theta_values):.3f}]')
```

---

## Scenario Design Principles

1. **Physical Realism**: All mappings derived from first principles or system ID
2. **Constraint Justification**: Every constraint has engineering rationale  
3. **Diverse Complexity**: Range from simple (2D) to complex (4D) dynamics
4. **Testable Hypotheses**: Each scenario enables specific causal questions
5. **Reproducibility**: Complete parameter documentation for replication

---

## See Also

- [examples/generate_engineering_scenarios.m](../examples/generate_engineering_scenarios.m) - Generation script
- [FRAMEWORK_OVERVIEW.md](FRAMEWORK_OVERVIEW.md) - Overall framework
- [METHODOLOGY.md](METHODOLOGY.md) - Experimental methodology
