# CONVIDE Scenarios - Engineering Applications

**Last Updated**: January 27, 2026  
**Status**: Production - 12 Scenarios Implemented

## Overview

**CONVIDE** (CONstrained VIrtual DEvelopment) scenarios provide physically realistic engineering contexts for testing causal uncertainty-inconsistency relationships. Each scenario represents a real cyber-physical system with:
- Authentic propagation rules (UPRs)
- Engineering-relevant constraints
- Realistic uncertainty characteristics

---

## Scenario Structure

### Common Format

```matlab
scenario = struct();
scenario.id = 'convide_2d_1';
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

### Scenario 1: Robot End-Effector Positioning
**Context**: 2-DOF planar robot arm with joint angle uncertainty
```matlab
% Mapping: Forward kinematics
F = [L1*cos(θ1) + L2*cos(θ1+θ2), -L1*sin(θ1) - L2*sin(θ1+θ2);
     L1*sin(θ1) + L2*sin(θ1+θ2),  L1*cos(θ1) + L2*cos(θ1+θ2)]
```
**Uncertainty Source**: Joint encoder noise, backlash  
**Constraint**: Workspace safety boundary (avoid collision zone)  
**Engineering Goal**: Ensure end-effector stays in safe region despite uncertainty

### Scenario 2: Thermal Sensor Calibration
**Context**: Temperature-pressure sensor with cross-sensitivity
```matlab
% Mapping: Coupled sensor response
F = [α_TT, α_TP;
     α_PT, α_PP]  % Sensitivity coefficients
```
**Uncertainty Source**: Sensor noise, thermal drift  
**Constraint**: Operating range limits (safety thresholds)  
**Engineering Goal**: Verify measurements within specification

### Scenario 3: Computer Vision Tracking
**Context**: Pixel-to-world coordinate transformation
```matlab
% Mapping: Homography + lens distortion
F = intrinsic_matrix * extrinsic_matrix
```
**Uncertainty Source**: Pixel quantization, lens distortion  
**Constraint**: Target detection zone  
**Engineering Goal**: Maintain tracking lock despite image noise

### Scenario 4: Control Design Conflict
**Context**: Observer-based state estimation
```matlab
% Mapping: Kalman filter prediction step
F = A_system  % System dynamics matrix
```
**Uncertainty Source**: Process noise, model mismatch  
**Constraint**: State constraints for stability  
**Engineering Goal**: Ensure estimated state respects physical limits

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

Each scenario is tested with **3 interventions × 20 parameter values**:

### 1. Widen (Uncertainty Increase)
```matlab
scales = logspace(log10(0.2), log10(20), 20);
% Tests: How does increasing uncertainty affect inconsistency?
```

### 2. Shrink (Uncertainty Decrease)  
```matlab
scales = logspace(log10(0.05), log10(5), 20);
% Tests: Does reducing uncertainty improve consistency?
```

### 3. Rotate (Uncertainty Reorientation)
```matlab
angles = linspace(0, 2*pi, 20);
% Tests: Does uncertainty direction matter for inconsistency?
```

**Total Experiments**: 12 scenarios × 3 interventions × 20 points = **720 experiments**

---

## Data Generation

```matlab
% Generate all CONVIDE scenarios
generate_convide_scenarios('output_dir', 'data/convide_2d_scenarios', ...
                           'dimensions', 2);
generate_convide_scenarios('output_dir', 'data/convide_3d_scenarios', ...
                           'dimensions', 3);
generate_convide_scenarios('output_dir', 'data/convide_4d_scenarios', ...
                           'dimensions', 4);
```

**Output Format**: JSON files per scenario with complete intervention results

---

## Scenario Properties

### Metadata Fields

Each scenario includes:
- `id`: Unique identifier (e.g., 'convide_2d_1')
- `name`: Human-readable description
- `dimension`: 2, 3, or 4
- `family`: 'forward' (all CONVIDE scenarios)
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
├── convide_2d_scenarios/
│   ├── scenario_2d_1.json    # Robot end-effector
│   ├── scenario_2d_2.json    # Thermal sensor
│   ├── scenario_2d_3.json    # Vision tracking
│   └── scenario_2d_4.json    # Control design
├── convide_3d_scenarios/
│   ├── scenario_3d_5.json    # IMU orientation
│   ├── scenario_3d_6.json    # Multi-sensor fusion
│   ├── scenario_3d_7.json    # Robotic welding
│   └── scenario_3d_8.json    # Chemical reactor
└── convide_4d_scenarios/
    ├── scenario_4d_9.json    # Quadrotor
    ├── scenario_4d_10.json   # Power grid
    ├── scenario_4d_11.json   # Spacecraft
    └── scenario_4d_12.json   # Hydraulic system
```

---

## Usage Example

```matlab
% Load scenario
scenario = jsondecode(fileread('data/convide_2d_scenarios/scenario_2d_1.json'));

% Run intervention
params = struct('scale_factor', 5.0);
result = causal_experiment_engine.run_intervention(scenario, 'widen', params);

% Extract causal effect
fprintf('Δ-Jaccard: %.4f\n', result.causal_effect.delta_jaccard);
fprintf('Δ-Volume: %.4f\n', result.causal_effect.delta_source_volume);
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

- [examples/generate_convide_scenarios.m](../examples/generate_convide_scenarios.m) - Generation script
- [FRAMEWORK_OVERVIEW.md](FRAMEWORK_OVERVIEW.md) - Overall framework
- [METHODOLOGY.md](METHODOLOGY.md) - Experimental methodology
