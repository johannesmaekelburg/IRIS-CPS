% GENERATE CPS DOMAIN SCENARIOS - TWO STEP WORKFLOW
%
% Eight additional CPS domains from real-world datasets, each contributing
% four 2-D uncertainty scenarios. Uncertainty bounds are taken directly from
% the JSON files in data/CPS-extra-domains/ (open datasets / literature).
%
% Domains  (4 scenarios each, IDs 1–36):
%   1–4   Automotive             (KaRaceIng / EV thermal, battery, mechanical)
%   5–8   Building HVAC          (EnergyPlus, ISO 7730)
%   9–12  Industrial Robot       (ISO 10218, welding, assembly)
%   13–16 Medical Device         (IEC 60601, insulin pump, ventilator, …)
%   17–20 Railway                (EN 15227 / SIL, braking, signalling, …)
%   21–24 Satellite Aerospace    (ECSS, attitude control, power, comms)
%   25–28 Smart Grid             (IEEE 1547, ENTSO-E, protection, …)
%   29–32 Water / Chemical       (SWaT dataset, Tennessee-Eastman, …)
%   33–36 Wind Turbine           (FAST / IEC 61400, fatigue, SCADA, …)
%
% Zonotope design (matches Formula Student / Engineering convention):
%   Each 2-D scenario pairs two physically related variables from the same
%   domain subsystem.  The source (M1) uses bounds / σ from the JSON directly:
%     interval:      c = (min+max)/2,  g = (max-min)/2
%     probabilistic: c = µ,            g = 1.96·σ  (95 % CI)
%   The target (M2) is derived from the fixed requirement threshold:
%     for "M1 ≤ threshold":  target_c = threshold − 0.25·g,  target_g = 0.25·g
%     for "M1 ≥ threshold":  target_c = threshold + 0.25·g,  target_g = 0.25·g
%   This guarantees a partial but non-trivial intersection for every scenario.
%
% Version: 1.0  —  April 2026

%% Setup
clear; close all; clc;

warning('off', 'CORA:deprecated');
warning('off', 'CORA:contSet:set:expMat');
warning('off', 'CORA:contSet:get:expMat');
warning('off', 'all');
lastwarn('');

%% ═══════════════════════════════════════════════════════════════
%%   CONFIGURATION
%% ═══════════════════════════════════════════════════════════════
run_step1 = true;    % Set false to skip zonotope generation (re-use existing)
run_step2 = true;    % Set false to skip consistency computation

use_saltelli_mode = true;   % true = Saltelli compound interventions
seed = 2025;
rng(seed);

% Consistency scoring (Step 2)
consistency_method   = 'aabb_and_mc';  % AABB Jaccard + MC Probability (includes MFMC)
mc_samples           = 1000;
sampling_methods     = {'sobol'};  % Sobol QMC (matches Saltelli sampling scheme)
use_parallel         = true;
n_workers            = 20;

% Scenarios to process (1–72)
% 1–36:  sequential pairs from original 8 entries  (CPS1–36)
% 37–45: coupled 2-D entries (entry 8 per domain)  (CPS37–45)
% 46–72: new sequential pairs from entries 9–14    (CPS46–72)
scenarios_to_run = 1:72;

%% ═══════════════════════════════════════════════════════════════
%%   PATHS
%% ═══════════════════════════════════════════════════════════════
script_dir   = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);

addpath(fullfile(project_root, 'src', 'matlab'));

%% Output directories
data_base    = fullfile(project_root, 'data');
zonotope_dir = fullfile(data_base, 'zonotopes_cps_v6');
results_dir  = fullfile(data_base, 'measurements_cps_v6');

if ~exist(data_base,    'dir'), mkdir(data_base);    end
if ~exist(zonotope_dir, 'dir'), mkdir(zonotope_dir); end
if ~exist(results_dir,  'dir'), mkdir(results_dir);  end

fprintf('===========================================\n');
fprintf('TWO-STEP CPS DOMAIN ANALYSIS\n');
fprintf('===========================================\n');
fprintf('Domains: 9  |  Scenarios: 36 seq. + 9 coupled + 27 new = 72 total\n');
fprintf('Zonotopes:    %s\n', zonotope_dir);
fprintf('Results:      %s\n\n', results_dir);

%% ═══════════════════════════════════════════════════════════════
%%   SALTELLI SAMPLES
%% ═══════════════════════════════════════════════════════════════
if use_saltelli_mode
    saltelli_params = {'scale_factor', 'center_delta', 'correlation_strength'};
    saltelli_file   = fullfile(data_base, 'saltelli_samples_3param_v6.csv');
    if ~exist(saltelli_file, 'file')
        % Fall back to unversioned generic file
        saltelli_file = fullfile(data_base, 'saltelli_samples_3param.csv');
    end
    if ~exist(saltelli_file, 'file')
        error('Saltelli CSV not found. Run generate_saltelli_samples.py first.\nExpected: %s', saltelli_file);
    end
    fprintf('Saltelli samples: %s\n', saltelli_file);
    saltelli_data  = readtable(saltelli_file);
    interventions  = struct('mode', 'saltelli', 'samples', saltelli_data, ...
                            'params', {saltelli_params});
    fprintf('  %d compound interventions loaded.\n\n', height(saltelli_data));
end

%% ═══════════════════════════════════════════════════════════════
%%   PARALLEL POOL
%% ═══════════════════════════════════════════════════════════════
if use_parallel
    delete(gcp('nocreate'));
    pool = gcp('nocreate');
    if isempty(pool)
        parpool('local', n_workers);
    end
end

%% ═══════════════════════════════════════════════════════════════
%%   HELPER: build 2-D conPolyZono
%% ═══════════════════════════════════════════════════════════════
% cpz2d(c1,g1,c2,g2) → conZonotope([c1;c2], diag([g1;g2]), [], [])
cpz2d = @(c1,g1,c2,g2) conZonotope([c1; c2], diag([g1; g2]), [], []);

%% ═══════════════════════════════════════════════════════════════
%%   SCENARIO DEFINITIONS (36 scenarios, 9 domains × 4)
%%
%%   Source M1 : uncertainty taken directly from JSON
%%   Target M2 : derived from fixed requirement threshold:
%%     "≤ thr" → target_c = thr − 0.25·g_src,  target_g = 0.25·g_src
%%     "≥ thr" → target_c = thr + 0.25·g_src,  target_g = 0.25·g_src
%% ═══════════════════════════════════════════════════════════════

all_scenarios = load_cps_dataset_pair_scenarios( ...
    fullfile(data_base, 'CPS-uncertainty-dataset'), ...
    fullfile(data_base, 'CPS-uncertainty-dataset-full'));
fprintf('Loaded %d CPS scenarios from raw JSON CR definitions.\n', length(all_scenarios));

%% ???????????????????????????????????????????????????????????????????????
%%   STEP 1: GENERATE ZONOTOPES
%% ═══════════════════════════════════════════════════════════════
if run_step1
    fprintf('\n===========================================\n');
    fprintf('STEP 1: GENERATING ZONOTOPES  (%d scenarios)\n', length(all_scenarios));
    fprintf('===========================================\n');

    gen_options = struct('n_repeats', 1, 'verbose', true, 'use_parallel', use_parallel);

    for k = 1:length(all_scenarios)
        scenario_def = all_scenarios{k};
        if ~ismember(scenario_def.id, scenarios_to_run)
            continue;
        end
        fprintf('\n  [%2d/%2d] Scenario %d: %s\n', k, length(all_scenarios), scenario_def.id, scenario_def.name);
        try
            causal_experiment_engine_twostep.generate_and_save_zonotopes( ...
                scenario_def, interventions, zonotope_dir, gen_options);
            fprintf('        + done\n');
        catch ME
            fprintf('        x ERROR: %s\n', ME.message);
        end
    end

    fprintf('\n===========================================\n');
    fprintf('STEP 1 COMPLETE → %s\n', zonotope_dir);
    fprintf('===========================================\n');
end

%% ═══════════════════════════════════════════════════════════════
%%   STEP 2: COMPUTE CONSISTENCY SCORES
%% ═══════════════════════════════════════════════════════════════
if run_step2
    fprintf('\n===========================================\n');
    fprintf('STEP 2: COMPUTING CONSISTENCY SCORES\n');
    fprintf('===========================================\n');

    consistency_opts = struct();
    consistency_opts.method          = consistency_method;
    consistency_opts.mc_samples      = mc_samples;
    consistency_opts.sampling_method = sampling_methods;
    consistency_opts.verbose         = false;
    consistency_opts.use_parallel    = use_parallel;

    causal_experiment_engine_twostep.compute_and_save_consistency( ...
        zonotope_dir, results_dir, consistency_opts);

    fprintf('\n===========================================\n');
    fprintf('STEP 2 COMPLETE → %s\n', results_dir);
    fprintf('===========================================\n');
end

%% Summary
fprintf('\n===========================================\n');
fprintf('CPS DOMAIN TWO-STEP WORKFLOW COMPLETE\n');
fprintf('===========================================\n');
fprintf('Zonotopes:    %s\n', zonotope_dir);
fprintf('Measurements: %s\n', results_dir);
fprintf('\nNext: python src/run_analysis.py --data data/measurements_cps\n');
fprintf('===========================================\n\n');


%% Local helper functions
function scenarios = load_cps_dataset_pair_scenarios(~, ~)
%LOAD_CPS_DATASET_PAIR_SCENARIOS Build all 72 CPS scenarios (fully hardcoded).
%
% Pass 1 (CPS 1–36):  sequential pairs from original 8 entries per domain.
% Pass 2 (CPS 37–45): coupled 2-D entries (entry 8 per domain).
% Pass 3 (CPS 46–72): new sequential pairs from entries 9–14 per domain.
%
% All values taken directly from the JSON dataset files.
% MATLAB jsondecode cannot reliably unify heterogeneous JSON struct arrays.
%
% Columns: name, scenario_id, src_c, src_g, tgt_c, tgt_g, scale, offset, upr_type
%
% For coupled entries (Pass 2) L=dim0, R=dim1 of the same physical entry.
% For new pairs (Pass 3) target zonotopes synthesised via 25% overlap rule:
%   op <= :  tgt_c = fv - 0.25*src_g,  tgt_g = src_g
%   op >= :  tgt_c = fv + 0.25*src_g,  tgt_g = src_g

    domains = { ...
        'Automotive', 'automotive_full.json', { ...
            'Motor Cooling',                   'auto_01_motor_cooling',          106.688,   8.0,    102.0,   8.0,   0.96, 2.0,  'parametric';
            'Radiator Mounting',               'auto_02_radiator_mounting',        1.8,     0.6,      2.0,   0.4,   1.0,  0.0,  'parametric';
            'Battery Overvoltage',             'auto_03_battery_overvoltage',      1.4411,  1.35,     0.65,  1.35,  1.0,  0.0,  'identity';
            'Wheel Carrier Manufacturing',     'auto_04_wheel_carrier',           10.0207,  0.05,    10.05,  0.05,  1.0,  0.0,  'identity';
            'Aerodynamic Part Quality',        'auto_05_aerodynamic_quality',      0.0,     2.352,    0.5,   1.5,   1.0,  0.0,  'identity_bidir';
            'Software Architecture Migration', 'auto_06_software_migration',      82.0,     8.0,     85.0,   5.0,   1.0,  0.0,  'identity_bidir';
            'LiDAR Sensor Quality',            'auto_07_lidar_quality',            0.924343,0.0588,   0.9588,0.0588, 0.95, 0.0,  'structural';
            'Power Electronics Efficiency',    'auto_08_power_electronics',       80.01449, 0.035,   80.035, 0.035,  0.97, 0.0,  'structural';
        }; ...
        'Building HVAC', 'building_hvac_full.json', { ...
            'Room Temperature Control',        'bldg_01_room_temperature',        23.172,   2.0,     22.0,   2.0,   0.99, 0.3,  'guarded';
            'CO2 Concentration',               'bldg_02_co2_concentration',      923.41,  185.0,    815.0, 185.0,   1.04, 0.0,  'guarded';
            'Chiller COP',                     'bldg_03_chiller_cop',              4.124576,0.784,    4.584, 0.784,  1.0,  0.0,  'constraint_based';
            'Pipe Pressure Drop',              'bldg_04_pipe_pressure_drop',   21068.5,  2250.0,  19750.0,2250.0,   1.0,  0.0,  'constraint_based';
            'Facade U-Value',                  'bldg_05_facade_u_value',           0.18551, 0.035,   0.165, 0.035,  1.06, 0.0,  'disambiguation';
            'Sprinkler Flow Rate',             'bldg_06_fire_suppression_flow',   84.14,   10.0,     90.0,  10.0,   0.97, 0.0,  'disambiguation';
            'Elevator Load',                   'bldg_07_elevator_load',           23.4475,  3.75,    21.25,  3.75,  1.02, 0.5,  'parametric';
            'BMS Latency',                     'bldg_08_bms_response_time',       16.481,   8.5,     11.5,   8.5,   1.0,  0.0,  'parametric';
        }; ...
        'Industrial Robot', 'industrial_robot_full.json', { ...
            'TCP Position Error',              'robot_01_tcp_position',            0.125657,0.0588,   0.0912,0.0588, 1.0,  0.0,  'identity';
            'Wrist Torque',                    'robot_02_joint_torque',           51.688,   8.0,     47.0,   8.0,   1.0,  0.0,  'identity';
            'Contact Force',                   'robot_03_collaborative_force',    94.475,  37.5,     72.5,  37.5,   1.0,  0.0,  'identity_bidir';
            'Cycle Time',                      'robot_04_cycle_time',             15.2548,  1.8,     14.2,   1.8,   1.0,  0.0,  'identity_bidir';
            'Weld Heat Input',                 'robot_05_welding_heat_input',      0.60032, 0.12,     0.53,  0.12,  0.96, 0.0,  'structural';
            'Vision Calibration Error',        'robot_06_vision_calibration',      0.167542,0.0784,  0.1216,0.0784, 1.0,  0.0,  'structural';
            'End-Effector Mass',               'robot_07_end_effector_mass',       4.8344,  0.4,      4.6,   0.4,   1.0,  0.0,  'guarded';
            'Bus Jitter',                      'robot_08_controller_latency',      6.688,   8.0,      2.0,   8.0,   1.08, 0.0,  'guarded';
        }; ...
        'Medical Device', 'medical_device_full.json', { ...
            'Insulin Dose',                    'med_01_insulin_dose',              5.3344,  0.4,      5.1,   0.4,   1.0,  0.0,  'constraint_based';
            'Ventilator Pressure',             'med_02_ventilator_pressure',      27.15996, 6.86,    23.14,  6.86,  1.0,  0.0,  'constraint_based';
            'Pacemaker Sensing',               'med_03_pacemaker_sensing',         6.704024,4.116,   9.116, 4.116,  0.94, 0.0,  'disambiguation';
            'Infusion Pressure',               'med_04_infusion_occlusion',      271.02,   70.0,    230.0,  70.0,  1.0,  0.0,  'disambiguation';
            'Defibrillator Energy',            'med_05_defibrillator_energy',    193.79,   15.0,    185.0,  15.0,  0.93, 2.0,  'parametric';
            'Radiation Dose',                  'med_06_radiation_dose',           59.21716, 2.94,    60.94,  2.94,  0.98, 0.0,  'parametric';
            'Surgical Force',                  'med_07_surgical_robot_force',      2.5032,  1.2,      1.8,   1.2,   1.0,  0.0,  'identity';
            'Drug Concentration',              'med_08_drug_concentration',       15.727968,5.488,   12.512, 5.488,  1.0,  0.0,  'identity';
        }; ...
        'Railway', 'railway_full.json', { ...
            'Braking Distance',                'rail_01_braking_distance',       946.18,  130.0,    870.0, 130.0,  1.0,  0.0,  'identity_bidir';
            'Axle Load',                       'rail_02_axle_load',             160.032,   12.0,    153.0,  12.0,  1.0,  0.0,  'identity_bidir';
            'Pantograph Force',                'rail_03_pantograph_force',       151.02,   70.0,    110.0,  70.0,  0.96, 0.5,  'structural';
            'GNSS Error',                      'rail_04_train_positioning',        4.269704,1.764,   3.236, 1.764,  1.08, 0.0,  'structural';
            'Door Gap',                        'rail_05_door_gap',               83.44,   40.0,     60.0,  40.0,  1.0,  0.0,  'guarded';
            'Signalling Latency',              'rail_06_signalling_latency',     417.2,  200.0,    300.0, 200.0,  1.1,  0.0,  'guarded';
            'Traction Energy',                 'rail_07_traction_energy',         31.916528,7.448,  27.552, 7.448,  1.0,  0.0,  'constraint_based';
            'Switch Heating Power',            'rail_08_switch_heating_power',  2234.4,  400.0,   2000.0, 400.0,  1.0,  0.0,  'constraint_based';
        }; ...
        'Satellite', 'satellite_aerospace_full.json', { ...
            'Attitude Error',                  'sat_01_attitude_error',            4.513136,1.176,   3.824, 1.176,  1.07, 0.0,  'disambiguation';
            'Battery DoD',                     'sat_02_battery_depth_of_discharge',37.102,  7.0,    33.0,   7.0,   1.0,  0.0,  'disambiguation';
            'Panel Temperature',               'sat_03_thermal_panel_temperature', 85.032,  12.0,   78.0,  12.0,   0.98, 1.5,  'parametric';
            'Propellant Mass',                 'sat_04_propellant_mass',           41.0971, 2.65,   42.65,  2.65,  1.0,  0.0,  'parametric';
            'Downlink Rate',                   'sat_05_downlink_data_rate',        23.408048,8.232, 28.232, 8.232,  1.0,  0.0,  'identity';
            'Reaction Wheel Torque',           'sat_06_reaction_wheel_torque',     20.9315, 2.25,   22.25,  2.25,  1.0,  0.0,  'identity';
            'Natural Frequency',               'sat_07_structural_frequency',      31.1385, 2.75,   32.75,  2.75,  1.0,  0.0,  'identity_bidir';
            'RAM Usage',                       'sat_08_onboard_sw_memory',         51.688,  8.0,    47.0,   8.0,   1.0,  0.0,  'identity_bidir';
        }; ...
        'Smart Grid', 'smart_grid_full.json', { ...
            'Frequency Nadir',                 'grid_01_frequency_deviation',      49.1035, 0.25,   49.25,  0.25,  1.0,  0.01, 'structural';
            'Transformer Load',                'grid_02_transformer_loading',     363.4852,88.2,   311.8,  88.2,   0.98, 0.0,  'structural';
            'Bus Voltage',                     'grid_03_voltage_deviation',         1.0793, 0.05,    1.05,  0.05,  1.01, 0.0,  'guarded';
            'Battery SOC',                     'grid_04_battery_storage_soc',      23.519,  8.5,    28.5,   8.5,   0.95, 2.0,  'guarded';
            'Relay Time',                      'grid_05_protection_relay_time',    72.548,  18.0,   62.0,  18.0,   1.0,  0.0,  'constraint_based';
            'Demand Response',                 'grid_06_demand_response_reduction',41.49152,15.68,  50.68, 15.68,  1.0,  0.0,  'constraint_based';
            'Cable Ampacity',                  'grid_07_cable_ampacity',          412.42,   30.0,  430.0,  30.0,   1.03, 0.0,  'disambiguation';
            'Meter Latency',                   'grid_08_smart_meter_latency',      23.79,   15.0,   15.0,  15.0,   1.0,  0.0,  'disambiguation';
        }; ...
        'Water & Chemical', 'water_chemical_process_full.json', { ...
            'Tank Level (SWaT)',               'proc_01_swat_tank_level',         879.90688,290.08, 709.92,290.08,  0.97, 1.0,  'parametric';
            'Inlet Flow (SWaT)',               'proc_02_swat_flow_rate',            0.65525, 0.375,  0.875, 0.375,  1.04, 0.0,  'parametric';
            'UF Pressure Drop (SWaT)',         'proc_03_swat_uf_pressure_drop',    33.79,   15.0,   25.0,  15.0,   1.0,  0.0,  'identity';
            'Chlorine Conc. (SWaT)',           'proc_04_swat_water_quality_chlorine',0.297373,0.2352,0.4352,0.2352, 1.0,  0.0,  'identity';
            'Reactor Temp (TEP)',              'proc_05_tep_reactor_temperature', 122.153475,1.8032,121.0968,1.8032,1.0,  0.0,  'identity_bidir';
            'Separator Pressure (TEP)',        'proc_06_tep_separator_pressure',   52.005022,1.1956,51.3044,1.1956, 1.0,  0.0,  'identity_bidir';
            'Distillation Temp',              'proc_07_distillation_column_temperature',80.039408,3.528,77.972,3.528,1.01,0.5, 'structural';
            'Actuator Pressure',               'proc_08_zema_actuator_pressure',    5.715996,0.686,  5.314, 0.686,  0.98, 0.0,  'structural';
        }; ...
        'Wind Turbine', 'wind_turbine_full.json', { ...
            'Blade Fatigue',                   'wind_01_blade_fatigue',         2146059.2,352800.0,2352800.0,352800.0,0.9, 0.0, 'guarded';
            'Nacelle Vibration',               'wind_02_nacelle_vibration',        3.9411,  1.35,    3.15,  1.35,  1.06, 0.0,  'guarded';
            'Tower Deflection',                'wind_03_tower_deflection',          0.54825, 0.125,  0.475, 0.125,  1.0,  0.0,  'constraint_based';
            'Power Deviation',                 'wind_04_power_curve',              -3.37712, 3.92,  -1.08,  3.92,   1.0,  0.0,  'constraint_based';
            'Pitch Response',                  'wind_05_pitch_actuator',            4.4618,  1.3,    3.7,   1.3,   1.04, 0.0,  'disambiguation';
            'Gearbox Oil Temp',                'wind_06_gearbox_oil_temp',         71.688,   8.0,   67.0,   8.0,   0.99, 1.0,  'disambiguation';
            'Foundation Settlement',           'wind_07_foundation_settlement',     2.513136, 1.176,  1.824, 1.176,  1.05, 0.0,  'parametric';
            'SCADA Latency',                   'wind_08_scada_latency',            84.475,  37.5,   62.5,  37.5,   1.0,  0.0,  'parametric';
        }; ...
    };

    % ── Pass 2: Coupled 2-D entries (one per domain, CPS37–45) ─────────────
    % Each row: {name_dim0, id, src_c0, src_g0, tgt_c0, tgt_g0, ...
    %                            src_c1, src_g1, tgt_c1, tgt_g1, upr_type}
    % M2 targets: pre-computed for dim0 where available; 25% rule for dim1.
    % op=<= : tgt_c = fv - 0.25*src_g   |   op=>= : tgt_c = fv + 0.25*src_g
    coupled_domains = { ...
        'Automotive',      'automotive_full.json',          'Brake Disc Geometry Tolerance', ...
            300.0,   5.0,  298.25, 1.25,   28.0,   1.5,  26.375,  1.5,  'constraint_based'; ...
        'Building HVAC',   'building_hvac_full.json',       'Room Temp & CO2 Ventilation Coupling', ...
            21.5,    1.96,  23.51, 0.49,  850.0, 196.0,  951.0,  196.0, 'constraint_based'; ...
        'Industrial Robot','industrial_robot_full.json',    'TCP X/Y Positioning Error', ...
            0.0,    0.0784, 0.1304,0.0784,  0.0,  0.0686, 0.1329, 0.0686,'constraint_based'; ...
        'Medical Device',  'medical_device_full.json',      'Ventilator Pressure & Tidal Volume', ...
            24.0,   6.86,  28.285, 1.715, 420.0,  78.4,  470.4,   78.4, 'constraint_based'; ...
        'Railway',         'railway_full.json',             'Braking Distance & Entry Speed', ...
            950.0, 127.4, 968.15, 31.85,  12.0,   9.8,   -2.45,   9.8,  'constraint_based'; ...
        'Satellite',       'satellite_aerospace_full.json', 'Solar Panel Power & Battery SOC', ...
            18.5,   4.312, 15.078, 1.078,  32.0,  11.76,  22.94,  11.76, 'constraint_based'; ...
        'Smart Grid',      'smart_grid_full.json',          'Frequency Nadir & BESS Response', ...
            49.55,  0.294, 49.0735,0.0735, 18.0,   7.84,  11.96,   7.84, 'constraint_based'; ...
        'Water & Chemical','water_chemical_process_full.json','Tank Level & Inlet Flow Coupling', ...
            525.0, 290.08, 927.48, 72.52,  1.175,  0.375,  0.40625,0.375,'constraint_based'; ...
        'Wind Turbine',    'wind_turbine_full.json',        'Blade Fatigue & Tower Deflection', ...
            2500000.0,352800.0,2088200.0,88200.0, 0.54,0.1176,0.6294,0.1176,'constraint_based'; ...
    };

    % ── Pass 3: New sequential pairs from entries 9–14 (CPS46–72) ───────────
    % Targets derived from M1 source geometry using n-D aware P_2D≈0.5 rule:
    %   P_per_dim = 0.5^(1/2) ≈ 0.7071
    %   offset    = 2 * src_g * (1 - 0.7071) ≈ 0.5858 * src_g
    %   op<=: tgt_c = src_c + offset;  op>=: tgt_c = src_c - offset;  tgt_g = src_g
    % Values match data/CPS-uncertainty-dataset-full/*_full.json (M2 zonotopes).
    % Columns: name_L, id_L, src_cL, src_gL, tgt_cL, tgt_gL, scale_L, off_L, upr_L, ...
    %          name_R, id_R, src_cR, src_gR, tgt_cR, tgt_gR, scale_R, off_R, upr_R
    new_pair_domains = { ...
        'Automotive', 'automotive_full.json', { ...
            'Traction Force from Motor Torque',  'auto_10_traction_parametric',  200.0,   25.0,   185.3553,  25.0,  1.0, 0.0, 'parametric'; ...
            'Motor Speed to Wheel Speed',        'auto_11_motor_speed_identity', 11000.0, 15.68, 11009.1851, 15.68, 1.0, 0.0, 'identity'; ...
            'Total Drivetrain Loss',             'auto_12_drivetrain_loss',        26.0,   4.0,    28.3431,   4.0,  1.0, 0.0, 'structural'; ...
            'Front/Rear Brake Balance',          'auto_13_brake_balance',          60.0,   3.0,    61.7574,   3.0,  1.0, 0.0, 'identity_bidir'; ...
            'Cooling Strategy Variant',          'auto_14_cooling_variant',        85.0,   7.0,    89.1005,   7.0,  1.0, 0.0, 'disambiguation'; ...
            'Aerodynamic Drag Coefficient',      'auto_15_aero_drag_parametric',   28.0,   3.92,   30.2963,   3.92, 1.0, 0.0, 'parametric'; ...
        }; ...
        'Building HVAC', 'building_hvac_full.json', { ...
            'Outdoor Temp to Zone Setpoint',     'bldg_10_outdoor_identity',        5.0,   3.0,     6.7574,   3.0,  1.0, 0.0, 'identity'; ...
            'Cooling Load from Solar/Occupancy', 'bldg_11_cooling_parametric',   1300.0, 200.0,  1417.1573, 200.0,  1.0, 0.0, 'parametric'; ...
            'Total Heating Load Structural',     'bldg_12_heating_structural',    850.0, 235.2,   987.7770, 235.2,  1.0, 0.0, 'structural'; ...
            'Supply/Return Air Temp Bidir',      'bldg_13_air_temp_bidir',         16.0,   2.0,    17.1716,   2.0,  1.0, 0.0, 'identity_bidir'; ...
            'Heating System Variant',            'bldg_14_heating_variant',        77.5,   7.5,    81.8934,   7.5,  1.0, 0.0, 'disambiguation'; ...
            'Annual Heating Energy Structural',  'bldg_15_annual_heating',         33.0,   6.86,   37.0185,   6.86, 1.0, 0.0, 'structural'; ...
        }; ...
        'Industrial Robot', 'industrial_robot_full.json', { ...
            'Force Sensor to Joint Torque',      'robot_10_force_identity',         0.0,   1.568,   0.9185,  1.568, 1.0, 0.0, 'identity'; ...
            'Weld Penetration from Heat Input',  'robot_11_weld_parametric',        0.55,  0.07,    0.5910,  0.07,  1.0, 0.0, 'parametric'; ...
            'Total Cycle Time Structural',       'robot_12_cycle_structural',       2.1,   0.3,     2.2757,  0.3,   1.0, 0.0, 'structural'; ...
            'TCP Error and Joint Angle Bidir',   'robot_13_tcp_bidir',              0.0,   0.098,   0.0574,  0.098, 1.0, 0.0, 'identity_bidir'; ...
            'Gripper Type Variant',              'robot_14_gripper_variant',       55.0,  10.0,    49.1421, 10.0,   1.0, 0.0, 'disambiguation'; ...
            'Path Deviation Structural',         'robot_15_path_structural',        0.0,   0.1568,  0.0919,  0.1568,1.0, 0.0, 'structural'; ...
        }; ...
        'Medical Device', 'medical_device_full.json', { ...
            'Drug Clearance from Body Weight',   'med_10_drug_parametric',         75.0,  17.64,   85.3333, 17.64,  1.0, 0.0, 'parametric'; ...
            'Infusion Pump Flow to Catheter',    'med_11_infusion_identity',        5.0,   0.5,     5.2929,  0.5,   1.0, 0.0, 'identity'; ...
            'Total Radiation Dose Structural',   'med_12_radiation_structural',   200.0,   7.84,  204.5926,  7.84,  1.0, 0.0, 'structural'; ...
            'Systolic/Diastolic BP Bidir',       'med_13_bp_bidir',               128.0,  23.52,  141.7777, 23.52,  1.0, 0.0, 'identity_bidir'; ...
            'Ventilation Mode Variant',          'med_14_ventilation_variant',     25.0,   3.0,    26.7574,  3.0,   1.0, 0.0, 'disambiguation'; ...
            'SpO2 from Haemoglobin Sat.',        'med_15_spo2_parametric',         95.2,   4.116,  92.7889,  4.116, 1.0, 0.0, 'parametric'; ...
        }; ...
        'Railway', 'railway_full.json', { ...
            'Track Geometry to Vehicle Dynamics','rail_10_track_identity',          0.0,   8.0,     4.6863,  8.0,   1.0, 0.0, 'identity'; ...
            'Traction Motor Current from Speed', 'rail_11_traction_parametric',     0.92,  0.0588,  0.9544,  0.0588,1.0, 0.0, 'parametric'; ...
            'Total Safe Stopping Distance',      'rail_12_stopping_structural',   950.0, 127.4,  1024.6292,127.4,  1.0, 0.0, 'structural'; ...
            'Entry Speed and Braking Bidir',     'rail_13_entry_speed_bidir',       0.12,  0.0588,  0.0856,  0.0588,1.0, 0.0, 'identity_bidir'; ...
            'Brake System Variant',              'rail_14_brake_variant',         100.0,  15.0,    91.2132, 15.0,   1.0, 0.0, 'disambiguation'; ...
            'Total Wheel-Rail Contact Force',    'rail_15_contact_structural',    165.0,  10.0,   170.8579, 10.0,   1.0, 0.0, 'structural'; ...
        }; ...
        'Satellite', 'satellite_aerospace_full.json', { ...
            'Link Budget Margin from Distance',  'sat_10_link_parametric',        550.0,  49.0,   521.2965, 49.0,   1.0, 0.0, 'parametric'; ...
            'Solar Panel Degradation to Power',  'sat_11_solar_identity',          2.5,   0.5,     2.2071,  0.5,   1.0, 0.0, 'identity'; ...
            'Total Power Consumption Structural','sat_12_power_structural',         1.5,   0.3,     1.6757,  0.3,   1.0, 0.0, 'structural'; ...
            'Solar Power and Battery SOC Bidir', 'sat_13_solar_bidir',            18.5,   4.312,  15.9741,  4.312, 1.0, 0.0, 'identity_bidir'; ...
            'Thruster Mode Variant',             'sat_14_thruster_variant',         0.6,   0.2,     0.4828,  0.2,   1.0, 0.0, 'disambiguation'; ...
            'Satellite Thermal Balance',         'sat_15_thermal_structural',       1.0,   0.2,     1.1172,  0.2,   1.0, 0.0, 'structural'; ...
        }; ...
        'Smart Grid', 'smart_grid_full.json', { ...
            'Primary to Secondary Voltage',      'grid_10_voltage_identity',       11.0,   0.55,   11.3222,  0.55,  1.0, 0.0, 'identity'; ...
            'EV Charging Load from Diversity',   'grid_11_ev_parametric',           0.65,  0.2352,  0.7878,  0.2352,1.0, 0.0, 'parametric'; ...
            'Total Grid Load Structural',        'grid_12_grid_load_structural',    0.0, 117.6,    68.8885,117.6,   1.0, 0.0, 'structural'; ...
            'Grid Frequency and BESS Bidir',     'grid_13_freq_bidir',             49.55,  0.294,  49.3778,  0.294, 1.0, 0.0, 'identity_bidir'; ...
            'Demand Response Variant',           'grid_14_demand_variant',         43.0,   5.0,    40.0711,  5.0,   1.0, 0.0, 'disambiguation'; ...
            'Total FCR Reserve Structural',      'grid_15_fcr_structural',         18.0,   7.84,   13.4074,  7.84,  1.0, 0.0, 'structural'; ...
        }; ...
        'Water & Chemical', 'water_chemical_process_full.json', { ...
            'Chlorine Conc. from Dosing Rate',   'proc_10_chlorine_parametric',     7.0,   2.0,     8.1716,  2.0,   1.0, 0.0, 'parametric'; ...
            'Reactor Temp to Separator Ident.',  'proc_11_reactor_identity',      120.4,   1.8032, 121.4563,  1.8032,1.0, 0.0, 'identity'; ...
            'Total Effluent Contaminant Load',   'proc_12_effluent_structural',     5.0,   3.0,     6.7574,  3.0,   1.0, 0.0, 'structural'; ...
            'Tank Level and Inlet Flow Bidir',   'proc_13_tank_bidir',            525.0, 290.08,  694.9249,290.08,  1.0, 0.0, 'identity_bidir'; ...
            'Treatment Process Variant',         'proc_14_treatment_variant',       0.165, 0.085,   0.2148,  0.085, 1.0, 0.0, 'disambiguation'; ...
            'Pump Head from Flow Rate',          'proc_15_pump_parametric',         1.175, 0.375,   0.9553,  0.375, 1.0, 0.0, 'parametric'; ...
        }; ...
        'Wind Turbine', 'wind_turbine_full.json', { ...
            'Hub Wind Speed to Rotor Thrust',    'wind_10_hub_speed_identity',      9.5,   2.94,   11.2222,  2.94,  1.0, 0.0, 'identity'; ...
            'Power Coeff. from Tip-Speed Ratio', 'wind_11_power_coeff_parametric',  0.795, 0.05,    0.7657,  0.05,  1.0, 0.0, 'parametric'; ...
            'Tower Base Moment Structural',      'wind_12_tower_moment_structural',100.0,  15.0,   108.7868, 15.0,  1.0, 0.0, 'structural'; ...
            'Blade Fatigue and Tower Bidir',     'wind_13_blade_tower_bidir',    4850.0, 627.2,  5217.4053,627.2,  1.0, 0.0, 'identity_bidir'; ...
            'Control Strategy Variant',          'wind_14_control_variant',        15.0,   0.686,  14.5982,  0.686, 1.0, 0.0, 'disambiguation'; ...
            'Annual Energy Yield Structural',    'wind_15_energy_structural',      25.0,   4.312,  22.4741,  4.312, 1.0, 0.0, 'structural'; ...
        }; ...
    };

    scenarios = {};
    next_id   = 1;

    % ── Pass 1: Original sequential pairs (CPS 1–36) ─────────────────────────
    for di = 1:size(domains, 1)
        domain_label = domains{di, 1};
        dataset_file = domains{di, 2};
        entries      = domains{di, 3};

        for pair_idx = 1:2:8
            L = entries(pair_idx,   :);
            R = entries(pair_idx+1, :);

            sl = L{7}; ol  = L{8};
            sr = R{7}; or_ = R{8};

            src = conZonotope([L{3}; R{3}], diag([L{4}, R{4}]), [], []);
            tgt = conZonotope([L{5}; R{5}], diag([L{6}, R{6}]), [], []);

            upr_l = L{9}; upr_r = R{9};
            if strcmp(upr_l, upr_r)
                upr_type = upr_l;
            else
                upr_type = sprintf('%s+%s', upr_l, upr_r);
            end

            rel_l = struct('upr_type', upr_l, 'mapping', struct('scale', sl, 'offset', ol));
            rel_r = struct('upr_type', upr_r, 'mapping', struct('scale', sr, 'offset', or_));

            scenario = struct();
            scenario.id     = next_id;
            scenario.name   = sprintf('%s - %s & %s', domain_label, L{1}, R{1});
            scenario.type   = upr_type;
            scenario.source = src;
            scenario.target = tgt;
            scenario.mapping     = struct('F', diag([sl, sr]), 'f', [ol; or_]);
            scenario.upr_type    = upr_type;
            scenario.upr_params  = struct();
            scenario.dataset_source          = dataset_file;
            scenario.paired_scenario_ids     = {L{2}, R{2}};
            scenario.paired_scenario_names   = {L{1}, R{1}};
            scenario.consistency_relations   = {rel_l, rel_r};
            scenario.relation_types          = {upr_l, upr_r};
            scenario.relation_operators      = {'leq', 'leq'};

            scenarios{end+1} = scenario; %#ok<AGROW>
            next_id = next_id + 1;
        end
    end

    % ── Pass 2: Coupled 2-D entries (CPS 37–45) ──────────────────────────────
    for di = 1:size(coupled_domains, 1)
        domain_label = coupled_domains{di, 1};
        dataset_file = coupled_domains{di, 2};
        cname        = coupled_domains{di, 3};
        sc0 = coupled_domains{di,  4};  sg0 = coupled_domains{di,  5};
        tc0 = coupled_domains{di,  6};  tg0 = coupled_domains{di,  7};
        sc1 = coupled_domains{di,  8};  sg1 = coupled_domains{di,  9};
        tc1 = coupled_domains{di, 10};  tg1 = coupled_domains{di, 11};
        upr = coupled_domains{di, 12};

        src = conZonotope([sc0; sc1], diag([sg0, sg1]), [], []);
        tgt = conZonotope([tc0; tc1], diag([tg0, tg1]), [], []);

        rel = struct('upr_type', upr, 'mapping', struct('scale', 1.0, 'offset', 0.0));

        scenario = struct();
        scenario.id     = next_id;
        scenario.name   = sprintf('%s - %s [coupled]', domain_label, cname);
        scenario.type   = upr;
        scenario.source = src;
        scenario.target = tgt;
        scenario.mapping     = struct('F', eye(2), 'f', zeros(2,1));
        scenario.upr_type    = upr;
        scenario.upr_params  = struct();
        scenario.dataset_source          = dataset_file;
        scenario.paired_scenario_ids     = {sprintf('%s_dim0', cname), sprintf('%s_dim1', cname)};
        scenario.paired_scenario_names   = {sprintf('%s (dim0)', cname), sprintf('%s (dim1)', cname)};
        scenario.consistency_relations   = {rel, rel};
        scenario.relation_types          = {upr, upr};
        scenario.relation_operators      = {'geq', 'geq'};

        scenarios{end+1} = scenario; %#ok<AGROW>
        next_id = next_id + 1;
    end

    % ── Pass 3: New sequential pairs from entries 9–14 (CPS 46–72) ───────────
    for di = 1:size(new_pair_domains, 1)
        domain_label = new_pair_domains{di, 1};
        dataset_file = new_pair_domains{di, 2};
        entries      = new_pair_domains{di, 3};

        for pair_idx = 1:2:6
            L = entries(pair_idx,   :);
            R = entries(pair_idx+1, :);

            sl = L{7}; ol  = L{8};
            sr = R{7}; or_ = R{8};

            src = conZonotope([L{3}; R{3}], diag([L{4}, R{4}]), [], []);
            tgt = conZonotope([L{5}; R{5}], diag([L{6}, R{6}]), [], []);

            upr_l = L{9}; upr_r = R{9};
            if strcmp(upr_l, upr_r)
                upr_type = upr_l;
            else
                upr_type = sprintf('%s+%s', upr_l, upr_r);
            end

            rel_l = struct('upr_type', upr_l, 'mapping', struct('scale', sl, 'offset', ol));
            rel_r = struct('upr_type', upr_r, 'mapping', struct('scale', sr, 'offset', or_));

            scenario = struct();
            scenario.id     = next_id;
            scenario.name   = sprintf('%s - %s & %s', domain_label, L{1}, R{1});
            scenario.type   = upr_type;
            scenario.source = src;
            scenario.target = tgt;
            scenario.mapping     = struct('F', diag([sl, sr]), 'f', [ol; or_]);
            scenario.upr_type    = upr_type;
            scenario.upr_params  = struct();
            scenario.dataset_source          = dataset_file;
            scenario.paired_scenario_ids     = {L{2}, R{2}};
            scenario.paired_scenario_names   = {L{1}, R{1}};
            scenario.consistency_relations   = {rel_l, rel_r};
            scenario.relation_types          = {upr_l, upr_r};
            scenario.relation_operators      = {'leq', 'leq'};

            scenarios{end+1} = scenario; %#ok<AGROW>
            next_id = next_id + 1;
        end
    end
end

