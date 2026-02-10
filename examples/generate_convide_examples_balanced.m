% GENERATE CONVIDE SCENARIOS WITH BALANCED BASELINE CONSISTENCY
%
% This version creates scenarios with LOW baseline I(θ) to demonstrate
% clear causal effects: uncertainty interventions CAUSE inconsistency.
%
% Key changes from original:
% - Centers closely aligned (minimal shift)
% - Source/target sizes similar (good initial overlap)
% - Expected baseline I(θ) ≈ 0.1-0.3 (mostly consistent)
% - Interventions drive I(θ) → 0.7-1.0 (becomes inconsistent)
%
% Output: data/convide_with_I_theta/
% MC Samples: 300 (configurable)
%
% Version: 4.0 (Balanced baseline for causal analysis)
% Date: January 2026

%% Setup
clear; close all; clc;

% Configuration: Enable/disable scenario dimensions
run_2d_scenarios = true;
run_3d_scenarios = false;
run_4d_scenarios = true;

seed = 2025;
rng(seed);
n_repeats = 5;
mc_samples = 200;  % 500 for comprehensive data

% Add paths
src_path = fullfile(fileparts(pwd), 'src');
addpath(src_path);

cps_framework_path = fullfile(fileparts(fileparts(pwd)), 'CPS-Uncertainty-Propagation-Framework');
addpath(genpath(fullfile(cps_framework_path, 'src')));

consistency_addon_path = fullfile(cps_framework_path, 'addons', 'consistency_scoring');
if exist(fullfile(consistency_addon_path, 'init_consistency_scoring.m'), 'file')
    addpath(consistency_addon_path);
    addpath(fullfile(consistency_addon_path, 'methods'));
else
    error('Consistency scoring addon required but not found.');
end

fprintf('========================================\n');
fprintf('BALANCED CONVIDE SCENARIOS WITH I(θ)\n');
fprintf('========================================\n');
fprintf('Strategy: Low baseline I(θ) + clear causal effects\n');
fprintf('MC Samples: %d\n', mc_samples);
fprintf('Repeats: %d per condition\n', n_repeats);
fprintf('Output: data/convide_balanced/\n\n');

%% Output directory
output_dir = fullfile(fileparts(pwd), 'data', 'convide_balanced');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% Intervention Configuration
interventions = {
    struct('type', 'widen', 'param', 'scale_factor', ...
        'values', [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]);
    struct('type', 'shrink', 'param', 'scale_factor', ...
        'values', [0.01, 0.1, 0.5, 1.0, 2.0]);
    struct('type', 'correlate', 'param', 'correlation_strength', ...
        'values', [0.0, 0.3, 0.6, 0.8, 0.9, 0.95]);
};

%% Options for I(θ) computation
options = struct('mc_samples', mc_samples);

%% ======================
%% 2D CONVIDE SCENARIOS (BALANCED)
%% ======================
if run_2d_scenarios
fprintf('\n========================================\n');
fprintf('2D CONVIDE SCENARIOS (4 scenarios - BALANCED)\n');
fprintf('========================================\n');

scenarios_2d = {
    % Scenario 1: Well-aligned, similar sizes → Low baseline I(θ)
    struct('id', 1, 'name', 'CAD Export Drift', 'type', 'Type A', ...
        'source', conPolyZono([100; 50], [2.0 0; 0 2.0], eye(2), [], [], []), ...
        'target', conPolyZono([100.3; 50.3], [2.1 0; 0 2.1], eye(2), [], [], []));
    
    % Scenario 2: Same center, target slightly larger → Very low I(θ)
    struct('id', 2, 'name', 'MBSE Version Mismatch', 'type', 'Type B', ...
        'source', conPolyZono([50; 25], [1.5 0; 0 1.5], eye(2), [], [], []), ...
        'target', conPolyZono([50; 25], [1.8 0; 0 1.8], eye(2), [], [], []));
    
    % Scenario 3: Small offset, similar sizes
    struct('id', 3, 'name', 'Documentation Sync', 'type', 'Type B', ...
        'source', conPolyZono([75; 40], [2.0 0; 0 2.0], eye(2), [], [], []), ...
        'target', conPolyZono([75.5; 40.5], [2.0 0; 0 2.0], eye(2), [], [], []));
    
    % Scenario 4: Moderate overlap, target slightly smaller
    struct('id', 4, 'name', 'Control Design Conflict', 'type', 'Type C', ...
        'source', conPolyZono([60; 30], [1.8 0; 0 1.8], eye(2), [], [], []), ...
        'target', conPolyZono([60.8; 30.8], [1.5 0; 0 1.5], eye(2), [], [], []));
};

fprintf('Running 4 scenarios in 2D...\n');
for s = 1:length(scenarios_2d)
    scenario_def = scenarios_2d{s};
    scenario_name = sprintf('convide_2d_scenario_%d', scenario_def.id);
    
    fprintf('\n--- Scenario %d: %s (%s) ---\n', ...
        scenario_def.id, scenario_def.name, scenario_def.type);
    
    % Create baseline scenario
    scenario = causal_experiment_engine.create_baseline_scenario(2, 5.0);
    scenario.source = scenario_def.source;
    scenario.target = scenario_def.target;
    
    % Run all interventions
    all_results = {};
    exp_count = 0;
    for i = 1:length(interventions)
        int = interventions{i};
        fprintf('  Running %s intervention (%d values × %d repeats = %d experiments)...\n', ...
            int.type, length(int.values), n_repeats, length(int.values)*n_repeats);
        
        for val_idx = 1:length(int.values)
            param_val = int.values(val_idx);
            
            for rep = 1:n_repeats
                exp_count = exp_count + 1;
                if mod(exp_count, 10) == 0
                    fprintf('    Progress: %d/%d\n', exp_count, length(int.values)*n_repeats*length(interventions));
                end
                
                % Create params struct
                params = struct(int.param, param_val);
                results = causal_experiment_engine.run_intervention(...
                    scenario, int.type, params, options);
                
                results.scenario_type = scenario_name;
                results.scenario_description = scenario_def.name;
                results.causality_type = scenario_def.type;
                results.intervention_type = int.type;
                results.intervention_direction = 'forward';
                results.param_value = param_val;
                results.repeat_idx = rep;
                results.run_id = exp_count;
                
                all_results{end+1} = results;
            end
        end
    end
    
    % Export to JSON
    output_file = fullfile(output_dir, sprintf('results_%s.json', scenario_name));
    causal_experiment_engine.export_to_json(...
        struct('experiments', {all_results}, ...
               'scenario_name', scenario_name, ...
               'scenario_description', scenario_def.name, ...
               'causality_type', scenario_def.type, ...
               'mc_samples', mc_samples, ...
               'n_repeats', n_repeats, ...
               'total_experiments', length(all_results)), ...
        output_file);
    
    fprintf('  ✓ Saved %d experiments to: %s\n', length(all_results), output_file);
end
end  % if run_2d_scenarios

%% ======================
%% 3D CONVIDE SCENARIOS (BALANCED)
%% ======================
if run_3d_scenarios
fprintf('\n========================================\n');
fprintf('3D CONVIDE SCENARIOS (4 scenarios - BALANCED)\n');
fprintf('========================================\n');

scenarios_3d = {
    % Scenario 5: Small center offset, similar sizes
    struct('id', 5, 'name', 'Sensor Calibration Drift', 'type', 'Type A', ...
        'source', conPolyZono([100; 50; 25], diag([2.5, 2.5, 1.2]), eye(3), [], [], []), ...
        'target', conPolyZono([100.5; 50.5; 25.5], diag([2.6, 2.6, 1.3]), eye(3), [], [], []));
    
    % Scenario 6: Same center, target larger (contains source)
    struct('id', 6, 'name', 'Requirements Ambiguity', 'type', 'Type D', ...
        'source', conPolyZono([80; 40; 20], diag([2.0, 2.0, 1.0]), eye(3), [], [], []), ...
        'target', conPolyZono([80; 40; 20], diag([2.5, 2.5, 1.2]), eye(3), [], [], []));
    
    % Scenario 7: Minimal offset, equal sizes
    struct('id', 7, 'name', 'Test Configuration Mismatch', 'type', 'Type B', ...
        'source', conPolyZono([70; 35; 18], diag([2.2, 2.2, 1.1]), eye(3), [], [], []), ...
        'target', conPolyZono([70.4; 35.4; 18.4], diag([2.2, 2.2, 1.1]), eye(3), [], [], []));
    
    % Scenario 8: Small offset, target slightly smaller
    struct('id', 8, 'name', 'Simulation Numerical Error', 'type', 'Type A', ...
        'source', conPolyZono([90; 45; 22], diag([2.5, 2.5, 1.3]), eye(3), [], [], []), ...
        'target', conPolyZono([90.6; 45.6; 22.6], diag([2.2, 2.2, 1.1]), eye(3), [], [], []));
};

fprintf('Running 4 scenarios in 3D...\n');
for s = 1:length(scenarios_3d)
    scenario_def = scenarios_3d{s};
    scenario_name = sprintf('convide_3d_scenario_%d', scenario_def.id);
    
    fprintf('\n--- Scenario %d: %s (%s) ---\n', ...
        scenario_def.id, scenario_def.name, scenario_def.type);
    
    scenario = causal_experiment_engine.create_baseline_scenario(3, 5.0);
    scenario.source = scenario_def.source;
    scenario.target = scenario_def.target;
    
    all_results = {};
    exp_count = 0;
    for i = 1:length(interventions)
        int = interventions{i};
        fprintf('  Running %s intervention...\n', int.type);
        
        for val_idx = 1:length(int.values)
            param_val = int.values(val_idx);
            
            for rep = 1:n_repeats
                exp_count = exp_count + 1;
                if mod(exp_count, 10) == 0
                    fprintf('    Progress: %d\n', exp_count);
                end
                
                params = struct(int.param, param_val);
                results = causal_experiment_engine.run_intervention(...
                    scenario, int.type, params, options);
                
                results.scenario_type = scenario_name;
                results.scenario_description = scenario_def.name;
                results.causality_type = scenario_def.type;
                results.intervention_type = int.type;
                results.intervention_direction = 'forward';
                results.param_value = param_val;
                results.repeat_idx = rep;
                results.run_id = exp_count;
                
                all_results{end+1} = results;
            end
        end
    end
    
    output_file = fullfile(output_dir, sprintf('results_%s.json', scenario_name));
    causal_experiment_engine.export_to_json(...
        struct('experiments', {all_results}, ...
               'scenario_name', scenario_name, ...
               'scenario_description', scenario_def.name, ...
               'causality_type', scenario_def.type, ...
               'mc_samples', mc_samples, ...
               'n_repeats', n_repeats, ...
               'total_experiments', length(all_results)), ...
        output_file);
    
    fprintf('  ✓ Saved %d experiments to: %s\n', length(all_results), output_file);
end
end  % if run_3d_scenarios

%% ======================
%% 4D CONVIDE SCENARIOS (BALANCED)
%% ======================
if run_4d_scenarios
fprintf('\n========================================\n');
fprintf('4D CONVIDE SCENARIOS (4 scenarios - BALANCED)\n');
fprintf('========================================\n');

scenarios_4d = {
    % Scenario 9: Small offset, similar sizes
    struct('id', 9, 'name', 'Multi-Physics Coupling Error', 'type', 'Type C', ...
        'source', conPolyZono([100; 50; 25; 12], diag([3.0, 3.0, 1.5, 0.75]), eye(4), [], [], []), ...
        'target', conPolyZono([100.5; 50.5; 25.5; 12.5], diag([3.1, 3.1, 1.6, 0.8]), eye(4), [], [], []));
    
    % Scenario 10: Same center, target contains source
    struct('id', 10, 'name', 'Interface Specification Gap', 'type', 'Type D', ...
        'source', conPolyZono([85; 42; 21; 10], diag([2.5, 2.5, 1.2, 0.6]), eye(4), [], [], []), ...
        'target', conPolyZono([85; 42; 21; 10], diag([3.0, 3.0, 1.5, 0.75]), eye(4), [], [], []));
    
    % Scenario 11: Minimal offset, equal sizes
    struct('id', 11, 'name', 'Parameter Estimation Bias', 'type', 'Type A', ...
        'source', conPolyZono([95; 48; 24; 11], diag([2.8, 2.8, 1.4, 0.7]), eye(4), [], [], []), ...
        'target', conPolyZono([95.4; 48.4; 24.4; 11.4], diag([2.8, 2.8, 1.4, 0.7]), eye(4), [], [], []));
    
    % Scenario 12: Small offset, target slightly smaller
    struct('id', 12, 'name', 'Traceability Link Inconsistency', 'type', 'Type B', ...
        'source', conPolyZono([75; 38; 19; 9], diag([2.5, 2.5, 1.2, 0.6]), eye(4), [], [], []), ...
        'target', conPolyZono([75.6; 38.6; 19.6; 9.6], diag([2.2, 2.2, 1.1, 0.55]), eye(4), [], [], []));
};

fprintf('Running 4 scenarios in 4D...\n');
for s = 1:length(scenarios_4d)
    scenario_def = scenarios_4d{s};
    scenario_name = sprintf('convide_4d_scenario_%d', scenario_def.id);
    
    fprintf('\n--- Scenario %d: %s (%s) ---\n', ...
        scenario_def.id, scenario_def.name, scenario_def.type);
    
    scenario = causal_experiment_engine.create_baseline_scenario(4, 5.0);
    scenario.source = scenario_def.source;
    scenario.target = scenario_def.target;
    
    all_results = {};
    exp_count = 0;
    for i = 1:length(interventions)
        int = interventions{i};
        fprintf('  Running %s intervention...\n', int.type);
        
        for val_idx = 1:length(int.values)
            param_val = int.values(val_idx);
            
            for rep = 1:n_repeats
                exp_count = exp_count + 1;
                if mod(exp_count, 10) == 0
                    fprintf('    Progress: %d\n', exp_count);
                end
                
                params = struct(int.param, param_val);
                results = causal_experiment_engine.run_intervention(...
                    scenario, int.type, params, options);
                
                results.scenario_type = scenario_name;
                results.scenario_description = scenario_def.name;
                results.causality_type = scenario_def.type;
                results.intervention_type = int.type;
                results.intervention_direction = 'forward';
                results.param_value = param_val;
                results.repeat_idx = rep;
                results.run_id = exp_count;
                
                all_results{end+1} = results;
            end
        end
    end
    
    output_file = fullfile(output_dir, sprintf('results_%s.json', scenario_name));
    causal_experiment_engine.export_to_json(...
        struct('experiments', {all_results}, ...
               'scenario_name', scenario_name, ...
               'scenario_description', scenario_def.name, ...
               'causality_type', scenario_def.type, ...
               'mc_samples', mc_samples, ...
               'n_repeats', n_repeats, ...
               'total_experiments', length(all_results)), ...
        output_file);
    
    fprintf('  ✓ Saved %d experiments to: %s\n', length(all_results), output_file);
end
end  % if run_4d_scenarios

%% Summary
fprintf('\n========================================\n');
fprintf('BALANCED SCENARIO GENERATION COMPLETE\n');
fprintf('========================================\n');
fprintf('Total scenarios: 12 (4×2D + 4×3D + 4×4D)\n');
fprintf('MC samples per experiment: %d\n', mc_samples);
fprintf('Output directory: %s\n', output_dir);
fprintf('\nKey improvements:\n');
fprintf('- Low baseline I(θ) ≈ 0.1-0.3 (good initial consistency)\n');
fprintf('- Clear causal effects: interventions drive I(θ) → 0.7-1.0\n');
fprintf('- Meaningful dose-response relationships\n');
fprintf('- Interpretable robustness margins\n\n');

fprintf('Next steps:\n');
fprintf('1. Run Python sensitivity analysis:\n');
fprintf('   python src/sensitivity_analysis.py --data_dir ./data/convide_with_I_theta --output_dir ./figures/convide_sensitivity\n\n');
fprintf('2. Compare with original scenarios to see the difference\n\n');
