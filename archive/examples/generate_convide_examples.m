% REGENERATE CONVIDE SCENARIOS WITH I(θ)
%
% This script regenerates CONVIDE scenarios with global inconsistency I(θ)
% instead of pairwise Jaccard indices. It runs the same engineering scenarios
% but computes I(θ) = 1 - P(consistent) across all models.
%
% Based on CONVIDE Real-World Patterns with NEW global inconsistency metric
%
% Output: data/convide_with_I_theta/
% MC Samples: 300 (configurable)
%
% Version: 3.0 (I(θ) format)
% Date: January 2026

%% Setup
clear; close all; clc;

% Configuration: Enable/disable scenario dimensions
run_2d_scenarios = true;
run_3d_scenarios = true;
run_4d_scenarios = true;

seed = 2025;
rng(seed);
n_repeats = 5;
mc_samples = 200;  % 500 for comprehensive data, can adjust

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
fprintf('CONVIDE SCENARIO REGENERATION WITH I(θ)\n');
fprintf('========================================\n');
fprintf('MC Samples: %d\n', mc_samples);
fprintf('Repeats: %d per condition\n', n_repeats);
fprintf('Output: data/convide_with_I_theta/\n\n');

%% Output directory
output_dir = fullfile(fileparts(pwd), 'data', 'convide_with_I_theta');
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
%% 2D CONVIDE SCENARIOS
%% ======================
if run_2d_scenarios
fprintf('\n========================================\n');
fprintf('2D CONVIDE SCENARIOS (4 scenarios)\n');
fprintf('========================================\n');

scenarios_2d = {
    struct('id', 1, 'name', 'CAD Export Drift', 'type', 'Type A', ...
        'source', conPolyZono([100; 50], [2.5 0; 0 2.5], eye(2), [], [], []), ...
        'target', conPolyZono([102.5; 51.5], [0.6 0; 0 0.6], eye(2), [], [], []));
    
    struct('id', 2, 'name', 'MBSE Version Mismatch', 'type', 'Type B', ...
        'source', conPolyZono([50; 25], [1.5 0; 0 1.5], eye(2), [], [], []), ...
        'target', conPolyZono([51; 26], [0.8 0; 0 0.8], eye(2), [], [], []));
    
    struct('id', 3, 'name', 'Documentation Sync', 'type', 'Type B', ...
        'source', conPolyZono([75; 40], [2.0 0; 0 2.0], eye(2), [], [], []), ...
        'target', conPolyZono([76; 41], [0.7 0; 0 0.7], eye(2), [], [], []));
    
    struct('id', 4, 'name', 'Control Design Conflict', 'type', 'Type C', ...
        'source', conPolyZono([60; 30], [1.8 0; 0 1.8], eye(2), [], [], []), ...
        'target', conPolyZono([61.5; 31], [0.5 0; 0 0.5], eye(2), [], [], []));
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
    total_experiments = 0;
    for i = 1:length(interventions)
        int = interventions{i};
        total_experiments = total_experiments + length(int.values) * n_repeats;
    end
    
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
                    fprintf('    Progress: %d/%d (%.1f%%)\n', ...
                        exp_count, total_experiments, 100*exp_count/total_experiments);
                end
                
                % Create params struct - FIXED
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
%% 3D CONVIDE SCENARIOS
%% ======================
if run_3d_scenarios
fprintf('\n========================================\n');
fprintf('3D CONVIDE SCENARIOS (4 scenarios)\n');
fprintf('========================================\n');

scenarios_3d = {
    struct('id', 5, 'name', 'Sensor Calibration Drift', 'type', 'Type A', ...
        'source', conPolyZono([100; 50; 25], diag([3.0, 3.0, 1.5]), eye(3), [], [], []), ...
        'target', conPolyZono([102; 51; 26], diag([0.8, 0.8, 0.4]), eye(3), [], [], []));
    
    struct('id', 6, 'name', 'Requirements Ambiguity', 'type', 'Type D', ...
        'source', conPolyZono([80; 40; 20], diag([2.5, 2.5, 1.2]), eye(3), [], [], []), ...
        'target', conPolyZono([81; 41; 21], diag([0.9, 0.9, 0.45]), eye(3), [], [], []));
    
    struct('id', 7, 'name', 'Test Configuration Mismatch', 'type', 'Type B', ...
        'source', conPolyZono([70; 35; 18], diag([2.2, 2.2, 1.1]), eye(3), [], [], []), ...
        'target', conPolyZono([71.5; 36; 18.5], diag([0.7, 0.7, 0.35]), eye(3), [], [], []));
    
    struct('id', 8, 'name', 'Simulation Numerical Error', 'type', 'Type A', ...
        'source', conPolyZono([90; 45; 22], diag([2.8, 2.8, 1.4]), eye(3), [], [], []), ...
        'target', conPolyZono([91; 46; 23], diag([0.75, 0.75, 0.38]), eye(3), [], [], []));
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
    total_experiments = 0;
    for i = 1:length(interventions)
        int = interventions{i};
        total_experiments = total_experiments + length(int.values) * n_repeats;
    end
    
    exp_count = 0;
    for i = 1:length(interventions)
        int = interventions{i};
        fprintf('  Running %s intervention...\n', int.type);
        
        for val_idx = 1:length(int.values)
            param_val = int.values(val_idx);
            
            for rep = 1:n_repeats
                exp_count = exp_count + 1;
                if mod(exp_count, 10) == 0
                    fprintf('    Progress: %d/%d (%.1f%%)\n', ...
                        exp_count, total_experiments, 100*exp_count/total_experiments);
                end
                
                % Create params struct - FIXED
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
%% 4D CONVIDE SCENARIOS
%% ======================
if run_4d_scenarios
fprintf('\n========================================\n');
fprintf('4D CONVIDE SCENARIOS (4 scenarios)\n');
fprintf('========================================\n');

scenarios_4d = {
    struct('id', 9, 'name', 'Multi-Physics Coupling Error', 'type', 'Type C', ...
        'source', conPolyZono([100; 50; 25; 12], diag([3.5, 3.5, 1.8, 0.9]), eye(4), [], [], []), ...
        'target', conPolyZono([102; 51; 26; 13], diag([0.85, 0.85, 0.43, 0.22]), eye(4), [], [], []));
    
    struct('id', 10, 'name', 'Interface Specification Gap', 'type', 'Type D', ...
        'source', conPolyZono([85; 42; 21; 10], diag([3.0, 3.0, 1.5, 0.75]), eye(4), [], [], []), ...
        'target', conPolyZono([86; 43; 22; 11], diag([0.95, 0.95, 0.48, 0.24]), eye(4), [], [], []));
    
    struct('id', 11, 'name', 'Parameter Estimation Bias', 'type', 'Type A', ...
        'source', conPolyZono([95; 48; 24; 11], diag([3.2, 3.2, 1.6, 0.8]), eye(4), [], [], []), ...
        'target', conPolyZono([96; 49; 25; 12], diag([0.8, 0.8, 0.4, 0.2]), eye(4), [], [], []));
    
    struct('id', 12, 'name', 'Traceability Link Inconsistency', 'type', 'Type B', ...
        'source', conPolyZono([75; 38; 19; 9], diag([2.8, 2.8, 1.4, 0.7]), eye(4), [], [], []), ...
        'target', conPolyZono([76; 39; 20; 10], diag([0.9, 0.9, 0.45, 0.23]), eye(4), [], [], []));
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
    total_experiments = 0;
    for i = 1:length(interventions)
        int = interventions{i};
        total_experiments = total_experiments + length(int.values) * n_repeats;
    end
    
    exp_count = 0;
    for i = 1:length(interventions)
        int = interventions{i};
        fprintf('  Running %s intervention...\n', int.type);
        
        for val_idx = 1:length(int.values)
            param_val = int.values(val_idx);
            
            for rep = 1:n_repeats
                exp_count = exp_count + 1;
                if mod(exp_count, 10) == 0
                    fprintf('    Progress: %d/%d (%.1f%%)\n', ...
                        exp_count, total_experiments, 100*exp_count/total_experiments);
                end
                
                % Create params struct - FIXED
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
fprintf('REGENERATION COMPLETE\n');
fprintf('========================================\n');
fprintf('Total scenarios: 12 (4×2D + 4×3D + 4×4D)\n');
fprintf('MC samples per experiment: %d\n', mc_samples);
fprintf('Output directory: %s\n', output_dir);
fprintf('\nAll data now includes I(θ) = 1 - P(consistent)\n');
fprintf('Ready for sensitivity analysis!\n\n');

fprintf('Next steps:\n');
fprintf('1. Run Python sensitivity analysis:\n');
fprintf('   cd src\n');
fprintf('   python sensitivity_analysis.py --data_dir ../data/convide_with_I_theta --output_dir ../figures/convide_sensitivity\n\n');
fprintf('2. Or run generate_comprehensive_sensitivity_data.m for parametric sweeps\n\n');
