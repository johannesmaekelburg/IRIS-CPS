% GENERATE COMPREHENSIVE DATASET FOR SENSITIVITY ANALYSIS
%
% This script generates a comprehensive dataset with global inconsistency I(θ)
% measurements across multiple intervention types and parameter ranges.
%
% Output: JSON files ready for Python sensitivity analysis
% Expected runtime: ~15-30 minutes depending on MC samples

%% Setup
clear; close all; clc;

src_path = fullfile(fileparts(pwd), 'src');
addpath(src_path);

cps_framework_path = fullfile(fileparts(fileparts(pwd)), 'CPS-Uncertainty-Propagation-Framework');
addpath(genpath(fullfile(cps_framework_path, 'src')));

consistency_addon_path = fullfile(cps_framework_path, 'addons', 'consistency_scoring');
if exist(fullfile(consistency_addon_path, 'init_consistency_scoring.m'), 'file')
    addpath(consistency_addon_path);
    addpath(fullfile(consistency_addon_path, 'methods'));
end

fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║   COMPREHENSIVE DATASET GENERATION FOR SENSITIVITY ANALYSIS    ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

%% Configuration
output_dir = fullfile(fileparts(pwd), 'data', 'sensitivity_ready');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Analysis parameters
dim = 2;
uncertainty_level = 5.0;
mc_samples = 500;  % Balance between speed and accuracy

% Intervention configurations
interventions = {
    struct('type', 'widen', 'param', 'scale_factor', ...
           'values', [0.5:0.1:1.0, 1.2:0.2:3.0], ...  % Dense near baseline, sparse far out
           'description', 'Widen uncertainty (tolerance relaxation)');
    
    struct('type', 'shrink', 'param', 'scale_factor', ...
           'values', [0.3:0.1:1.0], ...
           'description', 'Shrink uncertainty (tighter tolerances)');
    
    struct('type', 'shift', 'param', 'shift_distance', ...
           'values', [0:0.5:5.0], ...
           'description', 'Shift uncertainty center');
};

fprintf('Configuration:\n');
fprintf('  Dimension:      %d\n', dim);
fprintf('  Base uncertainty: ±%.1f\n', uncertainty_level);
fprintf('  MC samples:     %d\n', mc_samples);
fprintf('  Output:         %s\n\n', output_dir);

fprintf('Interventions to run:\n');
for i = 1:length(interventions)
    fprintf('  %d. %-10s: %s (%d parameter values)\n', ...
        i, interventions{i}.type, interventions{i}.description, ...
        length(interventions{i}.values));
end
fprintf('\n');

%% Create baseline scenario
fprintf('═══════════════════════════════════════════════════════════\n');
fprintf('Creating baseline scenario...\n');
fprintf('═══════════════════════════════════════════════════════════\n');

scenario = causal_experiment_engine.create_baseline_scenario(dim, uncertainty_level);

fprintf('  Source center: [%.2f, %.2f]\n', scenario.source.c(1), scenario.source.c(2));
fprintf('  Target center: [%.2f, %.2f]\n', scenario.target.c(1), scenario.target.c(2));
fprintf('  Uncertainty generators: %d\n', size(scenario.source.G, 2));
fprintf('\n');

% Set options for all experiments
options = struct(...
    'mc_samples', mc_samples, ...
    'consistency_method', 'mc_probability');  % Fast mode - MC only

%% Run experiments for each intervention type
total_start = tic;
all_results = struct();

for int_idx = 1:length(interventions)
    intervention = interventions{int_idx};
    
    fprintf('═══════════════════════════════════════════════════════════\n');
    fprintf('[%d/%d] Running: %s\n', int_idx, length(interventions), ...
        intervention.description);
    fprintf('═══════════════════════════════════════════════════════════\n');
    
    n_params = length(intervention.values);
    fprintf('  Parameter: %s\n', intervention.param);
    fprintf('  Values: %d points from %.2f to %.2f\n', ...
        n_params, min(intervention.values), max(intervention.values));
    fprintf('  Progress: ');
    
    % Run parameter sweep
    sweep_start = tic;
    
    try
        results = causal_experiment_engine.run_intervention_sweep(...
            scenario, ...
            intervention.type, ...
            intervention.param, ...
            intervention.values, ...
            options);
        
        sweep_time = toc(sweep_start);
        
        fprintf('✓ DONE\n');
        fprintf('  Completed: %d experiments in %.1f seconds (%.1f sec/exp)\n', ...
            length(results), sweep_time, sweep_time/length(results));
        
        % Quick validation: Check I_theta values
        I_theta_values = zeros(length(results), 1);
        for i = 1:length(results)
            if isfield(results{i}, 'post_state') && ...
               isfield(results{i}.post_state, 'inconsistency') && ...
               isfield(results{i}.post_state.inconsistency, 'I_theta')
                I_theta_values(i) = results{i}.post_state.inconsistency.I_theta;
            else
                I_theta_values(i) = NaN;
            end
        end
        
        valid_count = sum(~isnan(I_theta_values));
        fprintf('  I(θ) range: [%.4f, %.4f] (%d valid values)\n', ...
            min(I_theta_values), max(I_theta_values), valid_count);
        
        % Export to JSON
        output_filename = sprintf('results_%s_%dmc.json', ...
            intervention.type, mc_samples);
        output_path = fullfile(output_dir, output_filename);
        
        causal_experiment_engine.export_to_json(results, output_path);
        fprintf('  ✓ Saved: %s\n', output_filename);
        
        % Store for summary
        all_results.(intervention.type) = struct(...
            'n_experiments', length(results), ...
            'time_seconds', sweep_time, ...
            'I_theta_min', min(I_theta_values), ...
            'I_theta_max', max(I_theta_values), ...
            'I_theta_mean', mean(I_theta_values), ...
            'output_file', output_filename);
        
    catch ME
        fprintf('✗ FAILED\n');
        fprintf('  Error: %s\n', ME.message);
        all_results.(intervention.type) = struct('status', 'failed', 'error', ME.message);
    end
    
    fprintf('\n');
end

total_time = toc(total_start);

%% Summary Report
fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║                      GENERATION COMPLETE                       ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

fprintf('Total runtime: %.1f minutes (%.1f seconds)\n\n', total_time/60, total_time);

fprintf('Generated datasets:\n');
fprintf('%-12s | %-8s | %-10s | %-20s | %-15s\n', ...
    'Intervention', 'N_exp', 'Time(s)', 'I(θ) Range', 'File');
fprintf('─────────────────────────────────────────────────────────────────────────────────\n');

fields = fieldnames(all_results);
for i = 1:length(fields)
    result = all_results.(fields{i});
    if isfield(result, 'n_experiments')
        fprintf('%-12s | %-8d | %-10.1f | [%.3f, %.3f] | %-15s\n', ...
            fields{i}, ...
            result.n_experiments, ...
            result.time_seconds, ...
            result.I_theta_min, ...
            result.I_theta_max, ...
            result.output_file);
    else
        fprintf('%-12s | %-8s | %-10s | %-20s | %-15s\n', ...
            fields{i}, 'FAILED', '-', '-', '-');
    end
end
fprintf('\n');

%% Python Command Guide
fprintf('═══════════════════════════════════════════════════════════\n');
fprintf('NEXT STEP: Run Python Sensitivity Analysis\n');
fprintf('═══════════════════════════════════════════════════════════\n\n');

fprintf('For each intervention type, run:\n\n');

for i = 1:length(fields)
    result = all_results.(fields{i});
    if isfield(result, 'output_file')
        intervention_type = fields{i};
        
        fprintf('# %s analysis:\n', intervention_type);
        fprintf('python src/sensitivity_analysis.py \\\n');
        fprintf('  --data_dir ./data/sensitivity_ready \\\n');
        fprintf('  --output_dir ./figures/sensitivity_%s \\\n', intervention_type);
        fprintf('  --param %s \\\n', intervention_type);
        fprintf('  --threshold 0.5\n\n');
    end
end

fprintf('This will generate:\n');
fprintf('  ✓ Causal effects τ(a,b) plots\n');
fprintf('  ✓ Local sensitivity ∂I/∂θ curves\n');
fprintf('  ✓ Robustness margins s*(τ)\n');
fprintf('  ✓ Sobol indices (if multiple parameters)\n');
fprintf('  ✓ Surrogate model (GP/RF)\n\n');

fprintf('═══════════════════════════════════════════════════════════\n');
fprintf('✓ Dataset ready for sensitivity analysis!\n');
fprintf('═══════════════════════════════════════════════════════════\n');

%% Save summary
summary_file = fullfile(output_dir, 'generation_summary.json');
summary_data = struct(...
    'timestamp', char(datetime('now')), ...
    'configuration', struct(...
        'dim', dim, ...
        'uncertainty_level', uncertainty_level, ...
        'mc_samples', mc_samples), ...
    'total_time_seconds', total_time, ...
    'results', all_results);

fid = fopen(summary_file, 'w');
fprintf(fid, '%s', jsonencode(summary_data, 'PrettyPrint', true));
fclose(fid);
fprintf('\nSummary saved: %s\n', summary_file);
