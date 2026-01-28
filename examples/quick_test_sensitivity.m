% QUICK TEST: Global Inconsistency I(θ) and Sensitivity Analysis
%
% Fast version with reduced MC samples for testing the new approach
%
% Expected runtime: ~30 seconds

%% Setup
clear; close all; clc;

% Add paths
src_path = fullfile(fileparts(pwd), 'src');
addpath(src_path);

cps_framework_path = fullfile(fileparts(fileparts(pwd)), 'CPS-Uncertainty-Propagation-Framework');
addpath(genpath(fullfile(cps_framework_path, 'src')));

consistency_addon_path = fullfile(cps_framework_path, 'addons', 'consistency_scoring');
if exist(fullfile(consistency_addon_path, 'init_consistency_scoring.m'), 'file')
    addpath(consistency_addon_path);
    addpath(fullfile(consistency_addon_path, 'methods'));
end

fprintf('=== Quick Test: Global Inconsistency I(θ) ===\n\n');

%% Create Simple 2D Scenario
fprintf('Creating 2D scenario...\n');
dim = 2;
uncertainty_level = 5.0;

scenario = causal_experiment_engine.create_baseline_scenario(dim, uncertainty_level);
fprintf('  Source center: [%.1f, %.1f]\n', scenario.source.c(1), scenario.source.c(2));
fprintf('  Target center: [%.1f, %.1f]\n\n', scenario.target.c(1), scenario.target.c(2));

%% Run Quick Intervention Sweep (REDUCED SAMPLES)
fprintf('Running intervention sweep with REDUCED samples...\n');
intervention_type = 'widen';
param_name = 'scale_factor';
param_values = [0.8, 1.0, 1.5];  % Only 3 values instead of 6

% CRITICAL: Reduce MC samples from 2000 to 200
options = struct('mc_samples', 200, 'consistency_method', 'mc_probability');

fprintf('  Parameters: %s\n', mat2str(param_values));
fprintf('  MC samples: %d (reduced from 2000 for speed)\n\n', options.mc_samples);

tic;
results = causal_experiment_engine.run_intervention_sweep(...
    scenario, intervention_type, param_name, param_values, options);
elapsed = toc;

fprintf('  Completed %d experiments in %.1f seconds\n\n', length(results), elapsed);

%% Extract I(θ) Values
fprintf('=== Global Inconsistency Results ===\n');
fprintf('%-12s | %-10s | %-10s | %-15s\n', 'scale', 'I(θ)', 'SE', '95% CI');
fprintf('------------------------------------------------------------\n');

for i = 1:length(results)
    exp = results{i};
    
    % Extract from post-state
    if isfield(exp, 'post_state') && isfield(exp.post_state, 'inconsistency')
        inc = exp.post_state.inconsistency;
        
        if isfield(inc, 'I_theta')
            I_theta = inc.I_theta;
            I_se = inc.I_theta_se;
            I_ci_lower = inc.I_theta_ci95_lower;
            I_ci_upper = inc.I_theta_ci95_upper;
            
            fprintf('%-12.2f | %-10.4f | %-10.4f | [%.4f, %.4f]\n', ...
                param_values(i), I_theta, I_se, I_ci_lower, I_ci_upper);
        end
    end
end

%% Export to JSON for Python Analysis
fprintf('\n=== Exporting to JSON ===\n');
data_dir = fullfile(fileparts(pwd), 'data', 'test_sensitivity');
if ~exist(data_dir, 'dir')
    mkdir(data_dir);
end

output_file = fullfile(data_dir, 'quick_test_results.json');
causal_experiment_engine.export_to_json(results, output_file);
fprintf('Saved: %s\n', output_file);

%% Summary
fprintf('\n=== Summary ===\n');
fprintf('✓ MATLAB: Computed I(θ) for %d parameter values\n', length(results));
fprintf('✓ Data exported for Python sensitivity analysis\n');
fprintf('\n');
fprintf('Next steps:\n');
fprintf('1. Review I(θ) values above (should increase with scale_factor)\n');
fprintf('2. Run Python sensitivity analysis:\n');
fprintf('   cd Causality_Uncertainty_Inconsistency\n');
fprintf('   python src/sensitivity_analysis.py --data_dir ./data/test_sensitivity --output_dir ./figures/test_results --param scale_factor\n');
fprintf('\n');
fprintf('Expected runtime for full analysis (6 params, 2000 samples): ~5-10 minutes\n');
fprintf('This quick test: %.1f seconds\n', elapsed);
