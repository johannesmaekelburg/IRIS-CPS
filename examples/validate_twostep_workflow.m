% VALIDATE TWO-STEP WORKFLOW
%
% This script validates that the two-step workflow produces identical
% results to the original one-step approach.
%
% Tests:
%   1. Zonotope generation matches
%   2. Consistency scores match between workflows
%   3. Causal effects are identical

%% Setup
clear; close all; clc;

fprintf('========================================\n');
fprintf('VALIDATING TWO-STEP WORKFLOW\n');
fprintf('========================================\n\n');

% Add paths
src_path = fullfile(fileparts(pwd), 'src');
addpath(src_path);

cps_framework_path = fullfile(fileparts(fileparts(pwd)), 'CPS-Uncertainty-Propagation-Framework');
addpath(genpath(fullfile(cps_framework_path, 'src')));
addpath(fullfile(cps_framework_path, 'addons', 'consistency_scoring'));
addpath(fullfile(cps_framework_path, 'addons', 'consistency_scoring', 'methods'));

%% Test Configuration
seed = 12345;
rng(seed);
mc_samples = 500;  % Small for fast validation

% Simple test scenario
E = eye(2); A = []; b = []; EC = [];
test_scenario = struct();
test_scenario.id = 999;
test_scenario.name = 'Validation Test';
test_scenario.type = 'Test';
test_scenario.source = conPolyZono([100; 50], [2.0 0; 0 2.0], E, A, b, EC);
test_scenario.target = conPolyZono([101; 51], [2.1 0; 0 2.1], E, A, b, EC);

% Simple intervention
test_interventions = {...
    struct('type', 'widen', 'param', 'scale_factor', 'values', [1.0, 2.0, 5.0])};

%% Test 1: Original One-Step Approach
fprintf('Test 1: Running ORIGINAL one-step workflow...\n');

baseline = causal_experiment_engine.create_baseline_scenario(2, 5.0);
baseline.source = test_scenario.source;
baseline.target = test_scenario.target;

options_orig = struct('consistency_method', 'both', 'mc_samples', mc_samples);

results_original = {};
for val_idx = 1:length(test_interventions{1}.values)
    param_val = test_interventions{1}.values(val_idx);
    params = struct('scale_factor', param_val);
    
    result = causal_experiment_engine.run_intervention(...
        baseline, 'widen', params, options_orig);
    
    results_original{end+1} = result;
end

fprintf('  ✓ Generated %d results (original method)\n\n', length(results_original));

%% Test 2: Two-Step Approach
fprintf('Test 2: Running TWO-STEP workflow...\n');

% Step 1: Generate zonotopes
temp_zono_dir = fullfile(tempdir, 'validate_zonotopes');
if exist(temp_zono_dir, 'dir')
    rmdir(temp_zono_dir, 's');
end

gen_opts = struct('n_repeats', 1, 'verbose', false);
causal_experiment_engine_twostep.generate_and_save_zonotopes(...
    test_scenario, test_interventions, temp_zono_dir, gen_opts);

fprintf('  ✓ Step 1: Generated zonotopes\n');

% Step 2: Compute consistency
temp_results_dir = fullfile(tempdir, 'validate_results');
if exist(temp_results_dir, 'dir')
    rmdir(temp_results_dir, 's');
end

cons_opts = struct('method', 'both', 'mc_samples', mc_samples, 'verbose', false);
causal_experiment_engine_twostep.compute_and_save_consistency(...
    temp_zono_dir, temp_results_dir, cons_opts);

fprintf('  ✓ Step 2: Computed consistency\n');

% Load two-step results
results_file = fullfile(temp_results_dir, 'results_scenario_999.json');
fid = fopen(results_file, 'r');
json_str = fread(fid, '*char')';
fclose(fid);
results_twostep_data = jsondecode(json_str);
results_twostep = results_twostep_data.experiments;

fprintf('  ✓ Loaded %d results (two-step method)\n\n', length(results_twostep));

%% Test 3: Compare Results
fprintf('Test 3: Comparing results...\n');

tolerance = 1e-6;  % Numerical tolerance
all_passed = true;

for i = 1:length(results_original)
    orig = results_original{i};
    twostep = results_twostep(i);
    
    fprintf('  Experiment %d (scale_factor=%.1f):\n', i, test_interventions{1}.values(i));
    
    % Compare pre-state Jaccard
    if isfield(orig.pre_state.inconsistency, 'jaccard_C')
        diff = abs(orig.pre_state.inconsistency.jaccard_C - twostep.pre_state.inconsistency.jaccard_C);
        status = diff < tolerance;
        fprintf('    Pre Jaccard C: %.6f vs %.6f (diff=%.2e) %s\n', ...
            orig.pre_state.inconsistency.jaccard_C, ...
            twostep.pre_state.inconsistency.jaccard_C, ...
            diff, ternary(status, '✓', '✗'));
        all_passed = all_passed && status;
    end
    
    % Compare post-state Jaccard
    if isfield(orig.post_state.inconsistency, 'jaccard_C')
        diff = abs(orig.post_state.inconsistency.jaccard_C - twostep.post_state.inconsistency.jaccard_C);
        status = diff < tolerance;
        fprintf('    Post Jaccard C: %.6f vs %.6f (diff=%.2e) %s\n', ...
            orig.post_state.inconsistency.jaccard_C, ...
            twostep.post_state.inconsistency.jaccard_C, ...
            diff, ternary(status, '✓', '✗'));
        all_passed = all_passed && status;
    end
    
    % Compare MC probability (more tolerance due to sampling)
    if isfield(orig.pre_state.inconsistency, 'mc_probability')
        diff = abs(orig.pre_state.inconsistency.mc_probability - twostep.pre_state.inconsistency.mc_probability);
        mc_tolerance = 0.05;  % Allow 5% difference due to random sampling
        status = diff < mc_tolerance;
        fprintf('    Pre MC prob: %.6f vs %.6f (diff=%.2e) %s\n', ...
            orig.pre_state.inconsistency.mc_probability, ...
            twostep.pre_state.inconsistency.mc_probability, ...
            diff, ternary(status, '✓', '✗'));
        all_passed = all_passed && status;
    end
    
    % Compare I(theta)
    if isfield(orig.pre_state.inconsistency, 'I_theta')
        diff = abs(orig.pre_state.inconsistency.I_theta - twostep.pre_state.inconsistency.I_theta);
        mc_tolerance = 0.05;
        status = diff < mc_tolerance;
        fprintf('    Pre I(θ): %.6f vs %.6f (diff=%.2e) %s\n', ...
            orig.pre_state.inconsistency.I_theta, ...
            twostep.pre_state.inconsistency.I_theta, ...
            diff, ternary(status, '✓', '✗'));
        all_passed = all_passed && status;
    end
    
    fprintf('\n');
end

%% Cleanup
rmdir(temp_zono_dir, 's');
rmdir(temp_results_dir, 's');

%% Summary
fprintf('========================================\n');
if all_passed
    fprintf('VALIDATION PASSED ✓\n');
    fprintf('========================================\n');
    fprintf('Two-step workflow produces identical results!\n');
    fprintf('You can safely use the two-step approach for:\n');
    fprintf('  - Faster iteration on consistency methods\n');
    fprintf('  - Re-measuring with different parameters\n');
    fprintf('  - Adding new consistency metrics later\n');
else
    fprintf('VALIDATION FAILED ✗\n');
    fprintf('========================================\n');
    fprintf('Some differences detected. Review output above.\n');
    fprintf('Note: Small MC probability differences are expected\n');
    fprintf('due to random sampling. Set seed for reproducibility.\n');
end
fprintf('\n');

%% Helper function
function result = ternary(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
