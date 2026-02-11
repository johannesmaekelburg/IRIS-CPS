%% Test Two-Step Workflow Engine
% Quick test to verify the two-step workflow works correctly

clear; close all; clc;

fprintf('========================================\n');
fprintf('TESTING TWO-STEP WORKFLOW ENGINE\n');
fprintf('========================================\n\n');

%% Setup paths
causality_root = fileparts(fileparts(mfilename('fullpath')));
src_path = fullfile(causality_root, 'src');
addpath(src_path);

cps_framework_path = fullfile(fileparts(causality_root), 'CPS-Uncertainty-Propagation-Framework');
addpath(genpath(fullfile(cps_framework_path, 'src')));
addpath(fullfile(cps_framework_path, 'addons', 'consistency_scoring'));
addpath(fullfile(cps_framework_path, 'addons', 'consistency_scoring', 'methods'));

fprintf('Paths configured ✓\n\n');

%% Create test directories
test_dir = fullfile(tempdir, 'test_twostep_engine');
zonotope_dir = fullfile(test_dir, 'zonotopes');
results_dir = fullfile(test_dir, 'measurements');

if exist(test_dir, 'dir')
    rmdir(test_dir, 's');
end
mkdir(zonotope_dir);
mkdir(results_dir);

fprintf('Test directories created:\n');
fprintf('  Zonotopes:    %s\n', zonotope_dir);
fprintf('  Measurements: %s\n\n', results_dir);

%% Define simple test scenario
E = eye(2); A = []; b = []; EC = [];

scenario_def = struct();
scenario_def.id = 1;
scenario_def.name = 'Two-Step Test Scenario';
scenario_def.type = 'Test';
scenario_def.source = conPolyZono([100; 50], [3 0; 0 2], E, A, b, EC);
scenario_def.target = conPolyZono([102; 51], [3.5 0; 0 2.5], E, A, b, EC);

fprintf('Test scenario created:\n');
fprintf('  Source: center=[100, 50], size=3×2\n');
fprintf('  Target: center=[102, 51], size=3.5×2.5\n\n');

%% Define simple interventions
interventions = {...
    struct('type', 'widen', 'param', 'scale_factor', 'values', [1.0, 2.0, 5.0])};

fprintf('Interventions: widen with 3 values × 2 repeats = 6 experiments\n\n');

%% STEP 1: Generate and save zonotopes
fprintf('========================================\n');
fprintf('STEP 1: Generate and Save Zonotopes\n');
fprintf('========================================\n');

gen_options = struct();
gen_options.n_repeats = 2;
gen_options.verbose = true;

tic;
zonotope_data = causal_experiment_engine_twostep.generate_and_save_zonotopes(...
    scenario_def, interventions, zonotope_dir, gen_options);
step1_time = toc;

fprintf('Step 1 completed in %.2f seconds\n', step1_time);
fprintf('Generated %d zonotope files\n\n', length(zonotope_data));

%% Verify zonotope files exist
zono_files = dir(fullfile(zonotope_dir, 'zonotopes_*.mat'));
index_files = dir(fullfile(zonotope_dir, 'index_*.mat'));

fprintf('Verification:\n');
fprintf('  Zonotope files: %d ✓\n', length(zono_files));
fprintf('  Index files: %d ✓\n\n', length(index_files));

assert(length(zono_files) == 6, 'Should have 6 zonotope files');
assert(length(index_files) == 1, 'Should have 1 index file');

%% Load and inspect one zonotope file
sample_file = fullfile(zonotope_dir, zono_files(1).name);
sample_data = load(sample_file);

fprintf('Sample zonotope file contents:\n');
fprintf('  exp_id: %d\n', sample_data.exp_id);
fprintf('  intervention_type: %s\n', sample_data.intervention_type);
fprintf('  param_value: %.1f\n', sample_data.param_value);
fprintf('  Has Z_source_pre: %d\n', isfield(sample_data, 'Z_source_pre'));
fprintf('  Has Z_target: %d\n', isfield(sample_data, 'Z_target'));
fprintf('  Has Z_propagated: %d\n\n', isfield(sample_data, 'Z_propagated'));

%% STEP 2A: Compute consistency with Jaccard method
fprintf('========================================\n');
fprintf('STEP 2A: Compute Consistency (Jaccard)\n');
fprintf('========================================\n');

consistency_opts = struct();
consistency_opts.method = 'jaccard';
consistency_opts.mc_samples = 200;  % Reduced for fast testing (I(θ) still uses MC)
consistency_opts.verbose = true;

results_jaccard_dir = fullfile(results_dir, 'jaccard');
tic;
causal_experiment_engine_twostep.compute_and_save_consistency(...
    zonotope_dir, results_jaccard_dir, consistency_opts);
step2a_time = toc;

fprintf('Step 2A completed in %.2f seconds\n\n', step2a_time);

%% STEP 2B: Compute consistency with MC method (without regenerating!)
fprintf('========================================\n');
fprintf('STEP 2B: Compute Consistency (MC)\n');
fprintf('========================================\n');
fprintf('Re-measuring with different method WITHOUT regenerating zonotopes!\n\n');

consistency_opts.method = 'mc_probability';
consistency_opts.mc_samples = 200;  % Small for fast test

results_mc_dir = fullfile(results_dir, 'mc');
tic;
causal_experiment_engine_twostep.compute_and_save_consistency(...
    zonotope_dir, results_mc_dir, consistency_opts);
step2b_time = toc;

fprintf('Step 2B completed in %.2f seconds\n\n', step2b_time);

%% Verify result files
jaccard_results = dir(fullfile(results_jaccard_dir, 'results_*.json'));
mc_results = dir(fullfile(results_mc_dir, 'results_*.json'));

fprintf('Verification:\n');
fprintf('  Jaccard results: %d ✓\n', length(jaccard_results));
fprintf('  MC results: %d ✓\n\n', length(mc_results));

assert(length(jaccard_results) == 1, 'Should have 1 Jaccard result file');
assert(length(mc_results) == 1, 'Should have 1 MC result file');

%% Load and compare results
fprintf('========================================\n');
fprintf('COMPARING RESULTS\n');
fprintf('========================================\n');

% Load Jaccard results
jaccard_file = fullfile(results_jaccard_dir, jaccard_results(1).name);
fid = fopen(jaccard_file, 'r');
jaccard_json = jsondecode(fread(fid, '*char')');
fclose(fid);

% Load MC results
mc_file = fullfile(results_mc_dir, mc_results(1).name);
fid = fopen(mc_file, 'r');
mc_json = jsondecode(fread(fid, '*char')');
fclose(fid);

fprintf('Jaccard method:\n');
fprintf('  Total experiments: %d\n', jaccard_json.total_experiments);
fprintf('  Method: %s\n', jaccard_json.consistency_method);
fprintf('  Sample experiment 1:\n');
exp1_j = jaccard_json.experiments(1);
fprintf('    Pre-state Jaccard C: %.4f\n', exp1_j.pre_state.inconsistency.jaccard_C);
fprintf('    Post-state Jaccard C: %.4f\n', exp1_j.post_state.inconsistency.jaccard_C);

fprintf('\nMC method:\n');
fprintf('  Total experiments: %d\n', mc_json.total_experiments);
fprintf('  Method: %s\n', mc_json.consistency_method);
fprintf('  MC samples: %d\n', mc_json.mc_samples);
fprintf('  Sample experiment 1:\n');
exp1_m = mc_json.experiments(1);
fprintf('    Pre-state MC prob: %.4f\n', exp1_m.pre_state.inconsistency.mc_probability);
fprintf('    Post-state MC prob: %.4f\n\n', exp1_m.post_state.inconsistency.mc_probability);

%% Performance summary
fprintf('========================================\n');
fprintf('PERFORMANCE SUMMARY\n');
fprintf('========================================\n');
fprintf('Step 1 (Generate zonotopes): %.2f seconds\n', step1_time);
fprintf('Step 2A (Jaccard): %.2f seconds\n', step2a_time);
fprintf('Step 2B (MC): %.2f seconds\n', step2b_time);
fprintf('Total: %.2f seconds\n\n', step1_time + step2a_time + step2b_time);
fprintf('Key benefit: Step 2 can be repeated with different methods\n');
fprintf('without re-running Step 1 (saved %.2f seconds on Step 2B!)\n\n', step1_time);

%% Cleanup
rmdir(test_dir, 's');
fprintf('Test directory cleaned up ✓\n\n');

%% Summary
fprintf('========================================\n');
fprintf('ALL TESTS PASSED ✓\n');
fprintf('========================================\n');
fprintf('The two-step workflow is working correctly:\n');
fprintf('  ✓ Step 1: Zonotopes generated and saved\n');
fprintf('  ✓ Step 2: Consistency computed from saved zonotopes\n');
fprintf('  ✓ Re-measurement: Different methods without regenerating\n');
fprintf('  ✓ Results: JSON files created with correct structure\n\n');
fprintf('You can now use this workflow for your experiments!\n');
fprintf('See: examples/generate_convide_twostep.m\n\n');
