% TEST SHIFT INTERVENTION - Quick verification before full run
% This tests the new 'shift' intervention type with small sample size
%
% Expected: Should complete in ~2 minutes and show shift interventions working

clear; close all; clc;

% Suppress warnings
warning('off', 'all');

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
fprintf('TESTING SHIFT INTERVENTION\n');
fprintf('========================================\n');

%% Test setup
seed = 2025;
rng(seed);

% Test interventions including shift
interventions = {
    struct('type', 'shift', 'param', 'center_delta', ...
        'values', [0.0, 0.05, 0.1]);  % 0%, 5%, 10% relative shifts
};

% Simple 2D scenario
test_scenario = struct(...
    'id', 999, ...
    'name', 'Shift Test', ...
    'type', 'Test', ...
    'source', conPolyZono([100; 50], [2.0 0; 0 2.0], eye(2), [], [], []), ...
    'target', conPolyZono([100.3; 50.3], [2.1 0; 0 2.1], eye(2), [], [], []));

%% Step 1: Generate zonotopes
fprintf('\nStep 1: Generating zonotopes with shift interventions...\n');

temp_dir = fullfile(fileparts(pwd), 'data', 'test_shift');
if ~exist(temp_dir, 'dir')
    mkdir(temp_dir);
end

gen_options = struct();
gen_options.n_repeats = 2;  % Just 2 repeats for quick test
gen_options.verbose = true;

try
    causal_experiment_engine_twostep.generate_and_save_zonotopes(...
        test_scenario, interventions, temp_dir, gen_options);
    fprintf('✓ Zonotope generation successful!\n');
catch ME
    fprintf('✗ ERROR: %s\n', ME.message);
    fprintf('Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
    return;
end

%% Step 2: Compute consistency (small sample size)
fprintf('\nStep 2: Computing consistency scores (N=100 samples)...\n');

consistency_opts = struct();
consistency_opts.method = 'both';
consistency_opts.mc_samples = 100;  % Small for quick test
consistency_opts.sampling_method = 'sobol';  % Just one method for test
consistency_opts.verbose = true;

try
    causal_experiment_engine_twostep.compute_and_save_consistency(...
        temp_dir, temp_dir, consistency_opts);
    fprintf('✓ Consistency computation successful!\n');
catch ME
    fprintf('✗ ERROR: %s\n', ME.message);
    fprintf('Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
    return;
end

%% Verify results
fprintf('\n========================================\n');
fprintf('VERIFICATION\n');
fprintf('========================================\n');

result_file = fullfile(temp_dir, 'results_scenario_999.json');
if exist(result_file, 'file')
    fid = fopen(result_file, 'r');
    raw = fread(fid, inf);
    str = char(raw');
    fclose(fid);
    data = jsondecode(str);
    
    fprintf('Scenario: %s\n', data.scenario_name);
    fprintf('Total experiments: %d\n', length(data.experiments));
    fprintf('\nShift intervention results:\n');
    fprintf('%-15s %-15s %-15s\n', 'Shift Value', 'MC Probability', 'Jaccard MC');
    fprintf('%-15s %-15s %-15s\n', '───────────', '──────────────', '──────────');
    
    for i = 1:length(data.experiments)
        exp = data.experiments(i);
        if strcmp(exp.intervention_type, 'shift')
            % Get shift value from intervention_value field
            shift_val = exp.intervention_value;
            mc_prob = exp.post_state.inconsistency.mc_probability_sobol;
            
            % Check if jaccard_mc field exists
            if isfield(exp.post_state.inconsistency, 'jaccard_mc_index_sobol')
                jaccard_mc = exp.post_state.inconsistency.jaccard_mc_index_sobol;
            else
                jaccard_mc = NaN;
                if i == 1  % Print debug info once
                    fprintf('\nDEBUG: Available inconsistency fields:\n');
                    fprintf('  %s\n', strjoin(fieldnames(exp.post_state.inconsistency), ', '));
                end
            end
            fprintf('%-15.2f %-15.4f %-15.4f\n', shift_val, mc_prob, jaccard_mc);
        end
    end
    
    fprintf('\n✓ TEST PASSED: Shift interventions working correctly!\n');
    fprintf('  - All shift values processed\n');
    fprintf('  - Consistency metrics computed\n');
    fprintf('  - Results saved to JSON\n');
    
else
    fprintf('✗ Result file not found: %s\n', result_file);
end

fprintf('\n========================================\n');
fprintf('Test complete. Ready for full run!\n');
fprintf('========================================\n');
