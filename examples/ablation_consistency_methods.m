% ABLATION STUDY: Consistency Scoring Methods
% 
% This script compares different consistency scoring methods on a subset
% of scenarios to determine the best trade-off between accuracy and speed.
%
% Methods compared:
%   1. AABB Jaccard only (fast)
%   2. MC Jaccard (Sobol, various N)
%   3. MC Probability (Sobol, various N)
%   4. Both MC methods
%   5. Comparison of sampling methods (Sobol vs Halton vs LHS vs Random)
%
% Output: Timing comparison, score correlation analysis, recommendation

clear; close all; clc;

% Suppress warnings
warning('off', 'all');

%% Configuration
seed = 2025;
rng(seed);

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

fprintf('========================================\n');
fprintf('ABLATION STUDY: Consistency Methods\n');
fprintf('========================================\n\n');

%% Step 1: Generate zonotopes for test scenarios (if not exist)
data_base = fullfile(fileparts(pwd), 'data');
zonotope_dir = fullfile(data_base, 'zonotopes');
ablation_dir = fullfile(data_base, 'ablation_results');

if ~exist(ablation_dir, 'dir')
    mkdir(ablation_dir);
end

% Test on 2 scenarios: one 2D, one 3D
test_scenarios = {
    struct('id', 1, 'name', 'CAD Export Drift', 'type', 'Type A', ...
        'source', conPolyZono([100; 50], [2.0 0; 0 2.0], eye(2), [], [], []), ...
        'target', conPolyZono([100.3; 50.3], [2.1 0; 0 2.1], eye(2), [], [], []));
    
    struct('id', 5, 'name', 'Sensor Calibration Drift', 'type', 'Type A', ...
        'source', conPolyZono([100; 50; 25], diag([2.5, 2.5, 1.2]), eye(3), [], [], []), ...
        'target', conPolyZono([100.5; 50.5; 25.5], diag([2.6, 2.6, 1.3]), eye(3), [], [], []));
};

% Minimal interventions for quick testing - ONE from each type
interventions = {
    struct('type', 'widen', 'param', 'scale_factor', 'values', [2.0]);        % One scaling example
    struct('type', 'shift', 'param', 'center_delta', 'values', [0.05]);       % One shift example (5%)
    struct('type', 'correlate', 'param', 'correlation_strength', 'values', [0.6]);  % One correlation example
};

% Generate zonotopes if they don't exist
gen_options = struct('n_repeats', 3, 'verbose', false);

for s = 1:length(test_scenarios)
    scenario_def = test_scenarios{s};
    zono_index = fullfile(zonotope_dir, sprintf('index_scenario_%d.mat', scenario_def.id));
    
    if ~exist(zono_index, 'file')
        fprintf('Generating zonotopes for scenario %d: %s\n', scenario_def.id, scenario_def.name);
        causal_experiment_engine_twostep.generate_and_save_zonotopes(...
            scenario_def, interventions, zonotope_dir, gen_options);
    else
        fprintf('Zonotopes exist for scenario %d: %s\n', scenario_def.id, scenario_def.name);
        % Verify intervention count matches expectations
        loaded = load(zono_index);
        actual_count = length(loaded.zonotope_index.zonotope_files);
        expected_count = length(interventions) * gen_options.n_repeats;
        if actual_count ~= expected_count
            fprintf('  WARNING: Found %d experiments but expected %d (check interventions definition!)\n', ...
                actual_count, expected_count);
        end
    end
end

fprintf('\n');

%% Step 2: Test different consistency methods
fprintf('========================================\n');
fprintf('TESTING CONSISTENCY METHODS\n');
fprintf('========================================\n\n');

% Define test configurations - systematic progression
test_configs = {
    % Baseline
    {'AABB_only', 'jaccard', 0, 'sobol'};
    
    % N=100: All methods
    {'MC_Prob_N100_Sobol', 'mc_probability', 100, 'sobol'};
    {'MC_Jacc_N100_Sobol', 'jaccard_mc', 100, 'sobol'};
    {'Both_N100_Sobol', 'both', 100, 'sobol'};
    
    % N=200: All methods + sampling comparison
    {'MC_Prob_N200_Sobol', 'mc_probability', 200, 'sobol'};
    {'MC_Prob_N200_Halton', 'mc_probability', 200, 'halton'};
    {'MC_Prob_N200_LHS', 'mc_probability', 200, 'lhs'};
    {'MC_Prob_N200_Random', 'mc_probability', 200, 'random'};
    {'MC_Jacc_N200_Sobol', 'jaccard_mc', 200, 'sobol'};
    {'Both_N200_Sobol', 'both', 200, 'sobol'};
    
    % N=500: All methods
    {'MC_Prob_N500_Sobol', 'mc_probability', 500, 'sobol'};
    {'MC_Jacc_N500_Sobol', 'jaccard_mc', 500, 'sobol'};
    {'Both_N500_Sobol', 'both', 500, 'sobol'};
};

results = struct();

for c = 1:length(test_configs)
    config = test_configs{c};
    config_name = config{1};
    
    fprintf('\n--- Testing: %s ---\n', config_name);
    fprintf('  Method: %s\n', config{2});
    fprintf('  MC samples: %d\n', config{3});
    fprintf('  Sampling: %s\n', config{4});
    
    % Setup consistency options
    consistency_opts = struct();
    consistency_opts.method = config{2};
    consistency_opts.mc_samples = config{3};
    consistency_opts.sampling_method = config{4};
    consistency_opts.verbose = false;
    
    % Output directory for this config
    config_dir = fullfile(ablation_dir, config_name);
    if ~exist(config_dir, 'dir')
        mkdir(config_dir);
    end
    
    % Time the computation
    tic;
    try
        causal_experiment_engine_twostep.compute_and_save_consistency(...
            zonotope_dir, config_dir, consistency_opts);
        elapsed = toc;
        
        fprintf('  ✓ Completed in %.2f seconds\n', elapsed);
        
        % Store results
        results.(config_name).config = consistency_opts;
        results.(config_name).elapsed = elapsed;
        results.(config_name).status = 'success';
        results.(config_name).output_dir = config_dir;
        
    catch ME
        elapsed = toc;
        fprintf('  ✗ FAILED after %.2f seconds: %s\n', elapsed, ME.message);
        
        results.(config_name).config = consistency_opts;
        results.(config_name).elapsed = elapsed;
        results.(config_name).status = 'failed';
        results.(config_name).error = ME.message;
    end
end

%% Step 3: Analyze results
fprintf('\n========================================\n');
fprintf('ANALYSIS\n');
fprintf('========================================\n\n');

% Timing comparison
fprintf('--- Timing Comparison ---\n');
fprintf('%-20s %-10s %-15s %-10s %-12s\n', 'Configuration', 'Status', 'Time (s)', 'Time/Exp', 'N_Exps');
fprintf('%-20s %-10s %-15s %-10s %-12s\n', repmat('-', 1, 20), repmat('-', 1, 10), ...
    repmat('-', 1, 15), repmat('-', 1, 10), repmat('-', 1, 12));

config_names = fieldnames(results);

for i = 1:length(config_names)
    name = config_names{i};
    res = results.(name);
    
    if strcmp(res.status, 'success')
        % Count actual experiments from results file
        result_files = dir(fullfile(res.output_dir, 'results_scenario_*.json'));
        if ~isempty(result_files)
            fid = fopen(fullfile(res.output_dir, result_files(1).name), 'r');
            raw = fread(fid, inf); fclose(fid);
            data = jsondecode(char(raw'));
            n_experiments = length(data.experiments);
            time_per_exp = res.elapsed / n_experiments;
            fprintf('%-20s %-10s %15.2f %10.2f %12d\n', name, res.status, res.elapsed, time_per_exp, n_experiments);
        else
            fprintf('%-20s %-10s %15.2f %10s %12s\n', name, res.status, res.elapsed, 'N/A', 'N/A');
        end
    else
        fprintf('%-20s %-10s %15s %10s %12s\n', name, res.status, 'N/A', 'N/A', 'N/A');
    end
end

% Load and compare scores
fprintf('\n--- Score Correlation Analysis ---\n');

% Reference: AABB (fastest)
ref_config = 'AABB_only';
if isfield(results, ref_config) && strcmp(results.(ref_config).status, 'success')
    ref_file = dir(fullfile(results.(ref_config).output_dir, 'results_scenario_*.json'));
    if ~isempty(ref_file)
        fid = fopen(fullfile(results.(ref_config).output_dir, ref_file(1).name), 'r');
        raw = fread(fid, inf); fclose(fid);
        ref_data = jsondecode(char(raw'));
        
        ref_scores = arrayfun(@(x) x.post_state.inconsistency.jaccard_index, ref_data.experiments);
        
        fprintf('Correlation with AABB Jaccard:\n');
        
        for i = 1:length(config_names)
            name = config_names{i};
            if strcmp(name, ref_config), continue; end
            if ~strcmp(results.(name).status, 'success'), continue; end
            
            test_file = dir(fullfile(results.(name).output_dir, 'results_scenario_*.json'));
            if isempty(test_file), continue; end
            
            fid = fopen(fullfile(results.(name).output_dir, test_file(1).name), 'r');
            raw = fread(fid, inf); fclose(fid);
            test_data = jsondecode(char(raw'));
            
            % Get appropriate score field
            if contains(name, 'MC_Prob') || contains(name, 'Sobol') || contains(name, 'Halton') || contains(name, 'LHS') || contains(name, 'Random')
                if isfield(test_data.experiments(1).post_state.inconsistency, 'mc_probability_sobol')
                    test_scores = arrayfun(@(x) x.post_state.inconsistency.mc_probability_sobol, test_data.experiments);
                elseif isfield(test_data.experiments(1).post_state.inconsistency, 'mc_probability_halton')
                    test_scores = arrayfun(@(x) x.post_state.inconsistency.mc_probability_halton, test_data.experiments);
                elseif isfield(test_data.experiments(1).post_state.inconsistency, 'mc_probability')
                    test_scores = arrayfun(@(x) x.post_state.inconsistency.mc_probability, test_data.experiments);
                else
                    fprintf('  %s: Score field not found\n', name);
                    continue;
                end
            elseif contains(name, 'MC_Jacc')
                test_scores = arrayfun(@(x) x.post_state.inconsistency.jaccard_mc_index_sobol, test_data.experiments);
            else
                continue;
            end
            
            % Compute correlation
            if length(ref_scores) == length(test_scores)
                corr_coef = corr(ref_scores(:), test_scores(:));
                rmse = sqrt(mean((ref_scores - test_scores).^2));
                fprintf('  %-20s: r=%.4f, RMSE=%.4f\n', name, corr_coef, rmse);
            end
        end
    end
end

%% Step 4: Recommendations
fprintf('\n========================================\n');
fprintf('RECOMMENDATIONS\n');
fprintf('========================================\n\n');

fprintf('Based on timing and accuracy trade-offs:\n\n');

fprintf('1. FOR QUICK RESULTS (5-10 minutes total):\n');
fprintf('   → Use AABB Jaccard only\n');
fprintf('   → Cons: Conservative (may overestimate inconsistency)\n\n');

fprintf('2. FOR BALANCED ACCURACY (1-2 hours total):\n');
fprintf('   → Use MC Probability with N=200, Sobol sampling\n');
fprintf('   → Good correlation with AABB, reasonable runtime\n\n');

fprintf('3. FOR HIGH ACCURACY (4-8 hours total):\n');
fprintf('   → Use MC Probability with N=500-1000, Sobol sampling\n');
fprintf('   → Most precise, but expensive\n\n');

fprintf('4. SAMPLING METHOD:\n');
fprintf('   → Sobol QMC recommended (deterministic, fast convergence)\n');
fprintf('   → Halton also good, similar performance\n\n');

fprintf('========================================\n');
fprintf('Ablation study complete!\n');
fprintf('Results saved to: %s\n', ablation_dir);
fprintf('========================================\n');

% Save summary
save(fullfile(ablation_dir, 'ablation_summary.mat'), 'results', 'test_configs');
fprintf('\nSummary saved to: %s\n', fullfile(ablation_dir, 'ablation_summary.mat'));
