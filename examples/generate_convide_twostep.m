% GENERATE CONVIDE SCENARIOS - TWO STEP WORKFLOW
%
% This script demonstrates the two-step approach:
%   Step 1: Generate and save zonotopes (expensive, do once)
%   Step 2: Compute consistency scores (cheap, can repeat with different methods)
%
% Benefits:
%   - Try different consistency methods without regenerating zonotopes
%   - Add new metrics later without redoing interventions
%   - Faster iteration on analysis
%
% Version: 5.0 (Two-step workflow)
% Date: February 2026

%% Setup
clear; close all; clc;

% Configuration
run_step1 = true;   % Set to false to skip zonotope generation
run_step2 = true;   % Set to false to skip consistency computation

seed = 2025;
rng(seed);
n_repeats = 5;

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
fprintf('TWO-STEP CONVIDE DATA GENERATION\n');
fprintf('========================================\n');
fprintf('Step 1: Generate zonotopes (run_step1=%d)\n', run_step1);
fprintf('Step 2: Compute consistency (run_step2=%d)\n', run_step2);
fprintf('Output structure: data/zonotopes/ + data/measurements/\n\n');

%% Configuration: Enable/disable scenario dimensions
run_2d_scenarios = true;
run_3d_scenarios = true;
run_4d_scenarios = true;

%% Configuration: Select specific scenarios to run
% Specify which scenario IDs to process (1-12)
% Examples:
%   scenarios_to_run = 1:12;        % Run all scenarios
scenarios_to_run = 2:12;      % Skip scenario 1
%   scenarios_to_run = [1, 3, 5];   % Run only scenarios 1, 3, and 5
%scenarios_to_run = 1:12;  % Change this to control which scenarios run

%% Check if zonotopes already exist (skip generation if they do)
skip_existing = true;  % Set to false to regenerate all zonotopes

%% Output directories
data_base = fullfile(fileparts(pwd), 'data');
zonotope_dir = fullfile(data_base, 'zonotopes');
results_dir = fullfile(data_base, 'measurements');

%% Intervention Configuration
interventions = {
    struct('type', 'widen', 'param', 'scale_factor', ...
        'values', [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]);
    struct('type', 'shrink', 'param', 'scale_factor', ...
        'values', [0.01, 0.1, 0.5, 1.0, 2.0]);
    struct('type', 'correlate', 'param', 'correlation_strength', ...
        'values', [0.0, 0.3, 0.6, 0.8, 0.9, 0.95]);
};

%% STEP 1: GENERATE ZONOTOPES (DO ONCE)
%% ======================
if run_step1
    fprintf('\n========================================\n');
    fprintf('STEP 1: GENERATING ZONOTOPES\n');
    fprintf('========================================\n');
    
    % Generate options
    gen_options = struct();
    gen_options.n_repeats = n_repeats;
    gen_options.verbose = true;
    
    % Helper function to check if scenario zonotopes exist
    check_zonotopes_exist = @(scenario_id) ...
        exist(fullfile(zonotope_dir, sprintf('scenario_%d_index.mat', scenario_id)), 'file') == 2;
    
    %% 2D SCENARIOS
    if run_2d_scenarios
        fprintf('\n--- 2D SCENARIOS (4 scenarios) ---\n');
        
        scenarios_2d = {
            struct('id', 1, 'name', 'CAD Export Drift', 'type', 'Type A', ...
                'source', conPolyZono([100; 50], [2.0 0; 0 2.0], eye(2), [], [], []), ...
                'target', conPolyZono([100.3; 50.3], [2.1 0; 0 2.1], eye(2), [], [], []));
            
            struct('id', 2, 'name', 'MBSE Version Mismatch', 'type', 'Type B', ...
                'source', conPolyZono([50; 25], [1.5 0; 0 1.5], eye(2), [], [], []), ...
                'target', conPolyZono([50; 25], [1.8 0; 0 1.8], eye(2), [], [], []));
            
            struct('id', 3, 'name', 'Documentation Sync', 'type', 'Type B', ...
                'source', conPolyZono([75; 40], [2.0 0; 0 2.0], eye(2), [], [], []), ...
                'target', conPolyZono([75.5; 40.5], [2.0 0; 0 2.0], eye(2), [], [], []));
            
            struct('id', 4, 'name', 'Control Design Conflict', 'type', 'Type C', ...
                'source', conPolyZono([60; 30], [1.8 0; 0 1.8], eye(2), [], [], []), ...
                'target', conPolyZono([60.8; 30.8], [1.5 0; 0 1.5], eye(2), [], [], []));
        };
        
        for s = 1:length(scenarios_2d)
            scenario_def = scenarios_2d{s};
            
            % Check if this scenario should be run
            if ~ismember(scenario_def.id, scenarios_to_run)
                fprintf('\n  ⊘ Scenario %d: %s - not in scenarios_to_run, skipping\n', ...
                    scenario_def.id, scenario_def.name);
                continue;
            end
            
            % Check if already exists
            if skip_existing && check_zonotopes_exist(scenario_def.id)
                fprintf('\n  ⏭ Scenario %d: %s - zonotopes exist, skipping\n', ...
                    scenario_def.id, scenario_def.name);
                continue;
            end
            
            fprintf('\n  Generating zonotopes for Scenario %d: %s\n', ...
                scenario_def.id, scenario_def.name);
            
            causal_experiment_engine_twostep.generate_and_save_zonotopes(...
                scenario_def, interventions, zonotope_dir, gen_options);
        end
    end
    
    %% 3D SCENARIOS
    if run_3d_scenarios
        fprintf('\n--- 3D SCENARIOS (4 scenarios) ---\n');
        
        scenarios_3d = {
            struct('id', 5, 'name', 'Sensor Calibration Drift', 'type', 'Type A', ...
                'source', conPolyZono([100; 50; 25], diag([2.5, 2.5, 1.2]), eye(3), [], [], []), ...
                'target', conPolyZono([100.5; 50.5; 25.5], diag([2.6, 2.6, 1.3]), eye(3), [], [], []));
            
            struct('id', 6, 'name', 'Requirements Ambiguity', 'type', 'Type D', ...
                'source', conPolyZono([80; 40; 20], diag([2.0, 2.0, 1.0]), eye(3), [], [], []), ...
                'target', conPolyZono([80; 40; 20], diag([2.5, 2.5, 1.2]), eye(3), [], [], []));
            
            struct('id', 7, 'name', 'Test Configuration Mismatch', 'type', 'Type B', ...
                'source', conPolyZono([70; 35; 18], diag([2.2, 2.2, 1.1]), eye(3), [], [], []), ...
                'target', conPolyZono([70.4; 35.4; 18.4], diag([2.2, 2.2, 1.1]), eye(3), [], [], []));
            
            struct('id', 8, 'name', 'Simulation Numerical Error', 'type', 'Type A', ...
                'source', conPolyZono([90; 45; 22], diag([2.5, 2.5, 1.3]), eye(3), [], [], []), ...
                'target', conPolyZono([90.6; 45.6; 22.6], diag([2.2, 2.2, 1.1]), eye(3), [], [], []));
        };
        
        for s = 1:length(scenarios_3d)
            scenario_def = scenarios_3d{s};
            
            % Check if this scenario should be run
            if ~ismember(scenario_def.id, scenarios_to_run)
                fprintf('\n  ⊘ Scenario %d: %s - not in scenarios_to_run, skipping\n', ...
                    scenario_def.id, scenario_def.name);
                continue;
            end
            
            % Check if already exists
            if skip_existing && check_zonotopes_exist(scenario_def.id)
                fprintf('\n  ⏭ Scenario %d: %s - zonotopes exist, skipping\n', ...
                    scenario_def.id, scenario_def.name);
                continue;
            end
            
            fprintf('\n  Generating zonotopes for Scenario %d: %s\n', ...
                scenario_def.id, scenario_def.name);
            
            causal_experiment_engine_twostep.generate_and_save_zonotopes(...
                scenario_def, interventions, zonotope_dir, gen_options);
        end
    end
    
    %% 4D SCENARIOS
    if run_4d_scenarios
        fprintf('\n--- 4D SCENARIOS (4 scenarios) ---\n');
        
        scenarios_4d = {
            struct('id', 9, 'name', 'Multi-Physics Coupling Error', 'type', 'Type C', ...
                'source', conPolyZono([100; 50; 25; 12], diag([3.0, 3.0, 1.5, 0.75]), eye(4), [], [], []), ...
                'target', conPolyZono([100.5; 50.5; 25.5; 12.5], diag([3.1, 3.1, 1.6, 0.8]), eye(4), [], [], []));
            
            struct('id', 10, 'name', 'Interface Specification Gap', 'type', 'Type D', ...
                'source', conPolyZono([85; 42; 21; 10], diag([2.5, 2.5, 1.2, 0.6]), eye(4), [], [], []), ...
                'target', conPolyZono([85; 42; 21; 10], diag([3.0, 3.0, 1.5, 0.75]), eye(4), [], [], []));
            
            struct('id', 11, 'name', 'Parameter Estimation Bias', 'type', 'Type A', ...
                'source', conPolyZono([95; 48; 24; 11], diag([2.8, 2.8, 1.4, 0.7]), eye(4), [], [], []), ...
                'target', conPolyZono([95.4; 48.4; 24.4; 11.4], diag([2.8, 2.8, 1.4, 0.7]), eye(4), [], [], []));
            
            struct('id', 12, 'name', 'Traceability Link Inconsistency', 'type', 'Type B', ...
                'source', conPolyZono([75; 38; 19; 9], diag([2.5, 2.5, 1.2, 0.6]), eye(4), [], [], []), ...
                'target', conPolyZono([75.6; 38.6; 19.6; 9.6], diag([2.2, 2.2, 1.1, 0.55]), eye(4), [], [], []));
        };
        
        for s = 1:length(scenarios_4d)
            scenario_def = scenarios_4d{s};
            
            % Check if this scenario should be run
            if ~ismember(scenario_def.id, scenarios_to_run)
                fprintf('\n  ⊘ Scenario %d: %s - not in scenarios_to_run, skipping\n', ...
                    scenario_def.id, scenario_def.name);
                continue;
            end
            
            % Check if already exists
            if skip_existing && check_zonotopes_exist(scenario_def.id)
                fprintf('\n  ⏭ Scenario %d: %s - zonotopes exist, skipping\n', ...
                    scenario_def.id, scenario_def.name);
                continue;
            end
            
            fprintf('\n  Generating zonotopes for Scenario %d: %s\n', ...
                scenario_def.id, scenario_def.name);
            
            causal_experiment_engine_twostep.generate_and_save_zonotopes(...
                scenario_def, interventions, zonotope_dir, gen_options);
        end
    end
    
    fprintf('\n========================================\n');
    fprintf('STEP 1 COMPLETE: Zonotopes saved to:\n');
    fprintf('  %s\n', zonotope_dir);
    fprintf('========================================\n');
end

%% ======================
%% STEP 2: COMPUTE CONSISTENCY (CAN REPEAT)
%% ======================
if run_step2
    fprintf('\n========================================\n');
    fprintf('STEP 2: COMPUTING CONSISTENCY SCORES\n');
    fprintf('========================================\n');
    
    % Consistency options - TRY DIFFERENT METHODS HERE!
    consistency_opts = struct();
    consistency_opts.method = 'both';           % Try: 'jaccard', 'mc_probability', 'both'
    consistency_opts.mc_samples = 500;          % 500 is good for development, use 1000-2000 for final results
    consistency_opts.verbose = true;
    
    % Compute consistency from saved zonotopes
    causal_experiment_engine_twostep.compute_and_save_consistency(...
        zonotope_dir, results_dir, consistency_opts);
    
    fprintf('\n========================================\n');
    fprintf('STEP 2 COMPLETE: Results saved to:\n');
    fprintf('  %s\n', results_dir);
    fprintf('========================================\n');
end

%% ======================
%% DEMONSTRATION: RE-MEASURE WITH DIFFERENT METHOD
%% ======================
fprintf('\n========================================\n');
fprintf('EXAMPLE: HOW TO RE-MEASURE CONSISTENCY\n');
fprintf('========================================\n');
fprintf('You can now re-run Step 2 with different settings without\n');
fprintf('regenerating the expensive zonotopes:\n\n');
fprintf('Example 1: Try only Jaccard method\n');
fprintf('  consistency_opts.method = ''jaccard'';\n');
fprintf('  causal_experiment_engine_twostep.compute_and_save_consistency(...\n');
fprintf('      zonotope_dir, results_dir_jaccard, consistency_opts);\n\n');
fprintf('Example 2: Try MC with more samples\n');
fprintf('  consistency_opts.method = ''mc_probability'';\n');
fprintf('  consistency_opts.mc_samples = 10000;\n');
fprintf('  causal_experiment_engine_twostep.compute_and_save_consistency(...\n');
fprintf('      zonotope_dir, results_dir_mc_10k, consistency_opts);\n\n');
fprintf('Example 3: Use convenience function\n');
fprintf('  causal_experiment_engine_twostep.regenerate_consistency_all(...\n');
fprintf('      zonotope_dir, results_base_dir, ''mc_probability'', 5000);\n');
fprintf('========================================\n\n');

%% Summary
fprintf('========================================\n');
fprintf('TWO-STEP WORKFLOW COMPLETE\n');
fprintf('========================================\n');
fprintf('Zonotopes:    %s\n', zonotope_dir);
fprintf('Measurements: %s\n\n', results_dir);
fprintf('To re-measure consistency:\n');
fprintf('  1. Set run_step1=false, run_step2=true\n');
fprintf('  2. Change consistency_opts.method\n');
fprintf('  3. Re-run this script\n\n');
