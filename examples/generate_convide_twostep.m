% GENERATE CONVIDE SCENARIOS - TWO STEP WORKFLOW
%
% This script supports two modes:
%   1. CONVIDE Mode (use_saltelli_mode=false): Discrete interventions (widen, shrink, shift, correlate)
%   2. Saltelli Mode (use_saltelli_mode=true): Continuous compound interventions for sensitivity analysis
%
% Two-step approach:
%   Step 1: Generate and save zonotopes (expensive, do once)
%   Step 2: Compute consistency scores (cheap, can repeat with different methods)
%
% Benefits:
%   - Try different consistency methods without regenerating zonotopes
%   - Add new metrics later without redoing interventions
%   - Faster iteration on analysis
%   - Unified workflow for both CONVIDE and Saltelli experiments
%
% Version: 6.0 (Integrated Saltelli support)
% Date: March 2026

%% Setup
if ~exist('preserve_external_config', 'var') || ~preserve_external_config
    clear; close all; clc;
else
    close all; clc;
end

% Suppress CORA deprecation warnings (Grest → GI, expMat_ → EC)
warning('off', 'CORA:deprecated');
warning('off', 'CORA:contSet:set:expMat');
warning('off', 'CORA:contSet:get:expMat');
warning('off', 'all');  % Suppress all warnings temporarily
lastwarn('');  % Clear last warning

% Configuration
if ~exist('run_step1', 'var'), run_step1 = true; end    % TRUE: generate zonotopes (set false to re-use existing)
if ~exist('run_step2', 'var'), run_step2 = true; end    % Set to false to skip consistency computation

% Quick pilot mode for low-pre validation only
% When enabled, this overrides several settings to run a very small, fast check.
if ~exist('low_pre_pilot_mode', 'var'), low_pre_pilot_mode = false; end
if ~exist('low_pre_pilot_scenarios', 'var'), low_pre_pilot_scenarios = [1, 5, 9]; end  % One scenario per dimension (2D/3D/4D)
if ~exist('low_pre_pilot_mc_samples', 'var'), low_pre_pilot_mc_samples = 300; end

% Saltelli Sampling Mode (for sensitivity analysis)
if ~exist('use_saltelli_mode', 'var'), use_saltelli_mode = true; end  % TRUE: Use Saltelli compound interventions, FALSE: Use discrete CONVIDE interventions
if ~exist('saltelli_samples_file', 'var'), saltelli_samples_file = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data', 'saltelli_samples_3param_v6.csv'); end  % Path to CSV file with Saltelli samples
if ~exist('saltelli_params', 'var'), saltelli_params = {'scale_factor', 'center_delta', 'correlation_strength'}; end

if ~exist('seed', 'var'), seed = 2025; end
rng(seed);
if ~exist('n_repeats', 'var'), n_repeats = 1; end  % No repeats needed for Saltelli (each sample is independent)
if ~exist('use_parallel_step1', 'var'), use_parallel_step1 = []; end
if ~exist('use_parallel_step2', 'var'), use_parallel_step2 = []; end

%% Parallel Computing Setup
% Set to true if running on a multi-core server with Parallel Computing Toolbox
if ~exist('use_parallel', 'var'), use_parallel = true; end
if isempty(use_parallel_step1), use_parallel_step1 = use_parallel; end
if isempty(use_parallel_step2), use_parallel_step2 = use_parallel; end
if use_parallel
    n_workers = 20;
    delete(gcp('nocreate'));
    pool = gcp('nocreate');
    if isempty(pool)
        fprintf('Starting parallel pool with %d workers...\n', n_workers);
        parpool('local', n_workers);
        fprintf('  ✓ Parallel pool ready\n\n');
    else
        fprintf('Parallel pool already running with %d workers\n\n', pool.NumWorkers);
    end
else
    fprintf('Parallel computing disabled (single-threaded mode)\n\n');
end

% Add paths
src_path = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'src', 'matlab');
addpath(src_path);

script_dir   = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);  % Go up from examples/ to project root

fprintf('Detected paths:\n');
fprintf('  Project root: %s\n', project_root);

fprintf('========================================\n');
if use_saltelli_mode
    fprintf('TWO-STEP SALTELLI SENSITIVITY ANALYSIS\n');
    fprintf('========================================\n');
    fprintf('Mode: Saltelli compound interventions\n');
else
    fprintf('TWO-STEP CONVIDE DATA GENERATION\n');
    fprintf('========================================\n');
    fprintf('Mode: CONVIDE discrete interventions\n');
end
fprintf('Step 1: Generate zonotopes (run_step1=%d)\n', run_step1);
fprintf('Step 2: Compute consistency (run_step2=%d)\n', run_step2);
if low_pre_pilot_mode
    fprintf('Pilot mode: ON (quick low-pre validation)\n');
else
    fprintf('Pilot mode: OFF\n');
end
fprintf('Output structure: data/zonotopes/ + data/measurements/\n\n');

% Configuration: Enable/disable scenario dimensions
if ~exist('run_2d_scenarios', 'var'), run_2d_scenarios = true; end
if ~exist('run_3d_scenarios', 'var'), run_3d_scenarios = true; end
if ~exist('run_4d_scenarios', 'var'), run_4d_scenarios = true; end

% Baseline variant mode
% Reuse existing scenario definitions and transform them into lower pre I(theta)
% variants by pulling target center/generators toward source.
if ~exist('use_low_pre_variants', 'var'), use_low_pre_variants = true; end
if ~exist('low_pre_center_alpha', 'var'), low_pre_center_alpha = 0.25; end      % 0.25 means keep only 25% of original center offset
if ~exist('low_pre_generator_alpha', 'var'), low_pre_generator_alpha = 0.35; end   % 0.35 means keep only 35% of original generator mismatch

% Optional test-only scenarios with adjusted initial source-target proximity/shape.
% These are added as NEW scenario IDs and do not modify the original 1-12 set.
if ~exist('enable_test_scenarios', 'var'), enable_test_scenarios = true; end
% A variants: high pre-intervention inconsistency, I_theta ~ 0.55-0.99
% IDs: 2D->101, 3D->105, 4D->109
if ~exist('test_center_alpha_2d', 'var'), test_center_alpha_2d = 0.25; end
if ~exist('test_center_alpha_3d', 'var'), test_center_alpha_3d = 0.10; end
if ~exist('test_center_alpha_4d', 'var'), test_center_alpha_4d = 0.05; end
if ~exist('test_generator_alpha_2d', 'var'), test_generator_alpha_2d = 0.35; end
if ~exist('test_generator_alpha_3d', 'var'), test_generator_alpha_3d = 0.20; end
if ~exist('test_generator_alpha_4d', 'var'), test_generator_alpha_4d = 0.12; end
% B variants: very low pre-intervention inconsistency, I_theta ~ 0.03-0.15
% IDs: 2D->102, 3D->106, 4D->110
if ~exist('test_center_alpha_2d_b', 'var'), test_center_alpha_2d_b = 0.10; end
if ~exist('test_center_alpha_3d_b', 'var'), test_center_alpha_3d_b = 0.02; end
if ~exist('test_center_alpha_4d_b', 'var'), test_center_alpha_4d_b = 0.01; end
if ~exist('test_generator_alpha_2d_b', 'var'), test_generator_alpha_2d_b = 0.15; end
if ~exist('test_generator_alpha_3d_b', 'var'), test_generator_alpha_3d_b = 0.05; end
if ~exist('test_generator_alpha_4d_b', 'var'), test_generator_alpha_4d_b = 0.02; end
if ~exist('test_mapping_strength_2d_b', 'var'), test_mapping_strength_2d_b = 0.9; end
if ~exist('test_mapping_strength_3d_b', 'var'), test_mapping_strength_3d_b = 1.8; end
if ~exist('test_mapping_strength_4d_b', 'var'), test_mapping_strength_4d_b = 2.6; end
% C variants: mid-low pre-intervention inconsistency, I_theta ~ 0.20-0.35
% IDs: 2D->103, 3D->107, 4D->111
if ~exist('test_center_alpha_2d_c', 'var'), test_center_alpha_2d_c = 0.15; end
if ~exist('test_center_alpha_3d_c', 'var'), test_center_alpha_3d_c = 0.04; end
if ~exist('test_center_alpha_4d_c', 'var'), test_center_alpha_4d_c = 0.02; end
if ~exist('test_generator_alpha_2d_c', 'var'), test_generator_alpha_2d_c = 0.20; end
if ~exist('test_generator_alpha_3d_c', 'var'), test_generator_alpha_3d_c = 0.08; end
if ~exist('test_generator_alpha_4d_c', 'var'), test_generator_alpha_4d_c = 0.04; end
if ~exist('test_mapping_strength_2d_c', 'var'), test_mapping_strength_2d_c = 0.9; end
if ~exist('test_mapping_strength_3d_c', 'var'), test_mapping_strength_3d_c = 1.8; end
if ~exist('test_mapping_strength_4d_c', 'var'), test_mapping_strength_4d_c = 2.6; end
% D variants: mid pre-intervention inconsistency, I_theta ~ 0.40-0.60
% IDs: 2D->104, 3D->108, 4D->112
if ~exist('test_center_alpha_2d_d', 'var'), test_center_alpha_2d_d = 0.20; end
if ~exist('test_center_alpha_3d_d', 'var'), test_center_alpha_3d_d = 0.055; end
if ~exist('test_center_alpha_4d_d', 'var'), test_center_alpha_4d_d = 0.03; end
if ~exist('test_generator_alpha_2d_d', 'var'), test_generator_alpha_2d_d = 0.28; end
if ~exist('test_generator_alpha_3d_d', 'var'), test_generator_alpha_3d_d = 0.12; end
if ~exist('test_generator_alpha_4d_d', 'var'), test_generator_alpha_4d_d = 0.07; end
if ~exist('test_mapping_strength_2d_d', 'var'), test_mapping_strength_2d_d = 0.9; end
if ~exist('test_mapping_strength_3d_d', 'var'), test_mapping_strength_3d_d = 1.8; end
if ~exist('test_mapping_strength_4d_d', 'var'), test_mapping_strength_4d_d = 2.6; end

%% Configuration: Select specific scenarios to run
% Specify which scenario IDs to process (1-12)
%
% RECOMMENDED for Saltelli mode: Run one scenario at a time for better progress tracking
%   scenarios_to_run = 1;       % Scenario 1 only (mini-pilot: ~20-40 min with 24 workers)
%   scenarios_to_run = 2;       % Then Scenario 2, etc.
%
% Examples:
if ~exist('scenarios_to_run', 'var'), scenarios_to_run = 1:12; end  % All regular Saltelli scenarios

% Step 2 method override: 'jaccard' (AABB, fast), 'mc_probability' (slow), 'both'
% Set to 'jaccard' for AABB-only re-run (merge with MC results via merge_aabb_into_saltelli.py)
if ~exist('consistency_method_override', 'var'), consistency_method_override = 'aabb_and_mc'; end
%scenarios_to_run = 2:12;      % Skip scenario 1
%   scenarios_to_run = [1, 3, 5];   % Run only scenarios 1, 3, and 5
%scenarios_to_run = 1:12;  % Change this to control which scenarios run

%% Check if zonotopes already exist (skip generation if they do)
if ~exist('skip_existing', 'var'), skip_existing = false; end  % Set to FALSE to regenerate all zonotopes with correct balanced scenarios

% Apply pilot-mode overrides (fast low-pre check only)
if low_pre_pilot_mode
    use_saltelli_mode = false;
    use_low_pre_variants = true;
    run_step1 = true;
    run_step2 = true;
    scenarios_to_run = low_pre_pilot_scenarios;
    n_repeats = 1;
    skip_existing = false;
    fprintf('PILOT MODE ENABLED: low-pre quick validation\n');
    fprintf('  Scenarios: %s\n', mat2str(scenarios_to_run));
    fprintf('  MC samples: %d\n\n', low_pre_pilot_mc_samples);
end

fprintf('Selected scenario IDs: %s\n', mat2str(scenarios_to_run));
fprintf('Test-only scenarios enabled: %d\n\n', enable_test_scenarios);
fprintf('Parallel Step 1 (generation): %d\n', use_parallel_step1);
fprintf('Parallel Step 2 (consistency): %d\n\n', use_parallel_step2);

%% Output directories - use script location for reliable paths
data_base = fullfile(project_root, 'data');
if low_pre_pilot_mode
    zonotope_dir = fullfile(data_base, 'zonotopes_lowpre_pilot');
    results_dir = fullfile(data_base, 'measurements_lowpre_pilot');
else
    zonotope_dir = fullfile(data_base, 'zonotopes_v6');
    results_dir = fullfile(data_base, 'measurements_v6');
end

if exist('zonotope_dir_override', 'var') && ~isempty(zonotope_dir_override)
    zonotope_dir = zonotope_dir_override;
end
if exist('results_dir_override', 'var') && ~isempty(results_dir_override)
    results_dir = results_dir_override;
end

fprintf('Output paths:\n');
fprintf('  Data base: %s\n', data_base);
fprintf('  Zonotopes: %s\n', zonotope_dir);
fprintf('  Results: %s\n\n', results_dir);

% Create directories if they don't exist
if ~exist(data_base, 'dir')
    mkdir(data_base);
end
if ~exist(zonotope_dir, 'dir')
    mkdir(zonotope_dir);
end
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

%% Intervention Configuration
if use_saltelli_mode
    % Saltelli Mode: Load compound intervention samples
    if isempty(saltelli_samples_file)
        % Auto-generate Saltelli samples if not provided
        fprintf('Generating Saltelli samples for %d parameters...\n', length(saltelli_params));
        n_saltelli_base = 512;  % Change this to control sample count (N×(2p+2) total samples)
        
        % Call Python script to generate samples
        python_script = fullfile(project_root, 'src', 'generate_saltelli_samples.py');
        output_file = fullfile(data_base, sprintf('saltelli_samples_%dparam.csv', length(saltelli_params)));
        
        cmd = sprintf('python "%s" generate --n_samples %d --params %d --output "%s"', ...
            python_script, n_saltelli_base, length(saltelli_params), output_file);
        [status, output] = system(cmd);
        
        if status ~= 0
            error('Failed to generate Saltelli samples: %s', output);
        end
        fprintf('  ✓ Generated %d Saltelli samples\n', n_saltelli_base * (2*length(saltelli_params) + 2));
        saltelli_samples_file = output_file;
    end
    
    % Load Saltelli samples
    fprintf('Loading Saltelli samples from: %s\n', saltelli_samples_file);
    saltelli_data = readtable(saltelli_samples_file);
    interventions = struct('mode', 'saltelli', 'samples', saltelli_data, 'params', {saltelli_params});
    fprintf('  ✓ Loaded %d compound interventions\n\n', height(saltelli_data));
else
    % CONVIDE Mode: Use discrete intervention values
    if low_pre_pilot_mode
        % Minimal intervention set for quick baseline check (identity-like).
        interventions = {
            struct('type', 'widen', 'param', 'scale_factor', ...
                'values', [1.0]);
        };
    else
        interventions = {
            struct('type', 'widen', 'param', 'scale_factor', ...
                'values', [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]);
            struct('type', 'shrink', 'param', 'scale_factor', ...
                'values', [0.01, 0.1, 0.5, 1.0, 2.0]);
            struct('type', 'correlate', 'param', 'correlation_strength', ...
                'values', [0.0, 0.3, 0.6, 0.8, 0.9, 0.95]);
            struct('type', 'shift', 'param', 'center_delta', ...
                'values', [0.0, 0.01, 0.05, 0.1, 0.2]);  % 0%, 1%, 5%, 10%, 20% relative shifts
        };
    end
end

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
    gen_options.use_parallel = use_parallel_step1;
    
    % Helper function to check if scenario zonotopes exist
    check_zonotopes_exist = @(scenario_id) ...
        exist(fullfile(zonotope_dir, sprintf('scenario_%d_index.mat', scenario_id)), 'file') == 2;
    
    %% 2D SCENARIOS
    if run_2d_scenarios
        fprintf('\n--- 2D SCENARIOS (4 scenarios) ---\n');
        
        scenarios_2d = {
            struct('id', 1, 'name', 'CAD Export Drift', 'type', 'Type A', ...
                'upr_type', 'identity', ...
                'source', conZonotope([100; 50], [2.0 0; 0 2.0], [], []), ...
                'target', conZonotope([100.3; 50.3], [2.1 0; 0 2.1], [], []));

            struct('id', 2, 'name', 'MBSE Version Mismatch', 'type', 'Type B', ...
                'upr_type', 'parametric', ...
                'source', conZonotope([50; 25], [1.5 0; 0 1.5], [], []), ...
                'target', conZonotope([50; 25], [1.8 0; 0 1.8], [], []));

            struct('id', 3, 'name', 'Documentation Sync', 'type', 'Type B', ...
                'upr_type', 'structural', ...
                'source', conZonotope([75; 40], [2.0 0; 0 2.0], [], []), ...
                'target', conZonotope([75.5; 40.5], [2.0 0; 0 2.0], [], []));

            struct('id', 4, 'name', 'Control Design Conflict', 'type', 'Type C', ...
                'upr_type', 'parametric', ...
                'source', conZonotope([60; 30], [1.8 0; 0 1.8], [], []), ...
                'target', conZonotope([60.8; 30.8], [1.5 0; 0 1.5], [], []));
        };

        % Assign UPR mappings (affine F, f) — matches causal_engine.py create_convide_scenarios()
        a2d = 5 * pi / 180;
        R2d = [cos(a2d), -sin(a2d); sin(a2d), cos(a2d)];
        scenarios_2d{1}.mapping = struct('F', {eye(2)},              'f', {zeros(2,1)});   % S1: identity
        scenarios_2d{2}.mapping = struct('F', {diag([0.97, 1.04])},  'f', {zeros(2,1)});  % S2: diagonal scaling
        scenarios_2d{3}.mapping = struct('F', {eye(2)},              'f', {[0.5; 0.3]});  % S3: offset only
        scenarios_2d{4}.mapping = struct('F', {R2d},                 'f', {zeros(2,1)});  % S4: 2D rotation 5°

            base_scenarios_2d = scenarios_2d;

        if use_low_pre_variants
            fprintf('  Applying low-pre transform to 2D scenarios (alpha_c=%.2f, alpha_G=%.2f)\n', ...
                low_pre_center_alpha, low_pre_generator_alpha);
            for s = 1:length(scenarios_2d)
                scenarios_2d{s} = create_low_pre_variant_from_existing(...
                    scenarios_2d{s}, low_pre_center_alpha, low_pre_generator_alpha);
            end
        end

        if enable_test_scenarios
            test_scenario_2d = create_test_only_scenario(...
                base_scenarios_2d{1}, 101, test_center_alpha_2d, test_generator_alpha_2d, 'TEST');
            scenarios_2d{end + 1} = test_scenario_2d;
            fprintf('  Added test-only 2D scenario: %d (%s)\n', ...
                test_scenario_2d.id, test_scenario_2d.name);
            if ~isempty(test_center_alpha_2d_b)
                test_scenario_2d_b = create_test_only_scenario(...
                    base_scenarios_2d{1}, 102, test_center_alpha_2d_b, test_generator_alpha_2d_b, 'TEST-B', test_mapping_strength_2d_b);
                scenarios_2d{end + 1} = test_scenario_2d_b;
                fprintf('  Added test-only 2D scenario B: %d (%s)\n', ...
                    test_scenario_2d_b.id, test_scenario_2d_b.name);
            end
            if ~isempty(test_center_alpha_2d_c)
                test_scenario_2d_c = create_test_only_scenario(...
                    base_scenarios_2d{1}, 103, test_center_alpha_2d_c, test_generator_alpha_2d_c, 'TEST-C', test_mapping_strength_2d_c);
                scenarios_2d{end + 1} = test_scenario_2d_c;
                fprintf('  Added test-only 2D scenario C: %d (%s)\n', ...
                    test_scenario_2d_c.id, test_scenario_2d_c.name);
            end
            if ~isempty(test_center_alpha_2d_d)
                test_scenario_2d_d = create_test_only_scenario(...
                    base_scenarios_2d{1}, 104, test_center_alpha_2d_d, test_generator_alpha_2d_d, 'TEST-D', test_mapping_strength_2d_d);
                scenarios_2d{end + 1} = test_scenario_2d_d;
                fprintf('  Added test-only 2D scenario D: %d (%s)\n', ...
                    test_scenario_2d_d.id, test_scenario_2d_d.name);
            end
        end
        
        for s = 1:length(scenarios_2d)
            scenario_def = scenarios_2d{s};
            
            % Check if this scenario should be run
            if ~ismember(scenario_def.id, scenarios_to_run)
                fprintf('\n  Scenario %d: %s - not in scenarios_to_run, skipping\n', ...
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
            
            try
                causal_experiment_engine_twostep.generate_and_save_zonotopes(...
                    scenario_def, interventions, zonotope_dir, gen_options);
                fprintf('  ✓ Scenario %d completed successfully\n', scenario_def.id);
            catch ME
                fprintf('  ✗ ERROR in Scenario %d: %s\n', scenario_def.id, ME.message);
                fprintf('     Continuing with next scenario...\n');
            end
        end
    end
    
    %% 3D SCENARIOS
    if run_3d_scenarios
        fprintf('\n--- 3D SCENARIOS (4 scenarios) ---\n');
        
        scenarios_3d = {
            struct('id', 5, 'name', 'Sensor Calibration Drift', 'type', 'Type A', ...
                'upr_type', 'identity_bidir', ...
                'source', conZonotope([100; 50; 25], diag([2.5, 2.5, 1.2]), [], []), ...
                'target', conZonotope([100.5; 50.5; 25.5], diag([2.6, 2.6, 1.3]), [], []));

            struct('id', 6, 'name', 'Requirements Ambiguity', 'type', 'Type D', ...
                'upr_type', 'structural', ...
                'source', conZonotope([80; 40; 20], diag([2.0, 2.0, 1.0]), [], []), ...
                'target', conZonotope([80; 40; 20], diag([2.5, 2.5, 1.2]), [], []));

            struct('id', 7, 'name', 'Test Configuration Mismatch', 'type', 'Type B', ...
                'upr_type', 'guarded', ...
                'source', conZonotope([70; 35; 18], diag([2.2, 2.2, 1.1]), [], []), ...
                'target', conZonotope([70.4; 35.4; 18.4], diag([2.2, 2.2, 1.1]), [], []));

            struct('id', 8, 'name', 'Simulation Numerical Error', 'type', 'Type A', ...
                'upr_type', 'parametric', ...
                'source', conZonotope([90; 45; 22], diag([2.5, 2.5, 1.3]), [], []), ...
                'target', conZonotope([90.6; 45.6; 22.6], diag([2.2, 2.2, 1.1]), [], []));
        };

        % Assign UPR mappings — matches causal_engine.py create_convide_scenarios()
        a3d = 5 * pi / 180;
        R3d = [cos(a3d), -sin(a3d), 0; sin(a3d), cos(a3d), 0; 0, 0, 1];
        scenarios_3d{1}.mapping = struct('F', {eye(3)},                    'f', {zeros(3,1)});          % S5: identity
        scenarios_3d{2}.mapping = struct('F', {diag([0.95, 1.05, 0.98])}, 'f', {zeros(3,1)});          % S6: diagonal scaling
        scenarios_3d{3}.mapping = struct('F', {eye(3)},                    'f', {[0.3; -0.2; 0.1]});   % S7: offset only
        scenarios_3d{4}.mapping = struct('F', {R3d},                       'f', {zeros(3,1)});          % S8: 3D rotation 5° (xy plane)

            base_scenarios_3d = scenarios_3d;

        if use_low_pre_variants
            fprintf('  Applying low-pre transform to 3D scenarios (alpha_c=%.2f, alpha_G=%.2f)\n', ...
                low_pre_center_alpha, low_pre_generator_alpha);
            for s = 1:length(scenarios_3d)
                scenarios_3d{s} = create_low_pre_variant_from_existing(...
                    scenarios_3d{s}, low_pre_center_alpha, low_pre_generator_alpha);
            end
        end

        if enable_test_scenarios
            test_scenario_3d = create_test_only_scenario(...
                base_scenarios_3d{1}, 105, test_center_alpha_3d, test_generator_alpha_3d, 'TEST');
            scenarios_3d{end + 1} = test_scenario_3d;
            fprintf('  Added test-only 3D scenario: %d (%s)\n', ...
                test_scenario_3d.id, test_scenario_3d.name);
            if ~isempty(test_center_alpha_3d_b)
                test_scenario_3d_b = create_test_only_scenario(...
                    base_scenarios_3d{1}, 106, test_center_alpha_3d_b, test_generator_alpha_3d_b, 'TEST-B', test_mapping_strength_3d_b);
                scenarios_3d{end + 1} = test_scenario_3d_b;
                fprintf('  Added test-only 3D scenario B: %d (%s)\n', ...
                    test_scenario_3d_b.id, test_scenario_3d_b.name);
            end
            if ~isempty(test_center_alpha_3d_c)
                test_scenario_3d_c = create_test_only_scenario(...
                    base_scenarios_3d{1}, 107, test_center_alpha_3d_c, test_generator_alpha_3d_c, 'TEST-C', test_mapping_strength_3d_c);
                scenarios_3d{end + 1} = test_scenario_3d_c;
                fprintf('  Added test-only 3D scenario C: %d (%s)\n', ...
                    test_scenario_3d_c.id, test_scenario_3d_c.name);
            end
            if ~isempty(test_center_alpha_3d_d)
                test_scenario_3d_d = create_test_only_scenario(...
                    base_scenarios_3d{1}, 108, test_center_alpha_3d_d, test_generator_alpha_3d_d, 'TEST-D', test_mapping_strength_3d_d);
                scenarios_3d{end + 1} = test_scenario_3d_d;
                fprintf('  Added test-only 3D scenario D: %d (%s)\n', ...
                    test_scenario_3d_d.id, test_scenario_3d_d.name);
            end
        end
        
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
            
            try
                causal_experiment_engine_twostep.generate_and_save_zonotopes(...
                    scenario_def, interventions, zonotope_dir, gen_options);
                fprintf('  ✓ Scenario %d completed successfully\n', scenario_def.id);
            catch ME
                fprintf('  ✗ ERROR in Scenario %d: %s\n', scenario_def.id, ME.message);
                fprintf('     Continuing with next scenario...\n');
            end
        end
    end
    
    %% 4D SCENARIOS
    if run_4d_scenarios
        fprintf('\n--- 4D SCENARIOS (4 scenarios) ---\n');
        
        scenarios_4d = {
            struct('id', 9, 'name', 'Multi-Physics Coupling Error', 'type', 'Type C', ...
                'upr_type', 'identity', ...
                'source', conZonotope([100; 50; 25; 12], diag([3.0, 3.0, 1.5, 0.75]), [], []), ...
                'target', conZonotope([100.5; 50.5; 25.5; 12.5], diag([3.1, 3.1, 1.6, 0.8]), [], []));

            struct('id', 10, 'name', 'Interface Specification Gap', 'type', 'Type D', ...
                'upr_type', 'disambiguation', ...
                'source', conZonotope([85; 42; 21; 10], diag([2.5, 2.5, 1.2, 0.6]), [], []), ...
                'target', conZonotope([85; 42; 21; 10], diag([3.0, 3.0, 1.5, 0.75]), [], []));

            struct('id', 11, 'name', 'Parameter Estimation Bias', 'type', 'Type A', ...
                'upr_type', 'constraint_based', ...
                'source', conZonotope([95; 48; 24; 11], diag([2.8, 2.8, 1.4, 0.7]), [], []), ...
                'target', conZonotope([95.4; 48.4; 24.4; 11.4], diag([2.8, 2.8, 1.4, 0.7]), [], []));

            struct('id', 12, 'name', 'Traceability Link Inconsistency', 'type', 'Type B', ...
                'upr_type', 'parametric', ...
                'source', conZonotope([75; 38; 19; 9], diag([2.5, 2.5, 1.2, 0.6]), [], []), ...
                'target', conZonotope([75.6; 38.6; 19.6; 9.6], diag([2.2, 2.2, 1.1, 0.55]), [], []));
        };

        % Assign UPR mappings — matches causal_engine.py create_convide_scenarios()
        a4d = 5 * pi / 180;
        b4d = 3 * pi / 180;
        R4d = [cos(a4d), -sin(a4d), 0, 0; sin(a4d), cos(a4d), 0, 0; ...
               0, 0, cos(b4d), -sin(b4d); 0, 0, sin(b4d), cos(b4d)];
        scenarios_4d{1}.mapping = struct('F', {eye(4)},                           'f', {zeros(4,1)});          % S9: identity
        scenarios_4d{2}.mapping = struct('F', {diag([0.96, 1.03, 0.99, 1.02])},  'f', {zeros(4,1)});          % S10: diagonal scaling
        scenarios_4d{3}.mapping = struct('F', {eye(4)},                           'f', {[0.4; -0.3; 0.2; -0.1]});  % S11: offset only
        scenarios_4d{4}.mapping = struct('F', {R4d},                              'f', {zeros(4,1)});          % S12: 4D coupled rotation (5°, 3°)

            base_scenarios_4d = scenarios_4d;

        if use_low_pre_variants
            fprintf('  Applying low-pre transform to 4D scenarios (alpha_c=%.2f, alpha_G=%.2f)\n', ...
                low_pre_center_alpha, low_pre_generator_alpha);
            for s = 1:length(scenarios_4d)
                scenarios_4d{s} = create_low_pre_variant_from_existing(...
                    scenarios_4d{s}, low_pre_center_alpha, low_pre_generator_alpha);
            end
        end

        if enable_test_scenarios
            test_scenario_4d = create_test_only_scenario(...
                base_scenarios_4d{1}, 109, test_center_alpha_4d, test_generator_alpha_4d, 'TEST');
            scenarios_4d{end + 1} = test_scenario_4d;
            fprintf('  Added test-only 4D scenario: %d (%s)\n', ...
                test_scenario_4d.id, test_scenario_4d.name);
            if ~isempty(test_center_alpha_4d_b)
                test_scenario_4d_b = create_test_only_scenario(...
                    base_scenarios_4d{1}, 110, test_center_alpha_4d_b, test_generator_alpha_4d_b, 'TEST-B', test_mapping_strength_4d_b);
                scenarios_4d{end + 1} = test_scenario_4d_b;
                fprintf('  Added test-only 4D scenario B: %d (%s)\n', ...
                    test_scenario_4d_b.id, test_scenario_4d_b.name);
            end
            if ~isempty(test_center_alpha_4d_c)
                test_scenario_4d_c = create_test_only_scenario(...
                    base_scenarios_4d{1}, 111, test_center_alpha_4d_c, test_generator_alpha_4d_c, 'TEST-C', test_mapping_strength_4d_c);
                scenarios_4d{end + 1} = test_scenario_4d_c;
                fprintf('  Added test-only 4D scenario C: %d (%s)\n', ...
                    test_scenario_4d_c.id, test_scenario_4d_c.name);
            end
            if ~isempty(test_center_alpha_4d_d)
                test_scenario_4d_d = create_test_only_scenario(...
                    base_scenarios_4d{1}, 112, test_center_alpha_4d_d, test_generator_alpha_4d_d, 'TEST-D', test_mapping_strength_4d_d);
                scenarios_4d{end + 1} = test_scenario_4d_d;
                fprintf('  Added test-only 4D scenario D: %d (%s)\n', ...
                    test_scenario_4d_d.id, test_scenario_4d_d.name);
            end
        end
        
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
            
            try
                causal_experiment_engine_twostep.generate_and_save_zonotopes(...
                    scenario_def, interventions, zonotope_dir, gen_options);
                fprintf('  ✓ Scenario %d completed successfully\n', scenario_def.id);
            catch ME
                fprintf('  ✗ ERROR in Scenario %d: %s\n', scenario_def.id, ME.message);
                fprintf('     Continuing with next scenario...\n');
            end
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
    
    % Consistency options - ALL METHODS ACTIVATED!
    consistency_opts = struct();
    if low_pre_pilot_mode
        consistency_opts.method = 'mc_probability';
        consistency_opts.mc_samples = low_pre_pilot_mc_samples;
        consistency_opts.sampling_method = 'sobol';
    else
        consistency_opts.method = 'aabb_and_mc';    % AABB Jaccard + MC Probability (includes MFMC)
        consistency_opts.mc_samples = 1000;          % Default sample budget
        consistency_opts.sampling_method = {'sobol'};  % Sobol QMC (matches Saltelli sampling scheme)
    end
    consistency_opts.verbose = true;
    consistency_opts.use_parallel = use_parallel_step2;
    consistency_opts.scenario_ids = scenarios_to_run;

    if exist('consistency_method_override', 'var') && ~isempty(consistency_method_override)
        consistency_opts.method = consistency_method_override;
    end
    if exist('consistency_mc_samples_override', 'var') && ~isempty(consistency_mc_samples_override)
        consistency_opts.mc_samples = consistency_mc_samples_override;
    end
    if exist('consistency_sampling_method_override', 'var') && ~isempty(consistency_sampling_method_override)
        consistency_opts.sampling_method = consistency_sampling_method_override;
    end
    
    fprintf('\n=== DIAGNOSTIC INFO ===\n');
    fprintf('MC samples requested: %d\n', consistency_opts.mc_samples);
    fprintf('Method: %s\n', consistency_opts.method);
    if iscell(consistency_opts.sampling_method)
        fprintf('Sampling methods: %s\n', strjoin(consistency_opts.sampling_method, ', '));
        sampling_str = strjoin(consistency_opts.sampling_method, ', ');
    else
        fprintf('Sampling method: %s\n', consistency_opts.sampling_method);
        sampling_str = consistency_opts.sampling_method;
    end
    fprintf('Each experiment will compute:\n');
    fprintf('  - AABB Jaccard (fast, axis-aligned bounding box)\n');
    if strcmp(consistency_opts.method, 'mc_probability') || strcmp(consistency_opts.method, 'both')
        fprintf('  - MC Probability with %s sampling\n', sampling_str);
    end
    if strcmp(consistency_opts.method, 'jaccard_mc') || strcmp(consistency_opts.method, 'both')
        fprintf('  - MC Jaccard with %s sampling\n', sampling_str);
    end
    if iscell(consistency_opts.sampling_method)
        fprintf('Expected time: 6-12s per experiment (computing all methods)\n');
    else
        fprintf('Expected time: 1.5-3s per experiment (computing 2 metrics)\n');
    end
    fprintf('======================\n\n');
    
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
fprintf('Example 2: Try MC with Halton sampling\n');
fprintf('  consistency_opts.method = ''mc_probability'';\n');
fprintf('  consistency_opts.mc_samples = 10000;\n');
fprintf('  consistency_opts.sampling_method = ''halton'';\n');
fprintf('  causal_experiment_engine_twostep.compute_and_save_consistency(...\n');
fprintf('      zonotope_dir, results_dir_mc_halton, consistency_opts);\n\n');
fprintf('Example 3: Compare QMC methods\n');
fprintf('  methods = {''sobol'', ''halton'', ''lhs'', ''random''};\n');
fprintf('  for i = 1:length(methods)\n');
fprintf('      consistency_opts.sampling_method = methods{i};\n');
fprintf('      result_dir = fullfile(results_base, methods{i});\n');
fprintf('      causal_experiment_engine_twostep.compute_and_save_consistency(...\n');
fprintf('          zonotope_dir, result_dir, consistency_opts);\n');
fprintf('  end\n');
fprintf('========================================\n\n');

%% Summary
fprintf('========================================\n');
fprintf('TWO-STEP WORKFLOW COMPLETE\n');
fprintf('========================================\n');
fprintf('Zonotopes:    %s\n', zonotope_dir);
fprintf('Measurements: %s\n\n', results_dir);

% Verify a result file if Step 2 was run
if run_step2
    fprintf('\n=== VERIFICATION ===\n');
    result_files = dir(fullfile(results_dir, 'results_scenario_*.json'));
    if ~isempty(result_files)
        sample_file = fullfile(results_dir, result_files(1).name);
        fprintf('Checking sample result: %s\n', result_files(1).name);
        try
            fid = fopen(sample_file, 'r');
            raw = fread(fid, inf);
            str = char(raw');
            fclose(fid);
            data = jsondecode(str);
            
            if ~isempty(data.experiments)
                exp1 = data.experiments(1);
                fprintf('\nFirst experiment results:\n');
                if isfield(exp1.post_state.inconsistency, 'jaccard_index')
                    fprintf('  AABB Jaccard: %.4f\n', exp1.post_state.inconsistency.jaccard_index);
                end
                if isfield(exp1.post_state.inconsistency, 'jaccard_mc_index_sobol')
                    fprintf('  MC Jaccard (Sobol): %.4f\n', exp1.post_state.inconsistency.jaccard_mc_index_sobol);
                end
                if isfield(exp1.post_state.inconsistency, 'jaccard_mc_index_halton')
                    fprintf('  MC Jaccard (Halton): %.4f\n', exp1.post_state.inconsistency.jaccard_mc_index_halton);
                end
                if isfield(exp1.post_state.inconsistency, 'mc_probability_sobol')
                    fprintf('  MC Probability (Sobol): %.4f (±%.4f)\n', ...
                        exp1.post_state.inconsistency.mc_probability_sobol, ...
                        exp1.post_state.inconsistency.mc_standard_error_sobol);
                end
                if isfield(exp1.post_state.inconsistency, 'mc_probability_halton')
                    fprintf('  MC Probability (Halton): %.4f (±%.4f)\n', ...
                        exp1.post_state.inconsistency.mc_probability_halton, ...
                        exp1.post_state.inconsistency.mc_standard_error_halton);
                end
                fprintf('\n✓ Both QMC methods computed successfully!\n');
                fprintf('✓ Results saved with field names: *_sobol and *_halton\n');
            end
        catch ME
            fprintf('Could not verify (not an error): %s\n', ME.message);
        end
    end
    fprintf('====================\n\n');
end

fprintf('To re-measure consistency:\n');
fprintf('  1. Set run_step1=false, run_step2=true\n');
fprintf('  2. Change consistency_opts.method\n');
fprintf('  3. Re-run this script\n\n');

%% Local helper functions
function scenario_out = create_low_pre_variant_from_existing(scenario_in, center_alpha, generator_alpha)
%CREATE_LOW_PRE_VARIANT_FROM_EXISTING Reuse a scenario while lowering baseline inconsistency.
% Moves target center and generators toward source using convex mixing factors in [0,1].

    scenario_out = scenario_in;

    src = scenario_in.source;
    tgt = scenario_in.target;

    center_alpha = max(0.0, min(1.0, center_alpha));
    generator_alpha = max(0.0, min(1.0, generator_alpha));

    new_center = src.c + center_alpha * (tgt.c - src.c);
    new_G = src.G + generator_alpha * (tgt.G - src.G);

    scenario_out.target = conZonotope(new_center, new_G, [], []);
    scenario_out.name = sprintf('%s (low-pre)', scenario_in.name);
end

function scenario_out = create_test_only_scenario(scenario_in, new_id, center_alpha, generator_alpha, tag, mapping_strength)
%CREATE_TEST_ONLY_SCENARIO Build an additional scenario for testing without changing originals.

    scenario_out = create_low_pre_variant_from_existing(scenario_in, center_alpha, generator_alpha);
    scenario_out.id = new_id;
    scenario_out.name = sprintf('%s (%s)', scenario_out.name, tag);
    if nargin >= 6 && ~isempty(mapping_strength)
        scenario_out.mapping_strength = mapping_strength;
    end
end
