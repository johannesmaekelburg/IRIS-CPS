% GENERATE SYNTHETIC V6 MEASUREMENTS — TWO-STEP WORKFLOW
%
% Reads pre-generated synthetic_v6 JSON files (from surrogate/generate_synthetic.py)
% and runs the full MATLAB consistency pipeline (AABB + MC + MFMC) on each.
%
% WHY NOT READ PRE-STATE:
%   Each JSON stores the ALREADY-PROPAGATED source geometry
%   (c_prop = F @ c_int + f,  G_prop = F @ G_int) — the affine map F is baked in
%   and is NOT saved in the JSON.  Pre-state reconstruction is therefore not
%   possible.  Instead, the neutral experiment (theta closest to sf=1, cd=0, cs=0)
%   is used as the per-scenario base zonotope, and fresh Saltelli interventions
%   are applied around it.  UPR is set to identity because F is already included
%   in the stored geometry.
%
%   Output: data/measurements_synthetic_v6/results_scenario_*.json
%   These are NEW MATLAB-quality measurements (Saltelli theta distribution,
%   MFMC-scored) — not an in-place update of the Python synthetic_v6 files.
%
% Supports 2D / 3D / 4D scenarios.
%
% Version: 1.0 — May 2026

%% Setup
clear; close all; clc;

warning('off', 'CORA:deprecated');
warning('off', 'CORA:contSet:set:expMat');
warning('off', 'CORA:contSet:get:expMat');
warning('off', 'all');
lastwarn('');

%% ═══════════════════════════════════════════════════════════════
%%   CONFIGURATION
%% ═══════════════════════════════════════════════════════════════
run_step1 = true;    % Set false to re-use existing zonotope files
run_step2 = true;    % Set false to skip consistency scoring

use_saltelli_mode = true;
seed = 2025;
rng(seed);

consistency_method = 'aabb_and_mc';   % AABB Jaccard + MC (+ MFMC)
mc_samples         = 1000;
sampling_methods   = {'sobol'};
use_parallel       = true;
n_workers          = 20;

%% ═══════════════════════════════════════════════════════════════
%%   PATHS
%% ═══════════════════════════════════════════════════════════════
script_dir   = fileparts(mfilename('fullpath'));
project_root = fileparts(script_dir);
myCORA_dir   = fileparts(project_root);
cps_fw_path  = fullfile(myCORA_dir, 'CPS-Uncertainty-Propagation-Framework');

addpath(fullfile(project_root, 'src', 'matlab'));
addpath(genpath(fullfile(cps_fw_path, 'src')));

addon_path = fullfile(cps_fw_path, 'addons', 'consistency_scoring');
if ~exist(fullfile(addon_path, 'init_consistency_scoring.m'), 'file')
    error('Consistency scoring addon not found at: %s', addon_path);
end
addpath(addon_path);
addpath(fullfile(addon_path, 'methods'));

%% I/O directories
data_base     = fullfile(project_root, 'data');
synthetic_dir = fullfile(data_base, 'synthetic_v6');
zonotope_dir  = fullfile(data_base, 'zonotopes_synthetic_v6');
results_dir   = fullfile(data_base, 'measurements_synthetic_v6');

if ~exist(data_base,    'dir'), mkdir(data_base);    end
if ~exist(zonotope_dir, 'dir'), mkdir(zonotope_dir); end
if ~exist(results_dir,  'dir'), mkdir(results_dir);  end

fprintf('===========================================\n');
fprintf('SYNTHETIC V6  TWO-STEP ANALYSIS\n');
fprintf('===========================================\n');
fprintf('Input:      %s\n', synthetic_dir);
fprintf('Zonotopes:  %s\n', zonotope_dir);
fprintf('Results:    %s\n\n', results_dir);

%% ═══════════════════════════════════════════════════════════════
%%   SALTELLI SAMPLES
%% ═══════════════════════════════════════════════════════════════
if use_saltelli_mode
    saltelli_params = {'scale_factor', 'center_delta', 'correlation_strength'};

    % Prefer versioned file; fall back to generic
    saltelli_file = fullfile(data_base, 'saltelli_samples_3param_v6.csv');
    if ~exist(saltelli_file, 'file')
        saltelli_file = fullfile(data_base, 'saltelli_samples_3param.csv');
    end
    if ~exist(saltelli_file, 'file')
        error('Saltelli CSV not found.\nExpected: %s\nRun generate_saltelli_samples.py first.', ...
              saltelli_file);
    end

    fprintf('Saltelli samples: %s\n', saltelli_file);
    saltelli_data = readtable(saltelli_file);
    interventions = struct('mode',    'saltelli',      ...
                           'samples', saltelli_data,   ...
                           'params',  {saltelli_params});
    fprintf('  %d compound interventions loaded.\n\n', height(saltelli_data));
end

%% ═══════════════════════════════════════════════════════════════
%%   PARALLEL POOL
%% ═══════════════════════════════════════════════════════════════
if use_parallel
    delete(gcp('nocreate'));
    if isempty(gcp('nocreate'))
        parpool('local', n_workers);
    end
end

%% ═══════════════════════════════════════════════════════════════
%%   LOAD SCENARIOS FROM SYNTHETIC_V6 JSON FILES
%% ═══════════════════════════════════════════════════════════════
all_scenarios = load_synthetic_v6_scenarios(synthetic_dir);
fprintf('Loaded %d synthetic scenarios.\n\n', length(all_scenarios));

%% ═══════════════════════════════════════════════════════════════
%%   STEP 1 — GENERATE ZONOTOPES
%% ═══════════════════════════════════════════════════════════════
if run_step1
    fprintf('\n===========================================\n');
    fprintf('STEP 1: GENERATING ZONOTOPES  (%d scenarios)\n', length(all_scenarios));
    fprintf('===========================================\n');

    gen_options = struct('n_repeats', 1, 'verbose', true, 'use_parallel', use_parallel);

    for k = 1:length(all_scenarios)
        sc = all_scenarios{k};
        fprintf('\n  [%3d/%3d] Scenario %d: %s\n', ...
                k, length(all_scenarios), sc.id, sc.name);
        try
            causal_experiment_engine_twostep.generate_and_save_zonotopes( ...
                sc, interventions, zonotope_dir, gen_options);
            fprintf('          + done\n');
        catch ME
            fprintf('          x ERROR: %s\n', ME.message);
        end
    end

    fprintf('\n===========================================\n');
    fprintf('STEP 1 COMPLETE → %s\n', zonotope_dir);
    fprintf('===========================================\n');
end

%% ═══════════════════════════════════════════════════════════════
%%   STEP 2 — COMPUTE CONSISTENCY SCORES
%% ═══════════════════════════════════════════════════════════════
if run_step2
    fprintf('\n===========================================\n');
    fprintf('STEP 2: COMPUTING CONSISTENCY SCORES\n');
    fprintf('===========================================\n');

    consistency_opts = struct();
    consistency_opts.method          = consistency_method;
    consistency_opts.mc_samples      = mc_samples;
    consistency_opts.sampling_method = sampling_methods;
    consistency_opts.verbose         = false;
    consistency_opts.use_parallel    = use_parallel;

    causal_experiment_engine_twostep.compute_and_save_consistency( ...
        zonotope_dir, results_dir, consistency_opts);

    fprintf('\n===========================================\n');
    fprintf('STEP 2 COMPLETE → %s\n', results_dir);
    fprintf('===========================================\n');
end

fprintf('\n===========================================\n');
fprintf('SYNTHETIC V6  TWO-STEP WORKFLOW COMPLETE\n');
fprintf('===========================================\n');
fprintf('Zonotopes:    %s\n', zonotope_dir);
fprintf('Measurements: %s\n', results_dir);
fprintf('\nNext steps:\n');
fprintf('  python -m surrogate.run_analysis --data data/measurements_synthetic_v6\n');
fprintf('  python -m surrogate.run_analysis --data data/measurements_synthetic_v6 data/measurements\n');
fprintf('===========================================\n\n');


%% ═══════════════════════════════════════════════════════════════
%%   LOCAL HELPER — load scenarios
%% ═══════════════════════════════════════════════════════════════
function scenarios = load_synthetic_v6_scenarios(synthetic_dir)
%LOAD_SYNTHETIC_V6_SCENARIOS  Build scenario structs from synthetic_v6 JSON files.
%
% Design notes
% ─────────────
% The Python generator stores PROPAGATED geometry in each experiment:
%   post_state.uncertainty.source_center    = F @ c_int + f   (after mapping)
%   post_state.uncertainty.source_generators= F @ G_int        (after mapping)
%   post_state.uncertainty.target_center    = c_tgt            (fixed)
%   post_state.uncertainty.target_generators= G_tgt            (fixed)
%
% The affine map F is NOT stored in the JSON and differs per scenario
% (f_type ∈ {identity, near_identity, random}).  Because F is already
% embedded in the stored geometry we:
%   1. Set UPR = identity (no further transformation needed).
%   2. Use the "neutral" experiment (theta closest to sf=1, cd=0, cs=0)
%      as the per-scenario base zonotope for Step 1 Saltelli sweeps.
%
% Supports d = 2, 3, 4.

    json_files = dir(fullfile(synthetic_dir, 'results_scenario_*.json'));
    if isempty(json_files)
        error('No results_scenario_*.json files found in:\n  %s', synthetic_dir);
    end

    % Sort by numeric scenario ID
    raw_names = {json_files.name};
    ids = zeros(1, numel(raw_names));
    for n = 1:numel(raw_names)
        tok = regexp(raw_names{n}, 'results_scenario_(\d+)\.json', 'tokens');
        if ~isempty(tok), ids(n) = str2double(tok{1}{1}); end
    end
    [~, order] = sort(ids);
    json_files = json_files(order);

    scenarios = {};
    n_ok = 0;  n_err = 0;

    for k = 1:numel(json_files)
        json_path = fullfile(synthetic_dir, json_files(k).name);

        % ── Decode ────────────────────────────────────────────────────────
        try
            raw = jsondecode(fileread(json_path));
        catch ME
            fprintf('  [warn] Cannot parse %s: %s\n', json_files(k).name, ME.message);
            n_err = n_err + 1;  continue;
        end

        if ~isfield(raw, 'experiments') || isempty(raw.experiments)
            fprintf('  [warn] Empty experiments in %s\n', json_files(k).name);
            n_err = n_err + 1;  continue;
        end

        exps  = raw.experiments;   % struct array (one element per experiment)
        n_exp = numel(exps);

        % ── Find neutral experiment ────────────────────────────────────────
        %   The post-state of this experiment ≈ base geometry for Saltelli.
        dist = zeros(n_exp, 1);
        for ei = 1:n_exp
            e = exps(ei);
            dist(ei) = (e.scale_factor - 1.0)^2 + ...
                        e.center_delta^2           + ...
                        e.correlation_strength^2;
        end
        [~, ref] = min(dist);
        unc = exps(ref).post_state.uncertainty;

        % ── Geometry ──────────────────────────────────────────────────────
        %   source_generators in JSON: d rows × p columns (row = dimension).
        %   jsondecode returns (d×p) matrix directly — matches CORA convention.
        c_src = unc.source_center(:);     % (d,1)
        G_src = unc.source_generators;    % (d,p)
        c_tgt = unc.target_center(:);     % (d,1)
        G_tgt = unc.target_generators;    % (d,p_t)

        % Guard: jsondecode may return (p×1) vector for single-generator cases
        if isvector(G_src), G_src = G_src(:); end
        if isvector(G_tgt), G_tgt = G_tgt(:); end

        d = length(c_src);
        if d < 2 || size(G_src,1) ~= d || size(G_tgt,1) ~= d
            fprintf('  [warn] Dimension mismatch in %s (d=%d)\n', ...
                    json_files(k).name, d);
            n_err = n_err + 1;  continue;
        end

        % ── Build CORA zonotopes ───────────────────────────────────────────
        try
            src = conZonotope(c_src, G_src, [], []);
            tgt = conZonotope(c_tgt, G_tgt, [], []);
        catch ME
            fprintf('  [warn] conZonotope failed for %s: %s\n', ...
                    json_files(k).name, ME.message);
            n_err = n_err + 1;  continue;
        end

        % ── Scenario ID from filename ──────────────────────────────────────
        tok = regexp(json_files(k).name, 'results_scenario_(\d+)', 'tokens');
        sc_id = k;
        if ~isempty(tok), sc_id = str2double(tok{1}{1}); end

        % ── Metadata ──────────────────────────────────────────────────────
        dim_str = num2str(d);
        gen_str = 'unknown';   if isfield(raw,'gen_type'),       gen_str = raw.gen_type;       end
        ov_str  = 'unknown';   if isfield(raw,'overlap_regime'), ov_str  = raw.overlap_regime; end
        ft_str  = 'unknown';   if isfield(raw,'f_type'),         ft_str  = raw.f_type;         end

        sc_name = sprintf('Synthetic%sD_%s_%s_%s_id%d', ...
                          dim_str, gen_str, ov_str, ft_str, sc_id);

        % ── Per-dimension identity UPR (F already in stored geometry) ──────
        rel_cell = cell(1, d);
        for di = 1:d
            rel_cell{di} = struct('upr_type', 'identity', ...
                                  'mapping',  struct('scale', 1.0, 'offset', 0.0));
        end

        % ── Scenario struct ────────────────────────────────────────────────
        scenario = struct();
        scenario.id                    = sc_id;
        scenario.name                  = sc_name;
        scenario.type                  = 'identity';
        scenario.source                = src;
        scenario.target                = tgt;
        scenario.mapping               = struct('F', eye(d), 'f', zeros(d,1));
        scenario.upr_type              = 'identity';
        scenario.upr_params            = struct();
        scenario.dataset_source        = json_files(k).name;
        scenario.paired_scenario_ids   = arrayfun(@(i) sprintf('dim%d',i-1), 1:d, ...
                                                  'UniformOutput', false);
        scenario.paired_scenario_names = arrayfun(@(i) sprintf('dim%d',i-1), 1:d, ...
                                                  'UniformOutput', false);
        scenario.consistency_relations = rel_cell;
        scenario.relation_types        = repmat({'identity'}, 1, d);
        scenario.relation_operators    = repmat({'leq'},      1, d);

        scenarios{end+1} = scenario; %#ok<AGROW>
        n_ok = n_ok + 1;
    end

    fprintf('  Loaded %d scenarios  (%d skipped)\n', n_ok, n_err);
end
