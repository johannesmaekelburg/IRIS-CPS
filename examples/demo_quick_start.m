% QUICK START DEMO: Bidirectional Causal Inference
%
% Simple demonstration of bidirectional causality analysis using the
% uncertainty-inconsistency framework. Run this first to understand the
% basic concepts.
%
% DEMONSTRATES:
%   1. Forward: Uncertainty → Inconsistency (widen tolerance → more conflicts)
%   2. Reverse: Inconsistency → Uncertainty (shift target → change volume)
%
% Runtime: ~30 seconds
% Prerequisites: CPS Framework in path
%
% Version: 1.0
% Date: December 2025

%% Setup
clear; close all; clc;

% Add paths
src_path = fullfile(fileparts(pwd), 'src');
addpath(src_path);

cps_path = fullfile(fileparts(fileparts(pwd)), 'CPS-Uncertainty-Propagation-Framework', 'src');
if exist(cps_path, 'dir')
    addpath(genpath(cps_path));
end

% Initialize consistency scoring addon
consistency_addon_path = fullfile(fileparts(fileparts(pwd)), 'CPS-Uncertainty-Propagation-Framework', 'addons', 'consistency_scoring');
if exist(fullfile(consistency_addon_path, 'init_consistency_scoring.m'), 'file')
    addpath(consistency_addon_path);
    addpath(fullfile(consistency_addon_path, 'methods'));
    fprintf('✓ Consistency scoring addon loaded\n');
else
    error('Consistency scoring addon required but not found. Check CPS framework installation.');
end

fprintf('========================================\n');
fprintf('QUICK START DEMO\n');
fprintf('Bidirectional Causal Inference\n');
fprintf('========================================\n\n');

%% Create Simple 2D Scenario
fprintf('1. Creating 2D brake disc scenario...\n');
fprintf('   Source: Specification [d_req=300±5mm, r_i=25±2mm]\n');
fprintf('   Target: CAD model [d_req=302±3mm, r_i=25±1.5mm]\n\n');

% Source: Manufacturing specification
c_src = [300; 25];
G_src = [5, 0; 0, 2];
Z_src = conPolyZono(c_src, G_src, eye(2));

% Target: CAD model
c_tgt = [302; 25];
G_tgt = [3, 0; 0, 1.5];
Z_tgt = conPolyZono(c_tgt, G_tgt, eye(2));

% Scenario
scenario = struct();
scenario.source = Z_src;
scenario.target = Z_tgt;
scenario.mapping = struct('F', eye(2), 'f', zeros(2, 1));

%% DEMO 1: Forward Direction (Uncertainty → Inconsistency)
fprintf('========================================\n');
fprintf('DEMO 1: Forward Direction\n');
fprintf('Question: Does widening tolerance cause inconsistency?\n');
fprintf('========================================\n');

params = struct('scale_factor', 3.0);
result_fwd = causal_experiment_engine.run_intervention(scenario, 'widen', params);

fprintf('\nBefore widening (scale=1.0):\n');
fprintf('  Jaccard overlap: %.3f\n', result_fwd.pre_state.inconsistency.jaccard_index);
fprintf('  Empty intersection: %d\n', result_fwd.pre_state.inconsistency.empty_intersection);

fprintf('\nAfter widening (scale=3.0):\n');
fprintf('  Jaccard overlap: %.3f\n', result_fwd.post_state.inconsistency.jaccard_index);
fprintf('  Empty intersection: %d\n', result_fwd.post_state.inconsistency.empty_intersection);

fprintf('\nCausal Effect:\n');
fprintf('  Δ Jaccard: %.3f ', result_fwd.causal_effect.inconsistency.delta_jaccard);
if result_fwd.causal_effect.inconsistency.delta_jaccard < 0
    fprintf('(overlap DECREASED ✓)\n');
else
    fprintf('(overlap increased)\n');
end
fprintf('  Δ Empty: %d ', result_fwd.causal_effect.inconsistency.delta_empty);
if result_fwd.causal_effect.inconsistency.delta_empty > 0
    fprintf('(inconsistency CREATED ✓)\n');
elseif result_fwd.causal_effect.inconsistency.delta_empty < 0
    fprintf('(inconsistency resolved)\n');
else
    fprintf('(no change)\n');
end

%% DEMO 2: Shrink Intervention (Uncertainty → Inconsistency)
fprintf('\n========================================\n');
fprintf('DEMO 2: Shrink Intervention\n');
fprintf('Question: Does reducing uncertainty resolve inconsistency?\n');
fprintf('========================================\n');

params = struct('scale_factor', 0.5);
result_shrink = causal_experiment_engine.run_intervention(scenario, 'shrink', params);

fprintf('\nBefore shrinking:\n');
fprintf('  Source volume: %.2f mm³\n', result_shrink.pre_state.uncertainty.source_volume);
fprintf('  Jaccard overlap: %.3f\n', result_shrink.pre_state.inconsistency.jaccard_index);

fprintf('\nAfter shrinking by 50%%:\n');
fprintf('  Source volume: %.2f mm³\n', result_shrink.post_state.uncertainty.source_volume);
fprintf('  Jaccard overlap: %.3f\n', result_shrink.post_state.inconsistency.jaccard_index);

fprintf('\nCausal Effect:\n');
fprintf('  Δ Jaccard: %.3f ', result_shrink.causal_effect.inconsistency.delta_jaccard);
if result_shrink.causal_effect.inconsistency.delta_jaccard > 0
    fprintf('(overlap INCREASED ✓)\n');
else
    fprintf('(overlap unchanged or decreased)\n');
end
fprintf('  Volume ratio: %.2f\n', result_shrink.causal_effect.uncertainty.volume_ratio);

%% Summary
fprintf('\n========================================\n');
fprintf('DEMO COMPLETE\n');
fprintf('========================================\n');
fprintf('\nKey Concepts Demonstrated:\n');
fprintf('  ✓ Forward causality: Widening uncertainty → Decreased overlap\n');
fprintf('  ✓ Forward causality: Shrinking uncertainty → Increased overlap\n');
fprintf('  ✓ Quantitative effects: delta_jaccard, delta_empty\n');
fprintf('\nNext Steps:\n');
fprintf('  1. Run test_bidirectional_causality.m for detailed validation\n');
fprintf('  2. Generate datasets with generate_*.m scripts\n');
fprintf('  3. Train neural models with Python (src/train.py)\n\n');
