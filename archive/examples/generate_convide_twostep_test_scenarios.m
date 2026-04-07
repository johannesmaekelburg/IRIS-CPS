% GENERATE CONVIDE TEST SCENARIOS (WRAPPER)
%
% Runs dedicated test-only scenarios with adjusted initial source-target
% proximity/shape. Original scenarios (1-12) remain unchanged.

%% Setup wrapper options
clearvars; close all; clc;

% Core workflow switches
run_step1 = true;
run_step2 = true;

% Keep CONVIDE mode
use_saltelli_mode = false;

% No Parallel Computing Toolbox required
use_parallel = false;

% Disable pilot overrides so this wrapper controls scenario IDs directly
low_pre_pilot_mode = false;

% Keep original scenarios untouched in this run
use_low_pre_variants = false;

% Enable dedicated test-only scenarios
enable_test_scenarios = true;
scenarios_to_run = [102, 103, 104, 106, 107, 108, 110, 111, 112];

% Dimension-aware proximity/shape adjustment for test scenarios
% (A variants: high pre-intervention inconsistency, I_theta ~ 0.55-0.99)
% IDs: 2D->101, 3D->105, 4D->109
test_center_alpha_2d = 0.25;
test_center_alpha_3d = 0.10;
test_center_alpha_4d = 0.05;
test_generator_alpha_2d = 0.35;
test_generator_alpha_3d = 0.20;
test_generator_alpha_4d = 0.12;

% B variants: very low pre-intervention inconsistency, targeting I_theta ~ 0.03-0.15
% IDs: 2D->102, 3D->106, 4D->110
% Empirically validated: 2D->0.029, 3D->0.087, 4D->0.154
test_center_alpha_2d_b = 0.10;
test_center_alpha_3d_b = 0.02;
test_center_alpha_4d_b = 0.01;
test_generator_alpha_2d_b = 0.15;
test_generator_alpha_3d_b = 0.05;
test_generator_alpha_4d_b = 0.02;
test_mapping_strength_2d_b = 0.9;
test_mapping_strength_3d_b = 1.8;
test_mapping_strength_4d_b = 2.6;

% C variants: mid-low pre-intervention inconsistency, targeting I_theta ~ 0.20-0.35
% IDs: 2D->103, 3D->107, 4D->111
test_center_alpha_2d_c = 0.15;
test_center_alpha_3d_c = 0.04;
test_center_alpha_4d_c = 0.02;
test_generator_alpha_2d_c = 0.20;
test_generator_alpha_3d_c = 0.08;
test_generator_alpha_4d_c = 0.04;
test_mapping_strength_2d_c = 0.9;
test_mapping_strength_3d_c = 1.8;
test_mapping_strength_4d_c = 2.6;

% D variants: mid pre-intervention inconsistency, targeting I_theta ~ 0.40-0.60
% IDs: 2D->104, 3D->108, 4D->112
test_center_alpha_2d_d = 0.20;
test_center_alpha_3d_d = 0.055;
test_center_alpha_4d_d = 0.03;
test_generator_alpha_2d_d = 0.28;
test_generator_alpha_3d_d = 0.12;
test_generator_alpha_4d_d = 0.07;
test_mapping_strength_2d_d = 0.9;
test_mapping_strength_3d_d = 1.8;
test_mapping_strength_4d_d = 2.6;

% Use same output folders as regular scenarios
this_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(this_dir);

% 1000 MC samples — same as regular scenario runs
consistency_mc_samples_override = 1000;

% Keep wrapper-provided variables when running the main script
preserve_external_config = true;

fprintf('Launching test-scenarios wrapper...\n');
fprintf('  scenarios:  %s\n', mat2str(scenarios_to_run));
fprintf('  mc_samples: %d\n', consistency_mc_samples_override);
fprintf('  output:     data/zonotopes + data/measurements (shared with regular runs)\n\n');

%% Run main workflow script from same folder
run(fullfile(this_dir, 'generate_convide_twostep.m'));
