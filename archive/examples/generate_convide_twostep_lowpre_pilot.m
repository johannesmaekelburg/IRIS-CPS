% GENERATE CONVIDE LOW-PRE PILOT (WRAPPER)
%
% Wrapper for generate_convide_twostep.m that runs only a quick low-pre
% validation set. This avoids editing the main script each time.

%% Setup wrapper options
clearvars; close all; clc;

% Core workflow switches
run_step1 = true;
run_step2 = true;

% Keep CONVIDE mode for pilot
use_saltelli_mode = false;

% No parallel pool needed
use_parallel = false;

% Pilot mode (fast run)
low_pre_pilot_mode = true;
low_pre_pilot_scenarios = [1, 5, 9];   % One per dimension
low_pre_pilot_mc_samples = 300;

% Low-pre transform strength
use_low_pre_variants = true;
low_pre_center_alpha = 0.25;
low_pre_generator_alpha = 0.35;

% Optional: narrow scenario list further
% scenarios_to_run = [1];

fprintf('Launching low-pre pilot wrapper...\n');
fprintf('  scenarios: %s\n', mat2str(low_pre_pilot_scenarios));
fprintf('  mc_samples: %d\n\n', low_pre_pilot_mc_samples);

% Keep wrapper-provided variables when running the main script
preserve_external_config = true;

%% Run main workflow script from same folder
this_dir = fileparts(mfilename('fullpath'));
run(fullfile(this_dir, 'generate_convide_twostep.m'));
