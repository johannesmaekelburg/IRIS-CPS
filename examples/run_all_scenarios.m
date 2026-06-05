% RUN ALL SCENARIOS — Engineering + CPS DOMAINS
%
% Runs both generate_engineering_twostep.m and generate_cps_domains_twostep.m
% sequentially in a single MATLAB session, sharing the parallel pool.
%
% Usage:
%   run_all_scenarios          % run everything (Step 1 + Step 2)
%   run_step1 = false; run_all_scenarios  % skip zonotope generation, redo scoring only
%
% Version: 1.0

clear; close all; clc;
warning('off', 'all');
lastwarn('');

t_total = tic;
fprintf('==========================================\n');
fprintf('  RUN ALL SCENARIOS (Engineering + CPS)\n');
fprintf('==========================================\n\n');

%% ─── Shared configuration ────────────────────────────────────────────────────
% These are picked up by both sub-scripts via exist() guards.

preserve_external_config = true;   % prevents sub-scripts from clearing workspace

% Step control (override before calling this script if needed)
if ~exist('run_step1', 'var'), run_step1 = true; end
if ~exist('run_step2', 'var'), run_step2 = true; end

% Parallel pool — start once, both scripts reuse it
use_parallel = true;
n_workers    = 20;

% Clean up any stale parallel jobs before starting
try
    c = parcluster('local');
    jobs = c.Jobs;
    for k = 1:numel(jobs)
        delete(jobs(k));
    end
catch
end
delete(gcp('nocreate'));

pool = gcp('nocreate');
if isempty(pool)
    fprintf('Starting parallel pool with %d workers...\n', n_workers);
    parpool('local', n_workers);
    fprintf('  Parallel pool ready\n\n');
else
    fprintf('Parallel pool already running with %d workers\n\n', pool.NumWorkers);
end

%% ─── Part 1: Engineering scenarios ───────────────────────────────────────────────
fprintf('==========================================\n');
fprintf('  PART 1 / 2 — Engineering\n');
fprintf('==========================================\n');
t1 = tic;

run(fullfile(fileparts(mfilename('fullpath')), 'generate_engineering_twostep.m'));

fprintf('\n  Engineering done in %.1f min\n\n', toc(t1)/60);

%% ─── Part 2: CPS domain scenarios ───────────────────────────────────────────
fprintf('==========================================\n');
fprintf('  PART 2 / 2 — CPS DOMAINS\n');
fprintf('==========================================\n');
t2 = tic;

% Reset preserve flag so CPS script does not clear workspace but can still
% set its own configuration variables cleanly.
preserve_external_config = true;

run(fullfile(fileparts(mfilename('fullpath')), 'generate_cps_domains_twostep.m'));

fprintf('\n  CPS domains done in %.1f min\n\n', toc(t2)/60);

%% ─── Summary ─────────────────────────────────────────────────────────────────
fprintf('==========================================\n');
fprintf('  ALL DONE — total %.1f min\n', toc(t_total)/60);
fprintf('==========================================\n');
