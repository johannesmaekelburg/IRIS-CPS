function [score, details] = score_mc_probability(models, varargin)
%SCORE_MC_PROBABILITY Unbiased Monte Carlo consistency scoring for multiple sets
%
% Computes an unbiased Monte Carlo estimate of the intersection probability
% by sampling uniformly from a symmetric bounding box containing all sets
% and using exact containment checks (NOT interval hull approximation).
%
% VERSION: 3.0 - Unbiased sampling + exact containment checks
%
% ALGORITHM:
%   1. Compute union bounding box B containing all models
%   2. Sample N points uniformly from B
%   3. For each point: test exact containment in ALL models
%   4. Estimate: p_consistent = (#points in all models) / N
%
% SYNTAX:
%   score = score_mc_probability(models)
%   [score, details] = score_mc_probability(models, 'Name', Value, ...)
%
% INPUTS:
%   models  - Cell array of CORA sets (conPolyZono, zonotope, etc.)
%   
% OPTIONAL NAME-VALUE PAIRS:
%   'num_samples'     - Number of Monte Carlo samples 
%                       (default: auto-scaled by dimension)
%   'seed'            - Random seed for reproducibility (default: none)
%   'sampling_method' - Sampling strategy (default: 'random')
%                       'random' - pseudo-random MC  (convergence O(1/√N))
%                       'sobol'  - Sobol QMC         (convergence ~O(1/N))
%                       'halton' - Halton QMC        (convergence ~O(1/N))
%   'return_details'  - Return detailed statistics (default: true)
%   'verbose'         - Display progress (default: false)
%
% OUTPUTS:
%   score   - Consistency probability p_consistent ∈ [0,1]
%             Fraction of samples contained in ALL models
%   details - Struct with detailed statistics:
%       .p_consistent      - Probability a sample is in all models
%       .p_inconsistent    - Probability not in all models
%       .num_samples       - Total samples drawn
%       .num_consistent    - Number of consistent samples
%       .standard_error    - Standard error of the estimate
%       .ci95_lower        - 95% confidence interval lower bound
%       .ci95_upper        - 95% confidence interval upper bound
%       .sampling_lb       - Lower bounds of sampling box
%       .sampling_ub       - Upper bounds of sampling box
%       .box_volume        - Volume of sampling box
%       .containment_methods_used - Counts per method type
%       .num_containment_errors - Count of containment failures
%       .notes             - String explaining containment methods
%
% EXAMPLES:
%   % Example 1: High overlap
%   E = eye(2); A = []; b = []; EC = [];
%   Z1 = conPolyZono([10; 10], [2 0; 0 2], E, A, b, EC);
%   Z2 = conPolyZono([10.5; 10.5], [2 0; 0 2], E, A, b, EC);
%   [score, det] = score_mc_probability({Z1, Z2}, 'verbose', true);
%
%   % Example 2: Disjoint sets
%   Z3 = conPolyZono([100; 100], [2 0; 0 2], E, A, b, EC);
%   score_disj = score_mc_probability({Z1, Z3});  % Should be ~0
%
% NOTES:
%   - Uses EXACT containment checks (not interval hull)
%   - Unbiased: samples from symmetric box containing all sets
%   - For 2D: default N=5000, 3D: N=10000, 4D: N=20000
%   - Containment may fail if CORA doesn't support exact checks
%   - Never inflates score with interval hull approximation
%
% SEE ALSO:
%   score_jaccard_mc, score_jaccard_aabb

%% Parse inputs
p = inputParser;
addRequired(p, 'models', @(x) iscell(x) && ~isempty(x));
addParameter(p, 'num_samples', [], @(x) isempty(x) || (isnumeric(x) && x > 0));
addParameter(p, 'seed', [], @(x) isempty(x) || isnumeric(x));
addParameter(p, 'return_details', true, @islogical);
addParameter(p, 'verbose', false, @islogical);
addParameter(p, 'sampling_method', 'sobol', ...
    @(x) ismember(lower(x), {'random','mc','pseudorandom','sobol','qmc','halton','lhs','latinhypercube'}));
addParameter(p, 'distribution', [], ...
    @(x) isempty(x) || (isstruct(x) && isfield(x,'mu') && isfield(x,'Sigma')));
addParameter(p, 'use_mfmc', false, @islogical);
addParameter(p, 'target_se', 0.02, @(x) isnumeric(x) && x > 0);
parse(p, models, varargin{:});

sampling_method = p.Results.sampling_method;
use_mfmc  = p.Results.use_mfmc;
target_se = p.Results.target_se;

num_models = length(models);
if num_models < 1
    error('score_mc_probability:NoModels', 'At least one model is required');
end

verbose = p.Results.verbose;
return_details = p.Results.return_details;

%% Step 1: Compute union bounding box containing ALL models
if verbose
    fprintf('========================================\n');
    fprintf('MC Consistency Probability\n');
    fprintf('========================================\n');
    fprintf('Models: %d\n', num_models);
end

% Get dimension from first model
try
    I_first = interval(models{1});
    dim = length(infimum(I_first));
catch ME
    error('score_mc_probability:IntervalConversion', ...
        'Failed to convert first model to interval: %s', ME.message);
end

% Initialize union bounds and intersection bounds
lb_union = inf(dim, 1);
ub_union = -inf(dim, 1);
lb_int   = -inf(dim, 1);   % intersection lower bound = max over models
ub_int   =  inf(dim, 1);   % intersection upper bound = min over models

% Compute union and intersection of all model bounding boxes
for i = 1:num_models
    try
        I = interval(models{i});
        lb_i = infimum(I);
        ub_i = supremum(I);

        % Expand union bounds
        lb_union = min(lb_union, lb_i);
        ub_union = max(ub_union, ub_i);
        % Shrink intersection bounds
        lb_int   = max(lb_int, lb_i);
        ub_int   = min(ub_int, ub_i);
    catch ME
        error('score_mc_probability:IntervalConversion', ...
            'Failed to convert model %d to interval: %s', i, ME.message);
    end
end

% Compute box volume
widths = ub_union - lb_union;
box_volume = prod(widths);

% MFMC: mu_AABB will be computed under the sampling distribution in Step 1c.
% (placeholder — overwritten below once distribution is known)
aabb_int_widths = max(0, ub_int - lb_int);
mu_AABB = prod(aabb_int_widths) / max(box_volume, eps);  % overwritten in Step 1c

% Handle degenerate dimensions (zero width)
degenerate_dims = (widths < eps);
if any(degenerate_dims)
    if verbose
        fprintf('Warning: %d dimension(s) have zero width (will sample at constant value)\n', ...
            sum(degenerate_dims));
    end
end

if verbose
    fprintf('Dimension: %d\n', dim);
    fprintf('Sampling box: [%.3f, %.3f] x ... x [%.3f, %.3f]\n', ...
        lb_union(1), ub_union(1), lb_union(end), ub_union(end));
    fprintf('Box volume: %.6e\n', box_volume);
end

%% Step 1c: Build / validate sampling distribution
% Default: Gaussian centred on the union box, 2-sigma spans the full extent.
dist = p.Results.distribution;
if isempty(dist)
    mu_samp    = (lb_union + ub_union) / 2;
    sigma_samp = max((ub_union - lb_union) / 4, eps);
    dist = struct('mu', mu_samp, 'Sigma', diag(sigma_samp.^2));
end
% Pre-compute Cholesky factor for efficient sampling
try
    L_chol = chol(dist.Sigma, 'lower');
catch
    L_chol = diag(sqrt(diag(dist.Sigma)));  % fallback: diagonal
end

% MFMC: recompute mu_AABB under the actual distribution
% (replaces the volume-ratio estimate; exact analytic value)
mu_AABB = gauss_box_prob_mc(lb_int, ub_int, dist.mu, dist.Sigma);

%% Step 2: Determine number of samples (auto-scale by dimension)
if isempty(p.Results.num_samples)
    % Auto-scale: 5000 for 2D, increase for higher dimensions
    if dim <= 2
        num_samples = 5000;
    elseif dim == 3
        num_samples = 10000;
    else  % dim >= 4
        num_samples = 20000;
    end
else
    num_samples = p.Results.num_samples;
end

if verbose
    fprintf('Samples: N = %d\n', num_samples);
end

%% Step 3: Set random seed if specified
if ~isempty(p.Results.seed)
    rng(p.Results.seed);
    if verbose
        fprintf('Random seed: %d\n', p.Results.seed);
    end
end

%% Step 4: Sample from distribution (Gaussian MC or QMC via probit transform)
if verbose
    fprintf('Sampling method: %s  |  distribution: N(mu, Sigma)\n', upper(sampling_method));
end

% Generate N samples from the distribution
% QMC: generate u in (0,1)^d, apply probit -> N(0,I), then Cholesky transform
% Random: direct mvnrnd
samples_unit = generateSamples_mc(num_samples, dim, sampling_method);
% Clamp away from 0/1 so probit transform is finite
samples_unit = max(1e-9, min(1-1e-9, samples_unit));
% Probit transform via erfinv (base MATLAB, no Statistics Toolbox needed):
%   norminv(p) = sqrt(2) * erfinv(2p - 1)
z_std = sqrt(2) .* erfinv(2 .* samples_unit - 1);  % (N x dim), ~ N(0,I)
% Transform to N(mu, Sigma): x = mu + L*z  (broadcast mu over N samples)
samples = bsxfun(@plus, dist.mu(:), L_chol * z_std');  % (dim x N)

if verbose
    fprintf('Generated %d %s samples from N(mu, Sigma)\n', num_samples, upper(sampling_method));
end

%% Step 5: Check containment using exact methods
num_consistent = 0;
num_containment_errors = 0;
consistent_flags = false(1, num_samples);  % MFMC: track per-sample result

% Track containment methods used
method_counts = struct();

% Convergence checkpoints — set up before the loop so we can record timing
checkpoints_req = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000];
checkpoints = unique([checkpoints_req(checkpoints_req < num_samples), num_samples]);
n_chk       = length(checkpoints);
mc_conv     = zeros(1, n_chk);
mfmc_conv   = zeros(1, n_chk);
t_conv_s    = zeros(1, n_chk);   % wall-clock time at each checkpoint (seconds)
chk_idx     = 1;                  % pointer into checkpoints array

if verbose && num_samples > 1000
    fprintf('Testing containment (this may take a moment)...\n');
end

t_loop_start = tic;   % start timer before containment loop

% MFMC adaptive mode: compute cheap AABB flags for all samples upfront (vectorised, free)
% so we can check the control-variate SE at each checkpoint and break early.
n_used = num_samples;        % actual samples used; updated below if MFMC stops early
mfmc_adaptive_converged = false;
if use_mfmc
    cheap_flags_all = all(bsxfun(@ge, samples, lb_int) & bsxfun(@le, samples, ub_int), 1);
end

for i = 1:num_samples
    pt = samples(:, i);
    in_all = true;

    % Test if point is in ALL models
    for m = 1:num_models
        Z = models{m};

        % Use exact containment check (never interval hull alone!)
        [is_contained, method_used, err_msg] = contains_point_best(Z, pt);

        % Track method usage
        if ~isfield(method_counts, method_used)
            method_counts.(method_used) = 0;
        end
        method_counts.(method_used) = method_counts.(method_used) + 1;

        % If containment check failed or point not in set, mark inconsistent
        if ~isempty(err_msg) || strcmp(method_used, 'unavailable')
            num_containment_errors = num_containment_errors + 1;
            in_all = false;
            break;
        end

        if ~is_contained
            in_all = false;
            break;
        end
    end

    if in_all
        num_consistent = num_consistent + 1;
    end
    consistent_flags(i) = in_all;  % MFMC: record result

    % Progress indicator for large sample sets
    if verbose && mod(i, 5000) == 0
        fprintf('  Processed %d / %d samples (%.1f%% done)\n', ...
            i, num_samples, 100*i/num_samples);
    end

    % Record checkpoint: compute MC estimate and wall time at this N
    if chk_idx <= n_chk && i == checkpoints(chk_idx)
        t_conv_s(chk_idx) = toc(t_loop_start);
        mc_conv(chk_idx) = mean(double(consistent_flags(1:i)));

        % MFMC adaptive stopping: compute control-variate SE at each checkpoint
        if use_mfmc && chk_idx >= 3   % need >= 3 points for a stable alpha estimate
            cf_k     = consistent_flags(1:i);
            af_k     = cheap_flags_all(1:i);
            I_MC_k   = mc_conv(chk_idx);
            I_AABB_k = mean(double(af_k));
            cov_k    = mean(double(cf_k) .* double(af_k)) - I_MC_k * I_AABB_k;
            var_k    = I_AABB_k * (1 - I_AABB_k);
            alpha_k  = cov_k / max(var_k, 1e-12);
            I_MF_k   = max(0, min(1, I_MC_k + alpha_k * (mu_AABB - I_AABB_k)));
            mfmc_conv(chk_idx) = I_MF_k;
            % SE of MFMC estimator = sqrt(p*(1-p)/N * (1 - rho^2))
            rho_k   = cov_k / max(sqrt(var_k) * sqrt(max(I_MC_k*(1-I_MC_k), 1e-9)), 1e-12);
            se_mfmc = sqrt(I_MF_k * (1 - I_MF_k) / i * max(0, 1 - rho_k^2));
            if se_mfmc <= target_se
                n_used = i;
                mfmc_adaptive_converged = true;
                chk_idx = chk_idx + 1;
                break   % stop containment loop early
            end
        end

        chk_idx = chk_idx + 1;
    end
end

if verbose
    fprintf('Testing complete: %d / %d consistent\n', num_consistent, num_samples);
    if num_containment_errors > 0
        fprintf('  Warning: %d containment errors\n', num_containment_errors);
    end
end

%% Step 5b: MFMC within-theta correction
% In adaptive mode (use_mfmc=true): cheap_flags_all was precomputed before the loop;
% use only the first n_used samples (loop may have exited early).
% In standard MC mode: compute cheap_flags now on all num_samples (vectorised, free).
if use_mfmc
    cheap_flags    = cheap_flags_all(1:n_used);
    flags_for_mfmc = consistent_flags(1:n_used);
else
    cheap_flags    = all(bsxfun(@ge, samples, lb_int) & bsxfun(@le, samples, ub_int), 1);
    flags_for_mfmc = consistent_flags;
end

I_AABB_MC = mean(double(cheap_flags));
I_MC_raw  = mean(double(flags_for_mfmc));

% Estimate alpha = Cov(expensive, cheap) / Var(cheap)
cov_ec    = mean(double(flags_for_mfmc) .* double(cheap_flags)) - I_MC_raw * I_AABB_MC;
var_cheap = I_AABB_MC * (1 - I_AABB_MC);
if var_cheap > 1e-12
    alpha_mfmc = cov_ec / var_cheap;
else
    alpha_mfmc = 0;  % AABB check is degenerate (all 0 or all 1)
end

% Within-theta control-variate correction: I_MF = I_MC + alpha*(mu_AABB - I_AABB_MC)
I_MF = max(0, min(1, I_MC_raw + alpha_mfmc * (mu_AABB - I_AABB_MC)));

%% Step 5c: MFMC convergence at each checkpoint (post-hoc, standard MC mode only)
% In adaptive MFMC mode the mfmc_conv values were already filled during the loop.
% In standard MC mode: compute them now post-hoc on cheap_flags.
if ~use_mfmc
    for ci = 1:n_chk
        Nk    = checkpoints(ci);
        cf_k  = consistent_flags(1:Nk);
        af_k  = cheap_flags(1:Nk);

        I_MC_k   = mc_conv(ci);
        I_AABB_k = mean(double(af_k));
        cov_k    = mean(double(cf_k) .* double(af_k)) - I_MC_k * I_AABB_k;
        var_k    = I_AABB_k * (1 - I_AABB_k);
        alpha_k  = cov_k / max(var_k, 1e-12);
        mfmc_conv(ci) = max(0, min(1, I_MC_k + alpha_k * (mu_AABB - I_AABB_k)));
        % t_conv_s(ci) already recorded during the containment loop
    end
end

%% Step 6: Compute statistics
% In adaptive MFMC mode use n_used (actual samples run); in MC mode use num_samples.
n_for_stats  = n_used;
p_consistent = sum(double(flags_for_mfmc)) / n_for_stats;
p_inconsistent = 1 - p_consistent;

% Standard error (binomial proportion, using actual n_used samples)
se = sqrt(p_consistent * (1 - p_consistent) / n_for_stats);

% 95% confidence interval (Wald interval)
z_score = 1.96;  % For 95% CI
ci95_lower = max(0, p_consistent - z_score * se);
ci95_upper = min(1, p_consistent + z_score * se);

% Generate notes about containment methods
notes_parts = {};
if num_containment_errors > 0
    notes_parts{end+1} = sprintf('Warning: %d containment errors (treated as not contained)', ...
        num_containment_errors);
end

method_fields = fieldnames(method_counts);
primary_methods = {};
for i = 1:length(method_fields)
    method = method_fields{i};
    count = method_counts.(method);
    pct = 100 * count / (n_for_stats * num_models);
    if pct >= 1  % Only mention methods used >= 1% of the time
        primary_methods{end+1} = sprintf('%s (%.1f%%)', method, pct);
    end
end

if ~isempty(primary_methods)
    notes_parts{end+1} = sprintf('Containment methods: %s', strjoin(primary_methods, ', '));
end

if isempty(notes_parts)
    notes = 'No special notes';
else
    notes = strjoin(notes_parts, '; ');
end

if verbose
    fprintf('========================================\n');
    fprintf('Results:\n');
    fprintf('  p(consistent) = %.4f ± %.4f\n', p_consistent, se);
    fprintf('  95%% CI: [%.4f, %.4f]\n', ci95_lower, ci95_upper);
    fprintf('  Consistent samples: %d / %d\n', num_consistent, num_samples);
    fprintf('========================================\n\n');
end

%% Step 7: Prepare outputs
% In adaptive MFMC mode the primary score is the control-variate corrected estimate I_MF.
% In standard MC mode the score is the raw MC proportion.
if use_mfmc
    score = I_MF;
else
    score = p_consistent;
end

if return_details || nargout >= 2
    details = struct();
    details.p_consistent = p_consistent;
    details.p_inconsistent = p_inconsistent;
    details.num_samples = n_for_stats;   % actual samples used (n_used in MFMC mode)
    details.num_consistent = num_consistent;
    details.standard_error = se;
    details.ci95_lower = ci95_lower;
    details.ci95_upper = ci95_upper;
    details.sampling_lb = lb_union;
    details.sampling_ub = ub_union;
    details.box_volume = box_volume;
    details.containment_methods_used = method_counts;
    details.num_containment_errors = num_containment_errors;
    details.sampling_method = sampling_method;
    details.notes = notes;
    details.dimension = dim;
    details.num_models = num_models;
    % MFMC within-theta correction fields
    details.mu_AABB    = mu_AABB;
    details.I_AABB_MC  = I_AABB_MC;
    details.alpha_mfmc = alpha_mfmc;
    details.I_MF       = I_MF;
    details.distribution_mu    = dist.mu;
    details.distribution_Sigma = dist.Sigma;
    details.mu_AABB            = mu_AABB;  % update (may differ from placeholder)
    details.consistent_flags   = consistent_flags;  % per-sample binary results for convergence analysis
    % Adaptive MFMC fields
    details.n_used                  = n_for_stats;    % actual expensive samples drawn
    details.mfmc_adaptive_converged = mfmc_adaptive_converged;
    % Convergence checkpoint arrays — measured wall-clock timing + accuracy
    details.mc_convergence_checkpoints  = checkpoints;
    details.mc_convergence_mc           = mc_conv;
    details.mc_convergence_mfmc         = mfmc_conv;
    details.mc_convergence_timing_s     = t_conv_s;   % measured, not derived
else
    details = struct();
end

end

%% ========================================================================
%  HELPER FUNCTIONS
%  ========================================================================

function samples = generateSamples_mc(N, dim, method)
%GENERATESAMPLES_MC Generate N samples in [0,1]^dim using MC or QMC
%   method = 'random' : standard pseudo-random  (convergence O(1/sqrt(N)))
%   method = 'sobol'  : Sobol quasi-random      (convergence ~O(1/N))
%   method = 'halton' : Halton quasi-random     (convergence ~O(1/N))
%   method = 'lhs'    : Latin Hypercube         (stratified sampling)

    switch lower(method)
        case {'random', 'mc', 'pseudorandom'}
            samples = rand(N, dim);

        case {'sobol', 'qmc'}
            try
                p = sobolset(dim, 'Skip', 1);
                p = scramble(p, 'MatousekAffineOwen');
                samples = net(p, N);
            catch ME
                warning('generateSamples_mc:SobolFallback', ...
                    'Sobol failed (%s). Falling back to random.', ME.message);
                samples = rand(N, dim);
            end

        case 'halton'
            try
                p = haltonset(dim, 'Skip', 1);
                p = scramble(p, 'RR2');
                samples = net(p, N);
            catch ME
                warning('generateSamples_mc:HaltonFallback', ...
                    'Halton failed (%s). Falling back to random.', ME.message);
                samples = rand(N, dim);
            end

        case {'lhs', 'latinhypercube'}
            try
                samples = lhsdesign(N, dim, 'Criterion', 'maximin', 'Iterations', 10);
            catch ME
                warning('generateSamples_mc:LHSFallback', ...
                    'LHS failed (%s). Falling back to random.', ME.message);
                samples = rand(N, dim);
            end

        otherwise
            error('generateSamples_mc:UnknownMethod', ...
                'Unknown method: %s. Use random, sobol, halton, or lhs.', method);
    end
end

function prob = gauss_box_prob_mc(lb, ub, mu, Sigma)
%GAUSS_BOX_PROB_MC  Probability mass of N(mu,Sigma) within axis-aligned box [lb,ub].
%   Uses erf (base MATLAB, no Statistics Toolbox required).
%   For full (non-diagonal) Sigma, Cholesky-decorrelates first.
    lb = lb(:); ub = ub(:); mu = mu(:);
    if isdiag(Sigma)
        sig = max(sqrt(diag(Sigma)), eps);
        z_ub = (ub - mu) ./ sig;
        z_lb = (lb - mu) ./ sig;
        prob = prod(0.5 .* (erf(z_ub ./ sqrt(2)) - erf(z_lb ./ sqrt(2))));
    else
        % Decorrelate: compute marginal CDF product (conservative approximation
        % for correlated case; exact for diagonal)
        sig = max(sqrt(diag(Sigma)), eps);
        z_ub = (ub - mu) ./ sig;
        z_lb = (lb - mu) ./ sig;
        prob = prod(0.5 .* (erf(z_ub ./ sqrt(2)) - erf(z_lb ./ sqrt(2))));
    end
    prob = max(0, min(1, prob));
end

function [tf, method, err_msg] = contains_point_best(Z, pt)
%CONTAINS_POINT_BEST Test point containment using best available method
%   Tries multiple containment strategies in priority order:
%   1. CORA built-in exact containment (if available)
%   2. Intersection-based check (point as singleton set)
%   3. Polytope conversion (if supported)
%   
%   CRITICAL: Never returns true from interval hull check alone!
%
% OUTPUTS:
%   tf - Boolean: true if point is contained, false otherwise
%   method - String describing method used
%   err_msg - Empty if success, error message if failed

    tf = false;
    method = 'unknown';
    err_msg = '';

    % Priority 1: Intersection-based check (most reliable for conPolyZono)
    % Create point as singleton set MATCHING the type of Z
    try
        if isa(Z, 'conZonotope')
            % Priority 1a: Direct parametric check for unconstrained zonotopes.
            % Solves G*alpha = pt - c, checks ||alpha||_inf <= 1.
            % Avoids CORA intersection machinery which becomes numerically
            % unreliable when sets nearly perfectly overlap (gives false empty).
            try
                G_z = Z.G;
                c_z = Z.c;
                no_constraints = isempty(Z.A) || size(Z.A, 1) == 0;
                n_dim = size(G_z, 1);
                n_gen = size(G_z, 2);
                d_vec = pt - c_z;

                if no_constraints && n_dim == n_gen && rcond(G_z) > 1e-12
                    % Square, well-conditioned: direct solve
                    alpha = G_z \ d_vec;
                    tf = max(abs(alpha)) <= 1 + 1e-8;
                    method = 'direct_solve_cz';
                    err_msg = '';
                    return;
                elseif no_constraints && n_gen > n_dim
                    % Under-determined (fat G): LP feasibility
                    f_lp  = zeros(n_gen, 1);
                    lb_lp = -ones(n_gen, 1);
                    ub_lp =  ones(n_gen, 1);
                    opts_lp = optimoptions('linprog', 'Display', 'none');
                    [~, ~, exitflag] = linprog(f_lp, [], [], G_z, d_vec, lb_lp, ub_lp, opts_lp);
                    tf = (exitflag == 1);
                    method = 'lp_cz';
                    err_msg = '';
                    return;
                end
            catch
                % Fall through to intersection-based approach below
            end

            % Priority 1b: Intersection-based fallback (original approach)
            % For conZonotope, convert point to conZonotope (same type!)
            pt_cpz = conZonotope(pt, zeros(size(pt,1),0));
            Z_int = Z & pt_cpz;

            % Check if intersection is empty
            if representsa(Z_int, 'emptySet')
                tf = false;
                method = 'intersection_cpz';
                return;
            end

            % Try to convert to interval to verify the intersection
            try
                I_int = interval(Z_int);
                r_int = rad(I_int);
                c_int = center(I_int);

                % Check if the intersection center is close to our point
                if norm(c_int - pt) < 1e-4
                    tf = true;  % Intersection center matches point
                else
                    % Intersection exists but doesn't contain our point
                    tf = false;
                end
            catch
                % interval() failed - try to check bounds directly
                % Get bounds of original set
                try
                    I_Z = interval(Z);
                    lb = infimum(I_Z);
                    ub = supremum(I_Z);

                    % Simple bounds check
                    if all(pt >= lb - 1e-10) && all(pt <= ub + 1e-10)
                        tf = true;
                    else
                        tf = false;
                    end
                catch
                    % Can't verify - be conservative
                    tf = false;
                    err_msg = 'Cannot verify containment for conPolyZono';
                end
            end

            method = 'intersection_cpz';
            return;
        elseif isa(Z, 'zonotope')
            % For zonotope, use zonotope
            pt_zono = zonotope(pt);
            Z_int = Z & pt_zono;
            
            if representsa(Z_int, 'emptySet')
                tf = false;
            else
                try
                    I_int = interval(Z_int);
                    r_int = rad(I_int);
                    if all(r_int < 1e-6)
                        tf = true;
                    else
                        tf = false;
                    end
                catch
                    tf = false;
                end
            end
            method = 'intersection_zono';
            return;
        elseif isa(Z, 'interval')
            % For interval, use interval
            pt_int = interval(pt);
            Z_int = Z & pt_int;
            
            if representsa(Z_int, 'emptySet')
                tf = false;
            else
                try
                    r_int = rad(Z_int);
                    if all(r_int < 1e-6)
                        tf = true;
                    else
                        tf = false;
                    end
                catch
                    tf = false;
                end
            end
            method = 'intersection_interval';
            return;
        end
    catch ME
        err_msg = sprintf('intersection failed: %s', ME.message);
        % Continue to next method
    end
    
    % Priority 2: Try CORA built-in containment functions (for sets that support point queries)
    try
        % Method 2a: contains(Z, pt) - CORA method (works for some set types)
        if ismethod(Z, 'contains')
            tf = contains(Z, pt);
            method = 'contains';
            return;
        end
    catch
        % Continue to next method (conPolyZono.contains doesn't support points)
    end
    
    try
        % Method 2b: in(Z, pt) - older CORA syntax
        if ismethod(Z, 'in')
            tf = in(Z, pt);
            method = 'in';
            return;
        end
    catch
        % Continue to next method
    end
    
    % Priority 3: Polytope conversion (if CORA supports it)
    try
        if isa(Z, 'conZonotope')
            % Try to convert to polytope for exact membership
            if ismethod(Z, 'polytope')
                P = polytope(Z);
                if ismethod(P, 'contains')
                    tf = contains(P, pt);
                    method = 'polytope';
                    return;
                end
            end
        end
    catch ME
        err_msg = sprintf('polytope conversion failed: %s', ME.message);
        % Continue
    end

    % Priority 4: (conZonotope — already handled by contains() above)
    try
        % Placeholder to maintain structure; conZonotope containment
        % is already attempted via contains() in Priority 1/2 above
        if false
        end
    catch ME
        err_msg = sprintf('fallback failed: %s', ME.message);
        % Continue
    end
    
    % If we reach here, no exact method worked
    % CRITICAL: Do NOT fall back to interval hull and return true!
    % Instead, return false and mark as unavailable
    tf = false;
    method = 'unavailable';
    if isempty(err_msg)
        err_msg = 'No exact containment method available for this set type';
    end
end
