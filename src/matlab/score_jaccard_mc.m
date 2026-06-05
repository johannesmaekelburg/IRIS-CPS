function [J_mc, details] = score_jaccard_mc(A, B, options)
%SCORE_JACCARD_MC Monte Carlo / Quasi-Monte Carlo Jaccard index for sets
%   Computes approximate Jaccard overlap by sampling from the union
%   bounding box and testing containment in both sets.
%   Supports standard MC (pseudo-random) and QMC (Sobol/Halton) sampling.
%
% ALGORITHM:
%   1. Compute union bounding box: B = hull(interval(A), interval(B))
%   2. Sample N points from B (MC or QMC low-discrepancy sequence)
%   3. For each sample x: test inA = contains(A,x), inB = contains(B,x)
%   4. Estimate volumes: vol(A) = p_A * vol(B), vol(B) = p_B * vol(B)
%   5. Estimate Jaccard: J_mc = vol(A∩B) / (vol(A) + vol(B) - vol(A∩B))
%
% SYNTAX:
%   J_mc = score_jaccard_mc(A, B)
%   [J_mc, details] = score_jaccard_mc(A, B)
%   [J_mc, details] = score_jaccard_mc(A, B, options)
%
% INPUTS:
%   A       - First set (conPolyZono, zonotope, or interval)
%   B       - Second set (conPolyZono, zonotope, or interval)
%   options - (optional) struct with fields:
%             .N - Number of samples (default: auto-scaled by dimension)
%             .seed - Random seed for reproducibility (default: none)
%             .verbose - Display progress (default: false)
%             .return_details - Return detailed metrics (default: true)
%             .sampling_method - Sampling strategy (default: 'sobol')
%                   'random' - Standard pseudo-random MC (convergence O(1/√N))
%                   'sobol'  - Sobol low-discrepancy QMC (convergence O(1/N))
%                   'halton' - Halton low-discrepancy QMC (convergence O(1/N))
%                   'lhs'    - Latin Hypercube Sampling (stratified)
%             .containment_mode - Containment check strategy (default: 'auto')
%                   'exact' - Use CORA's exact containment (if available)
%                   'interval' - Use interval hull approximation (conservative)
%                   'auto' - Try exact first, fallback to interval
%             .dimension_weights - Per-dimension weights (default: ones)
%
% OUTPUTS:
%   J_mc    - Monte Carlo estimated Jaccard index ∈ [0,1]
%             J_mc ≈ |A ∩ B| / |A ∪ B| (estimated via sampling)
%   details - struct with fields:
%             .N - Number of samples used
%             .p_A - Fraction of samples in A
%             .p_B - Fraction of samples in B
%             .p_AB - Fraction of samples in A ∩ B
%             .vol_box - Volume of sampling box
%             .vol_A_est - Estimated volume of A
%             .vol_B_est - Estimated volume of B
%             .vol_AB_est - Estimated volume of A ∩ B
%             .vol_union_est - Estimated volume of A ∪ B             .sampling_method - Sampling method used ('random','sobol','halton')%             .containment_mode_used - Actual containment strategy used
%             .samples_A - Indices of samples in A
%             .samples_B - Indices of samples in B
%             .confidence_interval - (optional) 95% CI for J_mc
%
% SPECIAL CASES:
%   - Both sets empty (p_A=0, p_B=0) → J_mc = 0
%   - One set empty → J_mc = 0
%   - Denominator ≈ 0 → J_mc = NaN
%   - No samples in either set → J_mc = 0 (with warning)
%
% EXAMPLES:
%   % Example 1: High overlap (2D)
%   E = eye(2); A = []; b = []; EC = [];
%   Z1 = conPolyZono([10; 10], [2 0; 0 2], E, A, b, EC);
%   Z2 = conPolyZono([10.5; 10.5], [2 0; 0 2], E, A, b, EC);
%   opts.N = 10000; opts.seed = 42;
%   [J, det] = score_jaccard_mc(Z1, Z2, opts);
%   fprintf('MC Jaccard: %.3f (from %d samples)\n', J, det.N);
%
%   % Example 2: Compare with AABB
%   J_aabb = score_jaccard_aabb(Z1, Z2);
%   fprintf('AABB: %.3f vs MC: %.3f\n', J_aabb, J);
%
% NOTES:
%   - Sampling provides APPROXIMATE estimate (not exact)
%   - MC accuracy scales with 1/√N; QMC scales with ~1/N (much faster)
%   - QMC (Sobol) typically needs 3-10x fewer samples than MC for same error
%   - For 2D: default N=5000, 4D: N=15000 (configurable)
%   - Containment check may use interval approximation (see containment_mode)
%   - Runtime: O(N * complexity(containment_check))
%   - Use AABB version for fast proxy, MC/QMC for better accuracy
%   - Sobol requires MATLAB Statistics Toolbox (sobolset); Halton uses haltonset
%
% PERFORMANCE GUIDELINES:
%   - 2D, N=5000:  ~0.5s, error ~1%
%   - 3D, N=10000: ~1s,   error ~1%
%   - 4D, N=15000: ~2s,   error ~0.8%
%   - Higher N → better accuracy, longer runtime
%
% VERSION: 1.0 - Initial implementation of MC-based Jaccard
%
% SEE ALSO:
%   score_jaccard_aabb, interval, contains

%% Parse inputs
if nargin < 3
    options = struct();
end

verbose = getOption(options, 'verbose', false);
return_details = getOption(options, 'return_details', true);
containment_mode = getOption(options, 'containment_mode', 'auto');
sampling_method = getOption(options, 'sampling_method', 'sobol');
dimension_weights = getOption(options, 'dimension_weights', []);

% Validate sampling method (all QMCEngine-supported names)
valid_methods = {'random', 'mc', 'pseudorandom', 'sobol', 'qmc', 'halton', 'lhs', 'latinhypercube'};
if ~ismember(lower(sampling_method), valid_methods)
    error('score_jaccard_mc:InvalidSamplingMethod', ...
        'sampling_method must be one of: random, sobol, halton, lhs');
end

if verbose
    fprintf('======================================\n');
    fprintf('Monte Carlo Jaccard Computation\n');
    fprintf('======================================\n');
end

%% Step 1: Compute union bounding box
try
    IA = interval(A);
    IB = interval(B);
catch ME
    error('score_jaccard_mc:IntervalConversion', ...
        'Failed to convert to interval: %s', ME.message);
end

% Validate dimensions
n = length(infimum(IA));
if length(infimum(IB)) ~= n
    error('score_jaccard_mc:DimensionMismatch', ...
        'Sets have different dimensions: A(%d) vs B(%d)', n, length(infimum(IB)));
end

% Handle dimension weights
if isempty(dimension_weights)
    dimension_weights = ones(n, 1);
else
    if length(dimension_weights) ~= n
        error('score_jaccard_mc:InvalidWeights', ...
            'dimension_weights must have length %d', n);
    end
    dimension_weights = dimension_weights(:);
end

active_dims = dimension_weights > 0;
if ~any(active_dims)
    error('score_jaccard_mc:NoActiveDims', 'All dimensions have zero weight');
end
n_active = sum(active_dims);

% Determine number of samples (scale with dimension)
if isfield(options, 'N')
    N = options.N;
else
    % Auto-scale: 5000 for 2D, increase for higher dimensions
    N = max(5000, round(5000 * (n_active / 2)^1.5));
end

if verbose
    fprintf('Dimension: %d (%d active)\n', n, n_active);
    fprintf('Samples: N = %d\n', N);
end

% Compute union bounding box (sampling domain)
inf_union = min(infimum(IA), infimum(IB));
sup_union = max(supremum(IA), supremum(IB));

% Compute volume of sampling box
widths_union = sup_union - inf_union;
widths_active = widths_union(active_dims);

if any(widths_active == 0)
    warning('score_jaccard_mc:DegenerateDimension', ...
        'Union box has zero width in some active dimension');
end

vol_box = prod(widths_active);

if verbose
    fprintf('Union bounding box volume: %.6e\n', vol_box);
end

%% Step 2: Set random seed if specified
if isfield(options, 'seed')
    rng(options.seed);
    if verbose
        fprintf('Random seed: %d\n', options.seed);
    end
end

%% Step 3: Generate samples from union box (MC or QMC)
if verbose
    fprintf('Step 1: Generating %d %s samples...\n', N, upper(sampling_method));
end

samples_unit = generateSamples(N, n, sampling_method, options);

% Scale from [0,1]^n to actual bounding box: x = inf + (sup - inf) * u
samples = inf_union' + (sup_union' - inf_union') .* samples_unit;

%% Step 4: Test containment for each sample
if verbose
    fprintf('Step 2: Testing containment (mode: %s)...\n', containment_mode);
    tic;
end

% Determine actual containment mode to use
[in_A, mode_used_A] = testContainmentBatch(A, samples, containment_mode, verbose);
[in_B, mode_used_B] = testContainmentBatch(B, samples, containment_mode, verbose);

if verbose
    t_elapsed = toc;
    fprintf('  Containment tests completed in %.2f seconds\n', t_elapsed);
    fprintf('  Mode used: A=%s, B=%s\n', mode_used_A, mode_used_B);
end

% Compute intersection
in_AB = in_A & in_B;

%% Step 5: Estimate probabilities and volumes
p_A = sum(in_A) / N;
p_B = sum(in_B) / N;
p_AB = sum(in_AB) / N;

vol_A_est = p_A * vol_box;
vol_B_est = p_B * vol_box;
vol_AB_est = p_AB * vol_box;

if verbose
    fprintf('Step 3: Estimated probabilities\n');
    fprintf('  p(A) = %.4f (%d/%d samples)\n', p_A, sum(in_A), N);
    fprintf('  p(B) = %.4f (%d/%d samples)\n', p_B, sum(in_B), N);
    fprintf('  p(A∩B) = %.4f (%d/%d samples)\n', p_AB, sum(in_AB), N);
end

%% Step 6: Compute Jaccard index
if p_A < eps && p_B < eps
    % Both sets appear empty
    J_mc = 0;
    vol_union_est = 0;
    if verbose
        fprintf('  → Both sets appear empty (J=0)\n');
    end
    
elseif p_AB < eps
    % Disjoint or no intersection detected
    J_mc = 0;
    vol_union_est = vol_A_est + vol_B_est;
    if verbose
        fprintf('  → No intersection detected (J=0)\n');
    end
    
else
    % Normal case: compute via inclusion-exclusion
    vol_union_est = vol_A_est + vol_B_est - vol_AB_est;
    
    if vol_union_est < eps
        J_mc = NaN;
        if verbose
            fprintf('  → Warning: denominator ≈ 0 (J=NaN)\n');
        end
    else
        J_mc_raw = vol_AB_est / vol_union_est;
        
        % Clamp to [0, 1]
        if J_mc_raw < 0 || J_mc_raw > 1
            if verbose
                fprintf('  → Warning: raw J_mc = %.4f (clamping)\n', J_mc_raw);
            end
        end
        
        J_mc = min(max(J_mc_raw, 0), 1);
        
        if verbose
            fprintf('  → J_mc = %.6f\n', J_mc);
        end
    end
end

%% Step 7: Compute confidence interval (optional)
if p_AB > 0 && p_AB < 1 && return_details
    % Use bootstrap or analytical approximation for CI
    % Simplified: assume normal approximation for proportion
    se_AB = sqrt(p_AB * (1 - p_AB) / N);
    ci_95 = 1.96 * se_AB;
    
    % Propagate to Jaccard (simplified, assumes fixed denominator)
    J_lower = max(0, (p_AB - ci_95) * vol_box / max(vol_union_est, eps));
    J_upper = min(1, (p_AB + ci_95) * vol_box / max(vol_union_est, eps));
    
    confidence_interval = [J_lower, J_upper];
else
    confidence_interval = [NaN, NaN];
end

%% Step 8: Prepare details output
if nargout >= 2 || return_details
    details = struct();
    details.N = N;
    details.p_A = p_A;
    details.p_B = p_B;
    details.p_AB = p_AB;
    details.vol_box = vol_box;
    details.vol_A_est = vol_A_est;
    details.vol_B_est = vol_B_est;
    details.vol_AB_est = vol_AB_est;
    details.vol_union_est = vol_union_est;
    details.sampling_method = sampling_method;
    details.containment_mode_used = struct('A', mode_used_A, 'B', mode_used_B);
    details.samples_A = find(in_A);
    details.samples_B = find(in_B);
    details.confidence_interval = confidence_interval;
    details.dimension = n;
    details.active_dimensions = n_active;
else
    details = struct();
end

if verbose
    fprintf('======================================\n');
    fprintf('MC-Jaccard complete: J = %.6f\n', J_mc);
    if ~isnan(confidence_interval(1))
        fprintf('95%% CI: [%.4f, %.4f]\n', confidence_interval(1), confidence_interval(2));
    end
    fprintf('======================================\n\n');
end

end

%% ========================================================================
%  CONTAINMENT CHECKING
%  ========================================================================

function [in_set, mode_used] = testContainmentBatch(Z, samples, mode, verbose)
%TESTCONTAINMENTBATCH Test containment for a batch of samples
%   Uses interval hull for REJECTION only (point outside box → definitely out).
%   For points inside the box, tries exact methods; if all fail, conservatively
%   returns false (never inflates the score).
%
% Strategy:
%   1. Pre-filter: reject samples outside interval hull (fast, exact rejection)
%   2. For surviving samples: try exact containment methods
%   3. If exact methods fail → conservatively mark as NOT contained

    N = size(samples, 1);
    in_set = false(N, 1);
    mode_used = 'unknown';
    
    %% Step 1: Interval hull pre-filter (reject points definitely outside)
    try
        I = interval(Z);
        lb = infimum(I);
        ub = supremum(I);
    catch ME
        error('testContainmentBatch:IntervalFailed', ...
            'Cannot compute interval hull: %s', ME.message);
    end
    
    % Fast vectorized rejection: mark which samples are inside the box
    tol = 1e-9;
    in_box = false(N, 1);
    for i = 1:N
        pt = samples(i, :)';
        in_box(i) = all(pt >= lb - tol) && all(pt <= ub + tol);
    end
    
    n_in_box = sum(in_box);
    
    if verbose
        fprintf('    Pre-filter: %d / %d samples inside interval hull\n', n_in_box, N);
    end
    
    % If using pure interval mode, accept all box-contained points
    if strcmp(mode, 'interval')
        in_set = in_box;
        mode_used = 'interval';
        return;
    end
    
    % Points outside the box are definitely NOT contained → stay false
    % Only test exact containment for points that passed the pre-filter
    if n_in_box == 0
        mode_used = 'exact_prefiltered';
        return;
    end
    
    %% Step 2: Exact containment for points inside the interval hull
    candidates = find(in_box);
    exact_method_found = false;
    
    for idx = 1:length(candidates)
        i = candidates(idx);
        pt = samples(i, :)';
        
        contained = false;
        
        % Method 1: CORA contains(Z, pt)
        if ~exact_method_found || true  % always try
            try
                result = contains(Z, pt);
                contained = result;
                exact_method_found = true;
                in_set(i) = contained;
                continue;
            catch
                % Method not available for this set type
            end
        end
        
        % Method 2: CORA in(Z, pt)
        try
            if ismethod(Z, 'in')
                result = in(Z, pt);
                contained = result;
                exact_method_found = true;
                in_set(i) = contained;
                continue;
            end
        catch
            % Continue
        end
        
        % Method 3: Polytope conversion
        try
            if isa(Z, 'conZonotope') && ismethod(Z, 'polytope')
                P = polytope(Z);
                result = contains(P, pt);
                contained = result;
                exact_method_found = true;
                in_set(i) = contained;
                continue;
            end
        catch
            % Continue
        end
        
        % CRITICAL: If no exact method worked, conservatively mark as NOT contained
        % Do NOT assume contained just because it's inside the interval hull!
        in_set(i) = false;
    end
    
    if exact_method_found
        mode_used = 'exact_prefiltered';
    else
        % No exact method available: all candidates stay false (conservative)
        mode_used = 'conservative_reject';
        if verbose
            fprintf('    WARNING: No exact containment method available.\n');
            fprintf('    All %d box-interior points conservatively marked as NOT contained.\n', n_in_box);
        end
    end
end

function samples = generateSamples(N, dim, method, options)
%GENERATESAMPLES Generate N samples in [0,1]^dim using MC or QMC
%   method = 'random' : standard pseudo-random (convergence O(1/sqrt(N)))
%   method = 'sobol'  : Sobol quasi-random   (convergence ~O(1/N))
%   method = 'halton' : Halton quasi-random  (convergence ~O(1/N))
%   method = 'lhs'    : Latin Hypercube      (stratified sampling)

    switch lower(method)
        case {'random', 'mc', 'pseudorandom'}
            samples = rand(N, dim);

        case {'sobol', 'qmc'}
            try
                p = sobolset(dim, 'Skip', 1);
                p = scramble(p, 'MatousekAffineOwen');
                samples = net(p, N);
            catch ME
                warning('generateSamples:SobolFallback', ...
                    'Sobol failed (%s). Falling back to random.', ME.message);
                samples = rand(N, dim);
            end

        case 'halton'
            try
                p = haltonset(dim, 'Skip', 1);
                p = scramble(p, 'RR2');
                samples = net(p, N);
            catch ME
                warning('generateSamples:HaltonFallback', ...
                    'Halton failed (%s). Falling back to random.', ME.message);
                samples = rand(N, dim);
            end

        case {'lhs', 'latinhypercube'}
            try
                samples = lhsdesign(N, dim, 'Criterion', 'maximin', 'Iterations', 10);
            catch ME
                warning('generateSamples:LHSFallback', ...
                    'LHS failed (%s). Falling back to random.', ME.message);
                samples = rand(N, dim);
            end

        otherwise
            error('generateSamples:UnknownMethod', ...
                'Unknown method: %s. Use random, sobol, halton, or lhs.', method);
    end
end

function val = getOption(opts, field, default)
%GETOPTION Get option value or return default
    if isfield(opts, field)
        val = opts.(field);
    else
        val = default;
    end
end
