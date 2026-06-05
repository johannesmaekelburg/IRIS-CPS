function [C, Csym, details] = score_jaccard(Zsrc, Ztgt, F, f, options)
%SCORE_JACCARD Compute Jaccard-based consistency scores using geometric volumes
%   Computes how consistent a propagated source set is with a target set
%   after intersection, using CORA's conPolyZono operations and volume-based metrics.
%
% PRIMARY METRIC: C (directional consistency)
%   Use C as the headline metric in experiments and publications.
%   Csym is provided as the Jaccard index for reference.
%
% SYNTAX
%   C = score_jaccard(Zsrc, Ztgt, F, f)
%   [C, Csym] = score_jaccard(Zsrc, Ztgt, F, f)
%   [C, Csym, details] = score_jaccard(Zsrc, Ztgt, F, f, options)
%
% INPUTS
%   Zsrc    - Source set (zonotope or conPolyZono)
%   Ztgt    - Target set (zonotope or conPolyZono)
%   F       - Transformation matrix for affine map (n_out × n_state)
%   f       - Translation vector (n_out × 1), optional (default: zeros)
%   options - (optional) struct with fields:
%             .verbose - Display progress (default: false)
%             .return_details - Return detailed metrics (default: false)
%             .compute_symmetric - Compute Csym score (default: true)
%             .track_statistics - Track debugging statistics (default: false)
%             .warn_mixed_units - Warn about mixed unit scales (default: true)
%             .dimension_weights - Per-dimension weights for volume function (default: ones)
%                                   Set to 0 for boolean/logical dimensions to exclude them
%             .exclude_boolean - Auto-detect and exclude boolean dimensions (default: false)
%
% OUTPUTS
%   C       - PRIMARY: Directional consistency score ∈ [0,1]
%             C = vol(Zint) / vol(Zprop)
%             where vol(Z) = volume of interval hull (product of widths)
%             C=0 if intersection is empty, C=NaN if vol(Zprop)=0
%   Csym    - SECONDARY: Jaccard index (geometric overlap) ∈ [0,1]
%             Csym = vol(Zint) / (vol(Zprop) + vol(Ztgt) - vol(Zint))
%             (Over-approximation due to interval hull, but proper volume-based metric)
%   details - struct with fields:
%             .Zprop - Propagated source set
%             .Zint - Intersection of propagated and target
%             .vol_prop - Volume of propagated set
%             .vol_tgt - Volume of target set
%             .vol_int - Volume of intersection
%             .is_empty - Boolean, true if intersection is empty
%             .statistics - (if track_statistics=true) Debugging info
%
% ALGORITHM
%   1. Propagate: Zprop = F*Zsrc + f (using affineMap_cPZ)
%   2. Intersect: Zint = Zprop ∩ Ztgt (using and_)
%   3. Volume function: vol(Z) = prod(2*rad(interval(Z))) [geometric volume]
%   4. Consistency: C = vol(Zint) / vol(Zprop), clamped to [0,1]
%   5. Jaccard: Csym = vol(Zint) / (vol(Zprop) + vol(Ztgt) - vol(Zint))
%
% EXAMPLES
%   % Example 1: Basic usage
%   E = eye(2); A = []; b = []; EC = [];
%   Zsrc = conPolyZono([300; 25], [5 0; 0 2], E, A, b, EC);
%   Ztgt = conPolyZono([301; 25.5], [4 0; 0 1.5], E, A, b, EC);
%   F = [1.005 0; 0 0.995]; f = [0.5; 0.3];
%   [C, Csym] = score_jaccard(Zsrc, Ztgt, F, f);
%
% VERSION: 2.0 - Refactored to implement Jaccard directly (not via wrapper)
%                Changed from sum(radii) to prod(widths) for proper geometric volumes

%% Parse inputs
if nargin < 4 || isempty(f)
    f = zeros(size(F, 1), 1);
end

if nargin < 5
    options = struct();
end

verbose = getOption(options, 'verbose', false);
return_details = getOption(options, 'return_details', false);
compute_symmetric = getOption(options, 'compute_symmetric', true);
track_statistics = getOption(options, 'track_statistics', false);
warn_mixed_units = getOption(options, 'warn_mixed_units', true);
dimension_weights = getOption(options, 'dimension_weights', []);
exclude_boolean = getOption(options, 'exclude_boolean', false);

% Initialize statistics tracking
if track_statistics
    stats = struct();
    stats.cora_version = detectCORAVersion();
    stats.C_raw = NaN;
    stats.Csym_raw = NaN;
    stats.C_clamped = false;
    stats.Csym_clamped = false;
    stats.empty_count = 0;
end

if verbose
    fprintf('======================================\n');
    fprintf('Jaccard Consistency Score Computation\n');
    fprintf('======================================\n');
end

%% Step 0: Dimension consistency check
n_src = length(Zsrc.c);
n_tgt = length(Ztgt.c);
n_out = size(F, 1);

if n_out ~= n_tgt
    error('score_jaccard:DimensionMismatch', ...
        'Propagated dimension (%d) does not match target dimension (%d). F must map to Ztgt space.', ...
        n_out, n_tgt);
end

if size(F, 2) ~= n_src
    error('score_jaccard:DimensionMismatch', ...
        'Transformation matrix F (%dx%d) incompatible with source dimension (%d).', ...
        size(F, 1), size(F, 2), n_src);
end

if verbose
    fprintf('Dimension check: Zsrc(%d) -> F(%dx%d) -> Zprop(%d) vs Ztgt(%d) ✓\n', ...
        n_src, size(F, 1), size(F, 2), n_out, n_tgt);
end

%% Step 1: Propagate source set
if verbose
    fprintf('Step 1: Propagating source set...\n');
end

Zprop = CS_Types.affineMap_cPZ(Zsrc, F, f);

if verbose
    fprintf('  Source center: [%s]\n', sprintf('%.3f ', Zsrc.c));
    fprintf('  Propagated center: [%s]\n', sprintf('%.3f ', Zprop.c));
end

% Handle dimension weights and boolean detection
I_prop = interval(Zprop);
r_prop = rad(I_prop);

if isempty(dimension_weights)
    dimension_weights = ones(n_out, 1);
    
    if exclude_boolean
        c_prop = Zprop.c;
        is_boolean = (abs(c_prop - 0.5) < 0.1) & (abs(r_prop - 0.5) < 0.1);
        dimension_weights(is_boolean) = 0;
        
        if verbose && any(is_boolean)
            fprintf('  Auto-excluded %d boolean-like dimensions\n', sum(is_boolean));
        end
    end
else
    if length(dimension_weights) ~= n_out
        error('score_jaccard:InvalidWeights', ...
            'dimension_weights must have length %d (target dimension)', n_out);
    end
    dimension_weights = dimension_weights(:);
end

% Warn about mixed units/scales
if warn_mixed_units && n_out > 1
    active_dims = dimension_weights > 0;
    if any(active_dims)
        r_active = r_prop(active_dims);
        if max(r_active) / (min(r_active) + eps) > 100
            warning('score_jaccard:MixedScales', ...
                'State vector has mixed scales (active radii range: [%.2e, %.2e]). Consider normalization.', ...
                min(r_active), max(r_active));
        end
    end
end

%% Step 2: Compute intersection
if verbose
    fprintf('Step 2: Computing intersection...\n');
end

% Compute conPolyZono intersection for visualization/details
Zint = intersectCPZ(Zprop, Ztgt);

% Compute interval hulls for VOLUME computation (reliable, avoids
% interval(conPolyZono_intersection) which silently fails to 0 in CORA).
% The intersection of interval hulls is a valid over-approximation and
% is always numerically stable.
I_prop = interval(Zprop);
I_tgt  = interval(Ztgt);
I_int  = I_prop & I_tgt;  % interval intersection

%% Step 3: Check if intersection is empty
% Use interval-based check first (reliable); fall back to conPolyZono check.
r_int = rad(I_int);
r_int = r_int(:);  % flatten to column — guards against matrix-valued rad() output
is_empty_interval = isempty(r_int) || any(isnan(r_int)) || any(r_int < -1e-12);

if is_empty_interval
    is_empty = true;
else
    is_empty = false;
end

if ~is_empty && verbose
    fprintf('  Interval intersection widths: [%s]\n', sprintf('%.4f ', 2*r_int));
end

if is_empty && verbose
    fprintf('  ⚠ Intersection is empty (interval check)!\n');
end

if is_empty && track_statistics
    stats.empty_count = 1;
end

%% Step 4: Compute volumes via interval hulls (always numerically stable)
if verbose
    fprintf('Step 3: Computing set volumes...\n');
end

vol_prop = computeVolumeInterval(I_prop, dimension_weights);

if is_empty
    vol_int = 0;
    vol_tgt = NaN;
else
    vol_int = computeVolumeInterval(I_int, dimension_weights);
    vol_tgt = computeVolumeInterval(I_tgt, dimension_weights);
end

if verbose
    fprintf('  Volume vol(Zprop) = %.6f\n', vol_prop);
    if ~isnan(vol_tgt)
        fprintf('  Volume vol(Ztgt)  = %.6f\n', vol_tgt);
    end
    fprintf('  Volume vol(Zint)  = %.6f\n', vol_int);
end

%% Step 5: Compute consistency score C
if verbose
    fprintf('Step 4: Computing consistency scores...\n');
end

if is_empty
    C = 0;
    if verbose
        fprintf('  → C = 0 (empty intersection)\n');
    end
elseif vol_prop <= eps
    C = NaN;
    if verbose
        fprintf('  → C = NaN (degenerate propagated set: vol = %.2e)\n', vol_prop);
    end
else
    C_raw = vol_int / vol_prop;
    
    if track_statistics
        stats.C_raw = C_raw;
    end
    
    if C_raw > 1.0 || C_raw < 0
        if track_statistics
            stats.C_clamped = true;
        end
        if verbose
            fprintf('  ⚠ Over-approximation artifact: raw C = %.4f (clamping to [0,1])\n', C_raw);
        end
    end
    
    C = min(max(C_raw, 0), 1);
    
    if verbose
        fprintf('  → C = %.6f (PRIMARY directional consistency)\n', C);
    end
end

%% Step 6: Compute Jaccard index Csym
if is_empty
    Csym = 0;
    if verbose
        fprintf('  → Csym = 0 (empty intersection)\n');
    end
elseif vol_prop <= eps || vol_tgt <= eps
    Csym = NaN;
    if verbose
        fprintf('  → Csym = NaN (degenerate set: vol_prop=%.2e, vol_tgt=%.2e)\n', vol_prop, vol_tgt);
    end
else
    den = vol_prop + vol_tgt - vol_int;
    
    if den <= eps
        Csym = NaN;
        if verbose
            fprintf('  → Csym = NaN (denominator = %.2e <= eps)\n', den);
        end
        if track_statistics
            stats.Csym_raw = NaN;
        end
    else
        Csym_raw = vol_int / den;
        
        if track_statistics
            stats.Csym_raw = Csym_raw;
        end
        
        if Csym_raw > 1.0 || Csym_raw < 0
            if track_statistics
                stats.Csym_clamped = true;
            end
            if verbose
                fprintf('  ⚠ Over-approximation artifact: raw Csym = %.4f (clamping)\n', Csym_raw);
            end
        end
        
        Csym = min(max(Csym_raw, 0), 1);
        
        if verbose
            fprintf('  → Csym = %.6f (Jaccard index)\n', Csym);
        end
    end
end

%% Step 7: Prepare details output
if nargout >= 3 || return_details
    details = struct();
    details.Zprop = Zprop;
    details.Zint = Zint;
    details.vol_prop = vol_prop;
    details.vol_tgt = vol_tgt;
    details.vol_int = vol_int;
    details.is_empty = is_empty;
    
    % Legacy field names for backward compatibility
    details.s_prop = vol_prop;
    details.s_tgt = vol_tgt;
    details.s_int = vol_int;
    
    if track_statistics
        details.statistics = stats;
    end
else
    details = struct();
end

if verbose
    fprintf('======================================\n');
    fprintf('Jaccard computation complete!\n');
    fprintf('======================================\n\n');
end

end

%% ========================================================================
%  HELPER FUNCTIONS
%  ========================================================================

function val = getOption(opts, field, default)
%GETOPTION Get option value or return default
    if isfield(opts, field)
        val = opts.(field);
    else
        val = default;
    end
end

function version_str = detectCORAVersion()
%DETECTCORAVERSION Attempt to detect CORA version
    version_str = 'unknown';
    try
        if exist('CORAROOT', 'var')
            version_file = fullfile(CORAROOT, 'version.txt');
            if exist(version_file, 'file')
                fid = fopen(version_file, 'r');
                version_str = strtrim(fgetl(fid));
                fclose(fid);
                return;
            end
        end
        
        if exist('conZonotope', 'class')
            m = methods('conZonotope');
            if ismember('and', m)
                version_str = '2024+ (has and method)';
            else
                version_str = '2023 or earlier (no and method)';
            end
        end
    catch
    end
end

function vol = computeVolume(Z, weights)
%COMPUTEVOLUME Compute geometric volume of a set via interval hull
%   vol = computeVolume(Z, weights) computes vol(Z) = prod(widths(interval(Z)))
%   for active dimensions (where weight > 0)
%
%   NOTE: For intersection volumes use computeVolumeInterval() directly to
%   avoid CORA failures when converting constrained polynomial zonotopes.
%
% VERSION: 2.0 - Changed from sum(radii) to prod(widths) for proper geometric volume

    if nargin < 2 || isempty(weights)
        weights = 1;
    end
    
    try
        I = interval(Z);
        vol = computeVolumeInterval(I, weights);
    catch ME
        warning('score_jaccard:computeVolume:Failed', ...
            'Failed to compute volume: %s', ME.message);
        vol = 0;
    end
end

function vol = computeVolumeInterval(I, weights)
%COMPUTEVOLUMEINTERVAL Compute geometric volume directly from an interval object
%   This is the numerically stable core used for all volume computations.
%   Avoids calling interval() on complex constrained sets (which can fail silently).

    if nargin < 2 || isempty(weights)
        weights = 1;
    end
    
    r = rad(I);
    r = r(:);  % flatten to column vector — guards against matrix-valued rad() output

    if isscalar(weights) && weights > 0
        if isempty(r) || all(r == 0)
            vol = 0;
        else
            vol = prod(2 * r);
        end
    else
        if isempty(r)
            vol = 0;
            return;
        end
        active_dims = (weights(:) > 0);
        % Guard against r being shorter than active_dims (degenerate rad() output)
        if numel(r) < numel(active_dims)
            vol = 0;
            return;
        end
        r_active = r(active_dims);
        
        if isempty(r_active) || all(r_active == 0)
            vol = 0;
        else
            vol = prod(2 * r_active);
        end
    end
end

function is_empty = checkEmptiness(Z, verbose)
%CHECKEMPTINESS Check if set is empty using multiple methods
    is_empty = false;
    
    if exist('isEmptySet', 'file')
        is_empty = isEmptySet(Z);
    elseif exist('CS_Types', 'class') && ismethod(CS_Types, 'isEmptyCompat')
        is_empty = CS_Types.isEmptyCompat(Z);
    else
        try
            if representsa(Z, 'emptySet')
                is_empty = true;
            end
        catch
            try
                interval(Z);
                is_empty = false;
            catch
                warning('score_jaccard:EmptyDetection', ...
                    'Cannot determine emptiness - treating as empty');
                is_empty = true;
            end
        end
    end
    
    if verbose && is_empty
        fprintf('  ⚠ Intersection is empty!\n');
    end
end

function Zint = intersectCPZ(Z1, Z2)
%INTERSECTCPZ CORA compatibility wrapper for intersection
    try
        Zint = Z1 & Z2;
        return;
    catch
    end
    
    try
        Zint = and(Z1, Z2);
        return;
    catch
    end
    
    try
        Zint = intersect(Z1, Z2);
        return;
    catch
    end
    
    try
        Zint = and_(Z1, Z2);
        return;
    catch ME
        error('score_jaccard:IntersectionFailed', ...
            'All intersection methods failed. CORA version incompatible? Error: %s', ME.message);
    end
end
