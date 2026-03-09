% DEBUG MC CONTAINMENT ISSUE
% Test different containment methods for conPolyZono

clear; close all; clc;

% Add paths
src_path = fullfile(fileparts(pwd), 'src');
addpath(src_path);

cps_framework_path = fullfile(fileparts(fileparts(pwd)), 'CPS-Uncertainty-Propagation-Framework');
addpath(genpath(fullfile(cps_framework_path, 'src')));

consistency_addon_path = fullfile(cps_framework_path, 'addons', 'consistency_scoring');
addpath(consistency_addon_path);
addpath(fullfile(consistency_addon_path, 'methods'));

fprintf('========================================\n');
fprintf('DEBUGGING MC CONTAINMENT\n');
fprintf('========================================\n\n');

% Load first zonotope file
zonotope_dir = fullfile(fileparts(pwd), 'data', 'zonotopes');
zono_file = fullfile(zonotope_dir, 'zonotopes_scenario_1_exp0001.mat');

if ~exist(zono_file, 'file')
    error('Zonotope file not found: %s', zono_file);
end

fprintf('Loading: %s\n\n', zono_file);
data = load(zono_file);

Z_prop = data.Z_source_post;
Z_target = data.Z_target;

% Propagate
F = data.mapping_F;
f = data.mapping_f;
Z_propagated = CS_Types.affineMap_cPZ(Z_prop, F, f);

fprintf('Z_propagated:\n');
fprintf('  Center: [%s]\n', num2str(Z_propagated.c'));
fprintf('  Generators: %d\n', size(Z_propagated.G, 2));
fprintf('  Dimension: %d\n\n', length(Z_propagated.c));

fprintf('Z_target:\n');
fprintf('  Center: [%s]\n', num2str(Z_target.c'));
fprintf('  Generators: %d\n', size(Z_target.G, 2));
fprintf('  Dimension: %d\n\n', length(Z_target.c));

% Test 1: Sample a point from Z_propagated
fprintf('TEST 1: Sample point from Z_propagated\n');
alpha = 2 * rand(size(Z_propagated.G, 2), 1) - 1;
pt = Z_propagated.c + Z_propagated.G * alpha;
fprintf('  Sample point: [%s]\n', num2str(pt'));

% Test different containment methods
fprintf('\n  Testing containment in Z_target...\n');

% Method 1: contains_ function
try
    result = contains_(Z_target, pt, 'exact', 1e-6);
    fprintf('    Method 1 (contains_ exact): %d\n', result);
catch ME
    fprintf('    Method 1 (contains_ exact): FAILED - %s\n', ME.message);
end

% Method 2: in function
try
    result = in(Z_target, pt, 'exact', 1e-6);
    fprintf('    Method 2 (in exact): %d\n', result);
catch ME
    fprintf('    Method 2 (in exact): FAILED - %s\n', ME.message);
end

% Method 3: Simple contains without options
try
    result = contains_(Z_target, pt);
    fprintf('    Method 3 (contains_ simple): %d\n', result);
catch ME
    fprintf('    Method 3 (contains_ simple): FAILED - %s\n', ME.message);
end

% Method 4: Point zonotope intersection
try
    pt_zono = conPolyZono(pt, [], [], [], [], []);
    intersect_result = Z_target & pt_zono;
    is_empty = isEmptySet(intersect_result);
    fprintf('    Method 4 (intersection): %d (is_empty=%d)\n', ~is_empty, is_empty);
catch ME
    fprintf('    Method 4 (intersection): FAILED - %s\n', ME.message);
end

% Method 5: Bounding box check
try
    I = interval(Z_target);
    lb = infimum(I);
    ub = supremum(I);
    in_bbox = all(pt >= lb) && all(pt <= ub);
    fprintf('    Method 5 (bounding box): %d\n', in_bbox);
    fprintf('      Point: [%s]\n', num2str(pt'));
    fprintf('      Lower: [%s]\n', num2str(lb'));
    fprintf('      Upper: [%s]\n', num2str(ub'));
catch ME
    fprintf('    Method 5 (bounding box): FAILED - %s\n', ME.message);
end

% Test 2: Check Jaccard intersection
fprintf('\n\nTEST 2: Jaccard intersection (should be non-empty)\n');
try
    Z_intersect = Z_propagated & Z_target;
    is_empty_intersect = isEmptySet(Z_intersect);
    fprintf('  Intersection is empty: %d\n', is_empty_intersect);
    
    if ~is_empty_intersect
        fprintf('  Intersection center: [%s]\n', num2str(Z_intersect.c'));
        fprintf('  ✓ Sets DO overlap - MC should find some consistent samples!\n');
    else
        fprintf('  ✗ Sets do NOT overlap\n');
    end
catch ME
    fprintf('  FAILED: %s\n', ME.message);
end

% Test 3: Sample from intersection and check
fprintf('\n\nTEST 3: Sample from intersection and verify containment\n');
try
    Z_intersect = Z_propagated & Z_target;
    if ~isEmptySet(Z_intersect)
        % Sample from intersection
        alpha_int = 2 * rand(size(Z_intersect.G, 2), 1) - 1;
        pt_int = Z_intersect.c + Z_intersect.G * alpha_int;
        fprintf('  Sample from intersection: [%s]\n', num2str(pt_int'));
        
        % This MUST be in Z_target
        try
            result = contains_(Z_target, pt_int);
            fprintf('  In Z_target? %d\n', result);
        catch
            fprintf('  In Z_target? UNKNOWN (contains_ failed)\n');
        end
        
        % Try bounding box
        I = interval(Z_target);
        lb = infimum(I);
        ub = supremum(I);
        in_bbox = all(pt_int >= lb) && all(pt_int <= ub);
        fprintf('  In Z_target bbox? %d\n', in_bbox);
    end
catch ME
    fprintf('  FAILED: %s\n', ME.message);
end

fprintf('\n========================================\n');
fprintf('DIAGNOSIS COMPLETE\n');
fprintf('========================================\n');

% Test 4: Run actual MC probability with fixed method
fprintf('\n\nTEST 4: Run MC probability scoring with 100 samples\n');
try
    [mc_score, mc_details] = score_mc_probability(...
        {Z_propagated, Z_target}, ...
        'num_samples', 100, ...
        'return_details', true, ...
        'verbose', true);
    
    fprintf('\nResults:\n');
    fprintf('  MC Probability: %.4f\n', mc_score);
    fprintf('  Consistent samples: %d / %d\n', mc_details.num_consistent, mc_details.num_samples);
    fprintf('  95%% CI: [%.4f, %.4f]\n', mc_details.ci95_lower, mc_details.ci95_upper);
    
    if mc_score > 0
        fprintf('\n  ✓ SUCCESS! MC sampling is now working!\n');
    else
        fprintf('\n  ✗ Still returning 0 - needs further investigation\n');
    end
catch ME
    fprintf('  FAILED: %s\n', ME.message);
    fprintf('  Stack: %s\n', ME.stack(1).name);
end

fprintf('\n========================================\n');
