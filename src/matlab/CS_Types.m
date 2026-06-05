
classdef CS_Types
    % CS_TYPES Collection of all CS strategy functions as static methods
    % 
    % This class provides all change synchronization (CS) strategy
    % implementations as static methods that can be called from anywhere.
    %
    % Usage:
    %   [Z_safe, Z_new] = CS_Types.parametric_cs(ZB, ZC_old, F, f);
    %   [Z_resolved, Z_safe, k, ZC_all] = CS_Types.disambiguation_cs(...);
    
    methods (Static)
            

            function [ZC_resolved, ZC_resolved_safe, k, ZC_all] = disambiguation_cs(ZB, ZC_old, F_candidates, f_candidates, options)
            % DISAMBIGUATIONCPR Resolves multiple correspondence candidates
            % 
            % ⚠️ SAVE THIS FILE AS: disambiguationCPR.m
            %
            % Syntax:
            %   [ZC_resolved, ZC_resolved_safe, k, ZC_all] = disambiguationCPR(ZB, ZC_old, F_candidates, f_candidates)
            %   [ZC_resolved, ZC_resolved_safe, k, ZC_all] = disambiguationCPR(..., options)
            %
            % Inputs:
            %   ZB            - BrakeDisk source as conPolyZono
            %   ZC_old        - CAD before change as conPolyZono
            %   F_candidates  - Cell array of transformation matrices {F1, F2, F3, ...}
            %   f_candidates  - Cell array of translation vectors {f1, f2, f3, ...}
            %   options       - (optional) struct with fields:
            %                   .user_choice - Force specific candidate (default: [] = auto-select)
            %                   .criterion   - Selection criterion: 'nearest' (default) or 'custom'
            %
            % Outputs:
            %   ZC_resolved      - Selected/resolved CAD as conPolyZono
            %   ZC_resolved_safe - Safe intersection of resolved and old CAD
            %   k                - Index of selected candidate
            %   ZC_all           - Cell array of all candidate conPolyZono objects
            %
            % Example:
            %   % Create conPolyZono objects
            %   cB = [300.0; 28.0];
            %   G_B = [2.0 -0.10; 0.10 0.50];
            %   E = eye(2); A = zeros(0,0); b = zeros(0,1); EC = zeros(2,0);
            %   ZB = conPolyZono(cB, G_B, E, A, b, EC);
            %   
            %   cC0 = [301.0; 27.8];
            %   G_C0 = [2.5 -0.10; 0.10 0.40];
            %   ZC_old = conPolyZono(cC0, G_C0, E, A, b, EC);
            %   
            %   % Define multiple candidates
            %   F_candidates = {[1.005 0; 0 0.995], [0.995 0; 0 1.005], [0.985 0; 0 1.00]};
            %   f_candidates = {[+0.03; -0.03], [-0.02; +0.02], [-0.01; -0.01]};
            %   
            %   % Run disambiguation CPR
            %   [ZC_resolved, ZC_resolved_safe, k, ZC_all] = disambiguationCPR(ZB, ZC_old, F_candidates, f_candidates);
            %   fprintf('Selected candidate: %d\n', k);
            
            % Parse options
            if nargin < 5, options = struct(); end
            user_choice = CS_Types.getOption(options, 'user_choice', []);
            criterion   = CS_Types.getOption(options, 'criterion', 'nearest');
            verbose     = CS_Types.getOption(options, 'verbose', false);
            
            if verbose
                fprintf('   → Applying disambiguation CS\n');
                fprintf('     Number of candidates: %d\n', numel(F_candidates));
                fprintf('     Selection criterion: %s\n', criterion);
            end
            
            % Validate inputs
            if numel(F_candidates) ~= numel(f_candidates)
                error('F_candidates and f_candidates must have the same number of elements');
            end
            
            %% --- 1) Apply all candidate mappings
            n_candidates = numel(F_candidates);
            ZC_all = cell(1, n_candidates);
            
            for i = 1:n_candidates
                ZC_all{i} = CS_Types.affineMap_cPZ(ZB, F_candidates{i}, f_candidates{i});
            end
            
            %% --- 2) Ambiguity resolution
            if isempty(user_choice)
                % Automatic selection based on criterion
                switch lower(criterion)
                    case 'nearest'
                        % Select candidate with center nearest to source center
                        n_dim = size(ZB.c, 1);  % Get actual dimension from ZB
                        cCenters = zeros(n_dim, n_candidates);
                        for i = 1:n_candidates
                            cCenters(:,i) = ZC_all{i}.c;
                        end
                        d = vecnorm(cCenters - ZB.c, 2, 1);
                        [~, k] = min(d);
                        
                    case 'custom'
                        error('Custom criterion not implemented. Please provide user_choice or use "nearest".');
                        
                    otherwise
                        error('Unknown criterion: %s. Use "nearest" or "custom".', criterion);
                end
            else
                % User-specified choice
                k = user_choice;
                if k < 1 || k > n_candidates
                    error('user_choice must be between 1 and %d', n_candidates);
                end
                if verbose
                    fprintf('     User selected candidate: %d\n', k);
                end
            end
            
            if verbose && isempty(user_choice)
                fprintf('     Auto-selected candidate: %d\n', k);
            end
            
            %% --- 3) Extract resolved candidate
            ZC_resolved = ZC_all{k};
            
            if verbose
                fprintf('     Resolved center: [%.3f; %.3f]\n', ZC_resolved.c(1), ZC_resolved.c(2));
            end
            
            %% --- 4) Compute safe intersection
            ZC_resolved_safe = and_(ZC_resolved, ZC_old);
            
            % Empty-set detection after intersection
            if CS_Types.isEmptyCompat(ZC_resolved_safe)
                if verbose
                    fprintf('     ⚠ Safe intersection is empty\n');
                end
                ZC_resolved_safe = conZonotope(zeros(size(ZC_old.c,1),1), zeros(size(ZC_old.c,1),0));
            end
            
            try
                ZC_resolved_safe = reduce(ZC_resolved_safe, 'girard', 200);
                
                % Empty-set detection after reduction
                if CS_Types.isEmptyCompat(ZC_resolved_safe)
                    if verbose
                        fprintf('     ⚠ Safe region became empty after reduction\n');
                    end
                    ZC_resolved_safe = conZonotope(zeros(size(ZC_old.c,1),1), zeros(size(ZC_old.c,1),0));
                end
                
                if verbose, fprintf('     ✓ Safe region reduced\n'); end
            catch
                if verbose, fprintf('     ⚠ Reduction skipped\n'); end
            end
            
            if verbose
                fprintf('   ✓ Disambiguation CS complete\n');
            end
            
            end
            
            
            
            
            
            
            function [ZC_V_new, ZC_V_new_safe] = conditional_cs(ZB, ZC_V_old, F, f, options)
            % conditional_cs Performs conditional Change Propagation (ventilated flag)
            %
            % Syntax:
            %   [ZC_V_new, ZC_V_new_safe] = conditionalCPR(ZB, ZC_V_old, F, f)
            %   [ZC_V_new, ZC_V_new_safe] = conditionalCPR(ZB, ZC_V_old, F, f, options)
            %
            % Inputs:
            %   ZB        - BrakeDisk source as conPolyZono
            %   ZC_V_old  - CAD before change as conPolyZono
            %   F         - Transformation matrix (2x2)
            %   f         - Translation vector (2x1)
            %   options   - (optional) struct with fields:
            %               .addNoise - Add noise to propagated result (default: false)
            %               .noiseG   - Noise generator matrix (default: [0.10 0; 0 0.05])
            %
            % Outputs:
            %   ZC_V_new      - Propagated CAD (ventilated = true) as conPolyZono
            %   ZC_V_new_safe - Safe intersection of new and old CAD as conPolyZono
            %
            % Example:
            %   % Create conPolyZono objects
            %   cB = [300.0; 28.0];
            %   G_B = [2.0 -0.10; 0.10 0.50];
            %   E_B = eye(2); A_B = zeros(0,0); b_B = zeros(0,1); EC_B = zeros(2,0);
            %   ZB = conPolyZono(cB, G_B, E_B, A_B, b_B, EC_B);
            %   
            %   cC0 = [301.0; 27.8];
            %   G_C0 = [2.5 -0.10; 0.10 0.40];
            %   E_C0 = eye(2); A_C0 = zeros(0,0); b_C0 = zeros(0,1); EC_C0 = zeros(2,0);
            %   ZC_V_old = conPolyZono(cC0, G_C0, E_C0, A_C0, b_C0, EC_C0);
            %   
            %   % Define transformation
            %   F = 0.99*eye(2);
            %   f = [1.2; 0.4];
            %   
            %   % Run conditional CPR
            %   [ZC_V_new, ZC_V_new_safe] = conditionalCPR(ZB, ZC_V_old, F, f);
            
            % Parse options
            if nargin < 5, options = struct(); end
            addNoise = CS_Types.getOption(options, 'addNoise', false);
            noiseG   = CS_Types.getOption(options, 'noiseG', [0.10 0; 0 0.05]);
            verbose  = CS_Types.getOption(options, 'verbose', false);
            
            if verbose
                fprintf('   → Applying conditional CS\n');
                fprintf('     Add noise: %s\n', string(addNoise));
            end
            
            %% --- Mapping for ventilated == true
            ZC_V_new = CS_Types.affineMap_cPZ(ZB, F, f);

            if verbose
                fprintf('     Propagated center: [%.3f; %.3f]\n', ZC_V_new.c(1), ZC_V_new.c(2));
            end

            % Optionally add noise
            if addNoise
                W = conZonotope([0;0], noiseG);
                ZC_V_new = plus(ZC_V_new, W);
                if verbose
                    fprintf('     ✓ Noise added\n');
                end
            end
            
            %% --- Compute safe intersection
            ZC_V_new_safe = and_(ZC_V_new, ZC_V_old);
            try
                ZC_V_new_safe = reduce(ZC_V_new_safe, 'girard', 200);
                if verbose, fprintf('     ✓ Safe region reduced\n'); end
            catch
                if verbose, fprintf('     ⚠ Reduction skipped\n'); end
            end
            
            if verbose
                fprintf('   ✓ Conditional CS complete\n');
            end
            
            end
            
            
            
           function [Y_safe, Y_viol_act] = constraintBased_cs(ZB, ZC_old, d_max, F_act, f_act, options)
            % constraintBased_cs — Constraint-based Change Propagation with sync (y := x)
            %
            % Inputs:
            %   ZB, ZC_old : conPolyZono (2D, interpreted as [d_req; r_i])
            %   d_max      : scalar threshold on first coord (d_req)
            %   F_act      : 2x2 action matrix
            %   f_act      : 2x1 action offset
            %   options.epsSplit (optional, default 1e-9)
            %
            % Outputs:
            %   Y_safe     : conPolyZono (unchanged part after check)
            %   Y_viol_act : conPolyZono (violating part after action)
            
                if nargin < 6, options = struct(); end
                epsSplit = CS_Types.getOption(options, 'epsSplit', 1e-9);
                verbose  = CS_Types.getOption(options, 'verbose', false);
                if nargin < 5 || isempty(f_act), f_act = zeros(size(F_act,1),1); end
            
                if verbose
                    fprintf('   → Applying constraint-based CS\n');
                    fprintf('     Constraint: d_req <= %.3f\n', d_max);
                    fprintf('     Action matrix F_act applied to violations\n');
                end
            
                % ---- 1) Build joint space with shared generator vector (beta)
                nB = size(ZB.c,1);         % Source dimension
                nC = size(ZC_old.c,1);     % Target dimension
            
                mB = size(ZB.G,2);
                mC = size(ZC_old.G,2);
                m  = max(mB, mC);
            
                % Pad generators so both have m columns
                GB = ZB.G;  GC = ZC_old.G;
                if mB < m, GB = [GB, zeros(nB, m-mB)]; end
                if mC < m, GC = [GC, zeros(nC, m-mC)]; end
            
                c_joint = [ZB.c;     ZC_old.c];
                G_joint = [GB;       GC];
            
                A  = [];  b  = [];
                Z_joint = conZonotope(c_joint, G_joint, A, b);
            
                % ---- 2) Sync CAD to BrakeDisk: y := x (copy first nB dimensions)
                % State of Z_joint is [x; y], x∈R^nB from ZB, y∈R^nC from ZC_old.
                % Build selection matrix to extract first nB dimensions
                S_copy = [eye(nB), zeros(nB, nC)];
                Y_sync = CS_Types.affineMap_cPZ(Z_joint, S_copy, zeros(nB, 1)); % R^nB
            
                % ---- 3) Check constraint and split on first dimension
                % Safe: first dimension <= d_max
                h_safe = zeros(nB, 1);
                h_safe(1) = 1;
                H_safe = polytope(h_safe', d_max - epsSplit);
                
                % Violating: first dimension >= d_max
                h_viol = zeros(nB, 1);
                h_viol(1) = -1;
                H_viol = polytope(h_viol', -(d_max + epsSplit));
            
                Y_safe = and_(Y_sync, H_safe);
                Y_viol = and_(Y_sync, H_viol);
                
                % Empty-set detection after halfspace splitting
                if CS_Types.isEmptyCompat(Y_safe)
                    if verbose
                        fprintf('     ⚠ Safe region is empty (all values violate constraint)\n');
                    end
                    Y_safe = conZonotope(zeros(nB,1), zeros(nB,0));
                end
                
                if CS_Types.isEmptyCompat(Y_viol)
                    if verbose
                        fprintf('     ✓ Violating region is empty (all values satisfy constraint)\n');
                    end
                    Y_viol = conZonotope(zeros(nB,1), zeros(nB,0));
                end
            
                if verbose
                    fprintf('     Split into safe and violating regions\n');
                end
            
                % ---- 4) Reduction (optional, robust to failures)
                try
                    Y_safe = reduce(Y_safe, 'girard', 200);
                    
                    % Empty-set detection after reduction
                    if CS_Types.isEmptyCompat(Y_safe)
                        if verbose
                            fprintf('     ⚠ Safe region became empty after reduction\n');
                        end
                        Y_safe = conZonotope(zeros(nB,1), zeros(nB,0));
                    end
                    
                    if verbose, fprintf('     ✓ Safe region reduced\n'); end
                catch
                    if verbose, fprintf('     ⚠ Safe reduction skipped\n'); end
                end
                
                try
                    Y_viol = reduce(Y_viol, 'girard', 200);
                    
                    % Empty-set detection after reduction
                    if CS_Types.isEmptyCompat(Y_viol)
                        if verbose
                            fprintf('     ⚠ Violating region became empty after reduction\n');
                        end
                        Y_viol = conZonotope(zeros(nB,1), zeros(nB,0));
                    end
                    
                    if verbose, fprintf('     ✓ Violating region reduced\n'); end
                catch
                    if verbose, fprintf('     ⚠ Violating reduction skipped\n'); end
                end
            
                % ---- 5) Act ONLY on violating part
                Y_viol_act = CS_Types.affineMap_cPZ(Y_viol, F_act, f_act);
                
                if verbose
                    fprintf('     ✓ Action applied to violating region\n');
                    fprintf('   ✓ Constraint-based CS complete\n');
                end
            end

            
       
            
            function [ZC_safe, ZC_new] = parametric_cs(ZB, ZC_old, F, f, verbose)
            % parametric_cs
            % Propagates a source conPolyZono ZB via affine mapping (F,f),
            % replaces the target set (ZC_old), and computes the intersection.
            %
            % Inputs:
            %   ZB      - Source conPolyZono
            %   ZC_old  - Target conPolyZono (existing)
            %   F, f    - Affine mapping (ZC_new = F * ZB + f)
            %   verbose - (optional) Display progress messages (default: false)
            %
            % Outputs:
            %   ZC_safe - Intersection of old and new CAD sets
            %   ZC_new  - New propagated CAD set
            %
            % Example:
            %   [ZC_safe, ZC_new] = CS_Types.parametric_cs(ZB, ZC_old, F, f, true);
            
                if nargin < 5, verbose = false; end
                if nargin < 4 || isempty(f)
                    f = zeros(size(F,1),1);
                end
            
                if verbose
                    fprintf('   → Applying parametric transformation\n');
                    fprintf('     F = [%.4f %.4f; %.4f %.4f]\n', F(1,1), F(1,2), F(2,1), F(2,2));
                    fprintf('     f = [%.4f; %.4f]\n', f(1), f(2));
                    fprintf('     Source center: [%.3f; %.3f]\n', ZB.c(1), ZB.c(2));
                end
            
                % --- Propagation step
                ZC_new = CS_Types.affineMap_cPZ(ZB, F, f);
                
                if verbose
                    fprintf('     Propagated center: [%.3f; %.3f]\n', ZC_new.c(1), ZC_new.c(2));
                end
            
                % --- Intersection (safe zone)
                ZC_safe = and_(ZC_old, ZC_new);
                
                % Empty-set detection after intersection
                if CS_Types.isEmptyCompat(ZC_safe)
                    if verbose
                        fprintf('     ⚠ Intersection is empty (incompatible models)\n');
                    end
                    ZC_safe = conZonotope(zeros(size(ZC_old.c,1),1), zeros(size(ZC_old.c,1),0));
                end
                
                % Additional check: if and_() returned one of the inputs unchanged,
                % but the inputs are far apart, it indicates no real intersection
                dist_inputs = norm(ZC_old.c - ZC_new.c);
                max_gen_old = sum(sum(abs(ZC_old.G)));
                max_gen_new = sum(sum(abs(ZC_new.G)));
                max_combined_reach = max_gen_old + max_gen_new;
                
                % If inputs are far apart (>3× combined reach) but intersection succeeded,
                % check if result is just one of the inputs (indicates and_() failure)
                if dist_inputs > 3 * max_combined_reach
                    dist_to_old = norm(ZC_safe.c - ZC_old.c);
                    dist_to_new = norm(ZC_safe.c - ZC_new.c);
                    
                    % If result matches one input exactly, and_() likely failed to detect empty
                    if dist_to_old < 0.1 || dist_to_new < 0.1
                        if verbose
                            fprintf('     ⚠ Intersection appears empty (geometric check)\n');
                            fprintf('       Input separation: %.1f mm, combined reach: %.1f mm\n', ...
                                dist_inputs, max_combined_reach);
                        end
                        ZC_safe = conZonotope(zeros(size(ZC_old.c,1),1), zeros(size(ZC_old.c,1),0));
                    end
                end
            
                % --- Optional reduction for clarity
                try
                    ZC_safe = reduce(ZC_safe, 'girard', 200);
                    
                    % Empty-set detection after reduction
                    if CS_Types.isEmptyCompat(ZC_safe)
                        if verbose
                            fprintf('     ⚠ Set became empty after reduction\n');
                        end
                        ZC_safe = conZonotope(zeros(size(ZC_old.c,1),1), zeros(size(ZC_old.c,1),0));
                    end
                    
                    if verbose, fprintf('     ✓ Safe region reduced (girard, order 200)\n'); end
                catch ME
                    if verbose, fprintf('     ⚠ Reduction skipped (not supported)\n'); end
                end
                
                if verbose
                    fprintf('   ✓ Parametric CS complete\n');
                end
            end
            
            
            function [Z_D_safe, Z_T_safe] = structural_cs(ZB, F_D, f_D, F_T, f_T, varargin)
            % cPZ_param_div_intersections
            %   Apply parameter-specific affine divergences to a base conPolyZono ZB and
            %   return only the intersections with ZB (safe regions).
            %
            %   Z_D_new = F_D * ZB + f_D
            %   Z_T_new = F_T * ZB + f_T
            %   Z_D_safe = and_(Z_D_new, ZB)
            %   Z_T_safe = and_(Z_T_new, ZB)
            %
            % INPUTS
            %   ZB   : conPolyZono (base BrakeDisk set)
            %   F_D  : 2x2 double (divergence map for Diameter)
            %   f_D  : 2x1 double (offset for Diameter)
            %   F_T  : 2x2 double (divergence map for MinThickness)
            %   f_T  : 2x1 double (offset for MinThickness)
            %
            % Name-Value (optional)
            %   'DoReduce'  (logical, default=true)  -> apply reduce(...,'girard',200)
            %   'ReduceOrd' (double,  default=200)   -> reduction order
            %   'Verbose'   (logical, default=false) -> display progress messages
            %
            % OUTPUTS
            %   Z_D_safe : conPolyZono, intersection of ZB and Z_D_new
            %   Z_T_safe : conPolyZono, intersection of ZB and Z_T_new
            
                p = inputParser;
                addParameter(p, 'DoReduce',  true,  @(x) islogical(x) || isnumeric(x));
                addParameter(p, 'ReduceOrd', 200,   @(x) isnumeric(x) && isscalar(x));
                addParameter(p, 'Verbose',   false, @(x) islogical(x) || isnumeric(x));
                parse(p, varargin{:});
                doReduce  = logical(p.Results.DoReduce);
                reduceOrd = p.Results.ReduceOrd;
                verbose   = logical(p.Results.Verbose);
                
                if verbose
                    fprintf('   → Applying structural CS (parameter-specific divergences)\n');
                    fprintf('     Diameter divergence: F_D, f_D\n');
                    fprintf('     Thickness divergence: F_T, f_T\n');
                end
            
                % Diverged sets
                Z_D_new = CS_Types.affineMap_cPZ(ZB, F_D, f_D);
                Z_T_new = CS_Types.affineMap_cPZ(ZB, F_T, f_T);
                
                if verbose
                    fprintf('     Diameter diverged center: [%.3f; %.3f]\n', Z_D_new.c(1), Z_D_new.c(2));
                    fprintf('     Thickness diverged center: [%.3f; %.3f]\n', Z_T_new.c(1), Z_T_new.c(2));
                end
            
                % Intersections with base
                Z_D_safe = and_(Z_D_new, ZB);
                Z_T_safe = and_(Z_T_new, ZB);
                
                % Empty-set detection after intersections
                if CS_Types.isEmptyCompat(Z_D_safe)
                    if verbose
                        fprintf('     ⚠ Diameter safe region is empty\n');
                    end
                    Z_D_safe = conZonotope(zeros(size(ZB.c,1),1), zeros(size(ZB.c,1),0));
                end
                
                if CS_Types.isEmptyCompat(Z_T_safe)
                    if verbose
                        fprintf('     ⚠ Thickness safe region is empty\n');
                    end
                    Z_T_safe = conZonotope(zeros(size(ZB.c,1),1), zeros(size(ZB.c,1),0));
                end
            
                % Optional reduction
                if doReduce
                    try
                        Z_D_safe = reduce(Z_D_safe, 'girard', reduceOrd);
                        
                        % Empty-set detection after reduction
                        if CS_Types.isEmptyCompat(Z_D_safe)
                            if verbose
                                fprintf('     ⚠ Diameter safe region became empty after reduction\n');
                            end
                            Z_D_safe = conZonotope(zeros(size(ZB.c,1),1), zeros(size(ZB.c,1),0));
                        end
                        
                        if verbose, fprintf('     ✓ Diameter safe region reduced\n'); end
                    catch
                        if verbose, fprintf('     ⚠ Diameter reduction skipped\n'); end
                    end
                    
                    try
                        Z_T_safe = reduce(Z_T_safe, 'girard', reduceOrd);
                        
                        % Empty-set detection after reduction
                        if CS_Types.isEmptyCompat(Z_T_safe)
                            if verbose
                                fprintf('     ⚠ Thickness safe region became empty after reduction\n');
                            end
                            Z_T_safe = conZonotope(zeros(size(ZB.c,1),1), zeros(size(ZB.c,1),0));
                        end
                        
                        if verbose, fprintf('     ✓ Thickness safe region reduced\n'); end
                    catch
                        if verbose, fprintf('     ⚠ Thickness reduction skipped\n'); end
                    end
                end
                
                if verbose
                    fprintf('   ✓ Structural CS complete\n');
                end
            end
            
            
            function [ZB_safe, ZC_safe] = bidirectional_cs(ZB_old, ZC_old, verbose)
            % cPZ_bidirectional_intersect
            %   Performs a bidirectional identity swap between two conPolyZonos
            %   and returns only the intersections ("safe" reconciliations):
            %
            %       ZC_new = ZB_old
            %       ZB_new = ZC_old
            %
            %       ZC_safe = and_(ZC_old, ZC_new)
            %       ZB_safe = and_(ZB_old, ZB_new)
            %
            % INPUTS
            %   ZB_old  : conPolyZono  (e.g., BrakeDisk old)
            %   ZC_old  : conPolyZono  (e.g., CAD old)
            %   verbose : (optional) Display progress messages (default: false)
            %
            % OUTPUTS
            %   ZB_safe : conPolyZono intersection of old/new BrakeDisk
            %   ZC_safe : conPolyZono intersection of old/new CAD
            %
            % DEPENDENCIES
            %   Requires CORA toolbox (for conPolyZono, and_, reduce, etc.)
            
                if nargin < 3, verbose = false; end
                
                % Get actual dimension from input
                n_dim = size(ZB_old.c, 1);
                
                if verbose
                    fprintf('   → Applying bidirectional identity swap\n');
                    if n_dim <= 4
                        fprintf('     Source center: [%s]\n', sprintf('%.3f; ', ZB_old.c));
                        fprintf('     Target center: [%s]\n', sprintf('%.3f; ', ZC_old.c));
                    else
                        fprintf('     Source center: [%dD vector]\n', n_dim);
                        fprintf('     Target center: [%dD vector]\n', n_dim);
                    end
                end
                
                % --- Identity mapping (swap) - dimension-aware ---
                F = eye(n_dim);
                f = zeros(n_dim, 1);
                ZC_new = CS_Types.affineMap_cPZ(ZB_old, F, f);   % CAD <- old BrakeDisk
                ZB_new = CS_Types.affineMap_cPZ(ZC_old, F, f);   % BrakeDisk <- old CAD
            
            % --- Compute intersections ---
            ZC_safe = and_(ZC_old, ZC_new);
            ZB_safe = and_(ZB_old, ZB_new);
            
            % Empty-set detection after intersections
            if CS_Types.isEmptyCompat(ZC_safe)
                if verbose
                    fprintf('     ⚠ Target intersection is empty\n');
                end
                ZC_safe = conZonotope(zeros(size(ZC_old.c,1),1), zeros(size(ZC_old.c,1),0));
            else
                % Geometric separation check for ZC (old vs new)
                dist_ZC = norm(ZC_old.c - ZC_new.c);
                reach_old = sum(sum(abs(ZC_old.G)));
                reach_new = sum(sum(abs(ZC_new.G)));
                combined_reach_ZC = reach_old + reach_new;
                
                if dist_ZC > 3 * combined_reach_ZC
                    if verbose
                        fprintf('     ⚠ Target sets too far apart (dist=%.3f, reach=%.3f)\n', dist_ZC, combined_reach_ZC);
                    end
                    ZC_safe = conZonotope(zeros(size(ZC_old.c,1),1), zeros(size(ZC_old.c,1),0));
                end
            end
            
            if CS_Types.isEmptyCompat(ZB_safe)
                if verbose
                    fprintf('     ⚠ Source intersection is empty\n');
                end
                ZB_safe = conZonotope(zeros(size(ZB_old.c,1),1), zeros(size(ZB_old.c,1),0));
            else
                % Geometric separation check for ZB (old vs new)
                dist_ZB = norm(ZB_old.c - ZB_new.c);
                reach_old = sum(sum(abs(ZB_old.G)));
                reach_new = sum(sum(abs(ZB_new.G)));
                combined_reach_ZB = reach_old + reach_new;
                
                if dist_ZB > 3 * combined_reach_ZB
                    if verbose
                        fprintf('     ⚠ Source sets too far apart (dist=%.3f, reach=%.3f)\n', dist_ZB, combined_reach_ZB);
                    end
                    ZB_safe = conZonotope(zeros(size(ZB_old.c,1),1), zeros(size(ZB_old.c,1),0));
                end
            end                % --- Optional reduction (for compactness, not required) ---
                try
                    ZC_safe = reduce(ZC_safe, 'girard', 200);
                    
                    % Empty-set detection after reduction
                    if CS_Types.isEmptyCompat(ZC_safe)
                        if verbose
                            fprintf('     ⚠ Target became empty after reduction\n');
                        end
                        ZC_safe = conZonotope(zeros(size(ZC_old.c,1),1), zeros(size(ZC_old.c,1),0));
                    end
                    
                    if verbose, fprintf('     ✓ Target safe region reduced\n'); end
                catch ME
                    if verbose, fprintf('     ⚠ Target reduction skipped\n'); end
                end
                
                try
                    ZB_safe = reduce(ZB_safe, 'girard', 200);
                    
                    % Empty-set detection after reduction
                    if CS_Types.isEmptyCompat(ZB_safe)
                        if verbose
                            fprintf('     ⚠ Source became empty after reduction\n');
                        end
                        ZB_safe = conZonotope(zeros(size(ZB_old.c,1),1), zeros(size(ZB_old.c,1),0));
                    end
                    
                    if verbose, fprintf('     ✓ Source safe region reduced\n'); end
                catch ME
                    if verbose, fprintf('     ⚠ Source reduction skipped\n'); end
                end
                
                if verbose
                    fprintf('   ✓ Bidirectional CS complete\n');
                end
            end
            
            
            
            function [Z_propagated, Z_safe] = identity_cs(Z_source, Z_existing, verbose)
            % identity_cs
            %   Propagate a source conPolyZono via identity (F=I, f=0)
            %   and intersect with an existing conPolyZono.
            %
            % INPUTS
            %   Z_source   : conPolyZono (e.g., BrakeDisk)
            %   Z_existing : conPolyZono (e.g., existing CAD)
            %   verbose    : (optional) Display progress messages (default: false)
            %
            % OUTPUTS
            %   Z_propagated : identity-propagated set from Z_source
            %   Z_safe       : intersection(Z_source, Z_existing) reduced if possible
            %
            % DEPENDENCIES
            %   Requires CORA toolbox (for conPolyZono operations).
            
                if nargin < 3, verbose = false; end
                
                % Get actual dimension from input
                n_dim = size(Z_source.c, 1);
                
                if verbose
                    fprintf('   → Applying identity transformation (F=I, f=0)\n');
                    if n_dim <= 4
                        fprintf('     Source center: [%s]\n', sprintf('%.3f; ', Z_source.c));
                    else
                        fprintf('     Source center: [%dD vector]\n', n_dim);
                    end
                end
                
                % ---- 1) Propagation (identity map) - dimension-aware ----
                F = eye(n_dim);
                f = zeros(n_dim, 1);
                Z_propagated = CS_Types.affineMap_cPZ(Z_source, F, f);
            
                % ---- 2) Intersection and reduction ----
                Z_safe = and_(Z_source, Z_existing);
                
                % Empty-set detection after intersection
                if CS_Types.isEmptyCompat(Z_safe)
                    if verbose
                        fprintf('     ⚠ Intersection is empty (incompatible models)\n');
                    end
                    Z_safe = conZonotope(zeros(n_dim,1), zeros(n_dim,0));
                end
                
                % Additional check: if inputs are far apart but intersection succeeded
                dist_inputs = norm(Z_source.c - Z_existing.c);
                max_gen_source = sum(sum(abs(Z_source.G)));
                max_gen_existing = sum(sum(abs(Z_existing.G)));
                max_combined_reach = max_gen_source + max_gen_existing;
                
                if dist_inputs > 3 * max_combined_reach
                    dist_to_source = norm(Z_safe.c - Z_source.c);
                    dist_to_existing = norm(Z_safe.c - Z_existing.c);
                    
                    if dist_to_source < 0.1 || dist_to_existing < 0.1
                        if verbose
                            fprintf('     ⚠ Intersection appears empty (geometric check)\n');
                        end
                        Z_safe = conZonotope(zeros(n_dim,1), zeros(n_dim,0));
                    end
                end
                
                try
                    Z_safe = reduce(Z_safe, 'girard', 200);
                    
                    % Empty-set detection after reduction
                    if CS_Types.isEmptyCompat(Z_safe)
                        if verbose
                            fprintf('     ⚠ Set became empty after reduction\n');
                        end
                        Z_safe = conZonotope(zeros(n_dim,1), zeros(n_dim,0));
                    end
                    
                    if verbose, fprintf('     ✓ Safe region reduced\n'); end
                catch ME
                    if verbose, fprintf('     ⚠ Reduction skipped\n'); end
                end
                
                if verbose
                    fprintf('   ✓ Identity CS complete\n');
                end
            end
            
            
            
    end

    methods (Static)

        function P_C = affineMap_cPZ(P_B, F, f)
            % AFFINEMAP_CPZ Apply affine transformation to conPolyZono
            %
            % Syntax:
            %   P_C = affineMap_cPZ(P_B, F, f)
            %
            % Inputs:
            %   P_B - Input conPolyZono
            %   F   - Transformation matrix (n_out × n_state)
            %   f   - Translation vector (n_out × 1), optional
            %
            % Output:
            %   P_C - Transformed conPolyZono: P_C = F*P_B + f
            %
            % Example:
            %   P_C = CS_Types.affineMap_cPZ(P_B, [1.005 0; 0 1], [0.1; 0]);
            
            if nargin < 3 || isempty(f), f = zeros(size(F,1),1); end
            
            % Validate dimensions
            n_state = size(P_B.c, 1);
            assert(size(F, 2) == n_state, ...
                'CS_Types:affineMap_cPZ:DimensionMismatch', ...
                'F must have %d columns to match state dimension (has %d)', ...
                n_state, size(F, 2));
            
            assert(size(f, 1) == size(F, 1), ...
                'CS_Types:affineMap_cPZ:TranslationMismatch', ...
                'Translation vector f must have %d rows to match F output dimension (has %d)', ...
                size(F, 1), size(f, 1));
            
            assert(size(f, 2) == 1, ...
                'CS_Types:affineMap_cPZ:TranslationNotVector', ...
                'Translation f must be a column vector (has %d columns)', ...
                size(f, 2));
            
            % Apply affine transformation
            try
                A_out = P_B.A;
                b_out = P_B.b;
            catch
                A_out = [];
                b_out = [];
            end
            P_C = conZonotope(F*P_B.c + f, F*P_B.G, A_out, b_out);
        end
    
        function val = getOption(opts, field, default)
                % Get option value or return default
                if isfield(opts, field)
                    val = opts.(field);
                else
                    val = default;
                end
         end
         
         function isEmpty = isEmptyCompat(Z)
            % isEmptyCompat — Multi-layer empty-set detection for conPolyZono
            %
            % Combines representsa(), interval check, and geometric validation
            % to robustly detect empty sets even when and_() doesn't explicitly
            % return an empty set object.
            %
            % Input:
            %   Z : conPolyZono object
            %
            % Output:
            %   isEmpty : true if Z is determined to be empty, false otherwise
            
            isEmpty = false;
            
            % Layer 1: Standard CORA check
            try
                if representsa(Z, 'emptySet')
                    isEmpty = true;
                    return;
                end
            catch
                % representsa might fail on malformed objects
            end
            
            % Layer 2: Check for our explicit empty set marker (center at origin, no generators)
            try
                if norm(Z.c) < 1e-10 && size(Z.G, 2) == 0
                    isEmpty = true;
                    return;
                end
            catch
                % Marker check might fail
            end
            
            % Layer 3: Interval check (convert to interval and check bounds)
            try
                I = interval(Z);
                if any(infimum(I) > supremum(I))
                    isEmpty = true;
                    return;
                end
            catch
                % Interval conversion might fail
            end
            
            % Layer 4: Check if set has zero volume (all generators ~0)
            try
                if size(Z.G, 2) > 0
                    total_gen = sum(sum(abs(Z.G)));
                    if total_gen < 1e-10  % Essentially no generators = point or empty
                        isEmpty = true;
                        return;
                    end
                end
            catch
                % Generator check might fail
            end
         end
    end
end