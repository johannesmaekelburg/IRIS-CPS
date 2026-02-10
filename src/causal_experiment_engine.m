classdef causal_experiment_engine
    %CAUSAL_EXPERIMENT_ENGINE MATLAB backend for causal uncertainty-inconsistency experiments
    %
    % This class provides intervention mechanisms and measurement tools
    % to study the causal relationship between uncertainty and inconsistency
    % using the existing CPS framework (CS_Types).
    %
    % DESIGN: Wrapper around CS_Types to enable causal inference experiments
    % - Uses CS_Types.affineMap_cPZ for propagation
    % - Uses CS_Types intersection operations (and_)
    % - Uses isEmptySet from CPS framework
    %
    % CAUSAL PATHWAYS STUDIED:
    %   Uncertainty → Inconsistency (Forward Direction)
    %      - Interventions: widen, shrink, shift, rotate, correlate
    %      - Measure: Δ-inconsistency (delta_jaccard, delta_empty)
    %
    % USAGE:
    %   engine = causal_experiment_engine();
    %   result = engine.run_intervention(scenario_id, intervention_type, params);
    %
    % See also: CS_Types, integrated_cs_propagation, isEmptySet
    
    properties (Constant)
        % Forward direction: Uncertainty → Inconsistency
        UNCERTAINTY_INTERVENTIONS = {'widen', 'shrink', 'shift', 'rotate', 'correlate'};
        
        % Consistency scoring methods
        CONSISTENCY_METHODS = {'jaccard', 'mc_probability', 'both'};
        DEFAULT_METHOD = 'both';  % Use both methods by default
        DEFAULT_MC_SAMPLES = 2000;  % Monte Carlo samples
    end
    
    methods (Static)
        
        function result = run_intervention(scenario, intervention, params, options)
            %RUN_INTERVENTION Execute single causal intervention and measure effects
            %
            % Inputs:
            %   scenario      - Struct with .source, .target, .mapping fields
            %   intervention  - String: intervention type (see constants above)
            %   params        - Struct with intervention-specific parameters
            %   options       - (optional) Struct with:
            %       .consistency_method - 'jaccard', 'mc_probability', or 'both' (default: 'both')
            %       .mc_samples - Number of Monte Carlo samples (default: 2000)
            %
            % Outputs:
            %   result - Struct with:
            %       .pre_state      - Measurements before intervention
            %       .post_state     - Measurements after intervention
            %       .intervention   - Applied intervention details
            %       .causal_effect  - Delta measurements (post - pre)
            
            % Parse options
            if nargin < 4
                options = struct();
            end
            if ~isfield(options, 'consistency_method')
                options.consistency_method = causal_experiment_engine.DEFAULT_METHOD;
            end
            if ~isfield(options, 'mc_samples')
                options.mc_samples = causal_experiment_engine.DEFAULT_MC_SAMPLES;
            end
            
            % Measure baseline state
            result.pre_state = causal_experiment_engine.measure_state(scenario, options);
            
            % Apply intervention
            scenario_modified = causal_experiment_engine.apply_intervention(...
                scenario, intervention, params);
            
            % Measure post-intervention state
            result.post_state = causal_experiment_engine.measure_state(scenario_modified, options);
            
            % Compute causal effect
            result.causal_effect = causal_experiment_engine.compute_delta(...
                result.pre_state, result.post_state);
            
            % Store intervention metadata
            result.intervention = struct(...
                'type', intervention, ...
                'params', params, ...
                'consistency_method', options.consistency_method, ...
                'timestamp', datetime('now'));
        end
        
        function state = measure_state(scenario, options)
            %MEASURE_STATE Compute all observables for causal analysis
            %
            % Measures both uncertainty metrics and inconsistency metrics
            % Supports multiple consistency scoring methods
            %
            % Inputs:
            %   scenario - Struct with .source, .target, .mapping
            %   options  - (optional) Struct with .consistency_method, .mc_samples
            %
            % Outputs:
            %   state - Struct with:
            %       .uncertainty    - Volume, radius, entropy, correlation
            %       .inconsistency  - Emptiness, constraint violation, overlap
            %                        - Jaccard scores (C, Csym) if method='jaccard' or 'both'
            %                        - MC probability if method='mc_probability' or 'both'
            
            if nargin < 2
                options = struct();
            end
            if ~isfield(options, 'consistency_method')
                options.consistency_method = causal_experiment_engine.DEFAULT_METHOD;
            end
            if ~isfield(options, 'mc_samples')
                options.mc_samples = causal_experiment_engine.DEFAULT_MC_SAMPLES;
            end
            
            state = struct();
            
            % === UNCERTAINTY METRICS ===
            state.uncertainty = struct();
            
            % Source uncertainty
            if isfield(scenario, 'source') && isobject(scenario.source)
                Z_src = scenario.source;
                state.uncertainty.source_volume = causal_experiment_engine.compute_volume(Z_src);
                state.uncertainty.source_radius = causal_experiment_engine.compute_radius(Z_src);
                state.uncertainty.source_center = Z_src.c;
                state.uncertainty.source_n_generators = size(Z_src.G, 2);
                
                % Correlation measure (if multiple generators)
                if size(Z_src.G, 2) > 1
                    state.uncertainty.source_correlation = ...
                        causal_experiment_engine.compute_correlation(Z_src.G);
                else
                    state.uncertainty.source_correlation = 0;
                end
            end
            
            % Target uncertainty (if exists)
            if isfield(scenario, 'target') && isobject(scenario.target)
                Z_tgt = scenario.target;
                state.uncertainty.target_volume = causal_experiment_engine.compute_volume(Z_tgt);
                state.uncertainty.target_radius = causal_experiment_engine.compute_radius(Z_tgt);
                state.uncertainty.target_center = Z_tgt.c;
                state.uncertainty.target_n_generators = size(Z_tgt.G, 2);
            end
            
            % === INCONSISTENCY METRICS ===
            state.inconsistency = struct();
            
            % Propagate and check for inconsistencies
            if isfield(scenario, 'mapping') && isfield(scenario, 'source') && isobject(scenario.source)
                try
                    % Apply mapping to source
                    F = scenario.mapping.F;
                    f = scenario.mapping.f;
                    Z_propagated = CS_Types.affineMap_cPZ(scenario.source, F, f);
                    
                    state.inconsistency.propagation_success = true;
                    state.inconsistency.propagated_volume = ...
                        causal_experiment_engine.compute_volume(Z_propagated);
                    
                    % Check if propagated set is empty (use CPS framework function)
                    state.inconsistency.is_empty = isEmptySet(Z_propagated);
                    
                    % If target exists, use consistency scoring addon(s)
                    if isfield(scenario, 'target') && isobject(scenario.target)
                        method = lower(options.consistency_method);
                        
                        % === JACCARD METHOD ===
                        if strcmp(method, 'jaccard') || strcmp(method, 'both')
                            try
                                F_identity = eye(size(scenario.target.c, 1));
                                f_identity = zeros(size(scenario.target.c, 1), 1);
                                
                                [C, Csym, details] = score_jaccard(...
                                    Z_propagated, scenario.target, F_identity, f_identity, ...
                                    struct('return_details', true, 'verbose', false));
                                
                                % Store Jaccard-based scores
                                state.inconsistency.jaccard_C = C;               % Directional
                                state.inconsistency.jaccard_Csym = Csym;         % Symmetric
                                state.inconsistency.jaccard_index = Csym;        % Alias for compatibility
                                state.inconsistency.empty_intersection = details.is_empty;
                                state.inconsistency.has_intersection = ~details.is_empty;
                                
                                % Detailed Jaccard components
                                state.inconsistency.jaccard_vol_intersection = details.s_int;
                                state.inconsistency.jaccard_vol_union = details.s_prop + details.s_tgt - details.s_int;
                                
                                % Legacy volume (for backward compatibility)
                                if ~details.is_empty
                                    Z_intersect = Z_propagated & scenario.target;
                                    state.inconsistency.intersection_volume = ...
                                        causal_experiment_engine.compute_volume(Z_intersect);
                                else
                                    state.inconsistency.intersection_volume = 0;
                                end
                                
                            catch ME
                                warning('Jaccard scoring failed: %s', ME.message);
                                state.inconsistency.jaccard_C = NaN;
                                state.inconsistency.jaccard_Csym = NaN;
                                state.inconsistency.jaccard_index = NaN;
                            end
                        end
                        
                        % === MONTE CARLO METHOD ===
                        if strcmp(method, 'mc_probability') || strcmp(method, 'both')
                            try
                                [mc_score, mc_details] = score_mc_probability(...
                                    {Z_propagated, scenario.target}, ...
                                    'num_samples', options.mc_samples, ...
                                    'return_details', true, ...
                                    'verbose', false);
                                
                                % Store MC probability scores
                                state.inconsistency.mc_probability = mc_score;
                                state.inconsistency.mc_p_consistent = mc_details.p_consistent;
                                state.inconsistency.mc_p_inconsistent = mc_details.p_inconsistent;
                                state.inconsistency.mc_num_samples = mc_details.num_samples;
                                state.inconsistency.mc_num_consistent = mc_details.num_consistent;
                                state.inconsistency.mc_standard_error = mc_details.standard_error;
                                
                                % 95% confidence interval
                                state.inconsistency.mc_ci95_lower = mc_details.ci95_lower;
                                state.inconsistency.mc_ci95_upper = mc_details.ci95_upper;
                                
                            catch ME
                                warning('Monte Carlo scoring failed: %s', ME.message);
                                state.inconsistency.mc_probability = NaN;
                                state.inconsistency.mc_p_consistent = NaN;
                                state.inconsistency.mc_p_inconsistent = NaN;
                            end
                        end
                        
                        % Distance between centers (always computed)
                        try
                            state.inconsistency.center_distance = norm(...
                                Z_propagated.c - scenario.target.c);
                        catch
                            state.inconsistency.center_distance = NaN;
                        end
                        
                        % === GLOBAL INCONSISTENCY I(θ) ===
                        % Compute single scalar inconsistency across all models
                        try
                            % Collect all model zonotopes
                            all_models = {Z_propagated, scenario.target};
                            
                            % Build theta parameter vector from current scenario state
                            theta_params = struct();
                            if isfield(scenario, 'intervention_params')
                                theta_params = scenario.intervention_params;
                            end
                            
                            % Compute I(θ) = 1 - P(consistent across all models)
                            [I_theta, I_details] = global_inconsistency(all_models, ...
                                'n_samples', options.mc_samples, ...
                                'verbose', false, ...
                                'theta_params', theta_params);
                            
                            % Store global inconsistency metrics
                            state.inconsistency.I_theta = I_theta;
                            state.inconsistency.I_theta_se = I_details.standard_error;
                            state.inconsistency.I_theta_ci95_lower = I_details.ci95_lower;
                            state.inconsistency.I_theta_ci95_upper = I_details.ci95_upper;
                            state.inconsistency.I_theta_num_models = I_details.num_models;
                            
                            % Store theta parameters for sensitivity analysis
                            if ~isempty(fieldnames(theta_params))
                                state.inconsistency.theta_params = theta_params;
                            end
                            
                        catch ME
                            warning('Global inconsistency computation failed: %s', ME.message);
                            state.inconsistency.I_theta = NaN;
                            state.inconsistency.I_theta_se = NaN;
                        end
                    end
                    
                    % Constraint satisfaction metrics
                    if ~isempty(Z_propagated.A)
                        state.inconsistency.n_constraints = size(Z_propagated.A, 1);
                        state.inconsistency.constraint_rank = rank(Z_propagated.A);
                        state.inconsistency.constraint_redundancy = ...
                            state.inconsistency.n_constraints - state.inconsistency.constraint_rank;
                    else
                        state.inconsistency.n_constraints = 0;
                        state.inconsistency.constraint_rank = 0;
                        state.inconsistency.constraint_redundancy = 0;
                    end
                    
                catch ME
                    state.inconsistency.propagation_success = false;
                    state.inconsistency.error_message = ME.message;
                    state.inconsistency.is_empty = true;
                end
            end
        end
        
        function scenario_mod = apply_intervention(scenario, intervention, params)
            %APPLY_INTERVENTION Modify scenario according to intervention type
            %
            % This implements the do() operator from causal inference
            
            scenario_mod = scenario; % Copy
            
            % Store intervention parameters in scenario for theta tracking
            scenario_mod.intervention_params = params;
            scenario_mod.intervention_type = intervention;
            
            switch lower(intervention)
                % === UNCERTAINTY INTERVENTIONS ===
                case 'widen'
                    % Increase uncertainty by scaling generators
                    scale_factor = params.scale_factor; % e.g., 1.5
                    Z = scenario.source;
                    Z_new = conPolyZono(Z.c, Z.G * scale_factor, Z.E, Z.A, Z.b, Z.EC);
                    scenario_mod.source = Z_new;
                    
                case 'shrink'
                    % Decrease uncertainty by scaling generators
                    scale_factor = params.scale_factor; % e.g., 0.5
                    Z = scenario.source;
                    Z_new = conPolyZono(Z.c, Z.G * scale_factor, Z.E, Z.A, Z.b, Z.EC);
                    scenario_mod.source = Z_new;
                    
                case 'shift'
                    % Translate center without changing shape
                    shift_vector = params.shift_vector;
                    Z = scenario.source;
                    Z_new = conPolyZono(Z.c + shift_vector, Z.G, Z.E, Z.A, Z.b, Z.EC);
                    scenario_mod.source = Z_new;
                    
                case 'rotate'
                    % Rotate uncertainty region
                    rotation_matrix = params.rotation_matrix;
                    Z = scenario.source;
                    Z_new = conPolyZono(rotation_matrix * Z.c, ...
                                       rotation_matrix * Z.G, Z.E, Z.A, Z.b, Z.EC);
                    scenario_mod.source = Z_new;
                    
                case 'correlate'
                    % Add correlation between generators
                    correlation_strength = params.correlation_strength;
                    Z = scenario.source;
                    % Add dependent generator
                    new_gen = sum(Z.G, 2) * correlation_strength;
                    G_new = [Z.G, new_gen];
                    E_new = [Z.E, zeros(size(Z.E, 1), 1); zeros(1, size(Z.E, 2)), 1];
                    Z_new = conPolyZono(Z.c, G_new, E_new, Z.A, Z.b, Z.EC);
                    scenario_mod.source = Z_new;
                    
                otherwise
                    error('Unknown intervention type: %s', intervention);
            end
        end
        
        function metrics = compute_uncertainty_metrics(zonotope)
            %COMPUTE_UNCERTAINTY_METRICS Extract uncertainty-related measurements
            %
            % Outputs:
            %   metrics - Struct with volume, radii, norms, etc.
            
            metrics = struct();
            
            % Volume (zonotope measure)
            try
                metrics.volume = volume(zonotope);
            catch
                % Fallback: use generator-based approximation
                G = zonotope.G;
                metrics.volume = prod(vecnorm(G, 2, 1)) * 2^size(G, 2);
            end
            
            % Generator-based metrics
            G = zonotope.G;
            generator_norms = vecnorm(G, 2, 1);  % L2 norm of each column
            
            metrics.avg_radius = mean(generator_norms);
            metrics.max_radius = max(generator_norms);
            metrics.min_radius = min(generator_norms);
            metrics.frobenius_norm = norm(G, 'fro');
            metrics.n_generators = size(G, 2);
            metrics.dimension = size(G, 1);
            
            % Center
            metrics.center = zonotope.c;
            metrics.center_norm = norm(zonotope.c);
        end
        
        function metrics = compute_inconsistency_metrics(source, target, mapping)
            %COMPUTE_INCONSISTENCY_METRICS Extract inconsistency-related measurements
            %   Uses consistency scoring addon for robust Jaccard computation
            %
            % Outputs:
            %   metrics - Struct with consistency scores, jaccard_index, is_empty, etc.
            
            metrics = struct();
            
            % Propagate source through mapping
            F = mapping.F;
            f = mapping.f;
            Z_propagated = CS_Types.affineMap_cPZ(source, F, f);
            
            % Check emptiness of propagated set
            metrics.is_empty_propagated = isEmptySet(Z_propagated);
            
            % Use consistency scoring addon for robust metrics
            try
                % Identity mapping for direct comparison (Z_prop vs Z_tgt)
                F_identity = eye(size(target.c, 1));
                f_identity = zeros(size(target.c, 1), 1);
                
                % Call score_jaccard from consistency addon
                [C, Csym, details] = score_jaccard(...
                    Z_propagated, target, F_identity, f_identity, ...
                    struct('return_details', true, 'verbose', false));
                
                % Store addon-based consistency scores (primary metrics)
                metrics.consistency_score = C;         % Directional consistency
                metrics.consistency_score_sym = Csym;  % Symmetric (Jaccard-like)
                metrics.jaccard_index = Csym;          % Use addon's robust Jaccard
                metrics.is_empty = details.is_empty;
                metrics.has_intersection = ~details.is_empty;
                
                % Store size components
                metrics.vol_propagated = details.s_prop;
                metrics.vol_target = details.s_tgt;
                metrics.vol_intersection = details.s_int;
                
            catch ME
                warning('Consistency scoring failed: %s. Using fallback.', ME.message);
                % Fallback to basic intersection check
                try
                    Z_intersect = Z_propagated & target;
                    metrics.has_intersection = ~isEmptySet(Z_intersect);
                    metrics.is_empty = ~metrics.has_intersection;
                    if metrics.has_intersection
                        metrics.jaccard_index = 0.5;
                    else
                        metrics.jaccard_index = 0.0;
                    end
                    metrics.consistency_score = metrics.jaccard_index;
                    metrics.consistency_score_sym = metrics.jaccard_index;
                catch
                    metrics.has_intersection = false;
                    metrics.is_empty = true;
                    metrics.jaccard_index = 0.0;
                    metrics.consistency_score = 0.0;
                    metrics.consistency_score_sym = 0.0;
                end
            end
        end
        
        function delta = compute_delta(pre_state, post_state)
            %COMPUTE_DELTA Calculate change in measurements (causal effect)
            %
            % Computes deltas for BOTH causal directions:
            %   - Uncertainty → Inconsistency: use delta_jaccard, delta_empty
            %   - Inconsistency → Uncertainty: use volume_change, radius_change
            
            delta = struct();
            delta.uncertainty = struct();
            delta.inconsistency = struct();
            
            % === UNCERTAINTY DELTAS (for reverse direction: inconsistency → uncertainty) ===
            if isfield(pre_state.uncertainty, 'source_volume')
                delta.uncertainty.volume_change = ...
                    post_state.uncertainty.source_volume - pre_state.uncertainty.source_volume;
                delta.uncertainty.volume_ratio = ...
                    post_state.uncertainty.source_volume / pre_state.uncertainty.source_volume;
                delta.uncertainty.volume_change_pct = ...
                    100 * (post_state.uncertainty.source_volume - pre_state.uncertainty.source_volume) / ...
                    pre_state.uncertainty.source_volume;
            end
            
            if isfield(pre_state.uncertainty, 'source_radius')
                delta.uncertainty.radius_change = ...
                    post_state.uncertainty.source_radius - pre_state.uncertainty.source_radius;
                delta.uncertainty.radius_change_pct = ...
                    100 * (post_state.uncertainty.source_radius - pre_state.uncertainty.source_radius) / ...
                    pre_state.uncertainty.source_radius;
            end
            
            if isfield(pre_state.uncertainty, 'source_correlation')
                delta.uncertainty.correlation_change = ...
                    post_state.uncertainty.source_correlation - pre_state.uncertainty.source_correlation;
            end
            
            if isfield(pre_state.uncertainty, 'source_n_generators')
                delta.uncertainty.n_generators_change = ...
                    post_state.uncertainty.source_n_generators - pre_state.uncertainty.source_n_generators;
            end
            
            % === INCONSISTENCY DELTAS (for forward direction: uncertainty → inconsistency) ===
            if isfield(pre_state.inconsistency, 'is_empty')
                delta.inconsistency.emptiness_change = ...
                    double(post_state.inconsistency.is_empty) - double(pre_state.inconsistency.is_empty);
            end
            
            if isfield(pre_state.inconsistency, 'center_distance')
                delta.inconsistency.distance_change = ...
                    post_state.inconsistency.center_distance - pre_state.inconsistency.center_distance;
            end
            
            % === JACCARD METHOD DELTAS ===
            if isfield(pre_state.inconsistency, 'jaccard_C')
                delta.inconsistency.delta_jaccard_C = ...
                    post_state.inconsistency.jaccard_C - pre_state.inconsistency.jaccard_C;
            end
            
            if isfield(pre_state.inconsistency, 'jaccard_Csym')
                delta.inconsistency.delta_jaccard_Csym = ...
                    post_state.inconsistency.jaccard_Csym - pre_state.inconsistency.jaccard_Csym;
            end
            
            if isfield(pre_state.inconsistency, 'jaccard_index')
                delta.inconsistency.delta_jaccard = ...
                    post_state.inconsistency.jaccard_index - pre_state.inconsistency.jaccard_index;
                delta.inconsistency.delta_jaccard_abs = ...
                    abs(post_state.inconsistency.jaccard_index - pre_state.inconsistency.jaccard_index);
            end
            
            % === MONTE CARLO METHOD DELTAS ===
            if isfield(pre_state.inconsistency, 'mc_probability')
                delta.inconsistency.delta_mc_probability = ...
                    post_state.inconsistency.mc_probability - pre_state.inconsistency.mc_probability;
                delta.inconsistency.delta_mc_probability_abs = ...
                    abs(post_state.inconsistency.mc_probability - pre_state.inconsistency.mc_probability);
            end
            
            if isfield(pre_state.inconsistency, 'mc_p_consistent')
                delta.inconsistency.delta_mc_p_consistent = ...
                    post_state.inconsistency.mc_p_consistent - pre_state.inconsistency.mc_p_consistent;
            end
            
            if isfield(pre_state.inconsistency, 'mc_p_inconsistent')
                delta.inconsistency.delta_mc_p_inconsistent = ...
                    post_state.inconsistency.mc_p_inconsistent - pre_state.inconsistency.mc_p_inconsistent;
            end
            
            % === GLOBAL INCONSISTENCY DELTAS ===
            % These are critical for sensitivity analysis: τ_j(a,b) = E[I(θ)|do(θ_j=b)] - E[I(θ)|do(θ_j=a)]
            if isfield(pre_state.inconsistency, 'I_theta')
                delta.inconsistency.delta_I_theta = ...
                    post_state.inconsistency.I_theta - pre_state.inconsistency.I_theta;
                delta.inconsistency.delta_I_theta_abs = ...
                    abs(post_state.inconsistency.I_theta - pre_state.inconsistency.I_theta);
                
                % Relative change (percent)
                if pre_state.inconsistency.I_theta > 0
                    delta.inconsistency.delta_I_theta_pct = ...
                        100 * (post_state.inconsistency.I_theta - pre_state.inconsistency.I_theta) / ...
                        pre_state.inconsistency.I_theta;
                else
                    delta.inconsistency.delta_I_theta_pct = NaN;
                end
            end
            
            % Binary outcome change (−1 resolved, 0 unchanged, +1 created)
            if isfield(pre_state.inconsistency, 'empty_intersection') && ...
               isfield(post_state.inconsistency, 'empty_intersection')
                delta.inconsistency.delta_empty = ...
                    double(post_state.inconsistency.empty_intersection) - ...
                    double(pre_state.inconsistency.empty_intersection);
            end
        end
        
        % === HELPER FUNCTIONS ===
        
        function vol = compute_volume(Z)
            %COMPUTE_VOLUME Estimate zonotope volume
            try
                I = interval(Z);
                vol = prod(supremum(I) - infimum(I));
            catch
                vol = NaN;
            end
        end
        
        function rad = compute_radius(Z)
            %COMPUTE_RADIUS Compute maximum distance from center
            try
                I = interval(Z);
                rad = max(supremum(I) - infimum(I)) / 2;
            catch
                rad = NaN;
            end
        end
        function correlation = compute_correlation(G)
            %COMPUTE_CORRELATION Measure generator correlation
            if size(G, 2) <= 1
                correlation = 0;
                return;
            end
            % Compute correlation matrix of generators manually
            % Normalize each column (generator)
            G_norm = G ./ vecnorm(G, 2, 1);
            % Correlation matrix: C(i,j) = dot(g_i, g_j)
            C = G_norm' * G_norm;
            % Return mean off-diagonal correlation
            mask = ~eye(size(C));
            correlation = mean(abs(C(mask)));
        end
        
        % Note: check_emptiness() removed - now using isEmptySet() from CPS framework directly
        
        function scenario = create_baseline_scenario(dim, uncertainty_level)
            %CREATE_BASELINE_SCENARIO Generate CONSISTENT baseline scenario
            %
            % SEMANTICS (following CPS framework):
            %   - source (Z_B):   "New" measurement/state (e.g., BrakeDisk)
            %   - target (Z_old): "Existing" model/constraint (e.g., CAD)
            %   - mapping F:      Uncertainty Propagation Rule (UPR)
            %   - Consistency:    Z_propagated ∩ Z_old ≠ ∅ (safe region exists)
            %
            % Inputs:
            %   dim               - Dimension (2, 4, 8, ...)
            %   uncertainty_level - Scaling factor for generators
            %
            % Outputs:
            %   scenario - Struct with .source, .target, .mapping
            
            % Source zonotope (new measurement/update)
            c_src = randn(dim, 1) * 10;
            G_src = eye(dim) * uncertainty_level;
            E_src = eye(dim);
            Z_src = conPolyZono(c_src, G_src, E_src, [], [], []);
            
            % Mapping (Uncertainty Propagation Rule)
            F = eye(dim) * 1.0;  % Identity (no distortion baseline)
            f = zeros(dim, 1);   % No bias
            
            % Target (existing model) - must overlap with propagated source
            % Strategy: Center target near propagated center with overlapping radius
            c_propagated = F * c_src + f;  % = c_src for identity F
            
            % Place target slightly offset but still overlapping
            offset = randn(dim, 1) * 0.3 * uncertainty_level;  % Small random offset
            c_tgt = c_propagated + offset;
            
            % Target has similar size (ensures overlap for baseline)
            G_tgt = eye(dim) * uncertainty_level * 1.2;  % Slightly larger
            Z_tgt = conPolyZono(c_tgt, G_tgt, E_src, [], [], []);
            
            % Mapping
            mapping = struct('F', F, 'f', f);
            
            scenario = struct(...
                'source', Z_src, ...
                'target', Z_tgt, ...
                'mapping', mapping, ...
                'dim', dim);
        end
        
        function results = run_intervention_sweep(scenario, intervention, param_name, param_values, options)
            %RUN_INTERVENTION_SWEEP Run intervention with varying parameter
            %
            % Inputs:
            %   scenario      - Struct with .source, .target, .mapping
            %   intervention  - String: intervention type
            %   param_name    - String: parameter field name
            %   param_values  - Vector: parameter values to sweep
            %   options       - (optional) Struct: passed to run_intervention
            %
            % Example:
            %   options = struct('mc_samples', 500);
            %   results = run_intervention_sweep(scenario, 'widen', 'scale_factor', [0.5, 1.0, 1.5, 2.0], options);
            
            if nargin < 5
                options = struct();
            end
            
            n = length(param_values);
            results = cell(n, 1);
            
            for i = 1:n
                params = struct(param_name, param_values(i));
                results{i} = causal_experiment_engine.run_intervention(scenario, intervention, params, options);
                results{i}.param_value = param_values(i);
            end
        end
        
        function data = export_to_python(results)
            %EXPORT_TO_PYTHON Convert results to Python-friendly format
            %
            % Outputs JSON-serializable struct
            
            % Handle both formats: cell array or struct with metadata
            if isstruct(results) && isfield(results, 'experiments')
                % Struct format with metadata (from generate_convide_examples)
                data = results;
                experiments_list = results.experiments;
            else
                % Simple cell array format (legacy)
                data = struct();
                experiments_list = results;
            end
            
            % Process experiments
            data.n_experiments = length(experiments_list);
            data.experiments = cell(data.n_experiments, 1);
            
            for i = 1:data.n_experiments
                exp = experiments_list{i};
                data.experiments{i} = struct(...
                    'intervention', exp.intervention.type, ...
                    'param_value', exp.param_value, ...
                    'pre_uncertainty', causal_experiment_engine.flatten_struct(exp.pre_state.uncertainty), ...
                    'post_uncertainty', causal_experiment_engine.flatten_struct(exp.post_state.uncertainty), ...
                    'pre_inconsistency', causal_experiment_engine.flatten_struct(exp.pre_state.inconsistency), ...
                    'post_inconsistency', causal_experiment_engine.flatten_struct(exp.post_state.inconsistency), ...
                    'causal_effect', causal_experiment_engine.flatten_struct(exp.causal_effect));
            end
        end
        
        function export_to_json(results, filename)
            %EXPORT_TO_JSON Export results to JSON file for Python analysis
            %
            % Inputs:
            %   results  - Cell array of experiment results
            %   filename - Output JSON file path
            %
            % Example:
            %   causal_experiment_engine.export_to_json(results, 'data/results.json');
            
            % Convert to Python-friendly format
            data = causal_experiment_engine.export_to_python(results);
            
            % Write to JSON file
            json_text = jsonencode(data, 'PrettyPrint', true);
            
            % Create directory if needed
            [filepath, ~, ~] = fileparts(filename);
            if ~isempty(filepath) && ~exist(filepath, 'dir')
                mkdir(filepath);
            end
            
            % Write file
            fid = fopen(filename, 'w');
            if fid == -1
                error('Failed to open file: %s', filename);
            end
            fprintf(fid, '%s', json_text);
            fclose(fid);
        end
        
        function flat = flatten_struct(s)
            %FLATTEN_STRUCT Convert nested struct to flat key-value pairs
            flat = struct();
            fields = fieldnames(s);
            for i = 1:length(fields)
                val = s.(fields{i});
                if isnumeric(val) || islogical(val)
                    if isscalar(val)
                        flat.(fields{i}) = val;
                    else
                        flat.(fields{i}) = val(:)'; % Row vector
                    end
                elseif ischar(val)
                    flat.(fields{i}) = val;
                elseif isstruct(val)
                    % Recursively flatten
                    sub_flat = causal_experiment_engine.flatten_struct(val);
                    sub_fields = fieldnames(sub_flat);
                    for j = 1:length(sub_fields)
                        flat.([fields{i} '_' sub_fields{j}]) = sub_flat.(sub_fields{j});
                    end
                end
            end
        end
    end
end
