classdef causal_experiment_engine_twostep
    %CAUSAL_EXPERIMENT_ENGINE_TWOSTEP Self-contained two-step workflow for causal experiments
    %
    % This class is FULLY SELF-CONTAINED and does not depend on the one-step engine.
    % It separates zonotope generation from consistency measurement, allowing users
    % to recompute consistency scores without regenerating expensive zonotope operations.
    %
    % WORKFLOW:
    %   Step 1: generate_and_save_zonotopes()
    %           - Create scenarios
    %           - Apply interventions
    %           - Save zonotopes to .mat files
    %
    %   Step 2: compute_and_save_consistency()
    %           - Load zonotopes from .mat files
    %           - Compute consistency scores
    %           - Save results to JSON
    %
    % BENEFITS:
    %   - Re-run consistency measurements with different methods
    %   - Try new consistency metrics without regenerating data
    %   - Faster iteration on analysis
    %   - No dependency on one-step engine
    %
    % USAGE:
    %   % Step 1: Generate zonotopes (do once)
    %   causal_experiment_engine_twostep.generate_and_save_zonotopes(...
    %       scenario_def, interventions, output_dir, options);
    %
    %   % Step 2: Measure consistency (can repeat with different methods)
    %   causal_experiment_engine_twostep.compute_and_save_consistency(...
    %       zonotope_dir, output_dir, consistency_options);
    %
    % See also: consistency_score, score_jaccard, score_mc_probability
    
    properties (Constant)
        % Supported interventions
        UNCERTAINTY_INTERVENTIONS = {'widen', 'shrink', 'shift', 'rotate', 'correlate'};
        
        % Consistency scoring methods
        CONSISTENCY_METHODS = {'jaccard', 'mc_probability', 'both'};
        DEFAULT_METHOD = 'both';
        DEFAULT_MC_SAMPLES = 2000;
    end
    
    methods (Static)
        
        %% ================================================================
        %  STEP 1: ZONOTOPE GENERATION
        %  ================================================================
        
        function zonotope_data = generate_and_save_zonotopes(scenario_def, interventions, output_dir, options)
            %GENERATE_AND_SAVE_ZONOTOPES Step 1: Create and save zonotopes
            %
            % Inputs:
            %   scenario_def  - Struct with .id, .name, .source, .target, .type fields
            %   interventions - Cell array of intervention structs
            %   output_dir    - Directory to save zonotope .mat files
            %   options       - Struct with .n_repeats, .verbose
            %
            % Outputs:
            %   zonotope_data - Cell array of structs with zonotope metadata
            %
            % Saved files:
            %   <output_dir>/zonotopes_<scenario_name>_<exp_id>.mat
            
            if nargin < 4
                options = struct();
            end
            if ~isfield(options, 'n_repeats')
                options.n_repeats = 5;
            end
            if ~isfield(options, 'verbose')
                options.verbose = true;
            end
            
            % Create output directory
            if ~exist(output_dir, 'dir')
                mkdir(output_dir);
            end
            
            scenario_name = sprintf('scenario_%d', scenario_def.id);
            
            if options.verbose
                fprintf('Step 1: Generating zonotopes for %s (%s)...\n', ...
                    scenario_def.name, scenario_name);
            end
            
            zonotope_data = {};
            exp_count = 0;
            
            % Create baseline scenario
            baseline_scenario = causal_experiment_engine_twostep.create_baseline_scenario(...
                length(scenario_def.source.c), 5.0);
            baseline_scenario.source = scenario_def.source;
            baseline_scenario.target = scenario_def.target;
            
            % Loop over interventions
            for i = 1:length(interventions)
                int = interventions{i};
                
                if options.verbose
                    fprintf('  Intervention: %s (%d values × %d repeats)\n', ...
                        int.type, length(int.values), options.n_repeats);
                end
                
                for val_idx = 1:length(int.values)
                    param_val = int.values(val_idx);
                    
                    for rep = 1:options.n_repeats
                        exp_count = exp_count + 1;
                        
                        if options.verbose && mod(exp_count, 20) == 0
                            fprintf('    Progress: %d experiments generated\n', exp_count);
                        end
                        
                        % Apply intervention
                        params = struct(int.param, param_val);
                        scenario_modified = causal_experiment_engine_twostep.apply_intervention(...
                            baseline_scenario, int.type, params);
                        
                        % Propagate modified source
                        F = scenario_modified.mapping.F;
                        f = scenario_modified.mapping.f;
                        Z_propagated = CS_Types.affineMap_cPZ(scenario_modified.source, F, f);
                        
                        % Store zonotope data
                        zono_data = struct();
                        zono_data.exp_id = exp_count;
                        zono_data.scenario_id = scenario_def.id;
                        zono_data.scenario_name = scenario_name;
                        zono_data.scenario_description = scenario_def.name;
                        zono_data.causality_type = scenario_def.type;
                        zono_data.intervention_type = int.type;
                        zono_data.param_name = int.param;
                        zono_data.param_value = param_val;
                        zono_data.repeat_idx = rep;
                        zono_data.timestamp = datetime('now');
                        
                        % Store zonotopes
                        zono_data.Z_source_pre = baseline_scenario.source;
                        zono_data.Z_target = baseline_scenario.target;
                        zono_data.Z_source_post = scenario_modified.source;
                        zono_data.Z_propagated = Z_propagated;
                        
                        % Store mapping
                        zono_data.mapping_F = F;
                        zono_data.mapping_f = f;
                        
                        % Save to file
                        filename = sprintf('zonotopes_%s_exp%04d.mat', scenario_name, exp_count);
                        filepath = fullfile(output_dir, filename);
                        save(filepath, '-struct', 'zono_data');
                        
                        % Keep metadata for return
                        zono_metadata = struct();
                        zono_metadata.exp_id = exp_count;
                        zono_metadata.scenario_name = scenario_name;
                        zono_metadata.intervention_type = int.type;
                        zono_metadata.param_value = param_val;
                        zono_metadata.repeat_idx = rep;
                        zono_metadata.filepath = filepath;
                        
                        zonotope_data{end+1} = zono_metadata;
                    end
                end
            end
            
            % Save index file
            index_file = fullfile(output_dir, sprintf('index_%s.mat', scenario_name));
            zonotope_index = struct();
            zonotope_index.scenario_def = scenario_def;
            zonotope_index.interventions = interventions;
            zonotope_index.n_experiments = exp_count;
            zonotope_index.zonotope_files = zonotope_data;
            zonotope_index.generation_timestamp = datetime('now');
            save(index_file, 'zonotope_index');
            
            if options.verbose
                fprintf('  ✓ Generated %d zonotope files\n', exp_count);
                fprintf('  ✓ Saved index: %s\n\n', index_file);
            end
        end
        
        %% ================================================================
        %  STEP 2: CONSISTENCY COMPUTATION
        %  ================================================================
        
        function results = compute_and_save_consistency(zonotope_dir, output_dir, consistency_options)
            %COMPUTE_AND_SAVE_CONSISTENCY Step 2: Compute consistency from saved zonotopes
            %
            % Inputs:
            %   zonotope_dir         - Directory containing zonotope .mat files
            %   output_dir           - Directory to save results JSON
            %   consistency_options  - Struct with:
            %       .method          - 'jaccard', 'mc_probability', 'both' (default: 'both')
            %       .mc_samples      - Number of MC samples (default: 2000)
            %       .verbose         - Display progress (default: true)
            %       .scenario_filter - (optional) Only process specific scenario names
            %
            % Outputs:
            %   results - Cell array of result structs
            
            if nargin < 3
                consistency_options = struct();
            end
            if ~isfield(consistency_options, 'method')
                consistency_options.method = 'both';
            end
            if ~isfield(consistency_options, 'mc_samples')
                consistency_options.mc_samples = 2000;
            end
            if ~isfield(consistency_options, 'verbose')
                consistency_options.verbose = true;
            end
            
            % Create output directory
            if ~exist(output_dir, 'dir')
                mkdir(output_dir);
            end
            
            % Find all index files
            index_files = dir(fullfile(zonotope_dir, 'index_*.mat'));
            
            if isempty(index_files)
                error('No index files found in %s', zonotope_dir);
            end
            
            if consistency_options.verbose
                fprintf('Step 2: Computing consistency scores from saved zonotopes...\n');
                fprintf('Found %d scenario(s)\n', length(index_files));
                fprintf('Method: %s\n', consistency_options.method);
                if contains(consistency_options.method, 'mc')
                    fprintf('MC Samples: %d\n\n', consistency_options.mc_samples);
                end
            end
            
            results = {};
            
            % Process each scenario
            for idx_file_idx = 1:length(index_files)
                % Load index
                index_path = fullfile(zonotope_dir, index_files(idx_file_idx).name);
                loaded = load(index_path);
                zonotope_index = loaded.zonotope_index;
                
                scenario_name = sprintf('scenario_%d', zonotope_index.scenario_def.id);
                
                % Check filter
                if isfield(consistency_options, 'scenario_filter') && ...
                   ~isempty(consistency_options.scenario_filter)
                    if ~strcmp(scenario_name, consistency_options.scenario_filter)
                        continue;
                    end
                end
                
                n_total_exp = length(zonotope_index.zonotope_files);
                
                if consistency_options.verbose
                    fprintf('\nProcessing %s (%s)...\n', ...
                        scenario_name, zonotope_index.scenario_def.name);
                    fprintf('  Total experiments: %d\n', n_total_exp);
                    fprintf('  Method: %s\n', consistency_options.method);
                    if contains(lower(consistency_options.method), 'mc') || ...
                       strcmp(lower(consistency_options.method), 'both')
                        fprintf('  MC samples per experiment: %d\n', consistency_options.mc_samples);
                    end
                    fprintf('\n');
                end
                
                all_results = {};
                scenario_start_time = tic;
                
                % Process each zonotope file
                for exp_idx = 1:n_total_exp
                    zono_meta = zonotope_index.zonotope_files{exp_idx};
                    
                    exp_start_time = tic;
                    
                    % Load zonotope data
                    zono_data = load(zono_meta.filepath);
                    
                    % Measure pre-intervention state
                    pre_scenario = struct();
                    pre_scenario.source = zono_data.Z_source_pre;
                    pre_scenario.target = zono_data.Z_target;
                    pre_scenario.mapping = struct('F', zono_data.mapping_F, 'f', zono_data.mapping_f);
                    
                    pre_state = causal_experiment_engine_twostep.measure_state(...
                        pre_scenario, consistency_options);
                    
                    % Measure post-intervention state
                    post_scenario = struct();
                    post_scenario.source = zono_data.Z_source_post;
                    post_scenario.target = zono_data.Z_target;
                    post_scenario.mapping = struct('F', zono_data.mapping_F, 'f', zono_data.mapping_f);
                    
                    post_state = causal_experiment_engine_twostep.measure_state(...
                        post_scenario, consistency_options);
                    
                    exp_time = toc(exp_start_time);
                    
                    % Progress reporting with time estimates
                    if consistency_options.verbose
                        elapsed_total = toc(scenario_start_time);
                        avg_time_per_exp = elapsed_total / exp_idx;
                        remaining_exp = n_total_exp - exp_idx;
                        eta_seconds = remaining_exp * avg_time_per_exp;
                        
                        % Print every 5 experiments or every 10 seconds
                        if mod(exp_idx, 5) == 0 || exp_idx == 1 || exp_idx == n_total_exp
                            fprintf('  [%d/%d] %.2fs/exp | Elapsed: %.1fs | ETA: %.1fs (%.1f min)\n', ...
                                exp_idx, n_total_exp, exp_time, elapsed_total, eta_seconds, eta_seconds/60);
                        end
                    end
                    
                    % Compute causal effect
                    causal_effect = causal_experiment_engine_twostep.compute_delta(...
                        pre_state, post_state);
                    
                    % Build result struct
                    result = struct();
                    result.exp_id = zono_data.exp_id;
                    result.scenario_type = zono_data.scenario_name;
                    result.scenario_description = zono_data.scenario_description;
                    result.causality_type = zono_data.causality_type;
                    result.intervention_type = zono_data.intervention_type;
                    result.intervention_direction = 'forward';
                    result.param_value = zono_data.param_value;
                    result.repeat_idx = zono_data.repeat_idx;
                    result.run_id = zono_data.exp_id;
                    
                    result.pre_state = pre_state;
                    result.post_state = post_state;
                    result.causal_effect = causal_effect;
                    
                    result.consistency_method = consistency_options.method;
                    result.measurement_timestamp = datetime('now');
                    
                    all_results{end+1} = result;
                end
                
                scenario_total_time = toc(scenario_start_time);
                if consistency_options.verbose
                    fprintf('\n  ✓ Scenario complete in %.1f seconds (%.2f min)\n', ...
                        scenario_total_time, scenario_total_time/60);
                end
                
                % Save results to JSON
                output_file = fullfile(output_dir, sprintf('results_%s.json', scenario_name));
                
                metadata = struct();
                metadata.experiments = all_results;
                metadata.scenario_name = scenario_name;
                metadata.scenario_description = zonotope_index.scenario_def.name;
                metadata.causality_type = zonotope_index.scenario_def.type;
                metadata.consistency_method = consistency_options.method;
                metadata.mc_samples = consistency_options.mc_samples;
                metadata.n_repeats = zonotope_index.zonotope_files{1}.repeat_idx;
                metadata.total_experiments = length(all_results);
                metadata.zonotope_generation_timestamp = zonotope_index.generation_timestamp;
                metadata.consistency_measurement_timestamp = datetime('now');
                
                causal_experiment_engine_twostep.export_to_json(metadata, output_file);
                
                if consistency_options.verbose
                    fprintf('  ✓ Saved %d results to: %s\n\n', length(all_results), output_file);
                end
                
                results{end+1} = metadata;
            end
            
            if consistency_options.verbose
                fprintf('========================================\n');
                fprintf('Consistency computation complete!\n');
                fprintf('========================================\n');
            end
        end
        
        function regenerate_consistency_all(zonotope_dir, output_base_dir, new_method, mc_samples)
            %REGENERATE_CONSISTENCY_ALL Recompute consistency for all scenarios with new method
            
            if nargin < 4
                mc_samples = 2000;
            end
            
            % Create output directory with method name
            output_dir = fullfile(output_base_dir, sprintf('consistency_%s', new_method));
            if ~exist(output_dir, 'dir')
                mkdir(output_dir);
            end
            
            fprintf('========================================\n');
            fprintf('REGENERATING CONSISTENCY MEASUREMENTS\n');
            fprintf('========================================\n');
            fprintf('Method: %s\n', new_method);
            fprintf('MC Samples: %d\n', mc_samples);
            fprintf('Zonotopes: %s\n', zonotope_dir);
            fprintf('Output: %s\n\n', output_dir);
            
            % Compute consistency
            consistency_opts = struct();
            consistency_opts.method = new_method;
            consistency_opts.mc_samples = mc_samples;
            consistency_opts.verbose = true;
            
            causal_experiment_engine_twostep.compute_and_save_consistency(...
                zonotope_dir, output_dir, consistency_opts);
        end
        
        %% ================================================================
        %  MEASUREMENT FUNCTIONS (Self-contained)
        %  ================================================================
        
        function state = measure_state(scenario, options)
            %MEASURE_STATE Compute all observables for causal analysis
            %
            % Self-contained measurement function that doesn't depend on the one-step engine.
            
            if nargin < 2
                options = struct();
            end
            if ~isfield(options, 'method')
                options.method = causal_experiment_engine_twostep.DEFAULT_METHOD;
            end
            if ~isfield(options, 'mc_samples')
                options.mc_samples = causal_experiment_engine_twostep.DEFAULT_MC_SAMPLES;
            end
            
            state = struct();
            
            % === UNCERTAINTY METRICS ===
            state.uncertainty = struct();
            
            % Source uncertainty
            if isfield(scenario, 'source') && isobject(scenario.source)
                Z_src = scenario.source;
                state.uncertainty.source_volume = causal_experiment_engine_twostep.compute_volume(Z_src);
                state.uncertainty.source_radius = causal_experiment_engine_twostep.compute_radius(Z_src);
                state.uncertainty.source_center = Z_src.c;
                state.uncertainty.source_n_generators = size(Z_src.G, 2);
                
                % Correlation measure
                if size(Z_src.G, 2) > 1
                    state.uncertainty.source_correlation = ...
                        causal_experiment_engine_twostep.compute_correlation(Z_src.G);
                else
                    state.uncertainty.source_correlation = 0;
                end
            end
            
            % Target uncertainty
            if isfield(scenario, 'target') && isobject(scenario.target)
                Z_tgt = scenario.target;
                state.uncertainty.target_volume = causal_experiment_engine_twostep.compute_volume(Z_tgt);
                state.uncertainty.target_radius = causal_experiment_engine_twostep.compute_radius(Z_tgt);
                state.uncertainty.target_center = Z_tgt.c;
                state.uncertainty.target_n_generators = size(Z_tgt.G, 2);
            end
            
            % === INCONSISTENCY METRICS ===
            state.inconsistency = struct();
            
            % Propagate and check for inconsistencies
            if isfield(scenario, 'mapping') && isfield(scenario, 'source') && isobject(scenario.source)
                try
                    F = scenario.mapping.F;
                    f = scenario.mapping.f;
                    Z_propagated = CS_Types.affineMap_cPZ(scenario.source, F, f);
                    
                    state.inconsistency.propagation_success = true;
                    state.inconsistency.propagated_volume = ...
                        causal_experiment_engine_twostep.compute_volume(Z_propagated);
                    
                    % Check if propagated set is empty
                    state.inconsistency.is_empty = isEmptySet(Z_propagated);
                    
                    % If target exists, compute consistency scores
                    if isfield(scenario, 'target') && isobject(scenario.target)
                        method = lower(options.method);
                        
                        % === JACCARD METHOD ===
                        if strcmp(method, 'jaccard') || strcmp(method, 'both')
                            try
                                F_identity = eye(size(scenario.target.c, 1));
                                f_identity = zeros(size(scenario.target.c, 1), 1);
                                
                                [C, Csym, details] = score_jaccard(...
                                    Z_propagated, scenario.target, F_identity, f_identity, ...
                                    struct('return_details', true, 'verbose', false));
                                
                                state.inconsistency.jaccard_C = C;
                                state.inconsistency.jaccard_Csym = Csym;
                                state.inconsistency.jaccard_index = Csym;
                                state.inconsistency.empty_intersection = details.is_empty;
                                state.inconsistency.has_intersection = ~details.is_empty;
                                state.inconsistency.jaccard_vol_intersection = details.s_int;
                                state.inconsistency.jaccard_vol_union = details.s_prop + details.s_tgt - details.s_int;
                                
                                if ~details.is_empty
                                    Z_intersect = Z_propagated & scenario.target;
                                    state.inconsistency.intersection_volume = ...
                                        causal_experiment_engine_twostep.compute_volume(Z_intersect);
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
                                
                                state.inconsistency.mc_probability = mc_score;
                                state.inconsistency.mc_p_consistent = mc_details.p_consistent;
                                state.inconsistency.mc_p_inconsistent = mc_details.p_inconsistent;
                                state.inconsistency.mc_num_samples = mc_details.num_samples;
                                state.inconsistency.mc_num_consistent = mc_details.num_consistent;
                                state.inconsistency.mc_standard_error = mc_details.standard_error;
                                state.inconsistency.mc_ci95_lower = mc_details.ci95_lower;
                                state.inconsistency.mc_ci95_upper = mc_details.ci95_upper;
                            catch ME
                                warning('Monte Carlo scoring failed: %s', ME.message);
                                state.inconsistency.mc_probability = NaN;
                                state.inconsistency.mc_p_consistent = NaN;
                                state.inconsistency.mc_p_inconsistent = NaN;
                            end
                        end
                        
                        % Distance between centers
                        try
                            state.inconsistency.center_distance = norm(...
                                Z_propagated.c - scenario.target.c);
                        catch
                            state.inconsistency.center_distance = NaN;
                        end
                        
                        % === GLOBAL INCONSISTENCY I(θ) ===
                        try
                            all_models = {Z_propagated, scenario.target};
                            
                            % Determine method for I(theta)
                            if strcmp(method, 'jaccard')
                                i_theta_method = 'jaccard';
                            elseif strcmp(method, 'mc_probability')
                                i_theta_method = 'mc';
                            else
                                i_theta_method = 'jaccard';  % Default for 'both'
                            end
                            
                            [I_theta, I_details] = global_inconsistency(all_models, ...
                                'n_samples', options.mc_samples, ...
                                'method', i_theta_method, ...
                                'return_details', true, ...
                                'verbose', false);
                            
                            state.inconsistency.I_theta = I_theta;
                            state.inconsistency.I_theta_p_consistent = I_details.p_consistent;
                            state.inconsistency.I_theta_standard_error = I_details.standard_error;
                        catch ME
                            warning('Global I(theta) failed: %s', ME.message);
                            state.inconsistency.I_theta = NaN;
                        end
                    end
                catch ME
                    state.inconsistency.propagation_success = false;
                    state.inconsistency.propagation_error = ME.message;
                end
            end
        end
        
        function delta = compute_delta(pre_state, post_state)
            %COMPUTE_DELTA Compute difference between states (causal effect)
            
            delta = struct();
            
            % === UNCERTAINTY DELTAS ===
            delta.uncertainty = struct();
            
            % Volume change
            if isfield(pre_state.uncertainty, 'source_volume') && ...
               isfield(post_state.uncertainty, 'source_volume')
                delta.uncertainty.delta_source_volume = ...
                    post_state.uncertainty.source_volume - pre_state.uncertainty.source_volume;
            end
            
            % Radius change
            if isfield(pre_state.uncertainty, 'source_radius') && ...
               isfield(post_state.uncertainty, 'source_radius')
                delta.uncertainty.delta_source_radius = ...
                    post_state.uncertainty.source_radius - pre_state.uncertainty.source_radius;
            end
            
            % === INCONSISTENCY DELTAS ===
            delta.inconsistency = struct();
            
            % Jaccard score changes
            if isfield(pre_state.inconsistency, 'jaccard_C') && ...
               isfield(post_state.inconsistency, 'jaccard_C')
                delta.inconsistency.delta_jaccard_C = ...
                    post_state.inconsistency.jaccard_C - pre_state.inconsistency.jaccard_C;
                delta.inconsistency.delta_jaccard_Csym = ...
                    post_state.inconsistency.jaccard_Csym - pre_state.inconsistency.jaccard_Csym;
            end
            
            % MC probability changes
            if isfield(pre_state.inconsistency, 'mc_probability') && ...
               isfield(post_state.inconsistency, 'mc_probability')
                delta.inconsistency.delta_mc_probability = ...
                    post_state.inconsistency.mc_probability - pre_state.inconsistency.mc_probability;
            end
            
            % I(theta) changes
            if isfield(pre_state.inconsistency, 'I_theta') && ...
               isfield(post_state.inconsistency, 'I_theta')
                delta.inconsistency.delta_I_theta = ...
                    post_state.inconsistency.I_theta - pre_state.inconsistency.I_theta;
            end
            
            % Intersection volume change
            if isfield(pre_state.inconsistency, 'intersection_volume') && ...
               isfield(post_state.inconsistency, 'intersection_volume')
                delta.inconsistency.delta_intersection_volume = ...
                    post_state.inconsistency.intersection_volume - pre_state.inconsistency.intersection_volume;
            end
            
            % Center distance change
            if isfield(pre_state.inconsistency, 'center_distance') && ...
               isfield(post_state.inconsistency, 'center_distance')
                delta.inconsistency.delta_center_distance = ...
                    post_state.inconsistency.center_distance - pre_state.inconsistency.center_distance;
            end
        end
        
        %% ================================================================
        %  INTERVENTION FUNCTIONS (Self-contained)
        %  ================================================================
        
        function scenario = create_baseline_scenario(dim, uncertainty_scale)
            %CREATE_BASELINE_SCENARIO Create a default baseline scenario
            
            if nargin < 1
                dim = 2;
            end
            if nargin < 2
                uncertainty_scale = 5.0;
            end
            
            % Generate default centers
            source_center = 100 * ones(dim, 1);
            target_center = 100 * ones(dim, 1);
            
            % Generate default generators
            G_src = uncertainty_scale * eye(dim);
            G_tgt = uncertainty_scale * eye(dim);
            
            % Default mapping (identity + small perturbation)
            F = eye(dim) + 0.01 * randn(dim);
            f = 0.1 * randn(dim, 1);
            
            % Create conPolyZono objects
            E = eye(dim);
            A = []; b = []; EC = [];
            
            scenario.source = conPolyZono(source_center, G_src, E, A, b, EC);
            scenario.target = conPolyZono(target_center, G_tgt, E, A, b, EC);
            scenario.mapping = struct('F', F, 'f', f);
        end
        
        function scenario_modified = apply_intervention(scenario, intervention_type, params)
            %APPLY_INTERVENTION Apply an intervention to modify the scenario
            
            scenario_modified = scenario;
            
            switch lower(intervention_type)
                case 'widen'
                    % Increase source uncertainty
                    if isfield(params, 'scale_factor')
                        factor = params.scale_factor;
                    else
                        factor = params.factor;
                    end
                    scenario_modified.source = conPolyZono(...
                        scenario.source.c, ...
                        factor * scenario.source.G, ...
                        scenario.source.E, ...
                        scenario.source.A, ...
                        scenario.source.b, ...
                        scenario.source.EC);
                    
                case 'shrink'
                    % Decrease source uncertainty
                    if isfield(params, 'scale_factor')
                        factor = params.scale_factor;
                    else
                        factor = params.factor;
                    end
                    scenario_modified.source = conPolyZono(...
                        scenario.source.c, ...
                        factor * scenario.source.G, ...
                        scenario.source.E, ...
                        scenario.source.A, ...
                        scenario.source.b, ...
                        scenario.source.EC);
                    
                case 'shift'
                    % Shift source center
                    shift_amount = params.shift;
                    new_center = scenario.source.c + shift_amount;
                    scenario_modified.source = conPolyZono(...
                        new_center, ...
                        scenario.source.G, ...
                        scenario.source.E, ...
                        scenario.source.A, ...
                        scenario.source.b, ...
                        scenario.source.EC);
                    
                case 'rotate'
                    % Rotate generator matrix
                    angle = params.angle;
                    dim = length(scenario.source.c);
                    if dim == 2
                        R = [cos(angle), -sin(angle); sin(angle), cos(angle)];
                    else
                        % For higher dimensions, rotate in first two dims
                        R = eye(dim);
                        R(1:2, 1:2) = [cos(angle), -sin(angle); sin(angle), cos(angle)];
                    end
                    scenario_modified.source = conPolyZono(...
                        scenario.source.c, ...
                        R * scenario.source.G, ...
                        scenario.source.E, ...
                        scenario.source.A, ...
                        scenario.source.b, ...
                        scenario.source.EC);
                    
                case 'correlate'
                    % Introduce correlation in generators
                    if isfield(params, 'correlation_strength')
                        correlation = params.correlation_strength;
                    else
                        correlation = params.correlation;
                    end
                    G = scenario.source.G;
                    dim = size(G, 1);
                    n_gen = size(G, 2);
                    
                    % Add correlated generators
                    if n_gen >= 2
                        G_new = G;
                        G_new(:, 2) = G(:, 1) * correlation + G(:, 2) * sqrt(1 - correlation^2);
                        scenario_modified.source = conPolyZono(...
                            scenario.source.c, ...
                            G_new, ...
                            scenario.source.E, ...
                            scenario.source.A, ...
                            scenario.source.b, ...
                            scenario.source.EC);
                    end
                    
                otherwise
                    error('Unknown intervention type: %s', intervention_type);
            end
        end
        
        %% ================================================================
        %  UTILITY FUNCTIONS (Self-contained)
        %  ================================================================
        
        function vol = compute_volume(Z)
            %COMPUTE_VOLUME Compute set volume using interval hull approximation
            try
                I = interval(Z);
                r = rad(I);
                vol = prod(2 * r);
            catch
                vol = NaN;
            end
        end
        
        function rad_sum = compute_radius(Z)
            %COMPUTE_RADIUS Compute sum of radii
            try
                I = interval(Z);
                r = rad(I);
                rad_sum = sum(r);
            catch
                rad_sum = NaN;
            end
        end
        
        function corr = compute_correlation(G)
            %COMPUTE_CORRELATION Measure of correlation in generator matrix
            if size(G, 2) < 2
                corr = 0;
                return;
            end
            
            % Compute correlation matrix of generator columns
            try
                C = corrcoef(G);
                % Get off-diagonal elements
                off_diag = C - eye(size(C));
                corr = mean(abs(off_diag(:)));
            catch
                corr = 0;
            end
        end
        
        function export_to_json(data, filename)
            %EXPORT_TO_JSON Export results to JSON file for Python analysis
            
            % Prepare data for JSON export
            json_data = causal_experiment_engine_twostep.prepare_for_json(data);
            
            % Write to JSON file
            json_text = jsonencode(json_data, 'PrettyPrint', true);
            
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
        
        function json_data = prepare_for_json(data)
            %PREPARE_FOR_JSON Recursively convert data for JSON export
            
            if isstruct(data)
                fields = fieldnames(data);
                json_data = struct();
                for i = 1:length(fields)
                    field = fields{i};
                    json_data.(field) = causal_experiment_engine_twostep.prepare_for_json(data.(field));
                end
            elseif iscell(data)
                json_data = cell(size(data));
                for i = 1:numel(data)
                    json_data{i} = causal_experiment_engine_twostep.prepare_for_json(data{i});
                end
            elseif isdatetime(data)
                json_data = char(data);
            elseif isobject(data)
                % Convert CORA objects to serializable form
                json_data = sprintf('[%s object]', class(data));
            else
                json_data = data;
            end
        end
        
    end
end
