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
        CONSISTENCY_METHODS = {'jaccard', 'jaccard_mc', 'mc_probability', 'both', 'aabb_and_mc'};
        DEFAULT_METHOD = 'aabb_and_mc';  % AABB Jaccard + MC Probability (includes MFMC) — no jaccard_mc
        DEFAULT_MC_SAMPLES = 2000;
        DEFAULT_SAMPLING_METHOD = 'sobol';  % 'random', 'sobol', 'halton', 'lhs'
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
            if ~isfield(options, 'use_parallel')
                options.use_parallel = false;
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
            
            % Build baseline scenario from scenario_def (use provided mapping, not random F)
            baseline_scenario = struct();
            baseline_scenario.source = scenario_def.source;
            baseline_scenario.target = scenario_def.target;
            if isfield(scenario_def, 'mapping') && isstruct(scenario_def.mapping) && ...
                    isfield(scenario_def.mapping, 'F')
                % Use the mapping provided by the scenario definition (CPS dataset or CONVIDE)
                baseline_scenario.mapping = scenario_def.mapping;
            else
                % Fall back to identity mapping aligned to source/target centers
                dim = length(scenario_def.source.c);
                F   = eye(dim);
                f   = scenario_def.target.c - F * scenario_def.source.c;
                baseline_scenario.mapping = struct('F', F, 'f', f);
            end
            
            % Flatten experiments for optional parallelization
            experiments = [];
            
            % Check if Saltelli mode (struct) or CONVIDE mode (cell array)
            if isstruct(interventions) && isfield(interventions, 'mode') && strcmp(interventions.mode, 'saltelli')
                % SALTELLI MODE: Use compound interventions from samples
                saltelli_samples = interventions.samples;
                param_names = interventions.params;
                
                for sample_idx = 1:height(saltelli_samples)
                    for rep = 1:options.n_repeats
                        exp = struct();
                        exp.intervention_type = 'compound';
                        exp.sample_idx = sample_idx;
                        exp.repeat_idx = rep;
                        % Extract parameter values from table
                        for p = 1:length(param_names)
                            exp.(param_names{p}) = saltelli_samples.(param_names{p})(sample_idx);
                        end
                        experiments = [experiments; exp];
                    end
                end
            else
                % CONVIDE MODE: Use discrete interventions
                for i = 1:length(interventions)
                    int = interventions{i};
                    for val_idx = 1:length(int.values)
                        for rep = 1:options.n_repeats
                            exp = struct();
                            exp.intervention_type = int.type;
                            exp.param_name = int.param;
                            exp.param_value = int.values(val_idx);
                            exp.repeat_idx = rep;
                            experiments = [experiments; exp];
                        end
                    end
                end
            end
            
            n_total = length(experiments);
            if options.verbose
                fprintf('  Total experiments: %d\n', n_total);
                if options.use_parallel && ~isempty(gcp('nocreate'))
                    fprintf('  Parallel mode: ENABLED (parfor will distribute across workers)\n');
                end
            end
            
            zonotope_data = cell(n_total, 1);
            
            % Use parfor if parallel enabled, otherwise regular for
            if options.use_parallel
                progress_queue = [];
                if options.verbose
                    fprintf('  Starting parallel zonotope generation at %s...\n', char(datetime('now', 'Format', 'HH:mm:ss')));
                    tic;
                    % Setup progress monitoring
                    progress_queue = parallel.pool.DataQueue;
                    afterEach(progress_queue, @(~) fprintf('.'));
                end
                parfor exp_id = 1:n_total
                    % Suppress CORA warnings in each worker
                    warning('off', 'all');
                        
                    exp = experiments(exp_id);
                    
                    % Apply intervention (handle both CONVIDE and Saltelli modes)
                    if strcmp(exp.intervention_type, 'compound')
                        % SALTELLI MODE: Compound intervention
                        theta_vector = [exp.scale_factor, exp.center_delta, exp.correlation_strength];
                        param_names = {'scale_factor', 'center_delta', 'correlation_strength'};
                        scenario_modified = causal_experiment_engine_twostep.apply_compound_intervention(...
                            baseline_scenario, theta_vector, param_names);
                    else
                        % CONVIDE MODE: Discrete intervention
                        params = struct(exp.param_name, exp.param_value);
                        scenario_modified = causal_experiment_engine_twostep.apply_intervention(...
                            baseline_scenario, exp.intervention_type, params);
                    end
                        
                        % Propagate modified source
                        F = scenario_modified.mapping.F;
                        f = scenario_modified.mapping.f;
                        Z_propagated = CS_Types.affineMap_cPZ(scenario_modified.source, F, f);
                        
                    % Store zonotope data
                    zono_data = struct();
                    zono_data.exp_id = exp_id;
                    zono_data.scenario_id = scenario_def.id;
                    zono_data.scenario_name = scenario_name;
                    zono_data.scenario_description = scenario_def.name;
                    zono_data.causality_type = scenario_def.type;
                    if isfield(scenario_def, 'dataset_source')
                        zono_data.dataset_source = scenario_def.dataset_source;
                    end
                    if isfield(scenario_def, 'paired_scenario_ids')
                        zono_data.paired_scenario_ids = scenario_def.paired_scenario_ids;
                    end
                    if isfield(scenario_def, 'paired_scenario_names')
                        zono_data.paired_scenario_names = scenario_def.paired_scenario_names;
                    end
                    if isfield(scenario_def, 'consistency_relations')
                        zono_data.consistency_relations = scenario_def.consistency_relations;
                    end
                    if isfield(scenario_def, 'relation_types')
                        zono_data.relation_types = scenario_def.relation_types;
                    end
                    if isfield(scenario_def, 'relation_operators')
                        zono_data.relation_operators = scenario_def.relation_operators;
                    end
                    zono_data.intervention_type = exp.intervention_type;
                    zono_data.repeat_idx = exp.repeat_idx;
                    
                    % Store intervention parameters (different for CONVIDE vs Saltelli)
                    if strcmp(exp.intervention_type, 'compound')
                        zono_data.scale_factor = exp.scale_factor;
                        zono_data.center_delta = exp.center_delta;
                        zono_data.correlation_strength = exp.correlation_strength;
                        zono_data.sample_idx = exp.sample_idx;
                    else
                        zono_data.param_name = exp.param_name;
                        zono_data.intervention_value = exp.param_value;
                    end
                    
                    zono_data.timestamp = datetime('now');
                        
                    % Store zonotopes
                    zono_data.Z_source_pre = baseline_scenario.source;
                    zono_data.Z_target = baseline_scenario.target;
                    zono_data.Z_source_post = scenario_modified.source;
                    zono_data.Z_propagated = Z_propagated;
                    
                    % Store mapping
                    zono_data.mapping_F = F;
                    zono_data.mapping_f = f;
                    % Store UPR type and params (for type-specific propagation in Step 2)
                    if isfield(scenario_def, 'upr_type')
                        zono_data.upr_type   = scenario_def.upr_type;
                    else
                        zono_data.upr_type   = 'parametric';
                    end
                    if isfield(scenario_def, 'upr_params')
                        zono_data.upr_params = scenario_def.upr_params;
                    end
                    
                    % Save to file (use -fromstruct for parfor compatibility)
                    filename = sprintf('zonotopes_%s_exp%04d.mat', scenario_name, exp_id);
                    filepath = fullfile(output_dir, filename);
                    save(filepath, '-fromstruct', zono_data);
                    
                    % Return metadata
                    zono_metadata = struct();
                    zono_metadata.exp_id = exp_id;
                    zono_metadata.scenario_name = scenario_name;
                    zono_metadata.intervention_type = exp.intervention_type;
                    zono_metadata.repeat_idx = exp.repeat_idx;
                    
                    % Store intervention value (different for CONVIDE vs Saltelli) 
                    if strcmp(exp.intervention_type, 'compound')
                        zono_metadata.scale_factor = exp.scale_factor;
                        zono_metadata.center_delta = exp.center_delta;
                        zono_metadata.correlation_strength = exp.correlation_strength;
                        zono_metadata.sample_idx = exp.sample_idx;
                    else
                        zono_metadata.intervention_value = exp.param_value;
                    end
                    
                    zono_metadata.filepath = filepath;
                    
                    zonotope_data{exp_id} = zono_metadata;
                    
                    % Send progress update
                    if options.verbose
                        send(progress_queue, exp_id);
                    end
                end
                
                if options.verbose
                    fprintf('\n');  % New line after dots
                    elapsed = toc;
                    fprintf('  ✓ Parallel generation complete! %d files in %.1f min (%.1f files/min)\n', ...
                        n_total, elapsed/60, n_total/(elapsed/60));
                end
            else
                for exp_id = 1:n_total
                    % Suppress CORA warnings
                    warning('off', 'all');
                    
                    exp = experiments(exp_id);
                    
                    if options.verbose && mod(exp_id, 20) == 0
                        fprintf('    Progress: %d/%d\n', exp_id, n_total);
                    end
                    
                    % Apply intervention (handle both CONVIDE and Saltelli modes)
                    if strcmp(exp.intervention_type, 'compound')
                        % SALTELLI MODE: Compound intervention
                        theta_vector = [exp.scale_factor, exp.center_delta, exp.correlation_strength];
                        param_names = {'scale_factor', 'center_delta', 'correlation_strength'};
                        scenario_modified = causal_experiment_engine_twostep.apply_compound_intervention(...
                            baseline_scenario, theta_vector, param_names);
                    else
                        % CONVIDE MODE: Discrete intervention
                        params = struct(exp.param_name, exp.param_value);
                        scenario_modified = causal_experiment_engine_twostep.apply_intervention(...
                            baseline_scenario, exp.intervention_type, params);
                    end
                    
                    % Propagate modified source
                    F = scenario_modified.mapping.F;
                    f = scenario_modified.mapping.f;
                    Z_propagated = CS_Types.affineMap_cPZ(scenario_modified.source, F, f);
                    
                    % Store zonotope data
                    zono_data = struct();
                    zono_data.exp_id = exp_id;
                    zono_data.scenario_id = scenario_def.id;
                    zono_data.scenario_name = scenario_name;
                    zono_data.scenario_description = scenario_def.name;
                    zono_data.causality_type = scenario_def.type;
                    if isfield(scenario_def, 'dataset_source')
                        zono_data.dataset_source = scenario_def.dataset_source;
                    end
                    if isfield(scenario_def, 'paired_scenario_ids')
                        zono_data.paired_scenario_ids = scenario_def.paired_scenario_ids;
                    end
                    if isfield(scenario_def, 'paired_scenario_names')
                        zono_data.paired_scenario_names = scenario_def.paired_scenario_names;
                    end
                    if isfield(scenario_def, 'consistency_relations')
                        zono_data.consistency_relations = scenario_def.consistency_relations;
                    end
                    if isfield(scenario_def, 'relation_types')
                        zono_data.relation_types = scenario_def.relation_types;
                    end
                    if isfield(scenario_def, 'relation_operators')
                        zono_data.relation_operators = scenario_def.relation_operators;
                    end
                    zono_data.intervention_type = exp.intervention_type;
                    zono_data.repeat_idx = exp.repeat_idx;
                    
                    % Store intervention parameters (different for CONVIDE vs Saltelli)
                    if strcmp(exp.intervention_type, 'compound')
                        zono_data.scale_factor = exp.scale_factor;
                        zono_data.center_delta = exp.center_delta;
                        zono_data.correlation_strength = exp.correlation_strength;
                        zono_data.sample_idx = exp.sample_idx;
                    else
                        zono_data.param_name = exp.param_name;
                        zono_data.intervention_value = exp.param_value;
                    end
                    zono_data.timestamp = datetime('now');
                    
                    % Store zonotopes
                    zono_data.Z_source_pre = baseline_scenario.source;
                    zono_data.Z_target = baseline_scenario.target;
                    zono_data.Z_source_post = scenario_modified.source;
                    zono_data.Z_propagated = Z_propagated;
                    
                    % Store mapping
                    zono_data.mapping_F = F;
                    zono_data.mapping_f = f;
                    % Store UPR type and params (for type-specific propagation in Step 2)
                    if isfield(scenario_def, 'upr_type')
                        zono_data.upr_type   = scenario_def.upr_type;
                    else
                        zono_data.upr_type   = 'parametric';
                    end
                    if isfield(scenario_def, 'upr_params')
                        zono_data.upr_params = scenario_def.upr_params;
                    end
                    
                    % Save to file
                    filename = sprintf('zonotopes_%s_exp%04d.mat', scenario_name, exp_id);
                    filepath = fullfile(output_dir, filename);
                    save(filepath, '-fromstruct', zono_data);
                    
                    % Return metadata
                    zono_metadata = struct();
                    zono_metadata.exp_id = exp_id;
                    zono_metadata.scenario_name = scenario_name;
                    zono_metadata.intervention_type = exp.intervention_type;
                    zono_metadata.repeat_idx = exp.repeat_idx;
                    
                    % Store intervention value (different for CONVIDE vs Saltelli)
                    if strcmp(exp.intervention_type, 'compound')
                        zono_metadata.scale_factor = exp.scale_factor;
                        zono_metadata.center_delta = exp.center_delta;
                        zono_metadata.correlation_strength = exp.correlation_strength;
                        zono_metadata.sample_idx = exp.sample_idx;
                    else
                        zono_metadata.intervention_value = exp.param_value;
                    end
                    
                    zono_metadata.filepath = filepath;
                    
                    zonotope_data{exp_id} = zono_metadata;
                end
            end
            
            exp_count = n_total;
            
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
            %       .scenario_ids    - (optional) Numeric scenario IDs to process
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
            if ~isfield(consistency_options, 'sampling_method')
                consistency_options.sampling_method = {'sobol', 'halton'};  % Both QMC methods by default
            end
            % Convert single string to cell array for uniform processing
            if ischar(consistency_options.sampling_method)
                consistency_options.sampling_method = {consistency_options.sampling_method};
            end
            if ~isfield(consistency_options, 'verbose')
                consistency_options.verbose = true;
            end
            if ~isfield(consistency_options, 'use_parallel')
                consistency_options.use_parallel = false;
            end
            % Non-uniform inner sampling (optional; default OFF = auto Gaussian)
            if ~isfield(consistency_options, 'use_nonuniform_inner')
                consistency_options.use_nonuniform_inner = false;
            end
            if ~isfield(consistency_options, 'inner_distribution')
                consistency_options.inner_distribution = 'auto';  % 'auto' | 'gaussian' | 'truncated_normal'
            end
            if ~isfield(consistency_options, 'inner_dist_params')
                consistency_options.inner_dist_params = struct();  % struct with .mu, .Sigma
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
                if isfield(consistency_options, 'scenario_ids') && ...
                   ~isempty(consistency_options.scenario_ids)
                    if ~ismember(zonotope_index.scenario_def.id, consistency_options.scenario_ids)
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
                        if iscell(consistency_options.sampling_method)
                            fprintf('  Sampling methods: %s\n', strjoin(consistency_options.sampling_method, ', '));
                        else
                            fprintf('  Sampling method: %s\n', consistency_options.sampling_method);
                        end
                    end
                    fprintf('\n');
                end
                
                all_results = cell(1, n_total_exp);
                scenario_start_time = tic;
                
                % Process each zonotope file with optional parallelization
                if consistency_options.use_parallel
                    if consistency_options.verbose
                        fprintf('  Starting parallel consistency computation at %s...\n', ...
                            char(datetime('now', 'Format', 'HH:mm:ss')));
                    end
                    
                    % Setup progress monitoring for parallel execution
                    progress_queue = [];
                    if consistency_options.verbose
                        progress_queue = parallel.pool.DataQueue;
                        afterEach(progress_queue, @(~) fprintf('.'));
                    end

                    parfor exp_idx = 1:n_total_exp
                        % Suppress CORA warnings in each worker
                        warning('off', 'all');
                        
                        zono_meta = zonotope_index.zonotope_files{exp_idx};
                        
                        % Load zonotope data
                        zono_data = load(zono_meta.filepath);
                    
                    % Measure pre-intervention state
                    pre_scenario = struct();
                    pre_scenario.source = zono_data.Z_source_pre;
                    pre_scenario.target = zono_data.Z_target;
                    pre_scenario.mapping = struct('F', zono_data.mapping_F, 'f', zono_data.mapping_f);
                    if isfield(zono_data, 'upr_type'),   pre_scenario.upr_type   = zono_data.upr_type;   end
                    if isfield(zono_data, 'upr_params'), pre_scenario.upr_params = zono_data.upr_params; end

                    pre_state = causal_experiment_engine_twostep.measure_state(...
                        pre_scenario, consistency_options);

                    % Measure post-intervention state
                    post_scenario = struct();
                    post_scenario.source = zono_data.Z_source_post;
                    post_scenario.target = zono_data.Z_target;
                    post_scenario.mapping = struct('F', zono_data.mapping_F, 'f', zono_data.mapping_f);
                    if isfield(zono_data, 'upr_type'),   post_scenario.upr_type   = zono_data.upr_type;   end
                    if isfield(zono_data, 'upr_params'), post_scenario.upr_params = zono_data.upr_params; end

                    post_state = causal_experiment_engine_twostep.measure_state(...
                        post_scenario, consistency_options);
                    
                    % Compute causal effect
                    causal_effect = causal_experiment_engine_twostep.compute_delta(...
                        pre_state, post_state);
                    
                    % Build result struct
                    result = struct();
                    result.exp_id = zono_data.exp_id;
                    result.scenario_type = zono_data.scenario_name;
                    result.scenario_description = zono_data.scenario_description;
                    result.causality_type = zono_data.causality_type;
                    if isfield(zono_data, 'dataset_source')
                        result.dataset_source = zono_data.dataset_source;
                    end
                    if isfield(zono_data, 'paired_scenario_ids')
                        result.paired_scenario_ids = zono_data.paired_scenario_ids;
                    end
                    if isfield(zono_data, 'paired_scenario_names')
                        result.paired_scenario_names = zono_data.paired_scenario_names;
                    end
                    if isfield(zono_data, 'consistency_relations')
                        result.consistency_relations = zono_data.consistency_relations;
                    end
                    if isfield(zono_data, 'relation_types')
                        result.relation_types = zono_data.relation_types;
                    end
                    if isfield(zono_data, 'relation_operators')
                        result.relation_operators = zono_data.relation_operators;
                    end
                    result.intervention_type = zono_data.intervention_type;
                    result.intervention_direction = 'forward';
                    
                    % Store intervention parameters (different for CONVIDE vs Saltelli)
                    if strcmp(zono_data.intervention_type, 'compound')
                        result.scale_factor = zono_data.scale_factor;
                        result.center_delta = zono_data.center_delta;
                        result.correlation_strength = zono_data.correlation_strength;
                        if isfield(zono_data, 'sample_idx')
                            result.sample_idx = zono_data.sample_idx;
                        end
                    else
                        result.intervention_value = zono_data.intervention_value;
                    end
                    
                    result.repeat_idx = zono_data.repeat_idx;
                    result.run_id = zono_data.exp_id;
                    
                    result.pre_state = pre_state;
                    result.post_state = post_state;
                    result.causal_effect = causal_effect;
                    
                    result.consistency_method = consistency_options.method;
                    result.measurement_timestamp = datetime('now');
                    
                    all_results{exp_idx} = result;
                    
                    % Send progress update
                    if consistency_options.verbose
                        send(progress_queue, exp_idx);
                    end
                    end
                    
                    if consistency_options.verbose
                        fprintf('\n');  % New line after dots
                        elapsed = toc(scenario_start_time);
                        fprintf('  ✓ Parallel consistency complete! %d experiments in %.1f min (%.1f exp/min)\n', ...
                            n_total_exp, elapsed/60, n_total_exp/(elapsed/60));
                    end
                else
                    % Sequential processing with progress reporting
                    for exp_idx = 1:n_total_exp
                        % Suppress CORA warnings
                        warning('off', 'all');
                        
                        zono_meta = zonotope_index.zonotope_files{exp_idx};
                        
                        % Load zonotope data
                        zono_data = load(zono_meta.filepath);
                    
                        exp_start_time = tic;
                    
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
                        if isfield(zono_data, 'dataset_source')
                            result.dataset_source = zono_data.dataset_source;
                        end
                        if isfield(zono_data, 'paired_scenario_ids')
                            result.paired_scenario_ids = zono_data.paired_scenario_ids;
                        end
                        if isfield(zono_data, 'paired_scenario_names')
                            result.paired_scenario_names = zono_data.paired_scenario_names;
                        end
                        if isfield(zono_data, 'consistency_relations')
                            result.consistency_relations = zono_data.consistency_relations;
                        end
                        if isfield(zono_data, 'relation_types')
                            result.relation_types = zono_data.relation_types;
                        end
                        if isfield(zono_data, 'relation_operators')
                            result.relation_operators = zono_data.relation_operators;
                        end
                        result.intervention_type = zono_data.intervention_type;
                        result.intervention_direction = 'forward';
                        
                        % Store intervention parameters (different for CONVIDE vs Saltelli)
                        if strcmp(zono_data.intervention_type, 'compound')
                            result.scale_factor = zono_data.scale_factor;
                            result.center_delta = zono_data.center_delta;
                            result.correlation_strength = zono_data.correlation_strength;
                            if isfield(zono_data, 'sample_idx')
                                result.sample_idx = zono_data.sample_idx;
                            end
                        else
                            result.intervention_value = zono_data.intervention_value;
                        end
                        
                        result.repeat_idx = zono_data.repeat_idx;
                        result.run_id = zono_data.exp_id;
                        
                        result.pre_state = pre_state;
                        result.post_state = post_state;
                        result.causal_effect = causal_effect;
                        
                        result.consistency_method = consistency_options.method;
                        result.measurement_timestamp = datetime('now');
                        
                        all_results{exp_idx} = result;
                    end
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
                if isfield(zonotope_index.scenario_def, 'dataset_source')
                    metadata.dataset_source = zonotope_index.scenario_def.dataset_source;
                end
                if isfield(zonotope_index.scenario_def, 'paired_scenario_ids')
                    metadata.paired_scenario_ids = zonotope_index.scenario_def.paired_scenario_ids;
                end
                if isfield(zonotope_index.scenario_def, 'paired_scenario_names')
                    metadata.paired_scenario_names = zonotope_index.scenario_def.paired_scenario_names;
                end
                if isfield(zonotope_index.scenario_def, 'consistency_relations')
                    metadata.consistency_relations = zonotope_index.scenario_def.consistency_relations;
                end
                if isfield(zonotope_index.scenario_def, 'relation_types')
                    metadata.relation_types = zonotope_index.scenario_def.relation_types;
                end
                if isfield(zonotope_index.scenario_def, 'relation_operators')
                    metadata.relation_operators = zonotope_index.scenario_def.relation_operators;
                end
                metadata.consistency_method = consistency_options.method;
                metadata.mc_samples = consistency_options.mc_samples;
                metadata.use_nonuniform_inner = consistency_options.use_nonuniform_inner;
                metadata.inner_distribution = consistency_options.inner_distribution;
                if consistency_options.use_nonuniform_inner
                    metadata.inner_dist_params = consistency_options.inner_dist_params;
                end
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
                state.uncertainty.source_generators = Z_src.G;  % needed for GNN inference
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
                state.uncertainty.target_generators = Z_tgt.G;  % needed for GNN inference
                state.uncertainty.target_n_generators = size(Z_tgt.G, 2);
            end
            
            % === INCONSISTENCY METRICS ===
            % NOTE: Dual representation for unified analysis:
            %   - Raw metrics: jaccard_index, mc_probability (consistency: 0=bad, 1=good)
            %   - Derived metrics: jaccard_inconsistency, mc_p_inconsistent, I_theta (inconsistency: 0=good, 1=bad)
            %   - All *_inconsistency and I_theta metrics increase in the same direction for sensitivity analysis
            state.inconsistency = struct();
            
            % Propagate and check for inconsistencies
            if isfield(scenario, 'mapping') && isfield(scenario, 'source') && isobject(scenario.source)
                try
                    F = scenario.mapping.F;
                    f = scenario.mapping.f;

                    % ── UPR-type-specific propagation ──────────────────────────
                    upr_type = '';
                    if isfield(scenario, 'upr_type')
                        upr_type = lower(strtrim(char(string(scenario.upr_type))));
                    end

                    % For constraint_based: clip source to the safety bound
                    % before propagating (σ(x) = x if g(x) ≤ 0, else ∅).
                    % We use an interval-hull clip: conservative but exact in 1-D.
                    Z_source_eff = scenario.source;
                    if strcmp(upr_type, 'constraint_based') && ...
                            isfield(scenario, 'upr_params') && ...
                            isfield(scenario.upr_params, 'constraint_bound')
                        try
                            bounds = double(scenario.upr_params.constraint_bound);
                            dim    = length(bounds);
                            I_src  = interval(Z_source_eff);
                            lo_src = infimum(I_src);
                            hi_src = supremum(I_src);
                            hi_clipped = min(hi_src, bounds(:));
                            % If the source already violates the bound entirely,
                            % the propagated set is empty → use a degenerate zonotope.
                            if any(lo_src > hi_clipped)
                                % Empty set: degenerate point outside target
                                c_empty = bounds(:) + 1e6 * ones(dim, 1);
                                Z_source_eff = conZonotope(c_empty, zeros(dim,1), [], []);
                            else
                                c_clip = (lo_src + hi_clipped) / 2;
                                G_clip = diag((hi_clipped - lo_src) / 2);
                                Z_source_eff = conZonotope(c_clip, G_clip, [], []);
                            end
                        catch
                            % Fall back to unclipped source on error
                        end
                    end

                    Z_propagated = CS_Types.affineMap_cPZ(Z_source_eff, F, f);
                    
                    state.inconsistency.propagation_success = true;
                    state.inconsistency.propagated_volume = ...
                        causal_experiment_engine_twostep.compute_volume(Z_propagated);
                    
                    % Check if propagated set is empty
                    state.inconsistency.is_empty = isEmptySet(Z_propagated);
                    
                    % If target exists, compute consistency scores
                    if isfield(scenario, 'target') && isobject(scenario.target)
                        method = lower(options.method);
                        
                        % === JACCARD METHOD (AABB) ===
                        if strcmp(method, 'jaccard') || strcmp(method, 'both') || strcmp(method, 'aabb_and_mc')
                            try
                                F_identity = eye(size(scenario.target.c, 1));
                                f_identity = zeros(size(scenario.target.c, 1), 1);

                                t_jaccard = tic;
                                [C, Csym, details] = score_jaccard(...
                                    Z_propagated, scenario.target, F_identity, f_identity, ...
                                    struct('return_details', true, 'verbose', false));
                                state.inconsistency.timing_jaccard_s = toc(t_jaccard);

                                state.inconsistency.jaccard_C = C;
                                state.inconsistency.jaccard_Csym = Csym;
                                state.inconsistency.jaccard_index = Csym;
                                state.inconsistency.jaccard_inconsistency = 1 - Csym;  % Derived: unified direction (0=consistent, 1=inconsistent)
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
                                state.inconsistency.jaccard_inconsistency = NaN;
                            end
                        end
                        
                        % === JACCARD METHOD (MC with QMC support) ===
                        if strcmp(method, 'jaccard_mc') || strcmp(method, 'both')
                            sampling_methods = options.sampling_method;
                            for sm_idx = 1:length(sampling_methods)
                                samp_method = sampling_methods{sm_idx};
                                try
                                    % Create options struct for score_jaccard_mc
                                    jac_mc_opts = struct();
                                    jac_mc_opts.num_samples = options.mc_samples;
                                    jac_mc_opts.sampling_method = samp_method;
                                    jac_mc_opts.return_details = true;
                                    jac_mc_opts.verbose = false;

                                    t_jmc = tic;
                                    [jac_mc_score, jac_mc_details] = score_jaccard_mc(...
                                        Z_propagated, scenario.target, jac_mc_opts);
                                    state.inconsistency.(['timing_jaccard_mc_' samp_method '_s']) = toc(t_jmc);
                                    
                                    % Save with method-specific field names
                                    field_suffix = ['_' samp_method];
                                    state.inconsistency.(['jaccard_mc_index' field_suffix]) = jac_mc_score;
                                    state.inconsistency.(['jaccard_mc_inconsistency' field_suffix]) = 1 - jac_mc_score;  % Derived: unified direction
                                    state.inconsistency.(['jaccard_mc_vol_AB_est' field_suffix]) = jac_mc_details.vol_AB_est;
                                    state.inconsistency.(['jaccard_mc_vol_union_est' field_suffix]) = jac_mc_details.vol_union_est;
                                    state.inconsistency.(['jaccard_mc_num_samples' field_suffix]) = jac_mc_details.N;
                                catch ME
                                    warning('Jaccard MC scoring failed for %s: %s', samp_method, ME.message);
                                    state.inconsistency.(['jaccard_mc_index_' samp_method]) = NaN;
                                    state.inconsistency.(['jaccard_mc_inconsistency_' samp_method]) = NaN;
                                end
                            end
                        end
                        
                        % === MONTE CARLO PROBABILITY METHOD (Multi-set with QMC) ===
                        if strcmp(method, 'mc_probability') || strcmp(method, 'both') || strcmp(method, 'aabb_and_mc')
                            sampling_methods = options.sampling_method;
                            for sm_idx = 1:length(sampling_methods)
                                samp_method = sampling_methods{sm_idx};
                                try
                                    % Create options struct for score_mc_probability
                                    mc_opts = struct();
                                    mc_opts.num_samples = options.mc_samples;
                                    mc_opts.sampling_method = samp_method;
                                    mc_opts.return_details = true;
                                    mc_opts.verbose = false;
                                    % Pass explicit inner distribution when enabled
                                    if options.use_nonuniform_inner && ~isempty(fieldnames(options.inner_dist_params))
                                        mc_opts.distribution = options.inner_dist_params;
                                    end

                                    t_mc = tic;
                                    [mc_score, mc_details] = score_mc_probability(...
                                        {Z_propagated, scenario.target}, mc_opts);
                                    state.inconsistency.(['timing_mc_' samp_method '_s']) = toc(t_mc);
                                    
                                    % Save with method-specific field names
                                    field_suffix = ['_' samp_method];
                                    state.inconsistency.(['mc_probability' field_suffix]) = mc_score;
                                    state.inconsistency.(['mc_p_consistent' field_suffix]) = mc_details.p_consistent;
                                    state.inconsistency.(['mc_p_inconsistent' field_suffix]) = mc_details.p_inconsistent;
                                    state.inconsistency.(['mc_num_samples' field_suffix]) = mc_details.num_samples;
                                    state.inconsistency.(['mc_num_consistent' field_suffix]) = mc_details.num_consistent;
                                    state.inconsistency.(['mc_standard_error' field_suffix]) = mc_details.standard_error;
                                    state.inconsistency.(['mc_ci95_lower' field_suffix]) = mc_details.ci95_lower;
                                    state.inconsistency.(['mc_ci95_upper' field_suffix]) = mc_details.ci95_upper;
                                    % MFMC corrected estimate:
                                    %  - No intersection: I_MF = 1.0 exactly (certain inconsistency,
                                    %    AABB already proves the sets are disjoint).
                                    %  - Intersection + valid I_MF: use MFMC estimate.
                                    %  - Intersection + NaN I_MF: fall back to plain MC.
                                    has_intersection = false;
                                    if isfield(state.inconsistency, 'has_intersection')
                                        hi = state.inconsistency.has_intersection;
                                        if islogical(hi)
                                            has_intersection = hi;
                                        elseif isnumeric(hi)
                                            has_intersection = (hi ~= 0);
                                        end
                                    end
                                    if ~has_intersection
                                        state.inconsistency.(['I_MF_' samp_method]) = 1.0;
                                    elseif isfield(mc_details, 'I_MF') && ~isnan(mc_details.I_MF)
                                        state.inconsistency.(['I_MF_' samp_method]) = mc_details.I_MF;
                                    else
                                        % Fallback: plain MC when MFMC correction is numerically invalid
                                        state.inconsistency.(['I_MF_' samp_method]) = mc_details.p_inconsistent;
                                    end
                                    % Convergence curves: MC and MFMC with measured wall-clock timing
                                    if isfield(mc_details, 'mc_convergence_checkpoints') && ...
                                            ~isempty(mc_details.mc_convergence_checkpoints)
                                        chk_pts = mc_details.mc_convergence_checkpoints;
                                        state.inconsistency.mc_convergence_checkpoints              = chk_pts;
                                        state.inconsistency.(['mc_convergence_'      samp_method])  = mc_details.mc_convergence_mc;
                                        if has_intersection
                                            state.inconsistency.(['mfmc_convergence_'    samp_method])  = mc_details.mc_convergence_mfmc;
                                        end
                                        state.inconsistency.(['mc_convergence_timing_' samp_method]) = mc_details.mc_convergence_timing_s;

                                        % End-to-end MFMC runtime = AABB time + MC time
                                        % at the first crossover checkpoint.
                                        mfmc_total_s = NaN;
                                        mfmc_mc_until_stop_s = NaN;
                                        mfmc_stop_n = NaN;
                                        try
                                            mc_curve   = mc_details.mc_convergence_mc;
                                            mfmc_curve = mc_details.mc_convergence_mfmc;
                                            t_curve    = mc_details.mc_convergence_timing_s;
                                            if ~isempty(mc_curve) && ~isempty(mfmc_curve) && ~isempty(t_curve)
                                                target = mc_curve(end);
                                                tol    = max(std(mc_curve), 1e-3);
                                                crossed = find(abs(mfmc_curve - target) <= tol, 1, 'first');
                                                if isempty(crossed)
                                                    crossed = find(abs(mfmc_curve - target) <= 2 * tol, 1, 'first');
                                                end
                                                if ~isempty(crossed) && isfield(state.inconsistency, 'timing_jaccard_s')
                                                    mfmc_mc_until_stop_s = t_curve(crossed);
                                                    mfmc_stop_n = chk_pts(crossed);
                                                    mfmc_total_s = state.inconsistency.timing_jaccard_s + mfmc_mc_until_stop_s;
                                                end
                                            end
                                        catch
                                            mfmc_total_s = NaN;
                                            mfmc_mc_until_stop_s = NaN;
                                            mfmc_stop_n = NaN;
                                        end

                                        if has_intersection && ~isnan(mfmc_total_s)
                                            aabb_s = state.inconsistency.timing_jaccard_s;
                                            state.inconsistency.(['timing_mfmc_' samp_method '_s'])  = mfmc_total_s;
                                            state.inconsistency.(['timing_mfmc_' samp_method '_ms']) = 1000.0 * mfmc_total_s;
                                            state.inconsistency.(['timing_mfmc_total_' samp_method '_s'])  = mfmc_total_s;
                                            state.inconsistency.(['timing_mfmc_total_' samp_method '_ms']) = 1000.0 * mfmc_total_s;
                                            state.inconsistency.(['timing_mfmc_aabb_' samp_method '_s'])   = aabb_s;
                                            state.inconsistency.(['timing_mfmc_aabb_' samp_method '_ms'])  = 1000.0 * aabb_s;
                                            state.inconsistency.(['timing_mfmc_mc_until_stop_' samp_method '_s'])  = mfmc_mc_until_stop_s;
                                            state.inconsistency.(['timing_mfmc_mc_until_stop_' samp_method '_ms']) = 1000.0 * mfmc_mc_until_stop_s;
                                            state.inconsistency.(['timing_mfmc_stop_samples_' samp_method]) = mfmc_stop_n;
                                            if strcmp(samp_method, 'sobol')
                                                state.inconsistency.timing_mfmc_s  = mfmc_total_s;
                                                state.inconsistency.timing_mfmc_ms = 1000.0 * mfmc_total_s;
                                                state.inconsistency.timing_I_MF_s  = mfmc_total_s;
                                                state.inconsistency.timing_I_MF_ms = 1000.0 * mfmc_total_s;
                                                state.inconsistency.timing_mfmc_total_s = mfmc_total_s;
                                                state.inconsistency.timing_mfmc_total_ms = 1000.0 * mfmc_total_s;
                                                state.inconsistency.timing_mfmc_aabb_s = aabb_s;
                                                state.inconsistency.timing_mfmc_aabb_ms = 1000.0 * aabb_s;
                                                state.inconsistency.timing_mfmc_mc_until_stop_s = mfmc_mc_until_stop_s;
                                                state.inconsistency.timing_mfmc_mc_until_stop_ms = 1000.0 * mfmc_mc_until_stop_s;
                                                state.inconsistency.timing_mfmc_stop_samples = mfmc_stop_n;
                                            end
                                        end
                                    end
                                catch ME
                                    warning('Monte Carlo scoring failed for %s: %s', samp_method, ME.message);
                                    state.inconsistency.(['mc_probability_' samp_method]) = NaN;
                                    state.inconsistency.(['mc_p_consistent_' samp_method]) = NaN;
                                    state.inconsistency.(['mc_p_inconsistent_' samp_method]) = NaN;
                                end

                                % === MFMC ADAPTIVE (same sampling method, fewer expensive samples) ===
                                % Runs the containment loop with adaptive early stopping once the
                                % MFMC SE drops below target_se.  Provides a real measured runtime
                                % instead of a post-hoc crossover estimate.
                                try
                                    mfmc_opts = struct();
                                    mfmc_opts.num_samples    = options.mc_samples;
                                    mfmc_opts.sampling_method = samp_method;
                                    mfmc_opts.return_details  = true;
                                    mfmc_opts.verbose         = false;
                                    mfmc_opts.use_mfmc        = true;
                                    mfmc_opts.target_se       = 0.02;
                                    if options.use_nonuniform_inner && ~isempty(fieldnames(options.inner_dist_params))
                                        mfmc_opts.distribution = options.inner_dist_params;
                                    end

                                    t_mfmc_adaptive = tic;
                                    [mfmc_adaptive_score, mfmc_adaptive_details] = score_mc_probability( ...
                                        {Z_propagated, scenario.target}, mfmc_opts);
                                    state.inconsistency.(['timing_mfmc_adaptive_' samp_method '_s']) = toc(t_mfmc_adaptive);

                                    state.inconsistency.(['I_MF_adaptive_' samp_method])       = mfmc_adaptive_score;
                                    state.inconsistency.(['mfmc_adaptive_n_used_' samp_method]) = mfmc_adaptive_details.n_used;
                                    state.inconsistency.(['mfmc_adaptive_converged_' samp_method]) = ...
                                        mfmc_adaptive_details.mfmc_adaptive_converged;
                                catch ME
                                    warning('MFMC adaptive scoring failed for %s: %s', samp_method, ME.message);
                                    state.inconsistency.(['I_MF_adaptive_' samp_method])        = NaN;
                                    state.inconsistency.(['mfmc_adaptive_n_used_' samp_method]) = NaN;
                                end
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
                                i_theta_method = 'mc';  % Use MC for 'both' to match mc_probability values
                            end
                            
                            t_itheta = tic;
                            gi_args = {'n_samples', options.mc_samples, ...
                                       'method', i_theta_method, ...
                                       'return_details', true, ...
                                       'verbose', false};
                            if options.use_nonuniform_inner && ~isempty(fieldnames(options.inner_dist_params))
                                gi_args = [gi_args, {'inner_dist_params', options.inner_dist_params}];
                            end
                            [I_theta, I_details] = global_inconsistency(all_models, gi_args{:});
                            state.inconsistency.timing_I_theta_s = toc(t_itheta);

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
            
            % Jaccard inconsistency changes (derived metric)
            if isfield(pre_state.inconsistency, 'jaccard_inconsistency') && ...
               isfield(post_state.inconsistency, 'jaccard_inconsistency')
                delta.inconsistency.delta_jaccard_inconsistency = ...
                    post_state.inconsistency.jaccard_inconsistency - pre_state.inconsistency.jaccard_inconsistency;
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
            
            % Create conZonotope objects
            scenario.source = conZonotope(source_center, G_src, [], []);
            scenario.target = conZonotope(target_center, G_tgt, [], []);
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
                    scenario_modified.source = conZonotope(...
                        scenario.source.c, ...
                        factor * scenario.source.G, ...
                        [], []);
                    
                case 'shrink'
                    % Decrease source uncertainty
                    if isfield(params, 'scale_factor')
                        factor = params.scale_factor;
                    else
                        factor = params.factor;
                    end
                    scenario_modified.source = conZonotope(...
                        scenario.source.c, ...
                        factor * scenario.source.G, ...
                        [], []);
                    
                case 'shift'
                    % Shift source center by delta generator-widths per dimension.
                    % new_center = c + delta * diag(G)
                    % This is scale-invariant: delta=1 moves the center by exactly
                    % one generator half-width, regardless of the center magnitude.
                    % Supports zero-center scenarios and bidirectional shifts (delta < 0).
                    if isfield(params, 'center_delta')
                        shift_fraction = params.center_delta;
                    elseif isfield(params, 'shift')
                        shift_fraction = params.shift;
                    else
                        shift_fraction = params.delta;
                    end
                    % Per-dimension generator widths (diagonal of G for box generators)
                    gen_widths = diag(scenario.source.G);
                    if isscalar(shift_fraction)
                        shift_vec = shift_fraction * gen_widths;
                    else
                        shift_vec = shift_fraction .* gen_widths;
                    end
                    new_center = scenario.source.c + shift_vec;
                    scenario_modified.source = conZonotope(...
                        new_center, ...
                        scenario.source.G, ...
                        [], []);
                    
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
                    scenario_modified.source = conZonotope(...
                        scenario.source.c, ...
                        R * scenario.source.G, ...
                        [], []);
                    
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
                        scenario_modified.source = conZonotope(...
                            scenario.source.c, ...
                            G_new, ...
                            [], []);
                    end
                    
                otherwise
                    error('Unknown intervention type: %s', intervention_type);
            end
        end
        
        function scenario_modified = apply_compound_intervention(scenario, theta_vector, param_names, options)
            %APPLY_COMPOUND_INTERVENTION Apply multiple interventions simultaneously
            %
            % This function applies multiple intervention parameters at once, which
            % is required for Saltelli sampling in global sensitivity analysis.
            %
            % Inputs:
            %   scenario      - Baseline scenario struct
            %   theta_vector  - [1×p] vector of parameter values (e.g., [1.5, 0.1, 0.7])
            %   param_names   - {1×p} cell array of parameter names
            %                   Options: 'scale_factor', 'center_delta', 'correlation_strength'
            %   options       - (optional) struct with:
            %                   .shift_direction: 'x', 'y', 'radial' (default: 'radial')
            %                   .verbose: display progress (default: false)
            %
            % Outputs:
            %   scenario_modified - Modified scenario with all interventions applied
            %
            % Example:
            %   theta = [1.5, 0.05, 0.7];  % scale=1.5, shift=0.05, correlation=0.7
            %   params = {'scale_factor', 'center_delta', 'correlation_strength'};
            %   scenario_mod = causal_experiment_engine_twostep.apply_compound_intervention(...
            %       scenario, theta, params);
            %
            % Notes:
            %   - Interventions are applied in order: scale → shift → correlation
            %   - scale_factor < 1 = shrink, > 1 = widen
            %   - center_delta is shift in generator-width units: new_c = c + delta*diag(G)
            %   - correlation_strength ∈ [0,1]
            
            if nargin < 4
                options = struct();
            end
            if ~isfield(options, 'shift_direction')
                options.shift_direction = 'radial';
            end
            if ~isfield(options, 'verbose')
                options.verbose = false;
            end
            
            % Validate inputs
            assert(length(theta_vector) == length(param_names), ...
                'theta_vector length must match param_names length');
            
            % Start with baseline scenario
            scenario_modified = scenario;
            
            if options.verbose
                fprintf('  Compound: ');
                for i = 1:length(param_names)
                    fprintf('%s=%.3f ', param_names{i}, theta_vector(i));
                end
            end
            
            % Apply interventions in order: scale → shift → correlation
            for i = 1:length(param_names)
                param_name = param_names{i};
                param_value = theta_vector(i);
                
                switch param_name
                    case 'scale_factor'
                        % Scale intervention (widen if >1, shrink if <1)
                        params.scale_factor = param_value;
                        if param_value >= 1.0
                            scenario_modified = causal_experiment_engine_twostep.apply_intervention(...
                                scenario_modified, 'widen', params);
                        else
                            scenario_modified = causal_experiment_engine_twostep.apply_intervention(...
                                scenario_modified, 'shrink', params);
                        end
                        
                    case 'center_delta'
                        % Shift intervention (relative shift)
                        params.center_delta = param_value;
                        scenario_modified = causal_experiment_engine_twostep.apply_intervention(...
                            scenario_modified, 'shift', params);
                        
                    case 'correlation_strength'
                        % Correlation intervention
                        params.correlation_strength = param_value;
                        scenario_modified = causal_experiment_engine_twostep.apply_intervention(...
                            scenario_modified, 'correlate', params);
                        
                    otherwise
                        warning('Unknown parameter: %s (skipping)', param_name);
                end
            end
            
            if options.verbose
                fprintf('✓\n');
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
        
        function compute_and_save_mfmc(results_dir, options)
            %COMPUTE_AND_SAVE_MFMC  Step 3: MFMC variance reduction
            %
            % Reads each results_scenario_*.json in results_dir, computes the
            % Multi-Fidelity MC correction per sampling method (alpha per scenario),
            % and writes the corrected estimates alongside the original data.
            %
            % MFMC corrected estimator:
            %   I_MF_m = I_MC_m + alpha_m * (mu_AABB - I_AABB)
            %   alpha_m = Cov(I_MC_m, I_AABB) / Var(I_AABB)   [per scenario]
            %
            % New fields added to each experiment's post_state.inconsistency:
            %   I_MF_sobol, I_MF_halton, I_MF_lhs, I_MF_random  (where data exists)
            %   timing_mfmc_sobol_s, timing_mfmc_halton_s, ...
            %   timing_mfmc_s, timing_mfmc_ms, timing_I_MF_s, timing_I_MF_ms
            %
            % New top-level block added to each scenario file:
            %   mfmc_summary  ->  alpha, rho, rho_sq, variance_reduction, mu_AABB,
            %                     timing_s, timing_ms, timing_per_experiment_s, timing_per_experiment_ms
            %
            % The original I_MC_* values are preserved unchanged.
            %
            % Inputs:
            %   results_dir - Directory containing results_scenario_*.json files
            %   options     - (optional) struct:
            %       .verbose       [true]   Print progress
            %       .scenario_ids  []       Limit to these IDs (empty = all)
            %       .write_inplace [true]   Overwrite source JSON; false writes to
            %                               results_dir/mfmc/ instead

            if nargin < 2, options = struct(); end
            if ~isfield(options, 'verbose'),       options.verbose       = true;  end
            if ~isfield(options, 'scenario_ids'),  options.scenario_ids  = [];    end
            if ~isfield(options, 'write_inplace'), options.write_inplace = true;  end

            SAMPLING_METHODS = {'sobol', 'halton', 'lhs', 'random'};

            json_files = dir(fullfile(results_dir, 'results_scenario_*.json'));
            if isempty(json_files)
                warning('compute_and_save_mfmc: no result files found in %s', results_dir);
                return;
            end

            if options.verbose
                fprintf('Step 3: MFMC correction — %d scenario file(s)...\n', length(json_files));
            end

            for fi = 1:length(json_files)
                fname = json_files(fi).name;
                fpath = fullfile(results_dir, fname);

                % Parse scenario ID from filename
                tok = regexp(fname, 'results_scenario_(\d+)\.json', 'tokens');
                if isempty(tok), continue; end
                scenario_id = str2double(tok{1}{1});

                % Filter
                if ~isempty(options.scenario_ids) && ~ismember(scenario_id, options.scenario_ids)
                    continue;
                end

                if options.verbose
                    fprintf('  scenario_%d ... ', scenario_id);
                end

                % --- Load ---
                fid = fopen(fpath, 'r');
                raw = fread(fid, inf, 'uint8=>char')';
                fclose(fid);
                data = jsondecode(raw);

                if ~isfield(data, 'experiments') || isempty(data.experiments)
                    if options.verbose, fprintf('[SKIP] no experiments\n'); end
                    continue;
                end

                exps    = data.experiments;   % struct array or cell array
                n_exp   = numel(exps);
                is_cell = iscell(exps);

                % --- Extract I_AABB, I_MC, and jaccard_valid per method ---
                I_AABB        = NaN(n_exp, 1);
                jaccard_valid = false(n_exp, 1);  % true only when zonotopes intersect
                I_MC          = NaN(n_exp, length(SAMPLING_METHODS));

                for i = 1:n_exp
                    try
                        if is_cell
                            inc = exps{i}.post_state.inconsistency;
                        else
                            inc = exps(i).post_state.inconsistency;
                        end
                        % Only mark as valid when a true intersection exists and
                        % jaccard_index is a real number.
                        % Non-intersecting experiments are excluded
                        % from MFMC estimation — the control-variate assumption
                        % requires gradation in I_AABB, which is absent when all
                        % non-intersecting cases are collapsed to 0.
                        has_intersection = false;
                        if isfield(inc, 'has_intersection')
                            hi = inc.has_intersection;
                            if islogical(hi)
                                has_intersection = hi;
                            elseif isnumeric(hi)
                                has_intersection = (hi ~= 0);
                            end
                        end
                        if has_intersection && isfield(inc, 'jaccard_index') && ...
                           ~isempty(inc.jaccard_index) && ...
                           isnumeric(inc.jaccard_index) && ...
                           ~isnan(inc.jaccard_index)
                            I_AABB(i)        = inc.jaccard_index;
                            jaccard_valid(i) = true;
                        end
                        for m = 1:length(SAMPLING_METHODS)
                            f = sprintf('mc_probability_%s', SAMPLING_METHODS{m});
                            if isfield(inc, f) && ~isempty(inc.(f))
                                I_MC(i, m) = inc.(f);
                            end
                        end
                    catch
                        % Skip experiments with unexpected structure
                    end
                end

                % --- Compute MFMC alpha per sampling method ---
                % α and μ_AABB are estimated from intersecting experiments only.
                % The correction is then applied exclusively to those experiments;
                % non-intersecting experiments receive NaN (field not written).
                mfmc_summary = struct();
                mfmc_summary.computed_at = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
                I_MF = NaN(n_exp, length(SAMPLING_METHODS));
                primary_method = 'sobol';

                for m = 1:length(SAMPLING_METHODS)
                    method_name   = SAMPLING_METHODS{m};
                    mc           = I_MC(:, m);
                    % Restrict estimation to intersecting experiments
                    intersecting = jaccard_valid & ~isnan(mc);
                    if sum(intersecting) < 3, continue; end
                    min_abs_rho = 0.05;

                    t_mfmc = tic;
                    aabb_v  = I_AABB(intersecting);
                    mc_v    = mc(intersecting);
                    mu_aabb = mean(aabb_v);

                    C           = cov([mc_v, aabb_v]);   % 2x2 sample covariance
                    var_aabb    = C(2, 2);
                    cov_mc_aabb = C(1, 2);

                    if var_aabb < 1e-12, continue; end   % AABB has no variation

                    alpha = cov_mc_aabb / var_aabb;
                    rho   = cov_mc_aabb / (std(mc_v) * std(aabb_v) + 1e-12);
                    alpha_raw = alpha;
                    correction_enabled = abs(rho) >= min_abs_rho;
                    if ~correction_enabled
                        alpha = 0.0;
                    end

                    % Apply correction only to intersecting experiments
                    I_MF_m               = NaN(n_exp, 1);
                    I_MF_m(intersecting) = mc(intersecting) + alpha .* (mu_aabb - I_AABB(intersecting));
                    I_MF_m(intersecting) = max(0.0, min(1.0, I_MF_m(intersecting)));
                    I_MF(:, m)           = I_MF_m;

                    var_mc             = var(mc_v);
                    finite_mf          = I_MF_m(~isnan(I_MF_m));
                    var_mf             = var(finite_mf);
                    variance_reduction = 1.0 - var_mf / (var_mc + 1e-12);
                    elapsed_s          = toc(t_mfmc);
                    elapsed_ms         = 1000.0 * elapsed_s;
                    per_exp_s          = elapsed_s / n_exp;
                    per_exp_ms         = 1000.0 * per_exp_s;

                    s                    = struct();
                    s.alpha              = alpha;
                    s.alpha_raw          = alpha_raw;
                    s.rho                = rho;
                    s.rho_sq             = rho^2;
                    s.variance_reduction = variance_reduction;
                    s.correction_enabled = correction_enabled;
                    s.min_abs_rho_for_correction = min_abs_rho;
                    s.mu_AABB            = mu_aabb;
                    s.n_intersecting     = sum(intersecting);
                    s.n_experiments      = n_exp;
                    s.timing_s           = elapsed_s;
                    s.timing_ms          = elapsed_ms;
                    s.timing_per_experiment_s  = per_exp_s;
                    s.timing_per_experiment_ms = per_exp_ms;
                    mfmc_summary.(method_name) = s;
                end

                % --- Augment each experiment with I_MF_* fields ---
                for i = 1:n_exp
                    for m = 1:length(SAMPLING_METHODS)
                        method_name = SAMPLING_METHODS{m};
                        if isnan(I_MF(i, m)), continue; end
                        field = sprintf('I_MF_%s', method_name);
                        try
                            if is_cell
                                data.experiments{i}.post_state.inconsistency.(field) = I_MF(i, m);
                                if isfield(mfmc_summary, method_name)
                                    per_exp_s  = mfmc_summary.(method_name).timing_per_experiment_s;
                                    per_exp_ms = mfmc_summary.(method_name).timing_per_experiment_ms;
                                    data.experiments{i}.post_state.inconsistency.(sprintf('timing_mfmc_%s_s', method_name))  = per_exp_s;
                                    data.experiments{i}.post_state.inconsistency.(sprintf('timing_mfmc_%s_ms', method_name)) = per_exp_ms;
                                    if strcmp(method_name, primary_method)
                                        data.experiments{i}.post_state.inconsistency.timing_mfmc_s  = per_exp_s;
                                        data.experiments{i}.post_state.inconsistency.timing_mfmc_ms = per_exp_ms;
                                        data.experiments{i}.post_state.inconsistency.timing_I_MF_s  = per_exp_s;
                                        data.experiments{i}.post_state.inconsistency.timing_I_MF_ms = per_exp_ms;
                                    end
                                end
                            else
                                data.experiments(i).post_state.inconsistency.(field) = I_MF(i, m);
                                if isfield(mfmc_summary, method_name)
                                    per_exp_s  = mfmc_summary.(method_name).timing_per_experiment_s;
                                    per_exp_ms = mfmc_summary.(method_name).timing_per_experiment_ms;
                                    data.experiments(i).post_state.inconsistency.(sprintf('timing_mfmc_%s_s', method_name))  = per_exp_s;
                                    data.experiments(i).post_state.inconsistency.(sprintf('timing_mfmc_%s_ms', method_name)) = per_exp_ms;
                                    if strcmp(method_name, primary_method)
                                        data.experiments(i).post_state.inconsistency.timing_mfmc_s  = per_exp_s;
                                        data.experiments(i).post_state.inconsistency.timing_mfmc_ms = per_exp_ms;
                                        data.experiments(i).post_state.inconsistency.timing_I_MF_s  = per_exp_s;
                                        data.experiments(i).post_state.inconsistency.timing_I_MF_ms = per_exp_ms;
                                    end
                                end
                            end
                        catch
                            % Skip if field cannot be set
                        end
                    end
                end
                data.mfmc_summary = mfmc_summary;

                % --- Write ---
                if options.write_inplace
                    out_path = fpath;
                else
                    mfmc_dir = fullfile(results_dir, 'mfmc');
                    if ~exist(mfmc_dir, 'dir'), mkdir(mfmc_dir); end
                    out_path = fullfile(mfmc_dir, fname);
                end

                json_text = jsonencode(data, 'PrettyPrint', true);
                wid = fopen(out_path, 'w');
                if wid == -1
                    error('compute_and_save_mfmc: cannot write to %s', out_path);
                end
                fprintf(wid, '%s', json_text);
                fclose(wid);

                % Progress line
                if options.verbose
                    methods_done = fieldnames(mfmc_summary);
                    methods_done = methods_done(~strcmp(methods_done, 'computed_at'));
                    if ~isempty(methods_done)
                        parts = cellfun(@(m) sprintf('%s:rho=%.3f', m, mfmc_summary.(m).rho), ...
                            methods_done, 'UniformOutput', false);
                        fprintf('[%s]\n', strjoin(parts, '  '));
                    else
                        fprintf('[no MC data]\n');
                    end
                end
            end

            if options.verbose
                fprintf('  Done.\n');
            end
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
