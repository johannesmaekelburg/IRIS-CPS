function [I_theta, details] = global_inconsistency(models, varargin)
%GLOBAL_INCONSISTENCY Compute global inconsistency measure I(θ) across all models
%   Computes a single scalar inconsistency measure I(θ) ∈ [0,1] that
%   quantifies the degree of inconsistency across ALL model zonotopes
%   under a given parameter configuration θ (intervention state).
%
% DEFINITION (from PDF)
%   I(θ) = 1 - P_ξ~μ[C(ξ; θ) = 1]
%   
%   where C(ξ; θ) indicates if a realization x(ξ) = c + Gξ is consistent:
%   C(ξ; θ) = 1[x(ξ, θ) ∈ ⋂_i Z_Mi(θ)]
%   
%   In words: I(θ) is the probability that a random realization from the
%   uncertainty distribution FAILS to satisfy all model constraints simultaneously.
%
% INTERPRETATION
%   I(θ) = 0.0  →  Fully consistent (all realizations feasible)
%   I(θ) = 0.5  →  Half of realizations are inconsistent
%   I(θ) = 1.0  →  Fully inconsistent (no feasible realizations)
%
% SYNTAX
%   I_theta = global_inconsistency(models)
%   I_theta = global_inconsistency(scenario)
%   [I_theta, details] = global_inconsistency(..., 'n_samples', N)
%
% INPUTS
%   models - One of:
%       (1) Cell array {Z1, Z2, ..., Zn} of zonotope/conPolyZono objects
%           representing n different models under current θ
%       (2) Struct with field .models containing cell array
%       (3) Struct representing scenario with .source, .target, .mapping
%           (will propagate and extract models automatically)
%
% OPTIONAL NAME-VALUE PARAMETERS
%   'n_samples'      - Number of Monte Carlo samples (default: 2000)
%   'seed'           - Random seed for reproducibility (default: [])
%   'verbose'        - Display progress messages (default: false)
%   'method'         - Computation method: 'mc' (default), 'jaccard', 'exact'
%   'return_details' - Return detailed statistics (default: true)
%
% OUTPUTS
%   I_theta - Global inconsistency scalar ∈ [0, 1]
%             I(θ) = 1 - P(consistent across all models)
%   
%   details - Struct with fields:
%       .p_consistent        - P(realization ∈ ⋂ models) = 1 - I_theta
%       .p_inconsistent      - = I_theta
%       .num_models          - Number of models considered
%       .num_samples         - Number of MC samples used
%       .num_consistent      - Number of consistent samples
%       .standard_error      - Binomial standard error
%       .ci95_lower          - 95% CI lower bound for I_theta
%       .ci95_upper          - 95% CI upper bound for I_theta
%       .intersection_empty  - Boolean, true if ⋂ models is empty
%       .method              - Computation method used
%       .computation_time    - Elapsed time (seconds)
%       .theta_params        - Parameter configuration (if provided)
%
% ALGORITHM
%   1. Extract model zonotopes from input
%   2. Call score_mc_probability({Z1, ..., Zn}) to estimate P(consistent)
%   3. Compute I(θ) = 1 - P(consistent)
%   4. Return detailed statistics
%
% EXAMPLES
%   % Example 1: Three models after interventions
%   Z1 = conPolyZono([10; 20], eye(2), eye(2), [], [], []);
%   Z2 = conPolyZono([11; 21], 0.8*eye(2), eye(2), [], [], []);
%   Z3 = conPolyZono([9.5; 19.5], 1.2*eye(2), eye(2), [], [], []);
%   
%   [I_theta, details] = global_inconsistency({Z1, Z2, Z3}, ...
%       'n_samples', 5000, ...
%       'verbose', true);
%   
%   fprintf('Global Inconsistency I(θ) = %.3f ± %.3f\n', ...
%       I_theta, details.standard_error);
%   fprintf('P(consistent) = %.3f\n', details.p_consistent);
%   
%   % Example 2: Scenario struct
%   scenario.models = {Z1, Z2, Z3};
%   I_theta = global_inconsistency(scenario);
%
%   % Example 3: With theta parameter tracking
%   theta_state = struct('scale_source', 1.5, 'shift_target', [0.1; 0.2]);
%   [I_theta, details] = global_inconsistency({Z1, Z2, Z3}, ...
%       'theta_params', theta_state);
%
% USE IN SENSITIVITY ANALYSIS
%   This function is the foundation for computing:
%   - Total causal effects: τ_j(a,b) = E[I(θ)|do(θ_j=b)] - E[I(θ)|do(θ_j=a)]
%   - Local sensitivity: ∂I/∂θ_j ≈ (I(θ+ε) - I(θ))/ε
%   - Robustness margins: s*(τ) = sup{s : E[I(θ(s))] ≤ τ}
%   - Sobol indices: First-order and total-effect variance decomposition
%
% SEE ALSO
%   score_mc_probability, causal_experiment_engine, sensitivity_analysis

% Author: CPS-Uncertainty-Propagation-Framework / Causality Extension
% Date: 2026-01-28

%% Parse inputs
p = inputParser;
addRequired(p, 'models');
addParameter(p, 'n_samples', 2000, @(x) isnumeric(x) && x > 0);
addParameter(p, 'seed', [], @(x) isempty(x) || isnumeric(x));
addParameter(p, 'verbose', false, @islogical);
addParameter(p, 'method', 'mc', @(x) ismember(x, {'mc', 'jaccard', 'exact'}));
addParameter(p, 'return_details', true, @islogical);
addParameter(p, 'theta_params', [], @(x) isempty(x) || isstruct(x));
parse(p, models, varargin{:});

n_samples = p.Results.n_samples;
seed = p.Results.seed;
verbose = p.Results.verbose;
method = p.Results.method;
return_details = p.Results.return_details;
theta_params = p.Results.theta_params;

%% Extract model zonotopes
tic;

if verbose
    fprintf('\n=== Computing Global Inconsistency I(θ) ===\n');
end

% Handle different input formats
if iscell(models)
    % Already a cell array of zonotopes
    model_list = models;
elseif isstruct(models) && isfield(models, 'models')
    % Struct with .models field
    model_list = models.models;
elseif isstruct(models) && isfield(models, 'source') && isfield(models, 'mapping')
    % Scenario struct - extract propagated models
    % (For now, assume pre-propagated; extend later if needed)
    error('global_inconsistency:NotImplemented', ...
        'Automatic scenario propagation not yet implemented. Please provide cell array of model zonotopes.');
else
    error('global_inconsistency:InvalidInput', ...
        'Input must be cell array of zonotopes or struct with .models field.');
end

% Validate
if isempty(model_list)
    error('global_inconsistency:EmptyModels', 'No models provided.');
end

num_models = length(model_list);

if verbose
    fprintf('  Number of models: %d\n', num_models);
end

% Special case: single model → no inconsistency possible
if num_models == 1
    if verbose
        fprintf('  ⚠ Single model provided - no inconsistency possible\n');
        fprintf('  Returning I(θ) = 0 (fully consistent)\n');
    end
    I_theta = 0.0;
    if nargout > 1 || return_details
        details = struct();
        details.p_consistent = 1.0;
        details.p_inconsistent = 0.0;
        details.num_models = 1;
        details.num_samples = 0;
        details.num_consistent = n_samples;
        details.standard_error = 0.0;
        details.ci95_lower = 0.0;
        details.ci95_upper = 0.0;
        details.intersection_empty = false;
        details.method = 'trivial';
        details.computation_time = toc;
        if ~isempty(theta_params)
            details.theta_params = theta_params;
        end
    end
    return;
end

%% Compute P(consistent) using score_mc_probability
% This function computes P(realization ∈ ⋂ all models)

if verbose
    fprintf('  Computing P(consistent) via Monte Carlo...\n');
end

% Call parent's score_mc_probability function
[p_consistent, mc_details] = score_mc_probability(model_list, ...
    'num_samples', n_samples, ...
    'seed', seed, ...
    'verbose', verbose, ...
    'return_details', true);

%% Compute I(θ) = 1 - P(consistent)
I_theta = 1.0 - p_consistent;

if verbose
    fprintf('\n  ✓ P(consistent) = %.4f\n', p_consistent);
    fprintf('  ✓ I(θ) = %.4f\n', I_theta);
    if isfield(mc_details, 'standard_error')
        fprintf('  ✓ SE = %.4f\n', mc_details.standard_error);
    end
end

%% Construct detailed output
if nargout > 1 || return_details
    details = struct();
    
    % Core metrics
    details.p_consistent = p_consistent;
    details.p_inconsistent = I_theta;
    details.num_models = num_models;
    
    % Copy MC sampling details
    details.num_samples = mc_details.num_samples;
    details.num_consistent = mc_details.num_consistent;
    details.standard_error = mc_details.standard_error;
    
    % Confidence intervals for I(θ)
    % Note: CI bounds are flipped since I = 1 - P
    details.ci95_lower = 1.0 - mc_details.ci95_upper;
    details.ci95_upper = 1.0 - mc_details.ci95_lower;
    
    % Additional info
    details.intersection_empty = mc_details.intersection_empty;
    details.method = method;
    details.computation_time = toc;
    
    % Store theta parameters if provided
    if ~isempty(theta_params)
        details.theta_params = theta_params;
    end
    
    if verbose
        fprintf('\n  95%% CI for I(θ): [%.4f, %.4f]\n', ...
            details.ci95_lower, details.ci95_upper);
        fprintf('  Computation time: %.3f seconds\n', details.computation_time);
        fprintf('===========================================\n\n');
    end
end

end
