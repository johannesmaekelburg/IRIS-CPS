"""
Sensitivity Analysis for Global Inconsistency I(θ)

This module implements variance-based sensitivity analysis and causal effect
quantification for the global inconsistency measure I(θ) as described in the
Problem Formulation and Approach PDF.

IMPLEMENTED METRICS:
1. Total Causal Effects: τ_j(a,b) = E[I(θ)|do(θ_j=b)] - E[I(θ)|do(θ_j=a)]
2. Local Sensitivity: ∂I/∂θ_j ≈ (E[I(θ + εe)] - E[I(θ)])/ε
3. Robustness Margins: s*(τ) = sup{s : E[I(θ(s))] ≤ τ}
4. First-order Sobol Indices: S_i = Var_θi(E[I(θ)|θi])/Var(I(θ))
5. Total-effect Sobol Indices: S_i^T = 1 - Var_θ~i(E[I(θ)|θ~i])/Var(I(θ))

WORKFLOW:
1. Load experimental data from MATLAB JSON exports (with I_theta values)
2. Extract parameter configurations θ and corresponding I(θ) values
3. Compute sensitivity metrics using variance decomposition
4. Fit surrogate models (GP/RF) for expensive computations
5. Generate visualizations and export results

Author: CPS Uncertainty-Propagation Framework / Causality Extension
Date: 2026-01-28
"""

import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.interpolate import interp1d

# Machine learning for surrogate models
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Sobol indices computation
try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    SALIB_AVAILABLE = True
except ImportError:
    SALIB_AVAILABLE = False
    warnings.warn("SALib not installed. Sobol indices will not be available. "
                  "Install with: pip install SALib")


# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

def load_experimental_data(data_dir: str, pattern: str = 'results_*.json') -> pd.DataFrame:
    """
    Load experimental data from MATLAB JSON exports.
    
    Args:
        data_dir: Directory containing JSON files
        pattern: Filename pattern for JSON files
        
    Returns:
        DataFrame with columns: [theta_params, I_theta, pre/post state metrics]
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    json_files = sorted(data_path.glob(pattern))
    
    if len(json_files) == 0:
        raise ValueError(f"No files matching '{pattern}' found in {data_path}")
    
    print(f"Loading experimental data from {data_path}...")
    print(f"Found {len(json_files)} JSON files\n")
    
    all_records = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            warnings.warn(f"Failed to load {json_file.name}: {e}")
            continue
        
        # Handle MATLAB export format with 'experiments' array
        if isinstance(data, dict) and 'experiments' in data:
            records = data['experiments']
        elif isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            records = [data]
        else:
            warnings.warn(f"Unexpected data format in {json_file.name}")
            continue
        
        for record in records:
            # MATLAB export uses flattened structure
            # Check for I_theta in post_inconsistency (flattened) or nested post_state
            I_theta = None
            theta_params = {}
            
            # Option 1: Flattened MATLAB export (post_inconsistency.I_theta)
            if 'post_inconsistency' in record and isinstance(record['post_inconsistency'], dict):
                post_inc = record['post_inconsistency']
                if 'I_theta' in post_inc:
                    I_theta = post_inc['I_theta']
                    I_theta_se = post_inc.get('I_theta_se', np.nan)
                    I_theta_ci95_lower = post_inc.get('I_theta_ci95_lower', np.nan)
                    I_theta_ci95_upper = post_inc.get('I_theta_ci95_upper', np.nan)
                    
                    # Extract theta from record metadata
                    if 'param_value' in record:
                        param_name = record.get('intervention', 'parameter')
                        theta_params[param_name] = record['param_value']
            
            # Option 2: Nested structure (post_state.inconsistency.I_theta)
            elif 'post_state' in record:
                post_state = record['post_state']
                if 'inconsistency' not in post_state:
                    continue
                
                inconsistency = post_state['inconsistency']
                if 'I_theta' not in inconsistency:
                    continue
                
                I_theta = inconsistency['I_theta']
                I_theta_se = inconsistency.get('I_theta_se', np.nan)
                I_theta_ci95_lower = inconsistency.get('I_theta_ci95_lower', np.nan)
                I_theta_ci95_upper = inconsistency.get('I_theta_ci95_upper', np.nan)
                
                # Extract theta parameters
                if 'theta_params' in inconsistency:
                    theta_params = inconsistency['theta_params']
                elif 'intervention' in record:
                    intervention = record['intervention']
                    if 'params' in intervention:
                        theta_params = intervention['params']
            
            if I_theta is None:
                continue
            
            # Build flattened record
            flat_record = {
                'I_theta': I_theta,
                'I_theta_se': I_theta_se,
                'I_theta_ci95_lower': I_theta_ci95_lower,
                'I_theta_ci95_upper': I_theta_ci95_upper,
                'intervention_type': record.get('intervention', 'unknown'),
                **theta_params
            }
            
            # Add optional metrics
            if 'pre_inconsistency' in record and isinstance(record['pre_inconsistency'], dict):
                flat_record['I_theta_pre'] = record['pre_inconsistency'].get('I_theta', np.nan)
            elif 'pre_state' in record:
                pre_inc = record['pre_state'].get('inconsistency', {})
                flat_record['I_theta_pre'] = pre_inc.get('I_theta', np.nan)
            
            if 'causal_effect' in record:
                if isinstance(record['causal_effect'], dict):
                    flat_record['delta_I_theta'] = record['causal_effect'].get('delta_I_theta', np.nan)
                else:
                    delta = record['causal_effect'].get('inconsistency', {})
                    flat_record['delta_I_theta'] = delta.get('delta_I_theta', np.nan)
            
            all_records.append(flat_record)
    
    df = pd.DataFrame(all_records)
    
    print(f"Loaded {len(df)} records with I(θ) measurements")
    print(f"Parameter columns: {[col for col in df.columns if col not in ['I_theta', 'I_theta_se', 'intervention_type']]}")
    print(f"I(θ) range: [{df['I_theta'].min():.3f}, {df['I_theta'].max():.3f}]")
    print(f"I(θ) mean: {df['I_theta'].mean():.3f} ± {df['I_theta'].std():.3f}\n")
    
    return df


def extract_theta_and_I(df: pd.DataFrame, 
                        theta_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract parameter matrix θ and inconsistency vector I(θ).
    
    Args:
        df: DataFrame from load_experimental_data
        theta_columns: List of column names for θ parameters (auto-detect if None)
        
    Returns:
        (theta, I_theta) where theta is (n_samples, n_params) and I_theta is (n_samples,)
    """
    # Auto-detect theta columns if not provided
    if theta_columns is None:
        exclude_cols = ['I_theta', 'I_theta_se', 'I_theta_ci95_lower', 'I_theta_ci95_upper',
                       'intervention_type', 'I_theta_pre', 'delta_I_theta']
        theta_columns = [col for col in df.columns if col not in exclude_cols]
    
    # Extract theta matrix
    theta = df[theta_columns].values
    I_theta = df['I_theta'].values
    
    # Remove rows with NaN
    valid_mask = ~(np.isnan(theta).any(axis=1) | np.isnan(I_theta))
    theta = theta[valid_mask]
    I_theta = I_theta[valid_mask]
    
    print(f"Extracted θ matrix: {theta.shape}")
    print(f"Parameter ranges:")
    for i, col in enumerate(theta_columns):
        print(f"  {col}: [{theta[:, i].min():.3f}, {theta[:, i].max():.3f}]")
    
    return theta, I_theta, theta_columns


# ============================================================================
# 1. Total Causal Effects τ_j(a,b)
# ============================================================================

def compute_causal_effects(df: pd.DataFrame, 
                          param_name: str,
                          values: Optional[List[float]] = None) -> pd.DataFrame:
    """
    Compute total causal effects τ_j(a,b) = E[I(θ)|do(θ_j=b)] - E[I(θ)|do(θ_j=a)]
    
    Args:
        df: Experimental data
        param_name: Parameter to analyze (e.g., 'scale_factor')
        values: Specific values to compare (if None, compare all unique values)
        
    Returns:
        DataFrame with causal effects for all pairs (a, b)
    """
    if param_name not in df.columns:
        raise ValueError(f"Parameter '{param_name}' not found in data")
    
    # Get unique parameter values
    if values is None:
        values = sorted(df[param_name].dropna().unique())
    
    results = []
    
    # Compute pairwise effects
    for i, a in enumerate(values):
        for b in values[i+1:]:
            # Filter data for each intervention level
            data_a = df[df[param_name] == a]
            data_b = df[df[param_name] == b]
            
            if len(data_a) == 0 or len(data_b) == 0:
                continue
            
            # Compute expectations
            E_I_a = data_a['I_theta'].mean()
            E_I_b = data_b['I_theta'].mean()
            
            # Causal effect
            tau = E_I_b - E_I_a
            
            # Standard errors (assuming independence)
            se_a = data_a['I_theta'].sem()
            se_b = data_b['I_theta'].sem()
            tau_se = np.sqrt(se_a**2 + se_b**2)
            
            results.append({
                'parameter': param_name,
                'value_a': a,
                'value_b': b,
                'E_I_theta_a': E_I_a,
                'E_I_theta_b': E_I_b,
                'tau': tau,
                'tau_se': tau_se,
                'tau_abs': abs(tau),
                'n_samples_a': len(data_a),
                'n_samples_b': len(data_b)
            })
    
    results_df = pd.DataFrame(results)
    
    print(f"\n=== Causal Effects for {param_name} ===")
    print(results_df.to_string(index=False))
    print()
    
    return results_df


# ============================================================================
# 2. Local Sensitivity ∂I/∂θ_j
# ============================================================================

def compute_local_sensitivity(df: pd.DataFrame,
                              param_name: str,
                              method: str = 'finite_difference') -> Dict[str, Any]:
    """
    Compute local sensitivity ∂I/∂θ_j using finite differences.
    
    Args:
        df: Experimental data
        param_name: Parameter to analyze
        method: 'finite_difference' or 'regression'
        
    Returns:
        Dict with sensitivity estimates at different θ values
    """
    if param_name not in df.columns:
        raise ValueError(f"Parameter '{param_name}' not found in data")
    
    # Sort by parameter value
    df_sorted = df[[param_name, 'I_theta']].dropna().sort_values(param_name)
    theta_vals = df_sorted[param_name].values
    I_vals = df_sorted['I_theta'].values
    
    if len(theta_vals) < 2:
        raise ValueError(f"Need at least 2 data points for sensitivity analysis")
    
    sensitivities = []
    
    if method == 'finite_difference':
        # Central differences where possible, forward/backward at boundaries
        for i in range(len(theta_vals)):
            if i == 0:
                # Forward difference
                dI_dtheta = (I_vals[i+1] - I_vals[i]) / (theta_vals[i+1] - theta_vals[i])
            elif i == len(theta_vals) - 1:
                # Backward difference
                dI_dtheta = (I_vals[i] - I_vals[i-1]) / (theta_vals[i] - theta_vals[i-1])
            else:
                # Central difference
                dI_dtheta = (I_vals[i+1] - I_vals[i-1]) / (theta_vals[i+1] - theta_vals[i-1])
            
            sensitivities.append({
                'theta': theta_vals[i],
                'I_theta': I_vals[i],
                'dI_dtheta': dI_dtheta
            })
    
    elif method == 'regression':
        # Fit polynomial and compute derivative
        from numpy.polynomial import Polynomial
        p = Polynomial.fit(theta_vals, I_vals, deg=2)
        p_deriv = p.deriv()
        
        for theta, I in zip(theta_vals, I_vals):
            sensitivities.append({
                'theta': theta,
                'I_theta': I,
                'dI_dtheta': p_deriv(theta)
            })
    
    sens_df = pd.DataFrame(sensitivities)
    
    print(f"\n=== Local Sensitivity ∂I/∂{param_name} ===")
    print(f"Method: {method}")
    print(f"Range: [{sens_df['dI_dtheta'].min():.4f}, {sens_df['dI_dtheta'].max():.4f}]")
    print(f"Mean: {sens_df['dI_dtheta'].mean():.4f}")
    print()
    
    return {
        'parameter': param_name,
        'method': method,
        'sensitivities': sens_df,
        'mean_sensitivity': sens_df['dI_dtheta'].mean(),
        'max_sensitivity': sens_df['dI_dtheta'].abs().max()
    }


# ============================================================================
# 3. Robustness Margins s*(τ)
# ============================================================================

def compute_robustness_margins(df: pd.DataFrame,
                               param_name: str,
                               threshold: float = 0.5) -> Dict[str, Any]:
    """
    Compute robustness margin: s*(τ) = sup{s : E[I(θ(s))] ≤ τ}
    
    How much can we scale parameter before exceeding inconsistency threshold?
    
    Args:
        df: Experimental data
        param_name: Parameter to analyze (should be scaling factor)
        threshold: Inconsistency threshold τ
        
    Returns:
        Dict with critical values and safety margins
    """
    if param_name not in df.columns:
        raise ValueError(f"Parameter '{param_name}' not found in data")
    
    # Sort by parameter
    df_sorted = df[[param_name, 'I_theta']].dropna().sort_values(param_name)
    
    # Interpolate I(θ) as function of parameter
    # Use linear for <=3 points, quadratic for 4-5 points, cubic for >5
    n_points = len(df_sorted)
    if n_points < 2:
        raise ValueError("Need at least 2 data points for robustness analysis")
    elif n_points <= 3:
        interp_kind = 'linear'
    elif n_points <= 5:
        interp_kind = 'quadratic'
    else:
        interp_kind = 'cubic'
    
    interp_func = interp1d(df_sorted[param_name], df_sorted['I_theta'], 
                           kind=interp_kind, fill_value='extrapolate')
    
    # Find critical value where I(θ) = threshold
    param_range = np.linspace(df_sorted[param_name].min(), 
                             df_sorted[param_name].max(), 1000)
    I_range = interp_func(param_range)
    
    # Find where curve crosses threshold
    crossings = []
    for i in range(len(I_range) - 1):
        if (I_range[i] <= threshold and I_range[i+1] > threshold) or \
           (I_range[i] > threshold and I_range[i+1] <= threshold):
            # Linear interpolation for crossing point
            alpha = (threshold - I_range[i]) / (I_range[i+1] - I_range[i])
            s_critical = param_range[i] + alpha * (param_range[i+1] - param_range[i])
            crossings.append(s_critical)
    
    # Robustness margin (assuming baseline s=1.0)
    if len(crossings) > 0:
        s_star = crossings[0]  # First crossing
        margin = abs(s_star - 1.0)
    else:
        if I_range[0] > threshold:
            s_star = None
            margin = 0.0
        else:
            s_star = param_range[-1]
            margin = abs(s_star - 1.0)
    
    print(f"\n=== Robustness Margin for {param_name} ===")
    print(f"Threshold τ = {threshold}")
    print(f"Critical value s* = {s_star}")
    print(f"Safety margin = {margin:.3f}")
    print()
    
    return {
        'parameter': param_name,
        'threshold': threshold,
        's_star': s_star,
        'margin': margin,
        'crossings': crossings,
        'param_range': param_range,
        'I_range': I_range
    }


# ============================================================================
# 4. Sobol Indices (Variance-based Sensitivity)
# ============================================================================

def compute_sobol_indices(theta: np.ndarray,
                         I_theta: np.ndarray,
                         param_names: List[str],
                         n_samples: int = 1024,
                         calc_second_order: bool = False) -> Dict[str, Any]:
    """
    Compute Sobol sensitivity indices using SALib.
    
    First-order: S_i = Var_θi(E[I(θ)|θi])/Var(I(θ))
    Total-effect: S_i^T = 1 - Var_θ~i(E[I(θ)|θ~i])/Var(I(θ))
    
    Args:
        theta: Parameter matrix (n_samples, n_params)
        I_theta: Inconsistency values (n_samples,)
        param_names: Names of parameters
        n_samples: Number of samples for Saltelli sampling (if surrogate needed)
        calc_second_order: Compute second-order interactions
        
    Returns:
        Dict with first-order and total-effect Sobol indices
    """
    if not SALIB_AVAILABLE:
        raise ImportError("SALib is required for Sobol indices. Install with: pip install SALib")
    
    n_params = theta.shape[1]
    
    # Define problem for SALib
    problem = {
        'num_vars': n_params,
        'names': param_names,
        'bounds': [[theta[:, i].min(), theta[:, i].max()] for i in range(n_params)]
    }
    
    # Check if we have enough samples for direct analysis
    # SALib needs structured samples, so we'll fit a surrogate and use it
    print("\n=== Computing Sobol Indices ===")
    print(f"Fitting surrogate model for Sobol analysis...")
    
    # Fit surrogate (GP or RF)
    surrogate = fit_surrogate_model(theta, I_theta, model_type='gp')
    
    # Generate Saltelli samples
    param_values = saltelli.sample(problem, n_samples, calc_second_order=calc_second_order)
    
    # Evaluate surrogate on samples
    Y = surrogate.predict(param_values)
    
    # Perform Sobol analysis
    Si = sobol.analyze(problem, Y, calc_second_order=calc_second_order)
    
    # Extract results
    results = {
        'first_order': dict(zip(param_names, Si['S1'])),
        'first_order_conf': dict(zip(param_names, Si['S1_conf'])),
        'total_order': dict(zip(param_names, Si['ST'])),
        'total_order_conf': dict(zip(param_names, Si['ST_conf'])),
    }
    
    if calc_second_order:
        results['second_order'] = Si['S2']
        results['second_order_conf'] = Si['S2_conf']
    
    # Print summary
    print(f"\nFirst-order Sobol indices (main effects):")
    for name in param_names:
        S1 = results['first_order'][name]
        S1_conf = results['first_order_conf'][name]
        print(f"  {name:20s}: {S1:.4f} ± {S1_conf:.4f}")
    
    print(f"\nTotal-effect Sobol indices (with interactions):")
    for name in param_names:
        ST = results['total_order'][name]
        ST_conf = results['total_order_conf'][name]
        print(f"  {name:20s}: {ST:.4f} ± {ST_conf:.4f}")
    
    return results


# ============================================================================
# 5. Surrogate Model Fitting
# ============================================================================

def fit_surrogate_model(theta: np.ndarray,
                       I_theta: np.ndarray,
                       model_type: str = 'gp',
                       test_size: float = 0.2,
                       random_state: int = 42) -> Any:
    """
    Fit surrogate model I_hat(θ) for expensive I(θ) evaluations.
    
    Args:
        theta: Parameter matrix
        I_theta: Inconsistency values
        model_type: 'gp' (Gaussian Process) or 'rf' (Random Forest)
        test_size: Fraction for test set
        random_state: Random seed
        
    Returns:
        Trained surrogate model
    """
    # Train-test split
    theta_train, theta_test, I_train, I_test = train_test_split(
        theta, I_theta, test_size=test_size, random_state=random_state
    )
    
    print(f"\n=== Fitting Surrogate Model ({model_type.upper()}) ===")
    print(f"Training samples: {len(theta_train)}, Test samples: {len(theta_test)}")
    
    if model_type == 'gp':
        # Gaussian Process with RBF kernel
        kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0]*theta.shape[1], (1e-2, 1e2)) + \
                 WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 1e-1))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, 
                                        random_state=random_state, normalize_y=True)
    elif model_type == 'rf':
        # Random Forest
        model = RandomForestRegressor(n_estimators=100, max_depth=10, 
                                     random_state=random_state, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Fit model
    model.fit(theta_train, I_train)
    
    # Evaluate
    I_pred_train = model.predict(theta_train)
    I_pred_test = model.predict(theta_test)
    
    r2_train = r2_score(I_train, I_pred_train)
    r2_test = r2_score(I_test, I_pred_test)
    rmse_train = np.sqrt(mean_squared_error(I_train, I_pred_train))
    rmse_test = np.sqrt(mean_squared_error(I_test, I_pred_test))
    mae_test = mean_absolute_error(I_test, I_pred_test)
    
    print(f"\nPerformance:")
    print(f"  Train R²: {r2_train:.4f}")
    print(f"  Test R²:  {r2_test:.4f}")
    print(f"  Train RMSE: {rmse_train:.4f}")
    print(f"  Test RMSE:  {rmse_test:.4f}")
    print(f"  Test MAE:   {mae_test:.4f}")
    
    # Cross-validation (adaptive folds based on sample size)
    n_samples = len(theta)
    cv_folds = min(5, n_samples)  # Use min(5, n_samples) folds
    
    if cv_folds >= 2:
        cv_scores = cross_val_score(model, theta, I_theta, cv=cv_folds, 
                                    scoring='r2', n_jobs=-1)
        print(f"  {cv_folds}-fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    else:
        print(f"  Note: Too few samples ({n_samples}) for cross-validation")
    
    return model


# ============================================================================
# Visualization
# ============================================================================

def plot_causal_effects(results_df: pd.DataFrame, 
                       output_path: Optional[str] = None):
    """Plot total causal effects τ_j(a,b)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create labels for x-axis
    labels = [f"{row['value_a']:.1f}→{row['value_b']:.1f}" 
              for _, row in results_df.iterrows()]
    
    x = np.arange(len(labels))
    tau = results_df['tau'].values
    tau_se = results_df['tau_se'].values
    
    # Bar plot with error bars
    ax.bar(x, tau, yerr=tau_se, capsize=5, alpha=0.7, color='steelblue')
    ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel('Intervention: a → b')
    ax.set_ylabel('Causal Effect τ(a,b)')
    ax.set_title(f'Total Causal Effects on I(θ)\n{results_df["parameter"].iloc[0]}')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.show()


def plot_local_sensitivity(sens_dict: Dict[str, Any],
                           output_path: Optional[str] = None):
    """Plot local sensitivity ∂I/∂θ"""
    sens_df = sens_dict['sensitivities']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # I(θ) vs θ
    ax1.plot(sens_df['theta'], sens_df['I_theta'], 'o-', color='steelblue', label='I(θ)')
    ax1.set_xlabel(f'{sens_dict["parameter"]}')
    ax1.set_ylabel('I(θ)')
    ax1.set_title('Global Inconsistency I(θ)')
    ax1.grid(alpha=0.3)
    ax1.legend()
    
    # ∂I/∂θ vs θ
    ax2.plot(sens_df['theta'], sens_df['dI_dtheta'], 's-', color='coral', label='∂I/∂θ')
    ax2.axhline(0, color='k', linestyle='--', linewidth=0.8)
    ax2.set_xlabel(f'{sens_dict["parameter"]}')
    ax2.set_ylabel('∂I/∂θ')
    ax2.set_title(f'Local Sensitivity (method: {sens_dict["method"]})')
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.show()


def plot_robustness_margins(margins_dict: Dict[str, Any],
                            output_path: Optional[str] = None):
    """Plot robustness margins s*(τ)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(margins_dict['param_range'], margins_dict['I_range'], 
            '-', color='steelblue', linewidth=2, label='I(θ)')
    ax.axhline(margins_dict['threshold'], color='red', linestyle='--', 
              linewidth=2, label=f'Threshold τ={margins_dict["threshold"]:.2f}')
    
    if margins_dict['s_star'] is not None:
        ax.axvline(margins_dict['s_star'], color='green', linestyle='--',
                  linewidth=2, label=f's*={margins_dict["s_star"]:.3f}')
        ax.fill_betweenx([0, 1], 0, margins_dict['s_star'], 
                        alpha=0.2, color='green', label='Safe region')
    
    ax.set_xlabel(f'{margins_dict["parameter"]}')
    ax.set_ylabel('I(θ)')
    ax.set_title(f'Robustness Margin Analysis\nMargin = {margins_dict["margin"]:.3f}')
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.show()


def plot_sobol_indices(sobol_dict: Dict[str, Any],
                      output_path: Optional[str] = None):
    """Plot Sobol indices"""
    param_names = list(sobol_dict['first_order'].keys())
    S1 = [sobol_dict['first_order'][p] for p in param_names]
    ST = [sobol_dict['total_order'][p] for p in param_names]
    S1_conf = [sobol_dict['first_order_conf'][p] for p in param_names]
    ST_conf = [sobol_dict['total_order_conf'][p] for p in param_names]
    
    x = np.arange(len(param_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(x - width/2, S1, width, yerr=S1_conf, label='S₁ (First-order)', 
           alpha=0.8, capsize=5, color='steelblue')
    ax.bar(x + width/2, ST, width, yerr=ST_conf, label='Sᵀ (Total-effect)', 
           alpha=0.8, capsize=5, color='coral')
    
    ax.set_xlabel('Parameter')
    ax.set_ylabel('Sobol Index')
    ax.set_title('Variance-based Sensitivity: Sobol Indices')
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.show()


# ============================================================================
# Main Analysis Pipeline
# ============================================================================

def run_full_sensitivity_analysis(data_dir: str,
                                  output_dir: str,
                                  param_name: str = 'scale_factor',
                                  threshold: float = 0.5):
    """
    Run complete sensitivity analysis pipeline.
    
    Args:
        data_dir: Directory with MATLAB JSON exports
        output_dir: Directory for output figures and results
        param_name: Primary parameter to analyze
        threshold: Inconsistency threshold for robustness margins
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("SENSITIVITY ANALYSIS FOR GLOBAL INCONSISTENCY I(θ)")
    print("="*80)
    
    # 1. Load data
    df = load_experimental_data(data_dir)
    
    # 2. Total causal effects
    print("\n" + "="*80)
    print("1. TOTAL CAUSAL EFFECTS τ_j(a,b)")
    print("="*80)
    causal_effects = compute_causal_effects(df, param_name)
    plot_causal_effects(causal_effects, 
                       output_path / f'causal_effects_{param_name}.png')
    
    # 3. Local sensitivity
    print("\n" + "="*80)
    print("2. LOCAL SENSITIVITY ∂I/∂θ_j")
    print("="*80)
    local_sens = compute_local_sensitivity(df, param_name, method='finite_difference')
    plot_local_sensitivity(local_sens,
                          output_path / f'local_sensitivity_{param_name}.png')
    
    # 4. Robustness margins
    print("\n" + "="*80)
    print("3. ROBUSTNESS MARGINS s*(τ)")
    print("="*80)
    margins = compute_robustness_margins(df, param_name, threshold=threshold)
    plot_robustness_margins(margins,
                           output_path / f'robustness_margins_{param_name}.png')
    
    # 5. Sobol indices (if enough parameters)
    theta, I_theta, theta_columns = extract_theta_and_I(df)
    if theta.shape[1] >= 2 and SALIB_AVAILABLE:
        print("\n" + "="*80)
        print("4. SOBOL INDICES (Variance-based)")
        print("="*80)
        sobol_results = compute_sobol_indices(theta, I_theta, theta_columns, 
                                             n_samples=1024, calc_second_order=False)
        plot_sobol_indices(sobol_results,
                          output_path / 'sobol_indices.png')
        
        # Save results
        with open(output_path / 'sobol_indices.pkl', 'wb') as f:
            pickle.dump(sobol_results, f)
    
    # 6. Fit surrogate model
    print("\n" + "="*80)
    print("5. SURROGATE MODEL")
    print("="*80)
    surrogate = fit_surrogate_model(theta, I_theta, model_type='gp')
    
    # Save surrogate
    with open(output_path / 'surrogate_model.pkl', 'wb') as f:
        pickle.dump(surrogate, f)
    
    # Save summary
    summary = {
        'data_shape': df.shape,
        'parameters': theta_columns,
        'I_theta_stats': {
            'mean': float(I_theta.mean()),
            'std': float(I_theta.std()),
            'min': float(I_theta.min()),
            'max': float(I_theta.max())
        },
        'causal_effects': causal_effects.to_dict(orient='records'),
        'local_sensitivity': {
            'mean': float(local_sens['mean_sensitivity']),
            'max': float(local_sens['max_sensitivity'])
        },
        'robustness_margins': {
            's_star': margins['s_star'],
            'margin': float(margins['margin'])
        }
    }
    
    with open(output_path / 'sensitivity_analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {output_path}")
    print("="*80)


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sensitivity Analysis for Global Inconsistency I(θ)'
    )
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing MATLAB JSON exports')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory for output figures and results')
    parser.add_argument('--param', type=str, default='scale_factor',
                       help='Primary parameter to analyze')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Inconsistency threshold for robustness margins')
    
    args = parser.parse_args()
    
    run_full_sensitivity_analysis(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        param_name=args.param,
        threshold=args.threshold
    )
