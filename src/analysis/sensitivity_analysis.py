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
from datetime import datetime
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.interpolate import interp1d

# Import shared utilities
from analysis_utils import (
    load_experimental_data,
    extract_theta_and_I,
    compute_derived_metrics,
    save_report,
    create_report_header,
    setup_figure,
    save_figure,
    ensure_output_dir,
    print_analysis_header,
    print_analysis_footer,
    export_results_to_json
)

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
            # MATLAB export uses nested structure with post_state
            I_theta = None
            theta_params = {}
            
            # Option 1: Flattened MATLAB export (post_inconsistency.I_theta)
            if 'post_inconsistency' in record and isinstance(record['post_inconsistency'], dict):
                post_inc = record['post_inconsistency']
                _v = post_inc.get('I_MF_random')
                I_theta = (1.0 - _v) if _v is not None else None

                if I_theta is not None:
                    I_theta_se = post_inc.get('mc_standard_error_sobol', post_inc.get('I_theta_se', np.nan))
                    I_theta_ci95_lower = post_inc.get('I_theta_ci95_lower', np.nan)
                    I_theta_ci95_upper = post_inc.get('I_theta_ci95_upper', np.nan)
                    
                    # Extract theta from record metadata
                    if 'param_value' in record:
                        # Use 'intervention_value' as the parameter name for all interventions
                        theta_params['intervention_value'] = record['param_value']
                        # Also store intervention type for filtering
                        theta_params['intervention'] = record.get('intervention', 'unknown')
                    elif 'intervention_value' in record:
                        theta_params['intervention_value'] = record['intervention_value']
                        theta_params['intervention'] = record.get('intervention_type', 'unknown')
            
            # Option 2: Nested structure (post_state.inconsistency.I_theta)
            elif 'post_state' in record:
                post_state = record['post_state']
                if 'inconsistency' not in post_state:
                    continue
                
                inconsistency = post_state['inconsistency']
                _v = inconsistency.get('I_MF_random')
                if _v is None:
                    continue
                I_theta = 1.0 - _v
                
                I_theta_se = inconsistency.get('mc_standard_error_sobol', inconsistency.get('I_theta_se', np.nan))
                I_theta_ci95_lower = inconsistency.get('I_theta_ci95_lower', np.nan)
                I_theta_ci95_upper = inconsistency.get('I_theta_ci95_upper', np.nan)
                
                # Extract intervention info
                theta_params['intervention_value'] = record.get('intervention_value', record.get('param_value', np.nan))
                theta_params['intervention'] = record.get('intervention_type', record.get('intervention', 'unknown'))
            
            if I_theta is None:
                continue
            
            # Build flattened record
            flat_record = {
                'I_theta': I_theta,
                'I_theta_se': I_theta_se,
                'I_theta_ci95_lower': I_theta_ci95_lower,
                'I_theta_ci95_upper': I_theta_ci95_upper,
                'intervention_type': record.get('intervention_type', record.get('intervention', 'unknown')),
                **theta_params
            }
            
            # Add pre_uncertainty metrics (for Sobol analysis)
            pre_unc = record.get('pre_uncertainty', record.get('pre_state', {}).get('uncertainty', {}))
            if pre_unc:
                flat_record['pre_source_volume'] = pre_unc.get('source_volume', np.nan)
                flat_record['pre_source_radius'] = pre_unc.get('source_radius', np.nan)
                flat_record['pre_source_n_generators'] = pre_unc.get('source_n_generators', np.nan)
                flat_record['pre_source_correlation'] = pre_unc.get('source_correlation', np.nan)
                flat_record['pre_target_volume'] = pre_unc.get('target_volume', np.nan)
                flat_record['pre_target_radius'] = pre_unc.get('target_radius', np.nan)
                flat_record['pre_target_n_generators'] = pre_unc.get('target_n_generators', np.nan)
            
            # Add optional metrics - compute I_theta_pre if needed
            pre_inc = None
            if 'pre_inconsistency' in record and isinstance(record['pre_inconsistency'], dict):
                pre_inc = record['pre_inconsistency']
            elif 'pre_state' in record:
                pre_inc = record['pre_state'].get('inconsistency', {})
            
            if pre_inc:
                _v = pre_inc.get('I_MF_random')
                I_theta_pre = (1.0 - _v) if _v is not None else None
                flat_record['I_theta_pre'] = I_theta_pre
            
            # Compute delta_I_theta if available
            causal_eff = record.get('causal_effect', record.get('causal_effects', {}))
            if isinstance(causal_eff, dict):
                delta_I = causal_eff.get('delta_I_theta')
                if delta_I is None and 'I_theta_pre' in flat_record and flat_record['I_theta_pre'] is not None:
                    delta_I = flat_record['I_theta'] - flat_record['I_theta_pre']
                flat_record['delta_I_theta'] = delta_I if delta_I is not None else np.nan
            
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
                       'intervention_type', 'intervention', 'I_theta_pre', 'delta_I_theta',
                       'scenario_id', 'repeat_idx', 'run_id', 'mc_num_samples', 'mc_num_consistent']
        # Include numeric columns that represent uncertainty parameters
        # Prioritize: intervention_value, pre_source_volume, pre_source_radius, pre_source_correlation
        priority_cols = ['intervention_value', 'pre_source_volume', 'pre_source_radius', 
                        'pre_source_correlation', 'pre_source_n_generators',
                        'pre_target_volume', 'pre_target_radius']
        
        theta_columns = []
        for col in priority_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                # Only include if there's variation (not constant)
                if df[col].nunique() > 1:
                    theta_columns.append(col)
        
        # Add other numeric columns not excluded
        for col in df.columns:
            if col not in exclude_cols and col not in theta_columns and pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique() > 1:  # Only if varies
                    theta_columns.append(col)
    
    if len(theta_columns) == 0:
        raise ValueError("No numeric theta columns found in data")
    
    # Extract theta matrix
    theta = df[theta_columns].values.astype(float)
    I_theta = df['I_theta'].values.astype(float)
    
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
    
    # Remove duplicate theta values by averaging I_theta
    df_agg = df_sorted.groupby(param_name).agg({'I_theta': 'mean'}).reset_index()
    theta_vals = df_agg[param_name].values
    I_vals = df_agg['I_theta'].values
    
    if len(theta_vals) < 2:
        raise ValueError(f"Need at least 2 data points for sensitivity analysis")
    
    sensitivities = []
    
    if method == 'finite_difference':
        # Central differences where possible, forward/backward at boundaries
        for i in range(len(theta_vals)):
            if i == 0:
                # Forward difference
                delta_theta = theta_vals[i+1] - theta_vals[i]
                if delta_theta < 1e-10:  # Avoid division by near-zero
                    dI_dtheta = 0.0
                else:
                    dI_dtheta = (I_vals[i+1] - I_vals[i]) / delta_theta
            elif i == len(theta_vals) - 1:
                # Backward difference
                delta_theta = theta_vals[i] - theta_vals[i-1]
                if delta_theta < 1e-10:
                    dI_dtheta = 0.0
                else:
                    dI_dtheta = (I_vals[i] - I_vals[i-1]) / delta_theta
            else:
                # Central difference
                delta_theta = theta_vals[i+1] - theta_vals[i-1]
                if delta_theta < 1e-10:
                    dI_dtheta = 0.0
                else:
                    dI_dtheta = (I_vals[i+1] - I_vals[i-1]) / delta_theta
            
            # Clip extreme values to prevent Inf
            dI_dtheta = np.clip(dI_dtheta, -1e6, 1e6)
            
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
                               threshold: float = 0.5,
                               intervention_type: str = 'scale') -> Dict[str, Any]:
    """
    Compute robustness margin: s*(τ) = sup{s : E[I(θ(s))] ≤ τ}
    
    How much can we change parameter before exceeding inconsistency threshold?
    
    Args:
        df: Experimental data
        param_name: Parameter to analyze
        threshold: Inconsistency threshold τ
        intervention_type: Type of intervention ('scale' for widen/shrink, 'shift', 'correlate')
        
    Returns:
        Dict with critical values and safety margins
    """
    # Determine baseline (no-change value) based on intervention type
    if intervention_type in ['widen', 'shrink', 'scale']:
        baseline = 1.0  # 1.0 = no scaling
    elif intervention_type in ['shift', 'correlate']:
        baseline = 0.0  # 0.0 = no shift/correlation
    else:
        baseline = 0.0  # Default to 0.0 for unknown types
    if param_name not in df.columns:
        raise ValueError(f"Parameter '{param_name}' not found in data")
    
    # Aggregate data by parameter value (handle duplicates from repeats)
    df_agg = df[[param_name, 'I_theta']].dropna().groupby(param_name).agg({
        'I_theta': 'mean'
    }).reset_index()
    
    # Sort by parameter
    df_sorted = df_agg.sort_values(param_name)
    
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
    
    # Robustness margin from baseline
    if len(crossings) > 0:
        s_star = crossings[0]  # First crossing
        margin = abs(s_star - baseline)
        status = "THRESHOLD_CROSSED"
    else:
        if I_range.min() > threshold:
            # System always exceeds threshold (always inconsistent)
            s_star = None
            margin = 0.0
            status = "ALWAYS_ABOVE_THRESHOLD"
            print(f"⚠️  WARNING: I(θ) always > {threshold} (min={I_range.min():.4f})")
            print(f"    System is persistently inconsistent across all parameter values")
        else:
            # System always below threshold (always safe)
            s_star = param_range[-1]
            margin = abs(s_star - baseline)
            status = "ALWAYS_BELOW_THRESHOLD"
    
    print(f"\n=== Robustness Margin for {param_name} ===")
    print(f"Intervention type: {intervention_type}")
    print(f"Baseline (no-change): {baseline}")
    print(f"Threshold τ = {threshold}")
    print(f"Status: {status}")
    print(f"Critical value s* = {s_star}")
    print(f"Safety margin = {margin:.3f} (from baseline)")
    print(f"I(θ) range: [{I_range.min():.4f}, {I_range.max():.4f}]")
    print()
    
    return {
        'param_name': param_name,
        'intervention_type': intervention_type,
        'baseline': baseline,
        'threshold': threshold,
        's_star': s_star,
        'margin': margin,
        'status': status,
        'I_theta_min': float(I_range.min()),
        'I_theta_max': float(I_range.max()),
        'crossings': crossings,
        'param_range': param_range.tolist(),
        'I_range': I_range.tolist()
    }


# ============================================================================
# 4. Sobol Indices (Variance-based Sensitivity)
# ============================================================================

def compute_sobol_indices_raw_mc(theta: np.ndarray,
                                I_theta: np.ndarray,
                                param_names: List[str],
                                n_bootstrap: int = 100) -> Dict[str, Any]:
    """
    Compute Sobol sensitivity indices using RAW Monte Carlo (Saltelli scheme)
    directly from experimental data WITHOUT surrogate models.
    
    This implements the raw Saltelli formulas:
    - First-order: S_i ≈ (1/N) Σ f(B) * [f(A_B^(i)) - f(A)] / Var(Y)
    - Total-order: S_Ti ≈ (1/2N) Σ [f(A) - f(A_B^(i))]² / Var(Y)
    
    Args:
        theta: Parameter matrix (n_samples, n_params) - raw experimental data
        I_theta: Inconsistency values (n_samples,) - raw model outputs
        param_names: Names of parameters
        n_bootstrap: Number of bootstrap samples for confidence intervals
        
    Returns:
        Dict with first-order and total-effect Sobol indices
    """
    n_samples, n_params = theta.shape
    
    print("\n=== Computing Sobol Indices (Raw MC - Saltelli Scheme) ===")
    print(f"Using {n_samples} raw experimental samples (NO surrogate model)")
    print(f"Parameters: {param_names}")
    
    # Split data into two independent matrices A and B (Saltelli scheme)
    n_half = n_samples // 2
    if n_half < 10:
        warnings.warn(f"Too few samples ({n_samples}) for reliable Sobol estimation. Need at least 20.")
    
    # Random shuffle to decorrelate
    indices = np.random.permutation(n_samples)
    theta_shuffled = theta[indices]
    I_theta_shuffled = I_theta[indices]
    
    # Create matrices A and B
    A = theta_shuffled[:n_half]
    B = theta_shuffled[n_half:2*n_half]
    f_A = I_theta_shuffled[:n_half]
    f_B = I_theta_shuffled[n_half:2*n_half]
    
    # Variance of output
    var_Y = np.var(I_theta)
    
    if var_Y < 1e-10:
        warnings.warn("Output variance is near zero - Sobol indices undefined")
        return {
            'first_order': {name: 0.0 for name in param_names},
            'first_order_conf': {name: 0.0 for name in param_names},
            'total_order': {name: 0.0 for name in param_names},
            'total_order_conf': {name: 0.0 for name in param_names},
            'method': 'raw_mc_saltelli',
            'warning': 'insufficient_variance'
        }
    
    # For each parameter, create hybrid matrix A_B^(i)
    S1_estimates = np.zeros(n_params)  # First-order
    ST_estimates = np.zeros(n_params)  # Total-order
    
    for i in range(n_params):
        # Create A_B^(i): all columns from A except column i which comes from B
        A_Bi = A.copy()
        A_Bi[:, i] = B[:, i]
        
        # Need to evaluate f(A_B^(i)) - we'll use nearest neighbor from original data
        f_A_Bi = np.zeros(len(A_Bi))
        for j, sample in enumerate(A_Bi):
            # Find nearest sample in original theta
            distances = np.linalg.norm(theta - sample, axis=1)
            nearest_idx = np.argmin(distances)
            f_A_Bi[j] = I_theta[nearest_idx]
        
        # First-order index (Eq. 1 from Saltelli paper):
        # S_i ≈ (1/N) * Σ f(B) * [f(A_B^(i)) - f(A)] / Var(Y)
        numerator_S1 = np.mean(f_B * (f_A_Bi - f_A))
        S1_estimates[i] = numerator_S1 / var_Y
        
        # Total-order index (Eq. 2 from Saltelli paper):
        # S_Ti ≈ (1/2N) * Σ [f(A) - f(A_B^(i))]² / Var(Y)
        numerator_ST = 0.5 * np.mean((f_A - f_A_Bi)**2)
        ST_estimates[i] = numerator_ST / var_Y
    
    # Bootstrap confidence intervals
    S1_bootstrap = np.zeros((n_bootstrap, n_params))
    ST_bootstrap = np.zeros((n_bootstrap, n_params))
    
    for boot_idx in range(n_bootstrap):
        # Resample with replacement
        boot_indices = np.random.choice(n_half, size=n_half, replace=True)
        A_boot = A[boot_indices]
        B_boot = B[boot_indices]
        f_A_boot = f_A[boot_indices]
        f_B_boot = f_B[boot_indices]
        
        for i in range(n_params):
            A_Bi_boot = A_boot.copy()
            A_Bi_boot[:, i] = B_boot[:, i]
            
            f_A_Bi_boot = np.zeros(len(A_Bi_boot))
            for j, sample in enumerate(A_Bi_boot):
                distances = np.linalg.norm(theta - sample, axis=1)
                nearest_idx = np.argmin(distances)
                f_A_Bi_boot[j] = I_theta[nearest_idx]
            
            numerator_S1 = np.mean(f_B_boot * (f_A_Bi_boot - f_A_boot))
            S1_bootstrap[boot_idx, i] = numerator_S1 / var_Y
            
            numerator_ST = 0.5 * np.mean((f_A_boot - f_A_Bi_boot)**2)
            ST_bootstrap[boot_idx, i] = numerator_ST / var_Y
    
    # Compute confidence intervals
    S1_conf = np.std(S1_bootstrap, axis=0) * 1.96  # 95% CI
    ST_conf = np.std(ST_bootstrap, axis=0) * 1.96
    
    # Clip negative values (can occur due to sampling variance)
    S1_estimates = np.clip(S1_estimates, 0, 1)
    ST_estimates = np.clip(ST_estimates, 0, 1)
    
    results = {
        'first_order': dict(zip(param_names, S1_estimates)),
        'first_order_conf': dict(zip(param_names, S1_conf)),
        'total_order': dict(zip(param_names, ST_estimates)),
        'total_order_conf': dict(zip(param_names, ST_conf)),
        'method': 'raw_mc_saltelli',
        'n_samples': n_samples,
        'n_bootstrap': n_bootstrap
    }
    
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


def compute_sobol_indices(theta: np.ndarray,
                         I_theta: np.ndarray,
                         param_names: List[str],
                         n_samples: int = 1024,
                         calc_second_order: bool = False,
                         use_raw_mc: bool = True) -> Dict[str, Any]:
    """
    Compute Sobol sensitivity indices.
    
    First-order: S_i = Var_θi(E[I(θ)|θi])/Var(I(θ))
    Total-effect: S_i^T = 1 - Var_θ~i(E[I(θ)|θ~i])/Var(I(θ))
    
    Args:
        theta: Parameter matrix (n_samples, n_params)
        I_theta: Inconsistency values (n_samples,)
        param_names: Names of parameters
        n_samples: Number of samples for Saltelli sampling (if surrogate needed)
        calc_second_order: Compute second-order interactions
        use_raw_mc: If True, use raw MC Saltelli; if False, use surrogate+SALib
        
    Returns:
        Dict with first-order and total-effect Sobol indices
    """
    # Prefer raw MC for experimental data
    if use_raw_mc:
        return compute_sobol_indices_raw_mc(theta, I_theta, param_names, n_bootstrap=100)
    
    # Fallback to surrogate-based SALib method
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
    print("\n=== Computing Sobol Indices (Surrogate Method) ===")
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
        'method': 'surrogate_salib'
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
    ax.set_title(f'Total Causal Effects on I(θ)\n{format_parameter_name(results_df["parameter"].iloc[0])}')
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
    ax1.set_xlabel(f'{format_parameter_name(sens_dict["parameter"])}')
    ax1.set_ylabel('I(θ)')
    ax1.set_title('Global Inconsistency I(θ)')
    ax1.grid(alpha=0.3)
    ax1.legend()
    
    # ∂I/∂θ vs θ
    ax2.plot(sens_df['theta'], sens_df['dI_dtheta'], 's-', color='coral', label='∂I/∂θ')
    ax2.axhline(0, color='k', linestyle='--', linewidth=0.8)
    ax2.set_xlabel(f'{format_parameter_name(sens_dict["parameter"])}')
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
    
    # Show baseline (no-change value)
    baseline = margins_dict.get('baseline', 1.0)
    ax.axvline(baseline, color='gray', linestyle=':', linewidth=1.5, 
              label=f'Baseline={baseline}')
    
    if margins_dict['s_star'] is not None:
        ax.axvline(margins_dict['s_star'], color='green', linestyle='--',
                  linewidth=2, label=f's*={margins_dict["s_star"]:.3f}')
        # Fill safe region between baseline and s_star
        ax.fill_betweenx([0, 1], baseline, margins_dict['s_star'], 
                        alpha=0.2, color='green', label='Safe region')
    
    interv_type = margins_dict.get('intervention_type', 'unknown')
    ax.set_xlabel(f'{format_parameter_name(margins_dict["param_name"])}')
    ax.set_ylabel('I(θ)')
    ax.set_title(f'Robustness Margin Analysis ({interv_type})\nStatus: {margins_dict.get("status", "UNKNOWN")}\nMargin = {margins_dict["margin"]:.3f} from baseline={baseline}')
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.show()


def format_parameter_name(param_name: str) -> str:
    """Format parameter names for better readability in plots."""
    # Define common replacements with updated naming conventions
    name_map = {
        # Intervention parameters
        'param_value': 'Intervention Magnitude',
        'intervention_value': 'Intervention Magnitude',
        'scale_factor': 'Scale Factor',
        'correlation_strength': 'Correlation Strength',
        'center_delta': 'Center Shift',
        
        # Source uncertainty parameters
        'pre_source_volume': 'Source Volume (Pre)',
        'pre_source_radius': 'Source Radius (Pre)',
        'pre_source_n_generators': 'Source Generators (Pre)',
        'pre_source_correlation': 'Source Correlation (Pre)',
        
        # Target uncertainty parameters
        'pre_target_volume': 'Target Volume (Pre)',
        'pre_target_radius': 'Target Radius (Pre)',
        'pre_target_n_generators': 'Target Generators (Pre)',
        
        # Consistency metrics (new naming)
        'jaccard_C': 'Jaccard C',
        'jaccard_Csym': 'Jaccard C (Symmetric)',
        'jaccard_index': 'Jaccard Index',
        'mc_probability_sobol': 'MC P(Consistent) - Sobol',
        'mc_probability_halton': 'MC P(Consistent) - Halton',
        'mc_probability_lhs': 'MC P(Consistent) - LHS',
        'mc_probability_random': 'MC P(Consistent) - Random',
        'mc_p_inconsistent_sobol': 'MC P(Inconsistent) - Sobol',
        'mc_p_inconsistent_halton': 'MC P(Inconsistent) - Halton',
        'mc_p_inconsistent_lhs': 'MC P(Inconsistent) - LHS',
        'mc_p_inconsistent_random': 'MC P(Inconsistent) - Random',
        'I_theta': 'Global Inconsistency I(θ)',
        
        # Other parameters
        'uncertainty_scale': 'Uncertainty Scale',
    }
    
    # Check if exact match exists
    if param_name in name_map:
        return name_map[param_name]
    
    # Otherwise, format by replacing underscores and capitalizing
    formatted = param_name.replace('_', ' ').replace('pre ', '').title()
    return formatted


def plot_sobol_indices(sobol_dict: Dict[str, Any],
                      output_path: Optional[str] = None):
    """Plot Sobol indices"""
    param_names = list(sobol_dict['first_order'].keys())
    S1 = [sobol_dict['first_order'][p] for p in param_names]
    ST = [sobol_dict['total_order'][p] for p in param_names]
    S1_conf = [sobol_dict['first_order_conf'][p] for p in param_names]
    ST_conf = [sobol_dict['total_order_conf'][p] for p in param_names]
    
    # Format parameter names for display
    param_labels = [format_parameter_name(p) for p in param_names]
    
    x = np.arange(len(param_names))
    width = 0.35
    
    # Adjust figure size based on number of parameters
    fig_width = max(10, len(param_names) * 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    
    ax.bar(x - width/2, S1, width, yerr=S1_conf, label='S₁ (First-order)', 
           alpha=0.8, capsize=5, color='steelblue')
    ax.bar(x + width/2, ST, width, yerr=ST_conf, label='Sᵀ (Total-effect)', 
           alpha=0.8, capsize=5, color='coral')
    
    ax.set_xlabel('Parameter', fontsize=12)
    ax.set_ylabel('Sobol Index', fontsize=12)
    ax.set_title('Variance-based Sensitivity: Sobol Indices', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(param_labels, rotation=45, ha='right', fontsize=10)
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
                                  param_name: str = 'auto',
                                  threshold: float = 0.5,
                                  run_sobol: bool = True,
                                  split_by_intervention: bool = True):
    """
    Run complete sensitivity analysis pipeline.
    
    Args:
        data_dir: Directory with MATLAB JSON exports
        output_dir: Directory for output figures and results (timestamped subfolder will be created)
        param_name: Primary parameter to analyze ('auto' to auto-detect)
        threshold: Inconsistency threshold for robustness margins
        run_sobol: Whether to compute Sobol indices (requires SALib)
        split_by_intervention: If True, analyze each intervention type separately
    """
    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir_timestamped = Path(output_dir) / f"sensitivity_analysis_{timestamp}"
    output_path = output_dir_timestamped
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("SENSITIVITY ANALYSIS FOR GLOBAL INCONSISTENCY I(θ)")
    print("="*80)
    print(f"Output directory: {output_path}")
    print(f"Timestamp: {timestamp}")
    print(f"Split by intervention: {split_by_intervention}")
    print("="*80)
    
    # 1. Load data
    df = load_experimental_data(data_dir)
    
    # Check if we need to split by intervention type
    intervention_types = []
    if split_by_intervention and 'intervention_type' in df.columns:
        intervention_types = sorted(df['intervention_type'].dropna().unique())
        print(f"\n🔍 Detected {len(intervention_types)} intervention types: {intervention_types}")
    
    if len(intervention_types) <= 1:
        # No splitting needed or only one type
        split_by_intervention = False
        print("\n→ Analyzing all data together (no intervention split)")
    else:
        print(f"\n→ Will analyze each intervention type separately")
    
    # Auto-detect parameter name if needed
    if param_name == 'auto':
        # Look for common parameter names
        candidate_params = ['intervention_value', 'scale_factor', 'uncertainty_scale', 'alpha', 'beta']
        found_param = None
        for candidate in candidate_params:
            if candidate in df.columns and df[candidate].notna().sum() > 0:
                found_param = candidate
                break
        
        if found_param is None:
            # Try to find any numeric column that varies
            for col in df.columns:
                if col not in ['I_theta', 'I_theta_se', 'jaccard_index', 'mc_probability',
                              'source_volume', 'target_volume', 'post_I_theta', 'post_jaccard',
                              'delta_I_theta', 'delta_jaccard']:
                    if df[col].dtype in [np.float64, np.int64] and df[col].nunique() > 1:
                        found_param = col
                        break
        
        if found_param is None:
            raise ValueError("Could not auto-detect parameter. Available columns: " + 
                           ", ".join(df.columns))
        
        param_name = found_param
        print(f"\n🔍 Auto-detected parameter: '{param_name}'")
        print(f"   Values: {sorted(df[param_name].unique())}")
        print()
    
    # Function to run analysis on a subset
    def analyze_subset(df_subset, subset_name, subset_dir):
        """Run analysis on a data subset (e.g., one intervention type)"""
        print("\n" + "="*80)
        print(f"ANALYZING: {subset_name}")
        print(f"N samples: {len(df_subset)}")
        print("="*80)
        
        # 2. Total causal effects
        print("\n" + "-"*80)
        print("1. TOTAL CAUSAL EFFECTS τ_j(a,b)")
        print("-"*80)
        try:
            causal_effects = compute_causal_effects(df_subset, param_name)
            plot_causal_effects(causal_effects, 
                              subset_dir / f'causal_effects_{param_name}.png')
        except Exception as e:
            print(f"⚠ Causal effects failed: {e}")
        
        # 3. Local sensitivity
        print("\n" + "-"*80)
        print("2. LOCAL SENSITIVITY ∂I/∂θ_j")
        print("-"*80)
        try:
            local_sens = compute_local_sensitivity(df_subset, param_name, method='finite_difference')
            plot_local_sensitivity(local_sens,
                                 subset_dir / f'local_sensitivity_{param_name}.png')
        except Exception as e:
            print(f"⚠ Local sensitivity failed: {e}")
        
        # 4. Robustness margins
        print("\n" + "-"*80)
        print("3. ROBUSTNESS MARGINS s*(τ)")
        print("-"*80)
        try:
            # Extract intervention type for this subset
            interv_type = df_subset['intervention_type'].iloc[0] if 'intervention_type' in df_subset.columns else 'scale'
            margins = compute_robustness_margins(df_subset, param_name, threshold=threshold,
                                                intervention_type=interv_type)
            plot_robustness_margins(margins,
                                  subset_dir / f'robustness_margins_{param_name}.png')
        except Exception as e:
            print(f"⚠ Robustness margins failed: {e}")
        
        # 5. Sobol indices (if enough parameters)
        theta, I_theta, theta_columns = extract_theta_and_I(df_subset)
        print(f"\nExtracted parameters: {theta_columns}")
        print(f"Theta shape: {theta.shape}")
        
        # If only 1 parameter, try to add scenario-level features
        if theta.shape[1] == 1:
            print("\n⚠ Only 1 parameter found. Attempting to add scenario-level features...")
            additional_features = ['pre_source_volume', 'pre_source_radius', 
                                  'pre_target_volume', 'pre_target_radius',
                                  'pre_source_n_generators', 'pre_target_n_generators']
            
            added_cols = []
            for col in additional_features:
                if col in df_subset.columns and pd.api.types.is_numeric_dtype(df_subset[col]):
                    # Check if varies across dataset
                    if df_subset[col].nunique() > 1:
                        added_cols.append(col)
            
            if added_cols:
                print(f"  Adding scenario features: {added_cols}")
                theta_columns_expanded = theta_columns + added_cols
                theta_expanded = df_subset[theta_columns_expanded].values.astype(float)
                theta, I_theta, theta_columns = theta_expanded, I_theta, theta_columns_expanded
        
        # Sobol analysis
        if run_sobol and theta.shape[1] >= 2:
            print("\n" + "-"*80)
            print("4. SOBOL SENSITIVITY INDICES (Raw MC - Saltelli)")
            print("-"*80)
            try:
                # Use raw MC instead of surrogate
                sobol_results = compute_sobol_indices(theta, I_theta, theta_columns, 
                                                     use_raw_mc=True)
                plot_sobol_indices(sobol_results, 
                                 subset_dir / f'sobol_indices.png')
                
                # Save results
                with open(subset_dir / 'sobol_indices.json', 'w') as f:
                    # Convert to JSON-serializable format
                    json_results = {k: (v.tolist() if hasattr(v, 'tolist') else v) 
                                  for k, v in sobol_results.items()}
                    json.dump(json_results, f, indent=2)
                
            except Exception as e:
                print(f"⚠ Sobol analysis failed: {e}")
                import traceback
                traceback.print_exc()
        elif theta.shape[1] < 2:
            print(f"\n⚠ Skipping Sobol indices: need at least 2 parameters, found {theta.shape[1]}")
        
        print(f"\n✓ Analysis complete for {subset_name}")
        return True
    
    # Run analysis (split or combined)
    if split_by_intervention:
        # Analyze each intervention type separately
        print("\n" + "="*80)
        print(f"RUNNING SENSITIVITY ANALYSIS (SPLIT BY INTERVENTION)")
        print("="*80)
        for interv_type in intervention_types:
            df_subset = df[df['intervention_type'] == interv_type].copy()
            subset_dir = output_path / f'intervention_{interv_type}'
            subset_dir.mkdir(exist_ok=True)
            
            analyze_subset(df_subset, f"Intervention: {interv_type}", subset_dir)
    else:
        # Analyze all data together
        print("\n" + "="*80)
        print(f"RUNNING SENSITIVITY ANALYSIS (COMBINED)")
        print("="*80)
        analyze_subset(df, "All Data", output_path)
    
    print("\n" + "="*80)
    print("FULL SENSITIVITY ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll results saved to:")
    print(f"  {output_path.absolute()}")
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
    parser.add_argument('--param', type=str, default='intervention_value',
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
