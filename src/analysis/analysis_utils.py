"""
Shared Utilities for Sensitivity and Metric Comparison Analysis

Common functionality for:
- sensitivity_analysis.py: Parameter sensitivity and robustness analysis
- metric_comparison.py: Comparative metric evaluation

Author: CPS Uncertainty-Propagation Framework / Causality Extension
Date: 2026-02-10
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings

# Set consistent plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


# ============================================================================
# Data Loading (Common)
# ============================================================================

def load_experimental_data(data_dir: str, 
                          pattern: str = 'results_*.json',
                          verbose: bool = True) -> pd.DataFrame:
    """
    Load experimental data from MATLAB JSON exports.
    
    Args:
        data_dir: Directory containing JSON files
        pattern: Filename pattern for JSON files
        verbose: Print loading progress
        
    Returns:
        DataFrame with columns: [scenario, param_value, I_theta, metrics, etc.]
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    json_files = sorted(data_path.glob(pattern))
    
    if len(json_files) == 0:
        raise FileNotFoundError(f"No JSON files matching '{pattern}' in {data_path}")
    
    if verbose:
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
        
        # Extract scenario name
        scenario_name = data.get('scenario', json_file.stem)
        
        # Process each experiment
        experiments = data.get('experiments', [])
        
        for exp in experiments:
            # Basic info - handle MATLAB export format
            record = {
                'scenario': scenario_name,
                'param_value': exp.get('intervention_value', exp.get('param_value')),
                'intervention_type': exp.get('intervention_type', 'unknown'),
            }
            
            # Handle nested state structure (MATLAB export)
            pre_state = exp.get('pre_state', {})
            post_state = exp.get('post_state', {})
            
            # Pre-intervention inconsistency
            pre_inconsistency = exp.get('pre_inconsistency', pre_state.get('inconsistency', {}))
            
            # Prefer MFMC-corrected value; fall back to plain MC probability
            I_theta_pre = pre_inconsistency.get('I_MF_sobol',
                            pre_inconsistency.get('I_MF_halton',
                            pre_inconsistency.get('I_MF_random',
                            pre_inconsistency.get('I_theta'))))
            if I_theta_pre is None or np.isnan(I_theta_pre):
                mc_p_consistent = pre_inconsistency.get('mc_p_consistent_sobol',
                                                       pre_inconsistency.get('mc_probability'))
                if mc_p_consistent is not None:
                    I_theta_pre = 1.0 - mc_p_consistent
            
            record['I_theta'] = I_theta_pre
            record['I_theta_se'] = pre_inconsistency.get('mc_standard_error_sobol', np.nan)
            record['jaccard_index'] = pre_inconsistency.get('jaccard_index', np.nan)
            record['mc_probability'] = pre_inconsistency.get('mc_p_consistent_sobol', 
                                                            pre_inconsistency.get('mc_probability', np.nan))
            
            # Pre-intervention uncertainty
            pre_uncertainty = exp.get('pre_uncertainty', pre_state.get('uncertainty', {}))
            record['source_volume'] = pre_uncertainty.get('source_volume', np.nan)
            record['target_volume'] = pre_uncertainty.get('target_volume', np.nan)
            
            # Post-intervention inconsistency
            post_inconsistency = exp.get('post_inconsistency', post_state.get('inconsistency', {}))
            
            # Prefer MFMC-corrected value; fall back to plain MC probability
            I_theta_post = post_inconsistency.get('I_MF_sobol',
                             post_inconsistency.get('I_MF_halton',
                             post_inconsistency.get('I_MF_random',
                             post_inconsistency.get('I_theta'))))
            if I_theta_post is None or np.isnan(I_theta_post):
                mc_p_consistent_post = post_inconsistency.get('mc_p_consistent_sobol',
                                                              post_inconsistency.get('mc_probability'))
                if mc_p_consistent_post is not None:
                    I_theta_post = 1.0 - mc_p_consistent_post
            
            record['post_I_theta'] = I_theta_post
            record['post_jaccard'] = post_inconsistency.get('jaccard_index', np.nan)
            
            # Causal effects
            causal_effects = exp.get('causal_effects', exp.get('causal_effect', {}))
            delta_I = causal_effects.get('delta_I_theta')
            if delta_I is None and I_theta_post is not None and I_theta_pre is not None:
                delta_I = I_theta_post - I_theta_pre
            
            record['delta_I_theta'] = delta_I
            record['delta_jaccard'] = causal_effects.get('delta_jaccard', np.nan)
            
            # Extract parameter configurations (for sensitivity analysis)
            params = exp.get('parameters', {})
            for key, value in params.items():
                if isinstance(value, (int, float)):
                    record[f'param_{key}'] = value
            
            all_records.append(record)
    
    df = pd.DataFrame(all_records)
    
    if verbose:
        print(f"Loaded {len(df)} records with I(θ) measurements")
        if 'I_theta' in df.columns:
            valid_I = df['I_theta'].dropna()
            if len(valid_I) > 0:
                print(f"I(θ) range: [{valid_I.min():.3f}, {valid_I.max():.3f}]")
                print(f"I(θ) mean: {valid_I.mean():.3f} ± {valid_I.std():.3f}")
        print()
    
    return df


# ============================================================================
# Data Extraction and Preprocessing
# ============================================================================

def extract_theta_and_I(df: pd.DataFrame, 
                        theta_columns: Optional[List[str]] = None) -> tuple:
    """
    Extract parameter matrix θ and inconsistency vector I(θ).
    
    Args:
        df: DataFrame from load_experimental_data
        theta_columns: List of column names for θ parameters (auto-detect if None)
        
    Returns:
        (theta, I_theta, theta_column_names) where:
            theta is (n_samples, n_params)
            I_theta is (n_samples,)
            theta_column_names is list of parameter names
    """
    # Auto-detect theta columns if not provided
    if theta_columns is None:
        # Look for columns starting with 'param_' or numeric columns
        candidate_cols = [col for col in df.columns 
                         if col.startswith('param_') or 
                         (col not in ['I_theta', 'I_theta_se', 'scenario', 
                                     'intervention_type', 'jaccard_index',
                                     'mc_probability', 'source_volume', 'target_volume',
                                     'post_I_theta', 'post_jaccard',
                                     'delta_I_theta', 'delta_jaccard'] 
                          and df[col].dtype in [np.float64, np.int64])]
        
        # Verify at least one valid column
        valid_cols = []
        for col in candidate_cols:
            if df[col].notna().sum() > 0:
                valid_cols.append(col)
        
        theta_columns = valid_cols
    
    if len(theta_columns) == 0:
        raise ValueError("No valid parameter columns found in DataFrame")
    
    # Extract theta matrix
    theta = df[theta_columns].values.astype(float)
    I_theta = df['I_theta'].values.astype(float)
    
    # Remove rows with NaN
    valid_mask = ~(np.isnan(theta).any(axis=1) | np.isnan(I_theta))
    theta = theta[valid_mask]
    I_theta = I_theta[valid_mask]
    
    print(f"Extracted θ matrix: {theta.shape}")
    print(f"Parameter columns: {theta_columns}")
    print(f"Parameter ranges:")
    for i, col in enumerate(theta_columns):
        print(f"  {col}: [{theta[:, i].min():.4f}, {theta[:, i].max():.4f}]")
    print()
    
    return theta, I_theta, theta_columns


def compute_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived inconsistency metrics for comparison.
    
    Args:
        df: DataFrame with base metrics
        
    Returns:
        DataFrame with additional columns:
            - 1_minus_jaccard: 1 - jaccard_index
            - volume_ratio: |target_vol - source_vol| / source_vol
            - 1_minus_mc_prob: 1 - mc_probability
    """
    df = df.copy()
    
    # Derived metrics
    if 'jaccard_index' in df.columns:
        df['1_minus_jaccard'] = 1 - df['jaccard_index']
    
    if 'source_volume' in df.columns and 'target_volume' in df.columns:
        df['volume_ratio'] = np.abs(df['target_volume'] - df['source_volume']) / df['source_volume']
    
    if 'mc_probability' in df.columns:
        df['1_minus_mc_prob'] = 1 - df['mc_probability']
    
    return df


# ============================================================================
# Report Generation
# ============================================================================

def save_report(report_lines: List[str], output_path: Path, verbose: bool = True):
    """
    Save text report to file.
    
    Args:
        report_lines: List of strings (report content)
        output_path: Path to save report
        verbose: Print confirmation
    """
    report_text = "\n".join(report_lines)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    if verbose:
        print(f"✓ Saved: {output_path.name}")


def create_report_header(title: str, df: pd.DataFrame) -> List[str]:
    """
    Create standard report header.
    
    Args:
        title: Report title
        df: DataFrame with experimental data
        
    Returns:
        List of report lines
    """
    lines = []
    lines.append("=" * 80)
    lines.append(title)
    lines.append("=" * 80)
    lines.append(f"\nDataset: {len(df)} experiments from {df['scenario'].nunique()} scenarios")
    lines.append(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("\n")
    
    return lines


# ============================================================================
# Plot Utilities
# ============================================================================

def setup_figure(nrows: int = 1, ncols: int = 1, 
                figsize: Optional[tuple] = None,
                title: Optional[str] = None) -> tuple:
    """
    Create figure with consistent styling.
    
    Args:
        nrows: Number of subplot rows
        ncols: Number of subplot columns
        figsize: Figure size (auto if None)
        title: Super title for figure
        
    Returns:
        (fig, axes) matplotlib objects
    """
    if figsize is None:
        figsize = (7 * ncols, 5 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    return fig, axes


def save_figure(fig, output_path: Path, dpi: int = 300, verbose: bool = True):
    """
    Save figure with consistent settings.
    
    Args:
        fig: Matplotlib figure
        output_path: Path to save figure
        dpi: Resolution
        verbose: Print confirmation
    """
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    
    if verbose:
        print(f"✓ Saved: {output_path.name}")
    
    plt.close(fig)


# ============================================================================
# Statistical Utilities
# ============================================================================

def compute_summary_statistics(series: pd.Series, name: str = "Variable") -> Dict[str, float]:
    """
    Compute summary statistics for a series.
    
    Args:
        series: Pandas Series
        name: Variable name
        
    Returns:
        Dict with statistics
    """
    values = series.dropna()
    
    return {
        'name': name,
        'count': len(values),
        'mean': values.mean(),
        'std': values.std(),
        'min': values.min(),
        'q25': values.quantile(0.25),
        'median': values.median(),
        'q75': values.quantile(0.75),
        'max': values.max(),
        'range': values.max() - values.min()
    }


def format_statistics_table(stats_list: List[Dict]) -> str:
    """
    Format list of statistics as ASCII table.
    
    Args:
        stats_list: List of dicts from compute_summary_statistics
        
    Returns:
        Formatted table string
    """
    df = pd.DataFrame(stats_list)
    return df.to_string(index=False, float_format=lambda x: f'{x:.4f}')


# ============================================================================
# Path and Directory Utilities
# ============================================================================

def ensure_output_dir(output_dir: str) -> Path:
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_dir: Directory path
        
    Returns:
        Path object
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def print_analysis_header(analysis_name: str):
    """Print consistent analysis header."""
    print("=" * 80)
    print(analysis_name)
    print("=" * 80)
    print()


def print_analysis_footer(output_dir: Path, generated_files: List[str]):
    """Print consistent analysis footer."""
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    for file in generated_files:
        print(f"  - {file}")
    print()


# ============================================================================
# Export to JSON
# ============================================================================

def export_results_to_json(results: Dict[str, Any], 
                           output_path: Path,
                           verbose: bool = True):
    """
    Export analysis results to JSON file.
    
    Args:
        results: Dictionary with results
        output_path: Path to save JSON
        verbose: Print confirmation
    """
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert_to_json_serializable(results)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2)
    
    if verbose:
        print(f"✓ Saved: {output_path.name}")
