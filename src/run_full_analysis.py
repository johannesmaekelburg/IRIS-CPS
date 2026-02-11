"""
Master Analysis Pipeline

Runs both sensitivity analysis and metric comparison sequentially or independently.

Usage:
    # Run both analyses:
    python run_full_analysis.py --data_dir ../data/measurements/both --output_dir ../results/full_analysis
    
    # Run only sensitivity analysis:
    python run_full_analysis.py --data_dir ../data/measurements/both --output_dir ../results --only sensitivity
    
    # Run only metric comparison:
    python run_full_analysis.py --data_dir ../data/measurements/both --output_dir ../results --only comparison

    # Analyze Jaccard-only measurements:
    python run_full_analysis.py --data_dir ../data/measurements/jaccard --output_dir ../results/jaccard_analysis

Author: CPS Uncertainty-Propagation Framework / Causality Extension
Date: 2026-02-10
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import individual analysis modules
try:
    from sensitivity_analysis import run_full_sensitivity_analysis
    SENSITIVITY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import sensitivity_analysis: {e}")
    SENSITIVITY_AVAILABLE = False

try:
    from metric_comparison import run_metric_comparison_analysis
    COMPARISON_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import metric_comparison: {e}")
    COMPARISON_AVAILABLE = False

from analysis_utils import ensure_output_dir, print_analysis_header, print_analysis_footer


# ============================================================================
# Intervention & Dimension Comparison Plots
# ============================================================================

def load_data_for_comparison(data_dir: str) -> pd.DataFrame:
    """Load all experimental data with dimension and intervention info."""
    data_path = Path(data_dir)
    json_files = sorted(data_path.glob('results_*.json'))
    
    all_records = []
    
    for json_file in json_files:
        name = json_file.stem
        parts = name.split('_')
        
        # Extract dimension (2d, 3d, 4d) and scenario_id
        dim = None
        scenario_id = None
        for i, p in enumerate(parts):
            if p in ['2d', '3d', '4d']:
                dim = int(p[0])
            if p == 'scenario' and i + 1 < len(parts):
                scenario_id = int(parts[i + 1])
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        experiments = data.get('experiments', data)
        
        for exp in experiments:
            record = {
                'dimension': dim,
                'scenario_id': scenario_id,
                'intervention': exp.get('intervention', 'unknown'),
                'param_value': exp.get('param_value', np.nan),
            }
            
            post_inc = exp.get('post_inconsistency', {})
            record['I_theta'] = post_inc.get('I_theta', np.nan)
            record['jaccard_index'] = post_inc.get('jaccard_index', np.nan)
            
            pre_inc = exp.get('pre_inconsistency', {})
            record['I_theta_pre'] = pre_inc.get('I_theta', np.nan)
            
            if not np.isnan(record['I_theta']) and not np.isnan(record['I_theta_pre']):
                record['delta_I_theta'] = record['I_theta'] - record['I_theta_pre']
            else:
                record['delta_I_theta'] = np.nan
            
            all_records.append(record)
    
    return pd.DataFrame(all_records)


def plot_intervention_comparison(df: pd.DataFrame, output_path: Path):
    """Create intervention comparison plot (widen vs shrink vs correlate)."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    interventions = sorted(df['intervention'].unique())
    colors = {'widen': '#e74c3c', 'shrink': '#3498db', 'correlate': '#2ecc71'}
    markers = {'widen': 'o', 'shrink': 's', 'correlate': '^'}
    
    # Plot 1: Mean I(θ) vs param_value for each intervention
    ax1 = axes[0]
    for intervention in interventions:
        subset = df[df['intervention'] == intervention]
        grouped = subset.groupby('param_value')['I_theta'].agg(['mean', 'std']).reset_index()
        
        color = colors.get(intervention, '#95a5a6')
        marker = markers.get(intervention, 'o')
        
        ax1.errorbar(grouped['param_value'], grouped['mean'], 
                    yerr=grouped['std'], 
                    label=intervention.capitalize(),
                    marker=marker, color=color, 
                    capsize=3, linewidth=2, markersize=8)
    
    ax1.set_xlabel('Parameter Value', fontsize=12)
    ax1.set_ylabel('Mean I(θ)', fontsize=12)
    ax1.set_title('Global Inconsistency by Intervention Type', fontsize=14)
    ax1.legend(loc='best')
    ax1.set_ylim(0, 1.05)
    ax1.set_xscale('symlog', linthresh=0.1)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Delta I(θ) distribution by intervention (box plot)
    ax2 = axes[1]
    delta_data = []
    labels = []
    box_colors = []
    for intervention in interventions:
        subset = df[df['intervention'] == intervention]
        delta = subset['delta_I_theta'].dropna()
        if len(delta) > 0:
            delta_data.append(delta.values)
            labels.append(intervention.capitalize())
            box_colors.append(colors.get(intervention, '#95a5a6'))
    
    bp = ax2.boxplot(delta_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Intervention Type', fontsize=12)
    ax2.set_ylabel('ΔI(θ) = I(θ)_post - I(θ)_pre', fontsize=12)
    ax2.set_title('Change in Inconsistency by Intervention', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = output_path / 'intervention_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file.name}")
    plt.close()


def plot_dimension_comparison(df: pd.DataFrame, output_path: Path):
    """Create dimension comparison plot (2D vs 3D vs 4D)."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    dimensions = sorted(df['dimension'].dropna().unique())
    dim_colors = {2: '#3498db', 3: '#2ecc71', 4: '#e74c3c'}
    interventions = sorted(df['intervention'].unique())
    
    # Plot 1: Mean I(θ) by dimension for each intervention
    ax1 = axes[0]
    width = 0.25
    x = np.arange(len(interventions))
    
    for i, dim in enumerate(dimensions):
        means = []
        stds = []
        for intervention in interventions:
            subset = df[(df['dimension'] == dim) & (df['intervention'] == intervention)]
            means.append(subset['I_theta'].mean() if len(subset) > 0 else 0)
            stds.append(subset['I_theta'].std() if len(subset) > 0 else 0)
        
        offset = (i - len(dimensions)/2 + 0.5) * width
        ax1.bar(x + offset, means, width, yerr=stds,
               label=f'{int(dim)}D', color=dim_colors.get(dim, '#95a5a6'),
               alpha=0.8, capsize=3)
    
    ax1.set_xlabel('Intervention Type', fontsize=12)
    ax1.set_ylabel('Mean I(θ)', fontsize=12)
    ax1.set_title('I(θ) by Dimension and Intervention', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels([i.capitalize() for i in interventions])
    ax1.legend(title='Dimension')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: I(θ) trend across dimensions
    ax2 = axes[1]
    for intervention in interventions:
        dim_means = []
        dim_stds = []
        dims_present = []
        for dim in dimensions:
            subset = df[(df['dimension'] == dim) & (df['intervention'] == intervention)]
            if len(subset) > 0:
                dim_means.append(subset['I_theta'].mean())
                dim_stds.append(subset['I_theta'].std())
                dims_present.append(dim)
        
        if len(dims_present) > 0:
            ax2.errorbar(dims_present, dim_means, yerr=dim_stds,
                        marker='o', linewidth=2, markersize=10, capsize=5,
                        label=intervention.capitalize())
    
    ax2.set_xlabel('Dimension', fontsize=12)
    ax2.set_ylabel('Mean I(θ)', fontsize=12)
    ax2.set_title('I(θ) Trend Across Dimensions', fontsize=14)
    ax2.set_xticks(dimensions)
    ax2.set_xticklabels([f'{int(d)}D' for d in dimensions])
    ax2.legend(loc='best')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = output_path / 'dimension_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file.name}")
    plt.close()


def run_comparison_plots(data_dir: str, output_dir: Path):
    """Generate intervention and dimension comparison plots."""
    
    print("Loading data for comparison plots...")
    df = load_data_for_comparison(data_dir)
    
    dims = sorted(df['dimension'].dropna().unique())
    interventions = sorted(df['intervention'].unique())
    
    print(f"  Loaded {len(df)} experiments")
    print(f"  Dimensions: {[f'{int(d)}D' for d in dims]}")
    print(f"  Interventions: {interventions}")
    print()
    
    print("Generating plots...")
    plot_intervention_comparison(df, output_dir)
    plot_dimension_comparison(df, output_dir)


def run_complete_analysis(data_dir: str,
                         output_dir: str,
                         analysis_type: str = 'both',
                         param_name: str = 'auto',
                         threshold: float = 0.5,
                         run_sobol: bool = True):
    """
    Run complete analysis pipeline.
    
    Args:
        data_dir: Directory containing experimental JSON files
        output_dir: Root directory for all results
        analysis_type: 'both', 'sensitivity', or 'comparison'
        param_name: Parameter to analyze (for sensitivity)
        threshold: Inconsistency threshold (for robustness)
        run_sobol: Whether to compute Sobol indices
    """
    
    print_analysis_header("COMPLETE INCONSISTENCY ANALYSIS PIPELINE")
    
    start_time = datetime.now()
    
    # Create timestamped output directory
    timestamp = start_time.strftime('%Y%m%d_%H%M%S')
    analysis_name = analysis_type if analysis_type != 'both' else 'complete'
    output_dir_timestamped = Path(output_dir) / f"{analysis_name}_analysis_{timestamp}"
    output_path = ensure_output_dir(str(output_dir_timestamped))
    
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_path}")
    print(f"Analysis type: {analysis_type}")
    print()
    
    results_summary = {
        'data_dir': data_dir,
        'output_dir': str(output_path),
        'output_dir_name': output_path.name,
        'analysis_type': analysis_type,
        'timestamp': timestamp,
        'start_time': start_time.isoformat(),
        'completed': []
    }
    
    # Run metric comparison (validates I(θ) metric choice)
    if analysis_type in ['both', 'comparison']:
        if not COMPARISON_AVAILABLE:
            print("❌ Metric comparison not available - skipping")
        else:
            print("\n" + "="*80)
            print("PHASE 1: METRIC COMPARISON")
            print("="*80)
            print("Validating I(θ) metric against alternatives...")
            print()
            
            comparison_dir = output_path / 'metric_comparison'
            comparison_dir.mkdir(exist_ok=True)
            
            try:
                # Call the metric comparison analysis function directly
                run_metric_comparison_analysis(
                    data_dir=data_dir,
                    output_dir=str(comparison_dir)
                )
                
                results_summary['completed'].append('metric_comparison')
                print("\n✓ Metric comparison completed successfully")
                
            except Exception as e:
                print(f"\n❌ Metric comparison failed: {e}")
                import traceback
                traceback.print_exc()
    
    # Run sensitivity analysis (parameter effects and robustness)
    if analysis_type in ['both', 'sensitivity']:
        if not SENSITIVITY_AVAILABLE:
            print("❌ Sensitivity analysis not available - skipping")
        else:
            print("\n" + "="*80)
            print("PHASE 2: SENSITIVITY ANALYSIS")
            print("="*80)
            print("Computing parameter sensitivities and robustness margins...")
            print()
            
            sensitivity_dir = output_path / 'sensitivity_analysis'
            sensitivity_dir.mkdir(exist_ok=True)
            
            try:
                run_full_sensitivity_analysis(
                    data_dir=data_dir,
                    output_dir=str(sensitivity_dir),
                    param_name=param_name,
                    threshold=threshold,
                    run_sobol=run_sobol
                )
                
                results_summary['completed'].append('sensitivity_analysis')
                print("\n✓ Sensitivity analysis completed successfully")
                
            except Exception as e:
                print(f"\n❌ Sensitivity analysis failed: {e}")
                import traceback
                traceback.print_exc()
    
    # Run intervention and dimension comparison plots
    if analysis_type == 'both':
        print("\n" + "="*80)
        print("PHASE 3: INTERVENTION & DIMENSION COMPARISON")
        print("="*80)
        print("Comparing intervention effects and dimension scaling...")
        print()
        
        try:
            run_comparison_plots(data_dir, output_path)
            
            results_summary['completed'].append('comparison_plots')
            print("\n✓ Comparison plots completed successfully")
            
        except Exception as e:
            print(f"\n❌ Comparison plots failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    results_summary['end_time'] = end_time.isoformat()
    results_summary['duration_seconds'] = duration
    
    print("\n" + "="*80)
    print("ANALYSIS PIPELINE COMPLETE")
    print("="*80)
    print(f"\nEnd time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"\nCompleted analyses: {', '.join(results_summary['completed'])}")
    print(f"\nAll results saved to: {output_path.absolute()}")
    print()
    
    # Save summary
    import json
    summary_file = output_path / 'analysis_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"✓ Analysis summary saved: {summary_file.name}")
    
    return results_summary


def main():
    parser = argparse.ArgumentParser(
        description='Run complete inconsistency analysis pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run both analyses:
  python run_full_analysis.py --data_dir ../data --output_dir ../results
  
  # Run only sensitivity analysis:
  python run_full_analysis.py --data_dir ../data --output_dir ../results --only sensitivity
  
  # Run only metric comparison:
  python run_full_analysis.py --data_dir ../data --output_dir ../results --only comparison
  
  # Custom parameter analysis:
  python run_full_analysis.py --data_dir ../data --output_dir ../results \\
      --param param_alpha --threshold 0.3 --no-sobol
        """
    )
    
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing experimental JSON files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Root directory for all analysis results')
    parser.add_argument('--only', type=str, choices=['sensitivity', 'comparison'],
                       help='Run only one analysis (default: run both)')
    parser.add_argument('--param', type=str, default='auto',
                       help='Parameter name for sensitivity analysis (default: auto-detect)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Inconsistency threshold for robustness margins (default: 0.5)')
    parser.add_argument('--no-sobol', action='store_true',
                       help='Skip Sobol indices computation (faster)')
    
    args = parser.parse_args()
    
    # Determine analysis type
    if args.only:
        analysis_type = args.only
    else:
        analysis_type = 'both'
    
    # Check availability
    if analysis_type == 'sensitivity' and not SENSITIVITY_AVAILABLE:
        print("Error: Sensitivity analysis module not available")
        sys.exit(1)
    
    if analysis_type == 'comparison' and not COMPARISON_AVAILABLE:
        print("Error: Metric comparison module not available")
        sys.exit(1)
    
    if analysis_type == 'both' and not (SENSITIVITY_AVAILABLE and COMPARISON_AVAILABLE):
        print("Error: Both analysis modules must be available to run complete pipeline")
        sys.exit(1)
    
    # Run analysis
    try:
        run_complete_analysis(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            analysis_type=analysis_type,
            param_name=args.param,
            threshold=args.threshold,
            run_sobol=not args.no_sobol
        )
        
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Analysis pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
