#!/usr/bin/env python3
"""
Generate Saltelli Samples for Compound Intervention Experiments

This script generates Saltelli sampling matrices for joint sensitivity analysis
of multiple intervention parameters simultaneously. Output is consumed by MATLAB
to run compound intervention experiments.

Usage:
    python generate_saltelli_samples.py --n_samples 512 --output saltelli_samples.csv
"""

import argparse
import json
import numpy as np
from pathlib import Path

try:
    from SALib.sample import saltelli
    SALIB_AVAILABLE = True
except ImportError:
    SALIB_AVAILABLE = False
    print("⚠️  SALib not available — install with: pip install SALib")


def generate_3param_samples(
    N: int,
    output_dir: Path,
    output_format: str = 'csv',
    profile: str = 'generic'
):
    """
    Generate Saltelli samples for 3 intervention parameters:
    - scale_factor: Scaling intervention (widen: >1, shrink: <1)
    - center_delta: Spatial shift magnitude
    - correlation_strength: Correlation between generators
    
    Args:
        N: Base sample size (total samples = N*(2p+2) = 8*N for p=3)
        output_dir: Output directory for samples
        output_format: 'csv', 'json', or 'both'
        profile: Parameter profile ('generic' or 'formula_student')
    """
    if not SALIB_AVAILABLE:
        raise RuntimeError("SALib is required for Saltelli sampling")
    
    # Define parameter space
    base_problem = {
        'num_vars': 3,
        'names': ['scale_factor', 'center_delta', 'correlation_strength'],
    }

    if profile == 'formula_student':
        # Formula Student profile: narrower ranges for realistic engineering variation
        problem = {
            **base_problem,
            'bounds': [
                [0.7, 3.0],   # scale_factor: moderate shrink/widen range
                [0.0, 0.2],   # center_delta: up to 20% center shift
                [0.0, 0.85],  # correlation_strength: practical upper correlation
            ],
            'profile': 'formula_student'
        }
    else:
        # Generic profile: broad exploration bounds (backward-compatible)
        problem = {
            **base_problem,
            'bounds': [
                [0.5, 5.0],   # scale_factor: 0.5 (shrink 50%) to 5.0 (widen 500%)
                [0.0, 0.3],   # center_delta: 0 to 30% of radius
                [0.0, 0.95]   # correlation_strength: independent to highly correlated
            ],
            'profile': 'generic'
        }
    
    print(f"\n{'='*80}")
    print("GENERATING SALTELLI SAMPLES")
    print(f"{'='*80}")
    print(f"Profile: {problem['profile']}")
    print(f"Parameters: {problem['names']}")
    print(f"Bounds: {problem['bounds']}")
    print(f"Base sample size N = {N}")
    print(f"Total samples = N(2p+2) = {N * (2*problem['num_vars'] + 2)}")
    
    # Generate Saltelli samples
    samples = saltelli.sample(problem, N=N, calc_second_order=False)
    
    print(f"\n✓ Generated {samples.shape[0]} samples × {samples.shape[1]} parameters")

    suffix = '' if profile == 'generic' else f'_{profile}'
    
    # Save to CSV
    if output_format in ['csv', 'both']:
        csv_path = output_dir / f'saltelli_samples_3param{suffix}.csv'
        header = ','.join(problem['names'])
        np.savetxt(csv_path, samples, delimiter=',', header=header, comments='')
        print(f"✓ Saved CSV: {csv_path}")
    
    # Save to JSON (with metadata)
    if output_format in ['json', 'both']:
        json_path = output_dir / f'saltelli_samples_3param{suffix}.json'
        data = {
            'problem': problem,
            'N': N,
            'total_samples': samples.shape[0],
            'samples': samples.tolist(),
            'matlab_instructions': {
                'usage': 'Load samples and run compound interventions',
                'example': f'samples = readmatrix("saltelli_samples_3param{suffix}.csv"); for i=1:size(samples,1); theta = samples(i,:); ...'
            }
        }
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Saved JSON: {json_path}")
    
    # Statistics
    print(f"\n{'='*40}")
    print("SAMPLE STATISTICS")
    print(f"{'='*40}")
    for i, name in enumerate(problem['names']):
        print(f"{name:25s}: [{samples[:,i].min():.4f}, {samples[:,i].max():.4f}] (mean={samples[:,i].mean():.4f})")
    
    return samples, problem


def generate_7param_samples(N: int, output_dir: Path, output_format: str = 'csv'):
    """
    Generate Saltelli samples for 7 parameters (intervention + geometry):
    - 3 intervention parameters (as above)
    - 4 geometric parameters: source volume, target volume, source radius, target radius
    
    Note: This requires MATLAB to synthesize scenarios with prescribed geometry,
    which is more complex than simple interventions.
    
    Args:
        N: Base sample size (total samples = N*(2p+2) = 18*N for p=7)
        output_dir: Output directory for samples
        output_format: 'csv', 'json', or 'both'
    """
    if not SALIB_AVAILABLE:
        raise RuntimeError("SALib is required for Saltelli sampling")
    
    # Define parameter space
    problem = {
        'num_vars': 7,
        'names': [
            'scale_factor', 'center_delta', 'correlation_strength',
            'source_volume', 'target_volume', 'source_radius', 'target_radius'
        ],
        'bounds': [
            [0.5, 5.0],      # scale_factor
            [0.0, 0.3],      # center_delta
            [0.0, 0.95],     # correlation_strength
            [10.0, 50.0],    # source_volume
            [10.0, 50.0],    # target_volume
            [2.0, 6.0],      # source_radius
            [2.0, 6.0]       # target_radius
        ]
    }
    
    print(f"\n{'='*80}")
    print("GENERATING SALTELLI SAMPLES (7 PARAMETERS)")
    print(f"{'='*80}")
    print(f"Parameters: {problem['names']}")
    print(f"Base sample size N = {N}")
    print(f"Total samples = N(2p+2) = {N * (2*problem['num_vars'] + 2)}")
    print(f"\n⚠️  WARNING: 7-parameter sampling requires MATLAB to synthesize")
    print(f"   scenarios with prescribed geometric properties. This requires")
    print(f"   implementing create_scenario_with_geometry() function.")
    
    # Generate Saltelli samples
    samples = saltelli.sample(problem, N=N, calc_second_order=False)
    
    print(f"\n✓ Generated {samples.shape[0]} samples × {samples.shape[1]} parameters")
    
    # Save files
    if output_format in ['csv', 'both']:
        csv_path = output_dir / 'saltelli_samples_7param.csv'
        header = ','.join(problem['names'])
        np.savetxt(csv_path, samples, delimiter=',', header=header, comments='')
        print(f"✓ Saved CSV: {csv_path}")
    
    if output_format in ['json', 'both']:
        json_path = output_dir / 'saltelli_samples_7param.json'
        data = {
            'problem': problem,
            'N': N,
            'total_samples': samples.shape[0],
            'samples': samples.tolist()
        }
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Saved JSON: {json_path}")
    
    return samples, problem


def analyze_saltelli_results(results_path: Path, problem_path: Path):
    """
    Analyze results from MATLAB Saltelli experiments.
    
    Expects results JSON with:
    [
        {"sample_index": 0, "theta": [1.2, 0.05, 0.7], "I_theta": 0.45, ...},
        ...
    ]
    
    Args:
        results_path: Path to JSON with experimental results
        problem_path: Path to JSON with problem definition (from generation step)
    """
    if not SALIB_AVAILABLE:
        raise RuntimeError("SALib is required for Sobol analysis")
    
    from SALib.analyze import sobol
    
    # Load problem definition
    with open(problem_path, 'r') as f:
        data = json.load(f)
        problem = data['problem']
    
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Sort by sample index and extract I(θ)
    results_sorted = sorted(results, key=lambda r: r['sample_index'])
    Y = np.array([r['I_theta'] for r in results_sorted])
    
    print(f"\n{'='*80}")
    print("SOBOL SENSITIVITY ANALYSIS (Direct from Saltelli Samples)")
    print(f"{'='*80}")
    print(f"Total samples: {len(Y)}")
    print(f"I(θ) range: [{Y.min():.4f}, {Y.max():.4f}]")
    print(f"I(θ) mean ± std: {Y.mean():.4f} ± {Y.std():.4f}")
    
    # Compute Sobol indices
    Si = sobol.analyze(problem, Y, calc_second_order=False)
    
    # Display results
    print(f"\n{'='*80}")
    print("FIRST-ORDER INDICES (S_i)")
    print(f"{'='*80}")
    for i, name in enumerate(problem['names']):
        s = Si['S1'][i]
        s_conf = Si['S1_conf'][i]
        print(f"{name:25s}: {s:7.4f} ± {s_conf:7.4f}")
    
    print(f"\n{'='*80}")
    print("TOTAL-EFFECT INDICES (S_Ti)")
    print(f"{'='*80}")
    for i, name in enumerate(problem['names']):
        st = Si['ST'][i]
        st_conf = Si['ST_conf'][i]
        print(f"{name:25s}: {st:7.4f} ± {st_conf:7.4f}")
    
    # Save results
    output_path = results_path.parent / 'sobol_indices_saltelli.json'
    output = {
        'problem': problem,
        'first_order': {name: {'S1': float(Si['S1'][i]), 'S1_conf': float(Si['S1_conf'][i])}
                       for i, name in enumerate(problem['names'])},
        'total_effect': {name: {'ST': float(Si['ST'][i]), 'ST_conf': float(Si['ST_conf'][i])}
                        for i, name in enumerate(problem['names'])},
        'Y_stats': {
            'mean': float(Y.mean()),
            'std': float(Y.std()),
            'min': float(Y.min()),
            'max': float(Y.max())
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Saved Sobol indices: {output_path}")
    
    return Si


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate Saltelli samples for compound intervention experiments'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Generate samples
    gen_parser = subparsers.add_parser('generate', help='Generate Saltelli samples')
    gen_parser.add_argument('--n_samples', type=int, default=512,
                           help='Base sample size N (total = N*(2p+2))')
    gen_parser.add_argument('--params', type=str, choices=['3', '7'], default='3',
                           help='Number of parameters (3=interventions only, 7=+geometry)')
    gen_parser.add_argument('--profile', type=str, choices=['generic', 'formula_student'],
                           default='generic',
                           help='3-parameter bounds profile (generic or formula_student)')
    gen_parser.add_argument('--output_dir', type=str, default='../data',
                           help='Output directory')
    gen_parser.add_argument('--format', type=str, choices=['csv', 'json', 'both'], 
                           default='both', help='Output format')
    
    # Analyze results
    analyze_parser = subparsers.add_parser('analyze', help='Analyze Saltelli results')
    analyze_parser.add_argument('--results', type=str, required=True,
                               help='Path to results JSON from MATLAB')
    analyze_parser.add_argument('--problem', type=str, required=True,
                               help='Path to problem definition JSON')
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        if args.params == '3':
            generate_3param_samples(args.n_samples, output_dir, args.format, args.profile)
        else:
            generate_7param_samples(args.n_samples, output_dir, args.format)
    
    elif args.command == 'analyze':
        analyze_saltelli_results(Path(args.results), Path(args.problem))
    
    else:
        parser.print_help()
