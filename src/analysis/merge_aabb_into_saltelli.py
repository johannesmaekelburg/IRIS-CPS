#!/usr/bin/env python3
"""
Merge AABB Jaccard values into existing Saltelli MC-probability JSON files.

Usage:
    python merge_aabb_into_saltelli.py \
        --mc-dir   data/measurements/measurements \
        --aabb-dir data/measurements_aabb_only \
        --output   data/measurements_merged

The AABB JSONs were produced by running Step 2 with method='jaccard' only.
The MC JSONs were produced by running Step 2 with method='mc_probability' only.
The merged JSONs contain both, ready for MFMC post-processing.
"""

import json
import argparse
from pathlib import Path


AABB_FIELDS = [
    'jaccard_C', 'jaccard_Csym', 'jaccard_index',
    'empty_intersection', 'has_intersection',
    'jaccard_vol_intersection', 'jaccard_vol_union', 'intersection_volume',
]


def load_json(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: dict, path: Path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def build_exp_index(experiments: list, key: str = 'exp_id') -> dict:
    """Build lookup: exp_id -> experiment dict."""
    return {exp[key]: exp for exp in experiments if key in exp}


def merge_scenario(mc_path: Path, aabb_path: Path, output_path: Path):
    mc_data   = load_json(mc_path)
    aabb_data = load_json(aabb_path)

    aabb_index = build_exp_index(aabb_data.get('experiments', []))

    n_merged = 0
    n_missing = 0

    for exp in mc_data.get('experiments', []):
        eid = exp.get('exp_id')
        if eid not in aabb_index:
            n_missing += 1
            continue

        aabb_exp = aabb_index[eid]
        aabb_inc = aabb_exp.get('post_state', {}).get('inconsistency', {})
        mc_inc   = exp.get('post_state', {}).get('inconsistency', {})

        # Copy AABB fields into MC inconsistency block
        for field in AABB_FIELDS:
            if field in aabb_inc:
                mc_inc[field] = aabb_inc[field]

        # Also copy pre_state AABB fields if present
        aabb_pre_inc = aabb_exp.get('pre_state', {}).get('inconsistency', {})
        mc_pre_inc   = exp.get('pre_state', {}).get('inconsistency', {})
        for field in AABB_FIELDS:
            if field in aabb_pre_inc:
                mc_pre_inc[field] = aabb_pre_inc[field]

        n_merged += 1

    mc_data['merged_aabb'] = True
    mc_data['aabb_source'] = str(aabb_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(mc_data, output_path)

    return n_merged, n_missing


def main():
    parser = argparse.ArgumentParser(description='Merge AABB fields into Saltelli MC JSON files')
    parser.add_argument('--mc-dir',   required=True, help='Directory with MC-only JSONs')
    parser.add_argument('--aabb-dir', required=True, help='Directory with AABB-only JSONs')
    parser.add_argument('--output',   required=True, help='Output directory for merged JSONs')
    parser.add_argument('--scenario', nargs='*', type=int, default=None,
                        help='Scenario IDs to process (default: all)')
    args = parser.parse_args()

    mc_dir   = Path(args.mc_dir)
    aabb_dir = Path(args.aabb_dir)
    out_dir  = Path(args.output)

    mc_files = sorted(mc_dir.glob('results_scenario_*.json'))
    if args.scenario:
        mc_files = [f for f in mc_files
                    if any(f.name == f'results_scenario_{sid}.json' for sid in args.scenario)]

    print(f"Merging {len(mc_files)} scenario file(s)...\n")

    for mc_path in mc_files:
        aabb_path = aabb_dir / mc_path.name
        if not aabb_path.exists():
            print(f"  [SKIP] No AABB file for {mc_path.name}")
            continue

        out_path = out_dir / mc_path.name
        n_merged, n_missing = merge_scenario(mc_path, aabb_path, out_path)
        print(f"  {mc_path.name}: merged {n_merged} experiments"
              + (f", {n_missing} missing in AABB file" if n_missing else ""))

    print(f"\nDone. Merged files in: {out_dir}")
    print("\nNext step — run MFMC on merged results:")
    print(f"  python src/mfmc_correction.py --input {out_dir} --output results/mfmc")


if __name__ == '__main__':
    main()
