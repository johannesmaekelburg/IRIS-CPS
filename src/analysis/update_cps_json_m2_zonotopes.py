#!/usr/bin/env python3
"""
update_cps_json_m2_zonotopes.py
================================
Re-parameterise source (M1) and target (M2) zonotopes in every
CPS-extra-domains JSON file from the OLD 0.25-generator rule to the
NEW rule that yields P_dim ≈ 0.707 per dimension, P_2D ≈ 0.5.

Old rule:  source_c = (a+b)/2  (interval mid-point)
           target_c = thr − 0.25·g,  target_G = 0.25·g   (≤ case)

New rule:  source_c = thr − 0.414·g                       (≤ case)
           target_c = thr − 1.0·g,   target_G = g

For ≥ constraints the signs are reversed:
  source_c = thr + 0.414·g
  target_c = thr + 1.0·g

Threshold is back-calculated from the current M2:
  thr = M2_center + M2_G   (for ≤)
  thr = M2_center − M2_G   (for ≥)

Run from project root:
    python src/analysis/update_cps_json_m2_zonotopes.py
"""

import json
from pathlib import Path

SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
BASE_DIR     = PROJECT_ROOT / "data" / "CPS-extra-domains"

SKIP_FILES = {"cps_uncertainty_schema_v2.json", "nl_descriptions.json", "nl_descriptions.md"}

DELTA = 0.414  # (1 + delta)/2 = 0.707  →  P_dim ≈ 0.707,  P_2D ≈ 0.5


# ──────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _fmt(v: float, d: int = 4) -> str:
    s = f"{v:.{d}f}"
    s = s.rstrip("0")
    if s.endswith("."):
        s += "0"
    return s


# ──────────────────────────────────────────────────────────────────────────────
# Scenario-level update
# ──────────────────────────────────────────────────────────────────────────────

def _process_scenario(scenario: dict) -> bool:
    """Update M1 centre and M2 centre/G/formula in-place. Returns True if updated."""
    models = scenario.get("models", [])
    m1_model = next((m for m in models if m.get("id") == "M1"), None)
    m2_model = next((m for m in models if m.get("id") == "M2"), None)
    if not m1_model or not m2_model:
        return False

    m1_elem = next((e for e in m1_model.get("elements", []) if e.get("id") == "e1"), None)
    m2_elem = next((e for e in m2_model.get("elements", []) if e.get("id") == "e2"), None)
    if not m1_elem or not m2_elem:
        return False

    m1_zono = m1_elem.get("uncertainty", {}).get("zonotope", {})
    m2_zono = m2_elem.get("uncertainty", {}).get("zonotope", {})

    # Only handle real constrained_zonotopes in both M1 and M2
    if (m1_zono.get("type") != "constrained_zonotope"
            or m1_zono.get("domain") != "real"
            or m2_zono.get("type") != "constrained_zonotope"
            or m2_zono.get("domain") != "real"):
        return False

    try:
        g         = float(m1_zono["G"][0])
        old_m2_c  = float(m2_zono["center"])
        old_m2_g  = float(m2_zono["G"][0])
    except (KeyError, IndexError, TypeError, ValueError):
        return False

    # Determine constraint direction
    crs      = scenario.get("consistency_relations", [{}])
    operator = crs[0].get("operator", "<=") if crs else "<="
    is_leq   = "<" in operator or operator.lower() in ("leq", "subset_of", "<=")

    # Back-calculate threshold and compute new parameters
    if is_leq:
        threshold  = old_m2_c + old_m2_g
        new_src_c  = threshold - DELTA * g
        new_tgt_c  = threshold - g
    else:
        threshold  = old_m2_c - old_m2_g
        new_src_c  = threshold + DELTA * g
        new_tgt_c  = threshold + g
    new_tgt_g = g

    # ── Update M1 (source) ──────────────────────────────────────────────────
    m1_zono["center"] = round(new_src_c, 6)
    m1_zono["formula"] = f"CZ = \u27e8{_fmt(new_src_c)}, [{_fmt(g)}], 0, 0\u27e9"
    m1_zono["set_description"] = (
        f"{{{_fmt(new_src_c)} + {_fmt(g)}\u00b7\u03be | \u03be \u2208 [-1,1]}}"
    )
    m1_zono["matlab"] = f"zonotope([{_fmt(new_src_c)}; {_fmt(g)}])"

    # ── Update M2 (target) ──────────────────────────────────────────────────
    minus_sign = "\u2212" if is_leq else "+"   # − or +
    m2_zono["center"] = round(new_tgt_c, 6)
    m2_zono["G"]      = [round(new_tgt_g, 6)]
    m2_zono["formula"] = (
        f"CZ = \u27e8{_fmt(new_tgt_c)}, [{_fmt(new_tgt_g)}], 0, 0\u27e9"
    )
    m2_zono["set_description"] = (
        f"{{{_fmt(new_tgt_c)} + {_fmt(new_tgt_g)}\u00b7\u03be | \u03be \u2208 [-1,1]}}"
    )
    m2_zono["interval"] = [round(new_tgt_c - new_tgt_g, 6), round(new_tgt_c + new_tgt_g, 6)]
    m2_zono["matlab"]   = f"zonotope([{_fmt(new_tgt_c)}; {_fmt(new_tgt_g)}])"
    m2_zono["derivation"] = (
        f"Re-parameterised for P_2D \u2248 0.5 overlap target. "
        f"Threshold = {_fmt(threshold)}, generator g = {_fmt(g)}. "
        f"Target: c = threshold {minus_sign} g = {_fmt(new_tgt_c)}, G = g = {_fmt(new_tgt_g)}. "
        f"Source: c = threshold {minus_sign} {DELTA}\u00b7g = {_fmt(new_src_c)}. "
        f"P_dim = (1 + {DELTA})/2 = 0.707 per dimension, P_2D = 0.707\u00b2 \u2248 0.50."
    )

    # ── Update consistency_check_result (if present) ────────────────────────
    for cr in crs:
        ccr = cr.get("consistency_check_result")
        if not isinstance(ccr, dict):
            continue
        lo = round(new_src_c - g, 4)
        hi = round(new_src_c + g, 4)
        if is_leq:
            margin = round(threshold - hi, 4)
            ccr["margin"] = margin
            ccr["note"] = (
                f"CZ = [{lo}, {hi}], threshold = {_fmt(threshold)}. "
                f"Upper margin = {margin:.4f} (negative \u2192 upper end exceeds threshold, "
                f"partial inconsistency)."
            )
        else:
            margin = round(lo - threshold, 4)
            ccr["margin"] = margin
            ccr["note"] = (
                f"CZ = [{lo}, {hi}], threshold = {_fmt(threshold)}. "
                f"Lower margin = {margin:.4f} (negative \u2192 lower end falls below threshold, "
                f"partial inconsistency)."
            )

    return True


# ──────────────────────────────────────────────────────────────────────────────
# File-level processing
# ──────────────────────────────────────────────────────────────────────────────

def _process_file(filepath: Path) -> int:
    with open(filepath, encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        return 0

    n_updated = sum(_process_scenario(s) for s in data)

    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    return n_updated


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not BASE_DIR.exists():
        print(f"[ERROR] Directory not found: {BASE_DIR}")
        return

    total = 0
    for json_file in sorted(BASE_DIR.glob("*.json")):
        if json_file.name in SKIP_FILES:
            continue
        n = _process_file(json_file)
        print(f"  {json_file.name:<45}  updated {n} scenario(s)")
        total += n

    print(f"\n  Total scenarios updated: {total}")


if __name__ == "__main__":
    main()
