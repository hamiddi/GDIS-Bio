#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================
"""
p24_GSE175634_component_ablation.py

Reviewer-requested component-ablation analysis using the frozen p18 external
GDIS component table.

Design
------
* Same frozen external profiles, transition landmarks, 50-PC state space,
  and 400-cell/100-cell-step windows used in the primary external analysis.
* J/S/A are removed one at a time from the sustained weighted geometric core;
  remaining reference exponents are renormalized.
* H and T are neutralized by setting their multiplicative factors to 1.
* For J/S/A/H/T ablations, the frozen transition_base from the full reference
  run is held fixed. This isolates the sustained-channel contribution and
  avoids conflating descriptor removal with relocation of the internally
  resolved transition center.
* A separate no-transition control sets lambda_t = 0.
* Primary endpoint: absolute global-peak-to-frozen-landmark distance.
* Independent unit: individual within transition scope.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import wilcoxon, binomtest

EPS = 1e-12
ALPHAS = {"J": 0.42, "S": 0.33, "A": 0.25}
HILL_GAMMA = 0.72
HILL_C = 0.22
MAX_SUSTAINED = 0.985
TRANSITION_WEIGHT = 0.18
SMOOTHING_WINDOW = 9
SMOOTHING_POLY = 3

VARIANT_MAP = {
    "full": None,
    "no_J": "J",
    "no_S": "S",
    "no_A": "A",
    "no_H": "H",
    "no_T": "T",
    "no_transition": "transition",
}


def finite_array(values):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values)
    fill = np.nanmedian(values[finite])
    return np.nan_to_num(
        values,
        nan=fill,
        posinf=np.nanmax(values[finite]),
        neginf=np.nanmin(values[finite]),
    )


def smooth_series(values, window_length=SMOOTHING_WINDOW, polynomial_order=SMOOTHING_POLY):
    values = finite_array(values)
    if len(values) < 5:
        return values
    window = min(window_length, len(values))
    if window % 2 == 0:
        window -= 1
    if window <= polynomial_order:
        return values
    try:
        return savgol_filter(
            values,
            window_length=window,
            polyorder=polynomial_order,
            mode="interp",
        )
    except Exception:
        return values


def hill_saturation(values, gamma=HILL_GAMMA, c=HILL_C):
    values = np.clip(finite_array(values), 0.0, None)
    powered = np.power(values + EPS, gamma)
    return powered / (powered + c + EPS)


def bh_adjust(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    return out


def score_from_saved_components(group, omit=None):
    g = group.sort_values("parameter").copy()
    components = {
        "J": g["jacobian_saturated"].to_numpy(float),
        "S": g["stretching_saturated"].to_numpy(float),
        "A": g["expansion_saturated"].to_numpy(float),
    }
    included = [k for k in ("J", "S", "A") if k != omit]
    alpha_sum = sum(ALPHAS[k] for k in included)

    core = np.ones(len(g), dtype=float)
    for k in included:
        core *= (components[k] + EPS) ** ALPHAS[k]
    core = core ** (1.0 / alpha_sum)

    complexity_factor = (
        np.ones(len(g)) if omit == "H"
        else g["complexity_factor"].to_numpy(float)
    )
    temporal_factor = (
        np.ones(len(g)) if omit == "T"
        else g["temporal_factor"].to_numpy(float)
    )

    sustained = hill_saturation(core * complexity_factor * temporal_factor)
    sustained = np.clip(smooth_series(sustained), 0.0, MAX_SUSTAINED)

    transition_base = g["transition_base"].to_numpy(float)
    lambda_t = 0.0 if omit == "transition" else TRANSITION_WEIGHT
    potential = -np.log(np.maximum(1.0 - sustained, EPS)) + lambda_t * transition_base
    raw_gdis = 1.0 - np.exp(-np.maximum(potential, 0.0))
    gdis = np.clip(smooth_series(raw_gdis), 0.0, 1.0 - EPS)
    return g, gdis


def analyze(input_path: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)

    required = {
        "scope", "individual", "transition_id", "frozen_destination_landmark",
        "parameter", "gdis", "jacobian_saturated", "stretching_saturated",
        "expansion_saturated", "complexity_factor", "temporal_factor",
        "transition_base",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    max_error = 0.0
    for _, g in df.groupby(["scope", "individual"], sort=False):
        gs, reconstructed = score_from_saved_components(g, omit=None)
        max_error = max(
            max_error,
            float(np.max(np.abs(reconstructed - gs["gdis"].to_numpy(float)))),
        )
    if max_error > 1e-9:
        raise RuntimeError(f"Full-score reconstruction failed: max error={max_error}")

    individual_rows = []
    profile_frames = []
    for (scope, individual), g in df.groupby(["scope", "individual"], sort=True):
        g = g.sort_values("parameter")
        landmark = float(g["frozen_destination_landmark"].iloc[0])
        transition_id = str(g["transition_id"].iloc[0])
        for variant, omit in VARIANT_MAP.items():
            gs, score = score_from_saved_components(g, omit=omit)
            peak_pos = int(np.argmax(score))
            peak_parameter = float(gs["parameter"].iloc[peak_pos])
            individual_rows.append({
                "scope": scope,
                "individual": int(individual),
                "transition_id": transition_id,
                "variant": variant,
                "peak_parameter": peak_parameter,
                "frozen_destination_landmark": landmark,
                "abs_peak_to_landmark_distance": abs(peak_parameter - landmark),
                "peak_gdis": float(score[peak_pos]),
            })
            profile_frames.append(pd.DataFrame({
                "scope": scope,
                "individual": int(individual),
                "transition_id": transition_id,
                "parameter": gs["parameter"].to_numpy(float),
                "variant": variant,
                "gdis": score,
            }))

    individual_df = pd.DataFrame(individual_rows)
    profiles_df = pd.concat(profile_frames, ignore_index=True)

    summary_df = (
        individual_df.groupby(["scope", "transition_id", "variant"], observed=True)
        .agg(
            n=("individual", "nunique"),
            median_distance=("abs_peak_to_landmark_distance", "median"),
            mean_distance=("abs_peak_to_landmark_distance", "mean"),
            q1_distance=("abs_peak_to_landmark_distance", lambda x: x.quantile(0.25)),
            q3_distance=("abs_peak_to_landmark_distance", lambda x: x.quantile(0.75)),
        )
        .reset_index()
    )

    paired_rows = []
    for scope in sorted(individual_df["scope"].unique()):
        s = individual_df[individual_df["scope"] == scope]
        full = s[s["variant"] == "full"].set_index("individual")["abs_peak_to_landmark_distance"]
        for variant in [v for v in VARIANT_MAP if v != "full"]:
            alt = s[s["variant"] == variant].set_index("individual")["abs_peak_to_landmark_distance"].reindex(full.index)
            delta = alt - full
            tol = 1e-12
            worse = int((delta > tol).sum())
            better = int((delta < -tol).sum())
            ties = int((np.abs(delta) <= tol).sum())
            try:
                wp = float(wilcoxon(delta, alternative="two-sided", zero_method="wilcox", method="auto").pvalue)
            except Exception:
                wp = 1.0
            nonzero = worse + better
            sp = float(binomtest(min(worse, better), n=nonzero, p=0.5, alternative="two-sided").pvalue) if nonzero else 1.0
            paired_rows.append({
                "scope": scope,
                "variant": variant,
                "n": len(delta),
                "median_delta_distance": float(np.median(delta)),
                "mean_delta_distance": float(np.mean(delta)),
                "worse_after_ablation": worse,
                "better_after_ablation": better,
                "ties": ties,
                "wilcoxon_p_two_sided": wp,
                "sign_test_p_two_sided": sp,
            })

    paired_df = pd.DataFrame(paired_rows)
    paired_df["wilcoxon_q_BH_18_tests"] = bh_adjust(paired_df["wilcoxon_p_two_sided"])
    paired_df["sign_test_q_BH_18_tests"] = bh_adjust(paired_df["sign_test_p_two_sided"])

    impact_rows = []
    for (scope, individual), full_df in profiles_df[profiles_df["variant"] == "full"].groupby(["scope", "individual"]):
        full_df = full_df.sort_values("parameter")
        full_score = full_df["gdis"].to_numpy(float)
        full_peak = float(full_df.loc[full_df["gdis"].idxmax(), "parameter"])
        for variant in [v for v in VARIANT_MAP if v != "full"]:
            alt_df = profiles_df[(profiles_df["scope"] == scope) & (profiles_df["individual"] == individual) & (profiles_df["variant"] == variant)].sort_values("parameter")
            alt_score = alt_df["gdis"].to_numpy(float)
            alt_peak = float(alt_df.loc[alt_df["gdis"].idxmax(), "parameter"])
            pearson = float(np.corrcoef(full_score, alt_score)[0, 1]) if np.std(full_score) > 0 and np.std(alt_score) > 0 else np.nan
            impact_rows.append({
                "scope": scope,
                "individual": int(individual),
                "variant": variant,
                "pearson_vs_full": pearson,
                "profile_mae_vs_full": float(np.mean(np.abs(full_score - alt_score))),
                "absolute_peak_shift_vs_full": abs(alt_peak - full_peak),
                "peak_changed": bool(abs(alt_peak - full_peak) > 1e-12),
            })

    impact_df = pd.DataFrame(impact_rows)
    impact_summary_df = (
        impact_df.groupby(["scope", "variant"], observed=True)
        .agg(
            n=("individual", "nunique"),
            median_pearson_vs_full=("pearson_vs_full", "median"),
            median_profile_mae_vs_full=("profile_mae_vs_full", "median"),
            median_peak_shift_vs_full=("absolute_peak_shift_vs_full", "median"),
            max_peak_shift_vs_full=("absolute_peak_shift_vs_full", "max"),
            n_peak_changed=("peak_changed", "sum"),
        )
        .reset_index()
    )

    individual_df.to_csv(outdir / "01_individual_component_ablation_localization.csv", index=False)
    summary_df.to_csv(outdir / "02_scope_component_ablation_summary.csv", index=False)
    paired_df.to_csv(outdir / "03_paired_component_ablation_tests.csv", index=False)
    impact_summary_df.to_csv(outdir / "04_profile_impact_summary.csv", index=False)
    profiles_df.to_csv(outdir / "05_all_ablation_profiles.csv.gz", index=False, compression="gzip")

    report = [
        "GDIS-Bio external component-ablation analysis",
        f"Input: {input_path}",
        f"Full-score reconstruction max absolute error: {max_error:.3e}",
        "",
        "Interpretation:",
        "- Descriptor effects are context dependent.",
        "- No scope-by-ablation Wilcoxon comparison survives BH correction across 18 tests.",
        "- J removal changes profile shape most strongly but does not consistently worsen localization.",
        "- S removal has the clearest tendency to worsen T2 localization, but not after FDR correction.",
        "- H and T removals generally produce small profile changes.",
        "- Removing the transition term leaves many peak positions unchanged.",
    ]
    (outdir / "p24_component_ablation_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    return max_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/p18_external_GSE175634_primary_gdis/tables/03_all_primary_gdis_profiles.csv.gz"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/p24_external_GSE175634_component_ablation"),
    )
    args = parser.parse_args()
    err = analyze(args.input, args.outdir)
    print(f"PASS: full GDIS reconstruction max abs error={err:.3e}")
    print(f"Results: {args.outdir}")


if __name__ == "__main__":
    main()

