#!/usr/bin/env python3
"""P26: Reviewer 1 comment 6, dependence-aware *sensitivity* audit.

Read-only for P7/P10/P11. Writes exclusively to results/p26_comment6_blocked_null_audit.
Shared fractional phase shifts approximate branch coordination but do NOT perfectly
preserve shared-cell identity or all cross-branch correlations; this is not an
exact cluster-randomization test or mechanistic instability test.
"""
import argparse
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd

METRICS = ["gdis", "transition_energy", "total_variance",
           "mean_lag1_autocorrelation", "gaussian_differential_entropy",
           "mean_step_distance"]
BASELINES = [m for m in METRICS if m not in ("gdis", "transition_energy")]
GROUPS = [(b, d) for d in (1, 2) for b in (0, 1)]


def bh(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    q = np.empty(len(p), dtype=float)
    q[order] = np.minimum.accumulate((p[order] * len(p) /
                    np.arange(1, len(p)+1))[::-1])[::-1].clip(0, 1)
    return q


def exact_signflip(d):
    d = np.asarray(d, float)
    observed = d.mean()
    null = np.array([np.mean(d * np.asarray(s))
                     for s in product((-1., 1.), repeat=len(d))])
    # Exact enumeration: do not use a Monte Carlo +1 correction.
    return float(np.mean(null >= observed - 1e-12)), len(null)


def locate_landmark(root):
    expected = "07_primary_state_landmark_window_mapping.csv"
    hits = list((root / "results").rglob(expected))
    if len(hits) != 1:
        raise RuntimeError(f"Expected exactly one {expected}, found {len(hits)}: {hits}")
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--n-permutations", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261103)
    args = ap.parse_args()
    root = args.root.resolve()
    p10 = root / "results/p10_conventional_ews_benchmark/tables/01_all_metric_profiles.csv"
    p7 = locate_landmark(root)
    p11 = root / "results/p11_statistical_alignment_null_validation/tables/03_group_localization_errors.csv"
    for f in (p10, p7, p11):
        if not f.exists():
            raise FileNotFoundError(f)
    prof = pd.read_csv(p10)
    land = pd.read_csv(p7)
    archived = pd.read_csv(p11)
    for col in ["branch", "differentiation", "parameter", "dominant_state"] + METRICS:
        if col not in prof:
            raise ValueError(f"Missing profile column: {col}")
    for col in ["branch", "differentiation", "state", "state_median_pseudotime"]:
        if col not in land:
            raise ValueError(f"Missing landmark column: {col}")
    groups = {}
    for branch, diff in GROUPS:
        df = (prof[(prof.branch == branch) & (prof.differentiation == diff)]
              .sort_values("parameter", kind="stable").reset_index(drop=True))
        if len(df) < 3:
            raise ValueError(f"Too few windows for {(branch, diff)}")
        matches = land[(land.branch == branch) & (land.differentiation == diff) &
                       (land.state == "neurog3_early")]
        if len(matches) != 1:
            raise ValueError(f"Landmark mismatch in {(branch,diff)}: {len(matches)}")
        x = df.parameter.to_numpy(float)
        landmark = float(matches.state_median_pseudotime.iloc[0])
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite pseudotime")
        groups[branch, diff] = dict(x=x, target=landmark, df=df, n=len(df))

    # Recompute observed values and compare against the frozen P11 source.
    per_group = []
    for (branch, diff), g in groups.items():
        for metric in METRICS:
            vals = g["df"][metric].to_numpy(float)
            if not np.isfinite(vals).all():
                raise ValueError(f"Nonfinite values in {metric}, {(branch, diff)}")
            peak = float(g["x"][int(np.argmax(vals))])
            error = abs(peak - g["target"])
            reference = archived[(archived.metric == metric) &
                                 (archived.branch == branch) &
                                 (archived.differentiation == diff)]
            if len(reference) != 1 or not np.isclose(
                error, float(reference.absolute_localization_error.iloc[0]), atol=1e-7, rtol=0
            ):
                raise RuntimeError(f"Archived P11 mismatch: {metric}, {(branch,diff)}, {error}")
            per_group.append(dict(metric=metric, differentiation=diff, branch=branch,
                                  observed_absolute_localization_error=error))
    raw = pd.DataFrame(per_group)

    # Independent two-experiment inference, branches averaged *within* differentiation.
    paired = []
    gdis = raw[raw.metric == "gdis"].groupby("differentiation")[
        "observed_absolute_localization_error"].mean().sort_index()
    for metric in BASELINES + ["transition_energy"]:
        baseline = raw[raw.metric == metric].groupby("differentiation")[
            "observed_absolute_localization_error"].mean().sort_index()
        diffs = baseline.to_numpy() - gdis.to_numpy()
        p, n = exact_signflip(diffs)
        paired.append(dict(comparator=metric, mean_baseline_minus_gdis=float(diffs.mean()),
                           differentiation_1_difference=float(diffs[0]),
                           differentiation_2_difference=float(diffs[1]),
                           n_independent_differentiations=2, exact_one_sided_p=p,
                           n_exact_sign_flips=n))
    paired_df = pd.DataFrame(paired)

    # One fraction per differentiation across BOTH branches; different fractions
    # across the two differentiations. Map fraction to each branch's window count.
    # Exclude the identity shift in each profile, as in original P11.
    rng = np.random.default_rng(args.seed)
    nperm = args.n_permutations
    if nperm < 100:
        raise ValueError("Use >= 100 null draws")
    # Cache shifts so comparisons of different metrics use identical draws.
    u = rng.random((nperm, 2))
    keys = list(GROUPS)
    shifts = {key: 1 + np.floor(u[:, key[1] - 1] * (groups[key]["n"] - 1)).astype(int)
              for key in keys}
    null_metrics = {}
    observed_metric = {}
    for metric in METRICS:
        observed_metric[metric] = float(raw[raw.metric == metric][
            "observed_absolute_localization_error"].mean())
        arr = np.zeros(nperm, float)
        # With unique global peak, a roll by s relocates peak index (i+s) % n.
        # np.argmax keeps first occurrence for ties. To preserve exact original
        # argmax tie behavior, roll the array and argmax for every draw when tied.
        for key in keys:
            g = groups[key]
            y = g["df"][metric].to_numpy(float)
            if np.count_nonzero(y == y.max()) == 1:
                ipeak = int(np.argmax(y))
                ix = (ipeak + shifts[key]) % g["n"]
            else:
                ix = np.array([int(np.argmax(np.roll(y, s))) for s in shifts[key]], int)
            arr += np.abs(g["x"][ix] - g["target"]) / len(keys)
        null_metrics[metric] = arr

    rows = []
    for metric in METRICS:
        observed = observed_metric[metric]
        null = null_metrics[metric]
        p = (1 + np.count_nonzero(null <= observed + 1e-12)) / (nperm + 1)
        rows.append(dict(metric=metric, observed_mean_absolute_localization_error=observed,
                         null_mean=float(null.mean()), null_q025=float(np.quantile(null, .025)),
                         null_q975=float(np.quantile(null, .975)),
                         p_lower_blocked_fractional_shift=float(p), n_permutations=nperm))
    summary = pd.DataFrame(rows)
    summary["q_bh_6_metrics"] = bh(summary.p_lower_blocked_fractional_shift)
    out = root / "results/p26_comment6_blocked_null_audit"
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "01_verified_group_errors.csv", index=False)
    paired_df.to_csv(out / "02_differentiation_level_paired_tests.csv", index=False)
    summary.to_csv(out / "03_coordinated_fractional_shift_null.csv", index=False)
    pd.DataFrame(null_metrics).to_csv(out / "04_null_draws.csv", index=False)
    with (out / "README_LIMITATIONS.txt").open("w") as f:
        f.write("P26 supplementary sensitivity analysis. Does NOT overwrite archived P11.\n"
                "Within each differentiation the two branches share a circular-shift fraction,\n"
                "mapped onto each branch's number of windows; identity shifts excluded.\n"
                "This approximately coordinates branch perturbations but DOES NOT exactly\n"
                "preserve shared-cell correspondence or branch correlation. Not a rigorous\n"
                "biological cluster-randomization test, not proof of mechanistic instability.\n"
                "Circular wrapping creates artificial endpoints, as in P11.\n"
                "MC p: (1 + #null <= observed)/(N+1), BH for six metrics.\n"
                "Two-experiment exact paired tests use four sign flips (min p=0.25\n"
                "for two strictly positive experiment differences).\n")
    print("\n=== COORDINATED FRACTIONAL SHIFT SENSITIVITY ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("\n=== EXACT TWO-DIFFERENTIATION PAIRED TESTS ===")
    print(paired_df.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("\nVerified all 24 metric-by-trajectory frozen P11 errors.")
    print("OUTPUT:", out)

if __name__ == "__main__":
    main()

