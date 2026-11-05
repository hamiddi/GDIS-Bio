#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""P25-F9: Paired, scope-specific reconstruction sensitivity."""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

BASE = (
    ROOT / "results"
    / "p25_GSE175634_pseudotime_ordering_robustness"
)

SOURCE = (
    BASE / "geodesic_gdis_comparison"
    / "01_geodesic_gdis_full.csv"
)

OUT = BASE / "tables"
N_BOOT = 10000
SEED = 250639

EXPECTED = {
    "shared_backbone": 19,
    "cm_extension": 15,
    "cf_extension": 14,
}


def bootstrap_ci(values, statistic, seed):
    values = np.asarray(values, dtype=float)

    if len(values) < 2 or not np.isfinite(values).all():
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    indices = rng.integers(
        0, len(values),
        size=(N_BOOT, len(values)),
    )

    estimates = statistic(values[indices], axis=1)

    return tuple(
        np.quantile(estimates, [0.025, 0.975])
    )


def main():
    df = pd.read_csv(
        SOURCE, dtype={"individual": str}
    )

    if len(df) != 48:
        raise RuntimeError("Expected 48 profiles")

    if not (df["status"] == "PASS").all():
        raise RuntimeError("Failed profiles present")

    if df.duplicated(["scope", "individual"]).any():
        raise RuntimeError("Duplicate individual/scope records")

    for scope, count in EXPECTED.items():
        if (df["scope"] == scope).sum() != count:
            raise RuntimeError(f"Invalid scope count: {scope}")

    required = [
        "profile_pearson_r",
        "reference_distance",
        "geodesic_distance",
        "delta_distance",
        "absolute_peak_rank_shift",
    ]

    if not np.isfinite(
        df[required].to_numpy(dtype=float)
    ).all():
        raise RuntimeError("Nonfinite outcome values")

    if not np.allclose(
        df["delta_distance"],
        df["geodesic_distance"] - df["reference_distance"],
        atol=1e-12,
        rtol=0,
    ):
        raise RuntimeError("Paired difference mismatch")

    def parse_bool(series):
        values = series.astype(str).str.lower()
        if not values.isin(["true", "false"]).all():
            raise RuntimeError("Invalid boolean values")
        return values.eq("true").to_numpy(dtype=bool)

    for column in [
        "qualified",
        "reference_within_iqr",
        "geodesic_within_iqr",
    ]:
        df[column] = parse_bool(df[column])

    df["abs_delta_distance"] = df[
        "delta_distance"
    ].abs()

    df["iqr_change"] = (
        df["geodesic_within_iqr"].astype(int)
        - df["reference_within_iqr"].astype(int)
    )

    df["improved"] = df["delta_distance"] < -1e-12
    df["worsened"] = df["delta_distance"] > 1e-12

    rows = []
    counter = 0

    for scope in EXPECTED:
        subset = df[df["scope"] == scope]

        groups = [
            ("all", subset),
            ("qualified", subset[subset["qualified"]]),
            ("nonqualified", subset[~subset["qualified"]]),
        ]

        for category, group in groups:
            if group.empty:
                continue

            counter += 1

            delta = group[
                "delta_distance"
            ].to_numpy(dtype=float)

            abs_delta = np.abs(delta)

            ci_delta = bootstrap_ci(
                delta, np.median,
                SEED + counter * 10
            )

            ci_abs = bootstrap_ci(
                abs_delta, np.median,
                SEED + counter * 10 + 1
            )

            iqr_change = group[
                "iqr_change"
            ].to_numpy(dtype=float)

            ci_iqr = bootstrap_ci(
                iqr_change, np.mean,
                SEED + counter * 10 + 2
            )

            rows.append({
                "scope": scope,
                "qualification": category,
                "n_individuals": len(group),
                "median_profile_r": group[
                    "profile_pearson_r"
                ].median(),
                "median_window_overlap": group[
                    "mean_window_overlap"
                ].median(),
                "median_reference_distance": group[
                    "reference_distance"
                ].median(),
                "median_geodesic_distance": group[
                    "geodesic_distance"
                ].median(),
                "median_paired_delta": np.median(delta),
                "delta_ci_low": ci_delta[0],
                "delta_ci_high": ci_delta[1],
                "median_abs_delta": np.median(abs_delta),
                "abs_delta_ci_low": ci_abs[0],
                "abs_delta_ci_high": ci_abs[1],
                "median_abs_peak_shift": group[
                    "absolute_peak_rank_shift"
                ].median(),
                "fraction_improved": group[
                    "improved"
                ].mean(),
                "fraction_worsened": group[
                    "worsened"
                ].mean(),
                "reference_iqr_fraction": group[
                    "reference_within_iqr"
                ].mean(),
                "geodesic_iqr_fraction": group[
                    "geodesic_within_iqr"
                ].mean(),
                "paired_iqr_fraction_change":
                    iqr_change.mean(),
                "iqr_change_ci_low": ci_iqr[0],
                "iqr_change_ci_high": ci_iqr[1],
            })

    summary = pd.DataFrame(rows)

    OUT.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        OUT / "08_geodesic_paired_individual_results.csv",
        index=False,
    )

    summary.to_csv(
        OUT / "09_geodesic_paired_scope_statistics.csv",
        index=False,
    )

    columns = [
        "scope", "qualification", "n_individuals",
        "median_profile_r", "median_paired_delta",
        "delta_ci_low", "delta_ci_high",
        "median_abs_delta", "median_abs_peak_shift",
        "fraction_improved", "fraction_worsened",
        "reference_iqr_fraction", "geodesic_iqr_fraction",
    ]

    print("\nP25-F9 PAIRED STATISTICAL ANALYSIS: PASS")
    print("Bootstrap resamples:", N_BOOT)
    print()
    print(summary[columns].to_string(index=False))
    print("\nOutput:", OUT)


if __name__ == "__main__":
    main()
