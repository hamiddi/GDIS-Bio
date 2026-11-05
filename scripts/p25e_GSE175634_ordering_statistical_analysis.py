#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""P25-E: Individual-level analysis of ordering perturbations."""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

TABLES = (
    ROOT / "results"
    / "p25_GSE175634_pseudotime_ordering_robustness"
    / "tables"
)

SOURCE = TABLES / "02_ordering_robustness_48profiles_10reps.csv"

SEED = 250635
N_BOOT = 10000


def bootstrap_median_ci(values, seed):
    values = np.asarray(values, dtype=float)

    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Insufficient or invalid observations")

    rng = np.random.default_rng(seed)

    indices = rng.integers(
        0, len(values),
        size=(N_BOOT, len(values))
    )

    boot = np.median(values[indices], axis=1)

    return np.quantile(boot, [0.025, 0.975])


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)

    df = pd.read_csv(
        SOURCE, dtype={"individual": str}
    )

    if len(df) != 1440:
        raise RuntimeError("Expected 1440 perturbation runs")

    key = ["scope", "individual", "sigma_rank", "replicate"]

    if df.duplicated(key).any():
        raise RuntimeError("Duplicate perturbation records")

    if not np.isfinite(
        df[[
            "profile_pearson_r",
            "delta_distance",
            "absolute_peak_displacement",
            "mean_window_overlap",
        ]].to_numpy(dtype=float)
    ).all():
        raise RuntimeError("Nonfinite analysis endpoints")

    expected_counts = {
        "shared_backbone": 19,
        "cm_extension": 15,
        "cf_extension": 14,
    }

    for scope, n in expected_counts.items():
        for sigma in [10, 50, 100]:
            subset = df[
                (df["scope"] == scope)
                & (df["sigma_rank"] == sigma)
            ]

            if len(subset) != n * 10:
                raise RuntimeError(
                    f"Unexpected run count: {scope}/{sigma}"
                )

            if not (subset.groupby("individual").size() == 10).all():
                raise RuntimeError(
                    f"Incomplete replicates: {scope}/{sigma}"
                )

    df["peak_changed"] = (
        df["peak_window_shift"] > 0
    ).astype(float)

    individual = df.groupby(
        ["scope", "individual", "sigma_rank"],
        as_index=False
    ).agg(
        median_profile_r=("profile_pearson_r", "median"),
        median_delta_distance=("delta_distance", "median"),
        median_abs_delta_distance=(
            "delta_distance",
            lambda x: float(np.median(np.abs(x)))
        ),
        median_abs_peak_shift=(
            "absolute_peak_displacement", "median"
        ),
        median_window_overlap=(
            "mean_window_overlap", "median"
        ),
        peak_change_fraction=("peak_changed", "mean"),
        median_reference_distance=("reference_distance", "median"),
        median_perturbed_distance=("perturbed_distance", "median"),
    )

    individual.to_csv(
        TABLES / "04_individual_level_ordering_summary.csv",
        index=False
    )

    metrics = [
        "median_profile_r",
        "median_delta_distance",
        "median_abs_delta_distance",
        "median_abs_peak_shift",
        "median_window_overlap",
        "peak_change_fraction",
    ]

    rows = []

    for group_index, ((scope, sigma), group) in enumerate(
        individual.groupby(["scope", "sigma_rank"])
    ):
        for metric_index, metric in enumerate(metrics):
            values = group[metric].to_numpy(dtype=float)

            low, high = bootstrap_median_ci(
                values,
                SEED + group_index * 100 + metric_index
            )

            rows.append({
                "scope": scope,
                "sigma_rank": sigma,
                "metric": metric,
                "n_individuals": len(values),
                "individual_level_median": float(
                    np.median(values)
                ),
                "bootstrap_ci_lower": float(low),
                "bootstrap_ci_upper": float(high),
            })

    summary = pd.DataFrame(rows)

    summary.to_csv(
        TABLES / "05_individual_bootstrap_ci.csv",
        index=False
    )

    print("P25-E STATISTICAL ANALYSIS: COMPLETE")
    print("Individual-scope observations:", len(individual))
    print("Bootstrap replicates:", N_BOOT)
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
