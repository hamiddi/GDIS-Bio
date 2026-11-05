#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""P25-D: Full frozen-profile rank-order perturbation experiment."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from gdis import GDIS

from p25b_GSE175634_cell_window_reconstruction_audit import (
    ROOT, STATES, PC_COLUMNS, MANIFEST_FILE,
    load_state_space, resolve_path
)

BASE = ROOT / "results/p25_GSE175634_pseudotime_ordering_robustness"
WINDOW, STEP = 400, 100
SIGMAS = (10, 50, 100)
SEED = 250634
SCOPES = ("shared_backbone", "cm_extension", "cf_extension")


def calculate_profile(X, t, order, starts):
    X_ordered = X[order]
    t_ordered = t[order]

    windows = np.stack([
        X_ordered[s:s + WINDOW] for s in starts
    ]).astype(np.float64)

    parameters = np.array([
        np.median(t_ordered[s:s + WINDOW])
        for s in starts
    ], dtype=np.float64)

    if not np.isfinite(parameters).all():
        raise RuntimeError("Nonfinite parameters")

    if not np.all(np.diff(parameters) > 0):
        raise RuntimeError("Non-increasing window pseudotime")

    scores = GDIS().fit_transform(
        windows, parameters
    ).to_dataframe()["gdis"].to_numpy(dtype=float)

    if not np.isfinite(scores).all():
        raise RuntimeError("Nonfinite GDIS scores")

    return windows, parameters, scores


def analyze(row, state_space, replicates):
    scope = str(row["scope"])
    individual = str(row["individual"])

    group = state_space.loc[
        (state_space["individual"] == individual)
        & state_space["type"].isin(STATES[scope])
    ].sort_values(
        ["dpt_pseudotime", "cell"],
        kind="mergesort"
    ).reset_index(drop=True)

    X = group[PC_COLUMNS].to_numpy(
        dtype=np.float32, copy=True
    )

    t = group["dpt_pseudotime"].to_numpy(
        dtype=np.float64
    )

    n = len(group)
    starts = np.arange(0, n - WINDOW + 1, STEP)
    identity = np.arange(n)

    baseline_X, baseline_t, baseline_score = calculate_profile(
        X, t, identity, starts
    )

    with np.load(
        resolve_path(row["npz_path"]),
        allow_pickle=False
    ) as archive:
        if not np.array_equal(
            baseline_X, archive["trajectories"]
        ):
            raise RuntimeError("Baseline matrix mismatch")

        if not np.array_equal(
            baseline_t, archive["parameters"]
        ):
            raise RuntimeError("Baseline parameter mismatch")

    landmark = float(
        row["destination_state_median_pseudotime"]
    )

    baseline_peak = int(np.argmax(baseline_score))
    baseline_peak_t = float(baseline_t[baseline_peak])
    baseline_distance = abs(baseline_peak_t - landmark)

    records = []

    for sigma in SIGMAS:
        for rep in range(replicates):
            rng = np.random.default_rng(
                np.random.SeedSequence([
                    SEED, int(individual),
                    SCOPES.index(scope), sigma, rep
                ])
            )

            noisy = (
                np.arange(n, dtype=float)
                + rng.normal(0, sigma, n)
            )

            order = np.argsort(noisy, kind="mergesort")

            _, parameters, scores = calculate_profile(
                X, t, order, starts
            )

            new_rank = np.empty(n, dtype=np.int64)
            new_rank[order] = np.arange(n)

            displacement = np.abs(
                new_rank - identity
            )

            overlap = np.array([
                len(
                    set(identity[s:s + WINDOW])
                    & set(order[s:s + WINDOW])
                ) / WINDOW
                for s in starts
            ])

            peak = int(np.argmax(scores))
            peak_t = float(parameters[peak])
            distance = abs(peak_t - landmark)

            corr = (
                float(np.corrcoef(
                    baseline_score, scores
                )[0, 1])
                if np.std(scores) > 0
                and np.std(baseline_score) > 0
                else np.nan
            )

            records.append({
                "scope": scope,
                "individual": individual,
                "sigma_rank": sigma,
                "replicate": rep,
                "n_cells": n,
                "n_windows": len(starts),
                "mean_window_overlap": float(overlap.mean()),
                "median_rank_displacement": float(
                    np.median(displacement)
                ),
                "profile_pearson_r": corr,
                "max_parameter_change": float(
                    np.max(np.abs(parameters - baseline_t))
                ),
                "reference_peak_t": baseline_peak_t,
                "perturbed_peak_t": peak_t,
                "reference_distance": baseline_distance,
                "perturbed_distance": distance,
                "delta_distance": distance - baseline_distance,
                "absolute_peak_displacement": abs(
                    peak_t - baseline_peak_t
                ),
                "peak_window_shift": abs(
                    peak - baseline_peak
                ),
            })

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.replicates < 1 or args.limit < 0:
        parser.error("Replicates must be positive; limit nonnegative")

    manifest = pd.read_csv(
        MANIFEST_FILE, dtype={"individual": str}
    )

    if len(manifest) != 48:
        raise RuntimeError("Unexpected manifest size")

    if manifest.duplicated(["scope", "individual"]).any():
        raise RuntimeError("Duplicate manifest profiles")

    if args.limit:
        manifest = manifest.head(args.limit)

    state_space = load_state_space()

    output_dir = (
        BASE / "data"
        / f"rank_noise_reps_{args.replicates}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    completed = []

    for index, (_, row) in enumerate(manifest.iterrows(), 1):
        scope = str(row["scope"])
        individual = str(row["individual"])

        output = output_dir / f"{scope}_{individual}.csv"

        if output.exists():
            df = pd.read_csv(
                output, dtype={"individual": str}
            )
            expected = {
                (sigma, rep)
                for sigma in SIGMAS
                for rep in range(args.replicates)
            }
            observed = set(zip(
                df["sigma_rank"].astype(int),
                df["replicate"].astype(int)
            ))
            if observed != expected or len(df) != len(expected):
                raise RuntimeError(
                    f"Existing output incomplete: {output}"
                )
            print(f"[{index}/{len(manifest)}] VERIFIED {scope}/{individual}")
        else:
            df = analyze(row, state_space, args.replicates)

            # Write atomically to avoid partial CSV files.
            temporary = output.with_suffix(".tmp")
            df.to_csv(temporary, index=False)
            temporary.replace(output)

            print(f"[{index}/{len(manifest)}] PASS {scope}/{individual}")

        completed.append(df)

    combined = pd.concat(completed, ignore_index=True)

    tables = BASE / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    # Descriptive summaries only: randomized replicates are
    # NOT independent biological individuals.
    summary = combined.groupby(
        ["scope", "sigma_rank"]
    ).agg(
        n_individuals=("individual", "nunique"),
        n_runs=("replicate", "size"),
        median_profile_r=("profile_pearson_r", "median"),
        median_window_overlap=("mean_window_overlap", "median"),
        median_peak_distance=("perturbed_distance", "median"),
        median_delta_distance=("delta_distance", "median"),
        median_abs_peak_shift=("absolute_peak_displacement", "median"),
    ).reset_index()

    suffix = f"{len(manifest)}profiles_{args.replicates}reps"

    combined.to_csv(
        tables / f"02_ordering_robustness_{suffix}.csv",
        index=False
    )
    summary.to_csv(
        tables / f"03_ordering_summary_{suffix}.csv",
        index=False
    )

    print("\nP25-D COMPUTATION: COMPLETE")
    print("Profiles:", len(manifest))
    print("Perturbation runs:", len(combined))
    print("\nDescriptive summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
