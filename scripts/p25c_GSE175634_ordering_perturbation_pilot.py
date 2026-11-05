#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""P25-C: Controlled rank-perturbation pilot for GSE175634."""

from pathlib import Path
import numpy as np
import pandas as pd
from gdis import GDIS

from p25b_GSE175634_cell_window_reconstruction_audit import (
    ROOT, STATES, PC_COLUMNS, MANIFEST_FILE,
    load_state_space, resolve_path
)

OUT = (
    ROOT / "results"
    / "p25_GSE175634_pseudotime_ordering_robustness"
    / "tables"
)

SEED = 250634
SIGMAS = (10, 50, 100)
REPLICATES = 3
WINDOW = 400
STEP = 100


def main():
    manifest = pd.read_csv(
        MANIFEST_FILE,
        dtype={"individual": str}
    )

    selection = manifest.loc[
        (manifest["scope"] == "cf_extension")
        & (manifest["individual"] == "18489")
    ]

    if len(selection) != 1:
        raise RuntimeError("Pilot manifest selection invalid")

    row = selection.iloc[0]
    scope = str(row["scope"])
    individual = str(row["individual"])
    landmark = float(
        row["destination_state_median_pseudotime"]
    )

    state_space = load_state_space()

    group = state_space.loc[
        (state_space["individual"] == individual)
        & (state_space["type"].isin(STATES[scope]))
    ].sort_values(
        ["dpt_pseudotime", "cell"],
        kind="mergesort"
    ).reset_index(drop=True)

    X = group[PC_COLUMNS].to_numpy(
        dtype=np.float32, copy=True
    )
    pseudo = group["dpt_pseudotime"].to_numpy(
        dtype=np.float64
    )

    n = len(group)
    starts = np.arange(0, n - WINDOW + 1, STEP)

    with np.load(
        resolve_path(row["npz_path"]),
        allow_pickle=False
    ) as archive:
        reference_X = archive["trajectories"].copy()
        reference_t = archive["parameters"].copy()

    def build(order):
        ordered_X = X[order]
        ordered_t = pseudo[order]

        windows = np.stack(
            [ordered_X[s:s + WINDOW] for s in starts]
        ).astype(np.float64)

        parameters = np.array(
            [
                np.median(ordered_t[s:s + WINDOW])
                for s in starts
            ],
            dtype=np.float64
        )

        return windows, parameters

    identity = np.arange(n)
    baseline_X, baseline_t = build(identity)

    assert np.array_equal(baseline_X, reference_X)
    assert np.array_equal(baseline_t, reference_t)

    baseline_gdis = GDIS().fit_transform(
        baseline_X, baseline_t
    ).to_dataframe()["gdis"].to_numpy(dtype=float)

    ref_peak_index = int(np.argmax(baseline_gdis))
    ref_peak_t = float(baseline_t[ref_peak_index])
    ref_distance = abs(ref_peak_t - landmark)

    print("REFERENCE RECONSTRUCTION: PASS")
    print("Cells:", n, "Windows:", len(starts))
    print("Landmark:", landmark)
    print("Reference peak:", ref_peak_t)
    print("Reference peak distance:", ref_distance)

    results = []

    for sigma in SIGMAS:
        for replicate in range(REPLICATES):
            rng = np.random.default_rng(
                SEED + 1000 * sigma + replicate
            )

            noisy_ranks = (
                np.arange(n, dtype=float)
                + rng.normal(0, sigma, size=n)
            )

            order = np.argsort(
                noisy_ranks, kind="mergesort"
            )

            windows, parameters = build(order)

            if not np.all(np.diff(parameters) > 0):
                raise RuntimeError(
                    f"Non-increasing parameters: "
                    f"sigma={sigma}, replicate={replicate}"
                )

            # Inverse permutation gives each original cell
            # its new rank in the perturbed trajectory.
            new_rank = np.empty(n, dtype=np.int64)
            new_rank[order] = np.arange(n)

            rank_displacement = np.abs(
                new_rank - np.arange(n)
            )

            # Cell membership overlap, not just row order.
            overlaps = []
            for s in starts:
                original_cells = set(identity[s:s + WINDOW])
                new_cells = set(order[s:s + WINDOW])
                overlaps.append(
                    len(original_cells & new_cells) / WINDOW
                )

            result = GDIS().fit_transform(
                windows, parameters
            ).to_dataframe()

            gdis = result["gdis"].to_numpy(dtype=float)

            if len(gdis) != len(baseline_gdis):
                raise RuntimeError("GDIS length mismatch")

            if not np.isfinite(gdis).all():
                raise RuntimeError("Nonfinite GDIS values")

            peak_index = int(np.argmax(gdis))
            peak_t = float(parameters[peak_index])
            distance = abs(peak_t - landmark)

            if (
                np.std(gdis) > 0
                and np.std(baseline_gdis) > 0
            ):
                correlation = float(
                    np.corrcoef(
                        baseline_gdis, gdis
                    )[0, 1]
                )
            else:
                correlation = np.nan

            record = {
                "scope": scope,
                "individual": individual,
                "sigma_rank": sigma,
                "replicate": replicate,
                "n_cells": n,
                "n_windows": len(starts),
                "median_abs_rank_displacement":
                    float(np.median(rank_displacement)),
                "mean_window_membership_overlap":
                    float(np.mean(overlaps)),
                "max_abs_parameter_change":
                    float(np.max(np.abs(
                        parameters - baseline_t
                    ))),
                "profile_pearson_r": correlation,
                "reference_peak_t": ref_peak_t,
                "perturbed_peak_t": peak_t,
                "reference_distance": ref_distance,
                "perturbed_distance": distance,
                "delta_distance": distance - ref_distance,
                "peak_window_displacement":
                    abs(peak_index - ref_peak_index),
            }

            results.append(record)

            print(
                f"sigma={sigma:3d} "
                f"rep={replicate} "
                f"overlap={record['mean_window_membership_overlap']:.4f} "
                f"r={correlation:.4f} "
                f"distance={distance:.6f} "
                f"delta={record['delta_distance']:+.6f}"
            )

    OUT.mkdir(parents=True, exist_ok=True)

    output = OUT / "01_ordering_perturbation_pilot.csv"
    pd.DataFrame(results).to_csv(output, index=False)

    print()
    print("P25-C PILOT: COMPLETE")
    print("Output:", output)


if __name__ == "__main__":
    main()
