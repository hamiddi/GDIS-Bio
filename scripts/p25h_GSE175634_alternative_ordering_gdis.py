#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
P25-F5: Exploratory GDIS sensitivity to alternative orderings.

All methods use the same cells and frozen 50-PC coordinates.
GDIS is recomputed using a common normalized rank parameter.

Results do not replace frozen P18 reference calculations.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from gdis import GDIS

from p25b_GSE175634_cell_window_reconstruction_audit import (
    ROOT, PC_COLUMNS, STATES, load_state_space
)

SCOPE = "cf_extension"
INDIVIDUAL = "18489"
DESTINATION = "CF"
WINDOW = 400
STEP = 100

BASE = (
    ROOT / "results"
    / "p25_GSE175634_pseudotime_ordering_robustness"
)

SOURCE = BASE / "alternative_pseudotime_pilot"
OUT = BASE / "tables"

METHODS = {
    "deposited_dpt": "original_pseudotime",
    "scanpy_dpt": "alternative_pseudotime",
    "graph_geodesic": "geodesic_pseudotime",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    state_space = load_state_space()

    group = state_space.loc[
        (state_space["individual"] == INDIVIDUAL)
        & state_space["type"].isin(STATES[SCOPE])
    ].copy()

    original = pd.read_csv(
        SOURCE / "02_original_vs_alternative_pseudotime.csv",
        dtype={"cell": str},
    )

    geodesic = pd.read_csv(
        SOURCE / "08_graph_geodesic_cell_pseudotime.csv",
        dtype={"cell": str},
    )

    if (
        group["cell"].duplicated().any()
        or original["cell"].duplicated().any()
        or geodesic["cell"].duplicated().any()
    ):
        raise RuntimeError("Duplicate cell identifiers")

    if not (
        len(group) == len(original) == len(geodesic)
    ):
        raise RuntimeError("Cell population size mismatch")

    merged = group.merge(
        original[["cell", "alternative_pseudotime"]],
        on="cell",
        validate="one_to_one",
    ).merge(
        geodesic[["cell", "geodesic_pseudotime"]],
        on="cell",
        validate="one_to_one",
    )

    if len(merged) != len(group):
        raise RuntimeError("Cell population mismatch")

    merged["original_pseudotime"] = merged[
        "dpt_pseudotime"
    ]

    if not np.isfinite(
        merged[list(METHODS.values())].to_numpy(dtype=float)
    ).all():
        raise RuntimeError("Nonfinite pseudotime values")

    results = []
    profiles = []
    orders = {}
    parameters_by_method = {}

    n = len(merged)
    starts = np.arange(0, n - WINDOW + 1, STEP)
    rank_parameter = np.arange(n, dtype=np.float64) / (n - 1)

    for method, column in METHODS.items():
        ordered = merged.sort_values(
            [column, "cell"],
            kind="mergesort",
        ).reset_index(drop=True)

        orders[method] = ordered["cell"].to_numpy()

        X = ordered[PC_COLUMNS].to_numpy(
            dtype=np.float32, copy=True
        )

        windows = np.stack([
            X[s:s + WINDOW] for s in starts
        ]).astype(np.float64)

        parameters = np.asarray([
            np.median(rank_parameter[s:s + WINDOW])
            for s in starts
        ], dtype=np.float64)

        if not np.all(np.diff(parameters) > 0):
            raise RuntimeError("Invalid rank parameters")

        parameters_by_method[method] = parameters

        calculated = GDIS().fit_transform(
            windows, parameters
        ).to_dataframe()

        scores = calculated["gdis"].to_numpy(dtype=float)

        if not np.isfinite(scores).all():
            raise RuntimeError(f"Nonfinite scores: {method}")

        destination_ranks = rank_parameter[
            ordered["type"].to_numpy() == DESTINATION
        ]

        landmark = float(np.median(destination_ranks))
        q25, q75 = np.quantile(
            destination_ranks, [0.25, 0.75]
        )

        peak_index = int(np.argmax(scores))
        peak = float(parameters[peak_index])

        results.append({
            "method": method,
            "scope": SCOPE,
            "individual": INDIVIDUAL,
            "n_cells": n,
            "n_windows": len(starts),
            "destination_state": DESTINATION,
            "landmark_median_rank": landmark,
            "landmark_q25_rank": float(q25),
            "landmark_q75_rank": float(q75),
            "peak_window": peak_index,
            "peak_rank": peak,
            "peak_to_landmark_distance": abs(
                peak - landmark
            ),
            "peak_within_destination_iqr": bool(
                q25 <= peak <= q75
            ),
        })

        profiles.append(pd.DataFrame({
            "method": method,
            "window_index": np.arange(len(scores)),
            "parameter_rank": parameters,
            "gdis": scores,
        }))

    # Profile comparison on corresponding rank windows.
    all_profiles = pd.concat(profiles, ignore_index=True)

    reference_scores = all_profiles.loc[
        all_profiles["method"] == "deposited_dpt",
        "gdis",
    ].to_numpy()

    reference_order = orders["deposited_dpt"]

    for result in results:
        method = result["method"]

        scores = all_profiles.loc[
            all_profiles["method"] == method,
            "gdis",
        ].to_numpy()

        if method == "deposited_dpt":
            correlation = 1.0
        elif (
            np.std(reference_scores) > 0
            and np.std(scores) > 0
        ):
            correlation = float(
                np.corrcoef(reference_scores, scores)[0, 1]
            )
        else:
            correlation = np.nan

        alternative_order = orders[method]

        overlaps = []
        for start in starts:
            a = set(reference_order[start:start + WINDOW])
            b = set(alternative_order[start:start + WINDOW])
            overlaps.append(len(a & b) / WINDOW)

        result["profile_pearson_r_vs_deposited"] = correlation
        result["mean_window_membership_overlap"] = float(
            np.mean(overlaps)
        )

    summary = pd.DataFrame(results)

    summary.to_csv(
        OUT / "06_alternative_ordering_gdis_pilot.csv",
        index=False,
    )

    all_profiles.to_csv(
        OUT / "07_alternative_ordering_profiles.csv",
        index=False,
    )

    print("\nP25-F5 EXPLORATORY GDIS COMPARISON")
    print(summary.to_string(index=False))
    print("\nP25-F5: COMPLETE")


if __name__ == "__main__":
    main()
