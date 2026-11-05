#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""P25-F8: Frozen-cohort geodesic vs deposited-ordering GDIS."""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from gdis import GDIS
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.stats import spearmanr

from p25b_GSE175634_cell_window_reconstruction_audit import (
    ROOT, STATES, PC_COLUMNS, MANIFEST_FILE, load_state_space
)

BASE = ROOT / (
    "results/p25_GSE175634_pseudotime_ordering_robustness"
)
AUDIT = BASE / "alternative_reconstruction_cohort"
OUT = BASE / "geodesic_gdis_comparison"

N_NEIGHBORS = 30
SEED = 250636
WINDOW = 400
STEP = 100


def score_order(X, types, order, destination):
    n = len(order)
    ranks = np.arange(n, dtype=float) / (n - 1)
    starts = np.arange(0, n - WINDOW + 1, STEP)

    Xo = X[order]
    to = types[order]

    windows = np.stack([
        Xo[s:s + WINDOW] for s in starts
    ]).astype(np.float64)

    parameters = np.array([
        np.median(ranks[s:s + WINDOW])
        for s in starts
    ])

    if not np.all(np.diff(parameters) > 0):
        raise RuntimeError("Non-increasing parameters")

    result = GDIS().fit_transform(
        windows, parameters
    ).to_dataframe()

    scores = result["gdis"].to_numpy(dtype=float)

    if not np.isfinite(scores).all():
        raise RuntimeError("Nonfinite GDIS scores")

    destination_ranks = ranks[to == destination]

    if len(destination_ranks) == 0:
        raise RuntimeError("No destination cells")

    median = float(np.median(destination_ranks))
    q25, q75 = np.quantile(
        destination_ranks, [0.25, 0.75]
    )

    peak_index = int(np.argmax(scores))
    peak_rank = float(parameters[peak_index])

    return {
        "scores": scores,
        "peak_index": peak_index,
        "peak_rank": peak_rank,
        "landmark": median,
        "q25": float(q25),
        "q75": float(q75),
        "distance": abs(peak_rank - median),
        "within_iqr": bool(q25 <= peak_rank <= q75),
        "n_windows": len(starts),
    }


def analyze(row, source, qualification):
    scope = str(row["scope"])
    individual = str(row["individual"])

    group = source.loc[
        (source["individual"] == individual)
        & source["type"].isin(STATES[scope])
    ].sort_values(
        ["dpt_pseudotime", "cell"],
        kind="mergesort",
    ).reset_index(drop=True)

    X = group[PC_COLUMNS].to_numpy(
        dtype=np.float32, copy=True
    )
    types = group["type"].to_numpy(dtype=str)
    n = len(group)

    destination = str(row["destination_state"])

    if destination not in STATES[scope]:
        raise RuntimeError("Invalid destination state")

    roots = np.flatnonzero(types == "IPSC")

    centroid = X[roots].mean(
        axis=0, dtype=np.float64
    )

    root = int(roots[np.argmin(
        np.linalg.norm(
            X[roots].astype(np.float64) - centroid,
            axis=1
        )
    )])

    adata = ad.AnnData(X=X)
    adata.obsm["X_pca"] = X.copy()

    sc.pp.neighbors(
        adata,
        n_neighbors=N_NEIGHBORS,
        use_rep="X_pca",
        method=None,  # Only the distance graph is used; avoid Gaussian connectivity compatibility issues.
        random_state=SEED,
    )

    graph = csr_matrix(adata.obsp["distances"])
    graph = graph.maximum(graph.T).tocsr()

    components, labels = connected_components(
        graph, directed=False
    )

    if components != 1:
        raise RuntimeError("Disconnected graph")

    distances = np.asarray(
        dijkstra(graph, directed=False, indices=root),
        dtype=float
    )

    if not np.isfinite(distances).all():
        raise RuntimeError("Nonfinite geodesic distances")

    if np.ptp(distances) <= 0:
        raise RuntimeError("Degenerate geodesic distances")

    # Use the exact cell-rank ordering rules from the pilot.
    geodesic = (
        pd.Series(distances)
        .rank(method="average")
        .to_numpy(dtype=float)
    )
    geodesic = (geodesic - 1) / (n - 1)

    reference_order = np.arange(n)

    geodesic_order = np.lexsort((
        group["cell"].to_numpy(dtype=str),
        geodesic
    ))

    reference = score_order(
        X, types, reference_order, destination
    )

    alternative = score_order(
        X, types, geodesic_order, destination
    )

    starts = np.arange(0, n - WINDOW + 1, STEP)

    overlap = np.mean([
        len(
            set(reference_order[s:s + WINDOW])
            & set(geodesic_order[s:s + WINDOW])
        ) / WINDOW
        for s in starts
    ])

    if (
        np.std(reference["scores"]) <= 0
        or np.std(alternative["scores"]) <= 0
    ):
        raise RuntimeError("Degenerate GDIS profile")

    correlation = float(np.corrcoef(
        reference["scores"],
        alternative["scores"]
    )[0, 1])

    state_medians = np.array([
        np.median(geodesic[types == state])
        for state in STATES[scope]
    ])

    strict = bool(np.all(np.diff(state_medians) > 0))

    expected_strict = bool(qualification["qualified"])

    if strict != expected_strict:
        raise RuntimeError(
            "Reconstructed state qualification disagrees "
            "with frozen P25-F6 audit"
        )

    return {
        "scope": scope,
        "individual": individual,
        "qualified": strict,
        "n_cells": n,
        "n_windows": reference["n_windows"],
        "spearman_original_vs_geodesic": float(
            spearmanr(
                group["dpt_pseudotime"].to_numpy(),
                geodesic
            ).statistic
        ),
        "mean_window_overlap": float(overlap),
        "profile_pearson_r": correlation,
        "reference_peak_rank": reference["peak_rank"],
        "geodesic_peak_rank": alternative["peak_rank"],
        "absolute_peak_rank_shift": abs(
            reference["peak_rank"]
            - alternative["peak_rank"]
        ),
        "reference_landmark_rank": reference["landmark"],
        "geodesic_landmark_rank": alternative["landmark"],
        "reference_distance": reference["distance"],
        "geodesic_distance": alternative["distance"],
        "delta_distance": (
            alternative["distance"]
            - reference["distance"]
        ),
        "reference_within_iqr": reference["within_iqr"],
        "geodesic_within_iqr": alternative["within_iqr"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be nonnegative")

    manifest = pd.read_csv(
        MANIFEST_FILE, dtype={"individual": str}
    )

    qualification = pd.read_csv(
        AUDIT / "01_geodesic_qualification_full.csv",
        dtype={"individual": str}
    )

    if len(manifest) != 48 or len(qualification) != 48:
        raise RuntimeError("Frozen cohort count mismatch")

    key = ["scope", "individual"]

    if manifest.duplicated(key).any() or qualification.duplicated(key).any():
        raise RuntimeError("Duplicate cohort keys")

    merged = manifest.merge(
        qualification[key + ["qualified"]],
        on=key,
        how="left",
        validate="one_to_one"
    )

    if merged["qualified"].isna().any():
        raise RuntimeError("Missing qualification records")

    if args.limit:
        merged = merged.head(args.limit)

    state_space = load_state_space()
    OUT.mkdir(parents=True, exist_ok=True)

    records = []

    for i, (_, row) in enumerate(merged.iterrows(), 1):
        scope = str(row["scope"])
        individual = str(row["individual"])

        try:
            result = analyze(
                row, state_space,
                {"qualified": row["qualified"]}
            )
            result["status"] = "PASS"
            result["failure_reason"] = ""
        except Exception as exc:
            result = {
                "scope": scope,
                "individual": individual,
                "qualified": bool(row["qualified"]),
                "status": "FAIL",
                "failure_reason":
                    f"{type(exc).__name__}: {exc}"
            }

        records.append(result)

        # Checkpoint results after each profile.
        df = pd.DataFrame(records)
        suffix = "pilot" if args.limit else "full"
        target = OUT / f"01_geodesic_gdis_{suffix}.csv"
        temp = target.with_suffix(".tmp")
        df.to_csv(temp, index=False)
        temp.replace(target)

        print(
            f"[{i}/{len(merged)}] "
            f"{scope}/{individual} "
            f"{result['status']}",
            flush=True
        )

    df = pd.DataFrame(records)

    print("\nP25-F8 COMPARISON COMPLETE")
    print("Profiles:", len(df))
    print("Passed:", int((df["status"] == "PASS").sum()))
    print("Failed:", int((df["status"] == "FAIL").sum()))

    successful = df[df["status"] == "PASS"]

    if len(successful):
        print("\nBy qualification:")
        print(
            successful.groupby("qualified").agg(
                n=("individual", "size"),
                median_profile_r=("profile_pearson_r", "median"),
                median_reference_distance=(
                    "reference_distance", "median"
                ),
                median_geodesic_distance=(
                    "geodesic_distance", "median"
                ),
                median_delta_distance=(
                    "delta_distance", "median"
                ),
                geodesic_iqr_fraction=(
                    "geodesic_within_iqr", "mean"
                ),
            ).to_string()
        )

    if (df["status"] == "FAIL").any():
        raise SystemExit(
            "Some profiles failed; inspect failure_reason"
        )


if __name__ == "__main__":
    main()
