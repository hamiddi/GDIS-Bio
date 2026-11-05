#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""P25-F6: Cohort-wide graph-geodesic reconstruction qualification."""

import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

from p25b_GSE175634_cell_window_reconstruction_audit import (
    ROOT,
    STATES,
    PC_COLUMNS,
    MANIFEST_FILE,
    load_state_space,
)


OUT = (
    ROOT
    / "results"
    / "p25_GSE175634_pseudotime_ordering_robustness"
    / "alternative_reconstruction_cohort"
)

DAY_MAP = {
    "day0": 0,
    "day1": 1,
    "day3": 3,
    "day5": 5,
    "day7": 7,
    "day11": 11,
    "day15": 15,
}

N_NEIGHBORS = 30
SEED = 250636


def build_knn_distance_graph(X, n_neighbors=N_NEIGHBORS):
    """
    Construct a symmetric k-nearest-neighbor distance graph directly
    from the frozen PCA state space.

    This deliberately avoids scanpy.pp.neighbors(), because P25-F6
    requires only the distance graph for graph-geodesic shortest paths;
    Scanpy connectivity construction is not required.

    Parameters
    ----------
    X : ndarray, shape (n_cells, n_features)
        Frozen PCA coordinates.
    n_neighbors : int
        Requested number of nearest neighbors.

    Returns
    -------
    scipy.sparse.csr_matrix
        Symmetric sparse distance graph.
    """

    n_cells = X.shape[0]

    if n_cells < 2:
        raise RuntimeError(
            f"Need at least 2 cells to construct a neighbor graph; "
            f"found {n_cells}"
        )

    # Cannot request more non-self neighbors than n_cells - 1.
    k = min(int(n_neighbors), n_cells - 1)

    if k < 1:
        raise RuntimeError("Effective neighbor count is < 1")

    # Request k + 1 because the first neighbor is normally the point itself.
    nn = NearestNeighbors(
        n_neighbors=k + 1,
        algorithm="auto",
        metric="euclidean",
        n_jobs=-1,
    )
    nn.fit(X)

    distances, indices = nn.kneighbors(
        X,
        n_neighbors=k + 1,
        return_distance=True,
    )

    rows = []
    cols = []
    data = []

    tiny = np.finfo(np.float64).eps

    for i in range(n_cells):
        added = 0

        for dist, j in zip(distances[i], indices[i]):
            j = int(j)

            # Exclude self-neighbor.
            if j == i:
                continue

            # Preserve duplicate-point edges. Sparse matrices normally
            # discard explicit zero values, which could incorrectly make
            # identical points appear disconnected.
            d = max(float(dist), tiny)

            rows.append(i)
            cols.append(j)
            data.append(d)

            added += 1

            if added == k:
                break

        if added != k:
            raise RuntimeError(
                f"Only found {added} non-self neighbors for cell {i}; "
                f"expected {k}"
            )

    graph = coo_matrix(
        (data, (rows, cols)),
        shape=(n_cells, n_cells),
        dtype=np.float64,
    ).tocsr()

    # Convert directed kNN relation into an undirected graph.
    #
    # For Euclidean distance, reciprocal edges have the same weight.
    # maximum() also retains an edge that occurs in only one direction.
    graph = graph.maximum(graph.T)
    graph = csr_matrix(graph)

    graph.setdiag(0.0)
    graph.eliminate_zeros()
    graph.sort_indices()

    return graph


def safe_spearman(x, y):
    """
    Return a finite Spearman correlation or NaN if it cannot be estimated.
    """

    result = spearmanr(x, y)

    # scipy versions may expose either .statistic or a tuple-like result.
    rho = getattr(result, "statistic", result[0])

    if rho is None or not np.isfinite(rho):
        return np.nan

    return float(rho)


def reconstruct(row, state_space):
    scope = str(row["scope"])
    individual = str(row["individual"])

    if scope not in STATES:
        raise RuntimeError(f"Unknown scope: {scope}")

    expected_states = list(STATES[scope])

    group = (
        state_space.loc[
            (state_space["individual"].astype(str) == individual)
            & state_space["type"].isin(expected_states)
        ]
        .sort_values(
            ["dpt_pseudotime", "cell"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    if group.empty:
        raise RuntimeError(
            f"No cells found for {scope}/{individual}"
        )

    missing_pc = [
        col for col in PC_COLUMNS
        if col not in group.columns
    ]

    if missing_pc:
        raise RuntimeError(
            f"Missing PCA columns: {missing_pc[:5]}"
        )

    X = group[PC_COLUMNS].to_numpy(
        dtype=np.float64,
        copy=True,
    )

    if not np.isfinite(X).all():
        raise RuntimeError(
            "Non-finite values detected in frozen PCA state space"
        )

    types = group["type"].astype(str).to_numpy()

    # Confirm every prespecified state is represented.
    missing_states = [
        state
        for state in expected_states
        if not np.any(types == state)
    ]

    if missing_states:
        raise RuntimeError(
            f"Missing expected states: {missing_states}"
        )

    roots = np.flatnonzero(types == "IPSC")

    if len(roots) == 0:
        raise RuntimeError("No IPSC root candidates")

    # Frozen root definition:
    # IPSC cell closest to the IPSC centroid in the 50-PC state space.
    centroid = X[roots].mean(
        axis=0,
        dtype=np.float64,
    )

    root_distances = np.linalg.norm(
        X[roots] - centroid,
        axis=1,
    )

    root = int(
        roots[np.argmin(root_distances)]
    )

    # ------------------------------------------------------------
    # Construct the graph directly.
    # No Scanpy connectivity matrix is required for geodesic paths.
    # ------------------------------------------------------------
    graph = build_knn_distance_graph(
        X,
        n_neighbors=N_NEIGHBORS,
    )

    if graph.shape != (len(group), len(group)):
        raise RuntimeError(
            f"Unexpected graph shape {graph.shape}; "
            f"expected {(len(group), len(group))}"
        )

    if graph.nnz == 0:
        raise RuntimeError(
            "Neighbor graph contains no edges"
        )

    n_components, labels = connected_components(
        graph,
        directed=False,
        return_labels=True,
    )

    root_component = labels[root]

    record = {
        "scope": scope,
        "individual": individual,
        "n_cells": int(len(group)),
        "n_neighbors_requested": int(N_NEIGHBORS),
        "n_neighbors_effective": int(
            min(N_NEIGHBORS, len(group) - 1)
        ),
        "root_cell": str(group.iloc[root]["cell"]),
        "root_index": int(root),
        "n_components": int(n_components),
        "root_component_fraction": float(
            np.mean(labels == root_component)
        ),
        "strict_state_progression": False,
        "qualified": False,
        "failure_reason": "",
    }

    if n_components != 1:
        record["failure_reason"] = "disconnected_graph"
        return record

    # ------------------------------------------------------------
    # Rooted graph-geodesic distances.
    # ------------------------------------------------------------
    distances = np.asarray(
        dijkstra(
            graph,
            directed=False,
            indices=root,
        ),
        dtype=np.float64,
    ).reshape(-1)

    if distances.shape[0] != len(group):
        raise RuntimeError(
            "Unexpected number of geodesic distances"
        )

    if not np.isfinite(distances).all():
        record["failure_reason"] = "nonfinite_distances"
        return record

    distance_range = np.ptp(distances)

    if not np.isfinite(distance_range) or distance_range <= 0:
        record["failure_reason"] = "degenerate_distances"
        return record

    # Convert geodesic distances to [0, 1] rank pseudotime.
    ranks = (
        pd.Series(distances)
        .rank(method="average")
        .to_numpy(dtype=np.float64)
    )

    if len(ranks) <= 1:
        record["failure_reason"] = "insufficient_ranks"
        return record

    geodesic = (
        (ranks - 1.0)
        / (len(ranks) - 1.0)
    )

    original = group[
        "dpt_pseudotime"
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(original).all():
        record["failure_reason"] = "invalid_original_pseudotime"
        return record

    days = (
        group["diffday"]
        .map(DAY_MAP)
        .to_numpy(dtype=np.float64)
    )

    if not np.isfinite(days).all():
        record["failure_reason"] = "invalid_day"
        return record

    # ------------------------------------------------------------
    # Prespecified state progression qualification.
    # ------------------------------------------------------------
    medians = []

    for state in expected_states:
        mask = types == state

        if not np.any(mask):
            record["failure_reason"] = (
                f"missing_state:{state}"
            )
            return record

        median = float(
            np.median(geodesic[mask])
        )

        if not np.isfinite(median):
            record["failure_reason"] = (
                f"nonfinite_state_median:{state}"
            )
            return record

        medians.append(median)

    strict = bool(
        np.all(np.diff(medians) > 0)
    )

    rho_original = safe_spearman(
        original,
        geodesic,
    )

    rho_day = safe_spearman(
        days,
        geodesic,
    )

    record.update({
        "geodesic_min": float(
            np.min(distances)
        ),
        "geodesic_max": float(
            np.max(distances)
        ),
        "spearman_original": rho_original,
        "spearman_day": rho_day,
        "strict_state_progression": strict,
        "qualified": strict,
        "failure_reason": (
            ""
            if strict
            else "state_order_violation"
        ),
    })

    for state, median in zip(
        expected_states,
        medians,
    ):
        record[f"median_{state}"] = median

    return record


def main():
    parser = argparse.ArgumentParser(
        description=(
            "P25-F6 cohort-wide graph-geodesic "
            "reconstruction qualification."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Evaluate only the first N frozen profiles. "
            "Use 0 for the full 48-profile audit."
        ),
    )

    args = parser.parse_args()

    if args.limit < 0:
        parser.error(
            "--limit must be nonnegative"
        )

    manifest = pd.read_csv(
        MANIFEST_FILE,
        dtype={"individual": str},
    )

    if len(manifest) != 48:
        raise RuntimeError(
            f"Expected 48 profiles; found {len(manifest)}"
        )

    if manifest.duplicated(
        ["scope", "individual"]
    ).any():
        duplicates = manifest.loc[
            manifest.duplicated(
                ["scope", "individual"],
                keep=False,
            ),
            ["scope", "individual"],
        ]

        raise RuntimeError(
            "Duplicate frozen profiles detected:\n"
            + duplicates.to_string(index=False)
        )

    if args.limit:
        manifest = manifest.head(
            args.limit
        ).copy()

    state_space = load_state_space()

    print(
        f"STATE SPACE LOADED: "
        f"{len(state_space)} cells",
        flush=True,
    )

    # ------------------------------------------------------------
    # Restore frozen experimental-day metadata.
    # ------------------------------------------------------------
    source = (
        ROOT
        / "results"
        / "p15_external_GSE175634_preprocessing_state_space"
        / "data"
        / "external_state_space_pca_50.csv.gz"
    )

    if not source.exists():
        raise FileNotFoundError(
            f"Frozen state-space file not found: {source}"
        )

    days = pd.read_csv(
        source,
        usecols=["cell", "diffday"],
        dtype={
            "cell": str,
            "diffday": str,
        },
    )

    if days["cell"].duplicated().any():
        raise RuntimeError(
            "Duplicate metadata IDs"
        )

    # Ensure merge key types agree.
    state_space = state_space.copy()
    state_space["cell"] = (
        state_space["cell"].astype(str)
    )
    state_space["individual"] = (
        state_space["individual"].astype(str)
    )

    state_space = state_space.merge(
        days,
        on="cell",
        validate="one_to_one",
        how="left",
    )

    if state_space["diffday"].isna().any():
        missing = int(
            state_space["diffday"].isna().sum()
        )

        raise RuntimeError(
            f"Missing day metadata for {missing} cells"
        )

    records = []

    for index, (_, row) in enumerate(
        manifest.iterrows(),
        start=1,
    ):
        try:
            result = reconstruct(
                row,
                state_space,
            )

        except Exception as exc:
            traceback.print_exc()

            result = {
                "scope": str(row["scope"]),
                "individual": str(
                    row["individual"]
                ),
                "qualified": False,
                "failure_reason": (
                    "computation_error:"
                    f"{type(exc).__name__}:"
                    f"{exc}"
                ),
            }

        records.append(result)

        print(
            f"[{index}/{len(manifest)}] "
            f"{result['scope']}/"
            f"{result['individual']} "
            f"qualified="
            f"{result['qualified']} "
            f"reason="
            f"{result.get('failure_reason', '')}",
            flush=True,
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.DataFrame(records)

    suffix = (
        "pilot"
        if args.limit
        else "full"
    )

    output_file = (
        OUT
        / f"01_geodesic_qualification_{suffix}.csv"
    )

    df.to_csv(
        output_file,
        index=False,
    )

    print(
        "\nP25-F6 RECONSTRUCTION AUDIT COMPLETE"
    )

    print(
        "Profiles evaluated:",
        len(df),
    )

    qualified = (
        df["qualified"]
        .fillna(False)
        .astype(bool)
    )

    print(
        "Qualified:",
        int(qualified.sum()),
    )

    print(
        "Not qualified:",
        int((~qualified).sum()),
    )

    print("\nBy scope:")

    summary = (
        df.assign(qualified=qualified)
        .groupby("scope")["qualified"]
        .agg(["size", "sum"])
    )

    print(
        summary.to_string()
    )

    print("\nFailure reasons:")

    print(
        df["failure_reason"]
        .fillna("")
        .replace("", "qualified")
        .value_counts()
        .to_string()
    )

    print(
        "\nOutput:",
        output_file,
    )


if __name__ == "__main__":
    main()
