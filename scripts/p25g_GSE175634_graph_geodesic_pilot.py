#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""P25-F4: Prespecified graph-geodesic pseudotime pilot."""

from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

from scipy.sparse.csgraph import (
    connected_components,
    dijkstra,
)
from scipy.stats import spearmanr

from p25b_GSE175634_cell_window_reconstruction_audit import (
    ROOT, STATES, PC_COLUMNS, load_state_space
)

SCOPE = "cf_extension"
INDIVIDUAL = "18489"
N_NEIGHBORS = 30
SEED = 250636

OUT = (
    ROOT / "results"
    / "p25_GSE175634_pseudotime_ordering_robustness"
    / "alternative_pseudotime_pilot"
)

DAY_MAP = {
    "day0": 0, "day1": 1, "day3": 3,
    "day5": 5, "day7": 7,
    "day11": 11, "day15": 15,
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    state_space = load_state_space()

    # Restore experimental-day metadata from frozen P15 input.
    day_source = (
        ROOT / "results"
        / "p15_external_GSE175634_preprocessing_state_space"
        / "data/external_state_space_pca_50.csv.gz"
    )

    day_metadata = pd.read_csv(
        day_source,
        compression="gzip",
        usecols=["cell", "diffday"],
        dtype=str,
    )

    if day_metadata["cell"].duplicated().any():
        raise RuntimeError("Duplicate cell IDs in day metadata")

    state_space = state_space.merge(
        day_metadata,
        on="cell",
        how="left",
        validate="one_to_one",
    )

    if state_space["diffday"].isna().any():
        raise RuntimeError("Missing experimental-day metadata")

    group = state_space.loc[
        (state_space["individual"] == INDIVIDUAL)
        & state_space["type"].isin(STATES[SCOPE])
    ].sort_values(
        ["dpt_pseudotime", "cell"],
        kind="mergesort",
    ).reset_index(drop=True)

    X = group[PC_COLUMNS].to_numpy(
        dtype=np.float32, copy=True
    )

    types = group["type"].to_numpy(dtype=str)
    original = group["dpt_pseudotime"].to_numpy(
        dtype=float
    )

    root_candidates = np.flatnonzero(types == "IPSC")
    if not len(root_candidates):
        raise RuntimeError("No IPSC cells")

    centroid = X[root_candidates].mean(
        axis=0, dtype=np.float64
    )

    root_distances = np.linalg.norm(
        X[root_candidates].astype(np.float64)
        - centroid,
        axis=1,
    )

    root = int(root_candidates[np.argmin(root_distances)])

    adata = ad.AnnData(X=X)
    adata.obsm["X_pca"] = X.copy()

    sc.pp.neighbors(
        adata,
        n_neighbors=N_NEIGHBORS,
        use_rep="X_pca",
        method="gauss",
        random_state=SEED,
    )

    # Graph weights are PCA-space neighbor distances.
    graph = adata.obsp["distances"].tocsr()

    # An undirected graph: retain an edge whenever
    # either direction exists, with its available distance.
    graph = graph.maximum(graph.T).tocsr()

    n_components, labels = connected_components(
        graph, directed=False
    )

    print("Connected components:", n_components)

    if n_components != 1:
        raise RuntimeError(
            "Disconnected graph: geodesic ordering not qualified"
        )

    distances = np.asarray(
        dijkstra(
            graph,
            directed=False,
            indices=root
        ),
        dtype=float
    )

    if not np.isfinite(distances).all():
        raise RuntimeError("Unreachable cells")

    if np.ptp(distances) <= 0:
        raise RuntimeError("Degenerate geodesic distances")

    # Rank-normalized coordinate prevents a few unusually
    # long distances from dominating the coordinate scale.
    ranks = pd.Series(distances).rank(
        method="average"
    ).to_numpy(dtype=float)

    geodesic = (ranks - 1) / (len(ranks) - 1)

    day_numeric = group["diffday"].map(
        DAY_MAP
    ).to_numpy(dtype=float)

    if not np.isfinite(day_numeric).all():
        raise RuntimeError("Invalid day metadata")

    rho_original = spearmanr(
        original, geodesic
    ).statistic

    rho_day = spearmanr(
        day_numeric, geodesic
    ).statistic

    comparison = pd.DataFrame({
        "cell": group["cell"].to_numpy(),
        "individual": INDIVIDUAL,
        "scope": SCOPE,
        "type": types,
        "diffday": group["diffday"].to_numpy(),
        "original_pseudotime": original,
        "geodesic_pseudotime": geodesic,
        "geodesic_distance": distances,
    })

    state_summary = comparison.groupby("type").agg(
        n_cells=("cell", "size"),
        geodesic_q25=("geodesic_pseudotime",
                      lambda x: x.quantile(0.25)),
        geodesic_median=("geodesic_pseudotime", "median"),
        geodesic_q75=("geodesic_pseudotime",
                      lambda x: x.quantile(0.75)),
    ).reindex(STATES[SCOPE])

    medians = state_summary[
        "geodesic_median"
    ].to_numpy(dtype=float)

    progression = bool(
        np.all(np.diff(medians) > 0)
    )

    day_summary = comparison.groupby("diffday").agg(
        n_cells=("cell", "size"),
        geodesic_median=("geodesic_pseudotime", "median"),
    )

    day_summary["day_numeric"] = day_summary.index.map(
        DAY_MAP
    )
    day_summary = day_summary.sort_values("day_numeric")

    comparison.to_csv(
        OUT / "08_graph_geodesic_cell_pseudotime.csv",
        index=False
    )
    state_summary.to_csv(
        OUT / "09_graph_geodesic_state_summary.csv"
    )
    day_summary.to_csv(
        OUT / "10_graph_geodesic_day_summary.csv"
    )

    print("\nP25-F4 GRAPH-GEODESIC PILOT")
    print("Cells:", len(X))
    print("Root index:", root)
    print("Root cell:", group.iloc[root]["cell"])
    print("Spearman(original, geodesic):", rho_original)
    print("Spearman(day, geodesic):", rho_day)

    print("\nBIOLOGICAL STATE SUMMARY")
    print(state_summary.to_string())

    print("\nEXPERIMENTAL DAY SUMMARY")
    print(day_summary.to_string())

    print("\nStrict state progression:", progression)
    print("\nP25-F4: COMPUTATION COMPLETE")


if __name__ == "__main__":
    main()
