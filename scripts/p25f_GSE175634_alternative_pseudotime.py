#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
P25-F1: Independent trajectory reconstruction feasibility audit.

Reconstruct Scanpy DPT within one frozen individual/scope.
No original pseudotime is used in graph or root selection.
Original DPT is used only for post-reconstruction comparison.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr

from p25b_GSE175634_cell_window_reconstruction_audit import (
    ROOT, STATES, MANIFEST_FILE, PC_COLUMNS,
    load_state_space
)

SCOPE = "cf_extension"
INDIVIDUAL = "18489"
N_NEIGHBORS = 30
N_DCS = 10
SEED = 250636

BASE = (
    ROOT / "results"
    / "p25_GSE175634_pseudotime_ordering_robustness"
)

OUT = BASE / "alternative_pseudotime_pilot"


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(
        MANIFEST_FILE,
        dtype={"individual": str},
    )

    matches = manifest.loc[
        (manifest["scope"] == SCOPE)
        & (manifest["individual"] == INDIVIDUAL)
    ]

    if len(matches) != 1:
        raise RuntimeError("Frozen profile selection invalid")

    state_space = load_state_space()

    group = state_space.loc[
        (state_space["individual"] == INDIVIDUAL)
        & (state_space["type"].isin(STATES[SCOPE]))
    ].sort_values(
        ["dpt_pseudotime", "cell"],
        kind="mergesort",
    ).reset_index(drop=True)

    X = group[PC_COLUMNS].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    types = group["type"].astype(str).to_numpy()
    original_t = group["dpt_pseudotime"].to_numpy(
        dtype=float
    )

    # Choose root independently of deposited pseudotime.
    root_candidates = np.flatnonzero(types == "IPSC")

    if len(root_candidates) == 0:
        raise RuntimeError("No IPSC cells for root selection")

    centroid = X[root_candidates].mean(
        axis=0, dtype=np.float64
    )

    distances = np.linalg.norm(
        X[root_candidates].astype(np.float64)
        - centroid,
        axis=1,
    )

    root_index = int(root_candidates[np.argmin(distances)])

    print("P25-F1 INDEPENDENT DPT RECONSTRUCTION")
    print("Scope:", SCOPE)
    print("Individual:", INDIVIDUAL)
    print("Cells:", len(X))
    print("PC dimensions:", X.shape[1])
    print("Root index:", root_index)
    print("Root state:", types[root_index])
    print("Root cell:", group.iloc[root_index]["cell"])

    # PCA coordinates are already frozen.
    # Do not repeat preprocessing or PCA fitting.
    adata = ad.AnnData(X=X)
    adata.obsm["X_pca"] = X.copy()
    adata.obs["cell"] = group["cell"].to_numpy()
    adata.obs["type"] = types
    adata.uns["iroot"] = root_index

    sc.pp.neighbors(
        adata,
        n_neighbors=N_NEIGHBORS,
        use_rep="X_pca",
        method="gauss",
        random_state=SEED,
    )

    n_components, labels = connected_components(
        adata.obsp["connectivities"],
        directed=False,
    )

    component_sizes = np.bincount(labels)
    root_component = labels[root_index]
    reachable_fraction = float(
        component_sizes[root_component] / len(X)
    )

    print("\nGRAPH CONNECTIVITY")
    print("Connected components:", n_components)
    print("Largest component:", int(component_sizes.max()))
    print("Root-component fraction:", reachable_fraction)

    if n_components != 1:
        pd.DataFrame({
            "cell": group["cell"].to_numpy(),
            "type": types,
            "graph_component": labels,
        }).to_csv(
            OUT / "01_graph_connectivity_diagnostics.csv",
            index=False,
        )

        raise RuntimeError(
            "Neighbor graph disconnected. "
            "Inspect connectivity before interpreting DPT."
        )

    # Compute independently reconstructed diffusion map and DPT.
    sc.tl.diffmap(
        adata,
        n_comps=N_DCS + 1,
        random_state=SEED,
    )

    sc.tl.dpt(
        adata,
        n_dcs=N_DCS,
        n_branchings=0,
    )

    alternative_t = adata.obs[
        "dpt_pseudotime"
    ].to_numpy(dtype=float)

    if not np.isfinite(alternative_t).all():
        raise RuntimeError(
            "Alternative DPT contains nonfinite values"
        )

    if np.ptp(alternative_t) <= 0:
        raise RuntimeError("Degenerate alternative pseudotime")

    # Rank agreement does not require equal pseudotime scales.
    rho, _ = spearmanr(original_t, alternative_t)

    comparison = pd.DataFrame({
        "cell": group["cell"].to_numpy(),
        "individual": INDIVIDUAL,
        "scope": SCOPE,
        "type": types,
        "original_pseudotime": original_t,
        "alternative_pseudotime": alternative_t,
    })

    summary = comparison.groupby("type").agg(
        n_cells=("cell", "size"),
        original_median=("original_pseudotime", "median"),
        alternative_median=("alternative_pseudotime", "median"),
    )

    print("\nPSEUDOTIME RECONSTRUCTION")
    print("Spearman(original, alternative):", rho)
    print("Alternative min:", alternative_t.min())
    print("Alternative max:", alternative_t.max())

    print("\nBIOLOGICAL STATE MEDIANS")
    print(summary.to_string())

    # Report ordering consistency separately.
    expected = STATES[SCOPE]
    medians = summary.loc[
        expected, "alternative_median"
    ].to_numpy()

    progression_consistent = bool(
        np.all(np.diff(medians) > 0)
    )

    print("\nEXPECTED BIOLOGICAL PROGRESSION")
    print(" -> ".join(expected))
    print("Strictly increasing state medians:",
          progression_consistent)

    comparison.to_csv(
        OUT / "02_original_vs_alternative_pseudotime.csv",
        index=False,
    )

    summary.to_csv(
        OUT / "03_state_pseudotime_summary.csv"
    )

    pd.DataFrame([{
        "scope": SCOPE,
        "individual": INDIVIDUAL,
        "n_cells": len(X),
        "n_neighbors": N_NEIGHBORS,
        "n_components": n_components,
        "root_component_fraction": reachable_fraction,
        "spearman_original_vs_alternative": rho,
        "strict_state_progression": progression_consistent,
    }]).to_csv(
        OUT / "04_reconstruction_quality.csv",
        index=False,
    )

    print("\nP25-F1 RECONSTRUCTION: COMPLETE")
    print("Output:", OUT)


if __name__ == "__main__":
    main()
