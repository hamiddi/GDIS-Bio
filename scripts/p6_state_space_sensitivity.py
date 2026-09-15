#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================
"""
p6_state_space_sensitivity.py

State-space dimensionality sensitivity analysis for GDIS-Bio.

Purpose
-------
Compare the pre-specified PCA state spaces:

    5, 10, 20, 30, 50 PCs

BEFORE calculating GDIS.

The aim is to choose a defensible primary dimensionality based on
representation stability rather than selecting a dimensionality after
seeing a favorable GDIS result.

Inputs
------
Created by p5_preprocessing_state_space.py:

results/p5_preprocessing_state_space/
├── data/
│   ├── state_space_pca_05.csv.gz
│   ├── state_space_pca_10.csv.gz
│   ├── state_space_pca_20.csv.gz
│   ├── state_space_pca_30.csv.gz
│   └── state_space_pca_50.csv.gz
└── tables/
    └── 05_pca_explained_variance.csv

Analyses
--------
For each dimensionality, this script evaluates:

1. Cumulative explained variance.
2. Pairwise Euclidean-distance preservation relative to the 50-PC
   reference space on a fixed stratified cell sample.
3. 30-nearest-neighbor overlap relative to the 50-PC reference.
4. Biological-state centroid geometry preservation relative to 50 PCs.
5. Cross-replicate trajectory-geometry agreement.
6. Cross-replicate pseudotime prediction:
      train Differentiation 1 -> test Differentiation 2
      train Differentiation 2 -> test Differentiation 1
   using Ridge regression.
7. Cross-replicate terminal-fate classification (SC-beta vs SC-EC)
   using logistic regression.
8. A pre-specified selection rule that chooses the SMALLEST
   dimensionality satisfying all representation-stability thresholds.

IMPORTANT
---------
This script performs NO GDIS calculation.

It also does not:
- re-normalize expression
- re-select HVGs
- recompute PCA
- infer pseudotime
- alter branch assignments
- batch-correct the state space

Requirements
------------
numpy
pandas
scipy
matplotlib
scikit-learn
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

if SCRIPT_DIR.name == "scripts":
    PROJECT_DIR = SCRIPT_DIR.parent
else:
    PROJECT_DIR = SCRIPT_DIR

P5_DIR = PROJECT_DIR / "results" / "p5_preprocessing_state_space"
P5_DATA_DIR = P5_DIR / "data"
P5_TABLE_DIR = P5_DIR / "tables"

RESULTS_DIR = PROJECT_DIR / "results" / "p6_state_space_sensitivity"
TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR = RESULTS_DIR / "figures"
REPORT_FILE = RESULTS_DIR / "p6_state_space_sensitivity_report.txt"

PCA_VARIANCE_FILE = P5_TABLE_DIR / "05_pca_explained_variance.csv"

DIMENSIONS = [5, 10, 20, 30, 50]
REFERENCE_DIM = 50

RANDOM_SEED = 0
DISTANCE_SAMPLE_SIZE = 2000
K_NEIGHBORS = 30

# ---------------------------------------------------------------------
# Pre-specified dimensionality-selection thresholds
# ---------------------------------------------------------------------
#
# These thresholds are intentionally based on representation stability,
# NOT on GDIS values.
#
# The 50-PC space serves as the high-dimensional reference.
#

MIN_DISTANCE_SPEARMAN = 0.90
MIN_KNN_OVERLAP = 0.60
MIN_CENTROID_DISTANCE_SPEARMAN = 0.95
MIN_REPLICATE_GEOMETRY_CORRELATION = 0.95
MIN_PSEUDOTIME_GENERALIZATION_RHO = 0.70
MIN_TERMINAL_FATE_BALANCED_ACCURACY = 0.90


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def write_both(report, text=""):
    """Print text and write it to the report."""
    print(text)
    report.write(text + "\n")
    report.flush()


def save_table(df, filename, index=True):
    """Save dataframe as CSV."""
    path = TABLE_DIR / filename
    df.to_csv(path, index=index)
    return path


def state_space_path(n_dim):
    """Return state-space file path."""
    return P5_DATA_DIR / f"state_space_pca_{n_dim:02d}.csv.gz"


def pc_columns(n_dim):
    """Expected PC column names."""
    return [f"PC{i}" for i in range(1, n_dim + 1)]


def ensure_required_columns(df, columns, name):
    """Verify required columns are present."""
    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"{name} is missing required column(s): "
            + ", ".join(missing)
        )


def safe_spearman(x, y):
    """Return Spearman rho or NaN."""
    x = np.asarray(x)
    y = np.asarray(y)

    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 3:
        return np.nan

    result = spearmanr(
        x[mask],
        y[mask],
    )

    return float(result.statistic)


def safe_pearson(x, y):
    """Return Pearson correlation or NaN."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def stratified_sample_indices(metadata, n_total, random_seed):
    """
    Draw an approximately balanced sample over:
        branch x differentiation x day

    This prevents large Day-3 groups from dominating distance analysis.
    """
    rng = np.random.default_rng(random_seed)

    group_cols = [
        "Pseudotime_branch",
        "Differentiation",
        "CellDay",
    ]

    groups = list(
        metadata.groupby(
            group_cols,
            observed=True,
            sort=True,
        )
    )

    if not groups:
        raise RuntimeError(
            "No groups available for stratified sampling."
        )

    target_per_group = max(
        1,
        int(np.ceil(n_total / len(groups))),
    )

    chosen = []

    for _, group in groups:

        indices = group.index.to_numpy()

        n_take = min(
            target_per_group,
            len(indices),
        )

        sampled = rng.choice(
            indices,
            size=n_take,
            replace=False,
        )

        chosen.extend(sampled.tolist())

    chosen = np.array(
        sorted(set(chosen)),
        dtype=int,
    )

    # If stratification overshot the target, trim reproducibly.
    if len(chosen) > n_total:
        chosen = np.sort(
            rng.choice(
                chosen,
                size=n_total,
                replace=False,
            )
        )

    # If undershot, top up from remaining cells.
    if len(chosen) < n_total:

        all_indices = np.arange(
            len(metadata),
            dtype=int,
        )

        remaining = np.setdiff1d(
            all_indices,
            chosen,
            assume_unique=False,
        )

        n_needed = min(
            n_total - len(chosen),
            len(remaining),
        )

        extra = rng.choice(
            remaining,
            size=n_needed,
            replace=False,
        )

        chosen = np.sort(
            np.concatenate(
                [chosen, extra]
            )
        )

    return chosen


def neighborhood_overlap(X, X_ref, k):
    """
    Mean fraction of k nearest neighbors shared with the reference space.

    The cell itself is removed from each neighbor set.
    """
    n_neighbors = k + 1

    model_ref = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean",
        algorithm="auto",
        n_jobs=-1,
    )

    model_ref.fit(X_ref)

    ref_indices = model_ref.kneighbors(
        X_ref,
        return_distance=False,
    )[:, 1:]

    model = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean",
        algorithm="auto",
        n_jobs=-1,
    )

    model.fit(X)

    indices = model.kneighbors(
        X,
        return_distance=False,
    )[:, 1:]

    overlaps = np.empty(
        X.shape[0],
        dtype=float,
    )

    for i in range(X.shape[0]):

        overlaps[i] = (
            len(
                set(indices[i]).intersection(
                    ref_indices[i]
                )
            )
            / k
        )

    return {
        "mean": float(np.mean(overlaps)),
        "median": float(np.median(overlaps)),
        "q25": float(np.quantile(overlaps, 0.25)),
        "q75": float(np.quantile(overlaps, 0.75)),
    }


def centroid_table(df, n_dim):
    """
    Build biologically matched centroids for:
        branch x differentiation x state
    """
    pcs = pc_columns(n_dim)

    grouped = (
        df.groupby(
            [
                "Pseudotime_branch",
                "Differentiation",
                "Assigned_cluster",
            ],
            observed=True,
            sort=True,
        )[pcs]
        .mean()
        .reset_index()
    )

    grouped["centroid_key"] = (
        grouped["Pseudotime_branch"].astype(str)
        + "|"
        + grouped["Differentiation"].astype(str)
        + "|"
        + grouped["Assigned_cluster"].astype(str)
    )

    return grouped


def centroid_geometry_correlation(
    centroids,
    centroids_ref,
    n_dim,
    reference_dim,
):
    """
    Compare pairwise distances among matched biological centroids
    with the same centroid distances in the 50-PC reference space.
    """
    pcs = pc_columns(n_dim)
    ref_pcs = pc_columns(reference_dim)

    current = centroids.set_index(
        "centroid_key"
    )

    reference = centroids_ref.set_index(
        "centroid_key"
    )

    common = sorted(
        set(current.index).intersection(
            reference.index
        )
    )

    if len(common) < 3:
        return np.nan

    X = current.loc[
        common,
        pcs,
    ].to_numpy(dtype=float)

    X_ref = reference.loc[
        common,
        ref_pcs,
    ].to_numpy(dtype=float)

    d = pdist(X, metric="euclidean")
    d_ref = pdist(
        X_ref,
        metric="euclidean",
    )

    return safe_spearman(
        d,
        d_ref,
    )


def branch_state_sequence(branch):
    """Expected state sequence for each fate branch."""
    endpoint = (
        "sc_ec"
        if int(branch) == 0
        else "sc_beta"
    )

    return [
        "prog_nkx61",
        "neurog3_early",
        "neurog3_mid",
        "neurog3_late",
        endpoint,
    ]


def replicate_geometry_correlations(df, n_dim):
    """
    Compare trajectory-state geometry between Differentiation 1 and 2.

    For each branch:
    - compute state centroids separately in each differentiation
    - calculate all pairwise distances among the ordered state centroids
    - correlate those distance vectors between differentiations
    """
    pcs = pc_columns(n_dim)

    rows = []

    for branch in sorted(
        df["Pseudotime_branch"].unique()
    ):

        states = branch_state_sequence(branch)

        distance_vectors = {}

        for diff in sorted(
            df["Differentiation"].unique()
        ):

            subset = df[
                (df["Pseudotime_branch"] == branch)
                & (df["Differentiation"] == diff)
            ]

            centroids = (
                subset.groupby(
                    "Assigned_cluster",
                    observed=True,
                )[pcs]
                .mean()
            )

            if not set(states).issubset(
                centroids.index
            ):
                distance_vectors[diff] = None
                continue

            X = centroids.loc[
                states,
                pcs,
            ].to_numpy(dtype=float)

            distance_vectors[diff] = pdist(
                X,
                metric="euclidean",
            )

        diffs = sorted(
            distance_vectors.keys()
        )

        if (
            len(diffs) == 2
            and distance_vectors[diffs[0]]
            is not None
            and distance_vectors[diffs[1]]
            is not None
        ):
            corr = safe_pearson(
                distance_vectors[diffs[0]],
                distance_vectors[diffs[1]],
            )
        else:
            corr = np.nan

        rows.append(
            {
                "dimension": n_dim,
                "branch": int(branch),
                "replicate_geometry_correlation": corr,
            }
        )

    return pd.DataFrame(rows)


def cross_replicate_pseudotime_prediction(df, n_dim):
    """
    Train pseudotime regression in one differentiation and evaluate it
    in the other.

    Run for:
    - all trajectory cells
    - branch 0
    - branch 1

    Ridge regression is used without scaling because the PCA coordinate
    system itself is the representation being tested.
    """
    pcs = pc_columns(n_dim)
    diffs = sorted(
        df["Differentiation"].unique()
    )

    if len(diffs) != 2:
        raise ValueError(
            "Expected exactly two differentiations."
        )

    contexts = [
        ("all", None),
        ("branch_0", 0),
        ("branch_1", 1),
    ]

    rows = []

    for context_name, branch in contexts:

        if branch is None:
            context_df = df
        else:
            context_df = df[
                df["Pseudotime_branch"] == branch
            ]

        for train_diff, test_diff in [
            (diffs[0], diffs[1]),
            (diffs[1], diffs[0]),
        ]:

            train = context_df[
                context_df["Differentiation"]
                == train_diff
            ]

            test = context_df[
                context_df["Differentiation"]
                == test_diff
            ]

            model = Ridge(
                alpha=1.0,
            )

            model.fit(
                train[pcs].to_numpy(dtype=float),
                train["Pseudotime_value"].to_numpy(
                    dtype=float
                ),
            )

            pred = model.predict(
                test[pcs].to_numpy(dtype=float)
            )

            truth = test[
                "Pseudotime_value"
            ].to_numpy(dtype=float)

            rho = safe_spearman(
                pred,
                truth,
            )

            mae = float(
                np.mean(
                    np.abs(
                        pred - truth
                    )
                )
            )

            rows.append(
                {
                    "dimension": n_dim,
                    "context": context_name,
                    "train_differentiation":
                        int(train_diff),
                    "test_differentiation":
                        int(test_diff),
                    "n_train": len(train),
                    "n_test": len(test),
                    "spearman_predicted_vs_true":
                        rho,
                    "mean_absolute_error":
                        mae,
                }
            )

    return pd.DataFrame(rows)


def cross_replicate_terminal_fate(df, n_dim):
    """
    Cross-replicate classification of terminal fate:
        sc_ec   -> 0
        sc_beta -> 1

    Only terminal cells are included.
    """
    pcs = pc_columns(n_dim)

    terminal = df[
        df["Assigned_cluster"].isin(
            ["sc_ec", "sc_beta"]
        )
    ].copy()

    terminal["fate_label"] = (
        terminal["Assigned_cluster"]
        == "sc_beta"
    ).astype(int)

    diffs = sorted(
        terminal["Differentiation"].unique()
    )

    rows = []

    for train_diff, test_diff in [
        (diffs[0], diffs[1]),
        (diffs[1], diffs[0]),
    ]:

        train = terminal[
            terminal["Differentiation"]
            == train_diff
        ]

        test = terminal[
            terminal["Differentiation"]
            == test_diff
        ]

        model = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            random_state=RANDOM_SEED,
        )

        model.fit(
            train[pcs].to_numpy(dtype=float),
            train["fate_label"].to_numpy(
                dtype=int
            ),
        )

        pred = model.predict(
            test[pcs].to_numpy(dtype=float)
        )

        prob = model.predict_proba(
            test[pcs].to_numpy(dtype=float)
        )[:, 1]

        y_true = test[
            "fate_label"
        ].to_numpy(dtype=int)

        balanced_acc = float(
            balanced_accuracy_score(
                y_true,
                pred,
            )
        )

        auc = float(
            roc_auc_score(
                y_true,
                prob,
            )
        )

        rows.append(
            {
                "dimension": n_dim,
                "train_differentiation":
                    int(train_diff),
                "test_differentiation":
                    int(test_diff),
                "n_train": len(train),
                "n_test": len(test),
                "balanced_accuracy":
                    balanced_acc,
                "roc_auc": auc,
            }
        )

    return pd.DataFrame(rows)


def plot_metric(
    summary,
    metric,
    ylabel,
    output_path,
    threshold=None,
):
    """Line plot for one dimensionality-sensitivity metric."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    ax.plot(
        summary["dimension"],
        summary[metric],
        marker="o",
        linewidth=1.8,
    )

    if threshold is not None:
        ax.axhline(
            threshold,
            linestyle="--",
            linewidth=1.2,
        )

    ax.set_xlabel("Number of principal components")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"State-space sensitivity: {ylabel}"
    )
    ax.set_xticks(DIMENSIONS)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as report:

        write_both(
            report,
            "=" * 88,
        )

        write_both(
            report,
            "GDIS-Bio State-Space Dimensionality Sensitivity",
        )

        write_both(
            report,
            "Dataset: GSE114412 Stage-5 endocrine trajectory",
        )

        write_both(
            report,
            "=" * 88,
        )

        write_both(
            report,
            "\nNo GDIS calculation is performed in this script.",
        )

        # -------------------------------------------------------------
        # 1. Input validation
        # -------------------------------------------------------------

        write_both(
            report,
            "\n1. INPUT VALIDATION",
        )

        write_both(
            report,
            "-" * 88,
        )

        spaces = {}

        required_meta = [
            "library.barcode",
            "Assigned_cluster",
            "Pseudotime_value",
            "Pseudotime_branch",
            "Differentiation",
            "CellDay",
        ]

        reference_barcodes = None

        for dim in DIMENSIONS:

            path = state_space_path(dim)

            if not path.exists():
                raise FileNotFoundError(
                    f"Missing state-space file: {path}"
                )

            df = pd.read_csv(
                path,
                compression="gzip",
                low_memory=False,
            )

            ensure_required_columns(
                df,
                required_meta
                + pc_columns(dim),
                f"{dim}-PC state space",
            )

            df["library.barcode"] = (
                df["library.barcode"].astype(str)
            )

            if df["library.barcode"].duplicated().any():
                raise ValueError(
                    f"Duplicate cell barcodes in {dim}-PC space."
                )

            if reference_barcodes is None:

                reference_barcodes = (
                    df["library.barcode"].tolist()
                )

            else:

                if (
                    df["library.barcode"].tolist()
                    != reference_barcodes
                ):
                    raise ValueError(
                        f"Cell order differs in {dim}-PC space."
                    )

            spaces[dim] = df

            write_both(
                report,
                f"[OK] {dim:2d} PCs: "
                f"{len(df):,} cells",
            )

        ref = spaces[REFERENCE_DIM]

        # -------------------------------------------------------------
        # 2. Explained variance
        # -------------------------------------------------------------

        write_both(
            report,
            "\n2. EXPLAINED VARIANCE",
        )

        write_both(
            report,
            "-" * 88,
        )

        variance = pd.read_csv(
            PCA_VARIANCE_FILE,
        )

        ensure_required_columns(
            variance,
            [
                "PC",
                "variance_ratio",
                "cumulative_variance_ratio",
            ],
            "PCA explained variance table",
        )

        variance_rows = []

        for dim in DIMENSIONS:

            row = variance[
                variance["PC"] == dim
            ]

            if row.empty:
                cumulative = np.nan
            else:
                cumulative = float(
                    row.iloc[0][
                        "cumulative_variance_ratio"
                    ]
                )

            variance_rows.append(
                {
                    "dimension": dim,
                    "cumulative_explained_variance":
                        cumulative,
                }
            )

        variance_summary = pd.DataFrame(
            variance_rows
        )

        save_table(
            variance_summary,
            "01_explained_variance_by_dimension.csv",
            index=False,
        )

        write_both(
            report,
            variance_summary.round(6).to_string(
                index=False
            ),
        )

        # -------------------------------------------------------------
        # 3. Fixed stratified sample
        # -------------------------------------------------------------

        write_both(
            report,
            "\n3. FIXED STRATIFIED SAMPLE FOR DISTANCE ANALYSIS",
        )

        write_both(
            report,
            "-" * 88,
        )

        sample_indices = stratified_sample_indices(
            ref,
            min(
                DISTANCE_SAMPLE_SIZE,
                len(ref),
            ),
            RANDOM_SEED,
        )

        sample_meta = ref.iloc[
            sample_indices
        ][required_meta].copy()

        save_table(
            sample_meta,
            "02_distance_sample_cells.csv",
            index=False,
        )

        write_both(
            report,
            f"Distance-analysis sample size: "
            f"{len(sample_indices):,}",
        )

        # -------------------------------------------------------------
        # 4. Pairwise distance preservation
        # -------------------------------------------------------------

        write_both(
            report,
            "\n4. PAIRWISE DISTANCE PRESERVATION RELATIVE TO 50 PCs",
        )

        write_both(
            report,
            "-" * 88,
        )

        X_ref_sample = ref.iloc[
            sample_indices
        ][
            pc_columns(REFERENCE_DIM)
        ].to_numpy(dtype=float)

        reference_distances = pdist(
            X_ref_sample,
            metric="euclidean",
        )

        distance_rows = []

        for dim in DIMENSIONS:

            X = spaces[dim].iloc[
                sample_indices
            ][
                pc_columns(dim)
            ].to_numpy(dtype=float)

            d = pdist(
                X,
                metric="euclidean",
            )

            rho = safe_spearman(
                d,
                reference_distances,
            )

            distance_rows.append(
                {
                    "dimension": dim,
                    "distance_spearman_vs_50PC":
                        rho,
                }
            )

        distance_table = pd.DataFrame(
            distance_rows
        )

        save_table(
            distance_table,
            "03_pairwise_distance_preservation.csv",
            index=False,
        )

        write_both(
            report,
            distance_table.round(6).to_string(
                index=False
            ),
        )

        # -------------------------------------------------------------
        # 5. kNN overlap
        # -------------------------------------------------------------

        write_both(
            report,
            "\n5. 30-NEAREST-NEIGHBOR OVERLAP RELATIVE TO 50 PCs",
        )

        write_both(
            report,
            "-" * 88,
        )

        X_ref = ref[
            pc_columns(REFERENCE_DIM)
        ].to_numpy(dtype=np.float32)

        knn_rows = []

        # Compute reference neighbors once.
        ref_model = NearestNeighbors(
            n_neighbors=K_NEIGHBORS + 1,
            metric="euclidean",
            algorithm="auto",
            n_jobs=-1,
        )

        ref_model.fit(X_ref)

        ref_neighbors = ref_model.kneighbors(
            X_ref,
            return_distance=False,
        )[:, 1:]

        for dim in DIMENSIONS:

            X = spaces[dim][
                pc_columns(dim)
            ].to_numpy(dtype=np.float32)

            if dim == REFERENCE_DIM:

                stats = {
                    "mean": 1.0,
                    "median": 1.0,
                    "q25": 1.0,
                    "q75": 1.0,
                }

            else:

                model = NearestNeighbors(
                    n_neighbors=K_NEIGHBORS + 1,
                    metric="euclidean",
                    algorithm="auto",
                    n_jobs=-1,
                )

                model.fit(X)

                neighbors = model.kneighbors(
                    X,
                    return_distance=False,
                )[:, 1:]

                overlap = np.empty(
                    len(X),
                    dtype=float,
                )

                for i in range(len(X)):

                    overlap[i] = (
                        len(
                            set(
                                neighbors[i]
                            ).intersection(
                                ref_neighbors[i]
                            )
                        )
                        / K_NEIGHBORS
                    )

                stats = {
                    "mean": float(
                        np.mean(overlap)
                    ),
                    "median": float(
                        np.median(overlap)
                    ),
                    "q25": float(
                        np.quantile(
                            overlap,
                            0.25,
                        )
                    ),
                    "q75": float(
                        np.quantile(
                            overlap,
                            0.75,
                        )
                    ),
                }

            knn_rows.append(
                {
                    "dimension": dim,
                    "mean_knn_overlap_vs_50PC":
                        stats["mean"],
                    "median_knn_overlap_vs_50PC":
                        stats["median"],
                    "q25_knn_overlap_vs_50PC":
                        stats["q25"],
                    "q75_knn_overlap_vs_50PC":
                        stats["q75"],
                }
            )

        knn_table = pd.DataFrame(
            knn_rows
        )

        save_table(
            knn_table,
            "04_knn_overlap_vs_50PC.csv",
            index=False,
        )

        write_both(
            report,
            knn_table.round(6).to_string(
                index=False
            ),
        )

        # -------------------------------------------------------------
        # 6. Biological centroid geometry
        # -------------------------------------------------------------

        write_both(
            report,
            "\n6. BIOLOGICAL-CENTROID GEOMETRY PRESERVATION",
        )

        write_both(
            report,
            "-" * 88,
        )

        centroid_ref = centroid_table(
            ref,
            REFERENCE_DIM,
        )

        centroid_rows = []

        for dim in DIMENSIONS:

            centroids = centroid_table(
                spaces[dim],
                dim,
            )

            corr = centroid_geometry_correlation(
                centroids,
                centroid_ref,
                dim,
                REFERENCE_DIM,
            )

            centroid_rows.append(
                {
                    "dimension": dim,
                    "centroid_distance_spearman_vs_50PC":
                        corr,
                }
            )

        centroid_table_summary = pd.DataFrame(
            centroid_rows
        )

        save_table(
            centroid_table_summary,
            "05_centroid_geometry_preservation.csv",
            index=False,
        )

        write_both(
            report,
            centroid_table_summary.round(6).to_string(
                index=False
            ),
        )

        # -------------------------------------------------------------
        # 7. Replicate trajectory geometry
        # -------------------------------------------------------------

        write_both(
            report,
            "\n7. CROSS-REPLICATE TRAJECTORY-GEOMETRY AGREEMENT",
        )

        write_both(
            report,
            "-" * 88,
        )

        replicate_geometry_tables = []

        for dim in DIMENSIONS:

            result = replicate_geometry_correlations(
                spaces[dim],
                dim,
            )

            replicate_geometry_tables.append(
                result
            )

        replicate_geometry = pd.concat(
            replicate_geometry_tables,
            ignore_index=True,
        )

        save_table(
            replicate_geometry,
            "06_replicate_trajectory_geometry.csv",
            index=False,
        )

        write_both(
            report,
            replicate_geometry.round(6).to_string(
                index=False
            ),
        )

        # -------------------------------------------------------------
        # 8. Cross-replicate pseudotime prediction
        # -------------------------------------------------------------

        write_both(
            report,
            "\n8. CROSS-REPLICATE PSEUDOTIME GENERALIZATION",
        )

        write_both(
            report,
            "-" * 88,
        )

        prediction_tables = []

        for dim in DIMENSIONS:

            result = cross_replicate_pseudotime_prediction(
                spaces[dim],
                dim,
            )

            prediction_tables.append(
                result
            )

        prediction_table = pd.concat(
            prediction_tables,
            ignore_index=True,
        )

        save_table(
            prediction_table,
            "07_cross_replicate_pseudotime_prediction.csv",
            index=False,
        )

        prediction_summary = (
            prediction_table.groupby(
                "dimension",
                observed=True,
            )
            .agg(
                minimum_generalization_rho=(
                    "spearman_predicted_vs_true",
                    "min",
                ),
                mean_generalization_rho=(
                    "spearman_predicted_vs_true",
                    "mean",
                ),
                mean_absolute_error=(
                    "mean_absolute_error",
                    "mean",
                ),
            )
            .reset_index()
        )

        save_table(
            prediction_summary,
            "08_pseudotime_generalization_summary.csv",
            index=False,
        )

        write_both(
            report,
            prediction_summary.round(6).to_string(
                index=False
            ),
        )

        # -------------------------------------------------------------
        # 9. Cross-replicate terminal fate classification
        # -------------------------------------------------------------

        write_both(
            report,
            "\n9. CROSS-REPLICATE TERMINAL-FATE CLASSIFICATION",
        )

        write_both(
            report,
            "-" * 88,
        )

        fate_tables = []

        for dim in DIMENSIONS:

            result = cross_replicate_terminal_fate(
                spaces[dim],
                dim,
            )

            fate_tables.append(
                result
            )

        fate_table = pd.concat(
            fate_tables,
            ignore_index=True,
        )

        save_table(
            fate_table,
            "09_cross_replicate_terminal_fate.csv",
            index=False,
        )

        fate_summary = (
            fate_table.groupby(
                "dimension",
                observed=True,
            )
            .agg(
                minimum_balanced_accuracy=(
                    "balanced_accuracy",
                    "min",
                ),
                mean_balanced_accuracy=(
                    "balanced_accuracy",
                    "mean",
                ),
                minimum_roc_auc=(
                    "roc_auc",
                    "min",
                ),
                mean_roc_auc=(
                    "roc_auc",
                    "mean",
                ),
            )
            .reset_index()
        )

        save_table(
            fate_summary,
            "10_terminal_fate_summary.csv",
            index=False,
        )

        write_both(
            report,
            fate_summary.round(6).to_string(
                index=False
            ),
        )

        # -------------------------------------------------------------
        # 10. Integrated sensitivity table
        # -------------------------------------------------------------

        write_both(
            report,
            "\n10. INTEGRATED STATE-SPACE SENSITIVITY SUMMARY",
        )

        write_both(
            report,
            "-" * 88,
        )

        rep_summary = (
            replicate_geometry.groupby(
                "dimension",
                observed=True,
            )
            .agg(
                minimum_replicate_geometry_correlation=(
                    "replicate_geometry_correlation",
                    "min",
                ),
                mean_replicate_geometry_correlation=(
                    "replicate_geometry_correlation",
                    "mean",
                ),
            )
            .reset_index()
        )

        summary = variance_summary.merge(
            distance_table,
            on="dimension",
            how="left",
        )

        summary = summary.merge(
            knn_table[
                [
                    "dimension",
                    "mean_knn_overlap_vs_50PC",
                    "median_knn_overlap_vs_50PC",
                ]
            ],
            on="dimension",
            how="left",
        )

        summary = summary.merge(
            centroid_table_summary,
            on="dimension",
            how="left",
        )

        summary = summary.merge(
            rep_summary,
            on="dimension",
            how="left",
        )

        summary = summary.merge(
            prediction_summary[
                [
                    "dimension",
                    "minimum_generalization_rho",
                    "mean_generalization_rho",
                ]
            ],
            on="dimension",
            how="left",
        )

        summary = summary.merge(
            fate_summary[
                [
                    "dimension",
                    "minimum_balanced_accuracy",
                    "mean_balanced_accuracy",
                ]
            ],
            on="dimension",
            how="left",
        )

        # Pre-specified qualification rule.
        summary["pass_distance"] = (
            summary[
                "distance_spearman_vs_50PC"
            ]
            >= MIN_DISTANCE_SPEARMAN
        )

        summary["pass_knn"] = (
            summary[
                "mean_knn_overlap_vs_50PC"
            ]
            >= MIN_KNN_OVERLAP
        )

        summary["pass_centroid"] = (
            summary[
                "centroid_distance_spearman_vs_50PC"
            ]
            >= MIN_CENTROID_DISTANCE_SPEARMAN
        )

        summary["pass_replicate_geometry"] = (
            summary[
                "minimum_replicate_geometry_correlation"
            ]
            >= MIN_REPLICATE_GEOMETRY_CORRELATION
        )

        summary["pass_pseudotime_generalization"] = (
            summary[
                "minimum_generalization_rho"
            ]
            >= MIN_PSEUDOTIME_GENERALIZATION_RHO
        )

        summary["pass_terminal_fate"] = (
            summary[
                "minimum_balanced_accuracy"
            ]
            >= MIN_TERMINAL_FATE_BALANCED_ACCURACY
        )

        pass_cols = [
            "pass_distance",
            "pass_knn",
            "pass_centroid",
            "pass_replicate_geometry",
            "pass_pseudotime_generalization",
            "pass_terminal_fate",
        ]

        summary[
            "n_stability_checks_passed"
        ] = summary[pass_cols].sum(
            axis=1
        )

        summary[
            "all_stability_checks_pass"
        ] = summary[pass_cols].all(
            axis=1
        )

        save_table(
            summary,
            "11_integrated_state_space_sensitivity.csv",
            index=False,
        )

        display_cols = [
            "dimension",
            "cumulative_explained_variance",
            "distance_spearman_vs_50PC",
            "mean_knn_overlap_vs_50PC",
            "centroid_distance_spearman_vs_50PC",
            "minimum_replicate_geometry_correlation",
            "minimum_generalization_rho",
            "minimum_balanced_accuracy",
            "n_stability_checks_passed",
            "all_stability_checks_pass",
        ]

        write_both(
            report,
            summary[
                display_cols
            ].round(6).to_string(
                index=False
            ),
        )

        # -------------------------------------------------------------
        # 11. Pre-specified dimensionality selection
        # -------------------------------------------------------------

        write_both(
            report,
            "\n11. PRE-SPECIFIED PRIMARY DIMENSIONALITY SELECTION",
        )

        write_both(
            report,
            "-" * 88,
        )

        write_both(
            report,
            "Selection rule: choose the SMALLEST PC dimensionality "
            "that satisfies ALL representation-stability criteria.",
        )

        write_both(
            report,
            f"Thresholds:",
        )

        write_both(
            report,
            f"  distance Spearman vs 50 PCs >= "
            f"{MIN_DISTANCE_SPEARMAN:.2f}",
        )

        write_both(
            report,
            f"  mean 30-NN overlap vs 50 PCs >= "
            f"{MIN_KNN_OVERLAP:.2f}",
        )

        write_both(
            report,
            f"  centroid-distance Spearman vs 50 PCs >= "
            f"{MIN_CENTROID_DISTANCE_SPEARMAN:.2f}",
        )

        write_both(
            report,
            f"  minimum replicate-geometry correlation >= "
            f"{MIN_REPLICATE_GEOMETRY_CORRELATION:.2f}",
        )

        write_both(
            report,
            f"  minimum cross-replicate pseudotime rho >= "
            f"{MIN_PSEUDOTIME_GENERALIZATION_RHO:.2f}",
        )

        write_both(
            report,
            f"  minimum terminal-fate balanced accuracy >= "
            f"{MIN_TERMINAL_FATE_BALANCED_ACCURACY:.2f}",
        )

        qualifying = summary[
            summary[
                "all_stability_checks_pass"
            ]
        ].sort_values(
            "dimension"
        )

        if not qualifying.empty:

            selected_dim = int(
                qualifying.iloc[0][
                    "dimension"
                ]
            )

            write_both(
                report,
                f"\nPRIMARY DIMENSIONALITY SELECTED: "
                f"{selected_dim} PCs",
            )

            selection_status = "PASS"

        else:

            selected_dim = REFERENCE_DIM

            write_both(
                report,
                "\nNo reduced dimensionality satisfied all "
                "pre-specified criteria.",
            )

            write_both(
                report,
                f"PRIMARY DIMENSIONALITY FALLBACK: "
                f"{REFERENCE_DIM} PCs",
            )

            selection_status = "REFERENCE_FALLBACK"

        selection = pd.DataFrame(
            [
                {
                    "selected_primary_dimension":
                        selected_dim,
                    "selection_status":
                        selection_status,
                    "reference_dimension":
                        REFERENCE_DIM,
                    "selection_rule":
                        "smallest dimension passing all pre-specified stability criteria",
                }
            ]
        )

        save_table(
            selection,
            "12_primary_dimension_selection.csv",
            index=False,
        )

        # -------------------------------------------------------------
        # 12. Figures
        # -------------------------------------------------------------

        write_both(
            report,
            "\n12. SENSITIVITY FIGURES",
        )

        write_both(
            report,
            "-" * 88,
        )

        plot_metric(
            summary,
            "distance_spearman_vs_50PC",
            "Distance Spearman vs 50-PC reference",
            FIGURE_DIR / "01_distance_preservation.png",
            MIN_DISTANCE_SPEARMAN,
        )

        plot_metric(
            summary,
            "mean_knn_overlap_vs_50PC",
            "Mean 30-NN overlap vs 50-PC reference",
            FIGURE_DIR / "02_knn_overlap.png",
            MIN_KNN_OVERLAP,
        )

        plot_metric(
            summary,
            "centroid_distance_spearman_vs_50PC",
            "Centroid geometry Spearman vs 50-PC reference",
            FIGURE_DIR / "03_centroid_geometry.png",
            MIN_CENTROID_DISTANCE_SPEARMAN,
        )

        plot_metric(
            summary,
            "minimum_replicate_geometry_correlation",
            "Minimum replicate trajectory-geometry correlation",
            FIGURE_DIR / "04_replicate_geometry.png",
            MIN_REPLICATE_GEOMETRY_CORRELATION,
        )

        plot_metric(
            summary,
            "minimum_generalization_rho",
            "Minimum cross-replicate pseudotime rho",
            FIGURE_DIR / "05_pseudotime_generalization.png",
            MIN_PSEUDOTIME_GENERALIZATION_RHO,
        )

        plot_metric(
            summary,
            "minimum_balanced_accuracy",
            "Minimum terminal-fate balanced accuracy",
            FIGURE_DIR / "06_terminal_fate_accuracy.png",
            MIN_TERMINAL_FATE_BALANCED_ACCURACY,
        )

        # -------------------------------------------------------------
        # 13. Final status
        # -------------------------------------------------------------

        write_both(
            report,
            "\n13. FINAL STATUS",
        )

        write_both(
            report,
            "-" * 88,
        )

        selected_row = summary[
            summary["dimension"]
            == selected_dim
        ].iloc[0]

        if bool(
            selected_row[
                "all_stability_checks_pass"
            ]
        ):

            write_both(
                report,
                "FINAL STATUS: PASS",
            )

            write_both(
                report,
                f"{selected_dim} PCs is the pre-specified primary "
                "state-space dimensionality for the first GDIS analysis.",
            )

            write_both(
                report,
                "The remaining dimensions should be retained as "
                "sensitivity-analysis state spaces.",
            )

        else:

            write_both(
                report,
                "FINAL STATUS: REFERENCE FALLBACK",
            )

            write_both(
                report,
                "The 50-PC reference should be used provisionally, "
                "with all lower-dimensional spaces retained for sensitivity analysis.",
            )

        write_both(
            report,
            "\nNo GDIS values were calculated.",
        )

        write_both(
            report,
            f"\nReport:  {REPORT_FILE}",
        )

        write_both(
            report,
            f"Tables:  {TABLE_DIR}",
        )

        write_both(
            report,
            f"Figures: {FIGURE_DIR}",
        )

    print("\n" + "=" * 88)
    print("p6_state_space_sensitivity.py completed.")
    print("=" * 88)
    print(f"Report : {REPORT_FILE}")
    print(f"Tables : {TABLE_DIR}")
    print(f"Figures: {FIGURE_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

