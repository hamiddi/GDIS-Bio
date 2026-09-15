#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================
"""
p6b_reference_geometry_validation.py

Reference-geometry validation for the 50-PC GDIS-Bio state space.

Why this script exists
----------------------
p6_state_space_sensitivity.py selected 50 PCs because it was the only
candidate satisfying the pre-specified exact 30-nearest-neighbor overlap
criterion relative to the 50-PC reference.

All other major biological/trajectory metrics were already highly stable
at lower dimensions. Therefore, before using 50 PCs for the first GDIS
analysis, this script validates that the 50-PC reference itself has
well-behaved local geometry.

This script DOES NOT change the p6 selection rule and DOES NOT calculate
GDIS. It asks:

1. Do pairwise distances remain sufficiently dispersed in 50 PCs, or has
   the representation suffered severe distance concentration?
2. Does local k-nearest-neighbor distance scale vary biologically across
   pseudotime/day rather than becoming nearly constant?
3. Is local distance scale reproducible across Differentiation 1 and 2?
4. How smoothly do distance geometry and local neighborhoods converge
   as PCs are added from 5 to 50?
5. Is the 50-PC space safe to use as the primary GDIS reference while
   retaining lower-dimensional spaces for sensitivity analyses?

Input
-----
results/p5_preprocessing_state_space/data/
    GSE114412_endocrine_preprocessed_pca.h5ad

Requirements
------------
numpy
pandas
scipy
matplotlib
scikit-learn
anndata
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import anndata as ad

from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

if SCRIPT_DIR.name == "scripts":
    PROJECT_DIR = SCRIPT_DIR.parent
else:
    PROJECT_DIR = SCRIPT_DIR

P5_DATA_DIR = (
    PROJECT_DIR
    / "results"
    / "p5_preprocessing_state_space"
    / "data"
)

H5AD_FILE = (
    P5_DATA_DIR
    / "GSE114412_endocrine_preprocessed_pca.h5ad"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
    / "p6b_reference_geometry_validation"
)

TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR = RESULTS_DIR / "figures"

REPORT_FILE = (
    RESULTS_DIR
    / "p6b_reference_geometry_validation_report.txt"
)

DIMENSIONS = list(range(5, 51, 5))
REFERENCE_DIM = 50

RANDOM_SEED = 0
PAIRWISE_SAMPLE_SIZE = 2500
K_LOCAL = 30

# Pre-specified geometry-quality criteria.
#
# Pairwise-distance coefficient of variation:
# A value near zero would indicate severe distance concentration.
MIN_DISTANCE_CV = 0.10

# Relative contrast between the 30th-neighbor distance and nearest-neighbor
# distance. Values too close to 1 indicate weak local-distance discrimination.
MIN_NEIGHBOR_CONTRAST = 1.10

# Replicate agreement in median local scale across experimental days.
MIN_REPLICATE_LOCAL_SCALE_CORRELATION = 0.90

# Correlation of local scale with pseudotime is not required to be positive
# or negative; however, local scale must have non-trivial variation.
MIN_LOCAL_SCALE_CV = 0.05


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def write_both(report, text=""):
    print(text)
    report.write(text + "\n")
    report.flush()


def save_table(df, filename, index=True):
    path = TABLE_DIR / filename
    df.to_csv(path, index=index)
    return path


def safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 3:
        return np.nan

    return float(
        spearmanr(
            x[mask],
            y[mask],
        ).statistic
    )


def safe_pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return float(
        np.corrcoef(x, y)[0, 1]
    )


def stratified_sample(obs, n_total, random_seed=0):
    """
    Stratify over branch x differentiation x experimental day.
    """
    rng = np.random.default_rng(random_seed)

    temp = obs.reset_index(drop=True).copy()
    temp["_row_index"] = np.arange(len(temp))

    groups = list(
        temp.groupby(
            [
                "Pseudotime_branch",
                "Differentiation",
                "CellDay",
            ],
            observed=True,
            sort=True,
        )
    )

    n_per = max(
        1,
        int(np.ceil(n_total / len(groups))),
    )

    selected = []

    for _, group in groups:
        ids = group["_row_index"].to_numpy()

        take = min(
            n_per,
            len(ids),
        )

        selected.extend(
            rng.choice(
                ids,
                size=take,
                replace=False,
            ).tolist()
        )

    selected = np.unique(
        np.asarray(selected, dtype=int)
    )

    if len(selected) > n_total:
        selected = np.sort(
            rng.choice(
                selected,
                size=n_total,
                replace=False,
            )
        )

    if len(selected) < n_total:
        remaining = np.setdiff1d(
            np.arange(len(temp)),
            selected,
        )

        extra_n = min(
            n_total - len(selected),
            len(remaining),
        )

        extra = rng.choice(
            remaining,
            size=extra_n,
            replace=False,
        )

        selected = np.sort(
            np.concatenate(
                [selected, extra]
            )
        )

    return selected


def compute_knn_geometry(X, k):
    """
    Calculate nearest-neighbor geometry.

    Returns first-neighbor distance, kth-neighbor distance, and indices.
    """
    model = NearestNeighbors(
        n_neighbors=k + 1,
        metric="euclidean",
        n_jobs=-1,
    )

    model.fit(X)

    distances, indices = model.kneighbors(
        X,
        return_distance=True,
    )

    # Remove self neighbor at position 0.
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    return {
        "nearest_distance": distances[:, 0],
        "kth_distance": distances[:, -1],
        "mean_knn_distance": distances.mean(axis=1),
        "indices": indices,
    }


def neighbor_overlap(indices_a, indices_b):
    """
    Mean overlap between two equal-sized neighbor matrices.
    """
    if indices_a.shape != indices_b.shape:
        raise ValueError(
            "Neighbor matrices must have identical shape."
        )

    k = indices_a.shape[1]

    values = np.empty(
        indices_a.shape[0],
        dtype=float,
    )

    for i in range(indices_a.shape[0]):
        values[i] = (
            len(
                set(indices_a[i]).intersection(
                    indices_b[i]
                )
            )
            / k
        )

    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
    }


def plot_line(df, x, y, ylabel, title, output_file):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    ax.plot(
        df[x],
        df[y],
        marker="o",
        linewidth=1.8,
    )

    ax.set_xlabel("Number of principal components")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(DIMENSIONS)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        output_file,
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

    if not H5AD_FILE.exists():
        raise FileNotFoundError(
            f"Missing p5 AnnData file: {H5AD_FILE}"
        )

    adata = ad.read_h5ad(H5AD_FILE)

    if "X_pca" not in adata.obsm:
        raise ValueError(
            "AnnData does not contain X_pca."
        )

    X50 = np.asarray(
        adata.obsm["X_pca"],
        dtype=np.float32,
    )

    if X50.shape[1] < REFERENCE_DIM:
        raise ValueError(
            f"Expected at least {REFERENCE_DIM} PCs."
        )

    obs = adata.obs.copy()

    required_obs = [
        "Pseudotime_branch",
        "Differentiation",
        "CellDay",
        "Pseudotime_value",
        "Assigned_cluster",
    ]

    missing = [
        c for c in required_obs
        if c not in obs.columns
    ]

    if missing:
        raise ValueError(
            "Missing AnnData observation columns: "
            + ", ".join(missing)
        )

    obs["CellDay"] = pd.to_numeric(
        obs["CellDay"],
        errors="raise",
    )

    obs["Pseudotime_value"] = pd.to_numeric(
        obs["Pseudotime_value"],
        errors="raise",
    )

    # -------------------------------------------------------------
    # Fixed pairwise-distance sample
    # -------------------------------------------------------------

    sample_idx = stratified_sample(
        obs,
        min(
            PAIRWISE_SAMPLE_SIZE,
            adata.n_obs,
        ),
        RANDOM_SEED,
    )

    # -------------------------------------------------------------
    # Reference neighbors
    # -------------------------------------------------------------

    ref_knn = compute_knn_geometry(
        X50[:, :REFERENCE_DIM],
        K_LOCAL,
    )

    # -------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------

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
            "GDIS-Bio 50-PC Reference Geometry Validation",
        )

        write_both(
            report,
            "=" * 88,
        )

        write_both(
            report,
            f"\nCells: {adata.n_obs:,}",
        )

        write_both(
            report,
            f"Available PCA coordinates: "
            f"{X50.shape[1]}",
        )

        write_both(
            report,
            f"Local neighborhood size: "
            f"k={K_LOCAL}",
        )

        write_both(
            report,
            f"Pairwise-distance sample: "
            f"{len(sample_idx):,} cells",
        )

        # =========================================================
        # 1. Distance concentration
        # =========================================================

        write_both(
            report,
            "\n1. PAIRWISE-DISTANCE CONCENTRATION",
        )

        write_both(
            report,
            "-" * 88,
        )

        distance_rows = []

        pairwise_vectors = {}

        for dim in DIMENSIONS:

            X_sample = X50[
                sample_idx,
                :dim,
            ]

            distances = pdist(
                X_sample,
                metric="euclidean",
            )

            pairwise_vectors[dim] = distances

            mean_distance = float(
                np.mean(distances)
            )

            sd_distance = float(
                np.std(distances)
            )

            cv_distance = (
                sd_distance / mean_distance
                if mean_distance > 0
                else np.nan
            )

            q05 = float(
                np.quantile(
                    distances,
                    0.05,
                )
            )

            q95 = float(
                np.quantile(
                    distances,
                    0.95,
                )
            )

            contrast_95_05 = (
                q95 / q05
                if q05 > 0
                else np.nan
            )

            distance_rows.append(
                {
                    "dimension": dim,
                    "mean_pairwise_distance":
                        mean_distance,
                    "sd_pairwise_distance":
                        sd_distance,
                    "pairwise_distance_cv":
                        cv_distance,
                    "q05_pairwise_distance": q05,
                    "q95_pairwise_distance": q95,
                    "q95_q05_distance_ratio":
                        contrast_95_05,
                }
            )

        distance_table = pd.DataFrame(
            distance_rows
        )

        save_table(
            distance_table,
            "01_pairwise_distance_concentration.csv",
            index=False,
        )

        write_both(
            report,
            distance_table.round(6).to_string(
                index=False
            ),
        )

        # =========================================================
        # 2. Adjacent-dimensional distance convergence
        # =========================================================

        write_both(
            report,
            "\n2. ADJACENT-DIMENSION DISTANCE CONVERGENCE",
        )

        write_both(
            report,
            "-" * 88,
        )

        convergence_rows = []

        for i in range(
            len(DIMENSIONS) - 1
        ):

            dim_a = DIMENSIONS[i]
            dim_b = DIMENSIONS[i + 1]

            rho = safe_spearman(
                pairwise_vectors[dim_a],
                pairwise_vectors[dim_b],
            )

            convergence_rows.append(
                {
                    "dimension_from": dim_a,
                    "dimension_to": dim_b,
                    "distance_spearman":
                        rho,
                }
            )

        convergence_table = pd.DataFrame(
            convergence_rows
        )

        save_table(
            convergence_table,
            "02_adjacent_distance_convergence.csv",
            index=False,
        )

        write_both(
            report,
            convergence_table.round(6).to_string(
                index=False
            ),
        )

        # =========================================================
        # 3. Neighbor geometry and adjacent overlap
        # =========================================================

        write_both(
            report,
            "\n3. LOCAL NEIGHBOR GEOMETRY",
        )

        write_both(
            report,
            "-" * 88,
        )

        geometry_rows = []
        neighbor_indices = {}

        for dim in DIMENSIONS:

            geometry = compute_knn_geometry(
                X50[:, :dim],
                K_LOCAL,
            )

            neighbor_indices[dim] = (
                geometry["indices"]
            )

            nearest = (
                geometry["nearest_distance"]
            )

            kth = (
                geometry["kth_distance"]
            )

            local_scale = (
                geometry["mean_knn_distance"]
            )

            nearest_median = float(
                np.median(nearest)
            )

            kth_median = float(
                np.median(kth)
            )

            local_mean = float(
                np.mean(local_scale)
            )

            local_sd = float(
                np.std(local_scale)
            )

            geometry_rows.append(
                {
                    "dimension": dim,
                    "median_nearest_neighbor_distance":
                        nearest_median,
                    "median_30th_neighbor_distance":
                        kth_median,
                    "median_30th_to_first_ratio":
                        (
                            kth_median
                            / nearest_median
                            if nearest_median > 0
                            else np.nan
                        ),
                    "mean_local_scale":
                        local_mean,
                    "local_scale_cv":
                        (
                            local_sd
                            / local_mean
                            if local_mean > 0
                            else np.nan
                        ),
                    "spearman_local_scale_vs_pseudotime":
                        safe_spearman(
                            local_scale,
                            obs[
                                "Pseudotime_value"
                            ].to_numpy(),
                        ),
                }
            )

        geometry_table = pd.DataFrame(
            geometry_rows
        )

        save_table(
            geometry_table,
            "03_local_neighbor_geometry.csv",
            index=False,
        )

        write_both(
            report,
            geometry_table.round(6).to_string(
                index=False
            ),
        )

        adjacent_overlap_rows = []

        for i in range(
            len(DIMENSIONS) - 1
        ):

            dim_a = DIMENSIONS[i]
            dim_b = DIMENSIONS[i + 1]

            overlap = neighbor_overlap(
                neighbor_indices[dim_a],
                neighbor_indices[dim_b],
            )

            adjacent_overlap_rows.append(
                {
                    "dimension_from": dim_a,
                    "dimension_to": dim_b,
                    "mean_neighbor_overlap":
                        overlap["mean"],
                    "median_neighbor_overlap":
                        overlap["median"],
                    "q25_neighbor_overlap":
                        overlap["q25"],
                    "q75_neighbor_overlap":
                        overlap["q75"],
                }
            )

        adjacent_overlap = pd.DataFrame(
            adjacent_overlap_rows
        )

        save_table(
            adjacent_overlap,
            "04_adjacent_neighbor_overlap.csv",
            index=False,
        )

        write_both(
            report,
            "\nAdjacent dimensionality neighbor overlap:\n"
            + adjacent_overlap.round(6).to_string(
                index=False
            ),
        )

        # =========================================================
        # 4. 50-PC local scale by day and replicate
        # =========================================================

        write_both(
            report,
            "\n4. 50-PC LOCAL SCALE BY EXPERIMENTAL DAY AND DIFFERENTIATION",
        )

        write_both(
            report,
            "-" * 88,
        )

        local50 = compute_knn_geometry(
            X50[:, :REFERENCE_DIM],
            K_LOCAL,
        )["mean_knn_distance"]

        local_df = pd.DataFrame(
            {
                "local_scale_50PC": local50,
                "CellDay":
                    obs["CellDay"].to_numpy(),
                "Differentiation":
                    obs["Differentiation"].astype(
                        str
                    ).to_numpy(),
                "Pseudotime_branch":
                    obs["Pseudotime_branch"].astype(
                        str
                    ).to_numpy(),
                "Pseudotime_value":
                    obs[
                        "Pseudotime_value"
                    ].to_numpy(),
            }
        )

        day_rep = (
            local_df.groupby(
                [
                    "CellDay",
                    "Differentiation",
                ],
                observed=True,
            )[
                "local_scale_50PC"
            ]
            .median()
            .reset_index()
        )

        save_table(
            day_rep,
            "05_local_scale_by_day_differentiation.csv",
            index=False,
        )

        write_both(
            report,
            day_rep.round(6).to_string(
                index=False
            ),
        )

        # Cross-replicate correlation of daily local-scale profiles.
        diff_values = sorted(
            day_rep[
                "Differentiation"
            ].unique()
        )

        replicate_local_corr = np.nan

        if len(diff_values) == 2:

            a = (
                day_rep[
                    day_rep["Differentiation"]
                    == diff_values[0]
                ]
                .sort_values("CellDay")
            )

            b = (
                day_rep[
                    day_rep["Differentiation"]
                    == diff_values[1]
                ]
                .sort_values("CellDay")
            )

            replicate_local_corr = (
                safe_pearson(
                    a["local_scale_50PC"],
                    b["local_scale_50PC"],
                )
            )

        write_both(
            report,
            f"\nCross-replicate correlation of "
            f"daily median 50-PC local scale: "
            f"{replicate_local_corr:.6f}",
        )

        # =========================================================
        # 5. Branch-specific replicate local scale
        # =========================================================

        write_both(
            report,
            "\n5. 50-PC LOCAL SCALE REPRODUCIBILITY WITHIN BRANCHES",
        )

        write_both(
            report,
            "-" * 88,
        )

        branch_corr_rows = []

        for branch in sorted(
            local_df[
                "Pseudotime_branch"
            ].unique()
        ):

            subset = local_df[
                local_df[
                    "Pseudotime_branch"
                ] == branch
            ]

            table = (
                subset.groupby(
                    [
                        "CellDay",
                        "Differentiation",
                    ],
                    observed=True,
                )[
                    "local_scale_50PC"
                ]
                .median()
                .reset_index()
            )

            diffs = sorted(
                table[
                    "Differentiation"
                ].unique()
            )

            corr = np.nan

            if len(diffs) == 2:

                a = (
                    table[
                        table[
                            "Differentiation"
                        ] == diffs[0]
                    ]
                    .sort_values(
                        "CellDay"
                    )
                )

                b = (
                    table[
                        table[
                            "Differentiation"
                        ] == diffs[1]
                    ]
                    .sort_values(
                        "CellDay"
                    )
                )

                common_days = sorted(
                    set(
                        a["CellDay"]
                    ).intersection(
                        b["CellDay"]
                    )
                )

                a_vals = [
                    float(
                        a.loc[
                            a["CellDay"] == d,
                            "local_scale_50PC",
                        ].iloc[0]
                    )
                    for d in common_days
                ]

                b_vals = [
                    float(
                        b.loc[
                            b["CellDay"] == d,
                            "local_scale_50PC",
                        ].iloc[0]
                    )
                    for d in common_days
                ]

                corr = safe_pearson(
                    a_vals,
                    b_vals,
                )

            branch_corr_rows.append(
                {
                    "branch": branch,
                    "replicate_daily_local_scale_correlation":
                        corr,
                }
            )

        branch_corr_table = pd.DataFrame(
            branch_corr_rows
        )

        save_table(
            branch_corr_table,
            "06_branch_local_scale_reproducibility.csv",
            index=False,
        )

        write_both(
            report,
            branch_corr_table.round(6).to_string(
                index=False
            ),
        )

        # =========================================================
        # 6. Final reference qualification
        # =========================================================

        write_both(
            report,
            "\n6. FINAL 50-PC REFERENCE QUALIFICATION",
        )

        write_both(
            report,
            "-" * 88,
        )

        row50_dist = distance_table[
            distance_table[
                "dimension"
            ] == REFERENCE_DIM
        ].iloc[0]

        row50_geom = geometry_table[
            geometry_table[
                "dimension"
            ] == REFERENCE_DIM
        ].iloc[0]

        branch_repro_pass = bool(
            (
                branch_corr_table[
                    "replicate_daily_local_scale_correlation"
                ]
                >= MIN_REPLICATE_LOCAL_SCALE_CORRELATION
            ).all()
        )

        checks = [
            (
                f"50-PC pairwise-distance CV >= "
                f"{MIN_DISTANCE_CV}",
                (
                    row50_dist[
                        "pairwise_distance_cv"
                    ]
                    >= MIN_DISTANCE_CV
                ),
            ),
            (
                f"50-PC median 30th/1st neighbor "
                f"distance ratio >= "
                f"{MIN_NEIGHBOR_CONTRAST}",
                (
                    row50_geom[
                        "median_30th_to_first_ratio"
                    ]
                    >= MIN_NEIGHBOR_CONTRAST
                ),
            ),
            (
                f"50-PC local-scale CV >= "
                f"{MIN_LOCAL_SCALE_CV}",
                (
                    row50_geom[
                        "local_scale_cv"
                    ]
                    >= MIN_LOCAL_SCALE_CV
                ),
            ),
            (
                f"Overall daily local-scale replicate "
                f"correlation >= "
                f"{MIN_REPLICATE_LOCAL_SCALE_CORRELATION}",
                (
                    replicate_local_corr
                    >= MIN_REPLICATE_LOCAL_SCALE_CORRELATION
                ),
            ),
            (
                f"Branch-specific daily local-scale "
                f"replicate correlations >= "
                f"{MIN_REPLICATE_LOCAL_SCALE_CORRELATION}",
                branch_repro_pass,
            ),
        ]

        passed = 0

        for label, status in checks:

            if status:
                write_both(
                    report,
                    f"[PASS] {label}",
                )
                passed += 1
            else:
                write_both(
                    report,
                    f"[CHECK] {label}",
                )

        write_both(
            report,
            f"\n50-PC geometry checks passed: "
            f"{passed}/{len(checks)}",
        )

        if passed == len(checks):

            write_both(
                report,
                "\nFINAL STATUS: PASS",
            )

            write_both(
                report,
                "The 50-PC state space retains informative "
                "and reproducible local geometry and is suitable "
                "as the primary reference representation for the "
                "first GDIS-Bio calculation.",
            )

        else:

            write_both(
                report,
                "\nFINAL STATUS: REVIEW REQUIRED",
            )

            write_both(
                report,
                "One or more properties of the 50-PC local geometry "
                "should be reviewed before the first GDIS calculation.",
            )

        write_both(
            report,
            "\nThe p6 dimensionality decision is not modified by this script.",
        )

        write_both(
            report,
            "Lower-dimensional spaces remain mandatory sensitivity analyses.",
        )

        write_both(
            report,
            "\nNo GDIS values were calculated.",
        )

        # =========================================================
        # 7. Figures
        # =========================================================

        plot_line(
            distance_table,
            "dimension",
            "pairwise_distance_cv",
            "Pairwise-distance coefficient of variation",
            "Distance concentration across PCA dimensions",
            FIGURE_DIR
            / "01_pairwise_distance_cv.png",
        )

        plot_line(
            geometry_table,
            "dimension",
            "median_30th_to_first_ratio",
            "Median 30th / 1st neighbor distance",
            "Local distance contrast across PCA dimensions",
            FIGURE_DIR
            / "02_neighbor_distance_contrast.png",
        )

        overlap_plot = adjacent_overlap.copy()
        overlap_plot["dimension"] = (
            overlap_plot["dimension_to"]
        )

        plot_line(
            overlap_plot,
            "dimension",
            "mean_neighbor_overlap",
            "Mean neighbor overlap with previous dimension",
            "Adjacent-dimensional neighborhood convergence",
            FIGURE_DIR
            / "03_adjacent_neighbor_overlap.png",
        )

        # Local-scale by day figure.
        fig, ax = plt.subplots(
            figsize=(9, 5.5)
        )

        for diff in sorted(
            day_rep[
                "Differentiation"
            ].unique()
        ):

            group = (
                day_rep[
                    day_rep[
                        "Differentiation"
                    ] == diff
                ]
                .sort_values(
                    "CellDay"
                )
            )

            ax.plot(
                group["CellDay"],
                group[
                    "local_scale_50PC"
                ],
                marker="o",
                linewidth=1.8,
                label=f"Differentiation {diff}",
            )

        ax.set_xlabel(
            "Experimental day"
        )

        ax.set_ylabel(
            "Median mean 30-NN distance"
        )

        ax.set_title(
            "50-PC local trajectory scale by day"
        )

        ax.set_xticks(
            sorted(
                day_rep[
                    "CellDay"
                ].unique()
            )
        )

        ax.legend()
        ax.grid(alpha=0.25)

        fig.tight_layout()

        fig.savefig(
            FIGURE_DIR
            / "04_50PC_local_scale_by_day.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

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
    print("p6b_reference_geometry_validation.py completed.")
    print("=" * 88)
    print(f"Report : {REPORT_FILE}")
    print(f"Tables : {TABLE_DIR}")
    print(f"Figures: {FIGURE_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

