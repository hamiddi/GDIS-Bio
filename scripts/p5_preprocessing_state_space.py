#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================
"""
p5_preprocessing_state_space.py

Preprocessing and PCA state-space construction for GDIS-Bio.

Dataset
-------
GSE114412 Stage-5 endocrine trajectory

Primary analysis population
---------------------------
18,099 cells from:
GSE114412_Stage_5.endocrine_pseudotime.cell_metadata.tsv.gz

Goals
-----
1. Extract only the validated endocrine-trajectory cells from the
   51,274-cell processed-count matrix without loading the full matrix
   densely into memory.
2. Verify that the deposited processed values are non-negative and
   count-like before applying count-based normalization.
3. Preserve the deposited counts in an AnnData layer.
4. Normalize library size to 10,000 counts per cell.
5. Apply log1p transformation.
6. Select 2,000 highly variable genes with Differentiation as a batch
   key so that batch-specific variability is not favored.
7. Compute a common 50-PC state space across BOTH branches and BOTH
   independent differentiations.
8. Save PCA coordinates, explained variance, loadings, QC summaries,
   state-space tables, figures, and an H5AD object.
9. Do NOT calculate GDIS yet.

Design principle
----------------
The common PCA basis intentionally preserves biological branch signal.
We do NOT batch-correct expression or regress out differentiation here,
because such operations could distort the dynamical trajectory. The
two independent differentiations remain explicit metadata variables
for later replicate validation and sensitivity analysis.

Requirements
------------
numpy
pandas
scipy
matplotlib
anndata
scanpy

IMPORTANT
---------
This script performs preprocessing/state-space construction only.
It does NOT:
- infer a new pseudotime
- alter the deposited branch assignments
- run UMAP
- run clustering
- batch-correct the data
- calculate GDIS
"""

from pathlib import Path
import sys
import platform

import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt

import anndata as ad
import scanpy as sc


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

if SCRIPT_DIR.name == "scripts":
    PROJECT_DIR = SCRIPT_DIR.parent
else:
    PROJECT_DIR = SCRIPT_DIR

RAW_DIR = PROJECT_DIR / "raw_data"
RESULTS_DIR = PROJECT_DIR / "results" / "p5_preprocessing_state_space"
TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR = RESULTS_DIR / "figures"
DATA_DIR = RESULTS_DIR / "data"

COUNTS_FILE = RAW_DIR / "GSE114412_Stage_5.all.processed_counts.tsv.gz"
PSEUDOTIME_FILE = (
    RAW_DIR / "GSE114412_Stage_5.endocrine_pseudotime.cell_metadata.tsv.gz"
)

REPORT_FILE = RESULTS_DIR / "p5_preprocessing_state_space_report.txt"
H5AD_FILE = DATA_DIR / "GSE114412_endocrine_preprocessed_pca.h5ad"

COUNT_BARCODE_COLUMN = "# library.barcode"
META_BARCODE_COLUMN = "library.barcode"

N_TOP_HVG = 2000
N_PCS = 50
NORMALIZATION_TARGET = 1e4
CHUNK_SIZE = 500
RANDOM_SEED = 0

# We require deposited values to be overwhelmingly integer-like before
# applying count-based normalization.
MIN_INTEGER_LIKE_FRACTION = 0.999

# Convenience state-space exports for later sensitivity testing.
STATE_SPACE_DIMS = [5, 10, 20, 30, 50]


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def write_both(report, text=""):
    """Print and write the same text to the report."""
    print(text)
    report.write(text + "\n")
    report.flush()


def save_table(df, filename, index=True):
    """Save a dataframe to the table directory."""
    path = TABLE_DIR / filename
    df.to_csv(path, index=index)
    return path


def ensure_columns(df, columns, name):
    """Verify required columns."""
    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"{name} is missing required columns: "
            + ", ".join(missing)
        )


def safe_percentile(values, q):
    """Return percentile or NaN for empty input."""
    values = np.asarray(values)

    if values.size == 0:
        return np.nan

    return float(np.percentile(values, q))


def summarize_series(series, label):
    """Create one summary row for a numeric series."""
    values = pd.to_numeric(series, errors="coerce").dropna().values

    return {
        "metric": label,
        "n": int(len(values)),
        "min": float(np.min(values)) if len(values) else np.nan,
        "q25": safe_percentile(values, 25),
        "median": safe_percentile(values, 50),
        "mean": float(np.mean(values)) if len(values) else np.nan,
        "q75": safe_percentile(values, 75),
        "max": float(np.max(values)) if len(values) else np.nan,
    }


def extract_trajectory_counts(
    counts_file,
    selected_barcodes,
    report,
):
    """
    Stream through the full processed-count matrix and retain only the
    validated endocrine-trajectory cells.

    Returns
    -------
    X : scipy.sparse.csr_matrix
        Selected cell x gene count matrix.
    barcodes : list[str]
        Cell IDs in matrix row order.
    gene_names : list[str]
        Gene-column names.
    count_diagnostics : dict
        Non-negativity and integer-likeness diagnostics.
    """

    header = pd.read_csv(
        counts_file,
        sep="\t",
        compression="gzip",
        nrows=0,
    )

    columns = list(header.columns)

    if not columns or columns[0] != COUNT_BARCODE_COLUMN:
        raise ValueError(
            f"Expected first count-matrix column "
            f"'{COUNT_BARCODE_COLUMN}', found "
            f"'{columns[0] if columns else 'NONE'}'."
        )

    gene_names = columns[1:]
    selected_set = set(map(str, selected_barcodes))

    sparse_blocks = []
    retained_barcodes = []

    total_rows_seen = 0
    selected_rows_seen = 0
    nonzero_values = 0
    integer_like_values = 0
    negative_values = 0

    write_both(
        report,
        f"Streaming processed-count matrix in chunks of "
        f"{CHUNK_SIZE:,} rows..."
    )

    reader = pd.read_csv(
        counts_file,
        sep="\t",
        compression="gzip",
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(reader, start=1):

        total_rows_seen += len(chunk)

        barcode_values = chunk.iloc[:, 0].astype(str)
        mask = barcode_values.isin(selected_set)

        if not mask.any():
            continue

        selected = chunk.loc[mask]

        retained_barcodes.extend(
            selected.iloc[:, 0].astype(str).tolist()
        )

        numeric = selected.iloc[:, 1:].to_numpy(
            dtype=np.float32,
            copy=False,
        )

        selected_rows_seen += numeric.shape[0]

        if numeric.size:

            negative_values += int(
                np.count_nonzero(numeric < 0)
            )

            nz = numeric[numeric != 0]

            nonzero_values += int(nz.size)

            if nz.size:
                integer_like_values += int(
                    np.count_nonzero(
                        np.isclose(
                            nz,
                            np.rint(nz),
                            atol=1e-6,
                            rtol=0.0,
                        )
                    )
                )

        sparse_blocks.append(
            sparse.csr_matrix(numeric)
        )

        if chunk_number % 10 == 0:
            write_both(
                report,
                f"  chunks processed: {chunk_number:4d} | "
                f"matrix rows seen: {total_rows_seen:6,d} | "
                f"trajectory cells retained: "
                f"{selected_rows_seen:6,d}"
            )

    if not sparse_blocks:
        raise RuntimeError(
            "No trajectory cells were found in the count matrix."
        )

    X = sparse.vstack(
        sparse_blocks,
        format="csr",
        dtype=np.float32,
    )

    integer_like_fraction = (
        integer_like_values / nonzero_values
        if nonzero_values > 0
        else np.nan
    )

    diagnostics = {
        "total_matrix_rows_seen": total_rows_seen,
        "selected_rows_seen": selected_rows_seen,
        "n_genes": len(gene_names),
        "nonzero_values": nonzero_values,
        "integer_like_nonzero_values": integer_like_values,
        "integer_like_fraction": integer_like_fraction,
        "negative_values": negative_values,
    }

    return X, retained_barcodes, gene_names, diagnostics


def run_hvg(adata):
    """
    Batch-aware HVG selection.

    'seurat' flavor is used after normalize_total + log1p.
    Differentiation is used as batch_key to avoid favoring genes whose
    variability is specific to only one independent differentiation.
    """
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=N_TOP_HVG,
        flavor="seurat",
        batch_key="Differentiation",
        subset=False,
        inplace=True,
    )


def run_pca_compat(adata):
    """
    Compute PCA with compatibility across Scanpy API versions.

    Newer Scanpy versions use mask_var. Older releases used
    use_highly_variable. We prefer the modern mask_var API and fall
    back only if needed.
    """
    try:
        sc.pp.pca(
            adata,
            n_comps=N_PCS,
            mask_var="highly_variable",
            svd_solver="arpack",
            random_state=RANDOM_SEED,
        )
        return "sc.pp.pca(mask_var='highly_variable', random_state=0)"

    except TypeError:
        try:
            sc.pp.pca(
                adata,
                n_comps=N_PCS,
                mask_var="highly_variable",
                svd_solver="arpack",
                rng=RANDOM_SEED,
            )
            return "sc.pp.pca(mask_var='highly_variable', rng=0)"

        except TypeError:
            sc.tl.pca(
                adata,
                n_comps=N_PCS,
                use_highly_variable=True,
                svd_solver="arpack",
                random_state=RANDOM_SEED,
            )
            return "sc.tl.pca(use_highly_variable=True, random_state=0)"


def plot_explained_variance(variance_ratio, output_path):
    """Plot per-PC and cumulative explained variance."""
    pcs = np.arange(1, len(variance_ratio) + 1)
    cumulative = np.cumsum(variance_ratio)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(
        pcs,
        cumulative,
        marker="o",
        markersize=3,
        linewidth=1.5,
    )

    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance ratio")
    ax.set_title("PCA cumulative explained variance")
    ax.set_xlim(1, len(variance_ratio))
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pca_numeric(
    coords,
    values,
    xlabel,
    title,
    output_path,
):
    """PC1/PC2 scatter for a numeric metadata variable."""
    fig, ax = plt.subplots(figsize=(7.5, 6))

    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=np.asarray(values, dtype=float),
        s=5,
        alpha=0.55,
    )

    ax.set_xlabel(xlabel[0])
    ax.set_ylabel(xlabel[1])
    ax.set_title(title)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(title.split(":")[-1].strip())

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pca_categorical(
    coords,
    categories,
    xlabel,
    title,
    output_path,
):
    """PC1/PC2 scatter for categorical metadata."""
    categories = pd.Series(categories).astype(str)
    levels = sorted(categories.unique())

    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    for level in levels:

        mask = categories.values == level

        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=5,
            alpha=0.50,
            label=level,
        )

    ax.set_xlabel(xlabel[0])
    ax.set_ylabel(xlabel[1])
    ax.set_title(title)
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def top_loading_table(adata, n_pcs=10, top_n=20):
    """Return top absolute gene loadings for the first PCs."""
    loadings = np.asarray(adata.varm["PCs"])
    genes = np.asarray(adata.var_names)

    rows = []

    max_pc = min(n_pcs, loadings.shape[1])

    for pc_idx in range(max_pc):

        values = loadings[:, pc_idx]
        order = np.argsort(np.abs(values))[::-1][:top_n]

        for rank, gene_idx in enumerate(order, start=1):
            rows.append(
                {
                    "PC": pc_idx + 1,
                    "rank": rank,
                    "gene": genes[gene_idx],
                    "loading": float(values[gene_idx]),
                    "absolute_loading": float(
                        abs(values[gene_idx])
                    ),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(REPORT_FILE, "w", encoding="utf-8") as report:

        write_both(report, "=" * 86)
        write_both(report, "GDIS-Bio Preprocessing and State-Space Construction")
        write_both(report, "Dataset: GSE114412 Stage-5 endocrine trajectory")
        write_both(report, "=" * 86)

        # -------------------------------------------------------------
        # 1. Software / reproducibility information
        # -------------------------------------------------------------

        write_both(report, "\n1. SOFTWARE AND PARAMETERS")
        write_both(report, "-" * 86)

        write_both(report, f"Python: {platform.python_version()}")
        write_both(report, f"NumPy: {np.__version__}")
        write_both(report, f"Pandas: {pd.__version__}")
        write_both(report, f"Scanpy: {sc.__version__}")
        write_both(report, f"AnnData: {ad.__version__}")
        write_both(report, f"Random seed: {RANDOM_SEED}")
        write_both(report, f"Normalization target: {NORMALIZATION_TARGET:.0f}")
        write_both(report, f"Top HVGs: {N_TOP_HVG}")
        write_both(report, f"PCA components: {N_PCS}")
        write_both(
            report,
            "HVG batch key: Differentiation"
        )

        # -------------------------------------------------------------
        # 2. Load validated endocrine metadata
        # -------------------------------------------------------------

        write_both(report, "\n2. VALIDATED ENDOCRINE TRAJECTORY")
        write_both(report, "-" * 86)

        meta = pd.read_csv(
            PSEUDOTIME_FILE,
            sep="\t",
            compression="gzip",
            low_memory=False,
        )

        ensure_columns(
            meta,
            [
                META_BARCODE_COLUMN,
                "Assigned_cluster",
                "Pseudotime_value",
                "Pseudotime_branch",
                "Differentiation",
                "CellDay",
                "Lib_prep_batch",
            ],
            "Endocrine pseudotime metadata",
        )

        if meta[META_BARCODE_COLUMN].duplicated().any():
            raise ValueError(
                "Duplicate cell barcodes detected in trajectory metadata."
            )

        meta[META_BARCODE_COLUMN] = (
            meta[META_BARCODE_COLUMN].astype(str)
        )

        write_both(
            report,
            f"Validated trajectory cells requested: {len(meta):,}"
        )

        write_both(
            report,
            f"Branches: "
            f"{sorted(meta['Pseudotime_branch'].unique().tolist())}"
        )

        write_both(
            report,
            f"Differentiations: "
            f"{sorted(meta['Differentiation'].unique().tolist())}"
        )

        write_both(
            report,
            f"Experimental days: "
            f"{sorted(meta['CellDay'].unique().tolist())}"
        )

        # -------------------------------------------------------------
        # 3. Stream-extract trajectory expression matrix
        # -------------------------------------------------------------

        write_both(
            report,
            "\n3. TRAJECTORY CELL EXTRACTION FROM EXPRESSION MATRIX"
        )
        write_both(report, "-" * 86)

        (
            X_counts,
            matrix_barcodes,
            gene_names,
            diagnostics,
        ) = extract_trajectory_counts(
            COUNTS_FILE,
            meta[META_BARCODE_COLUMN],
            report,
        )

        write_both(
            report,
            f"\nFull count-matrix rows scanned: "
            f"{diagnostics['total_matrix_rows_seen']:,}"
        )

        write_both(
            report,
            f"Trajectory rows retained: "
            f"{diagnostics['selected_rows_seen']:,}"
        )

        write_both(
            report,
            f"Genes in deposited matrix: "
            f"{diagnostics['n_genes']:,}"
        )

        write_both(
            report,
            f"Sparse selected matrix shape: "
            f"{X_counts.shape[0]:,} x {X_counts.shape[1]:,}"
        )

        write_both(
            report,
            f"Selected matrix nonzero values: "
            f"{X_counts.nnz:,}"
        )

        density = (
            X_counts.nnz
            / (X_counts.shape[0] * X_counts.shape[1])
        )

        write_both(
            report,
            f"Selected matrix density: {density:.6f}"
        )

        # Confirm exact cell recovery.
        requested_set = set(meta[META_BARCODE_COLUMN])
        recovered_set = set(matrix_barcodes)

        missing_cells = requested_set - recovered_set
        extra_cells = recovered_set - requested_set

        write_both(
            report,
            f"Missing requested trajectory cells: "
            f"{len(missing_cells):,}"
        )

        write_both(
            report,
            f"Unexpected selected cells: "
            f"{len(extra_cells):,}"
        )

        if missing_cells or extra_cells:
            raise RuntimeError(
                "Trajectory-cell extraction did not exactly match "
                "the validated cell set."
            )

        # -------------------------------------------------------------
        # 4. Count-value qualification
        # -------------------------------------------------------------

        write_both(report, "\n4. DEPOSITED VALUE QUALIFICATION")
        write_both(report, "-" * 86)

        integer_fraction = diagnostics[
            "integer_like_fraction"
        ]

        write_both(
            report,
            f"Negative values: "
            f"{diagnostics['negative_values']:,}"
        )

        write_both(
            report,
            f"Nonzero values checked: "
            f"{diagnostics['nonzero_values']:,}"
        )

        write_both(
            report,
            f"Integer-like nonzero fraction: "
            f"{integer_fraction:.8f}"
        )

        count_like_pass = (
            diagnostics["negative_values"] == 0
            and np.isfinite(integer_fraction)
            and integer_fraction >= MIN_INTEGER_LIKE_FRACTION
        )

        if count_like_pass:
            write_both(
                report,
                "[PASS] Deposited processed values are sufficiently "
                "count-like for count-based normalization."
            )
        else:
            write_both(
                report,
                "[REVIEW REQUIRED] Deposited processed values do not "
                "satisfy the pre-specified count-like criterion."
            )

            write_both(
                report,
                "\nPreprocessing stopped before normalization."
            )

            return 2

        # -------------------------------------------------------------
        # 5. Build AnnData and align metadata
        # -------------------------------------------------------------

        write_both(report, "\n5. ANNDATA CONSTRUCTION")
        write_both(report, "-" * 86)

        meta_indexed = meta.set_index(
            META_BARCODE_COLUMN,
            drop=False,
        )

        obs = meta_indexed.loc[
            matrix_barcodes
        ].copy()

        # Keep useful metadata as explicit types.
        obs["Differentiation"] = (
            obs["Differentiation"]
            .astype(str)
            .astype("category")
        )

        obs["Pseudotime_branch"] = (
            obs["Pseudotime_branch"]
            .astype(str)
            .astype("category")
        )

        obs["Assigned_cluster"] = (
            obs["Assigned_cluster"]
            .astype(str)
            .astype("category")
        )

        obs["CellDay"] = pd.to_numeric(
            obs["CellDay"],
            errors="raise",
        ).astype(int)

        obs["Pseudotime_value"] = pd.to_numeric(
            obs["Pseudotime_value"],
            errors="raise",
        )

        var = pd.DataFrame(
            index=pd.Index(
                gene_names,
                name="gene",
            )
        )

        adata = ad.AnnData(
            X=X_counts,
            obs=obs,
            var=var,
            dtype=np.float32,
        )

        adata.obs_names = pd.Index(
            matrix_barcodes,
            dtype=str,
        )

        adata.var_names_make_unique()

        write_both(
            report,
            f"AnnData created: "
            f"{adata.n_obs:,} cells x {adata.n_vars:,} genes"
        )

        # -------------------------------------------------------------
        # 6. Objective QC summaries
        # -------------------------------------------------------------

        write_both(report, "\n6. QC SUMMARIES")
        write_both(report, "-" * 86)

        adata.var["mt"] = (
            adata.var_names
            .str.upper()
            .str.startswith("MT-")
        )

        sc.pp.calculate_qc_metrics(
            adata,
            qc_vars=["mt"],
            percent_top=None,
            log1p=False,
            inplace=True,
        )

        qc_summary = pd.DataFrame(
            [
                summarize_series(
                    adata.obs["total_counts"],
                    "total_counts",
                ),
                summarize_series(
                    adata.obs["n_genes_by_counts"],
                    "n_genes_by_counts",
                ),
                summarize_series(
                    adata.obs["pct_counts_mt"],
                    "pct_counts_mt",
                ),
            ]
        )

        save_table(
            qc_summary,
            "01_qc_summary.csv",
            index=False,
        )

        write_both(
            report,
            qc_summary.round(4).to_string(index=False)
        )

        qc_by_day = (
            adata.obs.groupby(
                "CellDay",
                observed=True,
            )[
                [
                    "total_counts",
                    "n_genes_by_counts",
                    "pct_counts_mt",
                ]
            ]
            .median()
        )

        save_table(
            qc_by_day,
            "02_qc_medians_by_day.csv",
            index=True,
        )

        qc_by_diff = (
            adata.obs.groupby(
                "Differentiation",
                observed=True,
            )[
                [
                    "total_counts",
                    "n_genes_by_counts",
                    "pct_counts_mt",
                ]
            ]
            .median()
        )

        save_table(
            qc_by_diff,
            "03_qc_medians_by_differentiation.csv",
            index=True,
        )

        # -------------------------------------------------------------
        # 7. Remove genes absent from the endocrine subset
        # -------------------------------------------------------------

        write_both(
            report,
            "\n7. REMOVE ZERO-EXPRESSION GENES IN ENDOCRINE SUBSET"
        )
        write_both(report, "-" * 86)

        genes_before = adata.n_vars

        sc.pp.filter_genes(
            adata,
            min_cells=1,
        )

        genes_after = adata.n_vars

        write_both(
            report,
            f"Genes before zero-expression removal: "
            f"{genes_before:,}"
        )

        write_both(
            report,
            f"Genes retained (expressed in >=1 trajectory cell): "
            f"{genes_after:,}"
        )

        write_both(
            report,
            f"Genes removed because they are absent from the "
            f"endocrine subset: {genes_before - genes_after:,}"
        )

        # Preserve deposited counts.
        adata.layers["counts"] = adata.X.copy()

        # -------------------------------------------------------------
        # 8. Normalization and log transformation
        # -------------------------------------------------------------

        write_both(
            report,
            "\n8. NORMALIZATION AND LOG1P TRANSFORMATION"
        )
        write_both(report, "-" * 86)

        sc.pp.normalize_total(
            adata,
            target_sum=NORMALIZATION_TARGET,
        )

        sc.pp.log1p(adata)

        write_both(
            report,
            f"Applied per-cell library-size normalization to "
            f"{NORMALIZATION_TARGET:.0f}, followed by log1p."
        )

        # -------------------------------------------------------------
        # 9. Batch-aware highly variable genes
        # -------------------------------------------------------------

        write_both(
            report,
            "\n9. HIGHLY VARIABLE GENE SELECTION"
        )
        write_both(report, "-" * 86)

        run_hvg(adata)

        n_hvg = int(
            adata.var["highly_variable"].sum()
        )

        write_both(
            report,
            f"Highly variable genes selected: {n_hvg:,}"
        )

        hvg_columns = [
            col
            for col in [
                "highly_variable",
                "means",
                "dispersions",
                "dispersions_norm",
                "highly_variable_nbatches",
                "highly_variable_intersection",
            ]
            if col in adata.var.columns
        ]

        hvg_table = (
            adata.var[hvg_columns]
            .copy()
            .reset_index()
        )

        if "highly_variable_nbatches" in hvg_table.columns:
            hvg_table = hvg_table.sort_values(
                [
                    "highly_variable",
                    "highly_variable_nbatches",
                ],
                ascending=[False, False],
            )
        else:
            hvg_table = hvg_table.sort_values(
                "highly_variable",
                ascending=False,
            )

        save_table(
            hvg_table,
            "04_highly_variable_genes.csv",
            index=False,
        )

        # -------------------------------------------------------------
        # 10. PCA common state space
        # -------------------------------------------------------------

        write_both(
            report,
            "\n10. COMMON PCA STATE SPACE"
        )
        write_both(report, "-" * 86)

        pca_method = run_pca_compat(adata)

        coords = np.asarray(
            adata.obsm["X_pca"],
            dtype=np.float32,
        )

        variance_ratio = np.asarray(
            adata.uns["pca"]["variance_ratio"],
            dtype=float,
        )

        variance = np.asarray(
            adata.uns["pca"]["variance"],
            dtype=float,
        )

        write_both(
            report,
            f"PCA implementation: {pca_method}"
        )

        write_both(
            report,
            f"PCA coordinate matrix: "
            f"{coords.shape[0]:,} cells x "
            f"{coords.shape[1]:,} PCs"
        )

        variance_table = pd.DataFrame(
            {
                "PC": np.arange(
                    1,
                    len(variance_ratio) + 1,
                ),
                "variance": variance,
                "variance_ratio": variance_ratio,
                "cumulative_variance_ratio":
                    np.cumsum(variance_ratio),
            }
        )

        save_table(
            variance_table,
            "05_pca_explained_variance.csv",
            index=False,
        )

        cumulative = np.cumsum(
            variance_ratio
        )

        for k in [5, 10, 20, 30, 50]:
            if k <= len(cumulative):
                write_both(
                    report,
                    f"Cumulative explained variance, "
                    f"first {k:2d} PCs: "
                    f"{cumulative[k - 1]:.6f}"
                )

        plot_explained_variance(
            variance_ratio,
            FIGURE_DIR / "01_pca_cumulative_explained_variance.png",
        )

        # -------------------------------------------------------------
        # 11. State-space exports
        # -------------------------------------------------------------

        write_both(
            report,
            "\n11. STATE-SPACE EXPORTS"
        )
        write_both(report, "-" * 86)

        meta_export = adata.obs[
            [
                META_BARCODE_COLUMN,
                "Assigned_cluster",
                "Pseudotime_value",
                "Pseudotime_branch",
                "Differentiation",
                "CellDay",
                "Lib_prep_batch",
            ]
        ].copy()

        for k in STATE_SPACE_DIMS:

            if k > coords.shape[1]:
                continue

            pc_df = pd.DataFrame(
                coords[:, :k],
                index=adata.obs_names,
                columns=[
                    f"PC{i}"
                    for i in range(1, k + 1)
                ],
            )

            export = pd.concat(
                [
                    meta_export,
                    pc_df,
                ],
                axis=1,
            )

            output_file = (
                DATA_DIR
                / f"state_space_pca_{k:02d}.csv.gz"
            )

            export.to_csv(
                output_file,
                index=False,
                compression="gzip",
            )

            write_both(
                report,
                f"Saved {k:2d}-PC state space: "
                f"{output_file.name}"
            )

        # -------------------------------------------------------------
        # 12. PCA loading interpretation
        # -------------------------------------------------------------

        write_both(
            report,
            "\n12. PCA GENE LOADINGS"
        )
        write_both(report, "-" * 86)

        loading_table = top_loading_table(
            adata,
            n_pcs=10,
            top_n=20,
        )

        save_table(
            loading_table,
            "06_top_gene_loadings_first10PCs.csv",
            index=False,
        )

        write_both(
            report,
            "Saved top 20 absolute gene loadings for PCs 1-10."
        )

        # -------------------------------------------------------------
        # 13. PCA descriptive figures
        # -------------------------------------------------------------

        write_both(
            report,
            "\n13. PCA DESCRIPTIVE FIGURES"
        )
        write_both(report, "-" * 86)

        labels = (
            "PC1",
            "PC2",
        )

        plot_pca_numeric(
            coords,
            adata.obs["CellDay"],
            labels,
            "PCA state space: experimental day",
            FIGURE_DIR / "02_pca_pc1_pc2_by_day.png",
        )

        plot_pca_numeric(
            coords,
            adata.obs["Pseudotime_value"],
            labels,
            "PCA state space: deposited pseudotime",
            FIGURE_DIR / "03_pca_pc1_pc2_by_pseudotime.png",
        )

        plot_pca_categorical(
            coords,
            adata.obs["Assigned_cluster"],
            labels,
            "PCA state space: biological state",
            FIGURE_DIR / "04_pca_pc1_pc2_by_cluster.png",
        )

        plot_pca_categorical(
            coords,
            adata.obs["Pseudotime_branch"],
            labels,
            "PCA state space: trajectory branch",
            FIGURE_DIR / "05_pca_pc1_pc2_by_branch.png",
        )

        plot_pca_categorical(
            coords,
            adata.obs["Differentiation"],
            labels,
            "PCA state space: independent differentiation",
            FIGURE_DIR / "06_pca_pc1_pc2_by_differentiation.png",
        )

        write_both(
            report,
            "Saved PC1-PC2 visualizations by day, pseudotime, "
            "biological state, branch, and differentiation."
        )

        # -------------------------------------------------------------
        # 14. Save AnnData
        # -------------------------------------------------------------

        write_both(
            report,
            "\n14. SAVE PREPROCESSED ANNDATA"
        )
        write_both(report, "-" * 86)

        # Record preprocessing parameters inside the object.
        adata.uns["gdis_bio_preprocessing"] = {
            "normalization_target": float(
                NORMALIZATION_TARGET
            ),
            "hvg_method": "seurat",
            "hvg_n_top_genes": int(N_TOP_HVG),
            "hvg_batch_key": "Differentiation",
            "pca_n_components": int(N_PCS),
            "random_seed": int(RANDOM_SEED),
            "count_like_threshold": float(
                MIN_INTEGER_LIKE_FRACTION
            ),
            "primary_population":
                "validated 18,099-cell endocrine trajectory",
            "batch_correction_applied": False,
            "gdis_calculated": False,
        }

        adata.write_h5ad(
            H5AD_FILE,
            compression="gzip",
        )

        write_both(
            report,
            f"Saved: {H5AD_FILE}"
        )

        # -------------------------------------------------------------
        # 15. Final checks
        # -------------------------------------------------------------

        write_both(
            report,
            "\n15. FINAL STATE-SPACE QUALIFICATION"
        )
        write_both(report, "-" * 86)

        checks = [
            (
                "Exactly 18,099 validated endocrine cells retained",
                adata.n_obs == 18099,
            ),
            (
                "All requested trajectory cells recovered exactly",
                len(missing_cells) == 0
                and len(extra_cells) == 0,
            ),
            (
                "No negative deposited values",
                diagnostics["negative_values"] == 0,
            ),
            (
                f"Integer-like nonzero fraction >= "
                f"{MIN_INTEGER_LIKE_FRACTION}",
                count_like_pass,
            ),
            (
                f"Exactly {N_TOP_HVG} HVGs selected",
                n_hvg == N_TOP_HVG,
            ),
            (
                f"{N_PCS} PCA coordinates computed",
                coords.shape == (adata.n_obs, N_PCS),
            ),
            (
                "PCA coordinates contain only finite values",
                bool(np.isfinite(coords).all()),
            ),
            (
                "AnnData output written",
                H5AD_FILE.exists(),
            ),
        ]

        passed = 0

        for label, status in checks:

            if status:
                write_both(
                    report,
                    f"[PASS] {label}"
                )
                passed += 1
            else:
                write_both(
                    report,
                    f"[CHECK] {label}"
                )

        write_both(
            report,
            f"\nState-space qualification checks passed: "
            f"{passed}/{len(checks)}"
        )

        if passed == len(checks):

            write_both(
                report,
                "\nFINAL STATUS: PASS"
            )

            write_both(
                report,
                "The validated endocrine trajectory has been "
                "successfully transformed into a common PCA state space."
            )

            write_both(
                report,
                "No GDIS calculation was performed."
            )

            write_both(
                report,
                "The next stage should evaluate state-space dimensionality "
                "and then calculate GDIS using pre-specified trajectory "
                "orderings and sensitivity analyses."
            )

        else:

            write_both(
                report,
                "\nFINAL STATUS: REVIEW REQUIRED"
            )

        write_both(
            report,
            f"\nReport:  {REPORT_FILE}"
        )

        write_both(
            report,
            f"Tables:  {TABLE_DIR}"
        )

        write_both(
            report,
            f"Figures: {FIGURE_DIR}"
        )

        write_both(
            report,
            f"Data:    {DATA_DIR}"
        )

    print("\n" + "=" * 86)
    print("p5_preprocessing_state_space.py completed.")
    print("=" * 86)
    print(f"Report : {REPORT_FILE}")
    print(f"Data   : {DATA_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

