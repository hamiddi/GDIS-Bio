#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================
"""
p15_GSE175634_preprocessing_state_space.py

Expression preprocessing and common state-space construction for the
GDIS-Bio external-validation dataset GSE175634.

This stage uses ONLY the cohorts and cell-inclusion rules frozen in p14.

Frozen external-validation design
---------------------------------
Primary:
    shared developmental backbone
    IPSC -> MES -> CMES -> PROG

Secondary:
    shared backbone -> CM
    shared backbone -> CF

Common state-space population
-----------------------------
The PCA state space is built ONCE from the union of all cells included in
ANY frozen p14 analysis:

    include_shared_backbone_primary
    OR include_cm_extension_secondary
    OR include_cf_extension_secondary

This ensures that the primary and both terminal-extension analyses use the
same expression coordinate system. No branch-specific PCA is allowed.

Preprocessing mirrors the discovery analysis
--------------------------------------------
1. Raw uncorrected UMI counts
2. No cell filtering at this stage
3. Remove genes detected in zero selected cells (min_cells = 1)
4. Library-size normalization to 10,000 counts/cell
5. log1p transformation
6. 2,000 highly variable genes using Scanpy "seurat" flavor
7. Batch-aware HVG selection using individual as batch_key
8. NO batch correction
9. Scale selected HVGs
10. PCA to 50 components
11. Export common 5/10/20/30/50-PC state spaces

IMPORTANT
---------
This script does NOT calculate GDIS.
It does NOT change the frozen p14 cohorts or biological landmarks.
It does NOT use experimental day to order cells.
It does NOT reconstruct pseudotime.
It does NOT perform batch correction.

Memory note
-----------
The compressed Matrix Market file is ~970 MiB and contains >300 million
nonzero entries. Loading/converting the sparse matrix can temporarily require
tens of GB of RAM. Run this script on the HPC system, not a laptop.

Requirements
------------
numpy
pandas
scipy
scanpy
anndata
matplotlib
"""

from pathlib import Path
import gc
import gzip
import platform
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import scipy
    import scipy.sparse as sp
    from scipy.io import mmread
except ImportError as exc:
    raise SystemExit(
        "ERROR: scipy is required.\n"
        "Install with: python -m pip install scipy"
    ) from exc

try:
    import scanpy as sc
    import anndata as ad
except ImportError as exc:
    raise SystemExit(
        "ERROR: scanpy/anndata are required.\n"
        "Install with: python -m pip install scanpy anndata"
    ) from exc


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR

RAW_DIR = (
    PROJECT_DIR
    / "raw_data"
    / "GSE175634"
)

COUNT_FILE = (
    RAW_DIR
    / "GSE175634_cell_counts.mtx.gz"
)

CELL_INDEX_FILE = (
    RAW_DIR
    / "GSE175634_cell_indices.tsv.gz"
)

GENE_INDEX_FILE = (
    RAW_DIR
    / "GSE175634_gene_indices_counts.tsv.gz"
)

P14_DIR = (
    PROJECT_DIR
    / "results"
    / "p14_external_GSE175634_design_freeze"
)

P14_TABLE_DIR = (
    P14_DIR
    / "tables"
)

FROZEN_CELL_FILE = (
    P14_TABLE_DIR
    / "02_frozen_cell_inclusion_flags.csv.gz"
)

FROZEN_TRANSITION_FILE = (
    P14_TABLE_DIR
    / "05_prespecified_biological_transitions.csv"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
    / "p15_external_GSE175634_preprocessing_state_space"
)

DATA_DIR = (
    RESULTS_DIR
    / "data"
)

TABLE_DIR = (
    RESULTS_DIR
    / "tables"
)

FIGURE_DIR = (
    RESULTS_DIR
    / "figures"
)

REPORT_FILE = (
    RESULTS_DIR
    / "p15_external_GSE175634_preprocessing_state_space_report.txt"
)


# ---------------------------------------------------------------------
# Frozen preprocessing parameters
# ---------------------------------------------------------------------

NORMALIZATION_TARGET = 10_000.0
MIN_GENE_CELLS = 1
N_HVG = 2_000
HVG_FLAVOR = "seurat"
HVG_BATCH_KEY = "individual"
N_PCS = 50
PCA_EXPORT_DIMS = [5, 10, 20, 30, 50]
PCA_SOLVER = "arpack"
PCA_RANDOM_STATE = 20260913
SCALE_MAX_VALUE = 10.0

INCLUSION_COLUMNS = [
    "include_shared_backbone_primary",
    "include_cm_extension_secondary",
    "include_cf_extension_secondary",
]


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


def human_bytes(value):
    value = float(value)

    for unit in [
        "B",
        "KiB",
        "MiB",
        "GiB",
        "TiB",
    ]:
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0


def robust_bool(series):
    """
    Convert common CSV boolean representations to bool.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    values = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }

    converted = values.map(
        mapping
    )

    if converted.isna().any():
        bad = sorted(
            values[
                converted.isna()
            ].unique().tolist()
        )

        raise ValueError(
            "Could not parse boolean values: "
            + str(bad)
        )

    return converted.astype(bool)


def load_index_table(
    path,
    index_column,
    name_column,
):
    """
    Read and validate one GEO index table.
    """
    df = pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )

    required = [
        index_column,
        name_column,
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path.name} missing columns: "
            + ", ".join(missing)
        )

    df[
        index_column
    ] = pd.to_numeric(
        df[
            index_column
        ],
        errors="raise",
    ).astype(
        np.int64
    )

    df[
        name_column
    ] = df[
        name_column
    ].astype(str)

    if df[
        index_column
    ].duplicated().any():
        raise ValueError(
            f"Duplicate values in "
            f"{index_column}"
        )

    if df[
        name_column
    ].duplicated().any():
        raise ValueError(
            f"Duplicate values in "
            f"{name_column}"
        )

    df = (
        df.sort_values(
            index_column
        )
        .reset_index(
            drop=True
        )
    )

    # Matrix positions are the row order after sorting by the deposited
    # matrix index. We do not assume whether the deposited index starts
    # at 0 or 1.
    df[
        "_matrix_position"
    ] = np.arange(
        len(df),
        dtype=np.int64,
    )

    expected_sequence = np.arange(
        int(
            df[
                index_column
            ].iloc[0]
        ),
        int(
            df[
                index_column
            ].iloc[0]
        )
        + len(df),
        dtype=np.int64,
    )

    contiguous = np.array_equal(
        df[
            index_column
        ].to_numpy(
            dtype=np.int64
        ),
        expected_sequence,
    )

    if not contiguous:
        raise ValueError(
            f"{index_column} is not contiguous "
            "after sorting."
        )

    return df


def summarize_numeric(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    finite = values[
        np.isfinite(values)
    ]

    if finite.size == 0:
        return {
            "min": np.nan,
            "q25": np.nan,
            "median": np.nan,
            "mean": np.nan,
            "q75": np.nan,
            "max": np.nan,
        }

    return {
        "min":
            float(
                np.min(
                    finite
                )
            ),
        "q25":
            float(
                np.quantile(
                    finite,
                    0.25,
                )
            ),
        "median":
            float(
                np.median(
                    finite
                )
            ),
        "mean":
            float(
                np.mean(
                    finite
                )
            ),
        "q75":
            float(
                np.quantile(
                    finite,
                    0.75,
                )
            ),
        "max":
            float(
                np.max(
                    finite
                )
            ),
    }


def make_variance_plot(
    variance_ratio,
    output_path,
):
    cumulative = np.cumsum(
        variance_ratio
    )

    x = np.arange(
        1,
        len(
            variance_ratio
        )
        + 1,
    )

    fig, ax = plt.subplots(
        figsize=(8.5, 5.5)
    )

    ax.plot(
        x,
        cumulative,
        marker="o",
        markersize=3,
        linewidth=1.5,
    )

    ax.set_xlabel(
        "Number of principal components"
    )

    ax.set_ylabel(
        "Cumulative explained variance ratio"
    )

    ax.set_title(
        "GSE175634 external-validation PCA"
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def make_pc_landmark_plot(
    state_space,
    output_path,
):
    """
    PC1/PC2 scatter of state centroids only.
    """
    centroids = (
        state_space.groupby(
            "type",
            observed=True,
        )[
            [
                "PC1",
                "PC2",
            ]
        ]
        .median()
        .reset_index()
    )

    fig, ax = plt.subplots(
        figsize=(7.5, 6)
    )

    ax.scatter(
        centroids[
            "PC1"
        ],
        centroids[
            "PC2"
        ],
        s=70,
    )

    for _, row in (
        centroids.iterrows()
    ):
        ax.annotate(
            str(
                row[
                    "type"
                ]
            ),
            (
                row[
                    "PC1"
                ],
                row[
                    "PC2"
                ],
            ),
            xytext=(
                5,
                5,
            ),
            textcoords="offset points",
        )

    ax.set_xlabel(
        "PC1"
    )

    ax.set_ylabel(
        "PC2"
    )

    ax.set_title(
        "Frozen external-validation state centroids"
    )

    ax.grid(
        alpha=0.25
    )

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

    for directory in [
        DATA_DIR,
        TABLE_DIR,
        FIGURE_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    required_files = [
        COUNT_FILE,
        CELL_INDEX_FILE,
        GENE_INDEX_FILE,
        FROZEN_CELL_FILE,
        FROZEN_TRANSITION_FILE,
    ]

    missing_files = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing required input files:\n"
            + "\n".join(
                missing_files
            )
        )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as report:

        write_both(
            report,
            "=" * 96,
        )

        write_both(
            report,
            "GDIS-Bio External Validation Preprocessing and State Space",
        )

        write_both(
            report,
            "Dataset: GSE175634",
        )

        write_both(
            report,
            "=" * 96,
        )

        write_both(
            report,
            "\nNo GDIS values are calculated in p15.",
        )

        write_both(
            report,
            "The p14 cohort definitions and biological transitions "
            "are used unchanged.",
        )

        # =========================================================
        # 1. Software and frozen parameters
        # =========================================================

        write_both(
            report,
            "\n1. SOFTWARE AND FROZEN PREPROCESSING PARAMETERS",
        )

        write_both(
            report,
            "-" * 96,
        )

        write_both(
            report,
            f"Python: {platform.python_version()}",
        )

        write_both(
            report,
            f"NumPy: {np.__version__}",
        )

        write_both(
            report,
            f"Pandas: {pd.__version__}",
        )

        write_both(
            report,
            f"SciPy: {scipy.__version__}",
        )

        write_both(
            report,
            f"Scanpy: {sc.__version__}",
        )

        write_both(
            report,
            f"AnnData: {ad.__version__}",
        )

        write_both(
            report,
            f"Normalization target: "
            f"{NORMALIZATION_TARGET:,.0f}",
        )

        write_both(
            report,
            f"Gene min_cells: "
            f"{MIN_GENE_CELLS}",
        )

        write_both(
            report,
            f"HVGs: {N_HVG}",
        )

        write_both(
            report,
            f"HVG flavor: "
            f"{HVG_FLAVOR}",
        )

        write_both(
            report,
            f"HVG batch key: "
            f"{HVG_BATCH_KEY}",
        )

        write_both(
            report,
            "Batch correction: NONE",
        )

        write_both(
            report,
            f"Scale max_value: "
            f"{SCALE_MAX_VALUE}",
        )

        write_both(
            report,
            f"PCA components: "
            f"{N_PCS}",
        )

        write_both(
            report,
            f"PCA solver: "
            f"{PCA_SOLVER}",
        )

        write_both(
            report,
            f"PCA random state: "
            f"{PCA_RANDOM_STATE}",
        )

        # =========================================================
        # 2. Frozen cell population
        # =========================================================

        write_both(
            report,
            "\n2. FROZEN COMMON STATE-SPACE POPULATION",
        )

        write_both(
            report,
            "-" * 96,
        )

        frozen = pd.read_csv(
            FROZEN_CELL_FILE,
            compression="gzip",
            low_memory=False,
        )

        required_frozen_columns = [
            "cell",
            "individual",
            "type",
            "diffday",
            "dpt_pseudotime",
        ] + INCLUSION_COLUMNS

        missing = [
            c
            for c in required_frozen_columns
            if c not in frozen.columns
        ]

        if missing:
            raise ValueError(
                "Frozen cell table missing columns: "
                + ", ".join(
                    missing
                )
            )

        frozen[
            "cell"
        ] = frozen[
            "cell"
        ].astype(str)

        frozen[
            "individual"
        ] = frozen[
            "individual"
        ].astype(str)

        frozen[
            "type"
        ] = frozen[
            "type"
        ].astype(str)

        frozen[
            "diffday"
        ] = frozen[
            "diffday"
        ].astype(str)

        frozen[
            "dpt_pseudotime"
        ] = pd.to_numeric(
            frozen[
                "dpt_pseudotime"
            ],
            errors="raise",
        )

        for column in INCLUSION_COLUMNS:
            frozen[
                column
            ] = robust_bool(
                frozen[
                    column
                ]
            )

        frozen[
            "include_any_frozen_analysis"
        ] = frozen[
            INCLUSION_COLUMNS
        ].any(
            axis=1
        )

        selected_meta = frozen[
            frozen[
                "include_any_frozen_analysis"
            ]
        ].copy()

        if selected_meta[
            "cell"
        ].duplicated().any():
            raise ValueError(
                "Duplicate cell IDs in frozen selection."
            )

        write_both(
            report,
            f"Frozen known cells: "
            f"{len(frozen):,}",
        )

        write_both(
            report,
            f"Union included in common state space: "
            f"{len(selected_meta):,}",
        )

        for column in INCLUSION_COLUMNS:

            write_both(
                report,
                f"  {column}: "
                f"{int(frozen[column].sum()):,}",
            )

        cohort_counts = (
            selected_meta.groupby(
                [
                    "type",
                    "individual",
                ],
                observed=True,
            )
            .size()
            .reset_index(
                name="n_cells"
            )
        )

        save_table(
            cohort_counts,
            "01_state_space_population_by_type_individual.csv",
            index=False,
        )

        # =========================================================
        # 3. Matrix / index validation and loading
        # =========================================================

        write_both(
            report,
            "\n3. RAW COUNT MATRIX ALIGNMENT",
        )

        write_both(
            report,
            "-" * 96,
        )

        cell_index = load_index_table(
            CELL_INDEX_FILE,
            "cell_index",
            "cell_name",
        )

        gene_index = load_index_table(
            GENE_INDEX_FILE,
            "gene_index",
            "gene_name",
        )

        write_both(
            report,
            f"Cell index rows: "
            f"{len(cell_index):,}",
        )

        write_both(
            report,
            f"Gene index rows: "
            f"{len(gene_index):,}",
        )

        # Align frozen cells to matrix columns.
        cell_map = cell_index.merge(
            selected_meta,
            left_on="cell_name",
            right_on="cell",
            how="inner",
            validate="one_to_one",
        )

        if len(
            cell_map
        ) != len(
            selected_meta
        ):
            missing_count = (
                len(
                    selected_meta
                )
                - len(
                    cell_map
                )
            )

            raise ValueError(
                f"{missing_count:,} frozen cells are missing "
                "from the count-matrix cell index."
            )

        cell_map = cell_map.sort_values(
            "_matrix_position"
        ).reset_index(
            drop=True
        )

        selected_positions = (
            cell_map[
                "_matrix_position"
            ].to_numpy(
                dtype=np.int64
            )
        )

        write_both(
            report,
            "Loading Matrix Market counts. "
            "This is the memory-intensive step...",
        )

        with gzip.open(
            COUNT_FILE,
            "rb",
        ) as handle:
            counts_gxc = mmread(
                handle
            )

        if not sp.issparse(
            counts_gxc
        ):
            counts_gxc = sp.coo_matrix(
                counts_gxc
            )

        write_both(
            report,
            f"Loaded matrix shape: "
            f"{counts_gxc.shape[0]:,} x "
            f"{counts_gxc.shape[1]:,}",
        )

        write_both(
            report,
            f"Loaded matrix nnz: "
            f"{counts_gxc.nnz:,}",
        )

        if counts_gxc.shape != (
            len(
                gene_index
            ),
            len(
                cell_index
            ),
        ):
            raise ValueError(
                "Count matrix dimensions do not match "
                "sorted gene/cell indices."
            )

        if counts_gxc.nnz == 0:
            raise ValueError(
                "Count matrix contains no nonzero values."
            )

        count_data_min = float(
            counts_gxc.data.min()
        )

        count_data_max = float(
            counts_gxc.data.max()
        )

        count_dtype = str(
            counts_gxc.dtype
        )

        count_integer_dtype = bool(
            np.issubdtype(
                counts_gxc.dtype,
                np.integer,
            )
        )

        count_nonnegative = bool(
            count_data_min >= 0
        )

        write_both(
            report,
            f"Raw count dtype: "
            f"{count_dtype}",
        )

        write_both(
            report,
            f"Integer count dtype: "
            f"{count_integer_dtype}",
        )

        write_both(
            report,
            f"Nonzero count range: "
            f"{count_data_min:g} to "
            f"{count_data_max:g}",
        )

        write_both(
            report,
            f"Nonnegative counts: "
            f"{count_nonnegative}",
        )

        # Convert to CSC because the deposited matrix is genes x cells
        # and we need a frozen subset of columns.
        write_both(
            report,
            "Converting genes x cells matrix to CSC "
            "for frozen-column subsetting...",
        )

        counts_gxc_csc = counts_gxc.tocsc()

        del counts_gxc
        gc.collect()

        selected_gxc = counts_gxc_csc[
            :,
            selected_positions,
        ]

        del counts_gxc_csc
        gc.collect()

        # Convert to cells x genes CSR for Scanpy.
        X = (
            selected_gxc.T
            .tocsr()
        )

        del selected_gxc
        gc.collect()

        write_both(
            report,
            f"Frozen cells x genes sparse matrix: "
            f"{X.shape[0]:,} x "
            f"{X.shape[1]:,}",
        )

        write_both(
            report,
            f"Frozen matrix nnz: "
            f"{X.nnz:,}",
        )

        write_both(
            report,
            f"Frozen sparse storage estimate: "
            f"{human_bytes(X.data.nbytes + X.indices.nbytes + X.indptr.nbytes)}",
        )

        if X.shape != (
            len(
                cell_map
            ),
            len(
                gene_index
            ),
        ):
            raise ValueError(
                "Frozen cells x genes matrix has "
                "unexpected dimensions."
            )

        # =========================================================
        # 4. AnnData and raw-count QC
        # =========================================================

        write_both(
            report,
            "\n4. RAW-COUNT QC WITHOUT CELL FILTERING",
        )

        write_both(
            report,
            "-" * 96,
        )

        obs_columns = [
            "cell",
            "individual",
            "type",
            "diffday",
            "dpt_pseudotime",
            "include_shared_backbone_primary",
            "include_cm_extension_secondary",
            "include_cf_extension_secondary",
            "include_any_frozen_analysis",
        ]

        obs = (
            cell_map[
                obs_columns
            ]
            .copy()
            .set_index(
                "cell",
                drop=False,
            )
        )

        var = pd.DataFrame(
            {
                "gene_index":
                    gene_index[
                        "gene_index"
                    ].to_numpy(),
                "gene_name_original":
                    gene_index[
                        "gene_name"
                    ].astype(str).to_numpy(),
            },
            index=gene_index[
                "gene_name"
            ].astype(str).to_numpy(),
        )

        adata = ad.AnnData(
            X=X,
            obs=obs,
            var=var,
        )

        del X
        gc.collect()

        # Keep original names but guarantee unique AnnData var_names.
        adata.var_names_make_unique()

        total_counts = np.asarray(
            adata.X.sum(
                axis=1
            )
        ).ravel()

        # In CSR, each stored positive count corresponds to a detected gene.
        # Matrix Market count data may theoretically contain explicit zeros,
        # so compute gene detection robustly if needed.
        if (
            adata.X.data.size > 0
            and np.all(
                adata.X.data > 0
            )
        ):
            n_genes_detected = np.diff(
                adata.X.indptr
            ).astype(
                np.int64
            )
        else:
            n_genes_detected = np.asarray(
                (
                    adata.X > 0
                ).sum(
                    axis=1
                )
            ).ravel().astype(
                np.int64
            )

        original_gene_names = adata.var[
            "gene_name_original"
        ].astype(str)

        mt_mask = (
            original_gene_names
            .str.upper()
            .str.startswith(
                "MT-"
            )
            .to_numpy()
        )

        n_mt_genes = int(
            mt_mask.sum()
        )

        if n_mt_genes > 0:

            mt_counts = np.asarray(
                adata.X[
                    :,
                    mt_mask,
                ].sum(
                    axis=1
                )
            ).ravel()

            pct_counts_mt = np.divide(
                100.0
                * mt_counts,
                total_counts,
                out=np.zeros_like(
                    total_counts,
                    dtype=float,
                ),
                where=(
                    total_counts > 0
                ),
            )

        else:

            mt_counts = np.zeros(
                adata.n_obs,
                dtype=float,
            )

            pct_counts_mt = np.full(
                adata.n_obs,
                np.nan,
                dtype=float,
            )

        adata.obs[
            "total_counts_raw"
        ] = total_counts

        adata.obs[
            "n_genes_by_counts_raw"
        ] = n_genes_detected

        adata.obs[
            "pct_counts_mt_raw"
        ] = pct_counts_mt

        total_summary = summarize_numeric(
            total_counts
        )

        gene_summary = summarize_numeric(
            n_genes_detected
        )

        mt_summary = summarize_numeric(
            pct_counts_mt
        )

        write_both(
            report,
            f"Cells with zero total counts: "
            f"{int(np.sum(total_counts <= 0)):,}",
        )

        write_both(
            report,
            f"Mitochondrial gene names detected: "
            f"{n_mt_genes:,}",
        )

        write_both(
            report,
            "\nRaw total counts/cell:",
        )

        write_both(
            report,
            str(
                total_summary
            ),
        )

        write_both(
            report,
            "\nRaw detected genes/cell:",
        )

        write_both(
            report,
            str(
                gene_summary
            ),
        )

        write_both(
            report,
            "\nRaw mitochondrial percent:",
        )

        write_both(
            report,
            str(
                mt_summary
            ),
        )

        qc_cell_table = adata.obs[
            [
                "cell",
                "individual",
                "type",
                "diffday",
                "dpt_pseudotime",
                "total_counts_raw",
                "n_genes_by_counts_raw",
                "pct_counts_mt_raw",
            ]
        ].copy()

        qc_cell_table.to_csv(
            TABLE_DIR
            / "02_raw_cell_qc.csv.gz",
            index=False,
            compression="gzip",
        )

        qc_group = (
            qc_cell_table.groupby(
                "type",
                observed=True,
            )
            .agg(
                n_cells=(
                    "cell",
                    "size",
                ),
                median_total_counts=(
                    "total_counts_raw",
                    "median",
                ),
                median_genes=(
                    "n_genes_by_counts_raw",
                    "median",
                ),
                median_pct_mt=(
                    "pct_counts_mt_raw",
                    "median",
                ),
            )
            .reset_index()
        )

        save_table(
            qc_group,
            "03_raw_qc_summary_by_type.csv",
            index=False,
        )

        # =========================================================
        # 5. Gene filtering, normalization, HVGs
        # =========================================================

        write_both(
            report,
            "\n5. NORMALIZATION AND BATCH-AWARE HVG SELECTION",
        )

        write_both(
            report,
            "-" * 96,
        )

        genes_before = int(
            adata.n_vars
        )

        sc.pp.filter_genes(
            adata,
            min_cells=MIN_GENE_CELLS,
        )

        genes_after = int(
            adata.n_vars
        )

        genes_removed = (
            genes_before
            - genes_after
        )

        write_both(
            report,
            f"Genes before min_cells filter: "
            f"{genes_before:,}",
        )

        write_both(
            report,
            f"Genes retained with min_cells >= "
            f"{MIN_GENE_CELLS}: "
            f"{genes_after:,}",
        )

        write_both(
            report,
            f"Genes removed: "
            f"{genes_removed:,}",
        )

        # Convert to float32 before normalization to control memory.
        adata.X = adata.X.astype(
            np.float32
        )

        sc.pp.normalize_total(
            adata,
            target_sum=NORMALIZATION_TARGET,
        )

        sc.pp.log1p(
            adata
        )

        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=N_HVG,
            flavor=HVG_FLAVOR,
            batch_key=HVG_BATCH_KEY,
            inplace=True,
        )

        n_hvg_selected = int(
            adata.var[
                "highly_variable"
            ].sum()
        )

        write_both(
            report,
            f"Highly variable genes selected: "
            f"{n_hvg_selected:,}",
        )

        if (
            n_hvg_selected
            != N_HVG
        ):
            raise ValueError(
                f"Expected {N_HVG} HVGs but selected "
                f"{n_hvg_selected}."
            )

        hvg_columns = [
            column
            for column in [
                "gene_index",
                "gene_name_original",
                "highly_variable",
                "highly_variable_rank",
                "highly_variable_nbatches",
                "means",
                "dispersions",
                "dispersions_norm",
            ]
            if column in adata.var.columns
        ]

        hvg_table = (
            adata.var.loc[
                adata.var[
                    "highly_variable"
                ],
                hvg_columns,
            ]
            .copy()
        )

        hvg_table.insert(
            0,
            "var_name_unique",
            hvg_table.index.astype(
                str
            ),
        )

        save_table(
            hvg_table.reset_index(
                drop=True
            ),
            "04_batch_aware_hvg_2000.csv",
            index=False,
        )

        hvg_nbatches_summary = None

        if (
            "highly_variable_nbatches"
            in hvg_table.columns
        ):

            hvg_nbatches_summary = (
                hvg_table[
                    "highly_variable_nbatches"
                ]
                .value_counts()
                .sort_index()
                .rename_axis(
                    "n_batches"
                )
                .reset_index(
                    name="n_hvgs"
                )
            )

            save_table(
                hvg_nbatches_summary,
                "05_hvg_batch_support_distribution.csv",
                index=False,
            )

        # =========================================================
        # 6. 50-PC common state space
        # =========================================================

        write_both(
            report,
            "\n6. COMMON 50-PC STATE SPACE",
        )

        write_both(
            report,
            "-" * 96,
        )

        adata_hvg = adata[
            :,
            adata.var[
                "highly_variable"
            ],
        ].copy()

        # Free the large full-gene normalized matrix before scaling.
        del adata
        gc.collect()

        write_both(
            report,
            f"HVG matrix: "
            f"{adata_hvg.n_obs:,} cells x "
            f"{adata_hvg.n_vars:,} genes",
        )

        write_both(
            report,
            "Scaling HVG expression. "
            "This may densify the 2,000-gene matrix...",
        )

        sc.pp.scale(
            adata_hvg,
            zero_center=True,
            max_value=SCALE_MAX_VALUE,
        )

        if (
            sp.issparse(
                adata_hvg.X
            )
        ):
            adata_hvg.X = (
                adata_hvg.X
                .toarray()
                .astype(
                    np.float32,
                    copy=False,
                )
            )
        else:
            adata_hvg.X = np.asarray(
                adata_hvg.X,
                dtype=np.float32,
            )

        write_both(
            report,
            f"Scaled HVG dense matrix storage: "
            f"{human_bytes(adata_hvg.X.nbytes)}",
        )

        sc.tl.pca(
            adata_hvg,
            n_comps=N_PCS,
            zero_center=True,
            svd_solver=PCA_SOLVER,
            random_state=PCA_RANDOM_STATE,
        )

        pcs = np.asarray(
            adata_hvg.obsm[
                "X_pca"
            ],
            dtype=np.float32,
        )

        if pcs.shape != (
            adata_hvg.n_obs,
            N_PCS,
        ):
            raise ValueError(
                f"Unexpected PCA shape: "
                f"{pcs.shape}"
            )

        finite_pcs = bool(
            np.isfinite(
                pcs
            ).all()
        )

        variance_ratio = np.asarray(
            adata_hvg.uns[
                "pca"
            ][
                "variance_ratio"
            ],
            dtype=float,
        )

        variance = np.asarray(
            adata_hvg.uns[
                "pca"
            ][
                "variance"
            ],
            dtype=float,
        )

        pca_variance_table = pd.DataFrame(
            {
                "pc":
                    np.arange(
                        1,
                        N_PCS + 1,
                    ),
                "variance":
                    variance,
                "variance_ratio":
                    variance_ratio,
                "cumulative_variance_ratio":
                    np.cumsum(
                        variance_ratio
                    ),
            }
        )

        save_table(
            pca_variance_table,
            "06_pca_explained_variance.csv",
            index=False,
        )

        cumulative_rows = []

        for n_dim in PCA_EXPORT_DIMS:

            cumulative = float(
                variance_ratio[
                    :n_dim
                ].sum()
            )

            cumulative_rows.append(
                {
                    "n_pcs":
                        n_dim,
                    "cumulative_variance_ratio":
                        cumulative,
                }
            )

            write_both(
                report,
                f"Cumulative variance, "
                f"{n_dim:>2} PCs: "
                f"{cumulative:.6f}",
            )

        cumulative_table = pd.DataFrame(
            cumulative_rows
        )

        save_table(
            cumulative_table,
            "07_pca_cumulative_variance_key_dimensions.csv",
            index=False,
        )

        # =========================================================
        # 7. Export common state spaces
        # =========================================================

        write_both(
            report,
            "\n7. STATE-SPACE EXPORTS",
        )

        write_both(
            report,
            "-" * 96,
        )

        state_space_metadata = (
            adata_hvg.obs[
                [
                    "cell",
                    "individual",
                    "type",
                    "diffday",
                    "dpt_pseudotime",
                    "include_shared_backbone_primary",
                    "include_cm_extension_secondary",
                    "include_cf_extension_secondary",
                    "include_any_frozen_analysis",
                    "total_counts_raw",
                    "n_genes_by_counts_raw",
                    "pct_counts_mt_raw",
                ]
            ]
            .reset_index(
                drop=True
            )
            .copy()
        )

        state_space_50 = (
            state_space_metadata.copy()
        )

        for j in range(
            N_PCS
        ):
            state_space_50[
                f"PC{j + 1}"
            ] = pcs[
                :,
                j,
            ]

        for n_dim in PCA_EXPORT_DIMS:

            columns = (
                list(
                    state_space_metadata.columns
                )
                + [
                    f"PC{i}"
                    for i in range(
                        1,
                        n_dim + 1,
                    )
                ]
            )

            output = (
                state_space_50[
                    columns
                ]
            )

            output_path = (
                DATA_DIR
                / (
                    f"external_state_space_pca_"
                    f"{n_dim}.csv.gz"
                )
            )

            output.to_csv(
                output_path,
                index=False,
                compression="gzip",
            )

            write_both(
                report,
                f"Saved {n_dim}-PC state space: "
                f"{output_path.name}",
            )

        # Compact AnnData containing ONLY the 50-PC state-space matrix.
        compact_obs = (
            state_space_metadata.copy()
        )

        compact_obs.index = (
            compact_obs[
                "cell"
            ].astype(str)
        )

        compact_var = pd.DataFrame(
            index=[
                f"PC{i}"
                for i in range(
                    1,
                    N_PCS + 1,
                )
            ]
        )

        compact = ad.AnnData(
            X=pcs,
            obs=compact_obs,
            var=compact_var,
        )

        compact.uns[
            "gdis_bio_preprocessing"
        ] = {
            "dataset":
                "GSE175634",
            "normalization_target":
                float(
                    NORMALIZATION_TARGET
                ),
            "min_gene_cells":
                int(
                    MIN_GENE_CELLS
                ),
            "n_hvg":
                int(
                    N_HVG
                ),
            "hvg_flavor":
                HVG_FLAVOR,
            "hvg_batch_key":
                HVG_BATCH_KEY,
            "batch_correction":
                "none",
            "scale_max_value":
                float(
                    SCALE_MAX_VALUE
                ),
            "n_pcs":
                int(
                    N_PCS
                ),
            "pca_solver":
                PCA_SOLVER,
            "pca_random_state":
                int(
                    PCA_RANDOM_STATE
                ),
            "state_space_population":
                (
                    "union of all frozen p14 analysis "
                    "inclusion flags"
                ),
        }

        compact.obsm[
            "X_pca"
        ] = pcs.copy()

        compact.write_h5ad(
            DATA_DIR
            / "external_state_space_pca50_compact.h5ad"
        )

        # =========================================================
        # 8. State-space descriptive summaries
        # =========================================================

        write_both(
            report,
            "\n8. STATE-SPACE DESCRIPTIVE SUMMARIES",
        )

        write_both(
            report,
            "-" * 96,
        )

        state_centroids = (
            state_space_50.groupby(
                "type",
                observed=True,
            )[
                [
                    f"PC{i}"
                    for i in range(
                        1,
                        N_PCS + 1,
                    )
                ]
            ]
            .median()
            .reset_index()
        )

        save_table(
            state_centroids,
            "08_state_median_centroids_50pc.csv",
            index=False,
        )

        individual_state_centroids = (
            state_space_50.groupby(
                [
                    "individual",
                    "type",
                ],
                observed=True,
            )[
                [
                    f"PC{i}"
                    for i in range(
                        1,
                        N_PCS + 1,
                    )
                ]
            ]
            .median()
            .reset_index()
        )

        save_table(
            individual_state_centroids,
            "09_individual_state_median_centroids_50pc.csv",
            index=False,
        )

        make_variance_plot(
            variance_ratio,
            FIGURE_DIR
            / "01_pca_cumulative_explained_variance.png",
        )

        make_pc_landmark_plot(
            state_space_50,
            FIGURE_DIR
            / "02_pc1_pc2_state_centroids.png",
        )

        # =========================================================
        # 9. Qualification
        # =========================================================

        write_both(
            report,
            "\n9. PREPROCESSING / STATE-SPACE QUALIFICATION",
        )

        write_both(
            report,
            "-" * 96,
        )

        zero_count_cells = int(
            np.sum(
                total_counts <= 0
            )
        )

        selected_id_match = (
            len(
                cell_map
            )
            == len(
                selected_meta
            )
        )

        all_cohort_flags_retained = True

        # Verify exact frozen inclusion counts after export.
        for column in INCLUSION_COLUMNS:

            frozen_count = int(
                frozen[
                    column
                ].sum()
            )

            state_space_count = int(
                state_space_50[
                    column
                ].sum()
            )

            if (
                frozen_count
                != state_space_count
            ):
                all_cohort_flags_retained = False

        checks = [
            (
                "Raw count matrix dimensions match GEO indices",
                True,
            ),
            (
                "All frozen union cell IDs were found in the "
                "count matrix",
                selected_id_match,
            ),
            (
                "Raw count entries are nonnegative",
                count_nonnegative,
            ),
            (
                "Raw count matrix uses integer count dtype",
                count_integer_dtype,
            ),
            (
                "No selected cells have zero total counts",
                zero_count_cells == 0,
            ),
            (
                f"Exactly {N_HVG:,} batch-aware HVGs were selected",
                n_hvg_selected
                == N_HVG,
            ),
            (
                "50-PC state-space coordinates are finite",
                finite_pcs,
            ),
            (
                "All frozen primary/secondary cohort counts are "
                "preserved in the common state space",
                all_cohort_flags_retained,
            ),
        ]

        passed = 0

        for label, status in checks:

            if status:
                prefix = "[PASS]"
                passed += 1
            else:
                prefix = "[REVIEW]"

            write_both(
                report,
                f"{prefix} {label}",
            )

        write_both(
            report,
            f"\nQualification checks passed: "
            f"{passed}/{len(checks)}",
        )

        if passed == len(
            checks
        ):

            write_both(
                report,
                "\nFINAL STATUS: PASS",
            )

            write_both(
                report,
                "The common GSE175634 expression state space "
                "has been constructed successfully.",
            )

            write_both(
                report,
                "The 50-PC representation is the provisional "
                "primary external-validation state space.",
            )

            write_both(
                report,
                "The 5/10/20/30-PC exports are frozen for "
                "pre-GDIS dimensionality sensitivity.",
            )

            write_both(
                report,
                "No batch correction was applied.",
            )

            write_both(
                report,
                "NO GDIS VALUES WERE CALCULATED.",
            )

        else:

            write_both(
                report,
                "\nFINAL STATUS: REVIEW REQUIRED",
            )

        write_both(
            report,
            "\nNext step:",
        )

        write_both(
            report,
            "Validate dimensionality and individual-level "
            "trajectory geometry before assembling GDIS windows.",
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

        write_both(
            report,
            f"Data:    {DATA_DIR}",
        )

    # Free large objects explicitly before exit.
    del state_space_50
    del adata_hvg
    del compact
    gc.collect()

    print("\n" + "=" * 96)
    print("p15_GSE175634_preprocessing_state_space.py completed.")
    print("=" * 96)
    print(f"Report : {REPORT_FILE}")
    print(f"Tables : {TABLE_DIR}")
    print(f"Figures: {FIGURE_DIR}")
    print(f"Data   : {DATA_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

