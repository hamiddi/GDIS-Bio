#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================
"""
p3_dataset_characterization.py

Biological characterization of GSE114412 Stage-5 data for GDIS-Bio.

Purpose
-------
This script characterizes the temporal and biological structure of the
approved GSE114412 Stage-5 dataset BEFORE any new normalization, filtering,
PCA, trajectory inference, or GDIS calculation.

It generates:
1. Cell counts by Day 0-7
2. Cell counts by day and differentiation
3. Cluster counts and proportions by day
4. Cluster composition separately for Differentiation 1 and 2
5. Subcluster counts and proportions by day
6. First appearance / temporal span of each cluster
7. Endocrine-trajectory state summaries
8. Marker-gene expression summaries by day
9. Marker-gene expression summaries by day and differentiation
10. Descriptive figures and a text report

Expected project structure
--------------------------
GDIS_Bio/
├── scripts/
│   ├── p1_download_data.py
│   ├── p2_preflight_data.py
│   └── p3_dataset_characterization.py
├── raw_data/
│   ├── GSE114412_Stage_5.all.processed_counts.tsv.gz
│   ├── GSE114412_Stage_5.all.cell_metadata.tsv.gz
│   └── GSE114412_Stage_5.endocrine_pseudotime.cell_metadata.tsv.gz
└── results/

Requirements
------------
pandas
numpy
matplotlib

IMPORTANT
---------
The expression matrix is already a deposited processed-count matrix.
This script does NOT perform any additional normalization or transformation.
Marker-expression summaries are therefore descriptive only.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

if SCRIPT_DIR.name == "scripts":
    PROJECT_DIR = SCRIPT_DIR.parent
else:
    PROJECT_DIR = SCRIPT_DIR

RAW_DIR = PROJECT_DIR / "raw_data"
RESULTS_DIR = PROJECT_DIR / "results" / "p3_dataset_characterization"
FIGURE_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"

COUNTS_FILE = RAW_DIR / "GSE114412_Stage_5.all.processed_counts.tsv.gz"
METADATA_FILE = RAW_DIR / "GSE114412_Stage_5.all.cell_metadata.tsv.gz"
PSEUDOTIME_FILE = (
    RAW_DIR / "GSE114412_Stage_5.endocrine_pseudotime.cell_metadata.tsv.gz"
)

REPORT_FILE = RESULTS_DIR / "p3_dataset_characterization_report.txt"

CELL_ID_COUNTS = "# library.barcode"
CELL_ID_METADATA = "library.barcode"
DAY_COLUMN = "CellWeek"
DIFF_COLUMN = "Differentiation"
CLUSTER_COLUMN = "Assigned_cluster"
SUBCLUSTER_COLUMN = "Assigned_subcluster"

# Marker genes already confirmed by p2_preflight_data.py.
MARKER_GENES = [
    "NEUROG3",
    "PDX1",
    "NKX6-1",
    "INS",
    "PAX4",
    "NEUROD1",
    "TPH1",
    "SLC18A1",
    "ISL1",
    "PTF1A",
]

# States of special interest for the proposed GDIS biological trajectory.
# These names come directly from the deposited metadata.
ENDOCRINE_STATES = [
    "prog_nkx61",
    "neurog3_early",
    "neurog3_mid",
    "neurog3_late",
    "fev_high_isl_low",
    "sc_beta",
    "sc_ec",
    "sc_alpha",
    "sst_hhex",
    "phox2a",
]


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def write_both(report, text=""):
    """Print text and also write it to the report file."""
    print(text)
    report.write(text + "\n")


def save_table(df, filename, index=True):
    """Save a dataframe as CSV in the p3 tables directory."""
    path = TABLE_DIR / filename
    df.to_csv(path, index=index)
    return path


def ensure_required_columns(df, columns, table_name):
    """Fail early if required columns are absent."""
    missing = [col for col in columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"{table_name} is missing required column(s): "
            + ", ".join(missing)
        )


def ordered_days(series):
    """Return sorted numeric day values."""
    values = pd.to_numeric(series, errors="coerce").dropna().unique()
    return sorted(int(v) for v in values)


def plot_cells_per_day(day_counts, output_path):
    """Bar plot of total cell counts by experimental day."""
    fig, ax = plt.subplots(figsize=(8, 5))

    x = day_counts.index.astype(int)
    y = day_counts.values

    ax.bar(x, y)
    ax.set_xlabel("Experimental day")
    ax.set_ylabel("Number of cells")
    ax.set_title("GSE114412 Stage 5: cells sampled by day")
    ax.set_xticks(x)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_day_by_differentiation(day_diff_counts, output_path):
    """Grouped bar plot of cell counts by day and differentiation."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    day_diff_counts.plot(kind="bar", ax=ax)

    ax.set_xlabel("Experimental day")
    ax.set_ylabel("Number of cells")
    ax.set_title("Cell counts by day and independent differentiation")
    ax.legend(title="Differentiation")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_proportions(cluster_props, output_path, title):
    """Stacked bar plot of cluster composition by day."""
    fig, ax = plt.subplots(figsize=(12, 7))

    cluster_props.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        width=0.85,
    )

    ax.set_xlabel("Experimental day")
    ax.set_ylabel("Fraction of cells")
    ax.set_title(title)
    ax.legend(
        title="Assigned cluster",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_selected_state_proportions(state_props, output_path):
    """Line plot for endocrine-state proportions across days."""
    fig, ax = plt.subplots(figsize=(11, 6.5))

    for state in state_props.columns:
        ax.plot(
            state_props.index,
            state_props[state],
            marker="o",
            linewidth=1.8,
            label=state,
        )

    ax.set_xlabel("Experimental day")
    ax.set_ylabel("Fraction of all cells")
    ax.set_title("Temporal abundance of selected endocrine states")
    ax.set_xticks(state_props.index)
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
    )
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_marker_detection(marker_detection, output_path):
    """Plot percentage of cells with non-zero marker expression by day."""
    fig, ax = plt.subplots(figsize=(11, 6.5))

    for gene in marker_detection.columns:
        ax.plot(
            marker_detection.index,
            marker_detection[gene] * 100.0,
            marker="o",
            linewidth=1.7,
            label=gene,
        )

    ax.set_xlabel("Experimental day")
    ax.set_ylabel("Cells with expression > 0 (%)")
    ax.set_title("Detection frequency of selected developmental markers")
    ax.set_xticks(marker_detection.index)
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
    )
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_marker_mean(marker_mean, output_path):
    """Plot deposited processed-count mean for markers by day."""
    fig, ax = plt.subplots(figsize=(11, 6.5))

    for gene in marker_mean.columns:
        ax.plot(
            marker_mean.index,
            marker_mean[gene],
            marker="o",
            linewidth=1.7,
            label=gene,
        )

    ax.set_xlabel("Experimental day")
    ax.set_ylabel("Mean deposited processed count")
    ax.set_title("Mean marker expression by experimental day")
    ax.set_xticks(marker_mean.index)
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
    )
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------

def main():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Load metadata
    # -----------------------------------------------------------------

    metadata = pd.read_csv(
        METADATA_FILE,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )

    ensure_required_columns(
        metadata,
        [
            CELL_ID_METADATA,
            DAY_COLUMN,
            DIFF_COLUMN,
            CLUSTER_COLUMN,
            SUBCLUSTER_COLUMN,
        ],
        "Main metadata",
    )

    # Convert experimental day and differentiation to numeric values.
    metadata[DAY_COLUMN] = pd.to_numeric(
        metadata[DAY_COLUMN],
        errors="raise",
    ).astype(int)

    metadata[DIFF_COLUMN] = pd.to_numeric(
        metadata[DIFF_COLUMN],
        errors="raise",
    ).astype(int)

    days = ordered_days(metadata[DAY_COLUMN])

    # -----------------------------------------------------------------
    # Read only barcode + selected markers from the expression matrix.
    # This avoids loading the complete 51,274 x 16,224 matrix into RAM.
    # -----------------------------------------------------------------

    counts_header = pd.read_csv(
        COUNTS_FILE,
        sep="\t",
        compression="gzip",
        nrows=0,
    )

    available_columns = set(counts_header.columns)

    marker_columns = [
        gene for gene in MARKER_GENES
        if gene in available_columns
    ]

    missing_markers = [
        gene for gene in MARKER_GENES
        if gene not in available_columns
    ]

    usecols = [CELL_ID_COUNTS] + marker_columns

    marker_counts = pd.read_csv(
        COUNTS_FILE,
        sep="\t",
        compression="gzip",
        usecols=usecols,
        low_memory=False,
    )

    marker_counts = marker_counts.rename(
        columns={CELL_ID_COUNTS: CELL_ID_METADATA}
    )

    # Merge expression summaries with metadata.
    analysis = metadata.merge(
        marker_counts,
        on=CELL_ID_METADATA,
        how="left",
        validate="one_to_one",
    )

    if analysis[marker_columns].isna().all(axis=None):
        raise RuntimeError(
            "Marker-expression columns did not merge correctly with metadata."
        )

    # -----------------------------------------------------------------
    # Generate report
    # -----------------------------------------------------------------

    with open(REPORT_FILE, "w", encoding="utf-8") as report:

        write_both(report, "=" * 80)
        write_both(report, "GDIS-Bio Dataset Characterization")
        write_both(report, "Dataset: GSE114412 Stage 5")
        write_both(report, "=" * 80)

        write_both(
            report,
            "\nPurpose: establish the biological and temporal transition "
            "landscape before any new preprocessing or GDIS analysis."
        )

        # =============================================================
        # 1. Dataset overview
        # =============================================================

        write_both(report, "\n1. DATASET OVERVIEW")
        write_both(report, "-" * 80)

        write_both(
            report,
            f"Total cells: {metadata.shape[0]:,}"
        )

        write_both(
            report,
            f"Experimental days: {', '.join(map(str, days))}"
        )

        write_both(
            report,
            f"Independent differentiations: "
            f"{metadata[DIFF_COLUMN].nunique()}"
        )

        write_both(
            report,
            f"Assigned clusters: "
            f"{metadata[CLUSTER_COLUMN].nunique()}"
        )

        write_both(
            report,
            f"Assigned subclusters: "
            f"{metadata[SUBCLUSTER_COLUMN].nunique()}"
        )

        write_both(
            report,
            f"Marker genes loaded: {len(marker_columns)}"
        )

        if missing_markers:
            write_both(
                report,
                "Markers not found: " + ", ".join(missing_markers)
            )

        # =============================================================
        # 2. Cell counts by day
        # =============================================================

        write_both(report, "\n2. CELL COUNTS BY EXPERIMENTAL DAY")
        write_both(report, "-" * 80)

        day_counts = (
            metadata.groupby(DAY_COLUMN)
            .size()
            .reindex(days, fill_value=0)
        )

        day_counts.name = "n_cells"
        day_counts.index.name = "day"

        save_table(
            day_counts.to_frame(),
            "01_cell_counts_by_day.csv",
            index=True,
        )

        for day, count in day_counts.items():
            write_both(
                report,
                f"Day {day}: {count:,} cells"
            )

        plot_cells_per_day(
            day_counts,
            FIGURE_DIR / "01_cell_counts_by_day.png",
        )

        # =============================================================
        # 3. Day x differentiation
        # =============================================================

        write_both(
            report,
            "\n3. CELL COUNTS BY DAY AND DIFFERENTIATION"
        )
        write_both(report, "-" * 80)

        day_diff_counts = pd.crosstab(
            metadata[DAY_COLUMN],
            metadata[DIFF_COLUMN],
        ).reindex(days, fill_value=0)

        day_diff_counts.index.name = "day"
        day_diff_counts.columns = [
            f"differentiation_{col}"
            for col in day_diff_counts.columns
        ]

        save_table(
            day_diff_counts,
            "02_cell_counts_day_by_differentiation.csv",
            index=True,
        )

        write_both(
            report,
            day_diff_counts.to_string()
        )

        plot_day_by_differentiation(
            day_diff_counts,
            FIGURE_DIR / "02_cell_counts_day_by_differentiation.png",
        )

        # =============================================================
        # 4. Cluster composition by day
        # =============================================================

        write_both(report, "\n4. CLUSTER COMPOSITION BY DAY")
        write_both(report, "-" * 80)

        cluster_counts = pd.crosstab(
            metadata[DAY_COLUMN],
            metadata[CLUSTER_COLUMN],
        ).reindex(days, fill_value=0)

        cluster_props = cluster_counts.div(
            cluster_counts.sum(axis=1),
            axis=0,
        )

        save_table(
            cluster_counts,
            "03_cluster_counts_by_day.csv",
            index=True,
        )

        save_table(
            cluster_props,
            "04_cluster_proportions_by_day.csv",
            index=True,
        )

        write_both(
            report,
            "\nCluster counts by day:\n"
            + cluster_counts.to_string()
        )

        plot_cluster_proportions(
            cluster_props,
            FIGURE_DIR / "03_cluster_proportions_by_day.png",
            "Cell-state composition across Stage-5 experimental days",
        )

        # =============================================================
        # 5. Cluster composition by day and differentiation
        # =============================================================

        write_both(
            report,
            "\n5. CLUSTER COMPOSITION WITHIN EACH DIFFERENTIATION"
        )
        write_both(report, "-" * 80)

        for diff in sorted(metadata[DIFF_COLUMN].unique()):

            subset = metadata[
                metadata[DIFF_COLUMN] == diff
            ].copy()

            diff_counts = pd.crosstab(
                subset[DAY_COLUMN],
                subset[CLUSTER_COLUMN],
            ).reindex(days, fill_value=0)

            diff_props = diff_counts.div(
                diff_counts.sum(axis=1).replace(0, np.nan),
                axis=0,
            ).fillna(0.0)

            save_table(
                diff_counts,
                f"05_cluster_counts_by_day_diff{diff}.csv",
                index=True,
            )

            save_table(
                diff_props,
                f"06_cluster_proportions_by_day_diff{diff}.csv",
                index=True,
            )

            plot_cluster_proportions(
                diff_props,
                FIGURE_DIR
                / f"04_cluster_proportions_by_day_diff{diff}.png",
                f"Cell-state composition: Differentiation {diff}",
            )

            write_both(
                report,
                f"\nDifferentiation {diff}: {len(subset):,} cells"
            )

        # =============================================================
        # 6. Subcluster composition
        # =============================================================

        write_both(report, "\n6. SUBCLUSTER COMPOSITION BY DAY")
        write_both(report, "-" * 80)

        subcluster_counts = pd.crosstab(
            metadata[DAY_COLUMN],
            metadata[SUBCLUSTER_COLUMN],
        ).reindex(days, fill_value=0)

        subcluster_props = subcluster_counts.div(
            subcluster_counts.sum(axis=1),
            axis=0,
        )

        save_table(
            subcluster_counts,
            "07_subcluster_counts_by_day.csv",
            index=True,
        )

        save_table(
            subcluster_props,
            "08_subcluster_proportions_by_day.csv",
            index=True,
        )

        write_both(
            report,
            f"Subcluster table saved with "
            f"{subcluster_counts.shape[1]} subclusters."
        )

        # =============================================================
        # 7. First appearance and temporal span
        # =============================================================

        write_both(report, "\n7. CLUSTER TEMPORAL APPEARANCE")
        write_both(report, "-" * 80)

        appearance_rows = []

        for cluster, group in metadata.groupby(CLUSTER_COLUMN):

            counts_by_day = (
                group.groupby(DAY_COLUMN)
                .size()
                .sort_index()
            )

            first_day = int(counts_by_day.index.min())
            last_day = int(counts_by_day.index.max())
            peak_day = int(counts_by_day.idxmax())
            peak_cells = int(counts_by_day.max())

            appearance_rows.append(
                {
                    "cluster": cluster,
                    "first_day": first_day,
                    "last_day": last_day,
                    "peak_day": peak_day,
                    "peak_cells": peak_cells,
                    "total_cells": int(len(group)),
                }
            )

        appearance = (
            pd.DataFrame(appearance_rows)
            .sort_values(
                ["first_day", "peak_day", "cluster"]
            )
            .reset_index(drop=True)
        )

        save_table(
            appearance,
            "09_cluster_temporal_appearance.csv",
            index=False,
        )

        write_both(
            report,
            appearance.to_string(index=False)
        )

        # =============================================================
        # 8. Selected endocrine trajectory states
        # =============================================================

        write_both(
            report,
            "\n8. SELECTED ENDOCRINE-TRAJECTORY STATES"
        )
        write_both(report, "-" * 80)

        present_endocrine_states = [
            state for state in ENDOCRINE_STATES
            if state in set(metadata[CLUSTER_COLUMN])
        ]

        state_counts = cluster_counts.reindex(
            columns=present_endocrine_states,
            fill_value=0,
        )

        state_props_all = state_counts.div(
            day_counts,
            axis=0,
        ).fillna(0.0)

        state_props_selected = state_counts.div(
            state_counts.sum(axis=1).replace(0, np.nan),
            axis=0,
        ).fillna(0.0)

        save_table(
            state_counts,
            "10_endocrine_state_counts_by_day.csv",
            index=True,
        )

        save_table(
            state_props_all,
            "11_endocrine_state_fraction_of_all_cells_by_day.csv",
            index=True,
        )

        save_table(
            state_props_selected,
            "12_endocrine_state_within_selected_states_by_day.csv",
            index=True,
        )

        write_both(
            report,
            "Selected states present:\n  "
            + "\n  ".join(present_endocrine_states)
        )

        write_both(
            report,
            "\nSelected endocrine-state counts by day:\n"
            + state_counts.to_string()
        )

        plot_selected_state_proportions(
            state_props_all,
            FIGURE_DIR / "05_endocrine_state_fraction_by_day.png",
        )

        # =============================================================
        # 9. Marker expression by day
        # =============================================================

        write_both(report, "\n9. MARKER EXPRESSION BY DAY")
        write_both(report, "-" * 80)

        marker_mean = (
            analysis.groupby(DAY_COLUMN)[marker_columns]
            .mean()
            .reindex(days)
        )

        marker_median = (
            analysis.groupby(DAY_COLUMN)[marker_columns]
            .median()
            .reindex(days)
        )

        marker_detection = (
            analysis.assign(
                **{
                    f"__det_{gene}": (
                        pd.to_numeric(
                            analysis[gene],
                            errors="coerce",
                        ).fillna(0) > 0
                    ).astype(float)
                    for gene in marker_columns
                }
            )
            .groupby(DAY_COLUMN)[
                [f"__det_{gene}" for gene in marker_columns]
            ]
            .mean()
            .reindex(days)
        )

        marker_detection.columns = marker_columns

        save_table(
            marker_mean,
            "13_marker_mean_expression_by_day.csv",
            index=True,
        )

        save_table(
            marker_median,
            "14_marker_median_expression_by_day.csv",
            index=True,
        )

        save_table(
            marker_detection,
            "15_marker_detection_fraction_by_day.csv",
            index=True,
        )

        write_both(
            report,
            "\nMarker detection fraction by day:\n"
            + marker_detection.round(4).to_string()
        )

        plot_marker_detection(
            marker_detection,
            FIGURE_DIR / "06_marker_detection_by_day.png",
        )

        plot_marker_mean(
            marker_mean,
            FIGURE_DIR / "07_marker_mean_expression_by_day.png",
        )

        # =============================================================
        # 10. Marker expression by day and differentiation
        # =============================================================

        write_both(
            report,
            "\n10. MARKER EXPRESSION BY DAY AND DIFFERENTIATION"
        )
        write_both(report, "-" * 80)

        marker_mean_diff = (
            analysis.groupby(
                [DAY_COLUMN, DIFF_COLUMN]
            )[marker_columns]
            .mean()
            .reset_index()
        )

        detection_rows = []

        for (day, diff), group in analysis.groupby(
            [DAY_COLUMN, DIFF_COLUMN]
        ):
            row = {
                "day": int(day),
                "differentiation": int(diff),
                "n_cells": int(len(group)),
            }

            for gene in marker_columns:
                values = pd.to_numeric(
                    group[gene],
                    errors="coerce",
                ).fillna(0)

                row[f"{gene}_detected_fraction"] = float(
                    (values > 0).mean()
                )

            detection_rows.append(row)

        marker_detection_diff = pd.DataFrame(
            detection_rows
        ).sort_values(
            ["day", "differentiation"]
        )

        save_table(
            marker_mean_diff,
            "16_marker_mean_by_day_differentiation.csv",
            index=False,
        )

        save_table(
            marker_detection_diff,
            "17_marker_detection_by_day_differentiation.csv",
            index=False,
        )

        write_both(
            report,
            "Day-by-differentiation marker tables saved."
        )

        # =============================================================
        # 11. Replicate similarity in cell-state composition
        # =============================================================

        write_both(
            report,
            "\n11. DIFFERENTIATION-TO-DIFFERENTIATION COMPOSITION AGREEMENT"
        )
        write_both(report, "-" * 80)

        diff_values = sorted(metadata[DIFF_COLUMN].unique())

        agreement_rows = []

        if len(diff_values) == 2:

            diff_a, diff_b = diff_values

            for day in days:

                day_a = metadata[
                    (metadata[DAY_COLUMN] == day)
                    & (metadata[DIFF_COLUMN] == diff_a)
                ]

                day_b = metadata[
                    (metadata[DAY_COLUMN] == day)
                    & (metadata[DIFF_COLUMN] == diff_b)
                ]

                prop_a = (
                    day_a[CLUSTER_COLUMN]
                    .value_counts(normalize=True)
                )

                prop_b = (
                    day_b[CLUSTER_COLUMN]
                    .value_counts(normalize=True)
                )

                states = sorted(
                    set(prop_a.index).union(prop_b.index)
                )

                va = np.array(
                    [prop_a.get(state, 0.0) for state in states],
                    dtype=float,
                )

                vb = np.array(
                    [prop_b.get(state, 0.0) for state in states],
                    dtype=float,
                )

                if len(states) >= 2 and np.std(va) > 0 and np.std(vb) > 0:
                    correlation = float(
                        np.corrcoef(va, vb)[0, 1]
                    )
                else:
                    correlation = np.nan

                l1_distance = float(np.abs(va - vb).sum())

                agreement_rows.append(
                    {
                        "day": day,
                        "n_cells_diff1": len(day_a),
                        "n_cells_diff2": len(day_b),
                        "cluster_proportion_correlation": correlation,
                        "cluster_proportion_L1_distance": l1_distance,
                    }
                )

            agreement = pd.DataFrame(agreement_rows)

            save_table(
                agreement,
                "18_differentiation_composition_agreement.csv",
                index=False,
            )

            write_both(
                report,
                agreement.round(4).to_string(index=False)
            )

        else:
            write_both(
                report,
                "Exactly two differentiations were not detected; "
                "agreement analysis skipped."
            )

        # =============================================================
        # 12. Endocrine pseudotime subset structure
        # =============================================================

        write_both(
            report,
            "\n12. ENDOCRINE PSEUDOTIME SUBSET"
        )
        write_both(report, "-" * 80)

        pseudo = pd.read_csv(
            PSEUDOTIME_FILE,
            sep="\t",
            compression="gzip",
            low_memory=False,
        )

        ensure_required_columns(
            pseudo,
            [
                "library.barcode",
                "Assigned_cluster",
                "Pseudotime_value",
                "Pseudotime_branch",
                "Differentiation",
                "CellDay",
            ],
            "Endocrine pseudotime metadata",
        )

        pseudo["CellDay"] = pd.to_numeric(
            pseudo["CellDay"],
            errors="raise",
        ).astype(int)

        pseudo_counts = pd.crosstab(
            pseudo["CellDay"],
            pseudo["Assigned_cluster"],
        )

        pseudo_branch_counts = pd.crosstab(
            pseudo["CellDay"],
            pseudo["Pseudotime_branch"],
        )

        save_table(
            pseudo_counts,
            "19_pseudotime_cluster_counts_by_day.csv",
            index=True,
        )

        save_table(
            pseudo_branch_counts,
            "20_pseudotime_branch_counts_by_day.csv",
            index=True,
        )

        write_both(
            report,
            f"Endocrine pseudotime cells: {len(pseudo):,}"
        )

        write_both(
            report,
            "\nPseudotime branch counts by day:\n"
            + pseudo_branch_counts.to_string()
        )

        # =============================================================
        # 13. Characterization summary
        # =============================================================

        write_both(report, "\n13. CHARACTERIZATION SUMMARY")
        write_both(report, "-" * 80)

        write_both(
            report,
            "The script completed descriptive characterization of:"
        )

        write_both(
            report,
            "  - experimental Day 0-7 sampling"
        )
        write_both(
            report,
            "  - the two independent differentiations"
        )
        write_both(
            report,
            "  - temporal cell-state composition"
        )
        write_both(
            report,
            "  - endocrine-state emergence"
        )
        write_both(
            report,
            "  - selected developmental marker behavior"
        )
        write_both(
            report,
            "  - the deposited endocrine pseudotime subset"
        )

        write_both(
            report,
            "\nNo new normalization, dimensionality reduction, "
            "trajectory reconstruction, or GDIS calculation was performed."
        )

        write_both(
            report,
            f"\nTables:  {TABLE_DIR}"
        )

        write_both(
            report,
            f"Figures: {FIGURE_DIR}"
        )

        write_both(
            report,
            f"Report:  {REPORT_FILE}"
        )

    print("\n" + "=" * 80)
    print("p3_dataset_characterization.py completed successfully.")
    print("=" * 80)
    print(f"Report : {REPORT_FILE}")
    print(f"Tables : {TABLE_DIR}")
    print(f"Figures: {FIGURE_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

