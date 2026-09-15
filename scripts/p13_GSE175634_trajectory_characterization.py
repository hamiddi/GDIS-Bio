#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================
"""
p13_GSE175634_trajectory_characterization.py

Trajectory characterization for the GDIS-Bio external validation dataset
GSE175634.

Purpose
-------
Characterize the deposited metadata and diffusion pseudotime BEFORE any
external-validation trajectory is defined.

This script does NOT:
    - load the large count matrix
    - normalize expression
    - select HVGs
    - compute PCA
    - reconstruct pseudotime
    - assign cells to CM/CF branches
    - calculate GDIS

Main questions
--------------
1. How complete is deposited diffusion pseudotime by day, cell type,
   and individual?
2. Does deposited pseudotime progress monotonically with experimental day?
3. Are the biological annotations ordered consistently along pseudotime?
4. How reproducible are state-level pseudotime medians across individuals?
5. Which individuals contain sufficient representation of early states
   plus CM and/or CF terminal states for later independent validation?
6. Do CM and CF terminal states occupy distinct or overlapping
   pseudotime ranges?

Important
---------
The biological annotations are treated as deposited labels. This script
does not infer or alter cell identities.

Candidate developmental sequence examined descriptively:
    IPSC -> MES -> CMES -> PROG -> CM
                           -> CF

This sequence is used only for characterization and ordering checks.
No final branch assignment is made here.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR

DATASET_DIR = PROJECT_DIR / "raw_data" / "GSE175634"

METADATA_FILE = (
    DATASET_DIR
    / "GSE175634_cell_metadata.tsv.gz"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
    / "p13_external_GSE175634_trajectory_characterization"
)

TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR = RESULTS_DIR / "figures"

REPORT_FILE = (
    RESULTS_DIR
    / "p13_external_GSE175634_trajectory_characterization_report.txt"
)


# ---------------------------------------------------------------------
# Frozen descriptive definitions
# ---------------------------------------------------------------------

EXPECTED_TYPES = [
    "IPSC",
    "MES",
    "CMES",
    "PROG",
    "CM",
    "CF",
]

COMMON_EARLY_TYPES = [
    "IPSC",
    "MES",
    "CMES",
    "PROG",
]

CM_SEQUENCE = [
    "IPSC",
    "MES",
    "CMES",
    "PROG",
    "CM",
]

CF_SEQUENCE = [
    "IPSC",
    "MES",
    "CMES",
    "PROG",
    "CF",
]

DAY_ORDER = [
    "day0",
    "day1",
    "day3",
    "day5",
    "day7",
    "day11",
    "day15",
]

DAY_TO_NUMERIC = {
    "day0": 0,
    "day1": 1,
    "day3": 3,
    "day5": 5,
    "day7": 7,
    "day11": 11,
    "day15": 15,
}

# These are descriptive coverage thresholds only.
# They are NOT used to exclude individuals from any later analysis.
MIN_STATE_CELLS_FOR_COVERAGE_FLAG = 50
MIN_PSEUDOTIME_CELLS_PER_INDIVIDUAL = 500


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


def monotonic_fraction(values):
    """
    Fraction of adjacent finite values that are non-decreasing.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        return np.nan

    return float(
        np.mean(
            np.diff(values) >= 0
        )
    )


def state_profile_correlation(a, b, states):
    """
    Pearson correlation of matched state median pseudotimes.
    """
    av = []
    bv = []

    for state in states:
        if state not in a.index or state not in b.index:
            continue

        xa = a.loc[state]
        xb = b.loc[state]

        if np.isfinite(xa) and np.isfinite(xb):
            av.append(float(xa))
            bv.append(float(xb))

    if len(av) < 3:
        return np.nan

    if np.std(av) == 0 or np.std(bv) == 0:
        return np.nan

    return float(
        np.corrcoef(av, bv)[0, 1]
    )


def summarize_pseudotime(group):
    """
    Standard pseudotime summary for a group.
    """
    pseudo = pd.to_numeric(
        group["dpt_pseudotime"],
        errors="coerce",
    )

    finite = pseudo.dropna()

    result = {
        "n_cells": int(len(group)),
        "n_with_pseudotime": int(finite.size),
        "pseudotime_fraction": (
            float(finite.size / len(group))
            if len(group)
            else np.nan
        ),
    }

    if finite.empty:
        for name in [
            "min",
            "q25",
            "median",
            "mean",
            "q75",
            "max",
        ]:
            result[f"pseudotime_{name}"] = np.nan
    else:
        result.update(
            {
                "pseudotime_min": float(finite.min()),
                "pseudotime_q25": float(finite.quantile(0.25)),
                "pseudotime_median": float(finite.median()),
                "pseudotime_mean": float(finite.mean()),
                "pseudotime_q75": float(finite.quantile(0.75)),
                "pseudotime_max": float(finite.max()),
            }
        )

    return pd.Series(result)


def make_state_day_heatmap(counts, output_path):
    """
    Cell-count heatmap using matplotlib only.
    """
    matrix = counts.reindex(
        index=EXPECTED_TYPES,
        columns=DAY_ORDER,
        fill_value=0,
    )

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    im = ax.imshow(
        np.log10(
            matrix.to_numpy(dtype=float)
            + 1.0
        ),
        aspect="auto",
    )

    ax.set_xticks(
        np.arange(
            len(matrix.columns)
        ),
        matrix.columns,
    )

    ax.set_yticks(
        np.arange(
            len(matrix.index)
        ),
        matrix.index,
    )

    ax.set_xlabel(
        "Experimental day"
    )

    ax.set_ylabel(
        "Deposited cell type"
    )

    ax.set_title(
        "GSE175634 cell abundance by type and day\n"
        "(log10 count + 1)"
    )

    fig.colorbar(
        im,
        ax=ax,
        label="log10(count + 1)",
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def make_state_pseudotime_boxplot(metadata, output_path):
    """
    Pseudotime distribution by biological state.
    """
    data = []

    labels = []

    for state in EXPECTED_TYPES:

        values = pd.to_numeric(
            metadata.loc[
                metadata["type"] == state,
                "dpt_pseudotime",
            ],
            errors="coerce",
        ).dropna()

        if len(values) > 0:
            data.append(
                values.to_numpy(
                    dtype=float
                )
            )
            labels.append(state)

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    ax.boxplot(
        data,
        tick_labels=labels,
        showfliers=False,
    )

    ax.set_xlabel(
        "Deposited cell type"
    )

    ax.set_ylabel(
        "Deposited diffusion pseudotime"
    )

    ax.set_title(
        "Pseudotime distributions across deposited cell states"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def make_day_pseudotime_plot(day_summary, output_path):
    """
    Median pseudotime by experimental day.
    """
    ordered = day_summary.copy()

    ordered["day_numeric"] = (
        ordered[
            "diffday"
        ].map(
            DAY_TO_NUMERIC
        )
    )

    ordered = ordered.sort_values(
        "day_numeric"
    )

    fig, ax = plt.subplots(
        figsize=(8.5, 5.5)
    )

    ax.plot(
        ordered[
            "day_numeric"
        ],
        ordered[
            "pseudotime_median"
        ],
        marker="o",
        linewidth=1.8,
    )

    ax.set_xlabel(
        "Experimental differentiation day"
    )

    ax.set_ylabel(
        "Median deposited diffusion pseudotime"
    )

    ax.set_title(
        "Experimental day versus deposited pseudotime"
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


def make_individual_state_median_plot(
    individual_state,
    output_path,
):
    """
    Plot individual state-median profiles.
    """
    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for individual, group in individual_state.groupby(
        "individual"
    ):

        pivot = (
            group.set_index(
                "type"
            )[
                "pseudotime_median"
            ]
        )

        x = []
        y = []

        for idx, state in enumerate(
            EXPECTED_TYPES
        ):
            if (
                state in pivot.index
                and np.isfinite(
                    pivot.loc[state]
                )
            ):
                x.append(idx)
                y.append(
                    float(
                        pivot.loc[state]
                    )
                )

        if len(x) >= 3:
            ax.plot(
                x,
                y,
                marker="o",
                markersize=2.5,
                linewidth=0.8,
                alpha=0.55,
            )

    ax.set_xticks(
        np.arange(
            len(EXPECTED_TYPES)
        ),
        EXPECTED_TYPES,
    )

    ax.set_xlabel(
        "Deposited cell state"
    )

    ax.set_ylabel(
        "Median deposited pseudotime"
    )

    ax.set_title(
        "Individual-level biological-state pseudotime profiles"
    )

    ax.grid(
        alpha=0.20
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

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing metadata file: "
            f"{METADATA_FILE}"
        )

    metadata = pd.read_csv(
        METADATA_FILE,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )

    required_columns = [
        "cell",
        "diffday",
        "individual",
        "type",
        "dpt_pseudotime",
    ]

    missing = [
        col
        for col in required_columns
        if col not in metadata.columns
    ]

    if missing:
        raise ValueError(
            "Metadata is missing required columns: "
            + ", ".join(missing)
        )

    metadata["cell"] = (
        metadata[
            "cell"
        ].astype(str)
    )

    metadata["individual"] = (
        metadata[
            "individual"
        ].astype(str)
    )

    metadata["type"] = (
        metadata[
            "type"
        ].astype(str)
    )

    metadata["diffday"] = (
        metadata[
            "diffday"
        ].astype(str)
    )

    metadata[
        "dpt_pseudotime"
    ] = pd.to_numeric(
        metadata[
            "dpt_pseudotime"
        ],
        errors="coerce",
    )

    metadata["day_numeric"] = (
        metadata[
            "diffday"
        ].map(
            DAY_TO_NUMERIC
        )
    )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as report:

        write_both(
            report,
            "=" * 94,
        )

        write_both(
            report,
            "GDIS-Bio External Dataset Trajectory Characterization",
        )

        write_both(
            report,
            "Dataset: GSE175634",
        )

        write_both(
            report,
            "=" * 94,
        )

        write_both(
            report,
            "\nNo branch assignment, preprocessing, PCA, "
            "or GDIS is performed in p13.",
        )

        # =========================================================
        # 1. Basic metadata inventory
        # =========================================================

        write_both(
            report,
            "\n1. BASIC METADATA INVENTORY",
        )

        write_both(
            report,
            "-" * 94,
        )

        write_both(
            report,
            f"Cells: {len(metadata):,}",
        )

        write_both(
            report,
            f"Individuals: "
            f"{metadata['individual'].nunique():,}",
        )

        write_both(
            report,
            f"Cell types: "
            f"{sorted(metadata['type'].unique().tolist())}",
        )

        write_both(
            report,
            f"Experimental days: "
            f"{sorted(metadata['diffday'].unique().tolist())}",
        )

        pseudo_finite = (
            metadata[
                "dpt_pseudotime"
            ].notna()
        )

        write_both(
            report,
            f"Cells with deposited pseudotime: "
            f"{pseudo_finite.sum():,} / "
            f"{len(metadata):,} "
            f"({pseudo_finite.mean():.4f})",
        )

        # =========================================================
        # 2. Pseudotime missingness
        # =========================================================

        write_both(
            report,
            "\n2. PSEUDOTIME COMPLETENESS BY CELL TYPE",
        )

        write_both(
            report,
            "-" * 94,
        )

        by_type = (
            metadata.groupby(
                "type",
                observed=True,
            )
            .apply(
                summarize_pseudotime,
                include_groups=False,
            )
            .reset_index()
        )

        save_table(
            by_type,
            "01_pseudotime_summary_by_type.csv",
            index=False,
        )

        write_both(
            report,
            by_type.round(
                6
            ).to_string(
                index=False
            ),
        )

        write_both(
            report,
            "\n3. PSEUDOTIME COMPLETENESS BY EXPERIMENTAL DAY",
        )

        write_both(
            report,
            "-" * 94,
        )

        by_day = (
            metadata.groupby(
                "diffday",
                observed=True,
            )
            .apply(
                summarize_pseudotime,
                include_groups=False,
            )
            .reset_index()
        )

        by_day[
            "day_numeric"
        ] = by_day[
            "diffday"
        ].map(
            DAY_TO_NUMERIC
        )

        by_day = by_day.sort_values(
            "day_numeric"
        )

        save_table(
            by_day,
            "02_pseudotime_summary_by_day.csv",
            index=False,
        )

        write_both(
            report,
            by_day.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 4. Day vs pseudotime
        # =========================================================

        write_both(
            report,
            "\n4. EXPERIMENTAL DAY vs DEPOSITED PSEUDOTIME",
        )

        write_both(
            report,
            "-" * 94,
        )

        valid_day_pseudo = metadata[
            metadata[
                "day_numeric"
            ].notna()
            & metadata[
                "dpt_pseudotime"
            ].notna()
        ]

        overall_day_rho = safe_spearman(
            valid_day_pseudo[
                "day_numeric"
            ],
            valid_day_pseudo[
                "dpt_pseudotime"
            ],
        )

        write_both(
            report,
            f"Overall cell-level Spearman(day, pseudotime): "
            f"{overall_day_rho:.6f}",
        )

        day_medians = (
            by_day[
                "pseudotime_median"
            ].to_numpy(
                dtype=float
            )
        )

        day_monotonicity = (
            monotonic_fraction(
                day_medians
            )
        )

        write_both(
            report,
            f"Adjacent-day median non-decreasing fraction: "
            f"{day_monotonicity:.6f}",
        )

        # =========================================================
        # 5. Biological-state ordering
        # =========================================================

        write_both(
            report,
            "\n5. BIOLOGICAL-STATE PSEUDOTIME ORDERING",
        )

        write_both(
            report,
            "-" * 94,
        )

        state_summary = (
            by_type[
                by_type[
                    "type"
                ].isin(
                    EXPECTED_TYPES
                )
            ]
            .copy()
        )

        state_summary[
            "expected_order"
        ] = state_summary[
            "type"
        ].map(
            {
                state: i
                for i, state
                in enumerate(
                    EXPECTED_TYPES
                )
            }
        )

        state_summary = (
            state_summary.sort_values(
                "expected_order"
            )
        )

        write_both(
            report,
            state_summary[
                [
                    "type",
                    "n_cells",
                    "n_with_pseudotime",
                    "pseudotime_fraction",
                    "pseudotime_median",
                    "pseudotime_q25",
                    "pseudotime_q75",
                ]
            ].round(
                6
            ).to_string(
                index=False
            ),
        )

        cm_medians = (
            state_summary[
                state_summary[
                    "type"
                ].isin(
                    CM_SEQUENCE
                )
            ]
            .set_index(
                "type"
            )
            .reindex(
                CM_SEQUENCE
            )[
                "pseudotime_median"
            ]
            .to_numpy(
                dtype=float
            )
        )

        cf_medians = (
            state_summary[
                state_summary[
                    "type"
                ].isin(
                    CF_SEQUENCE
                )
            ]
            .set_index(
                "type"
            )
            .reindex(
                CF_SEQUENCE
            )[
                "pseudotime_median"
            ]
            .to_numpy(
                dtype=float
            )
        )

        cm_monotonicity = (
            monotonic_fraction(
                cm_medians
            )
        )

        cf_monotonicity = (
            monotonic_fraction(
                cf_medians
            )
        )

        write_both(
            report,
            f"\nCM sequence median monotonicity "
            f"(IPSC→MES→CMES→PROG→CM): "
            f"{cm_monotonicity:.6f}",
        )

        write_both(
            report,
            f"CF sequence median monotonicity "
            f"(IPSC→MES→CMES→PROG→CF): "
            f"{cf_monotonicity:.6f}",
        )

        # =========================================================
        # 6. Individual-level state coverage
        # =========================================================

        write_both(
            report,
            "\n6. INDIVIDUAL-LEVEL STATE COVERAGE",
        )

        write_both(
            report,
            "-" * 94,
        )

        individual_type_counts = (
            metadata[
                metadata[
                    "type"
                ].isin(
                    EXPECTED_TYPES
                )
            ]
            .groupby(
                [
                    "individual",
                    "type",
                ],
                observed=True,
            )
            .size()
            .unstack(
                fill_value=0
            )
            .reindex(
                columns=EXPECTED_TYPES,
                fill_value=0,
            )
        )

        individual_type_counts[
            "total_expected_state_cells"
        ] = individual_type_counts[
            EXPECTED_TYPES
        ].sum(
            axis=1
        )

        for state in EXPECTED_TYPES:
            individual_type_counts[
                f"has_{state}_{MIN_STATE_CELLS_FOR_COVERAGE_FLAG}"
            ] = (
                individual_type_counts[
                    state
                ]
                >= MIN_STATE_CELLS_FOR_COVERAGE_FLAG
            )

        individual_type_counts[
            "common_early_states_all_ge_threshold"
        ] = individual_type_counts[
            [
                f"has_{state}_{MIN_STATE_CELLS_FOR_COVERAGE_FLAG}"
                for state
                in COMMON_EARLY_TYPES
            ]
        ].all(
            axis=1
        )

        individual_type_counts[
            "candidate_CM_coverage_flag"
        ] = (
            individual_type_counts[
                "common_early_states_all_ge_threshold"
            ]
            & individual_type_counts[
                f"has_CM_{MIN_STATE_CELLS_FOR_COVERAGE_FLAG}"
            ]
        )

        individual_type_counts[
            "candidate_CF_coverage_flag"
        ] = (
            individual_type_counts[
                "common_early_states_all_ge_threshold"
            ]
            & individual_type_counts[
                f"has_CF_{MIN_STATE_CELLS_FOR_COVERAGE_FLAG}"
            ]
        )

        individual_type_counts = (
            individual_type_counts.reset_index()
        )

        save_table(
            individual_type_counts,
            "03_individual_state_cell_counts.csv",
            index=False,
        )

        display_cols = (
            [
                "individual",
            ]
            + EXPECTED_TYPES
            + [
                "candidate_CM_coverage_flag",
                "candidate_CF_coverage_flag",
            ]
        )

        write_both(
            report,
            individual_type_counts[
                display_cols
            ].to_string(
                index=False
            ),
        )

        write_both(
            report,
            "\nNOTE: these coverage flags are descriptive only "
            "and do not exclude any individual.",
        )

        # =========================================================
        # 7. Individual pseudotime coverage and day correlation
        # =========================================================

        write_both(
            report,
            "\n7. INDIVIDUAL-LEVEL PSEUDOTIME / DAY AGREEMENT",
        )

        write_both(
            report,
            "-" * 94,
        )

        individual_rows = []

        for individual, group in metadata.groupby(
            "individual",
            observed=True,
        ):

            valid = group[
                group[
                    "day_numeric"
                ].notna()
                & group[
                    "dpt_pseudotime"
                ].notna()
            ]

            day_rho = safe_spearman(
                valid[
                    "day_numeric"
                ],
                valid[
                    "dpt_pseudotime"
                ],
            )

            day_med = (
                valid.groupby(
                    "day_numeric",
                    observed=True,
                )[
                    "dpt_pseudotime"
                ]
                .median()
                .sort_index()
            )

            individual_rows.append(
                {
                    "individual":
                        individual,
                    "n_cells":
                        int(
                            len(group)
                        ),
                    "n_with_pseudotime":
                        int(
                            group[
                                "dpt_pseudotime"
                            ].notna().sum()
                        ),
                    "pseudotime_fraction":
                        float(
                            group[
                                "dpt_pseudotime"
                            ].notna().mean()
                        ),
                    "n_days":
                        int(
                            group[
                                "diffday"
                            ].nunique()
                        ),
                    "n_days_with_pseudotime":
                        int(
                            valid[
                                "diffday"
                            ].nunique()
                        ),
                    "spearman_day_vs_pseudotime":
                        day_rho,
                    "daily_median_monotonicity":
                        monotonic_fraction(
                            day_med.to_numpy(
                                dtype=float
                            )
                        ),
                    "has_at_least_min_pseudotime_cells":
                        bool(
                            group[
                                "dpt_pseudotime"
                            ].notna().sum()
                            >= MIN_PSEUDOTIME_CELLS_PER_INDIVIDUAL
                        ),
                }
            )

        individual_pseudo = pd.DataFrame(
            individual_rows
        )

        save_table(
            individual_pseudo,
            "04_individual_pseudotime_day_agreement.csv",
            index=False,
        )

        write_both(
            report,
            individual_pseudo.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 8. Individual state median pseudotimes
        # =========================================================

        write_both(
            report,
            "\n8. INDIVIDUAL BIOLOGICAL-STATE PSEUDOTIME PROFILES",
        )

        write_both(
            report,
            "-" * 94,
        )

        individual_state = (
            metadata[
                metadata[
                    "type"
                ].isin(
                    EXPECTED_TYPES
                )
            ]
            .groupby(
                [
                    "individual",
                    "type",
                ],
                observed=True,
            )
            .apply(
                summarize_pseudotime,
                include_groups=False,
            )
            .reset_index()
        )

        save_table(
            individual_state,
            "05_individual_state_pseudotime_summary.csv",
            index=False,
        )

        profile_rows = []

        global_state_profile = (
            state_summary.set_index(
                "type"
            )[
                "pseudotime_median"
            ]
        )

        for individual, group in individual_state.groupby(
            "individual",
            observed=True,
        ):

            profile = (
                group.set_index(
                    "type"
                )[
                    "pseudotime_median"
                ]
            )

            cm_corr = state_profile_correlation(
                profile,
                global_state_profile,
                CM_SEQUENCE,
            )

            cf_corr = state_profile_correlation(
                profile,
                global_state_profile,
                CF_SEQUENCE,
            )

            cm_values = np.asarray(
                [
                    profile.get(
                        state,
                        np.nan,
                    )
                    for state
                    in CM_SEQUENCE
                ],
                dtype=float,
            )

            cf_values = np.asarray(
                [
                    profile.get(
                        state,
                        np.nan,
                    )
                    for state
                    in CF_SEQUENCE
                ],
                dtype=float,
            )

            profile_rows.append(
                {
                    "individual":
                        individual,
                    "cm_state_profile_correlation_to_global":
                        cm_corr,
                    "cf_state_profile_correlation_to_global":
                        cf_corr,
                    "cm_state_median_monotonicity":
                        monotonic_fraction(
                            cm_values
                        ),
                    "cf_state_median_monotonicity":
                        monotonic_fraction(
                            cf_values
                        ),
                }
            )

        profile_agreement = pd.DataFrame(
            profile_rows
        )

        save_table(
            profile_agreement,
            "06_individual_state_profile_agreement.csv",
            index=False,
        )

        write_both(
            report,
            profile_agreement.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 9. Terminal CM vs CF pseudotime
        # =========================================================

        write_both(
            report,
            "\n9. TERMINAL CM vs CF PSEUDOTIME COMPARISON",
        )

        write_both(
            report,
            "-" * 94,
        )

        terminal = state_summary[
            state_summary[
                "type"
            ].isin(
                [
                    "CM",
                    "CF",
                ]
            )
        ][
            [
                "type",
                "n_cells",
                "n_with_pseudotime",
                "pseudotime_median",
                "pseudotime_q25",
                "pseudotime_q75",
                "pseudotime_min",
                "pseudotime_max",
            ]
        ]

        write_both(
            report,
            terminal.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 10. Figures
        # =========================================================

        write_both(
            report,
            "\n10. FIGURES",
        )

        write_both(
            report,
            "-" * 94,
        )

        state_day_counts = pd.crosstab(
            metadata[
                "type"
            ],
            metadata[
                "diffday"
            ],
        )

        make_state_day_heatmap(
            state_day_counts,
            FIGURE_DIR
            / "01_cell_type_by_day_heatmap.png",
        )

        make_state_pseudotime_boxplot(
            metadata[
                metadata[
                    "type"
                ].isin(
                    EXPECTED_TYPES
                )
            ],
            FIGURE_DIR
            / "02_pseudotime_by_cell_type.png",
        )

        make_day_pseudotime_plot(
            by_day,
            FIGURE_DIR
            / "03_day_vs_pseudotime_median.png",
        )

        make_individual_state_median_plot(
            individual_state,
            FIGURE_DIR
            / "04_individual_state_pseudotime_profiles.png",
        )

        write_both(
            report,
            "Saved cell-type/day, state-pseudotime, "
            "day-pseudotime, and individual-state figures.",
        )

        # =========================================================
        # 11. Descriptive qualification
        # =========================================================

        write_both(
            report,
            "\n11. TRAJECTORY-CHARACTERIZATION QUALIFICATION",
        )

        write_both(
            report,
            "-" * 94,
        )

        expected_type_presence = all(
            state
            in set(
                metadata[
                    "type"
                ]
            )
            for state
            in EXPECTED_TYPES
        )

        pseudo_fraction = float(
            metadata[
                "dpt_pseudotime"
            ].notna().mean()
        )

        checks = [
            (
                "All expected developmental cell types are present",
                expected_type_presence,
            ),
            (
                "At least 90% of all cells have deposited pseudotime",
                pseudo_fraction >= 0.90,
            ),
            (
                "Overall experimental day is positively associated "
                "with deposited pseudotime",
                np.isfinite(
                    overall_day_rho
                )
                and overall_day_rho > 0,
            ),
            (
                "Experimental-day median pseudotime is fully non-decreasing",
                np.isfinite(
                    day_monotonicity
                )
                and day_monotonicity == 1.0,
            ),
            (
                "CM candidate state medians are fully non-decreasing",
                np.isfinite(
                    cm_monotonicity
                )
                and cm_monotonicity == 1.0,
            ),
            (
                "CF candidate state medians are fully non-decreasing",
                np.isfinite(
                    cf_monotonicity
                )
                and cf_monotonicity == 1.0,
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
            f"\nCharacterization checks passed: "
            f"{passed}/{len(checks)}",
        )

        if passed == len(checks):
            write_both(
                report,
                "\nFINAL STATUS: PASS",
            )

            write_both(
                report,
                "The deposited biological annotations and diffusion "
                "pseudotime are structurally consistent with further "
                "external-validation trajectory design.",
            )
        else:
            write_both(
                report,
                "\nFINAL STATUS: REVIEW REQUIRED",
            )

            write_both(
                report,
                "At least one trajectory-ordering property should be "
                "reviewed before defining CM/CF validation trajectories.",
            )

        write_both(
            report,
            "\nIMPORTANT: p13 does not assign cells to CM or CF "
            "trajectory branches. Branch construction must be justified "
            "after reviewing these characterization results.",
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

    print("\n" + "=" * 94)
    print("p13_GSE175634_trajectory_characterization.py completed.")
    print("=" * 94)
    print(f"Report : {REPORT_FILE}")
    print(f"Tables : {TABLE_DIR}")
    print(f"Figures: {FIGURE_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

