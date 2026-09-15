#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================
"""
p7_trajectory_assembly_validation.py

Trajectory-window assembly and validation for GDIS-Bio.

Dataset
-------
GSE114412 Stage-5 validated endocrine trajectory.

Purpose
-------
Convert the validated single-cell PCA state space into fixed,
pseudotime-ordered local state sequences suitable for the first GDIS-Bio
calculation.

IMPORTANT SCIENTIFIC INTERPRETATION
-----------------------------------
These are NOT longitudinal measurements of the same individual cell.
Each sequence is a pseudotemporal ordering of independent single cells
along the deposited endocrine trajectory.

Primary analysis
----------------
Each branch and independent differentiation is treated separately:

    Branch 0 / Differentiation 1  -> SC-EC lineage replicate 1
    Branch 0 / Differentiation 2  -> SC-EC lineage replicate 2
    Branch 1 / Differentiation 1  -> SC-beta lineage replicate 1
    Branch 1 / Differentiation 2  -> SC-beta lineage replicate 2

The primary windowing configuration is frozen BEFORE GDIS:

    400 cells per window
    100-cell step
    75% overlap

Two pre-specified sensitivity configurations are also assembled:

    300 cells / 75-cell step
    500 cells / 125-cell step

All configurations preserve 75% overlap.

For each window:
    - cells are sorted by deposited pseudotime
    - state vector = first 50 PCs
    - window parameter = median deposited pseudotime
    - experimental-day and biological-state composition are retained

Outputs
-------
1. Compressed .npz trajectory arrays for each configuration/group.
2. Window-level metadata tables.
3. Primary-window cell-membership table.
4. Coverage and ordering diagnostics.
5. Landmark-to-window mapping.
6. Validation figures.
7. Final PASS / REVIEW REQUIRED assessment.

NO GDIS IS CALCULATED HERE.
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

P5_DATA_DIR = (
    PROJECT_DIR
    / "results"
    / "p5_preprocessing_state_space"
    / "data"
)

STATE_SPACE_FILE = (
    P5_DATA_DIR
    / "state_space_pca_50.csv.gz"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
    / "p7_trajectory_assembly_validation"
)

TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR = RESULTS_DIR / "figures"
DATA_DIR = RESULTS_DIR / "data"

REPORT_FILE = (
    RESULTS_DIR
    / "p7_trajectory_assembly_validation_report.txt"
)

# Primary representation already frozen by p6/p6b.
N_PCS = 50
PC_COLUMNS = [f"PC{i}" for i in range(1, N_PCS + 1)]

# Primary + pre-specified sensitivity configurations.
WINDOW_CONFIGS = [
    {
        "name": "sensitivity_w300_s75",
        "window_size": 300,
        "step_size": 75,
        "primary": False,
    },
    {
        "name": "primary_w400_s100",
        "window_size": 400,
        "step_size": 100,
        "primary": True,
    },
    {
        "name": "sensitivity_w500_s125",
        "window_size": 500,
        "step_size": 125,
        "primary": False,
    },
]

PRIMARY_CONFIG_NAME = "primary_w400_s100"

# Expected biological interpretation frozen from p4/p4b.
BRANCH_ENDPOINTS = {
    0: "sc_ec",
    1: "sc_beta",
}

EXPECTED_STATE_SEQUENCE = {
    0: [
        "prog_nkx61",
        "neurog3_early",
        "neurog3_mid",
        "neurog3_late",
        "sc_ec",
    ],
    1: [
        "prog_nkx61",
        "neurog3_early",
        "neurog3_mid",
        "neurog3_late",
        "sc_beta",
    ],
}

# Validation criteria selected before GDIS.
MIN_WINDOWS_PER_GROUP = 30
MIN_CELL_COVERAGE_FRACTION = 0.98
MIN_STRICT_PARAMETER_INCREMENT_FRACTION = 0.95
MAX_PARAMETER_TIE_FRACTION = 0.05
MIN_EXPECTED_STATE_COVERAGE = 1.00
MIN_DAY_COVERAGE = 1.00

RANDOM_SEED = 0


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def write_both(report, text=""):
    """Print and write the same text."""
    print(text)
    report.write(text + "\n")
    report.flush()


def save_table(df, filename, index=True):
    """Save a dataframe under the p7 tables directory."""
    path = TABLE_DIR / filename
    df.to_csv(path, index=index)
    return path


def ensure_columns(df, columns, table_name):
    """Verify required columns."""
    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"{table_name} is missing required column(s): "
            + ", ".join(missing)
        )


def group_label(branch, differentiation):
    """Compact group name for file names and reports."""
    return f"branch{int(branch)}_diff{int(differentiation)}"


def config_label(window_size, step_size):
    """Readable window configuration label."""
    overlap = 1.0 - (step_size / window_size)

    return (
        f"w={window_size}, step={step_size}, "
        f"overlap={overlap:.0%}"
    )


def expected_window_count(n_cells, window_size, step_size):
    """Number of complete equal-cell windows."""
    if n_cells < window_size:
        return 0

    return ((n_cells - window_size) // step_size) + 1


def assemble_windows(
    group_df,
    branch,
    differentiation,
    config_name,
    window_size,
    step_size,
):
    """
    Assemble equal-cell pseudotime windows for one branch/replicate.

    Returns
    -------
    trajectories : ndarray
        Shape = n_windows x window_size x 50
    parameters : ndarray
        Median pseudotime for each window.
    window_table : DataFrame
        One row per window.
    membership_table : DataFrame
        One row per cell per window.
    coverage_table : DataFrame
        One row per cell in the group with number of windows containing it.
    """

    # Stable ordering: pseudotime first, barcode second for deterministic
    # treatment of any pseudotime ties.
    ordered = (
        group_df
        .sort_values(
            ["Pseudotime_value", "library.barcode"],
            kind="mergesort",
        )
        .reset_index(drop=True)
        .copy()
    )

    n_cells = len(ordered)

    n_windows = expected_window_count(
        n_cells,
        window_size,
        step_size,
    )

    if n_windows == 0:
        raise RuntimeError(
            f"{group_label(branch, differentiation)} has "
            f"{n_cells} cells, fewer than window size {window_size}."
        )

    trajectories = np.empty(
        (n_windows, window_size, N_PCS),
        dtype=np.float32,
    )

    parameters = np.empty(
        n_windows,
        dtype=np.float64,
    )

    coverage_counts = np.zeros(
        n_cells,
        dtype=np.int32,
    )

    window_rows = []
    membership_parts = []

    day_values_all = sorted(
        ordered["CellDay"].unique().tolist()
    )

    expected_states = EXPECTED_STATE_SEQUENCE[int(branch)]

    for window_idx in range(n_windows):

        start = window_idx * step_size
        end = start + window_size

        window = ordered.iloc[start:end].copy()

        coords = window[
            PC_COLUMNS
        ].to_numpy(
            dtype=np.float32,
            copy=True,
        )

        if coords.shape != (
            window_size,
            N_PCS,
        ):
            raise RuntimeError(
                f"Unexpected trajectory shape for "
                f"{group_label(branch, differentiation)} "
                f"window {window_idx}: {coords.shape}"
            )

        if not np.isfinite(coords).all():
            raise RuntimeError(
                f"Non-finite PCA values in "
                f"{group_label(branch, differentiation)} "
                f"window {window_idx}."
            )

        pseudotime = window[
            "Pseudotime_value"
        ].to_numpy(dtype=float)

        if np.any(np.diff(pseudotime) < 0):
            raise RuntimeError(
                f"Pseudotime ordering failure in "
                f"{group_label(branch, differentiation)} "
                f"window {window_idx}."
            )

        trajectories[window_idx] = coords

        parameter = float(
            np.median(pseudotime)
        )

        parameters[window_idx] = parameter

        coverage_counts[start:end] += 1

        state_counts = (
            window["Assigned_cluster"]
            .value_counts()
        )

        state_fractions = (
            window["Assigned_cluster"]
            .value_counts(normalize=True)
        )

        day_counts = (
            window["CellDay"]
            .value_counts()
        )

        day_fractions = (
            window["CellDay"]
            .value_counts(normalize=True)
        )

        row = {
            "config": config_name,
            "branch": int(branch),
            "differentiation": int(differentiation),
            "window_index": window_idx,
            "window_id": (
                f"{group_label(branch, differentiation)}"
                f"_window{window_idx:03d}"
            ),
            "start_rank": int(start),
            "end_rank_exclusive": int(end),
            "n_cells": int(len(window)),
            "parameter_median_pseudotime": parameter,
            "pseudotime_min": float(
                pseudotime.min()
            ),
            "pseudotime_q25": float(
                np.quantile(pseudotime, 0.25)
            ),
            "pseudotime_q75": float(
                np.quantile(pseudotime, 0.75)
            ),
            "pseudotime_max": float(
                pseudotime.max()
            ),
            "pseudotime_span": float(
                pseudotime.max()
                - pseudotime.min()
            ),
            "median_day": float(
                window["CellDay"].median()
            ),
            "min_day": int(
                window["CellDay"].min()
            ),
            "max_day": int(
                window["CellDay"].max()
            ),
            "n_unique_days": int(
                window["CellDay"].nunique()
            ),
            "dominant_state": str(
                state_counts.index[0]
            ),
            "dominant_state_fraction": float(
                state_fractions.iloc[0]
            ),
            "n_unique_states": int(
                window["Assigned_cluster"].nunique()
            ),
        }

        # Wide state fractions.
        for state in expected_states:
            row[f"state_fraction__{state}"] = float(
                state_fractions.get(
                    state,
                    0.0,
                )
            )

        # Wide day fractions for all days observed in this group.
        for day in day_values_all:
            row[f"day_fraction__{int(day)}"] = float(
                day_fractions.get(
                    day,
                    0.0,
                )
            )

        window_rows.append(row)

        membership = pd.DataFrame(
            {
                "config": config_name,
                "branch": int(branch),
                "differentiation": int(differentiation),
                "window_index": window_idx,
                "window_id": row["window_id"],
                "cell_rank_within_window":
                    np.arange(
                        window_size,
                        dtype=int,
                    ),
                "group_pseudotime_rank":
                    np.arange(
                        start,
                        end,
                        dtype=int,
                    ),
                "library.barcode":
                    window[
                        "library.barcode"
                    ].astype(str).to_numpy(),
                "Pseudotime_value":
                    pseudotime,
                "CellDay":
                    window[
                        "CellDay"
                    ].to_numpy(dtype=int),
                "Assigned_cluster":
                    window[
                        "Assigned_cluster"
                    ].astype(str).to_numpy(),
            }
        )

        membership_parts.append(
            membership
        )

    window_table = pd.DataFrame(
        window_rows
    )

    membership_table = pd.concat(
        membership_parts,
        ignore_index=True,
    )

    coverage_table = pd.DataFrame(
        {
            "config": config_name,
            "branch": int(branch),
            "differentiation": int(differentiation),
            "group_pseudotime_rank":
                np.arange(n_cells),
            "library.barcode":
                ordered[
                    "library.barcode"
                ].astype(str).to_numpy(),
            "Pseudotime_value":
                ordered[
                    "Pseudotime_value"
                ].to_numpy(dtype=float),
            "CellDay":
                ordered[
                    "CellDay"
                ].to_numpy(dtype=int),
            "Assigned_cluster":
                ordered[
                    "Assigned_cluster"
                ].astype(str).to_numpy(),
            "n_windows_containing_cell":
                coverage_counts,
        }
    )

    return (
        trajectories,
        parameters,
        window_table,
        membership_table,
        coverage_table,
    )


def build_landmark_mapping(
    group_df,
    window_table,
    branch,
    differentiation,
):
    """
    Map biological-state median pseudotime landmarks to the closest
    primary window parameter.
    """
    rows = []

    expected_states = EXPECTED_STATE_SEQUENCE[int(branch)]

    for state in expected_states:

        state_df = group_df[
            group_df[
                "Assigned_cluster"
            ] == state
        ]

        if state_df.empty:
            continue

        landmark = float(
            state_df[
                "Pseudotime_value"
            ].median()
        )

        distances = np.abs(
            window_table[
                "parameter_median_pseudotime"
            ].to_numpy(dtype=float)
            - landmark
        )

        nearest_pos = int(
            np.argmin(distances)
        )

        nearest = window_table.iloc[
            nearest_pos
        ]

        rows.append(
            {
                "branch": int(branch),
                "differentiation":
                    int(differentiation),
                "state": state,
                "state_n_cells":
                    int(len(state_df)),
                "state_median_pseudotime":
                    landmark,
                "nearest_window_index":
                    int(
                        nearest[
                            "window_index"
                        ]
                    ),
                "nearest_window_parameter":
                    float(
                        nearest[
                            "parameter_median_pseudotime"
                        ]
                    ),
                "absolute_parameter_difference":
                    float(
                        abs(
                            nearest[
                                "parameter_median_pseudotime"
                            ]
                            - landmark
                        )
                    ),
                "nearest_window_dominant_state":
                    str(
                        nearest[
                            "dominant_state"
                        ]
                    ),
                "nearest_window_median_day":
                    float(
                        nearest[
                            "median_day"
                        ]
                    ),
            }
        )

    return pd.DataFrame(rows)


def plot_window_parameter_coverage(
    primary_windows,
    output_path,
):
    """
    Plot primary window parameters for all four branch/replicate groups.
    """
    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for (
        branch,
        differentiation,
    ), group in primary_windows.groupby(
        [
            "branch",
            "differentiation",
        ]
    ):

        group = group.sort_values(
            "window_index"
        )

        ax.plot(
            group["window_index"],
            group[
                "parameter_median_pseudotime"
            ],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=(
                f"Branch {branch}, "
                f"Diff {differentiation}"
            ),
        )

    ax.set_xlabel(
        "Window index"
    )

    ax.set_ylabel(
        "Median deposited pseudotime"
    )

    ax.set_title(
        "Primary pseudotemporal window coverage"
    )

    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_window_span(
    primary_windows,
    output_path,
):
    """Plot pseudotime span of each primary window."""
    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for (
        branch,
        differentiation,
    ), group in primary_windows.groupby(
        [
            "branch",
            "differentiation",
        ]
    ):

        group = group.sort_values(
            "parameter_median_pseudotime"
        )

        ax.plot(
            group[
                "parameter_median_pseudotime"
            ],
            group[
                "pseudotime_span"
            ],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=(
                f"Branch {branch}, "
                f"Diff {differentiation}"
            ),
        )

    ax.set_xlabel(
        "Window median pseudotime"
    )

    ax.set_ylabel(
        "Window pseudotime span"
    )

    ax.set_title(
        "Adaptive pseudotime span of equal-cell windows"
    )

    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_dominant_state(
    primary_windows,
    output_path,
):
    """
    Plot dominant-state index across pseudotime.

    Uses expected biological order only for visualization.
    """
    state_order = {
        "prog_nkx61": 0,
        "neurog3_early": 1,
        "neurog3_mid": 2,
        "neurog3_late": 3,
        "sc_ec": 4,
        "sc_beta": 4,
    }

    state_labels = [
        "prog_nkx61",
        "neurog3_early",
        "neurog3_mid",
        "neurog3_late",
        "endpoint",
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for (
        branch,
        differentiation,
    ), group in primary_windows.groupby(
        [
            "branch",
            "differentiation",
        ]
    ):

        group = group.sort_values(
            "parameter_median_pseudotime"
        )

        y = (
            group[
                "dominant_state"
            ]
            .map(state_order)
            .to_numpy(dtype=float)
        )

        ax.plot(
            group[
                "parameter_median_pseudotime"
            ],
            y,
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=(
                f"Branch {branch}, "
                f"Diff {differentiation}"
            ),
        )

    ax.set_xlabel(
        "Window median pseudotime"
    )

    ax.set_ylabel(
        "Dominant biological state"
    )

    ax.set_yticks(
        list(
            range(
                len(state_labels)
            )
        ),
        state_labels,
    )

    ax.set_title(
        "Biological-state progression across primary windows"
    )

    ax.legend()
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

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not STATE_SPACE_FILE.exists():
        raise FileNotFoundError(
            f"Missing 50-PC state-space file: "
            f"{STATE_SPACE_FILE}"
        )

    # -----------------------------------------------------------------
    # Load frozen 50-PC state space
    # -----------------------------------------------------------------

    df = pd.read_csv(
        STATE_SPACE_FILE,
        compression="gzip",
        low_memory=False,
    )

    required_columns = [
        "library.barcode",
        "Assigned_cluster",
        "Pseudotime_value",
        "Pseudotime_branch",
        "Differentiation",
        "CellDay",
    ] + PC_COLUMNS

    ensure_columns(
        df,
        required_columns,
        "50-PC state space",
    )

    df["library.barcode"] = (
        df["library.barcode"].astype(str)
    )

    df["Pseudotime_value"] = pd.to_numeric(
        df["Pseudotime_value"],
        errors="raise",
    )

    df["Pseudotime_branch"] = pd.to_numeric(
        df["Pseudotime_branch"],
        errors="raise",
    ).astype(int)

    df["Differentiation"] = pd.to_numeric(
        df["Differentiation"],
        errors="raise",
    ).astype(int)

    df["CellDay"] = pd.to_numeric(
        df["CellDay"],
        errors="raise",
    ).astype(int)

    # -----------------------------------------------------------------
    # Analysis
    # -----------------------------------------------------------------

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as report:

        write_both(
            report,
            "=" * 90,
        )

        write_both(
            report,
            "GDIS-Bio Pseudotemporal Trajectory Assembly Validation",
        )

        write_both(
            report,
            "=" * 90,
        )

        write_both(
            report,
            "\nSCIENTIFIC INTERPRETATION:",
        )

        write_both(
            report,
            "These windows are pseudotime-ordered sequences of "
            "independent single cells, not longitudinal recordings "
            "of the same cell.",
        )

        write_both(
            report,
            "\nPrimary state space: 50 PCs",
        )

        write_both(
            report,
            "Primary windowing: "
            "400 cells/window, 100-cell step, 75% overlap",
        )

        write_both(
            report,
            "Pre-specified sensitivity windowing: "
            "300/75 and 500/125",
        )

        # =============================================================
        # 1. Input validation
        # =============================================================

        write_both(
            report,
            "\n1. INPUT VALIDATION",
        )

        write_both(
            report,
            "-" * 90,
        )

        write_both(
            report,
            f"Input cells: {len(df):,}",
        )

        write_both(
            report,
            f"PC dimensions: {N_PCS}",
        )

        write_both(
            report,
            f"Branches: "
            f"{sorted(df['Pseudotime_branch'].unique().tolist())}",
        )

        write_both(
            report,
            f"Differentiations: "
            f"{sorted(df['Differentiation'].unique().tolist())}",
        )

        write_both(
            report,
            f"Experimental days: "
            f"{sorted(df['CellDay'].unique().tolist())}",
        )

        finite_pass = bool(
            np.isfinite(
                df[
                    PC_COLUMNS
                ].to_numpy(dtype=float)
            ).all()
        )

        write_both(
            report,
            f"Finite PCA values: {finite_pass}",
        )

        # =============================================================
        # 2. Group sizes
        # =============================================================

        write_both(
            report,
            "\n2. BRANCH x DIFFERENTIATION GROUP SIZES",
        )

        write_both(
            report,
            "-" * 90,
        )

        group_sizes = (
            df.groupby(
                [
                    "Pseudotime_branch",
                    "Differentiation",
                ],
                observed=True,
            )
            .size()
            .reset_index(
                name="n_cells"
            )
            .rename(
                columns={
                    "Pseudotime_branch":
                        "branch",
                    "Differentiation":
                        "differentiation",
                }
            )
        )

        save_table(
            group_sizes,
            "01_group_sizes.csv",
            index=False,
        )

        write_both(
            report,
            group_sizes.to_string(
                index=False
            ),
        )

        # =============================================================
        # 3. Assemble every frozen configuration
        # =============================================================

        write_both(
            report,
            "\n3. WINDOW ASSEMBLY",
        )

        write_both(
            report,
            "-" * 90,
        )

        all_config_summaries = []
        all_primary_windows = []
        all_primary_memberships = []
        all_primary_coverage = []
        all_primary_landmarks = []
        all_window_tables = []

        groups = sorted(
            df[
                [
                    "Pseudotime_branch",
                    "Differentiation",
                ]
            ]
            .drop_duplicates()
            .itertuples(
                index=False,
                name=None,
            )
        )

        for config in WINDOW_CONFIGS:

            config_name = config["name"]
            window_size = config["window_size"]
            step_size = config["step_size"]

            config_dir = (
                DATA_DIR
                / config_name
            )

            config_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            write_both(
                report,
                f"\nConfiguration: {config_name}",
            )

            write_both(
                report,
                "  "
                + config_label(
                    window_size,
                    step_size,
                ),
            )

            for branch, differentiation in groups:

                group_df = df[
                    (
                        df[
                            "Pseudotime_branch"
                        ] == branch
                    )
                    & (
                        df[
                            "Differentiation"
                        ] == differentiation
                    )
                ].copy()

                (
                    trajectories,
                    parameters,
                    window_table,
                    membership_table,
                    coverage_table,
                ) = assemble_windows(
                    group_df=group_df,
                    branch=branch,
                    differentiation=differentiation,
                    config_name=config_name,
                    window_size=window_size,
                    step_size=step_size,
                )

                label = group_label(
                    branch,
                    differentiation,
                )

                # Save fixed arrays for p8.
                npz_path = (
                    config_dir
                    / f"{label}.npz"
                )

                np.savez_compressed(
                    npz_path,
                    trajectories=trajectories,
                    parameters=parameters,
                    branch=np.array(
                        int(branch),
                        dtype=np.int16,
                    ),
                    differentiation=np.array(
                        int(differentiation),
                        dtype=np.int16,
                    ),
                    window_size=np.array(
                        window_size,
                        dtype=np.int32,
                    ),
                    step_size=np.array(
                        step_size,
                        dtype=np.int32,
                    ),
                    n_pcs=np.array(
                        N_PCS,
                        dtype=np.int32,
                    ),
                )

                # Save window table per group/config.
                window_table.to_csv(
                    config_dir
                    / f"{label}_windows.csv",
                    index=False,
                )

                all_window_tables.append(
                    window_table
                )

                # Coverage diagnostics.
                covered = (
                    coverage_table[
                        "n_windows_containing_cell"
                    ] > 0
                )

                coverage_fraction = float(
                    covered.mean()
                )

                n_uncovered = int(
                    (~covered).sum()
                )

                parameter_diffs = np.diff(
                    parameters
                )

                strict_fraction = (
                    float(
                        np.mean(
                            parameter_diffs > 0
                        )
                    )
                    if len(
                        parameter_diffs
                    ) > 0
                    else np.nan
                )

                tie_fraction = (
                    float(
                        np.mean(
                            parameter_diffs == 0
                        )
                    )
                    if len(
                        parameter_diffs
                    ) > 0
                    else np.nan
                )

                nondecreasing_pass = bool(
                    np.all(
                        parameter_diffs >= 0
                    )
                )

                expected_states = set(
                    EXPECTED_STATE_SEQUENCE[
                        int(branch)
                    ]
                )

                covered_state_set = set(
                    coverage_table.loc[
                        covered,
                        "Assigned_cluster",
                    ]
                )

                expected_state_coverage = (
                    len(
                        expected_states.intersection(
                            covered_state_set
                        )
                    )
                    / len(expected_states)
                )

                group_days = set(
                    group_df[
                        "CellDay"
                    ].unique()
                )

                covered_days = set(
                    coverage_table.loc[
                        covered,
                        "CellDay",
                    ].unique()
                )

                day_coverage = (
                    len(
                        group_days.intersection(
                            covered_days
                        )
                    )
                    / len(group_days)
                    if len(group_days)
                    else np.nan
                )

                expected_n_windows = (
                    expected_window_count(
                        len(group_df),
                        window_size,
                        step_size,
                    )
                )

                summary_row = {
                    "config": config_name,
                    "primary": bool(
                        config["primary"]
                    ),
                    "window_size":
                        window_size,
                    "step_size":
                        step_size,
                    "overlap_fraction":
                        1.0
                        - (
                            step_size
                            / window_size
                        ),
                    "branch":
                        int(branch),
                    "differentiation":
                        int(differentiation),
                    "group_cells":
                        int(len(group_df)),
                    "expected_windows":
                        int(
                            expected_n_windows
                        ),
                    "assembled_windows":
                        int(
                            trajectories.shape[0]
                        ),
                    "trajectory_rows_per_window":
                        int(
                            trajectories.shape[1]
                        ),
                    "trajectory_dimensions":
                        int(
                            trajectories.shape[2]
                        ),
                    "cell_coverage_fraction":
                        coverage_fraction,
                    "n_uncovered_tail_cells":
                        n_uncovered,
                    "strict_parameter_increment_fraction":
                        strict_fraction,
                    "parameter_tie_fraction":
                        tie_fraction,
                    "parameters_nondecreasing":
                        nondecreasing_pass,
                    "expected_state_coverage_fraction":
                        expected_state_coverage,
                    "experimental_day_coverage_fraction":
                        day_coverage,
                    "parameter_min":
                        float(
                            parameters.min()
                        ),
                    "parameter_max":
                        float(
                            parameters.max()
                        ),
                    "median_window_pseudotime_span":
                        float(
                            window_table[
                                "pseudotime_span"
                            ].median()
                        ),
                    "max_window_pseudotime_span":
                        float(
                            window_table[
                                "pseudotime_span"
                            ].max()
                        ),
                    "npz_file":
                        str(npz_path),
                }

                all_config_summaries.append(
                    summary_row
                )

                write_both(
                    report,
                    f"  {label}: "
                    f"{len(group_df):,} cells -> "
                    f"{trajectories.shape[0]} windows; "
                    f"coverage={coverage_fraction:.4f}; "
                    f"parameter strict increments="
                    f"{strict_fraction:.4f}",
                )

                if config["primary"]:

                    all_primary_windows.append(
                        window_table
                    )

                    all_primary_memberships.append(
                        membership_table
                    )

                    all_primary_coverage.append(
                        coverage_table
                    )

                    landmark_table = (
                        build_landmark_mapping(
                            group_df=group_df,
                            window_table=window_table,
                            branch=branch,
                            differentiation=
                                differentiation,
                        )
                    )

                    all_primary_landmarks.append(
                        landmark_table
                    )

        config_summary = pd.DataFrame(
            all_config_summaries
        )

        save_table(
            config_summary,
            "02_window_configuration_summary.csv",
            index=False,
        )

        # Save all window tables in one file.
        all_windows = pd.concat(
            all_window_tables,
            ignore_index=True,
        )

        save_table(
            all_windows,
            "03_all_window_metadata.csv",
            index=False,
        )

        # =============================================================
        # 4. Primary-window outputs
        # =============================================================

        write_both(
            report,
            "\n4. PRIMARY WINDOW OUTPUTS",
        )

        write_both(
            report,
            "-" * 90,
        )

        primary_windows = pd.concat(
            all_primary_windows,
            ignore_index=True,
        )

        primary_membership = pd.concat(
            all_primary_memberships,
            ignore_index=True,
        )

        primary_coverage = pd.concat(
            all_primary_coverage,
            ignore_index=True,
        )

        primary_landmarks = pd.concat(
            all_primary_landmarks,
            ignore_index=True,
        )

        save_table(
            primary_windows,
            "04_primary_window_metadata.csv",
            index=False,
        )

        # Membership table can be moderately large; gzip it.
        membership_path = (
            TABLE_DIR
            / "05_primary_window_cell_membership.csv.gz"
        )

        primary_membership.to_csv(
            membership_path,
            index=False,
            compression="gzip",
        )

        save_table(
            primary_coverage,
            "06_primary_cell_coverage.csv",
            index=False,
        )

        save_table(
            primary_landmarks,
            "07_primary_state_landmark_window_mapping.csv",
            index=False,
        )

        write_both(
            report,
            f"Primary windows: "
            f"{len(primary_windows):,}",
        )

        write_both(
            report,
            f"Primary window-membership rows: "
            f"{len(primary_membership):,}",
        )

        primary_covered_unique_cells = int(
            primary_coverage.loc[
                primary_coverage[
                    "n_windows_containing_cell"
                ] > 0,
                "library.barcode",
            ].nunique()
        )

        write_both(
            report,
            f"Primary covered unique cells: "
            f"{primary_covered_unique_cells:,}",
        )

        # =============================================================
        # 5. Primary biological landmark mapping
        # =============================================================

        write_both(
            report,
            "\n5. BIOLOGICAL LANDMARKS MAPPED TO PRIMARY WINDOWS",
        )

        write_both(
            report,
            "-" * 90,
        )

        write_both(
            report,
            primary_landmarks.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =============================================================
        # 6. Cross-replicate window parameter agreement
        # =============================================================

        write_both(
            report,
            "\n6. CROSS-REPLICATE WINDOW-PARAMETER AGREEMENT",
        )

        write_both(
            report,
            "-" * 90,
        )

        agreement_rows = []

        for branch in sorted(
            primary_windows[
                "branch"
            ].unique()
        ):

            branch_df = primary_windows[
                primary_windows[
                    "branch"
                ] == branch
            ]

            diffs = sorted(
                branch_df[
                    "differentiation"
                ].unique()
            )

            if len(diffs) != 2:
                continue

            a = (
                branch_df[
                    branch_df[
                        "differentiation"
                    ] == diffs[0]
                ]
                .sort_values(
                    "parameter_median_pseudotime"
                )
                .reset_index(drop=True)
            )

            b = (
                branch_df[
                    branch_df[
                        "differentiation"
                    ] == diffs[1]
                ]
                .sort_values(
                    "parameter_median_pseudotime"
                )
                .reset_index(drop=True)
            )

            # Compare normalized window position because replicate groups
            # contain slightly different cell counts/window counts.
            grid = np.linspace(
                0.0,
                1.0,
                101,
            )

            pos_a = np.linspace(
                0.0,
                1.0,
                len(a),
            )

            pos_b = np.linspace(
                0.0,
                1.0,
                len(b),
            )

            interp_a = np.interp(
                grid,
                pos_a,
                a[
                    "parameter_median_pseudotime"
                ].to_numpy(dtype=float),
            )

            interp_b = np.interp(
                grid,
                pos_b,
                b[
                    "parameter_median_pseudotime"
                ].to_numpy(dtype=float),
            )

            corr = float(
                np.corrcoef(
                    interp_a,
                    interp_b,
                )[0, 1]
            )

            mae = float(
                np.mean(
                    np.abs(
                        interp_a
                        - interp_b
                    )
                )
            )

            agreement_rows.append(
                {
                    "branch":
                        int(branch),
                    "diff1_windows":
                        int(len(a)),
                    "diff2_windows":
                        int(len(b)),
                    "normalized_position_parameter_correlation":
                        corr,
                    "normalized_position_parameter_mae":
                        mae,
                }
            )

        agreement_table = pd.DataFrame(
            agreement_rows
        )

        save_table(
            agreement_table,
            "08_primary_cross_replicate_parameter_agreement.csv",
            index=False,
        )

        write_both(
            report,
            agreement_table.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =============================================================
        # 7. Primary validation checks
        # =============================================================

        write_both(
            report,
            "\n7. PRIMARY TRAJECTORY-ASSEMBLY QUALIFICATION",
        )

        write_both(
            report,
            "-" * 90,
        )

        primary_summary = config_summary[
            config_summary[
                "config"
            ] == PRIMARY_CONFIG_NAME
        ].copy()

        checks = []

        checks.append(
            (
                f"Every group has at least "
                f"{MIN_WINDOWS_PER_GROUP} complete windows",
                bool(
                    (
                        primary_summary[
                            "assembled_windows"
                        ]
                        >= MIN_WINDOWS_PER_GROUP
                    ).all()
                ),
            )
        )

        checks.append(
            (
                f"Every group covers at least "
                f"{MIN_CELL_COVERAGE_FRACTION:.0%} "
                f"of its cells",
                bool(
                    (
                        primary_summary[
                            "cell_coverage_fraction"
                        ]
                        >= MIN_CELL_COVERAGE_FRACTION
                    ).all()
                ),
            )
        )

        checks.append(
            (
                "Window parameters are non-decreasing "
                "in every group",
                bool(
                    primary_summary[
                        "parameters_nondecreasing"
                    ].all()
                ),
            )
        )

        checks.append(
            (
                f"Strict window-parameter increments occur in "
                f"at least "
                f"{MIN_STRICT_PARAMETER_INCREMENT_FRACTION:.0%} "
                f"of adjacent windows",
                bool(
                    (
                        primary_summary[
                            "strict_parameter_increment_fraction"
                        ]
                        >= MIN_STRICT_PARAMETER_INCREMENT_FRACTION
                    ).all()
                ),
            )
        )

        checks.append(
            (
                f"Parameter ties are at most "
                f"{MAX_PARAMETER_TIE_FRACTION:.0%}",
                bool(
                    (
                        primary_summary[
                            "parameter_tie_fraction"
                        ]
                        <= MAX_PARAMETER_TIE_FRACTION
                    ).all()
                ),
            )
        )

        checks.append(
            (
                "All expected biological states are represented "
                "among covered cells",
                bool(
                    (
                        primary_summary[
                            "expected_state_coverage_fraction"
                        ]
                        >= MIN_EXPECTED_STATE_COVERAGE
                    ).all()
                ),
            )
        )

        checks.append(
            (
                "All experimental days represented in each group "
                "remain represented among covered cells",
                bool(
                    (
                        primary_summary[
                            "experimental_day_coverage_fraction"
                        ]
                        >= MIN_DAY_COVERAGE
                    ).all()
                ),
            )
        )

        checks.append(
            (
                "All primary trajectory arrays have shape "
                "n_windows x 400 x 50",
                bool(
                    (
                        primary_summary[
                            "trajectory_rows_per_window"
                        ] == 400
                    ).all()
                    and (
                        primary_summary[
                            "trajectory_dimensions"
                        ] == 50
                    ).all()
                ),
            )
        )

        checks.append(
            (
                "Input PCA coordinates are finite",
                finite_pass,
            )
        )

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
            f"\nPrimary assembly checks passed: "
            f"{passed}/{len(checks)}",
        )

        # =============================================================
        # 8. Sensitivity configuration check
        # =============================================================

        write_both(
            report,
            "\n8. PRE-SPECIFIED WINDOW-SENSITIVITY CONFIGURATIONS",
        )

        write_both(
            report,
            "-" * 90,
        )

        sensitivity_table = (
            config_summary.groupby(
                [
                    "config",
                    "window_size",
                    "step_size",
                    "overlap_fraction",
                    "primary",
                ],
                observed=True,
            )
            .agg(
                minimum_windows=(
                    "assembled_windows",
                    "min",
                ),
                maximum_windows=(
                    "assembled_windows",
                    "max",
                ),
                minimum_cell_coverage=(
                    "cell_coverage_fraction",
                    "min",
                ),
                minimum_strict_parameter_increment_fraction=(
                    "strict_parameter_increment_fraction",
                    "min",
                ),
                maximum_parameter_tie_fraction=(
                    "parameter_tie_fraction",
                    "max",
                ),
            )
            .reset_index()
        )

        save_table(
            sensitivity_table,
            "09_window_sensitivity_configuration_summary.csv",
            index=False,
        )

        write_both(
            report,
            sensitivity_table.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =============================================================
        # 9. Figures
        # =============================================================

        write_both(
            report,
            "\n9. FIGURES",
        )

        write_both(
            report,
            "-" * 90,
        )

        plot_window_parameter_coverage(
            primary_windows,
            FIGURE_DIR
            / "01_primary_window_parameter_coverage.png",
        )

        plot_window_span(
            primary_windows,
            FIGURE_DIR
            / "02_primary_window_pseudotime_span.png",
        )

        plot_dominant_state(
            primary_windows,
            FIGURE_DIR
            / "03_primary_window_dominant_state_progression.png",
        )

        write_both(
            report,
            "Saved primary-window coverage, span, and "
            "biological-progression figures.",
        )

        # =============================================================
        # 10. Final status
        # =============================================================

        write_both(
            report,
            "\n10. FINAL STATUS",
        )

        write_both(
            report,
            "-" * 90,
        )

        if passed == len(checks):

            write_both(
                report,
                "FINAL STATUS: PASS",
            )

            write_both(
                report,
                "The four branch x differentiation pseudotemporal "
                "sequences have been assembled successfully.",
            )

            write_both(
                report,
                "Primary GDIS input is frozen at "
                "400 cells/window, 100-cell step, 50 PCs.",
            )

            write_both(
                report,
                "Sensitivity inputs are frozen at "
                "300/75 and 500/125 using the same 50-PC state space.",
            )

            write_both(
                report,
                "The next step may calculate GDIS without changing "
                "the windowing design based on observed GDIS values.",
            )

        else:

            write_both(
                report,
                "FINAL STATUS: REVIEW REQUIRED",
            )

            write_both(
                report,
                "One or more primary trajectory-assembly criteria "
                "require review before any GDIS calculation.",
            )

        write_both(
            report,
            "\nNO GDIS VALUES WERE CALCULATED.",
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

    print("\n" + "=" * 90)
    print("p7_trajectory_assembly_validation.py completed.")
    print("=" * 90)
    print(f"Report : {REPORT_FILE}")
    print(f"Tables : {TABLE_DIR}")
    print(f"Figures: {FIGURE_DIR}")
    print(f"Data   : {DATA_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

