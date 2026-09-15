#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================
"""
p17_GSE175634_trajectory_window_assembly.py

Assemble and validate pseudotemporal trajectory windows for the frozen
GSE175634 external-validation design.

NO GDIS values are calculated in this script.

Transferred primary analysis choices
------------------------------------
State space:
    50 PCs, frozen after p16 geometry validation.

Windowing:
    Primary:     400 cells/window, 100-cell step
    Sensitivity: 300 cells/window,  75-cell step
    Sensitivity: 500 cells/window, 125-cell step

These are transferred from the discovery analysis and are NOT optimized on
GSE175634.

Independent replicate unit
--------------------------
INDIVIDUAL.

Trajectory scopes frozen in p14
-------------------------------
1. shared_backbone:
       IPSC -> MES -> CMES -> PROG
       19 eligible individuals

2. cm_extension:
       IPSC -> MES -> CMES -> PROG -> CM
       16 eligible individuals

3. cf_extension:
       IPSC -> MES -> CMES -> PROG -> CF
       15 eligible individuals

Important biological interpretation
-----------------------------------
The rows inside each window are independent single cells ordered by deposited
diffusion pseudotime. They are NOT repeated longitudinal measurements of one
cell.

Common early cells are not assigned an eventual CM or CF fate. The same
shared-backbone cells may therefore appear in both terminal-extension analyses
for an eligible individual. This is deliberate and follows the p14 frozen
design.

Window parameter
----------------
For each window:
    parameter = median deposited diffusion pseudotime

Rows inside each window are sorted by:
    deposited diffusion pseudotime, then cell ID

A parameter family must be strictly increasing for later pyGDIS use.

Outputs
-------
For each configuration / scope / individual:
    .npz trajectory array
    .csv window metadata

Aggregate summary, landmark mapping, configuration sensitivity, and
qualification tables are also written.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR

P14_DIR = (
    PROJECT_DIR
    / "results"
    / "p14_external_GSE175634_design_freeze"
)

P14_TABLE_DIR = P14_DIR / "tables"

ELIGIBILITY_FILE = (
    P14_TABLE_DIR
    / "01_frozen_individual_eligibility.csv"
)

INDIVIDUAL_LANDMARK_FILE = (
    P14_TABLE_DIR
    / "03_frozen_state_landmarks_by_individual.csv"
)

TRANSITION_FILE = (
    P14_TABLE_DIR
    / "05_prespecified_biological_transitions.csv"
)

P15_DATA_DIR = (
    PROJECT_DIR
    / "results"
    / "p15_external_GSE175634_preprocessing_state_space"
    / "data"
)

STATE_SPACE_FILE = (
    P15_DATA_DIR
    / "external_state_space_pca_50.csv.gz"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
    / "p17_external_GSE175634_trajectory_window_assembly"
)

DATA_DIR = RESULTS_DIR / "data"
TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR = RESULTS_DIR / "figures"

REPORT_FILE = (
    RESULTS_DIR
    / "p17_external_GSE175634_trajectory_window_assembly_report.txt"
)


# ---------------------------------------------------------------------
# Frozen trajectory design
# ---------------------------------------------------------------------

PRIMARY_DIMENSION = 50

WINDOW_CONFIGS = {
    "sensitivity_w300_s75": {
        "window_size": 300,
        "step_size": 75,
    },
    "primary_w400_s100": {
        "window_size": 400,
        "step_size": 100,
    },
    "sensitivity_w500_s125": {
        "window_size": 500,
        "step_size": 125,
    },
}

PRIMARY_CONFIG = "primary_w400_s100"

SCOPE_DEFINITIONS = {
    "shared_backbone": {
        "states": [
            "IPSC",
            "MES",
            "CMES",
            "PROG",
        ],
        "eligibility_column":
            "shared_backbone_eligible",
        "landmark_states": [
            "IPSC",
            "MES",
            "CMES",
            "PROG",
        ],
        "primary_transition_id":
            "T2",
    },
    "cm_extension": {
        "states": [
            "IPSC",
            "MES",
            "CMES",
            "PROG",
            "CM",
        ],
        "eligibility_column":
            "cm_extension_eligible",
        "landmark_states": [
            "IPSC",
            "MES",
            "CMES",
            "PROG",
            "CM",
        ],
        "primary_transition_id":
            "T3_CM",
    },
    "cf_extension": {
        "states": [
            "IPSC",
            "MES",
            "CMES",
            "PROG",
            "CF",
        ],
        "eligibility_column":
            "cf_extension_eligible",
        "landmark_states": [
            "IPSC",
            "MES",
            "CMES",
            "PROG",
            "CF",
        ],
        "primary_transition_id":
            "T3_CF",
    },
}

DAY_TO_NUMERIC = {
    "day0": 0,
    "day1": 1,
    "day3": 3,
    "day5": 5,
    "day7": 7,
    "day11": 11,
    "day15": 15,
}

PC_COLUMNS = [
    f"PC{i}"
    for i in range(
        1,
        PRIMARY_DIMENSION + 1,
    )
]

# Qualification thresholds are assembly diagnostics, not tuning criteria.
MIN_PRIMARY_WINDOWS_PER_GROUP = 15
MIN_PRIMARY_CELL_COVERAGE = 0.95
MIN_STRICT_PARAMETER_INCREMENT_FRACTION = 0.99
MAX_PARAMETER_TIE_FRACTION = 0.01


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


def robust_bool(series):
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

    parsed = values.map(
        mapping
    )

    if parsed.isna().any():
        bad = sorted(
            values[
                parsed.isna()
            ].unique().tolist()
        )

        raise ValueError(
            f"Could not parse boolean values: {bad}"
        )

    return parsed.astype(bool)


def group_file_stem(
    scope,
    individual,
):
    return (
        f"{scope}_"
        f"individual_{individual}"
    )


def deterministic_mode(series):
    """
    Deterministic categorical mode.

    If multiple categories tie, use lexicographically smallest label.
    """
    counts = (
        series.astype(str)
        .value_counts()
    )

    maximum = counts.max()

    tied = sorted(
        counts[
            counts == maximum
        ].index.tolist()
    )

    return tied[0]


def state_fraction_columns(
    state_series,
    expected_states,
):
    n = len(
        state_series
    )

    result = {}

    counts = (
        state_series.astype(str)
        .value_counts()
    )

    for state in expected_states:

        count = int(
            counts.get(
                state,
                0,
            )
        )

        result[
            f"n_{state}"
        ] = count

        result[
            f"fraction_{state}"
        ] = (
            count / n
            if n
            else np.nan
        )

    return result


def assemble_windows(
    group,
    expected_states,
    window_size,
    step_size,
):
    """
    Assemble complete equal-cell windows.

    The final incomplete tail is not padded or duplicated.
    """
    ordered = (
        group.sort_values(
            [
                "dpt_pseudotime",
                "cell",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    n_cells = len(
        ordered
    )

    if n_cells < window_size:
        return (
            np.empty(
                (
                    0,
                    window_size,
                    PRIMARY_DIMENSION,
                ),
                dtype=np.float32,
            ),
            pd.DataFrame(),
            ordered,
        )

    starts = list(
        range(
            0,
            n_cells
            - window_size
            + 1,
            step_size,
        )
    )

    trajectories = np.empty(
        (
            len(starts),
            window_size,
            PRIMARY_DIMENSION,
        ),
        dtype=np.float32,
    )

    metadata_rows = []

    day_numeric_all = (
        ordered[
            "diffday"
        ].map(
            DAY_TO_NUMERIC
        )
    )

    if day_numeric_all.isna().any():
        bad = sorted(
            ordered.loc[
                day_numeric_all.isna(),
                "diffday",
            ].astype(str).unique().tolist()
        )

        raise ValueError(
            "Unknown experimental-day labels: "
            + str(bad)
        )

    ordered = ordered.copy()

    ordered[
        "_day_numeric"
    ] = day_numeric_all.astype(
        float
    )

    for window_index, start in enumerate(
        starts
    ):

        stop = (
            start
            + window_size
        )

        window = ordered.iloc[
            start:stop
        ]

        X = window[
            PC_COLUMNS
        ].to_numpy(
            dtype=np.float32,
            copy=True,
        )

        if X.shape != (
            window_size,
            PRIMARY_DIMENSION,
        ):
            raise RuntimeError(
                "Unexpected window matrix shape."
            )

        if not np.isfinite(
            X
        ).all():
            raise ValueError(
                "Non-finite PCA values in a trajectory window."
            )

        trajectories[
            window_index,
            :,
            :,
        ] = X

        pseudo = window[
            "dpt_pseudotime"
        ].to_numpy(
            dtype=float
        )

        days = window[
            "_day_numeric"
        ].to_numpy(
            dtype=float
        )

        states = window[
            "type"
        ].astype(str)

        dominant_state = (
            deterministic_mode(
                states
            )
        )

        dominant_fraction = float(
            (
                states
                == dominant_state
            ).mean()
        )

        row = {
            "window_index":
                window_index,
            "start_rank":
                start,
            "end_rank_exclusive":
                stop,
            "n_cells":
                window_size,
            "parameter_median_pseudotime":
                float(
                    np.median(
                        pseudo
                    )
                ),
            "pseudotime_min":
                float(
                    np.min(
                        pseudo
                    )
                ),
            "pseudotime_max":
                float(
                    np.max(
                        pseudo
                    )
                ),
            "pseudotime_span":
                float(
                    np.max(
                        pseudo
                    )
                    - np.min(
                        pseudo
                    )
                ),
            "median_day_numeric":
                float(
                    np.median(
                        days
                    )
                ),
            "day_min_numeric":
                float(
                    np.min(
                        days
                    )
                ),
            "day_max_numeric":
                float(
                    np.max(
                        days
                    )
                ),
            "dominant_state":
                dominant_state,
            "dominant_state_fraction":
                dominant_fraction,
            "first_cell":
                str(
                    window.iloc[
                        0
                    ][
                        "cell"
                    ]
                ),
            "last_cell":
                str(
                    window.iloc[
                        -1
                    ][
                        "cell"
                    ]
                ),
        }

        row.update(
            state_fraction_columns(
                states,
                expected_states,
            )
        )

        metadata_rows.append(
            row
        )

    window_meta = pd.DataFrame(
        metadata_rows
    )

    return (
        trajectories,
        window_meta,
        ordered,
    )


def coverage_from_window_metadata(
    window_meta,
    ordered,
):
    """
    Unique-cell coverage across all complete windows.
    """
    n_cells = len(
        ordered
    )

    if (
        n_cells == 0
        or window_meta.empty
    ):
        return (
            0,
            0.0,
        )

    covered = np.zeros(
        n_cells,
        dtype=bool,
    )

    for _, row in (
        window_meta.iterrows()
    ):

        start = int(
            row[
                "start_rank"
            ]
        )

        stop = int(
            row[
                "end_rank_exclusive"
            ]
        )

        covered[
            start:stop
        ] = True

    n_covered = int(
        covered.sum()
    )

    return (
        n_covered,
        float(
            n_covered
            / n_cells
        ),
    )


def parameter_increment_diagnostics(
    window_meta,
):
    if len(
        window_meta
    ) < 2:
        return {
            "strict_increment_fraction":
                np.nan,
            "tie_fraction":
                np.nan,
            "parameters_non_decreasing":
                False,
            "parameters_strictly_increasing":
                False,
        }

    parameters = window_meta[
        "parameter_median_pseudotime"
    ].to_numpy(
        dtype=float
    )

    increments = np.diff(
        parameters
    )

    return {
        "strict_increment_fraction":
            float(
                np.mean(
                    increments > 0
                )
            ),
        "tie_fraction":
            float(
                np.mean(
                    increments == 0
                )
            ),
        "parameters_non_decreasing":
            bool(
                np.all(
                    increments >= 0
                )
            ),
        "parameters_strictly_increasing":
            bool(
                np.all(
                    increments > 0
                )
            ),
    }


def nearest_window_to_landmark(
    window_meta,
    landmark,
):
    parameters = window_meta[
        "parameter_median_pseudotime"
    ].to_numpy(
        dtype=float
    )

    position = int(
        np.argmin(
            np.abs(
                parameters
                - float(
                    landmark
                )
            )
        )
    )

    row = window_meta.iloc[
        position
    ]

    return {
        "nearest_window_index":
            int(
                row[
                    "window_index"
                ]
            ),
        "nearest_window_parameter":
            float(
                row[
                    "parameter_median_pseudotime"
                ]
            ),
        "absolute_parameter_difference":
            float(
                abs(
                    float(
                        row[
                            "parameter_median_pseudotime"
                        ]
                    )
                    - float(
                        landmark
                    )
                )
            ),
        "nearest_window_median_day":
            float(
                row[
                    "median_day_numeric"
                ]
            ),
        "nearest_window_dominant_state":
            str(
                row[
                    "dominant_state"
                ]
            ),
    }


def make_window_count_plot(
    primary_summary,
    output_path,
):
    plot_df = (
        primary_summary.sort_values(
            [
                "scope",
                "individual",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    x = np.arange(
        len(
            plot_df
        )
    )

    fig, ax = plt.subplots(
        figsize=(13, 6)
    )

    ax.bar(
        x,
        plot_df[
            "n_windows"
        ],
    )

    ax.set_xticks(
        x,
        (
            plot_df[
                "scope"
            ]
            + "\n"
            + plot_df[
                "individual"
            ].astype(str)
        ),
        rotation=90,
        fontsize=7,
    )

    ax.set_ylabel(
        "Number of complete primary windows"
    )

    ax.set_title(
        "GSE175634 frozen primary-window counts"
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


def make_coverage_plot(
    primary_summary,
    output_path,
):
    plot_df = (
        primary_summary.sort_values(
            [
                "scope",
                "individual",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    x = np.arange(
        len(
            plot_df
        )
    )

    fig, ax = plt.subplots(
        figsize=(13, 6)
    )

    ax.bar(
        x,
        plot_df[
            "cell_coverage_fraction"
        ],
    )

    ax.axhline(
        MIN_PRIMARY_CELL_COVERAGE,
        linestyle="--",
        linewidth=1,
        label=(
            f"Qualification threshold "
            f"{MIN_PRIMARY_CELL_COVERAGE:.2f}"
        ),
    )

    ax.set_xticks(
        x,
        (
            plot_df[
                "scope"
            ]
            + "\n"
            + plot_df[
                "individual"
            ].astype(str)
        ),
        rotation=90,
        fontsize=7,
    )

    ax.set_ylim(
        0.90,
        1.005,
    )

    ax.set_ylabel(
        "Unique-cell coverage fraction"
    )

    ax.set_title(
        "Primary trajectory-window cell coverage"
    )

    ax.legend()

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
        ELIGIBILITY_FILE,
        INDIVIDUAL_LANDMARK_FILE,
        TRANSITION_FILE,
        STATE_SPACE_FILE,
    ]

    missing = [
        str(
            path
        )
        for path in required_files
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required inputs:\n"
            + "\n".join(
                missing
            )
        )

    eligibility = pd.read_csv(
        ELIGIBILITY_FILE,
        low_memory=False,
    )

    landmarks = pd.read_csv(
        INDIVIDUAL_LANDMARK_FILE,
        low_memory=False,
    )

    transitions = pd.read_csv(
        TRANSITION_FILE,
        low_memory=False,
    )

    state_space = pd.read_csv(
        STATE_SPACE_FILE,
        compression="gzip",
        low_memory=False,
    )

    # -----------------------------------------------------------------
    # Input validation
    # -----------------------------------------------------------------

    required_state_space_columns = [
        "cell",
        "individual",
        "type",
        "diffday",
        "dpt_pseudotime",
    ] + PC_COLUMNS

    missing_columns = [
        column
        for column in required_state_space_columns
        if column not in state_space.columns
    ]

    if missing_columns:
        raise ValueError(
            "State-space file missing columns: "
            + ", ".join(
                missing_columns
            )
        )

    state_space[
        "cell"
    ] = state_space[
        "cell"
    ].astype(str)

    state_space[
        "individual"
    ] = state_space[
        "individual"
    ].astype(str)

    state_space[
        "type"
    ] = state_space[
        "type"
    ].astype(str)

    state_space[
        "diffday"
    ] = state_space[
        "diffday"
    ].astype(str)

    state_space[
        "dpt_pseudotime"
    ] = pd.to_numeric(
        state_space[
            "dpt_pseudotime"
        ],
        errors="raise",
    )

    for column in PC_COLUMNS:
        state_space[
            column
        ] = pd.to_numeric(
            state_space[
                column
            ],
            errors="raise",
        ).astype(
            np.float32
        )

    if state_space[
        "cell"
    ].duplicated().any():
        raise ValueError(
            "Duplicate cell IDs detected in p15 state space."
        )

    if not np.isfinite(
        state_space[
            PC_COLUMNS
        ].to_numpy(
            dtype=np.float32
        )
    ).all():
        raise ValueError(
            "Non-finite PC values in state space."
        )

    eligibility[
        "individual"
    ] = eligibility[
        "individual"
    ].astype(str)

    for definition in (
        SCOPE_DEFINITIONS.values()
    ):

        column = definition[
            "eligibility_column"
        ]

        if column not in eligibility.columns:
            raise ValueError(
                f"Eligibility table missing {column}"
            )

        eligibility[
            column
        ] = robust_bool(
            eligibility[
                column
            ]
        )

    landmarks[
        "individual"
    ] = landmarks[
        "individual"
    ].astype(str)

    landmarks[
        "state"
    ] = landmarks[
        "state"
    ].astype(str)

    landmarks[
        "state_median_pseudotime"
    ] = pd.to_numeric(
        landmarks[
            "state_median_pseudotime"
        ],
        errors="raise",
    )

    # -----------------------------------------------------------------
    # Main assembly
    # -----------------------------------------------------------------

    all_summary_rows = []
    all_window_metadata = []
    all_landmark_rows = []
    manifest_rows = []

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as report:

        write_both(
            report,
            "=" * 100,
        )

        write_both(
            report,
            "GDIS-Bio External Validation Trajectory-Window Assembly",
        )

        write_both(
            report,
            "Dataset: GSE175634",
        )

        write_both(
            report,
            "=" * 100,
        )

        write_both(
            report,
            "\nNO GDIS VALUES ARE CALCULATED.",
        )

        write_both(
            report,
            "Rows within each trajectory window are independent cells "
            "ordered by deposited diffusion pseudotime.",
        )

        write_both(
            report,
            "\nTransferred window configurations:",
        )

        for config_name, config in (
            WINDOW_CONFIGS.items()
        ):

            write_both(
                report,
                f"  {config_name}: "
                f"{config['window_size']} cells/window, "
                f"{config['step_size']}-cell step",
            )

        write_both(
            report,
            "\nIndependent replicate unit: individual",
        )

        # =========================================================
        # 1. Frozen scope inventory
        # =========================================================

        write_both(
            report,
            "\n1. FROZEN TRAJECTORY-SCOPE INVENTORY",
        )

        write_both(
            report,
            "-" * 100,
        )

        for scope, definition in (
            SCOPE_DEFINITIONS.items()
        ):

            eligible_ids = (
                eligibility.loc[
                    eligibility[
                        definition[
                            "eligibility_column"
                        ]
                    ],
                    "individual",
                ]
                .astype(str)
                .tolist()
            )

            write_both(
                report,
                f"{scope}: "
                f"{len(eligible_ids)} eligible individuals | "
                f"states={definition['states']}",
            )

        # =========================================================
        # 2. Assemble all configurations
        # =========================================================

        write_both(
            report,
            "\n2. TRAJECTORY ASSEMBLY",
        )

        write_both(
            report,
            "-" * 100,
        )

        for config_name, config in (
            WINDOW_CONFIGS.items()
        ):

            window_size = int(
                config[
                    "window_size"
                ]
            )

            step_size = int(
                config[
                    "step_size"
                ]
            )

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
                f"\nCONFIGURATION: "
                f"{config_name}",
            )

            for scope, definition in (
                SCOPE_DEFINITIONS.items()
            ):

                scope_dir = (
                    config_dir
                    / scope
                )

                scope_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                eligible_ids = (
                    eligibility.loc[
                        eligibility[
                            definition[
                                "eligibility_column"
                            ]
                        ],
                        "individual",
                    ]
                    .astype(str)
                    .tolist()
                )

                expected_states = (
                    definition[
                        "states"
                    ]
                )

                for individual in (
                    eligible_ids
                ):

                    group = state_space[
                        (
                            state_space[
                                "individual"
                            ] == individual
                        )
                        & (
                            state_space[
                                "type"
                            ].isin(
                                expected_states
                            )
                        )
                    ].copy()

                    if group.empty:
                        raise ValueError(
                            f"No cells for "
                            f"{scope}, individual {individual}"
                        )

                    observed_states = set(
                        group[
                            "type"
                        ].unique().tolist()
                    )

                    missing_states = [
                        state
                        for state in expected_states
                        if state
                        not in observed_states
                    ]

                    if missing_states:
                        raise ValueError(
                            f"{scope}, individual {individual}: "
                            f"missing expected states "
                            f"{missing_states}"
                        )

                    (
                        trajectories,
                        window_meta,
                        ordered,
                    ) = assemble_windows(
                        group=group,
                        expected_states=
                            expected_states,
                        window_size=
                            window_size,
                        step_size=
                            step_size,
                    )

                    if window_meta.empty:
                        raise ValueError(
                            f"No complete windows for "
                            f"{scope}, individual {individual}, "
                            f"{config_name}"
                        )

                    (
                        n_covered,
                        coverage_fraction,
                    ) = coverage_from_window_metadata(
                        window_meta,
                        ordered,
                    )

                    increment_info = (
                        parameter_increment_diagnostics(
                            window_meta
                        )
                    )

                    n_windows = int(
                        len(
                            window_meta
                        )
                    )

                    # Attach provenance columns before export.
                    window_meta.insert(
                        0,
                        "individual",
                        individual,
                    )

                    window_meta.insert(
                        0,
                        "scope",
                        scope,
                    )

                    window_meta.insert(
                        0,
                        "config",
                        config_name,
                    )

                    window_meta.insert(
                        3,
                        "window_size",
                        window_size,
                    )

                    window_meta.insert(
                        4,
                        "step_size",
                        step_size,
                    )

                    # Store compact arrays needed by pyGDIS.
                    stem = group_file_stem(
                        scope,
                        individual,
                    )

                    npz_path = (
                        scope_dir
                        / f"{stem}.npz"
                    )

                    metadata_path = (
                        scope_dir
                        / f"{stem}_windows.csv"
                    )

                    parameters = window_meta[
                        "parameter_median_pseudotime"
                    ].to_numpy(
                        dtype=np.float64
                    )

                    np.savez_compressed(
                        npz_path,
                        trajectories=
                            trajectories,
                        parameters=
                            parameters,
                        window_indices=
                            window_meta[
                                "window_index"
                            ].to_numpy(
                                dtype=np.int64
                            ),
                    )

                    window_meta.to_csv(
                        metadata_path,
                        index=False,
                    )

                    summary_row = {
                        "config":
                            config_name,
                        "scope":
                            scope,
                        "individual":
                            individual,
                        "window_size":
                            window_size,
                        "step_size":
                            step_size,
                        "n_cells":
                            int(
                                len(
                                    ordered
                                )
                            ),
                        "n_windows":
                            n_windows,
                        "n_covered_unique_cells":
                            n_covered,
                        "cell_coverage_fraction":
                            coverage_fraction,
                        "parameter_min":
                            float(
                                parameters.min()
                            ),
                        "parameter_max":
                            float(
                                parameters.max()
                            ),
                        "strict_increment_fraction":
                            increment_info[
                                "strict_increment_fraction"
                            ],
                        "tie_fraction":
                            increment_info[
                                "tie_fraction"
                            ],
                        "parameters_non_decreasing":
                            increment_info[
                                "parameters_non_decreasing"
                            ],
                        "parameters_strictly_increasing":
                            increment_info[
                                "parameters_strictly_increasing"
                            ],
                        "all_expected_states_present":
                            (
                                len(
                                    missing_states
                                ) == 0
                            ),
                        "trajectory_shape":
                            (
                                f"{trajectories.shape[0]}x"
                                f"{trajectories.shape[1]}x"
                                f"{trajectories.shape[2]}"
                            ),
                        "trajectory_values_finite":
                            bool(
                                np.isfinite(
                                    trajectories
                                ).all()
                            ),
                    }

                    all_summary_rows.append(
                        summary_row
                    )

                    all_window_metadata.append(
                        window_meta
                    )

                    manifest_rows.append(
                        {
                            "config":
                                config_name,
                            "scope":
                                scope,
                            "individual":
                                individual,
                            "npz_path":
                                str(
                                    npz_path
                                ),
                            "window_metadata_path":
                                str(
                                    metadata_path
                                ),
                            "n_windows":
                                n_windows,
                            "window_size":
                                window_size,
                            "n_dimensions":
                                PRIMARY_DIMENSION,
                        }
                    )

                    write_both(
                        report,
                        f"  {scope:16s} | "
                        f"individual {individual} | "
                        f"cells={len(ordered):5d} | "
                        f"windows={n_windows:3d} | "
                        f"coverage={coverage_fraction:.4f} | "
                        f"strict={increment_info['strict_increment_fraction']:.4f}",
                    )

        summary = pd.DataFrame(
            all_summary_rows
        )

        all_windows = pd.concat(
            all_window_metadata,
            ignore_index=True,
        )

        manifest = pd.DataFrame(
            manifest_rows
        )

        save_table(
            summary,
            "01_all_group_configuration_summary.csv",
            index=False,
        )

        save_table(
            manifest,
            "02_trajectory_file_manifest.csv",
            index=False,
        )

        all_windows.to_csv(
            TABLE_DIR
            / "03_all_window_metadata.csv.gz",
            index=False,
            compression="gzip",
        )

        primary_summary = (
            summary[
                summary[
                    "config"
                ] == PRIMARY_CONFIG
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        primary_windows = (
            all_windows[
                all_windows[
                    "config"
                ] == PRIMARY_CONFIG
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        save_table(
            primary_summary,
            "04_primary_group_summary.csv",
            index=False,
        )

        primary_windows.to_csv(
            TABLE_DIR
            / "05_primary_window_metadata.csv.gz",
            index=False,
            compression="gzip",
        )

        # =========================================================
        # 3. Map frozen biological landmarks to primary windows
        # =========================================================

        write_both(
            report,
            "\n3. FROZEN BIOLOGICAL-LANDMARK MAPPING "
            "TO PRIMARY WINDOWS",
        )

        write_both(
            report,
            "-" * 100,
        )

        for scope, definition in (
            SCOPE_DEFINITIONS.items()
        ):

            eligible_ids = (
                eligibility.loc[
                    eligibility[
                        definition[
                            "eligibility_column"
                        ]
                    ],
                    "individual",
                ]
                .astype(str)
                .tolist()
            )

            for individual in (
                eligible_ids
            ):

                group_windows = (
                    primary_windows[
                        (
                            primary_windows[
                                "scope"
                            ] == scope
                        )
                        & (
                            primary_windows[
                                "individual"
                            ] == individual
                        )
                    ]
                    .sort_values(
                        "window_index"
                    )
                    .reset_index(
                        drop=True
                    )
                )

                individual_landmarks = (
                    landmarks[
                        landmarks[
                            "individual"
                        ] == individual
                    ]
                    .set_index(
                        "state"
                    )
                )

                for state in (
                    definition[
                        "landmark_states"
                    ]
                ):

                    if (
                        state
                        not in individual_landmarks.index
                    ):
                        raise ValueError(
                            f"Missing frozen landmark "
                            f"{state} for individual "
                            f"{individual}"
                        )

                    landmark = float(
                        individual_landmarks.loc[
                            state,
                            "state_median_pseudotime",
                        ]
                    )

                    mapping = (
                        nearest_window_to_landmark(
                            group_windows,
                            landmark,
                        )
                    )

                    row = {
                        "scope":
                            scope,
                        "individual":
                            individual,
                        "state":
                            state,
                        "state_median_pseudotime":
                            landmark,
                    }

                    row.update(
                        mapping
                    )

                    all_landmark_rows.append(
                        row
                    )

        landmark_mapping = pd.DataFrame(
            all_landmark_rows
        )

        save_table(
            landmark_mapping,
            "06_primary_state_landmark_window_mapping.csv",
            index=False,
        )

        write_both(
            report,
            landmark_mapping.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 4. Pre-specified transition landmark coverage
        # =========================================================

        write_both(
            report,
            "\n4. PRE-SPECIFIED TRANSITION LANDMARK COVERAGE",
        )

        write_both(
            report,
            "-" * 100,
        )

        transition_rows = []

        transition_lookup = (
            transitions.set_index(
                "transition_id"
            )
        )

        for scope, definition in (
            SCOPE_DEFINITIONS.items()
        ):

            transition_id = (
                definition[
                    "primary_transition_id"
                ]
            )

            if (
                transition_id
                not in transition_lookup.index
            ):
                raise ValueError(
                    f"Missing transition "
                    f"{transition_id}"
                )

            destination_state = str(
                transition_lookup.loc[
                    transition_id,
                    "destination_state",
                ]
            )

            eligible_ids = (
                eligibility.loc[
                    eligibility[
                        definition[
                            "eligibility_column"
                        ]
                    ],
                    "individual",
                ]
                .astype(str)
                .tolist()
            )

            for individual in (
                eligible_ids
            ):

                summary_row = (
                    primary_summary[
                        (
                            primary_summary[
                                "scope"
                            ] == scope
                        )
                        & (
                            primary_summary[
                                "individual"
                            ] == individual
                        )
                    ]
                    .iloc[
                        0
                    ]
                )

                landmark_row = (
                    landmark_mapping[
                        (
                            landmark_mapping[
                                "scope"
                            ] == scope
                        )
                        & (
                            landmark_mapping[
                                "individual"
                            ] == individual
                        )
                        & (
                            landmark_mapping[
                                "state"
                            ] == destination_state
                        )
                    ]
                    .iloc[
                        0
                    ]
                )

                landmark_value = float(
                    landmark_row[
                        "state_median_pseudotime"
                    ]
                )

                parameter_min = float(
                    summary_row[
                        "parameter_min"
                    ]
                )

                parameter_max = float(
                    summary_row[
                        "parameter_max"
                    ]
                )

                inside = bool(
                    parameter_min
                    <= landmark_value
                    <= parameter_max
                )

                transition_rows.append(
                    {
                        "scope":
                            scope,
                        "individual":
                            individual,
                        "transition_id":
                            transition_id,
                        "destination_state":
                            destination_state,
                        "destination_state_median_pseudotime":
                            landmark_value,
                        "primary_parameter_min":
                            parameter_min,
                        "primary_parameter_max":
                            parameter_max,
                        "landmark_inside_primary_parameter_range":
                            inside,
                        "nearest_window_index":
                            int(
                                landmark_row[
                                    "nearest_window_index"
                                ]
                            ),
                        "nearest_window_parameter":
                            float(
                                landmark_row[
                                    "nearest_window_parameter"
                                ]
                            ),
                        "absolute_parameter_difference":
                            float(
                                landmark_row[
                                    "absolute_parameter_difference"
                                ]
                            ),
                        "nearest_window_median_day":
                            float(
                                landmark_row[
                                    "nearest_window_median_day"
                                ]
                            ),
                        "nearest_window_dominant_state":
                            str(
                                landmark_row[
                                    "nearest_window_dominant_state"
                                ]
                            ),
                    }
                )

        transition_coverage = pd.DataFrame(
            transition_rows
        )

        save_table(
            transition_coverage,
            "07_primary_transition_landmark_coverage.csv",
            index=False,
        )

        write_both(
            report,
            transition_coverage.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 5. Window-size sensitivity summary
        # =========================================================

        write_both(
            report,
            "\n5. WINDOW-SIZE SENSITIVITY SUMMARY",
        )

        write_both(
            report,
            "-" * 100,
        )

        sensitivity_summary = (
            summary.groupby(
                [
                    "config",
                    "scope",
                ],
                observed=True,
            )
            .agg(
                n_individuals=(
                    "individual",
                    "nunique",
                ),
                min_windows=(
                    "n_windows",
                    "min",
                ),
                median_windows=(
                    "n_windows",
                    "median",
                ),
                max_windows=(
                    "n_windows",
                    "max",
                ),
                min_cell_coverage=(
                    "cell_coverage_fraction",
                    "min",
                ),
                mean_cell_coverage=(
                    "cell_coverage_fraction",
                    "mean",
                ),
                min_strict_increment_fraction=(
                    "strict_increment_fraction",
                    "min",
                ),
                max_tie_fraction=(
                    "tie_fraction",
                    "max",
                ),
            )
            .reset_index()
        )

        save_table(
            sensitivity_summary,
            "08_window_size_sensitivity_summary.csv",
            index=False,
        )

        write_both(
            report,
            sensitivity_summary.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 6. Qualification
        # =========================================================

        write_both(
            report,
            "\n6. TRAJECTORY-ASSEMBLY QUALIFICATION",
        )

        write_both(
            report,
            "-" * 100,
        )

        expected_group_count = int(
            sum(
                eligibility[
                    definition[
                        "eligibility_column"
                    ]
                ].sum()
                for definition in (
                    SCOPE_DEFINITIONS.values()
                )
            )
        )

        primary_group_count = int(
            len(
                primary_summary
            )
        )

        all_expected_groups_present = (
            primary_group_count
            == expected_group_count
        )

        min_primary_windows = int(
            primary_summary[
                "n_windows"
            ].min()
        )

        min_primary_coverage = float(
            primary_summary[
                "cell_coverage_fraction"
            ].min()
        )

        min_strict_increment = float(
            summary[
                "strict_increment_fraction"
            ].min()
        )

        max_tie_fraction = float(
            summary[
                "tie_fraction"
            ].max()
        )

        all_parameters_non_decreasing = bool(
            summary[
                "parameters_non_decreasing"
            ].all()
        )

        all_primary_expected_states = bool(
            primary_summary[
                "all_expected_states_present"
            ].all()
        )

        all_arrays_finite = bool(
            summary[
                "trajectory_values_finite"
            ].all()
        )

        all_landmarks_inside = bool(
            transition_coverage[
                "landmark_inside_primary_parameter_range"
            ].all()
        )

        checks = [
            (
                "Every frozen scope x individual group is present",
                all_expected_groups_present,
            ),
            (
                f"Every primary group has >= "
                f"{MIN_PRIMARY_WINDOWS_PER_GROUP} complete windows",
                min_primary_windows
                >= MIN_PRIMARY_WINDOWS_PER_GROUP,
            ),
            (
                f"Every primary group has >= "
                f"{MIN_PRIMARY_CELL_COVERAGE:.2f} unique-cell coverage",
                min_primary_coverage
                >= MIN_PRIMARY_CELL_COVERAGE,
            ),
            (
                "Parameters are non-decreasing in every "
                "configuration/group",
                all_parameters_non_decreasing,
            ),
            (
                f"Minimum strict parameter-increment fraction >= "
                f"{MIN_STRICT_PARAMETER_INCREMENT_FRACTION:.2f}",
                min_strict_increment
                >= MIN_STRICT_PARAMETER_INCREMENT_FRACTION,
            ),
            (
                f"Maximum parameter-tie fraction <= "
                f"{MAX_PARAMETER_TIE_FRACTION:.2f}",
                max_tie_fraction
                <= MAX_PARAMETER_TIE_FRACTION,
            ),
            (
                "All expected biological states are represented "
                "in every primary group",
                all_primary_expected_states,
            ),
            (
                "All trajectory arrays contain finite 50-PC values",
                all_arrays_finite,
            ),
            (
                "Every scope's pre-specified primary transition "
                "landmark lies inside its primary parameter range",
                all_landmarks_inside,
            ),
        ]

        qualification_rows = []

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

            qualification_rows.append(
                {
                    "check":
                        label,
                    "pass":
                        bool(
                            status
                        ),
                }
            )

        save_table(
            pd.DataFrame(
                qualification_rows
            ),
            "09_trajectory_assembly_qualification.csv",
            index=False,
        )

        write_both(
            report,
            f"\nQualification checks passed: "
            f"{passed}/{len(checks)}",
        )

        # =========================================================
        # 7. Figures
        # =========================================================

        write_both(
            report,
            "\n7. FIGURES",
        )

        write_both(
            report,
            "-" * 100,
        )

        make_window_count_plot(
            primary_summary,
            FIGURE_DIR
            / "01_primary_window_counts.png",
        )

        make_coverage_plot(
            primary_summary,
            FIGURE_DIR
            / "02_primary_cell_coverage.png",
        )

        write_both(
            report,
            "Saved primary window-count and cell-coverage figures.",
        )

        # =========================================================
        # 8. Final status
        # =========================================================

        write_both(
            report,
            "\n8. FINAL STATUS",
        )

        write_both(
            report,
            "-" * 100,
        )

        write_both(
            report,
            f"Primary groups assembled: "
            f"{primary_group_count}",
        )

        write_both(
            report,
            f"Primary windows total: "
            f"{int(primary_summary['n_windows'].sum()):,}",
        )

        write_both(
            report,
            f"Minimum primary windows/group: "
            f"{min_primary_windows}",
        )

        write_both(
            report,
            f"Minimum primary unique-cell coverage: "
            f"{min_primary_coverage:.6f}",
        )

        write_both(
            report,
            f"Minimum strict parameter-increment fraction "
            f"across all configurations: "
            f"{min_strict_increment:.6f}",
        )

        write_both(
            report,
            f"Maximum parameter-tie fraction "
            f"across all configurations: "
            f"{max_tie_fraction:.6f}",
        )

        if (
            passed
            == len(
                checks
            )
        ):

            write_both(
                report,
                "\nFINAL STATUS: PASS",
            )

            write_both(
                report,
                "GSE175634 external-validation trajectory families "
                "have been assembled successfully using the frozen "
                "50-PC state space and transferred windowing design.",
            )

            write_both(
                report,
                "The next stage may run the unchanged pyGDIS "
                "formulation independently for each eligible individual "
                "and trajectory scope.",
            )

        else:

            write_both(
                report,
                "\nFINAL STATUS: REVIEW REQUIRED",
            )

            write_both(
                report,
                "Review the flagged assembly property before "
                "running external GDIS.",
            )

        write_both(
            report,
            "\nIMPORTANT:",
        )

        write_both(
            report,
            "No biological landmark was used to define a window.",
        )

        write_both(
            report,
            "No window size or step was optimized based on GDIS.",
        )

        write_both(
            report,
            "No common early cell was assigned an eventual CM or CF fate.",
        )

        write_both(
            report,
            "NO GDIS VALUES WERE CALCULATED.",
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

    print("\n" + "=" * 100)
    print("p17_GSE175634_trajectory_window_assembly.py completed.")
    print("=" * 100)
    print(f"Report : {REPORT_FILE}")
    print(f"Tables : {TABLE_DIR}")
    print(f"Figures: {FIGURE_DIR}")
    print(f"Data   : {DATA_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

