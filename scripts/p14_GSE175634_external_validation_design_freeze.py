#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================
"""
p14_GSE175634_external_validation_design_freeze.py

Freeze the external-validation cohort and biological trajectory design
for GDIS-Bio using GSE175634.

This script is metadata-only.

It DOES NOT:
    - load the count matrix
    - normalize expression
    - select genes
    - compute PCA
    - reconstruct pseudotime
    - calculate GDIS
    - tune any GDIS parameter

Scientific design
-----------------
The deposited annotations support a shared developmental backbone:

    IPSC -> MES -> CMES -> PROG

The terminal states are treated as separate extensions:

    shared backbone -> CM
    shared backbone -> CF

IMPORTANT:
The common early cells are NOT assigned an eventual CM or CF fate.
The same biological backbone may therefore be used in both terminal-extension
analyses, but this must not be interpreted as branch membership of the
individual early cells.

Primary external-validation analysis:
    Shared backbone across individuals.

Secondary external-validation analyses:
    CM terminal extension across eligible individuals.
    CF terminal extension across eligible individuals.

Biological transitions frozen BEFORE GDIS
-----------------------------------------
T1: IPSC -> MES
    Mesoderm induction.

T2: MES -> CMES
    Cardiac-mesoderm specification.
    This is the PRIMARY common transition for external validation.

T3-CM: PROG -> CM
    Cardiomyocyte terminal differentiation.

T3-CF: PROG -> CF
    Cardiac-fibroblast terminal differentiation.

The destination-state median pseudotime is retained as a reproducible
state landmark, NOT as a claim of true transition-onset time.

Pre-specified coverage rules
----------------------------
For individual-level replication:

Shared-backbone eligibility:
    >= 100 cells in EACH of IPSC, MES, CMES, PROG.

CM-extension eligibility:
    shared-backbone eligible AND >= 100 CM cells.

CF-extension eligibility:
    shared-backbone eligible AND >= 100 CF cells.

These thresholds are frozen here before expression preprocessing or GDIS.
They are coverage requirements, not biological optimization criteria.
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

DATASET_DIR = PROJECT_DIR / "raw_data" / "GSE175634"

METADATA_FILE = (
    DATASET_DIR
    / "GSE175634_cell_metadata.tsv.gz"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
    / "p14_external_GSE175634_design_freeze"
)

TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR = RESULTS_DIR / "figures"

REPORT_FILE = (
    RESULTS_DIR
    / "p14_external_GSE175634_design_freeze_report.txt"
)


# ---------------------------------------------------------------------
# Frozen design
# ---------------------------------------------------------------------

CORE_TYPES = [
    "IPSC",
    "MES",
    "CMES",
    "PROG",
]

CM_TYPES = CORE_TYPES + [
    "CM",
]

CF_TYPES = CORE_TYPES + [
    "CF",
]

KNOWN_TYPES = [
    "IPSC",
    "MES",
    "CMES",
    "PROG",
    "CM",
    "CF",
]

MIN_CORE_STATE_CELLS = 100
MIN_TERMINAL_STATE_CELLS = 100

TRANSITIONS = [
    {
        "transition_id": "T1",
        "analysis_scope": "shared_backbone",
        "source_state": "IPSC",
        "destination_state": "MES",
        "biological_label": "mesoderm_induction",
        "primary_external_transition": False,
    },
    {
        "transition_id": "T2",
        "analysis_scope": "shared_backbone",
        "source_state": "MES",
        "destination_state": "CMES",
        "biological_label": "cardiac_mesoderm_specification",
        "primary_external_transition": True,
    },
    {
        "transition_id": "T3_CM",
        "analysis_scope": "cm_extension",
        "source_state": "PROG",
        "destination_state": "CM",
        "biological_label": "cardiomyocyte_terminal_differentiation",
        "primary_external_transition": False,
    },
    {
        "transition_id": "T3_CF",
        "analysis_scope": "cf_extension",
        "source_state": "PROG",
        "destination_state": "CF",
        "biological_label": "cardiac_fibroblast_terminal_differentiation",
        "primary_external_transition": False,
    },
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


def monotonic_fraction(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) < 2:
        return np.nan

    return float(
        np.mean(
            np.diff(values) >= 0
        )
    )


def median_pseudotime(
    metadata,
    individual,
    state,
):
    values = metadata.loc[
        (
            metadata[
                "individual"
            ] == individual
        )
        & (
            metadata[
                "type"
            ] == state
        ),
        "dpt_pseudotime",
    ].dropna()

    if values.empty:
        return np.nan

    return float(
        values.median()
    )


def global_state_landmark_table(
    metadata,
):
    rows = []

    for state in KNOWN_TYPES:

        values = metadata.loc[
            metadata[
                "type"
            ] == state,
            "dpt_pseudotime",
        ].dropna()

        rows.append(
            {
                "state":
                    state,
                "n_cells":
                    int(
                        len(values)
                    ),
                "pseudotime_min":
                    float(
                        values.min()
                    ),
                "pseudotime_q25":
                    float(
                        values.quantile(
                            0.25
                        )
                    ),
                "pseudotime_median":
                    float(
                        values.median()
                    ),
                "pseudotime_mean":
                    float(
                        values.mean()
                    ),
                "pseudotime_q75":
                    float(
                        values.quantile(
                            0.75
                        )
                    ),
                "pseudotime_max":
                    float(
                        values.max()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def make_eligibility_plot(
    eligibility,
    output_path,
):
    """
    Plot per-individual cell counts in the six known states.
    """
    plot_columns = (
        CORE_TYPES
        + [
            "CM",
            "CF",
        ]
    )

    matrix = eligibility[
        plot_columns
    ].to_numpy(
        dtype=float
    )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    image = ax.imshow(
        np.log10(
            matrix + 1.0
        ),
        aspect="auto",
    )

    ax.set_xticks(
        np.arange(
            len(
                plot_columns
            )
        ),
        plot_columns,
    )

    ax.set_yticks(
        np.arange(
            len(
                eligibility
            )
        ),
        eligibility[
            "individual"
        ].astype(str),
    )

    ax.set_xlabel(
        "Deposited biological state"
    )

    ax.set_ylabel(
        "Individual"
    )

    ax.set_title(
        "External-validation cell coverage by individual\n"
        "(log10 count + 1)"
    )

    fig.colorbar(
        image,
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


def make_landmark_plot(
    individual_landmarks,
    output_path,
):
    """
    Individual state-median pseudotime profiles.
    """
    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    state_order = (
        CORE_TYPES
        + [
            "CM",
            "CF",
        ]
    )

    state_to_x = {
        state: i
        for i, state
        in enumerate(
            state_order
        )
    }

    for individual, group in (
        individual_landmarks.groupby(
            "individual"
        )
    ):

        backbone = group[
            group[
                "state"
            ].isin(
                CORE_TYPES
            )
        ].copy()

        backbone[
            "x"
        ] = backbone[
            "state"
        ].map(
            state_to_x
        )

        backbone = backbone.sort_values(
            "x"
        )

        if len(
            backbone
        ) == len(
            CORE_TYPES
        ):
            ax.plot(
                backbone[
                    "x"
                ],
                backbone[
                    "state_median_pseudotime"
                ],
                linewidth=0.8,
                alpha=0.35,
            )

        for terminal in [
            "CM",
            "CF",
        ]:

            terminal_row = group[
                group[
                    "state"
                ] == terminal
            ]

            if terminal_row.empty:
                continue

            prog_row = group[
                group[
                    "state"
                ] == "PROG"
            ]

            if prog_row.empty:
                continue

            x = [
                state_to_x[
                    "PROG"
                ],
                state_to_x[
                    terminal
                ],
            ]

            y = [
                float(
                    prog_row.iloc[
                        0
                    ][
                        "state_median_pseudotime"
                    ]
                ),
                float(
                    terminal_row.iloc[
                        0
                    ][
                        "state_median_pseudotime"
                    ]
                ),
            ]

            ax.plot(
                x,
                y,
                linewidth=0.7,
                alpha=0.25,
            )

    ax.set_xticks(
        np.arange(
            len(
                state_order
            )
        ),
        state_order,
    )

    ax.set_xlabel(
        "Deposited state"
    )

    ax.set_ylabel(
        "Median deposited diffusion pseudotime"
    )

    ax.set_title(
        "Frozen external-validation biological landmarks"
    )

    ax.grid(
        alpha=0.2
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

    required = [
        "cell",
        "individual",
        "type",
        "diffday",
        "dpt_pseudotime",
    ]

    missing = [
        column
        for column in required
        if column not in metadata.columns
    ]

    if missing:
        raise ValueError(
            "Missing required metadata columns: "
            + ", ".join(
                missing
            )
        )

    metadata[
        "cell"
    ] = metadata[
        "cell"
    ].astype(str)

    metadata[
        "individual"
    ] = metadata[
        "individual"
    ].astype(str)

    metadata[
        "type"
    ] = metadata[
        "type"
    ].astype(str)

    metadata[
        "dpt_pseudotime"
    ] = pd.to_numeric(
        metadata[
            "dpt_pseudotime"
        ],
        errors="coerce",
    )

    # Known annotated cells with deposited pseudotime.
    known = metadata[
        metadata[
            "type"
        ].isin(
            KNOWN_TYPES
        )
        & metadata[
            "dpt_pseudotime"
        ].notna()
    ].copy()

    # -----------------------------------------------------------------
    # Per-individual state counts and eligibility
    # -----------------------------------------------------------------

    counts = (
        known.groupby(
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
            columns=KNOWN_TYPES,
            fill_value=0,
        )
        .reset_index()
    )

    for state in CORE_TYPES:

        counts[
            f"core_{state}_pass"
        ] = (
            counts[
                state
            ]
            >= MIN_CORE_STATE_CELLS
        )

    counts[
        "shared_backbone_eligible"
    ] = counts[
        [
            f"core_{state}_pass"
            for state
            in CORE_TYPES
        ]
    ].all(
        axis=1
    )

    counts[
        "cm_terminal_pass"
    ] = (
        counts[
            "CM"
        ]
        >= MIN_TERMINAL_STATE_CELLS
    )

    counts[
        "cf_terminal_pass"
    ] = (
        counts[
            "CF"
        ]
        >= MIN_TERMINAL_STATE_CELLS
    )

    counts[
        "cm_extension_eligible"
    ] = (
        counts[
            "shared_backbone_eligible"
        ]
        & counts[
            "cm_terminal_pass"
        ]
    )

    counts[
        "cf_extension_eligible"
    ] = (
        counts[
            "shared_backbone_eligible"
        ]
        & counts[
            "cf_terminal_pass"
        ]
    )

    counts[
        "dual_terminal_extension_eligible"
    ] = (
        counts[
            "cm_extension_eligible"
        ]
        & counts[
            "cf_extension_eligible"
        ]
    )

    eligibility = counts.copy()

    # -----------------------------------------------------------------
    # Cell inclusion flags
    # -----------------------------------------------------------------

    eligibility_index = (
        eligibility.set_index(
            "individual"
        )
    )

    known[
        "shared_backbone_cell"
    ] = known[
        "type"
    ].isin(
        CORE_TYPES
    )

    known[
        "shared_backbone_individual_eligible"
    ] = known[
        "individual"
    ].map(
        eligibility_index[
            "shared_backbone_eligible"
        ]
    ).fillna(
        False
    ).astype(
        bool
    )

    known[
        "cm_extension_individual_eligible"
    ] = known[
        "individual"
    ].map(
        eligibility_index[
            "cm_extension_eligible"
        ]
    ).fillna(
        False
    ).astype(
        bool
    )

    known[
        "cf_extension_individual_eligible"
    ] = known[
        "individual"
    ].map(
        eligibility_index[
            "cf_extension_eligible"
        ]
    ).fillna(
        False
    ).astype(
        bool
    )

    known[
        "include_shared_backbone_primary"
    ] = (
        known[
            "shared_backbone_individual_eligible"
        ]
        & known[
            "type"
        ].isin(
            CORE_TYPES
        )
    )

    known[
        "include_cm_extension_secondary"
    ] = (
        known[
            "cm_extension_individual_eligible"
        ]
        & known[
            "type"
        ].isin(
            CM_TYPES
        )
    )

    known[
        "include_cf_extension_secondary"
    ] = (
        known[
            "cf_extension_individual_eligible"
        ]
        & known[
            "type"
        ].isin(
            CF_TYPES
        )
    )

    # -----------------------------------------------------------------
    # Biological state landmarks
    # -----------------------------------------------------------------

    landmark_rows = []

    for (
        individual,
        state,
    ), group in known.groupby(
        [
            "individual",
            "type",
        ],
        observed=True,
    ):

        values = group[
            "dpt_pseudotime"
        ].dropna()

        landmark_rows.append(
            {
                "individual":
                    individual,
                "state":
                    state,
                "n_cells":
                    int(
                        len(values)
                    ),
                "state_pseudotime_min":
                    float(
                        values.min()
                    ),
                "state_pseudotime_q25":
                    float(
                        values.quantile(
                            0.25
                        )
                    ),
                "state_median_pseudotime":
                    float(
                        values.median()
                    ),
                "state_pseudotime_q75":
                    float(
                        values.quantile(
                            0.75
                        )
                    ),
                "state_pseudotime_max":
                    float(
                        values.max()
                    ),
            }
        )

    individual_landmarks = pd.DataFrame(
        landmark_rows
    )

    global_landmarks = (
        global_state_landmark_table(
            known
        )
    )

    global_landmark_index = (
        global_landmarks.set_index(
            "state"
        )
    )

    # -----------------------------------------------------------------
    # Frozen transition table
    # -----------------------------------------------------------------

    transition_rows = []

    for transition in TRANSITIONS:

        source = transition[
            "source_state"
        ]

        destination = transition[
            "destination_state"
        ]

        source_median = float(
            global_landmark_index.loc[
                source,
                "pseudotime_median",
            ]
        )

        destination_median = float(
            global_landmark_index.loc[
                destination,
                "pseudotime_median",
            ]
        )

        transition_rows.append(
            {
                **transition,
                "source_global_median_pseudotime":
                    source_median,
                "destination_global_median_pseudotime":
                    destination_median,
                "destination_minus_source_median":
                    (
                        destination_median
                        - source_median
                    ),
                "reference_landmark_definition":
                    (
                        "destination-state median deposited "
                        "diffusion pseudotime"
                    ),
                "true_onset_claimed":
                    False,
            }
        )

    transition_table = pd.DataFrame(
        transition_rows
    )

    # -----------------------------------------------------------------
    # Individual monotonicity under frozen design
    # -----------------------------------------------------------------

    individual_order_rows = []

    for individual in sorted(
        known[
            "individual"
        ].unique()
    ):

        state_map = (
            individual_landmarks[
                individual_landmarks[
                    "individual"
                ] == individual
            ]
            .set_index(
                "state"
            )[
                "state_median_pseudotime"
            ]
        )

        core_values = np.asarray(
            [
                state_map.get(
                    state,
                    np.nan,
                )
                for state
                in CORE_TYPES
            ],
            dtype=float,
        )

        cm_values = np.asarray(
            [
                state_map.get(
                    state,
                    np.nan,
                )
                for state
                in CM_TYPES
            ],
            dtype=float,
        )

        cf_values = np.asarray(
            [
                state_map.get(
                    state,
                    np.nan,
                )
                for state
                in CF_TYPES
            ],
            dtype=float,
        )

        individual_order_rows.append(
            {
                "individual":
                    individual,
                "shared_backbone_median_monotonicity":
                    monotonic_fraction(
                        core_values
                    ),
                "cm_extension_median_monotonicity":
                    monotonic_fraction(
                        cm_values
                    ),
                "cf_extension_median_monotonicity":
                    monotonic_fraction(
                        cf_values
                    ),
            }
        )

    individual_order = pd.DataFrame(
        individual_order_rows
    )

    # -----------------------------------------------------------------
    # Save tables
    # -----------------------------------------------------------------

    save_table(
        eligibility,
        "01_frozen_individual_eligibility.csv",
        index=False,
    )

    cell_inclusion_path = (
        TABLE_DIR
        / "02_frozen_cell_inclusion_flags.csv.gz"
    )

    known[
        [
            "cell",
            "individual",
            "type",
            "diffday",
            "dpt_pseudotime",
            "shared_backbone_cell",
            "shared_backbone_individual_eligible",
            "cm_extension_individual_eligible",
            "cf_extension_individual_eligible",
            "include_shared_backbone_primary",
            "include_cm_extension_secondary",
            "include_cf_extension_secondary",
        ]
    ].to_csv(
        cell_inclusion_path,
        index=False,
        compression="gzip",
    )

    save_table(
        individual_landmarks,
        "03_frozen_state_landmarks_by_individual.csv",
        index=False,
    )

    save_table(
        global_landmarks,
        "04_frozen_global_state_landmarks.csv",
        index=False,
    )

    save_table(
        transition_table,
        "05_prespecified_biological_transitions.csv",
        index=False,
    )

    save_table(
        individual_order,
        "06_individual_frozen_sequence_monotonicity.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------

    make_eligibility_plot(
        eligibility,
        FIGURE_DIR
        / "01_individual_state_coverage.png",
    )

    make_landmark_plot(
        individual_landmarks,
        FIGURE_DIR
        / "02_frozen_state_landmarks_by_individual.png",
    )

    # -----------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------

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
            "GDIS-Bio External Validation Design Freeze",
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
            "\nSCIENTIFIC DESIGN:",
        )

        write_both(
            report,
            "Primary external-validation trajectory = "
            "shared developmental backbone:",
        )

        write_both(
            report,
            "  IPSC -> MES -> CMES -> PROG",
        )

        write_both(
            report,
            "\nSecondary terminal extensions:",
        )

        write_both(
            report,
            "  shared backbone -> CM",
        )

        write_both(
            report,
            "  shared backbone -> CF",
        )

        write_both(
            report,
            "\nCommon early cells are NOT assigned an eventual "
            "CM or CF fate.",
        )

        write_both(
            report,
            "\nCoverage thresholds frozen before expression "
            "preprocessing/GDIS:",
        )

        write_both(
            report,
            f"  >= {MIN_CORE_STATE_CELLS} cells in EACH "
            f"shared-backbone state",
        )

        write_both(
            report,
            f"  >= {MIN_TERMINAL_STATE_CELLS} terminal-state cells "
            f"for CM/CF extension analyses",
        )

        # ---------------------------------------------------------
        # Cohort counts
        # ---------------------------------------------------------

        write_both(
            report,
            "\n1. FROZEN INDIVIDUAL COHORTS",
        )

        write_both(
            report,
            "-" * 96,
        )

        n_total = int(
            eligibility[
                "individual"
            ].nunique()
        )

        n_core = int(
            eligibility[
                "shared_backbone_eligible"
            ].sum()
        )

        n_cm = int(
            eligibility[
                "cm_extension_eligible"
            ].sum()
        )

        n_cf = int(
            eligibility[
                "cf_extension_eligible"
            ].sum()
        )

        n_dual = int(
            eligibility[
                "dual_terminal_extension_eligible"
            ].sum()
        )

        write_both(
            report,
            f"Individuals in dataset: {n_total}",
        )

        write_both(
            report,
            f"Shared-backbone eligible: {n_core}",
        )

        write_both(
            report,
            f"CM-extension eligible: {n_cm}",
        )

        write_both(
            report,
            f"CF-extension eligible: {n_cf}",
        )

        write_both(
            report,
            f"Eligible for both terminal extensions: {n_dual}",
        )

        display_columns = (
            [
                "individual",
            ]
            + KNOWN_TYPES
            + [
                "shared_backbone_eligible",
                "cm_extension_eligible",
                "cf_extension_eligible",
                "dual_terminal_extension_eligible",
            ]
        )

        write_both(
            report,
            "\n"
            + eligibility[
                display_columns
            ].to_string(
                index=False
            ),
        )

        # ---------------------------------------------------------
        # Included cell counts
        # ---------------------------------------------------------

        write_both(
            report,
            "\n2. FROZEN CELL COUNTS",
        )

        write_both(
            report,
            "-" * 96,
        )

        n_known = int(
            len(
                known
            )
        )

        n_core_cells = int(
            known[
                "include_shared_backbone_primary"
            ].sum()
        )

        n_cm_cells = int(
            known[
                "include_cm_extension_secondary"
            ].sum()
        )

        n_cf_cells = int(
            known[
                "include_cf_extension_secondary"
            ].sum()
        )

        write_both(
            report,
            f"Known annotated cells with pseudotime: "
            f"{n_known:,}",
        )

        write_both(
            report,
            f"Primary shared-backbone cells: "
            f"{n_core_cells:,}",
        )

        write_both(
            report,
            f"Secondary CM-extension cells: "
            f"{n_cm_cells:,}",
        )

        write_both(
            report,
            f"Secondary CF-extension cells: "
            f"{n_cf_cells:,}",
        )

        # ---------------------------------------------------------
        # Biological transitions
        # ---------------------------------------------------------

        write_both(
            report,
            "\n3. PRE-SPECIFIED BIOLOGICAL TRANSITIONS",
        )

        write_both(
            report,
            "-" * 96,
        )

        write_both(
            report,
            transition_table.round(
                6
            ).to_string(
                index=False
            ),
        )

        write_both(
            report,
            "\nPRIMARY EXTERNAL TRANSITION: "
            "T2 MES -> CMES "
            "(cardiac-mesoderm specification)",
        )

        write_both(
            report,
            "Destination-state median pseudotime is a reproducible "
            "state landmark, not a true-onset estimate.",
        )

        # ---------------------------------------------------------
        # Global landmarks
        # ---------------------------------------------------------

        write_both(
            report,
            "\n4. FROZEN GLOBAL STATE LANDMARKS",
        )

        write_both(
            report,
            "-" * 96,
        )

        write_both(
            report,
            global_landmarks.round(
                6
            ).to_string(
                index=False
            ),
        )

        # ---------------------------------------------------------
        # Individual sequence ordering
        # ---------------------------------------------------------

        write_both(
            report,
            "\n5. INDIVIDUAL FROZEN-SEQUENCE ORDERING",
        )

        write_both(
            report,
            "-" * 96,
        )

        write_both(
            report,
            individual_order.round(
                6
            ).to_string(
                index=False
            ),
        )

        # ---------------------------------------------------------
        # Qualification
        # ---------------------------------------------------------

        write_both(
            report,
            "\n6. DESIGN-FREEZE QUALIFICATION",
        )

        write_both(
            report,
            "-" * 96,
        )

        destination_after_source = bool(
            (
                transition_table[
                    "destination_minus_source_median"
                ]
                > 0
            ).all()
        )

        core_all_19 = (
            n_core == n_total
        )

        enough_cm = (
            n_cm >= 10
        )

        enough_cf = (
            n_cf >= 10
        )

        enough_dual = (
            n_dual >= 8
        )

        checks = [
            (
                "All individuals satisfy shared-backbone "
                "coverage requirements",
                core_all_19,
            ),
            (
                "At least 10 independent individuals are eligible "
                "for CM-extension validation",
                enough_cm,
            ),
            (
                "At least 10 independent individuals are eligible "
                "for CF-extension validation",
                enough_cf,
            ),
            (
                "At least 8 individuals support both terminal "
                "extensions",
                enough_dual,
            ),
            (
                "Every frozen transition destination has a larger "
                "global median pseudotime than its source",
                destination_after_source,
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
            f"\nDesign checks passed: "
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
                "The GSE175634 external-validation cohort, "
                "trajectory scopes, replicate units, coverage rules, "
                "and biological transition landmarks are now frozen.",
            )

            write_both(
                report,
                "The next stage may preprocess expression counts "
                "without changing these definitions based on GDIS.",
            )

        else:

            write_both(
                report,
                "\nFINAL STATUS: REVIEW REQUIRED",
            )

            write_both(
                report,
                "Review the flagged coverage/design condition before "
                "expression preprocessing.",
            )

        write_both(
            report,
            "\nIMPORTANT:",
        )

        write_both(
            report,
            "Experimental day remains an independent biological "
            "covariate and is NOT used as the primary ordering "
            "coordinate.",
        )

        write_both(
            report,
            "Deposited diffusion pseudotime remains the primary "
            "ordering coordinate.",
        )

        write_both(
            report,
            "No common-backbone cell has been assigned an eventual "
            "CM or CF fate.",
        )

        write_both(
            report,
            "No GDIS result was examined or calculated in this stage.",
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

    print("\n" + "=" * 96)
    print("p14_GSE175634_external_validation_design_freeze.py completed.")
    print("=" * 96)
    print(f"Report : {REPORT_FILE}")
    print(f"Tables : {TABLE_DIR}")
    print(f"Figures: {FIGURE_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

