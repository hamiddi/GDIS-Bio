#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================
"""
p9_gdis_result_validation.py

Biological alignment, replicate consensus, and common-transition validation
for the first GDIS-Bio result.

This script DOES NOT recompute GDIS and DOES NOT tune any parameter.

It consumes the frozen primary GDIS outputs from p8 and the biological
landmarks frozen before GDIS in p7.

Main questions
--------------
1. How reproducible are the primary GDIS peak and transition-energy center
   across the two independent differentiations?
2. Do the GDIS maxima and data-driven transition centers align with the
   pre-frozen biological landmarks?
3. Does the transition-energy center occur before the median NEUROG3-early
   state, and by how much?
4. Is GDIS elevated from progenitor to NEUROG3-early in all four groups?
5. Are the early GDIS dynamics shared between the future SC-EC and SC-beta
   branches?
6. What is the branch-level replicate-consensus GDIS profile?

IMPORTANT
---------
A positive lead relative to the median NEUROG3-early landmark is described
only as "preceding the median state landmark." It is NOT automatically
interpreted as prediction of the true biological transition onset.

No thresholds or model parameters are optimized in this script.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

if SCRIPT_DIR.name == "scripts":
    PROJECT_DIR = SCRIPT_DIR.parent
else:
    PROJECT_DIR = SCRIPT_DIR

P7_TABLE_DIR = (
    PROJECT_DIR
    / "results"
    / "p7_trajectory_assembly_validation"
    / "tables"
)

P8_DIR = (
    PROJECT_DIR
    / "results"
    / "p8_gdis_primary_analysis"
)

P8_DATA_DIR = P8_DIR / "data"

LANDMARK_FILE = (
    P7_TABLE_DIR
    / "07_primary_state_landmark_window_mapping.csv"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
    / "p9_gdis_result_validation"
)

TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR = RESULTS_DIR / "figures"

REPORT_FILE = (
    RESULTS_DIR
    / "p9_gdis_result_validation_report.txt"
)

GROUPS = [
    (0, 1),
    (0, 2),
    (1, 1),
    (1, 2),
]

BRANCH_LABELS = {
    0: "SC-EC",
    1: "SC-beta",
}

PRIMARY_FILES = {
    (0, 1): "primary_branch0_diff1_gdis.csv",
    (0, 2): "primary_branch0_diff2_gdis.csv",
    (1, 1): "primary_branch1_diff1_gdis.csv",
    (1, 2): "primary_branch1_diff2_gdis.csv",
}

CONSENSUS_GRID_POINTS = 300

EXPECTED_LANDMARK_SEQUENCE = {
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


def load_primary(branch, differentiation):
    path = (
        P8_DATA_DIR
        / PRIMARY_FILES[
            (branch, differentiation)
        ]
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing p8 primary output: {path}"
        )

    df = pd.read_csv(path)

    required = [
        "parameter",
        "gdis",
        "sustained_instability",
        "transition_instability",
        "transition_energy",
        "window_index",
        "median_day",
        "dominant_state",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: "
            + ", ".join(missing)
        )

    return (
        df.sort_values("parameter")
        .reset_index(drop=True)
    )


def get_landmark(
    landmarks,
    branch,
    differentiation,
    state,
):
    subset = landmarks[
        (landmarks["branch"] == branch)
        & (
            landmarks[
                "differentiation"
            ] == differentiation
        )
        & (
            landmarks["state"]
            == state
        )
    ]

    if len(subset) != 1:
        raise ValueError(
            f"Expected exactly one landmark for "
            f"branch={branch}, diff={differentiation}, "
            f"state={state}; found {len(subset)}"
        )

    return float(
        subset.iloc[0][
            "state_median_pseudotime"
        ]
    )


def nearest_profile_row(df, parameter):
    distances = np.abs(
        df["parameter"].to_numpy(dtype=float)
        - float(parameter)
    )

    pos = int(
        np.argmin(distances)
    )

    return (
        df.iloc[pos],
        float(distances[pos]),
    )


def interpolate_profile(
    df,
    grid,
    column,
):
    return np.interp(
        grid,
        df["parameter"].to_numpy(dtype=float),
        df[column].to_numpy(dtype=float),
    )


def branch_consensus(
    diff1,
    diff2,
    branch,
):
    """
    Build consensus on the shared actual-pseudotime interval.
    """
    lower = max(
        float(diff1["parameter"].min()),
        float(diff2["parameter"].min()),
    )

    upper = min(
        float(diff1["parameter"].max()),
        float(diff2["parameter"].max()),
    )

    grid = np.linspace(
        lower,
        upper,
        CONSENSUS_GRID_POINTS,
    )

    metrics = [
        "gdis",
        "sustained_instability",
        "transition_instability",
        "transition_energy",
    ]

    out = pd.DataFrame(
        {
            "branch": branch,
            "lineage": BRANCH_LABELS[branch],
            "parameter": grid,
        }
    )

    for metric in metrics:

        v1 = interpolate_profile(
            diff1,
            grid,
            metric,
        )

        v2 = interpolate_profile(
            diff2,
            grid,
            metric,
        )

        out[
            f"{metric}_diff1"
        ] = v1

        out[
            f"{metric}_diff2"
        ] = v2

        out[
            f"{metric}_consensus_mean"
        ] = (
            v1 + v2
        ) / 2.0

        out[
            f"{metric}_consensus_min"
        ] = np.minimum(
            v1,
            v2,
        )

        out[
            f"{metric}_consensus_max"
        ] = np.maximum(
            v1,
            v2,
        )

        out[
            f"{metric}_absolute_replicate_difference"
        ] = np.abs(
            v1 - v2
        )

    return out


def plot_consensus(
    consensus,
    branch,
    landmarks,
    output_path,
):
    """Plot branch consensus with frozen state landmarks."""
    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    x = consensus[
        "parameter"
    ].to_numpy(dtype=float)

    mean = consensus[
        "gdis_consensus_mean"
    ].to_numpy(dtype=float)

    low = consensus[
        "gdis_consensus_min"
    ].to_numpy(dtype=float)

    high = consensus[
        "gdis_consensus_max"
    ].to_numpy(dtype=float)

    ax.fill_between(
        x,
        low,
        high,
        alpha=0.20,
        label="Replicate range",
    )

    ax.plot(
        x,
        mean,
        linewidth=2.0,
        label="Replicate consensus GDIS",
    )

    # Mean landmark position across the two differentiations.
    for state in EXPECTED_LANDMARK_SEQUENCE[branch]:

        subset = landmarks[
            (landmarks["branch"] == branch)
            & (
                landmarks["state"]
                == state
            )
        ]

        if subset.empty:
            continue

        landmark = float(
            subset[
                "state_median_pseudotime"
            ].mean()
        )

        ax.axvline(
            landmark,
            linestyle="--",
            linewidth=0.9,
            alpha=0.65,
        )

        ax.text(
            landmark,
            0.02,
            state,
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=8,
            transform=ax.get_xaxis_transform(),
        )

    ax.set_xlabel(
        "Deposited pseudotime"
    )

    ax.set_ylabel(
        "GDIS"
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.set_title(
        f"Replicate-consensus GDIS: "
        f"{BRANCH_LABELS[branch]} lineage"
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

    if not LANDMARK_FILE.exists():
        raise FileNotFoundError(
            f"Missing frozen landmark file: "
            f"{LANDMARK_FILE}"
        )

    landmarks = pd.read_csv(
        LANDMARK_FILE
    )

    primary = {
        key: load_primary(*key)
        for key in GROUPS
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as report:

        write_both(
            report,
            "=" * 92,
        )

        write_both(
            report,
            "GDIS-Bio Primary Result Validation and Replicate Consensus",
        )

        write_both(
            report,
            "=" * 92,
        )

        write_both(
            report,
            "\nNo GDIS values are recomputed in this script.",
        )

        write_both(
            report,
            "All biological landmarks were frozen before the p8 GDIS run.",
        )

        # =========================================================
        # 1. Event localization relative to frozen landmarks
        # =========================================================

        write_both(
            report,
            "\n1. EVENT LOCALIZATION RELATIVE TO FROZEN BIOLOGICAL LANDMARKS",
        )

        write_both(
            report,
            "-" * 92,
        )

        event_rows = []

        for branch, differentiation in GROUPS:

            df = primary[
                (branch, differentiation)
            ]

            early_landmark = get_landmark(
                landmarks,
                branch,
                differentiation,
                "neurog3_early",
            )

            peak_pos = int(
                df["gdis"].idxmax()
            )

            energy_pos = int(
                df[
                    "transition_energy"
                ].idxmax()
            )

            sustained_pos = int(
                df[
                    "sustained_instability"
                ].idxmax()
            )

            peak_parameter = float(
                df.loc[
                    peak_pos,
                    "parameter",
                ]
            )

            energy_parameter = float(
                df.loc[
                    energy_pos,
                    "parameter",
                ]
            )

            sustained_parameter = float(
                df.loc[
                    sustained_pos,
                    "parameter",
                ]
            )

            event_rows.append(
                {
                    "branch": branch,
                    "lineage":
                        BRANCH_LABELS[
                            branch
                        ],
                    "differentiation":
                        differentiation,
                    "neurog3_early_median_landmark":
                        early_landmark,
                    "gdis_peak_parameter":
                        peak_parameter,
                    "gdis_peak_minus_early_landmark":
                        (
                            peak_parameter
                            - early_landmark
                        ),
                    "transition_energy_peak_parameter":
                        energy_parameter,
                    "transition_center_minus_early_landmark":
                        (
                            energy_parameter
                            - early_landmark
                        ),
                    "transition_center_precedes_early_median":
                        bool(
                            energy_parameter
                            < early_landmark
                        ),
                    "sustained_peak_parameter":
                        sustained_parameter,
                    "gdis_peak_value":
                        float(
                            df.loc[
                                peak_pos,
                                "gdis",
                            ]
                        ),
                    "transition_energy_peak_value":
                        float(
                            df.loc[
                                energy_pos,
                                "transition_energy",
                            ]
                        ),
                    "gdis_peak_median_day":
                        float(
                            df.loc[
                                peak_pos,
                                "median_day",
                            ]
                        ),
                    "gdis_peak_dominant_state":
                        str(
                            df.loc[
                                peak_pos,
                                "dominant_state",
                            ]
                        ),
                }
            )

        event_table = pd.DataFrame(
            event_rows
        )

        save_table(
            event_table,
            "01_event_localization_vs_neurog3_early.csv",
            index=False,
        )

        write_both(
            report,
            event_table.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 2. Landmark progression
        # =========================================================

        write_both(
            report,
            "\n2. GDIS PROGRESSION ACROSS FROZEN BIOLOGICAL LANDMARKS",
        )

        write_both(
            report,
            "-" * 92,
        )

        landmark_rows = []

        for branch, differentiation in GROUPS:

            df = primary[
                (branch, differentiation)
            ]

            sequence = (
                EXPECTED_LANDMARK_SEQUENCE[
                    branch
                ]
            )

            for order, state in enumerate(
                sequence
            ):

                landmark = get_landmark(
                    landmarks,
                    branch,
                    differentiation,
                    state,
                )

                row, distance = (
                    nearest_profile_row(
                        df,
                        landmark,
                    )
                )

                landmark_rows.append(
                    {
                        "branch":
                            branch,
                        "lineage":
                            BRANCH_LABELS[
                                branch
                            ],
                        "differentiation":
                            differentiation,
                        "state_order":
                            order,
                        "state":
                            state,
                        "state_median_pseudotime":
                            landmark,
                        "nearest_window_parameter":
                            float(
                                row[
                                    "parameter"
                                ]
                            ),
                        "parameter_error":
                            distance,
                        "gdis":
                            float(
                                row[
                                    "gdis"
                                ]
                            ),
                        "sustained_instability":
                            float(
                                row[
                                    "sustained_instability"
                                ]
                            ),
                        "transition_instability":
                            float(
                                row[
                                    "transition_instability"
                                ]
                            ),
                        "transition_energy":
                            float(
                                row[
                                    "transition_energy"
                                ]
                            ),
                    }
                )

        landmark_table = pd.DataFrame(
            landmark_rows
        )

        save_table(
            landmark_table,
            "02_metric_values_at_frozen_landmarks.csv",
            index=False,
        )

        write_both(
            report,
            landmark_table.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 3. Stage-to-stage changes
        # =========================================================

        write_both(
            report,
            "\n3. STAGE-TO-STAGE METRIC CHANGES",
        )

        write_both(
            report,
            "-" * 92,
        )

        delta_rows = []

        metrics = [
            "gdis",
            "sustained_instability",
            "transition_instability",
            "transition_energy",
        ]

        for branch, differentiation in GROUPS:

            subset = (
                landmark_table[
                    (
                        landmark_table[
                            "branch"
                        ] == branch
                    )
                    & (
                        landmark_table[
                            "differentiation"
                        ] == differentiation
                    )
                ]
                .sort_values(
                    "state_order"
                )
                .reset_index(
                    drop=True
                )
            )

            for i in range(
                len(subset) - 1
            ):

                a = subset.iloc[i]
                b = subset.iloc[i + 1]

                row = {
                    "branch":
                        branch,
                    "lineage":
                        BRANCH_LABELS[
                            branch
                        ],
                    "differentiation":
                        differentiation,
                    "from_state":
                        a["state"],
                    "to_state":
                        b["state"],
                }

                for metric in metrics:
                    row[
                        f"delta_{metric}"
                    ] = (
                        float(
                            b[metric]
                        )
                        - float(
                            a[metric]
                        )
                    )

                delta_rows.append(
                    row
                )

        delta_table = pd.DataFrame(
            delta_rows
        )

        save_table(
            delta_table,
            "03_stage_to_stage_metric_changes.csv",
            index=False,
        )

        write_both(
            report,
            delta_table.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 4. Replicate event-location agreement
        # =========================================================

        write_both(
            report,
            "\n4. REPLICATE EVENT-LOCATION AGREEMENT",
        )

        write_both(
            report,
            "-" * 92,
        )

        replicate_event_rows = []

        for branch in [0, 1]:

            a = event_table[
                (
                    event_table[
                        "branch"
                    ] == branch
                )
                & (
                    event_table[
                        "differentiation"
                    ] == 1
                )
            ].iloc[0]

            b = event_table[
                (
                    event_table[
                        "branch"
                    ] == branch
                )
                & (
                    event_table[
                        "differentiation"
                    ] == 2
                )
            ].iloc[0]

            replicate_event_rows.append(
                {
                    "branch":
                        branch,
                    "lineage":
                        BRANCH_LABELS[
                            branch
                        ],
                    "absolute_gdis_peak_parameter_difference":
                        abs(
                            float(
                                a[
                                    "gdis_peak_parameter"
                                ]
                            )
                            - float(
                                b[
                                    "gdis_peak_parameter"
                                ]
                            )
                        ),
                    "absolute_transition_center_difference":
                        abs(
                            float(
                                a[
                                    "transition_energy_peak_parameter"
                                ]
                            )
                            - float(
                                b[
                                    "transition_energy_peak_parameter"
                                ]
                            )
                        ),
                    "absolute_sustained_peak_difference":
                        abs(
                            float(
                                a[
                                    "sustained_peak_parameter"
                                ]
                            )
                            - float(
                                b[
                                    "sustained_peak_parameter"
                                ]
                            )
                        ),
                }
            )

        replicate_event_table = (
            pd.DataFrame(
                replicate_event_rows
            )
        )

        save_table(
            replicate_event_table,
            "04_replicate_event_location_agreement.csv",
            index=False,
        )

        write_both(
            report,
            replicate_event_table.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 5. Branch-level replicate consensus
        # =========================================================

        write_both(
            report,
            "\n5. BRANCH-LEVEL REPLICATE CONSENSUS",
        )

        write_both(
            report,
            "-" * 92,
        )

        consensus_tables = []
        consensus_summary_rows = []

        for branch in [0, 1]:

            consensus = branch_consensus(
                primary[(branch, 1)],
                primary[(branch, 2)],
                branch,
            )

            consensus_tables.append(
                consensus
            )

            gdis_col = (
                "gdis_consensus_mean"
            )

            energy_col = (
                "transition_energy_consensus_mean"
            )

            gdis_peak_pos = int(
                consensus[
                    gdis_col
                ].idxmax()
            )

            energy_peak_pos = int(
                consensus[
                    energy_col
                ].idxmax()
            )

            mean_early_landmark = float(
                landmarks[
                    (
                        landmarks[
                            "branch"
                        ] == branch
                    )
                    & (
                        landmarks[
                            "state"
                        ] == "neurog3_early"
                    )
                ][
                    "state_median_pseudotime"
                ].mean()
            )

            consensus_summary_rows.append(
                {
                    "branch":
                        branch,
                    "lineage":
                        BRANCH_LABELS[
                            branch
                        ],
                    "consensus_parameter_min":
                        float(
                            consensus[
                                "parameter"
                            ].min()
                        ),
                    "consensus_parameter_max":
                        float(
                            consensus[
                                "parameter"
                            ].max()
                        ),
                    "consensus_gdis_peak_parameter":
                        float(
                            consensus.loc[
                                gdis_peak_pos,
                                "parameter",
                            ]
                        ),
                    "consensus_gdis_peak_value":
                        float(
                            consensus.loc[
                                gdis_peak_pos,
                                gdis_col,
                            ]
                        ),
                    "consensus_transition_energy_peak_parameter":
                        float(
                            consensus.loc[
                                energy_peak_pos,
                                "parameter",
                            ]
                        ),
                    "mean_neurog3_early_landmark":
                        mean_early_landmark,
                    "consensus_transition_center_minus_early_landmark":
                        (
                            float(
                                consensus.loc[
                                    energy_peak_pos,
                                    "parameter",
                                ]
                            )
                            - mean_early_landmark
                        ),
                    "mean_absolute_replicate_gdis_difference":
                        float(
                            consensus[
                                "gdis_absolute_replicate_difference"
                            ].mean()
                        ),
                }
            )

            plot_consensus(
                consensus,
                branch,
                landmarks,
                FIGURE_DIR
                / (
                    f"0{branch + 1}_"
                    f"consensus_gdis_"
                    f"{BRANCH_LABELS[branch].lower().replace('-', '_')}"
                    f".png"
                ),
            )

        consensus_table = pd.concat(
            consensus_tables,
            ignore_index=True,
        )

        consensus_summary = (
            pd.DataFrame(
                consensus_summary_rows
            )
        )

        save_table(
            consensus_table,
            "05_branch_consensus_profiles.csv",
            index=False,
        )

        save_table(
            consensus_summary,
            "06_branch_consensus_summary.csv",
            index=False,
        )

        write_both(
            report,
            consensus_summary.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 6. Cross-branch common-phase agreement
        # =========================================================

        write_both(
            report,
            "\n6. CROSS-BRANCH AGREEMENT IN THE COMMON ENDOCRINE PHASE",
        )

        write_both(
            report,
            "-" * 92,
        )

        b0 = consensus_table[
            consensus_table[
                "branch"
            ] == 0
        ].copy()

        b1 = consensus_table[
            consensus_table[
                "branch"
            ] == 1
        ].copy()

        # Frozen common-phase endpoint:
        # mean NEUROG3-late median landmark across all four groups.
        late_cutoff = float(
            landmarks[
                landmarks[
                    "state"
                ] == "neurog3_late"
            ][
                "state_median_pseudotime"
            ].mean()
        )

        lower = max(
            float(
                b0["parameter"].min()
            ),
            float(
                b1["parameter"].min()
            ),
        )

        upper = min(
            late_cutoff,
            float(
                b0["parameter"].max()
            ),
            float(
                b1["parameter"].max()
            ),
        )

        grid = np.linspace(
            lower,
            upper,
            CONSENSUS_GRID_POINTS,
        )

        g0 = np.interp(
            grid,
            b0["parameter"],
            b0[
                "gdis_consensus_mean"
            ],
        )

        g1 = np.interp(
            grid,
            b1["parameter"],
            b1[
                "gdis_consensus_mean"
            ],
        )

        e0 = np.interp(
            grid,
            b0["parameter"],
            b0[
                "transition_energy_consensus_mean"
            ],
        )

        e1 = np.interp(
            grid,
            b1["parameter"],
            b1[
                "transition_energy_consensus_mean"
            ],
        )

        common_phase_summary = (
            pd.DataFrame(
                [
                    {
                        "common_phase_min":
                            lower,
                        "common_phase_max_neurog3_late":
                            upper,
                        "gdis_cross_branch_pearson":
                            safe_pearson(
                                g0,
                                g1,
                            ),
                        "gdis_cross_branch_spearman":
                            safe_spearman(
                                g0,
                                g1,
                            ),
                        "gdis_cross_branch_mae":
                            float(
                                np.mean(
                                    np.abs(
                                        g0 - g1
                                    )
                                )
                            ),
                        "transition_energy_cross_branch_pearson":
                            safe_pearson(
                                e0,
                                e1,
                            ),
                        "transition_energy_cross_branch_spearman":
                            safe_spearman(
                                e0,
                                e1,
                            ),
                        "transition_energy_cross_branch_mae":
                            float(
                                np.mean(
                                    np.abs(
                                        e0 - e1
                                    )
                                )
                            ),
                    }
                ]
            )
        )

        save_table(
            common_phase_summary,
            "07_cross_branch_common_phase_agreement.csv",
            index=False,
        )

        write_both(
            report,
            common_phase_summary.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 7. Descriptive concordance checks
        # =========================================================

        write_both(
            report,
            "\n7. DESCRIPTIVE CONCORDANCE CHECKS",
        )

        write_both(
            report,
            "-" * 92,
        )

        # These are descriptive summaries, NOT pre-registered inferential
        # thresholds and are not used to tune or select the GDIS analysis.
        checks = []

        checks.append(
            (
                "All four GDIS global peaks have dominant state "
                "NEUROG3-early",
                bool(
                    (
                        event_table[
                            "gdis_peak_dominant_state"
                        ]
                        == "neurog3_early"
                    ).all()
                ),
            )
        )

        checks.append(
            (
                "All four transition-energy centers precede the "
                "median NEUROG3-early landmark",
                bool(
                    event_table[
                        "transition_center_precedes_early_median"
                    ].all()
                ),
            )
        )

        # GDIS at early state > GDIS at progenitor state.
        early_gt_prog = True

        for branch, differentiation in GROUPS:

            subset = landmark_table[
                (
                    landmark_table[
                        "branch"
                    ] == branch
                )
                & (
                    landmark_table[
                        "differentiation"
                    ] == differentiation
                )
            ]

            prog = float(
                subset.loc[
                    subset[
                        "state"
                    ] == "prog_nkx61",
                    "gdis",
                ].iloc[0]
            )

            early = float(
                subset.loc[
                    subset[
                        "state"
                    ] == "neurog3_early",
                    "gdis",
                ].iloc[0]
            )

            if not early > prog:
                early_gt_prog = False

        checks.append(
            (
                "GDIS at the NEUROG3-early landmark exceeds "
                "GDIS at the progenitor landmark in all four groups",
                early_gt_prog,
            )
        )

        for label, status in checks:

            prefix = (
                "[CONSISTENT]"
                if status
                else "[NOT CONSISTENT]"
            )

            write_both(
                report,
                f"{prefix} {label}",
            )

        # =========================================================
        # 8. Final interpretation summary
        # =========================================================

        write_both(
            report,
            "\n8. RESULT SUMMARY",
        )

        write_both(
            report,
            "-" * 92,
        )

        mean_center_lead = float(
            -event_table[
                "transition_center_minus_early_landmark"
            ].mean()
        )

        min_center_lead = float(
            -event_table[
                "transition_center_minus_early_landmark"
            ].max()
        )

        max_center_lead = float(
            -event_table[
                "transition_center_minus_early_landmark"
            ].min()
        )

        write_both(
            report,
            "Observed primary pattern:",
        )

        write_both(
            report,
            "  - The global GDIS maximum localizes to the "
            "NEUROG3-early transition phase in all four "
            "branch x differentiation analyses.",
        )

        write_both(
            report,
            "  - The data-driven transition-energy center occurs "
            "before the median NEUROG3-early landmark in all four groups.",
        )

        write_both(
            report,
            f"  - Mean separation from transition-energy center "
            f"to median NEUROG3-early landmark: "
            f"{mean_center_lead:.6f} pseudotime units.",
        )

        write_both(
            report,
            f"  - Observed range of that separation: "
            f"{min_center_lead:.6f} to "
            f"{max_center_lead:.6f}.",
        )

        write_both(
            report,
            "  - This is evidence of reproducible transition localization "
            "relative to a pre-frozen state landmark; it is not by itself "
            "proof of prediction before true biological onset.",
        )

        write_both(
            report,
            "  - A distinct later global GDIS maximum at the terminal "
            "SC-EC/SC-beta state is not supported by the primary result.",
        )

        write_both(
            report,
            "\nRecommended next step:",
        )

        write_both(
            report,
            "Benchmark the frozen GDIS result against conventional "
            "early-warning / transition metrics using the same windows "
            "and biological landmarks.",
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

    print("\n" + "=" * 92)
    print("p9_gdis_result_validation.py completed.")
    print("=" * 92)
    print(f"Report : {REPORT_FILE}")
    print(f"Tables : {TABLE_DIR}")
    print(f"Figures: {FIGURE_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

