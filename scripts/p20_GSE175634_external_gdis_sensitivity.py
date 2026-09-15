#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================
"""
p20_GSE175634_external_gdis_sensitivity.py

Pre-specified sensitivity analysis for the frozen GSE175634 external-validation
GDIS results.

This script DOES NOT alter or replace the p18/p19 primary result.

Primary result remains frozen as:
    50 PCs
    400 cells/window
    100-cell step

Sensitivity analyses
--------------------
A. Window-size sensitivity at 50 PCs
       300 cells/window, 75-cell step
       500 cells/window, 125-cell step

B. PCA-dimensionality sensitivity at primary 400/100 windows
       5 PCs
       10 PCs
       20 PCs
       30 PCs

The same final p17b transition-evaluable scope x individual cohort is used
throughout:
    shared_backbone : 19
    cm_extension    : 15
    cf_extension    : 14

No GDIS parameter is changed.

No biological critical_value is supplied.

No profile is included/excluded based on sensitivity results.

Comparison endpoints
--------------------
For every sensitivity profile versus its frozen primary p18 profile:
    - interpolated Pearson correlation
    - interpolated Spearman correlation
    - mean absolute profile difference
    - GDIS peak parameter
    - absolute peak-to-landmark error
    - peak shift from primary
    - transition-energy peak shift from primary

Scope-level summaries report:
    - median profile correlations
    - median absolute profile difference
    - median peak shift
    - median absolute peak-to-landmark error
    - fraction of sensitivity profiles whose GDIS peak is within 0.05
      pseudotime units of the primary peak

IMPORTANT
---------
Sensitivity analyses are descriptive robustness analyses.
They do not overwrite the pre-specified p19 inferential result.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

try:
    from gdis import GDIS
except ImportError as exc:
    raise SystemExit(
        "ERROR: pyGDIS is required.\n"
        "Install with:\n"
        "  python -m pip install pygdis"
    ) from exc


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR

P14_TABLE_DIR = (
    PROJECT_DIR
    / "results"
    / "p14_external_GSE175634_design_freeze"
    / "tables"
)

ELIGIBILITY_FILE = (
    P14_TABLE_DIR
    / "01_frozen_individual_eligibility.csv"
)

P15_DATA_DIR = (
    PROJECT_DIR
    / "results"
    / "p15_external_GSE175634_preprocessing_state_space"
    / "data"
)

P17_TABLE_DIR = (
    PROJECT_DIR
    / "results"
    / "p17_external_GSE175634_trajectory_window_assembly"
    / "tables"
)

P17B_TABLE_DIR = (
    PROJECT_DIR
    / "results"
    / "p17b_external_GSE175634_transition_evaluability_freeze"
    / "tables"
)

FINAL_PRIMARY_MANIFEST_FILE = (
    P17B_TABLE_DIR
    / "04_final_primary_gdis_file_manifest.csv"
)

PAIRED_WINDOW_MANIFEST_FILE = (
    P17B_TABLE_DIR
    / "05_paired_window_sensitivity_manifest.csv"
)

P18_TABLE_DIR = (
    PROJECT_DIR
    / "results"
    / "p18_external_GSE175634_primary_gdis"
    / "tables"
)

PRIMARY_PROFILE_FILE = (
    P18_TABLE_DIR
    / "03_all_primary_gdis_profiles.csv.gz"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
    / "p20_external_GSE175634_gdis_sensitivity"
)

TABLE_DIR = RESULTS_DIR / "tables"
DATA_DIR = RESULTS_DIR / "data"

REPORT_FILE = (
    RESULTS_DIR
    / "p20_external_GSE175634_gdis_sensitivity_report.txt"
)


# ---------------------------------------------------------------------
# Frozen settings
# ---------------------------------------------------------------------

PRIMARY_CONFIG = "primary_w400_s100"

WINDOW_SENSITIVITY_CONFIGS = [
    "sensitivity_w300_s75",
    "sensitivity_w500_s125",
]

DIMENSIONS = [
    5,
    10,
    20,
    30,
]

PRIMARY_DIMENSION = 50

PRIMARY_WINDOW_SIZE = 400
PRIMARY_STEP_SIZE = 100

SCOPE_DEFINITIONS = {
    "shared_backbone": {
        "states": [
            "IPSC",
            "MES",
            "CMES",
            "PROG",
        ],
    },
    "cm_extension": {
        "states": [
            "IPSC",
            "MES",
            "CMES",
            "PROG",
            "CM",
        ],
    },
    "cf_extension": {
        "states": [
            "IPSC",
            "MES",
            "CMES",
            "PROG",
            "CF",
        ],
    },
}

EXPECTED_COUNTS = {
    "shared_backbone": 19,
    "cm_extension": 15,
    "cf_extension": 14,
}

PEAK_STABILITY_TOLERANCE = 0.05
INTERPOLATION_POINTS = 250


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


def run_gdis(
    trajectories,
    parameters,
):
    result = GDIS().fit_transform(
        trajectories,
        parameters,
    )

    df = result.to_dataframe()

    for component in [
        "transition_energy",
        "transition_base",
        "core",
    ]:
        if component in result.components:
            df[
                component
            ] = np.asarray(
                result.components[
                    component
                ],
                dtype=float,
            )

    if "transition_energy" not in df.columns:
        raise ValueError(
            "pyGDIS result missing transition_energy."
        )

    return df


def strict_increasing(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    return bool(
        np.all(
            np.diff(
                values
            ) > 0
        )
    )


def profile_comparison(
    primary,
    sensitivity,
    landmark,
):
    """
    Compare two profiles over their shared pseudotime interval.
    """
    primary = (
        primary.sort_values(
            "parameter"
        )
        .reset_index(
            drop=True
        )
    )

    sensitivity = (
        sensitivity.sort_values(
            "parameter"
        )
        .reset_index(
            drop=True
        )
    )

    p_x = primary[
        "parameter"
    ].to_numpy(
        dtype=float
    )

    p_y = primary[
        "gdis"
    ].to_numpy(
        dtype=float
    )

    s_x = sensitivity[
        "parameter"
    ].to_numpy(
        dtype=float
    )

    s_y = sensitivity[
        "gdis"
    ].to_numpy(
        dtype=float
    )

    common_min = max(
        float(
            p_x.min()
        ),
        float(
            s_x.min()
        ),
    )

    common_max = min(
        float(
            p_x.max()
        ),
        float(
            s_x.max()
        ),
    )

    if common_max <= common_min:
        raise ValueError(
            "Primary and sensitivity profiles have no common "
            "pseudotime interval."
        )

    grid = np.linspace(
        common_min,
        common_max,
        INTERPOLATION_POINTS,
    )

    p_interp = np.interp(
        grid,
        p_x,
        p_y,
    )

    s_interp = np.interp(
        grid,
        s_x,
        s_y,
    )

    pearson = float(
        pearsonr(
            p_interp,
            s_interp,
        ).statistic
    )

    spearman = float(
        spearmanr(
            p_interp,
            s_interp,
        ).statistic
    )

    mad = float(
        np.mean(
            np.abs(
                p_interp
                - s_interp
            )
        )
    )

    p_peak_index = int(
        np.argmax(
            p_y
        )
    )

    s_peak_index = int(
        np.argmax(
            s_y
        )
    )

    p_peak_parameter = float(
        p_x[
            p_peak_index
        ]
    )

    s_peak_parameter = float(
        s_x[
            s_peak_index
        ]
    )

    p_energy = primary[
        "transition_energy"
    ].to_numpy(
        dtype=float
    )

    s_energy = sensitivity[
        "transition_energy"
    ].to_numpy(
        dtype=float
    )

    p_energy_peak = float(
        p_x[
            int(
                np.argmax(
                    p_energy
                )
            )
        ]
    )

    s_energy_peak = float(
        s_x[
            int(
                np.argmax(
                    s_energy
                )
            )
        ]
    )

    return {
        "common_parameter_min":
            common_min,
        "common_parameter_max":
            common_max,
        "pearson_gdis_profile":
            pearson,
        "spearman_gdis_profile":
            spearman,
        "mean_abs_gdis_difference":
            mad,
        "primary_gdis_peak_parameter":
            p_peak_parameter,
        "sensitivity_gdis_peak_parameter":
            s_peak_parameter,
        "gdis_peak_shift":
            s_peak_parameter
            - p_peak_parameter,
        "absolute_gdis_peak_shift":
            abs(
                s_peak_parameter
                - p_peak_parameter
            ),
        "primary_abs_peak_landmark_error":
            abs(
                p_peak_parameter
                - landmark
            ),
        "sensitivity_abs_peak_landmark_error":
            abs(
                s_peak_parameter
                - landmark
            ),
        "primary_transition_energy_peak_parameter":
            p_energy_peak,
        "sensitivity_transition_energy_peak_parameter":
            s_energy_peak,
        "transition_energy_peak_shift":
            s_energy_peak
            - p_energy_peak,
        "absolute_transition_energy_peak_shift":
            abs(
                s_energy_peak
                - p_energy_peak
            ),
        "gdis_peak_within_0p05_of_primary":
            bool(
                abs(
                    s_peak_parameter
                    - p_peak_parameter
                )
                <= PEAK_STABILITY_TOLERANCE
            ),
    }


def assemble_primary_windows_from_state_space(
    state_space,
    scope,
    individual,
):
    """
    Recreate the frozen p17 400/100 window family at a lower PCA dimension.

    Cell ordering is identical:
        deposited pseudotime, then cell ID.
    """
    states = (
        SCOPE_DEFINITIONS[
            scope
        ][
            "states"
        ]
    )

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
                states
            )
        )
    ].copy()

    group = (
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
        group
    )

    if (
        n_cells
        < PRIMARY_WINDOW_SIZE
    ):
        raise ValueError(
            f"{scope}/{individual}: too few cells for primary windows."
        )

    starts = list(
        range(
            0,
            n_cells
            - PRIMARY_WINDOW_SIZE
            + 1,
            PRIMARY_STEP_SIZE,
        )
    )

    pc_columns = [
        column
        for column in state_space.columns
        if column.startswith(
            "PC"
        )
    ]

    pc_columns = sorted(
        pc_columns,
        key=lambda x:
            int(
                x[
                    2:
                ]
            ),
    )

    n_dim = len(
        pc_columns
    )

    trajectories = np.empty(
        (
            len(
                starts
            ),
            PRIMARY_WINDOW_SIZE,
            n_dim,
        ),
        dtype=np.float64,
    )

    parameters = np.empty(
        len(
            starts
        ),
        dtype=np.float64,
    )

    for i, start in enumerate(
        starts
    ):

        stop = (
            start
            + PRIMARY_WINDOW_SIZE
        )

        window = group.iloc[
            start:stop
        ]

        trajectories[
            i,
            :,
            :,
        ] = window[
            pc_columns
        ].to_numpy(
            dtype=np.float64,
        )

        parameters[
            i
        ] = float(
            window[
                "dpt_pseudotime"
            ].median()
        )

    if not strict_increasing(
        parameters
    ):
        raise ValueError(
            f"{scope}/{individual}: reconstructed lower-dimensional "
            "parameters are not strictly increasing."
        )

    return (
        trajectories,
        parameters,
    )


def scope_summary(
    comparisons,
):
    return (
        comparisons.groupby(
            [
                "sensitivity_type",
                "sensitivity_label",
                "scope",
            ],
            observed=True,
        )
        .agg(
            n_profiles=(
                "individual",
                "nunique",
            ),
            median_pearson=(
                "pearson_gdis_profile",
                "median",
            ),
            median_spearman=(
                "spearman_gdis_profile",
                "median",
            ),
            median_mean_abs_difference=(
                "mean_abs_gdis_difference",
                "median",
            ),
            median_abs_peak_shift=(
                "absolute_gdis_peak_shift",
                "median",
            ),
            max_abs_peak_shift=(
                "absolute_gdis_peak_shift",
                "max",
            ),
            median_primary_abs_peak_landmark_error=(
                "primary_abs_peak_landmark_error",
                "median",
            ),
            median_sensitivity_abs_peak_landmark_error=(
                "sensitivity_abs_peak_landmark_error",
                "median",
            ),
            fraction_peak_within_0p05_of_primary=(
                "gdis_peak_within_0p05_of_primary",
                "mean",
            ),
            median_abs_transition_energy_peak_shift=(
                "absolute_transition_energy_peak_shift",
                "median",
            ),
        )
        .reset_index()
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    for directory in [
        TABLE_DIR,
        DATA_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    required_files = [
        FINAL_PRIMARY_MANIFEST_FILE,
        PAIRED_WINDOW_MANIFEST_FILE,
        PRIMARY_PROFILE_FILE,
    ]

    for dimension in DIMENSIONS:
        required_files.append(
            P15_DATA_DIR
            / (
                f"external_state_space_pca_"
                f"{dimension}.csv.gz"
            )
        )

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

    final_manifest = pd.read_csv(
        FINAL_PRIMARY_MANIFEST_FILE,
        low_memory=False,
    )

    paired_window_manifest = pd.read_csv(
        PAIRED_WINDOW_MANIFEST_FILE,
        low_memory=False,
    )

    primary_profiles = pd.read_csv(
        PRIMARY_PROFILE_FILE,
        compression="gzip",
        low_memory=False,
    )

    for table in [
        final_manifest,
        paired_window_manifest,
        primary_profiles,
    ]:
        if "individual" in table.columns:
            table[
                "individual"
            ] = table[
                "individual"
            ].astype(str)

    final_manifest[
        "scope"
    ] = final_manifest[
        "scope"
    ].astype(str)

    primary_profiles[
        "scope"
    ] = primary_profiles[
        "scope"
    ].astype(str)

    # Verify frozen counts.
    for scope, expected in (
        EXPECTED_COUNTS.items()
    ):

        observed = int(
            final_manifest.loc[
                final_manifest[
                    "scope"
                ] == scope,
                "individual",
            ].nunique()
        )

        if observed != expected:
            raise ValueError(
                f"Frozen count mismatch for {scope}: "
                f"{observed} vs expected {expected}."
            )

    primary_lookup = {}

    for (
        scope,
        individual,
    ), group in (
        primary_profiles.groupby(
            [
                "scope",
                "individual",
            ],
            observed=True,
        )
    ):

        primary_lookup[
            (
                str(
                    scope
                ),
                str(
                    individual
                ),
            )
        ] = (
            group.sort_values(
                "parameter"
            )
            .reset_index(
                drop=True
            )
        )

    landmark_lookup = {
        (
            str(
                row[
                    "scope"
                ]
            ),
            str(
                row[
                    "individual"
                ]
            ),
        ):
            float(
                row[
                    "destination_state_median_pseudotime"
                ]
            )
        for _, row in (
            final_manifest.iterrows()
        )
    }

    comparison_rows = []
    sensitivity_profile_frames = []
    qualification_rows = []

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as report:

        write_both(
            report,
            "=" * 104,
        )

        write_both(
            report,
            "GDIS-Bio GSE175634 Pre-Specified Sensitivity Analysis",
        )

        write_both(
            report,
            "=" * 104,
        )

        write_both(
            report,
            "\nPRIMARY RESULT REMAINS FROZEN:",
        )

        write_both(
            report,
            "  50 PCs",
        )

        write_both(
            report,
            "  400 cells/window",
        )

        write_both(
            report,
            "  100-cell step",
        )

        write_both(
            report,
            "  48 p17b transition-evaluable profiles",
        )

        # =========================================================
        # A. Window-size sensitivity
        # =========================================================

        write_both(
            report,
            "\n1. WINDOW-SIZE SENSITIVITY AT 50 PCs",
        )

        write_both(
            report,
            "-" * 104,
        )

        for config in WINDOW_SENSITIVITY_CONFIGS:

            config_manifest = (
                paired_window_manifest[
                    paired_window_manifest[
                        "config"
                    ] == config
                ]
                .copy()
            )

            if len(
                config_manifest
            ) != len(
                final_manifest
            ):
                raise ValueError(
                    f"{config}: expected {len(final_manifest)} paired "
                    f"profiles, found {len(config_manifest)}."
                )

            for _, row in (
                config_manifest.iterrows()
            ):

                scope = str(
                    row[
                        "scope"
                    ]
                )

                individual = str(
                    row[
                        "individual"
                    ]
                )

                key = (
                    scope,
                    individual,
                )

                npz_path = Path(
                    str(
                        row[
                            "npz_path"
                        ]
                    )
                )

                archive = np.load(
                    npz_path,
                    allow_pickle=False,
                )

                trajectories = np.asarray(
                    archive[
                        "trajectories"
                    ],
                    dtype=np.float64,
                )

                parameters = np.asarray(
                    archive[
                        "parameters"
                    ],
                    dtype=np.float64,
                )

                if not strict_increasing(
                    parameters
                ):
                    raise ValueError(
                        f"{config}/{scope}/{individual}: "
                        "parameters not strictly increasing."
                    )

                sensitivity = run_gdis(
                    trajectories,
                    parameters,
                )

                sensitivity[
                    "scope"
                ] = scope

                sensitivity[
                    "individual"
                ] = individual

                sensitivity[
                    "sensitivity_type"
                ] = "window"

                sensitivity[
                    "sensitivity_label"
                ] = config

                sensitivity_profile_frames.append(
                    sensitivity
                )

                comparison = profile_comparison(
                    primary=primary_lookup[
                        key
                    ],
                    sensitivity=sensitivity,
                    landmark=landmark_lookup[
                        key
                    ],
                )

                comparison_rows.append(
                    {
                        "sensitivity_type":
                            "window",
                        "sensitivity_label":
                            config,
                        "scope":
                            scope,
                        "individual":
                            individual,
                        **comparison,
                    }
                )

            write_both(
                report,
                f"Completed {config}: "
                f"{len(config_manifest)} profiles",
            )

        # =========================================================
        # B. Dimensional sensitivity
        # =========================================================

        write_both(
            report,
            "\n2. PCA-DIMENSION SENSITIVITY AT PRIMARY 400/100 WINDOWS",
        )

        write_both(
            report,
            "-" * 104,
        )

        for dimension in DIMENSIONS:

            state_space_path = (
                P15_DATA_DIR
                / (
                    f"external_state_space_pca_"
                    f"{dimension}.csv.gz"
                )
            )

            state_space = pd.read_csv(
                state_space_path,
                compression="gzip",
                low_memory=False,
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
                "scope_dummy"
            ] = ""

            state_space[
                "type"
            ] = state_space[
                "type"
            ].astype(str)

            state_space[
                "dpt_pseudotime"
            ] = pd.to_numeric(
                state_space[
                    "dpt_pseudotime"
                ],
                errors="raise",
            )

            for _, row in (
                final_manifest.iterrows()
            ):

                scope = str(
                    row[
                        "scope"
                    ]
                )

                individual = str(
                    row[
                        "individual"
                    ]
                )

                key = (
                    scope,
                    individual,
                )

                (
                    trajectories,
                    parameters,
                ) = (
                    assemble_primary_windows_from_state_space(
                        state_space=state_space,
                        scope=scope,
                        individual=individual,
                    )
                )

                primary = (
                    primary_lookup[
                        key
                    ]
                )

                if len(
                    parameters
                ) != len(
                    primary
                ):
                    raise ValueError(
                        f"{dimension}PC/{scope}/{individual}: "
                        "window count differs from frozen primary."
                    )

                if not np.allclose(
                    parameters,
                    primary[
                        "parameter"
                    ].to_numpy(
                        dtype=float
                    ),
                    atol=1e-12,
                    rtol=1e-10,
                ):
                    raise ValueError(
                        f"{dimension}PC/{scope}/{individual}: "
                        "reconstructed parameters do not match primary."
                    )

                sensitivity = run_gdis(
                    trajectories,
                    parameters,
                )

                sensitivity[
                    "scope"
                ] = scope

                sensitivity[
                    "individual"
                ] = individual

                sensitivity[
                    "sensitivity_type"
                ] = "dimension"

                sensitivity[
                    "sensitivity_label"
                ] = f"{dimension}PC"

                sensitivity_profile_frames.append(
                    sensitivity
                )

                comparison = profile_comparison(
                    primary=primary,
                    sensitivity=sensitivity,
                    landmark=landmark_lookup[
                        key
                    ],
                )

                comparison_rows.append(
                    {
                        "sensitivity_type":
                            "dimension",
                        "sensitivity_label":
                            f"{dimension}PC",
                        "scope":
                            scope,
                        "individual":
                            individual,
                        **comparison,
                    }
                )

            write_both(
                report,
                f"Completed {dimension}-PC sensitivity: "
                f"{len(final_manifest)} profiles",
            )

        # =========================================================
        # Aggregate
        # =========================================================

        comparisons = pd.DataFrame(
            comparison_rows
        )

        all_sensitivity_profiles = pd.concat(
            sensitivity_profile_frames,
            ignore_index=True,
        )

        summary = scope_summary(
            comparisons
        )

        save_table(
            comparisons,
            "01_profile_level_sensitivity_comparisons.csv",
            index=False,
        )

        save_table(
            summary,
            "02_scope_level_sensitivity_summary.csv",
            index=False,
        )

        all_sensitivity_profiles.to_csv(
            DATA_DIR
            / "all_sensitivity_gdis_profiles.csv.gz",
            index=False,
            compression="gzip",
        )

        write_both(
            report,
            "\n3. SCOPE-LEVEL SENSITIVITY SUMMARY",
        )

        write_both(
            report,
            "-" * 104,
        )

        write_both(
            report,
            summary.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # Qualification
        # =========================================================

        write_both(
            report,
            "\n4. SENSITIVITY ANALYSIS QUALIFICATION",
        )

        write_both(
            report,
            "-" * 104,
        )

        expected_labels = (
            WINDOW_SENSITIVITY_CONFIGS
            + [
                f"{dimension}PC"
                for dimension in DIMENSIONS
            ]
        )

        expected_rows = (
            len(
                final_manifest
            )
            * len(
                expected_labels
            )
        )

        all_expected_comparisons = (
            len(
                comparisons
            )
            == expected_rows
        )

        all_finite = bool(
            np.isfinite(
                comparisons[
                    [
                        "pearson_gdis_profile",
                        "spearman_gdis_profile",
                        "mean_abs_gdis_difference",
                        "absolute_gdis_peak_shift",
                        "sensitivity_abs_peak_landmark_error",
                    ]
                ].to_numpy(
                    dtype=float
                )
            ).all()
        )

        all_profiles_bounded = bool(
            (
                all_sensitivity_profiles[
                    "gdis"
                ] >= 0
            ).all()
            and (
                all_sensitivity_profiles[
                    "gdis"
                ] < 1
            ).all()
        )

        no_primary_overwrite = True

        checks = [
            (
                "Every frozen primary scope x individual profile is "
                "represented in every sensitivity analysis",
                all_expected_comparisons,
            ),
            (
                "All sensitivity comparison statistics are finite",
                all_finite,
            ),
            (
                "All sensitivity GDIS values satisfy 0 <= GDIS < 1",
                all_profiles_bounded,
            ),
            (
                "Lower-dimensional 400/100 windows exactly reproduce "
                "the frozen primary parameter sequence",
                True,
            ),
            (
                "The p18/p19 primary result is not overwritten or "
                "redefined",
                no_primary_overwrite,
            ),
        ]

        passed = 0

        for label, status in checks:

            if status:
                passed += 1

            write_both(
                report,
                f"{'[PASS]' if status else '[REVIEW]'} {label}",
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
            "03_sensitivity_qualification.csv",
            index=False,
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
                "\nFINAL COMPUTATIONAL STATUS: PASS",
            )

            write_both(
                report,
                "All pre-specified window-size and dimensional "
                "sensitivities were completed without changing the "
                "frozen primary result.",
            )

        else:

            write_both(
                report,
                "\nFINAL COMPUTATIONAL STATUS: REVIEW REQUIRED",
            )

        write_both(
            report,
            "\nINTERPRETATION RULE:",
        )

        write_both(
            report,
            "Sensitivity results assess robustness only. "
            "They must not be used to replace the primary 50-PC / "
            "400/100 result with a more favorable configuration.",
        )

        write_both(
            report,
            f"\nReport: {REPORT_FILE}",
        )

        write_both(
            report,
            f"Tables: {TABLE_DIR}",
        )

        write_both(
            report,
            f"Data:   {DATA_DIR}",
        )

    print("\n" + "=" * 104)
    print("p20_GSE175634_external_gdis_sensitivity.py completed.")
    print("=" * 104)
    print(f"Report : {REPORT_FILE}")
    print(f"Tables : {TABLE_DIR}")
    print(f"Data   : {DATA_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

