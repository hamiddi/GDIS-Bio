#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================
"""
p17b_GSE175634_transition_evaluability_freeze.py

Resolve the single p17 trajectory-assembly review condition and freeze the
FINAL external-validation GDIS group manifest before any GDIS calculation.

Background
----------
p17 assembled all frozen GSE175634 trajectory families successfully, but the
pre-specified destination-state median landmark was outside the PRIMARY
400/100 parameter range for:

    individual 19108 / cm_extension
    individual 19108 / cf_extension

No GDIS values have been calculated.

Scientific rule
---------------
A scope x individual profile is eligible for transition-specific PRIMARY
external validation only when the pre-specified destination-state median
landmark lies inside the analyzable PRIMARY 400/100 parameter range.

This rule was already part of the p17 qualification and is therefore NOT a
post-GDIS optimization.

Consequences
------------
- shared_backbone remains unchanged.
- individual 19108 remains in shared_backbone.
- individual 19108 is excluded only from CM- and CF-terminal-transition
  external GDIS analyses.
- no window is padded.
- no incomplete tail is added.
- no landmark is moved.
- no window size or PCA dimension is changed.
- no GDIS parameter is changed.

This script also audits landmark coverage for all three transferred window
configurations, but the FINAL PRIMARY GDIS manifest is defined ONLY from the
frozen primary_w400_s100 evaluability rule.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


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

P17_DIR = (
    PROJECT_DIR
    / "results"
    / "p17_external_GSE175634_trajectory_window_assembly"
)

P17_TABLE_DIR = P17_DIR / "tables"
P17_DATA_DIR = P17_DIR / "data"

TRANSITION_FILE = (
    P14_TABLE_DIR
    / "05_prespecified_biological_transitions.csv"
)

SUMMARY_FILE = (
    P17_TABLE_DIR
    / "01_all_group_configuration_summary.csv"
)

MANIFEST_FILE = (
    P17_TABLE_DIR
    / "02_trajectory_file_manifest.csv"
)

ALL_WINDOWS_FILE = (
    P17_TABLE_DIR
    / "03_all_window_metadata.csv.gz"
)

PRIMARY_TRANSITION_COVERAGE_FILE = (
    P17_TABLE_DIR
    / "07_primary_transition_landmark_coverage.csv"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
    / "p17b_external_GSE175634_transition_evaluability_freeze"
)

TABLE_DIR = RESULTS_DIR / "tables"

REPORT_FILE = (
    RESULTS_DIR
    / "p17b_external_GSE175634_transition_evaluability_freeze_report.txt"
)


# ---------------------------------------------------------------------
# Frozen definitions
# ---------------------------------------------------------------------

PRIMARY_CONFIG = "primary_w400_s100"

CONFIGS = [
    "sensitivity_w300_s75",
    "primary_w400_s100",
    "sensitivity_w500_s125",
]

SCOPE_TRANSITION = {
    "shared_backbone": "T2",
    "cm_extension": "T3_CM",
    "cf_extension": "T3_CF",
}

MIN_FINAL_GROUPS_PER_SCOPE = 10


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


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    required = [
        TRANSITION_FILE,
        SUMMARY_FILE,
        MANIFEST_FILE,
        ALL_WINDOWS_FILE,
        PRIMARY_TRANSITION_COVERAGE_FILE,
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required inputs:\n"
            + "\n".join(missing)
        )

    transitions = pd.read_csv(
        TRANSITION_FILE,
        low_memory=False,
    )

    summary = pd.read_csv(
        SUMMARY_FILE,
        low_memory=False,
    )

    manifest = pd.read_csv(
        MANIFEST_FILE,
        low_memory=False,
    )

    windows = pd.read_csv(
        ALL_WINDOWS_FILE,
        compression="gzip",
        low_memory=False,
    )

    primary_coverage = pd.read_csv(
        PRIMARY_TRANSITION_COVERAGE_FILE,
        low_memory=False,
    )

    for table in [
        summary,
        manifest,
        windows,
        primary_coverage,
    ]:
        if "individual" in table.columns:
            table[
                "individual"
            ] = table[
                "individual"
            ].astype(str)

    transition_lookup = (
        transitions.set_index(
            "transition_id"
        )
    )

    # -----------------------------------------------------------------
    # Individual-specific frozen landmark values from p17.
    # -----------------------------------------------------------------

    landmark_lookup = {}

    for _, row in (
        primary_coverage.iterrows()
    ):

        key = (
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
        )

        landmark_lookup[
            key
        ] = float(
            row[
                "destination_state_median_pseudotime"
            ]
        )

    # -----------------------------------------------------------------
    # Audit every transferred window configuration.
    # -----------------------------------------------------------------

    audit_rows = []

    grouped = (
        windows.groupby(
            [
                "config",
                "scope",
                "individual",
            ],
            observed=True,
            sort=True,
        )
    )

    for (
        config,
        scope,
        individual,
    ), group in grouped:

        config = str(
            config
        )

        scope = str(
            scope
        )

        individual = str(
            individual
        )

        key = (
            scope,
            individual,
        )

        if key not in landmark_lookup:
            raise ValueError(
                f"Missing frozen transition landmark for "
                f"{scope}, individual {individual}"
            )

        transition_id = (
            SCOPE_TRANSITION[
                scope
            ]
        )

        destination_state = str(
            transition_lookup.loc[
                transition_id,
                "destination_state",
            ]
        )

        landmark = float(
            landmark_lookup[
                key
            ]
        )

        parameters = pd.to_numeric(
            group[
                "parameter_median_pseudotime"
            ],
            errors="raise",
        ).to_numpy(
            dtype=float
        )

        parameter_min = float(
            np.min(
                parameters
            )
        )

        parameter_max = float(
            np.max(
                parameters
            )
        )

        inside = bool(
            parameter_min
            <= landmark
            <= parameter_max
        )

        audit_rows.append(
            {
                "config":
                    config,
                "scope":
                    scope,
                "individual":
                    individual,
                "transition_id":
                    transition_id,
                "destination_state":
                    destination_state,
                "destination_state_median_pseudotime":
                    landmark,
                "parameter_min":
                    parameter_min,
                "parameter_max":
                    parameter_max,
                "landmark_minus_parameter_max":
                    (
                        landmark
                        - parameter_max
                    ),
                "landmark_inside_parameter_range":
                    inside,
                "n_windows":
                    int(
                        len(
                            group
                        )
                    ),
            }
        )

    audit = pd.DataFrame(
        audit_rows
    )

    save_table(
        audit,
        "01_transition_evaluability_all_configs.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # Freeze primary-evaluable scope x individual groups.
    # -----------------------------------------------------------------

    primary_audit = (
        audit[
            audit[
                "config"
            ] == PRIMARY_CONFIG
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    primary_audit[
        "final_primary_gdis_eligible"
    ] = primary_audit[
        "landmark_inside_parameter_range"
    ].astype(bool)

    excluded = (
        primary_audit[
            ~primary_audit[
                "final_primary_gdis_eligible"
            ]
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    included = (
        primary_audit[
            primary_audit[
                "final_primary_gdis_eligible"
            ]
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    save_table(
        included,
        "02_final_primary_gdis_groups.csv",
        index=False,
    )

    save_table(
        excluded,
        "03_excluded_primary_groups.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # Final file manifest for p18.
    # -----------------------------------------------------------------

    final_manifest = (
        manifest[
            manifest[
                "config"
            ] == PRIMARY_CONFIG
        ]
        .merge(
            included[
                [
                    "scope",
                    "individual",
                    "transition_id",
                    "destination_state",
                    "destination_state_median_pseudotime",
                ]
            ],
            on=[
                "scope",
                "individual",
            ],
            how="inner",
            validate="one_to_one",
        )
        .sort_values(
            [
                "scope",
                "individual",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # Confirm all referenced files exist before freezing the manifest.
    file_checks = []

    for _, row in (
        final_manifest.iterrows()
    ):

        npz_path = Path(
            str(
                row[
                    "npz_path"
                ]
            )
        )

        metadata_path = Path(
            str(
                row[
                    "window_metadata_path"
                ]
            )
        )

        file_checks.append(
            npz_path.exists()
            and metadata_path.exists()
        )

    final_manifest[
        "referenced_files_exist"
    ] = file_checks

    if not bool(
        final_manifest[
            "referenced_files_exist"
        ].all()
    ):
        bad = final_manifest[
            ~final_manifest[
                "referenced_files_exist"
            ]
        ]

        raise FileNotFoundError(
            "One or more final GDIS trajectory files are missing:\n"
            + bad.to_string(
                index=False
            )
        )

    save_table(
        final_manifest,
        "04_final_primary_gdis_file_manifest.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # Sensitivity manifest:
    # keep the SAME frozen primary-evaluable scope x individual groups
    # in every transferred window configuration.
    # -----------------------------------------------------------------

    paired_sensitivity_manifest = (
        manifest.merge(
            included[
                [
                    "scope",
                    "individual",
                ]
            ].drop_duplicates(),
            on=[
                "scope",
                "individual",
            ],
            how="inner",
            validate="many_to_one",
        )
        .merge(
            audit[
                [
                    "config",
                    "scope",
                    "individual",
                    "landmark_inside_parameter_range",
                ]
            ],
            on=[
                "config",
                "scope",
                "individual",
            ],
            how="left",
            validate="one_to_one",
        )
        .sort_values(
            [
                "config",
                "scope",
                "individual",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    save_table(
        paired_sensitivity_manifest,
        "05_paired_window_sensitivity_manifest.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # Counts
    # -----------------------------------------------------------------

    count_rows = []

    for scope in [
        "shared_backbone",
        "cm_extension",
        "cf_extension",
    ]:

        original_count = int(
            primary_audit[
                primary_audit[
                    "scope"
                ] == scope
            ][
                "individual"
            ].nunique()
        )

        final_count = int(
            included[
                included[
                    "scope"
                ] == scope
            ][
                "individual"
            ].nunique()
        )

        excluded_count = (
            original_count
            - final_count
        )

        count_rows.append(
            {
                "scope":
                    scope,
                "p17_primary_groups":
                    original_count,
                "final_primary_gdis_groups":
                    final_count,
                "excluded_for_transition_evaluability":
                    excluded_count,
            }
        )

    counts = pd.DataFrame(
        count_rows
    )

    save_table(
        counts,
        "06_final_group_counts.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # Qualification
    # -----------------------------------------------------------------

    all_final_inside = bool(
        included[
            "landmark_inside_parameter_range"
        ].all()
    )

    all_files_exist = bool(
        final_manifest[
            "referenced_files_exist"
        ].all()
    )

    counts_sufficient = bool(
        (
            counts[
                "final_primary_gdis_groups"
            ]
            >= MIN_FINAL_GROUPS_PER_SCOPE
        ).all()
    )

    no_shared_exclusions = bool(
        counts.loc[
            counts[
                "scope"
            ] == "shared_backbone",
            "excluded_for_transition_evaluability",
        ].iloc[
            0
        ]
        == 0
    )

    # Confirm p17's two known terminal failures are the only exclusions.
    exclusion_pairs = set(
        zip(
            excluded[
                "scope"
            ].astype(str),
            excluded[
                "individual"
            ].astype(str),
        )
    )

    expected_exclusions = {
        (
            "cm_extension",
            "19108",
        ),
        (
            "cf_extension",
            "19108",
        ),
    }

    exclusions_match_expected = (
        exclusion_pairs
        == expected_exclusions
    )

    qualification = pd.DataFrame(
        [
            {
                "check":
                    "All final primary GDIS groups contain their "
                    "pre-specified transition landmark",
                "pass":
                    all_final_inside,
            },
            {
                "check":
                    "All final trajectory and metadata files exist",
                "pass":
                    all_files_exist,
            },
            {
                "check":
                    f"Each final scope retains at least "
                    f"{MIN_FINAL_GROUPS_PER_SCOPE} independent individuals",
                "pass":
                    counts_sufficient,
            },
            {
                "check":
                    "Shared-backbone cohort remains unchanged",
                "pass":
                    no_shared_exclusions,
            },
            {
                "check":
                    "Only the two p17-flagged individual 19108 terminal "
                    "profiles are excluded",
                "pass":
                    exclusions_match_expected,
            },
        ]
    )

    save_table(
        qualification,
        "07_final_evaluability_qualification.csv",
        index=False,
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
            "=" * 100,
        )

        write_both(
            report,
            "GDIS-Bio GSE175634 Transition-Evaluability Freeze",
        )

        write_both(
            report,
            "=" * 100,
        )

        write_both(
            report,
            "\nNO GDIS VALUES HAVE BEEN CALCULATED.",
        )

        write_both(
            report,
            "This step resolves the pre-GDIS p17 transition-landmark "
            "coverage criterion without changing windows, landmarks, "
            "PCA dimensions, or GDIS settings.",
        )

        write_both(
            report,
            "\n1. P17 PRIMARY GROUPS FAILING TRANSITION EVALUABILITY",
        )

        write_both(
            report,
            "-" * 100,
        )

        if excluded.empty:

            write_both(
                report,
                "None.",
            )

        else:

            write_both(
                report,
                excluded[
                    [
                        "scope",
                        "individual",
                        "transition_id",
                        "destination_state",
                        "destination_state_median_pseudotime",
                        "parameter_min",
                        "parameter_max",
                        "landmark_minus_parameter_max",
                    ]
                ].round(
                    6
                ).to_string(
                    index=False
                ),
            )

        write_both(
            report,
            "\nInterpretation:",
        )

        write_both(
            report,
            "These profiles do not extend far enough in the transferred "
            "400/100 complete-window family to reach the frozen terminal "
            "state median landmark.",
        )

        write_both(
            report,
            "They are excluded from terminal-transition GDIS validation "
            "rather than modifying the window construction.",
        )

        write_both(
            report,
            "\n2. FINAL PRIMARY EXTERNAL GDIS COHORT",
        )

        write_both(
            report,
            "-" * 100,
        )

        write_both(
            report,
            counts.to_string(
                index=False
            ),
        )

        write_both(
            report,
            f"\nTotal final primary GDIS profiles: "
            f"{len(included)}",
        )

        write_both(
            report,
            "\n3. WINDOW-CONFIGURATION LANDMARK AUDIT",
        )

        write_both(
            report,
            "-" * 100,
        )

        config_summary = (
            audit.groupby(
                [
                    "config",
                    "scope",
                ],
                observed=True,
            )
            .agg(
                n_groups=(
                    "individual",
                    "nunique",
                ),
                n_landmark_inside=(
                    "landmark_inside_parameter_range",
                    "sum",
                ),
                min_parameter_max=(
                    "parameter_max",
                    "min",
                ),
                max_landmark_overrun=(
                    "landmark_minus_parameter_max",
                    "max",
                ),
            )
            .reset_index()
        )

        write_both(
            report,
            config_summary.round(
                6
            ).to_string(
                index=False
            ),
        )

        save_table(
            config_summary,
            "08_window_configuration_evaluability_summary.csv",
            index=False,
        )

        write_both(
            report,
            "\n4. FINAL QUALIFICATION",
        )

        write_both(
            report,
            "-" * 100,
        )

        passed = 0

        for _, row in (
            qualification.iterrows()
        ):

            status = bool(
                row[
                    "pass"
                ]
            )

            if status:
                passed += 1

            write_both(
                report,
                f"{'[PASS]' if status else '[REVIEW]'} "
                f"{row['check']}",
            )

        write_both(
            report,
            f"\nQualification checks passed: "
            f"{passed}/{len(qualification)}",
        )

        if passed == len(
            qualification
        ):

            write_both(
                report,
                "\nFINAL STATUS: PASS",
            )

            write_both(
                report,
                "The transition-evaluable external-validation cohort "
                "is now frozen before GDIS.",
            )

            write_both(
                report,
                "Final primary replicate counts:",
            )

            for _, row in (
                counts.iterrows()
            ):

                write_both(
                    report,
                    f"  {row['scope']}: "
                    f"{int(row['final_primary_gdis_groups'])}",
                )

            write_both(
                report,
                f"  TOTAL: {len(included)}",
            )

            write_both(
                report,
                "\nThe next step may run unchanged pyGDIS independently "
                "on the final manifest.",
            )

        else:

            write_both(
                report,
                "\nFINAL STATUS: REVIEW REQUIRED",
            )

        write_both(
            report,
            "\nIMPORTANT:",
        )

        write_both(
            report,
            "Individual 19108 remains in the shared-backbone analysis.",
        )

        write_both(
            report,
            "No incomplete window tail was added or padded.",
        )

        write_both(
            report,
            "No transition landmark was moved.",
        )

        write_both(
            report,
            "No PCA dimension, window size, step size, or GDIS parameter "
            "was changed.",
        )

        write_both(
            report,
            "Sensitivity analyses use the same frozen primary-evaluable "
            "scope x individual groups for paired comparability.",
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

    print("\n" + "=" * 100)
    print("p17b_GSE175634_transition_evaluability_freeze.py completed.")
    print("=" * 100)
    print(f"Report : {REPORT_FILE}")
    print(f"Tables : {TABLE_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

