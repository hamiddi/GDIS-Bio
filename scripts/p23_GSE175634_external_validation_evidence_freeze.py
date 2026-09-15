#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================
"""
p23_GSE175634_external_validation_evidence_freeze.py

Freeze and consolidate the completed GSE175634 external-validation evidence
WITHOUT performing any new inferential analysis and WITHOUT recomputing GDIS.

Purpose
-------
The external validation is complete through p22. This script creates a compact,
manuscript-ready evidence freeze that combines:

    p17b  final transition-evaluable cohort
    p19   primary GDIS statistical validation
    p20   pre-specified sensitivity analysis
    p21   conventional benchmark
    p22   paired statistical benchmark comparison

No biological landmark, cohort, window configuration, PCA dimension, metric,
or statistical result is changed.

Frozen scientific interpretation
--------------------------------
1. GDIS transition-region localization generalizes to independent human
   cardiac differentiation data.

2. GDIS localization is significantly better than its structure-preserving
   circular-shift null for T2, T3_CM, and T3_CF.

3. GDIS is NOT universally superior to conventional early-warning metrics.
   Variance, Gaussian entropy, and pseudotemporal step distance are competitive
   and are descriptively better in some terminal-transition settings.

4. GDIS shows statistically supported superiority over lag-1 pseudotemporal
   autocorrelation for T3_CM and T3_CF after FDR correction.

5. The internal transition-energy component does not generalize as a locator
   of these later cardiac transitions and is consistently less accurate than
   full GDIS.

6. The external result is robust to the pre-specified window-size sensitivity
   and especially to 30-PC versus 50-PC state-space sensitivity.

7. Destination-state medians/IQRs are biological state landmarks, not true
   transition-onset times. No prospective onset-prediction claim is made.

Outputs
-------
results/p23_external_GSE175634_evidence_freeze/
    p23_external_GSE175634_evidence_freeze_report.txt
    tables/
        01_final_external_gdis_evidence.csv
        02_paired_gdis_vs_conventional_evidence.csv
        03_sensitivity_evidence.csv
        04_external_validation_claims_guardrails.csv
        05_final_external_validation_cohort.csv
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

P17B_TABLE_DIR = (
    PROJECT_DIR
    / "results"
    / "p17b_external_GSE175634_transition_evaluability_freeze"
    / "tables"
)

P19_TABLE_DIR = (
    PROJECT_DIR
    / "results"
    / "p19_external_GSE175634_gdis_statistical_validation"
    / "tables"
)

P20_TABLE_DIR = (
    PROJECT_DIR
    / "results"
    / "p20_external_GSE175634_gdis_sensitivity"
    / "tables"
)

P21_TABLE_DIR = (
    PROJECT_DIR
    / "results"
    / "p21_external_GSE175634_conventional_ews_benchmark"
    / "tables"
)

P22_TABLE_DIR = (
    PROJECT_DIR
    / "results"
    / "p22_external_GSE175634_benchmark_statistical_comparison"
    / "tables"
)

FINAL_COHORT_FILE = (
    P17B_TABLE_DIR
    / "06_final_group_counts.csv"
)

P19_GDIS_SUMMARY_FILE = (
    P19_TABLE_DIR
    / "07_primary_gdis_external_validation_summary.csv"
)

P20_SENSITIVITY_FILE = (
    P20_TABLE_DIR
    / "02_scope_level_sensitivity_summary.csv"
)

P21_BENCHMARK_FILE = (
    P21_TABLE_DIR
    / "08_integrated_descriptive_benchmark.csv"
)

P22_NULL_FILE = (
    P22_TABLE_DIR
    / "01_all_metric_circular_shift_tests.csv"
)

P22_PAIRED_FILE = (
    P22_TABLE_DIR
    / "04_paired_gdis_vs_conventional_localization.csv"
)

P22_INTERNAL_FILE = (
    P22_TABLE_DIR
    / "05_gdis_vs_transition_energy_localization.csv"
)

P22_RANKING_FILE = (
    P22_TABLE_DIR
    / "06_descriptive_localization_ranking.csv"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
    / "p23_external_GSE175634_evidence_freeze"
)

TABLE_DIR = RESULTS_DIR / "tables"

REPORT_FILE = (
    RESULTS_DIR
    / "p23_external_GSE175634_evidence_freeze_report.txt"
)


# ---------------------------------------------------------------------
# Frozen expectations
# ---------------------------------------------------------------------

SCOPE_ORDER = [
    "shared_backbone",
    "cm_extension",
    "cf_extension",
]

EXPECTED_SCOPE_COUNTS = {
    "shared_backbone": 19,
    "cm_extension": 15,
    "cf_extension": 14,
}

SCOPE_TRANSITION = {
    "shared_backbone": "T2",
    "cm_extension": "T3_CM",
    "cf_extension": "T3_CF",
}

CONVENTIONAL_BASELINES = [
    "total_variance",
    "mean_lag1_autocorrelation",
    "gaussian_differential_entropy",
    "mean_step_distance",
]

ALPHA = 0.05


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def write_both(report, text=""):
    print(text)
    report.write(text + "\n")
    report.flush()


def save_table(df, filename, index=False):
    path = TABLE_DIR / filename
    df.to_csv(path, index=index)
    return path


def bool_text(value):
    return "YES" if bool(value) else "NO"


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    required_files = [
        FINAL_COHORT_FILE,
        P19_GDIS_SUMMARY_FILE,
        P20_SENSITIVITY_FILE,
        P21_BENCHMARK_FILE,
        P22_NULL_FILE,
        P22_PAIRED_FILE,
        P22_INTERNAL_FILE,
        P22_RANKING_FILE,
    ]

    missing = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required completed-analysis files:\n"
            + "\n".join(missing)
        )

    cohort = pd.read_csv(
        FINAL_COHORT_FILE,
        low_memory=False,
    )

    gdis = pd.read_csv(
        P19_GDIS_SUMMARY_FILE,
        low_memory=False,
    )

    sensitivity = pd.read_csv(
        P20_SENSITIVITY_FILE,
        low_memory=False,
    )

    benchmark = pd.read_csv(
        P21_BENCHMARK_FILE,
        low_memory=False,
    )

    null_tests = pd.read_csv(
        P22_NULL_FILE,
        low_memory=False,
    )

    paired = pd.read_csv(
        P22_PAIRED_FILE,
        low_memory=False,
    )

    internal = pd.read_csv(
        P22_INTERNAL_FILE,
        low_memory=False,
    )

    ranking = pd.read_csv(
        P22_RANKING_FILE,
        low_memory=False,
    )

    # -----------------------------------------------------------------
    # Validate completed frozen cohort.
    # -----------------------------------------------------------------

    cohort[
        "scope"
    ] = cohort[
        "scope"
    ].astype(str)

    for scope, expected in EXPECTED_SCOPE_COUNTS.items():

        row = cohort[
            cohort[
                "scope"
            ] == scope
        ]

        if len(row) != 1:
            raise ValueError(
                f"Expected one cohort row for {scope}."
            )

        observed = int(
            row[
                "final_primary_gdis_groups"
            ].iloc[0]
        )

        if observed != expected:
            raise ValueError(
                f"{scope}: expected frozen count {expected}, "
                f"observed {observed}."
            )

    # -----------------------------------------------------------------
    # 1. Final GDIS evidence table.
    # -----------------------------------------------------------------

    gdis_rows = []

    for scope in SCOPE_ORDER:

        transition_id = (
            SCOPE_TRANSITION[
                scope
            ]
        )

        p19_row = gdis[
            (
                gdis[
                    "scope"
                ] == scope
            )
            & (
                gdis[
                    "transition_id"
                ] == transition_id
            )
        ]

        if len(
            p19_row
        ) != 1:
            raise ValueError(
                f"Missing/duplicate p19 GDIS summary for "
                f"{scope}/{transition_id}."
            )

        p19_row = p19_row.iloc[
            0
        ]

        p22_gdis = null_tests[
            (
                null_tests[
                    "scope"
                ] == scope
            )
            & (
                null_tests[
                    "metric"
                ] == "gdis"
            )
        ]

        if len(
            p22_gdis
        ) != 1:
            raise ValueError(
                f"Missing/duplicate p22 GDIS null result for {scope}."
            )

        p22_gdis = p22_gdis.iloc[
            0
        ]

        rank_row = ranking[
            (
                ranking[
                    "scope"
                ] == scope
            )
            & (
                ranking[
                    "metric"
                ] == "gdis"
            )
        ]

        rank_value = (
            float(
                rank_row[
                    "localization_rank_within_scope"
                ].iloc[0]
            )
            if len(
                rank_row
            ) == 1
            else np.nan
        )

        gdis_rows.append(
            {
                "scope":
                    scope,
                "transition_id":
                    transition_id,
                "n_profiles":
                    int(
                        p19_row[
                            "n_profiles"
                        ]
                    ),
                "median_absolute_gdis_peak_error":
                    float(
                        p19_row[
                            "observed_median_abs_peak_error"
                        ]
                    ),
                "median_signed_gdis_peak_offset":
                    float(
                        p19_row[
                            "observed_median_signed_peak_offset"
                        ]
                    ),
                "gdis_localization_p":
                    float(
                        p22_gdis[
                            "p_localization_lower"
                        ]
                    ),
                "gdis_localization_q":
                    float(
                        p22_gdis[
                            "q_localization_bh"
                        ]
                    ),
                "gdis_localization_significant_fdr":
                    bool(
                        float(
                            p22_gdis[
                                "q_localization_bh"
                            ]
                        )
                        < ALPHA
                    ),
                "n_positive_source_destination_change":
                    int(
                        p19_row[
                            "n_positive_delta"
                        ]
                    ),
                "n_negative_source_destination_change":
                    int(
                        p19_row[
                            "n_negative_delta"
                        ]
                    ),
                "median_source_destination_gdis_change":
                    float(
                        p19_row[
                            "observed_median_source_to_destination_delta"
                        ]
                    ),
                "source_destination_wilcoxon_q":
                    float(
                        p19_row[
                            "q_wilcoxon_greater_bh"
                        ]
                    ),
                "source_destination_increase_significant_fdr":
                    bool(
                        float(
                            p19_row[
                                "q_wilcoxon_greater_bh"
                            ]
                        )
                        < ALPHA
                    ),
                "gdis_descriptive_localization_rank_among_6_metrics":
                    rank_value,
            }
        )

    final_gdis = pd.DataFrame(
        gdis_rows
    )

    save_table(
        final_gdis,
        "01_final_external_gdis_evidence.csv",
    )

    # -----------------------------------------------------------------
    # 2. Paired conventional comparator evidence.
    # -----------------------------------------------------------------

    paired_out = paired.copy()

    paired_out[
        "sign_test_superiority_significant_fdr"
    ] = (
        paired_out[
            "q_sign_gdis_better_bh"
        ]
        < ALPHA
    )

    paired_out[
        "wilcoxon_superiority_significant_fdr"
    ] = (
        paired_out[
            "q_wilcoxon_gdis_better_bh"
        ]
        < ALPHA
    )

    paired_out[
        "paired_superiority_supported"
    ] = (
        paired_out[
            "sign_test_superiority_significant_fdr"
        ]
        | paired_out[
            "wilcoxon_superiority_significant_fdr"
        ]
    )

    save_table(
        paired_out,
        "02_paired_gdis_vs_conventional_evidence.csv",
    )

    # -----------------------------------------------------------------
    # 3. Compact sensitivity evidence.
    # -----------------------------------------------------------------

    keep_sensitivity = sensitivity[
        (
            (
                sensitivity[
                    "sensitivity_type"
                ] == "window"
            )
            & (
                sensitivity[
                    "sensitivity_label"
                ].isin(
                    [
                        "sensitivity_w300_s75",
                        "sensitivity_w500_s125",
                    ]
                )
            )
        )
        | (
            (
                sensitivity[
                    "sensitivity_type"
                ] == "dimension"
            )
            & (
                sensitivity[
                    "sensitivity_label"
                ] == "30PC"
            )
        )
    ].copy()

    save_table(
        keep_sensitivity,
        "03_sensitivity_evidence.csv",
    )

    # -----------------------------------------------------------------
    # 4. Claims / guardrails freeze.
    # -----------------------------------------------------------------

    claims = pd.DataFrame(
        [
            {
                "claim_id":
                    "C1",
                "status":
                    "SUPPORTED",
                "manuscript_safe_statement":
                    (
                        "GDIS transition-region localization generalized "
                        "to the independent GSE175634 cardiac differentiation "
                        "dataset for T2, T3-CM, and T3-CF."
                    ),
                "guardrail":
                    (
                        "This means significant alignment to independently "
                        "frozen state landmarks versus circular-shift nulls; "
                        "it does not establish exact onset prediction."
                    ),
            },
            {
                "claim_id":
                    "C2",
                "status":
                    "SUPPORTED",
                "manuscript_safe_statement":
                    (
                        "The shared-backbone T2 transition showed both "
                        "significant GDIS localization and a significant "
                        "positive MES-to-CMES GDIS change across individuals."
                    ),
                "guardrail":
                    (
                        "Do not generalize this monotonic increase to all "
                        "terminal transitions."
                    ),
            },
            {
                "claim_id":
                    "C3",
                "status":
                    "SUPPORTED",
                "manuscript_safe_statement":
                    (
                        "GDIS localized the T3-CM and T3-CF terminal "
                        "transition regions significantly versus "
                        "structure-preserving nulls even though the score "
                        "did not show a consistent source-to-destination "
                        "increase."
                    ),
                "guardrail":
                    (
                        "GDIS need not peak at the mature destination-state "
                        "median and should not be interpreted as a monotonic "
                        "maturation score."
                    ),
            },
            {
                "claim_id":
                    "C4",
                "status":
                    "SUPPORTED",
                "manuscript_safe_statement":
                    (
                        "GDIS outperformed lag-1 pseudotemporal "
                        "autocorrelation for T3-CM and T3-CF in paired "
                        "individual-level localization tests after FDR "
                        "correction."
                    ),
                "guardrail":
                    (
                        "T2 showed the same direction but did not remain "
                        "significant after FDR correction."
                    ),
            },
            {
                "claim_id":
                    "C5",
                "status":
                    "NOT SUPPORTED",
                "manuscript_safe_statement":
                    (
                        "Do not claim that GDIS is universally superior "
                        "to variance, Gaussian entropy, or pseudotemporal "
                        "step distance."
                    ),
                "guardrail":
                    (
                        "Those conventional metrics were competitive or "
                        "descriptively better localized in several external "
                        "transition settings, and paired superiority tests "
                        "did not support a general GDIS advantage."
                    ),
            },
            {
                "claim_id":
                    "C6",
                "status":
                    "SUPPORTED",
                "manuscript_safe_statement":
                    (
                        "Full GDIS localized all three external transitions "
                        "substantially better than its internal transition-"
                        "energy component."
                    ),
                "guardrail":
                    (
                        "Transition energy frequently peaked at an earlier "
                        "developmental transition and should not be presented "
                        "as an independently validated late-transition locator."
                    ),
            },
            {
                "claim_id":
                    "C7",
                "status":
                    "SUPPORTED",
                "manuscript_safe_statement":
                    (
                        "The external GDIS result was robust to the "
                        "pre-specified window-size analyses and particularly "
                        "stable between 30-PC and 50-PC state spaces."
                    ),
                "guardrail":
                    (
                        "The primary analysis remains the frozen 50-PC, "
                        "400-cell, 100-step configuration; sensitivity results "
                        "do not replace it."
                    ),
            },
            {
                "claim_id":
                    "C8",
                "status":
                    "NOT SUPPORTED",
                "manuscript_safe_statement":
                    (
                        "Do not describe the external result as prospective "
                        "prediction of biological transition onset."
                    ),
                "guardrail":
                    (
                        "The deposited pseudotime is a rank ordering of "
                        "independent cells, and state medians/IQRs are "
                        "landmarks rather than true onset times."
                    ),
            },
        ]
    )

    save_table(
        claims,
        "04_external_validation_claims_guardrails.csv",
    )

    save_table(
        cohort,
        "05_final_external_validation_cohort.csv",
    )

    # -----------------------------------------------------------------
    # Internal transition-energy compact table.
    # -----------------------------------------------------------------

    internal_out = internal.copy()

    internal_out[
        "gdis_better_sign_test_significant"
    ] = (
        internal_out[
            "p_sign_gdis_better"
        ]
        < ALPHA
    )

    internal_out[
        "gdis_better_wilcoxon_significant"
    ] = (
        internal_out[
            "p_wilcoxon_gdis_better"
        ]
        < ALPHA
    )

    save_table(
        internal_out,
        "06_gdis_vs_transition_energy_evidence.csv",
    )

    # -----------------------------------------------------------------
    # Final report
    # -----------------------------------------------------------------

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as report:

        write_both(
            report,
            "=" * 106,
        )

        write_both(
            report,
            "GDIS-Bio GSE175634 External Validation Evidence Freeze",
        )

        write_both(
            report,
            "=" * 106,
        )

        write_both(
            report,
            "\nNO NEW GDIS OR INFERENTIAL ANALYSIS IS PERFORMED HERE.",
        )

        write_both(
            report,
            "This report consolidates the already completed p17b-p22 "
            "external-validation evidence.",
        )

        write_both(
            report,
            "\n1. FINAL EXTERNAL COHORT",
        )

        write_both(
            report,
            "-" * 106,
        )

        write_both(
            report,
            cohort.to_string(
                index=False
            ),
        )

        write_both(
            report,
            f"\nTotal final primary profiles: "
            f"{int(cohort['final_primary_gdis_groups'].sum())}",
        )

        write_both(
            report,
            "\n2. FINAL PRIMARY GDIS EVIDENCE",
        )

        write_both(
            report,
            "-" * 106,
        )

        write_both(
            report,
            final_gdis.round(
                6
            ).to_string(
                index=False
            ),
        )

        write_both(
            report,
            "\n3. PAIRED GDIS-vs-CONVENTIONAL EVIDENCE",
        )

        write_both(
            report,
            "-" * 106,
        )

        display_columns = [
            "scope",
            "transition_id",
            "baseline_label",
            "n_individuals",
            "median_gdis_abs_error",
            "median_baseline_abs_error",
            "gdis_wins",
            "ties",
            "gdis_losses",
            "q_sign_gdis_better_bh",
            "q_wilcoxon_gdis_better_bh",
            "paired_superiority_supported",
        ]

        write_both(
            report,
            paired_out[
                display_columns
            ].round(
                6
            ).to_string(
                index=False
            ),
        )

        write_both(
            report,
            "\n4. PRE-SPECIFIED SENSITIVITY EVIDENCE",
        )

        write_both(
            report,
            "-" * 106,
        )

        write_both(
            report,
            keep_sensitivity.round(
                6
            ).to_string(
                index=False
            ),
        )

        write_both(
            report,
            "\n5. FROZEN SCIENTIFIC CLAIMS",
        )

        write_both(
            report,
            "-" * 106,
        )

        for _, row in (
            claims.iterrows()
        ):

            write_both(
                report,
                f"\n[{row['status']}] {row['claim_id']}",
            )

            write_both(
                report,
                str(
                    row[
                        "manuscript_safe_statement"
                    ]
                ),
            )

            write_both(
                report,
                "Guardrail: "
                + str(
                    row[
                        "guardrail"
                    ]
                ),
            )

        # ----------------------------------------------------------
        # Automated consistency qualification.
        # ----------------------------------------------------------

        all_gdis_localization_supported = bool(
            final_gdis[
                "gdis_localization_significant_fdr"
            ].all()
        )

        paired_terminal_autocorr = paired_out[
            (
                paired_out[
                    "baseline_metric"
                ] == "mean_lag1_autocorrelation"
            )
            & (
                paired_out[
                    "scope"
                ].isin(
                    [
                        "cm_extension",
                        "cf_extension",
                    ]
                )
            )
        ]

        terminal_autocorr_supported = bool(
            len(
                paired_terminal_autocorr
            ) == 2
            and paired_terminal_autocorr[
                "paired_superiority_supported"
            ].all()
        )

        no_general_variance_entropy_step_superiority = bool(
            not paired_out[
                paired_out[
                    "baseline_metric"
                ].isin(
                    [
                        "total_variance",
                        "gaussian_differential_entropy",
                        "mean_step_distance",
                    ]
                )
            ][
                "paired_superiority_supported"
            ].any()
        )

        internal_gdis_better = bool(
            (
                internal_out[
                    "p_wilcoxon_gdis_better"
                ]
                < ALPHA
            ).all()
        )

        expected_sensitivity_rows = (
            len(
                SCOPE_ORDER
            )
            * 3
        )

        sensitivity_complete = bool(
            len(
                keep_sensitivity
            )
            == expected_sensitivity_rows
        )

        checks = [
            (
                "All three primary GDIS transition-localization tests "
                "remain significant after FDR",
                all_gdis_localization_supported,
            ),
            (
                "Paired GDIS superiority over lag-1 autocorrelation is "
                "supported for both terminal transitions",
                terminal_autocorr_supported,
            ),
            (
                "No universal GDIS superiority over variance, entropy, "
                "or step distance is asserted",
                no_general_variance_entropy_step_superiority,
            ),
            (
                "Full GDIS is significantly better localized than "
                "transition energy in all three scopes",
                internal_gdis_better,
            ),
            (
                "Frozen 30-PC and both window-size sensitivity summaries "
                "are present for all scopes",
                sensitivity_complete,
            ),
        ]

        write_both(
            report,
            "\n6. EVIDENCE-FREEZE QUALIFICATION",
        )

        write_both(
            report,
            "-" * 106,
        )

        passed = 0

        for label, status in checks:

            if status:
                passed += 1

            write_both(
                report,
                f"{'[PASS]' if status else '[REVIEW]'} "
                f"{label}",
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
                "GSE175634 external validation is complete and frozen.",
            )

        else:

            write_both(
                report,
                "\nFINAL STATUS: REVIEW REQUIRED",
            )

        write_both(
            report,
            "\nFINAL EXTERNAL-VALIDATION INTERPRETATION:",
        )

        write_both(
            report,
            "GDIS significantly localized independently defined "
            "transition regions across the shared cardiac developmental "
            "transition and both terminal extensions. The effect was "
            "robust to pre-specified window and state-space sensitivity "
            "analyses. However, conventional variance, Gaussian entropy, "
            "and pseudotemporal step distance were also strong transition "
            "markers and were equally competitive or better in several "
            "settings. GDIS showed its clearest comparative advantage "
            "over lag-1 pseudotemporal autocorrelation in the terminal "
            "transitions. These results support GDIS as a complementary "
            "integrative dynamical-instability measure rather than a "
            "universally superior early-warning statistic.",
        )

        write_both(
            report,
            "\nNO PROSPECTIVE BIOLOGICAL ONSET-PREDICTION CLAIM IS MADE.",
        )

        write_both(
            report,
            f"\nReport: {REPORT_FILE}",
        )

        write_both(
            report,
            f"Tables: {TABLE_DIR}",
        )

    print("\n" + "=" * 106)
    print("p23_GSE175634_external_validation_evidence_freeze.py completed.")
    print("=" * 106)
    print(f"Report : {REPORT_FILE}")
    print(f"Tables : {TABLE_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

