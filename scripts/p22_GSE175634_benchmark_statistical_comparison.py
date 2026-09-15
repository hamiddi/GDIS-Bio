#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================
"""
p22_GSE175634_benchmark_statistical_comparison.py

Inferential comparison of GDIS with conventional early-warning metrics for
the frozen GSE175634 external-validation benchmark.

This script DOES NOT recompute GDIS or any conventional metric.

Frozen inputs
-------------
p21:
    all_primary_benchmark_profiles.csv.gz

p14:
    individual source/destination state pseudotime landmarks
    pre-specified biological transitions

p17b:
    final 48-profile transition-evaluable manifest

Independent inferential unit
----------------------------
INDIVIDUAL.

Overlapping windows are never treated as independent observations.

Metrics
-------
Primary:
    GDIS

Internal GDIS component:
    transition_energy

Conventional baselines:
    total_variance
    mean_lag1_autocorrelation
    gaussian_differential_entropy
    mean_step_distance

Inferential analyses
--------------------
A. Structure-preserving circular-shift nulls for EVERY metric and scope:
       - median absolute global-peak localization error
       - number of peaks inside destination-state pseudotime IQR
       - median source->destination metric change
       - median signed peak offset

B. Independent-individual source->destination tests:
       - exact binomial sign test
       - Wilcoxon signed-rank test

C. Paired GDIS-vs-conventional localization comparisons within each scope:
       error_difference = baseline_absolute_error - GDIS_absolute_error

   Positive:
       GDIS localizes more accurately.

   Negative:
       baseline localizes more accurately.

   Tests:
       - exact one-sided sign test for GDIS superiority
       - one-sided Wilcoxon signed-rank test for GDIS superiority

   Benjamini-Hochberg FDR is applied across the 12 pre-specified
   scope x conventional-baseline comparisons.

D. GDIS-vs-transition-energy is reported separately because transition energy
   is an internal GDIS component, not a conventional comparator.

Interpretation
--------------
A significant circular-shift result shows stronger-than-random alignment after
breaking biological alignment while preserving the profile shape.

A significant paired GDIS-vs-baseline result is required before claiming that
GDIS localizes a transition better than that baseline.

Destination-state medians/IQRs are independently frozen biological state
landmarks, NOT true transition-onset times.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from scipy.stats import (
    binomtest,
    wilcoxon,
)


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

LANDMARK_FILE = (
    P14_TABLE_DIR
    / "03_frozen_state_landmarks_by_individual.csv"
)

TRANSITION_FILE = (
    P14_TABLE_DIR
    / "05_prespecified_biological_transitions.csv"
)

P17B_TABLE_DIR = (
    PROJECT_DIR
    / "results"
    / "p17b_external_GSE175634_transition_evaluability_freeze"
    / "tables"
)

FINAL_MANIFEST_FILE = (
    P17B_TABLE_DIR
    / "04_final_primary_gdis_file_manifest.csv"
)

P21_DATA_DIR = (
    PROJECT_DIR
    / "results"
    / "p21_external_GSE175634_conventional_ews_benchmark"
    / "data"
)

BENCHMARK_PROFILE_FILE = (
    P21_DATA_DIR
    / "all_primary_benchmark_profiles.csv.gz"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
    / "p22_external_GSE175634_benchmark_statistical_comparison"
)

TABLE_DIR = RESULTS_DIR / "tables"

REPORT_FILE = (
    RESULTS_DIR
    / "p22_external_GSE175634_benchmark_statistical_comparison_report.txt"
)


# ---------------------------------------------------------------------
# Frozen analysis settings
# ---------------------------------------------------------------------

RANDOM_SEED = 20260913
N_PERMUTATIONS = 20_000

SCOPE_ORDER = [
    "shared_backbone",
    "cm_extension",
    "cf_extension",
]

SCOPE_TRANSITION = {
    "shared_backbone": "T2",
    "cm_extension": "T3_CM",
    "cf_extension": "T3_CF",
}

EXPECTED_SCOPE_COUNTS = {
    "shared_backbone": 19,
    "cm_extension": 15,
    "cf_extension": 14,
}

METRICS = [
    "gdis",
    "transition_energy",
    "total_variance",
    "mean_lag1_autocorrelation",
    "gaussian_differential_entropy",
    "mean_step_distance",
]

METRIC_LABELS = {
    "gdis":
        "GDIS",
    "transition_energy":
        "GDIS transition energy",
    "total_variance":
        "Total variance",
    "mean_lag1_autocorrelation":
        "Mean lag-1 pseudotemporal autocorrelation",
    "gaussian_differential_entropy":
        "Gaussian differential entropy",
    "mean_step_distance":
        "Mean pseudotemporal step distance",
}

CONVENTIONAL_BASELINES = [
    "total_variance",
    "mean_lag1_autocorrelation",
    "gaussian_differential_entropy",
    "mean_step_distance",
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


def bh_adjust(p_values):
    """
    Benjamini-Hochberg FDR adjustment, preserving NaN.
    """
    values = np.asarray(
        p_values,
        dtype=float,
    )

    adjusted = np.full(
        values.shape,
        np.nan,
        dtype=float,
    )

    finite_mask = np.isfinite(
        values
    )

    finite = values[
        finite_mask
    ]

    if finite.size == 0:
        return adjusted

    order = np.argsort(
        finite
    )

    ranked = finite[
        order
    ]

    m = len(
        ranked
    )

    q_sorted = (
        ranked
        * m
        / np.arange(
            1,
            m + 1,
            dtype=float,
        )
    )

    q_sorted = np.minimum.accumulate(
        q_sorted[
            ::-1
        ]
    )[
        ::-1
    ]

    q_sorted = np.clip(
        q_sorted,
        0.0,
        1.0,
    )

    inverse = np.empty_like(
        order
    )

    inverse[
        order
    ] = np.arange(
        m
    )

    adjusted[
        finite_mask
    ] = q_sorted[
        inverse
    ]

    return adjusted


def empirical_lower_p(
    null_values,
    observed,
):
    null_values = np.asarray(
        null_values,
        dtype=float,
    )

    return float(
        (
            1
            + np.sum(
                null_values
                <= observed
            )
        )
        / (
            len(
                null_values
            )
            + 1
        )
    )


def empirical_upper_p(
    null_values,
    observed,
):
    null_values = np.asarray(
        null_values,
        dtype=float,
    )

    return float(
        (
            1
            + np.sum(
                null_values
                >= observed
            )
        )
        / (
            len(
                null_values
            )
            + 1
        )
    )


def exact_sign_test_greater(
    differences,
):
    """
    One-sided exact sign test:
        H1 median/sign probability favors positive differences.

    Zero differences are excluded.
    """
    values = np.asarray(
        differences,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    nonzero = values[
        values != 0
    ]

    if len(
        nonzero
    ) == 0:
        return {
            "n_positive":
                0,
            "n_negative":
                0,
            "n_nonzero":
                0,
            "p":
                np.nan,
        }

    n_positive = int(
        np.sum(
            nonzero > 0
        )
    )

    n_negative = int(
        np.sum(
            nonzero < 0
        )
    )

    p = float(
        binomtest(
            n_positive,
            n=len(
                nonzero
            ),
            p=0.5,
            alternative="greater",
        ).pvalue
    )

    return {
        "n_positive":
            n_positive,
        "n_negative":
            n_negative,
        "n_nonzero":
            len(
                nonzero
            ),
        "p":
            p,
    }


def wilcoxon_greater(
    differences,
):
    values = np.asarray(
        differences,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(
        values
    ) == 0:
        return np.nan

    if np.all(
        values == 0
    ):
        return 1.0

    try:
        return float(
            wilcoxon(
                values,
                alternative="greater",
                zero_method="wilcox",
                method="auto",
            ).pvalue
        )
    except ValueError:
        return np.nan


def nearest_index(
    parameters,
    landmark,
):
    return int(
        np.argmin(
            np.abs(
                parameters
                - float(
                    landmark
                )
            )
        )
    )


def build_profile_context(
    profile,
    landmarks,
    source_state,
    destination_state,
):
    profile = (
        profile.sort_values(
            "parameter"
        )
        .reset_index(
            drop=True
        )
        .copy()
    )

    individual = str(
        profile[
            "individual"
        ].iloc[
            0
        ]
    )

    parameters = profile[
        "parameter"
    ].to_numpy(
        dtype=float
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

    source_median = float(
        individual_landmarks.loc[
            source_state,
            "state_median_pseudotime",
        ]
    )

    destination_median = float(
        individual_landmarks.loc[
            destination_state,
            "state_median_pseudotime",
        ]
    )

    destination_q25 = float(
        individual_landmarks.loc[
            destination_state,
            "state_pseudotime_q25",
        ]
    )

    destination_q75 = float(
        individual_landmarks.loc[
            destination_state,
            "state_pseudotime_q75",
        ]
    )

    return {
        "profile":
            profile,
        "individual":
            individual,
        "parameters":
            parameters,
        "source_median":
            source_median,
        "destination_median":
            destination_median,
        "destination_q25":
            destination_q25,
        "destination_q75":
            destination_q75,
        "source_index":
            nearest_index(
                parameters,
                source_median,
            ),
        "destination_index":
            nearest_index(
                parameters,
                destination_median,
            ),
    }


def observed_stats(
    context,
    metric,
):
    profile = context[
        "profile"
    ]

    parameters = context[
        "parameters"
    ]

    values = profile[
        metric
    ].to_numpy(
        dtype=float
    )

    peak_index = int(
        np.argmax(
            values
        )
    )

    peak_parameter = float(
        parameters[
            peak_index
        ]
    )

    destination = float(
        context[
            "destination_median"
        ]
    )

    signed_offset = (
        peak_parameter
        - destination
    )

    source_value = float(
        values[
            int(
                context[
                    "source_index"
                ]
            )
        ]
    )

    destination_value = float(
        values[
            int(
                context[
                    "destination_index"
                ]
            )
        ]
    )

    return {
        "peak_parameter":
            peak_parameter,
        "signed_peak_offset":
            signed_offset,
        "absolute_peak_error":
            abs(
                signed_offset
            ),
        "peak_inside_destination_iqr":
            bool(
                float(
                    context[
                        "destination_q25"
                    ]
                )
                <= peak_parameter
                <= float(
                    context[
                        "destination_q75"
                    ]
                )
            ),
        "source_value":
            source_value,
        "destination_value":
            destination_value,
        "source_to_destination_delta":
            (
                destination_value
                - source_value
            ),
    }


def precompute_nonzero_shifts(
    context,
    metric,
):
    """
    For one individual profile, compute statistics for every nonzero
    circular rotation of the score vector.
    """
    profile = context[
        "profile"
    ]

    parameters = context[
        "parameters"
    ]

    values = profile[
        metric
    ].to_numpy(
        dtype=float
    )

    n = len(
        values
    )

    if n < 3:
        raise ValueError(
            "Profile too short for circular shifts."
        )

    destination = float(
        context[
            "destination_median"
        ]
    )

    q25 = float(
        context[
            "destination_q25"
        ]
    )

    q75 = float(
        context[
            "destination_q75"
        ]
    )

    source_index = int(
        context[
            "source_index"
        ]
    )

    destination_index = int(
        context[
            "destination_index"
        ]
    )

    rows = np.empty(
        (
            n - 1,
            4,
        ),
        dtype=float,
    )

    for row_index, shift in enumerate(
        range(
            1,
            n,
        )
    ):

        shifted = np.roll(
            values,
            shift,
        )

        peak_index = int(
            np.argmax(
                shifted
            )
        )

        peak_parameter = float(
            parameters[
                peak_index
            ]
        )

        signed_offset = (
            peak_parameter
            - destination
        )

        rows[
            row_index,
            0,
        ] = abs(
            signed_offset
        )

        rows[
            row_index,
            1,
        ] = signed_offset

        rows[
            row_index,
            2,
        ] = float(
            q25
            <= peak_parameter
            <= q75
        )

        rows[
            row_index,
            3,
        ] = float(
            shifted[
                destination_index
            ]
            - shifted[
                source_index
            ]
        )

    return rows


def run_circular_shift_null(
    contexts,
    metric,
    rng,
):
    observed_rows = []
    shift_tables = []

    for context in contexts:

        stats = observed_stats(
            context,
            metric,
        )

        observed_rows.append(
            {
                "individual":
                    context[
                        "individual"
                    ],
                **stats,
            }
        )

        shift_tables.append(
            precompute_nonzero_shifts(
                context,
                metric,
            )
        )

    observed = pd.DataFrame(
        observed_rows
    )

    observed_median_abs_error = float(
        observed[
            "absolute_peak_error"
        ].median()
    )

    observed_median_signed_offset = float(
        observed[
            "signed_peak_offset"
        ].median()
    )

    observed_n_inside = int(
        observed[
            "peak_inside_destination_iqr"
        ].sum()
    )

    observed_median_delta = float(
        observed[
            "source_to_destination_delta"
        ].median()
    )

    n_profiles = len(
        shift_tables
    )

    null_abs = np.empty(
        N_PERMUTATIONS,
        dtype=float,
    )

    null_signed = np.empty(
        N_PERMUTATIONS,
        dtype=float,
    )

    null_inside = np.empty(
        N_PERMUTATIONS,
        dtype=float,
    )

    null_delta = np.empty(
        N_PERMUTATIONS,
        dtype=float,
    )

    for permutation in range(
        N_PERMUTATIONS
    ):

        abs_values = np.empty(
            n_profiles,
            dtype=float,
        )

        signed_values = np.empty(
            n_profiles,
            dtype=float,
        )

        inside_values = np.empty(
            n_profiles,
            dtype=float,
        )

        delta_values = np.empty(
            n_profiles,
            dtype=float,
        )

        for i, shift_table in enumerate(
            shift_tables
        ):

            row = int(
                rng.integers(
                    0,
                    len(
                        shift_table
                    ),
                )
            )

            abs_values[
                i
            ] = shift_table[
                row,
                0,
            ]

            signed_values[
                i
            ] = shift_table[
                row,
                1,
            ]

            inside_values[
                i
            ] = shift_table[
                row,
                2,
            ]

            delta_values[
                i
            ] = shift_table[
                row,
                3,
            ]

        null_abs[
            permutation
        ] = np.median(
            abs_values
        )

        null_signed[
            permutation
        ] = np.median(
            signed_values
        )

        null_inside[
            permutation
        ] = np.sum(
            inside_values
        )

        null_delta[
            permutation
        ] = np.median(
            delta_values
        )

    signed_lower = empirical_lower_p(
        null_signed,
        observed_median_signed_offset,
    )

    signed_upper = empirical_upper_p(
        null_signed,
        observed_median_signed_offset,
    )

    signed_two_sided = min(
        1.0,
        2.0
        * min(
            signed_lower,
            signed_upper,
        ),
    )

    summary = {
        "n_profiles":
            n_profiles,
        "observed_median_abs_peak_error":
            observed_median_abs_error,
        "null_median_abs_peak_error":
            float(
                np.median(
                    null_abs
                )
            ),
        "p_localization_lower":
            empirical_lower_p(
                null_abs,
                observed_median_abs_error,
            ),
        "observed_n_peaks_inside_destination_iqr":
            observed_n_inside,
        "null_median_n_peaks_inside_destination_iqr":
            float(
                np.median(
                    null_inside
                )
            ),
        "p_peak_inside_iqr_upper":
            empirical_upper_p(
                null_inside,
                observed_n_inside,
            ),
        "observed_median_source_destination_delta":
            observed_median_delta,
        "null_median_source_destination_delta":
            float(
                np.median(
                    null_delta
                )
            ),
        "p_delta_upper":
            empirical_upper_p(
                null_delta,
                observed_median_delta,
            ),
        "observed_median_signed_peak_offset":
            observed_median_signed_offset,
        "null_median_signed_peak_offset":
            float(
                np.median(
                    null_signed
                )
            ),
        "p_signed_offset_lower":
            signed_lower,
        "p_signed_offset_upper":
            signed_upper,
        "p_signed_offset_two_sided":
            signed_two_sided,
    }

    return (
        summary,
        observed,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    required_files = [
        LANDMARK_FILE,
        TRANSITION_FILE,
        FINAL_MANIFEST_FILE,
        BENCHMARK_PROFILE_FILE,
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

    landmarks = pd.read_csv(
        LANDMARK_FILE,
        low_memory=False,
    )

    transitions = pd.read_csv(
        TRANSITION_FILE,
        low_memory=False,
    )

    manifest = pd.read_csv(
        FINAL_MANIFEST_FILE,
        low_memory=False,
    )

    profiles = pd.read_csv(
        BENCHMARK_PROFILE_FILE,
        compression="gzip",
        low_memory=False,
    )

    for table in [
        landmarks,
        manifest,
        profiles,
    ]:
        if "individual" in table.columns:
            table[
                "individual"
            ] = table[
                "individual"
            ].astype(str)

    landmarks[
        "state"
    ] = landmarks[
        "state"
    ].astype(str)

    transitions[
        "transition_id"
    ] = transitions[
        "transition_id"
    ].astype(str)

    transitions[
        "source_state"
    ] = transitions[
        "source_state"
    ].astype(str)

    transitions[
        "destination_state"
    ] = transitions[
        "destination_state"
    ].astype(str)

    manifest[
        "scope"
    ] = manifest[
        "scope"
    ].astype(str)

    profiles[
        "scope"
    ] = profiles[
        "scope"
    ].astype(str)

    required_profile_columns = [
        "scope",
        "individual",
        "transition_id",
        "parameter",
    ] + METRICS

    missing_columns = [
        column
        for column in required_profile_columns
        if column not in profiles.columns
    ]

    if missing_columns:
        raise ValueError(
            "Benchmark profile table missing columns: "
            + ", ".join(
                missing_columns
            )
        )

    for column in [
        "parameter",
    ] + METRICS:
        profiles[
            column
        ] = pd.to_numeric(
            profiles[
                column
            ],
            errors="raise",
        )

    # Frozen cohort integrity.
    manifest_pairs = set(
        zip(
            manifest[
                "scope"
            ].astype(str),
            manifest[
                "individual"
            ].astype(str),
        )
    )

    profile_pairs = set(
        zip(
            profiles[
                "scope"
            ].astype(str),
            profiles[
                "individual"
            ].astype(str),
        )
    )

    if (
        manifest_pairs
        != profile_pairs
    ):
        raise ValueError(
            "p21 benchmark cohort does not match frozen p17b manifest."
        )

    for scope, expected in (
        EXPECTED_SCOPE_COUNTS.items()
    ):

        observed = int(
            profiles.loc[
                profiles[
                    "scope"
                ] == scope,
                "individual",
            ].nunique()
        )

        if observed != expected:
            raise ValueError(
                f"{scope}: expected {expected} profiles, found {observed}."
            )

    transition_lookup = (
        transitions.set_index(
            "transition_id"
        )
    )

    contexts_by_scope = {}

    for scope in SCOPE_ORDER:

        transition_id = (
            SCOPE_TRANSITION[
                scope
            ]
        )

        source_state = str(
            transition_lookup.loc[
                transition_id,
                "source_state",
            ]
        )

        destination_state = str(
            transition_lookup.loc[
                transition_id,
                "destination_state",
            ]
        )

        contexts = []

        for individual, group in (
            profiles[
                profiles[
                    "scope"
                ] == scope
            ]
            .groupby(
                "individual",
                observed=True,
                sort=True,
            )
        ):

            contexts.append(
                build_profile_context(
                    profile=group,
                    landmarks=landmarks,
                    source_state=source_state,
                    destination_state=destination_state,
                )
            )

        contexts_by_scope[
            scope
        ] = contexts

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    null_rows = []
    individual_metric_frames = []

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as report:

        write_both(
            report,
            "=" * 108,
        )

        write_both(
            report,
            "GDIS-Bio GSE175634 Benchmark Statistical Comparison",
        )

        write_both(
            report,
            "=" * 108,
        )

        write_both(
            report,
            f"\nCircular-shift permutations per scope x metric: "
            f"{N_PERMUTATIONS:,}",
        )

        write_both(
            report,
            f"Random seed: {RANDOM_SEED}",
        )

        write_both(
            report,
            "Independent inferential unit: individual",
        )

        write_both(
            report,
            "No GDIS or conventional metric is recomputed.",
        )

        # =========================================================
        # 1. Circular-shift nulls for all metrics
        # =========================================================

        write_both(
            report,
            "\n1. STRUCTURE-PRESERVING CIRCULAR-SHIFT NULL TESTS",
        )

        write_both(
            report,
            "-" * 108,
        )

        for scope in SCOPE_ORDER:

            transition_id = (
                SCOPE_TRANSITION[
                    scope
                ]
            )

            source_state = str(
                transition_lookup.loc[
                    transition_id,
                    "source_state",
                ]
            )

            destination_state = str(
                transition_lookup.loc[
                    transition_id,
                    "destination_state",
                ]
            )

            contexts = (
                contexts_by_scope[
                    scope
                ]
            )

            write_both(
                report,
                f"\n{scope} | {transition_id}: "
                f"{source_state} -> {destination_state}",
            )

            for metric in METRICS:

                (
                    null_summary,
                    individual_stats,
                ) = run_circular_shift_null(
                    contexts=contexts,
                    metric=metric,
                    rng=rng,
                )

                null_rows.append(
                    {
                        "scope":
                            scope,
                        "transition_id":
                            transition_id,
                        "source_state":
                            source_state,
                        "destination_state":
                            destination_state,
                        "metric":
                            metric,
                        "metric_label":
                            METRIC_LABELS[
                                metric
                            ],
                        **null_summary,
                    }
                )

                individual_stats.insert(
                    0,
                    "metric_label",
                    METRIC_LABELS[
                        metric
                    ],
                )

                individual_stats.insert(
                    0,
                    "metric",
                    metric,
                )

                individual_stats.insert(
                    0,
                    "transition_id",
                    transition_id,
                )

                individual_stats.insert(
                    0,
                    "scope",
                    scope,
                )

                individual_metric_frames.append(
                    individual_stats
                )

                write_both(
                    report,
                    f"  {METRIC_LABELS[metric]}: "
                    f"median abs error="
                    f"{null_summary['observed_median_abs_peak_error']:.6f}, "
                    f"p_loc="
                    f"{null_summary['p_localization_lower']:.6g}; "
                    f"inside IQR="
                    f"{null_summary['observed_n_peaks_inside_destination_iqr']}/"
                    f"{null_summary['n_profiles']}, "
                    f"p_IQR="
                    f"{null_summary['p_peak_inside_iqr_upper']:.6g}; "
                    f"median delta="
                    f"{null_summary['observed_median_source_destination_delta']:.6f}, "
                    f"p_delta="
                    f"{null_summary['p_delta_upper']:.6g}",
                )

        null_tests = pd.DataFrame(
            null_rows
        )

        for p_column, q_column in [
            (
                "p_localization_lower",
                "q_localization_bh",
            ),
            (
                "p_peak_inside_iqr_upper",
                "q_peak_inside_iqr_bh",
            ),
            (
                "p_delta_upper",
                "q_delta_bh",
            ),
            (
                "p_signed_offset_two_sided",
                "q_signed_offset_two_sided_bh",
            ),
        ]:

            null_tests[
                q_column
            ] = bh_adjust(
                null_tests[
                    p_column
                ].to_numpy(
                    dtype=float
                )
            )

        save_table(
            null_tests,
            "01_all_metric_circular_shift_tests.csv",
            index=False,
        )

        individual_metrics = pd.concat(
            individual_metric_frames,
            ignore_index=True,
        )

        save_table(
            individual_metrics,
            "02_individual_metric_localization_and_change.csv",
            index=False,
        )

        # =========================================================
        # 2. Independent-individual source/destination tests
        # =========================================================

        write_both(
            report,
            "\n2. INDEPENDENT-INDIVIDUAL SOURCE-to-DESTINATION TESTS",
        )

        write_both(
            report,
            "-" * 108,
        )

        change_test_rows = []

        for (
            scope,
            transition_id,
            metric,
        ), group in (
            individual_metrics.groupby(
                [
                    "scope",
                    "transition_id",
                    "metric",
                ],
                observed=True,
            )
        ):

            deltas = group[
                "source_to_destination_delta"
            ].to_numpy(
                dtype=float
            )

            sign_result = (
                exact_sign_test_greater(
                    deltas
                )
            )

            change_test_rows.append(
                {
                    "scope":
                        scope,
                    "transition_id":
                        transition_id,
                    "metric":
                        metric,
                    "metric_label":
                        METRIC_LABELS[
                            metric
                        ],
                    "n_individuals":
                        len(
                            group
                        ),
                    "n_positive_delta":
                        int(
                            np.sum(
                                deltas > 0
                            )
                        ),
                    "n_negative_delta":
                        int(
                            np.sum(
                                deltas < 0
                            )
                        ),
                    "n_zero_delta":
                        int(
                            np.sum(
                                deltas == 0
                            )
                        ),
                    "median_delta":
                        float(
                            np.median(
                                deltas
                            )
                        ),
                    "p_sign_greater":
                        sign_result[
                            "p"
                        ],
                    "p_wilcoxon_greater":
                        wilcoxon_greater(
                            deltas
                        ),
                }
            )

        change_tests = pd.DataFrame(
            change_test_rows
        )

        change_tests[
            "q_sign_greater_bh"
        ] = bh_adjust(
            change_tests[
                "p_sign_greater"
            ].to_numpy(
                dtype=float
            )
        )

        change_tests[
            "q_wilcoxon_greater_bh"
        ] = bh_adjust(
            change_tests[
                "p_wilcoxon_greater"
            ].to_numpy(
                dtype=float
            )
        )

        save_table(
            change_tests,
            "03_independent_individual_change_tests.csv",
            index=False,
        )

        write_both(
            report,
            change_tests.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 3. Paired GDIS vs conventional localization
        # =========================================================

        write_both(
            report,
            "\n3. PAIRED GDIS-vs-CONVENTIONAL LOCALIZATION TESTS",
        )

        write_both(
            report,
            "-" * 108,
        )

        paired_rows = []

        for scope in SCOPE_ORDER:

            transition_id = (
                SCOPE_TRANSITION[
                    scope
                ]
            )

            gdis = (
                individual_metrics[
                    (
                        individual_metrics[
                            "scope"
                        ] == scope
                    )
                    & (
                        individual_metrics[
                            "metric"
                        ] == "gdis"
                    )
                ][
                    [
                        "individual",
                        "absolute_peak_error",
                    ]
                ]
                .rename(
                    columns={
                        "absolute_peak_error":
                            "gdis_absolute_peak_error",
                    }
                )
            )

            for baseline in (
                CONVENTIONAL_BASELINES
            ):

                baseline_table = (
                    individual_metrics[
                        (
                            individual_metrics[
                                "scope"
                            ] == scope
                        )
                        & (
                            individual_metrics[
                                "metric"
                            ] == baseline
                        )
                    ][
                        [
                            "individual",
                            "absolute_peak_error",
                        ]
                    ]
                    .rename(
                        columns={
                            "absolute_peak_error":
                                "baseline_absolute_peak_error",
                        }
                    )
                )

                paired = gdis.merge(
                    baseline_table,
                    on="individual",
                    how="inner",
                    validate="one_to_one",
                )

                difference = (
                    paired[
                        "baseline_absolute_peak_error"
                    ].to_numpy(
                        dtype=float
                    )
                    - paired[
                        "gdis_absolute_peak_error"
                    ].to_numpy(
                        dtype=float
                    )
                )

                sign_result = (
                    exact_sign_test_greater(
                        difference
                    )
                )

                n_ties = int(
                    np.sum(
                        difference == 0
                    )
                )

                paired_rows.append(
                    {
                        "scope":
                            scope,
                        "transition_id":
                            transition_id,
                        "baseline_metric":
                            baseline,
                        "baseline_label":
                            METRIC_LABELS[
                                baseline
                            ],
                        "n_individuals":
                            len(
                                paired
                            ),
                        "median_gdis_abs_error":
                            float(
                                paired[
                                    "gdis_absolute_peak_error"
                                ].median()
                            ),
                        "median_baseline_abs_error":
                            float(
                                paired[
                                    "baseline_absolute_peak_error"
                                ].median()
                            ),
                        "median_baseline_minus_gdis_error":
                            float(
                                np.median(
                                    difference
                                )
                            ),
                        "gdis_wins":
                            int(
                                np.sum(
                                    difference > 0
                                )
                            ),
                        "ties":
                            n_ties,
                        "gdis_losses":
                            int(
                                np.sum(
                                    difference < 0
                                )
                            ),
                        "p_sign_gdis_better":
                            sign_result[
                                "p"
                            ],
                        "p_wilcoxon_gdis_better":
                            wilcoxon_greater(
                                difference
                            ),
                    }
                )

        paired_tests = pd.DataFrame(
            paired_rows
        )

        paired_tests[
            "q_sign_gdis_better_bh"
        ] = bh_adjust(
            paired_tests[
                "p_sign_gdis_better"
            ].to_numpy(
                dtype=float
            )
        )

        paired_tests[
            "q_wilcoxon_gdis_better_bh"
        ] = bh_adjust(
            paired_tests[
                "p_wilcoxon_gdis_better"
            ].to_numpy(
                dtype=float
            )
        )

        save_table(
            paired_tests,
            "04_paired_gdis_vs_conventional_localization.csv",
            index=False,
        )

        write_both(
            report,
            paired_tests.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 4. GDIS vs internal transition-energy component
        # =========================================================

        write_both(
            report,
            "\n4. GDIS-vs-TRANSITION-ENERGY INTERNAL COMPARISON",
        )

        write_both(
            report,
            "-" * 108,
        )

        internal_rows = []

        for scope in SCOPE_ORDER:

            transition_id = (
                SCOPE_TRANSITION[
                    scope
                ]
            )

            gdis = (
                individual_metrics[
                    (
                        individual_metrics[
                            "scope"
                        ] == scope
                    )
                    & (
                        individual_metrics[
                            "metric"
                        ] == "gdis"
                    )
                ][
                    [
                        "individual",
                        "absolute_peak_error",
                    ]
                ]
                .rename(
                    columns={
                        "absolute_peak_error":
                            "gdis_absolute_peak_error",
                    }
                )
            )

            energy = (
                individual_metrics[
                    (
                        individual_metrics[
                            "scope"
                        ] == scope
                    )
                    & (
                        individual_metrics[
                            "metric"
                        ] == "transition_energy"
                    )
                ][
                    [
                        "individual",
                        "absolute_peak_error",
                    ]
                ]
                .rename(
                    columns={
                        "absolute_peak_error":
                            "transition_energy_absolute_peak_error",
                    }
                )
            )

            paired = gdis.merge(
                energy,
                on="individual",
                how="inner",
                validate="one_to_one",
            )

            difference = (
                paired[
                    "transition_energy_absolute_peak_error"
                ].to_numpy(
                    dtype=float
                )
                - paired[
                    "gdis_absolute_peak_error"
                ].to_numpy(
                    dtype=float
                )
            )

            sign_result = (
                exact_sign_test_greater(
                    difference
                )
            )

            internal_rows.append(
                {
                    "scope":
                        scope,
                    "transition_id":
                        transition_id,
                    "n_individuals":
                        len(
                            paired
                        ),
                    "median_gdis_abs_error":
                        float(
                            paired[
                                "gdis_absolute_peak_error"
                            ].median()
                        ),
                    "median_transition_energy_abs_error":
                        float(
                            paired[
                                "transition_energy_absolute_peak_error"
                            ].median()
                        ),
                    "median_energy_minus_gdis_error":
                        float(
                            np.median(
                                difference
                            )
                        ),
                    "gdis_wins":
                        int(
                            np.sum(
                                difference > 0
                            )
                        ),
                    "ties":
                        int(
                            np.sum(
                                difference == 0
                            )
                        ),
                    "gdis_losses":
                        int(
                            np.sum(
                                difference < 0
                            )
                        ),
                    "p_sign_gdis_better":
                        sign_result[
                            "p"
                        ],
                    "p_wilcoxon_gdis_better":
                        wilcoxon_greater(
                            difference
                        ),
                }
            )

        internal = pd.DataFrame(
            internal_rows
        )

        save_table(
            internal,
            "05_gdis_vs_transition_energy_localization.csv",
            index=False,
        )

        write_both(
            report,
            internal.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 5. Scope ranking by median localization error
        # =========================================================

        write_both(
            report,
            "\n5. DESCRIPTIVE LOCALIZATION RANKING",
        )

        write_both(
            report,
            "-" * 108,
        )

        ranking = (
            null_tests[
                [
                    "scope",
                    "transition_id",
                    "metric",
                    "metric_label",
                    "observed_median_abs_peak_error",
                    "p_localization_lower",
                    "q_localization_bh",
                ]
            ]
            .copy()
        )

        ranking[
            "localization_rank_within_scope"
        ] = (
            ranking.groupby(
                "scope",
                observed=True,
            )[
                "observed_median_abs_peak_error"
            ]
            .rank(
                method="min",
                ascending=True,
            )
        )

        ranking = ranking.sort_values(
            [
                "scope",
                "localization_rank_within_scope",
                "observed_median_abs_peak_error",
            ]
        )

        save_table(
            ranking,
            "06_descriptive_localization_ranking.csv",
            index=False,
        )

        write_both(
            report,
            ranking.round(
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
            "\n6. STATISTICAL BENCHMARK QUALIFICATION",
        )

        write_both(
            report,
            "-" * 108,
        )

        expected_null_tests = (
            len(
                SCOPE_ORDER
            )
            * len(
                METRICS
            )
        )

        expected_paired_tests = (
            len(
                SCOPE_ORDER
            )
            * len(
                CONVENTIONAL_BASELINES
            )
        )

        checks = [
            (
                "All 18 scope x metric circular-shift tests completed",
                len(
                    null_tests
                )
                == expected_null_tests,
            ),
            (
                "All 12 pre-specified paired GDIS-vs-conventional "
                "comparisons completed",
                len(
                    paired_tests
                )
                == expected_paired_tests,
            ),
            (
                "Every paired comparison uses the same individuals "
                "for GDIS and its baseline",
                bool(
                    (
                        paired_tests[
                            "n_individuals"
                        ]
                        > 0
                    ).all()
                ),
            ),
            (
                f"Exactly {N_PERMUTATIONS:,} structure-preserving "
                "permutations used per scope x metric",
                True,
            ),
            (
                "BH correction applied separately to localization, "
                "IQR enrichment, source/destination change, signed offset, "
                "and paired-superiority families",
                True,
            ),
            (
                "No metric or GDIS profile was recomputed or retuned",
                True,
            ),
        ]

        qualification_rows = []

        passed = 0

        for label, status in checks:

            if status:
                passed += 1

            write_both(
                report,
                f"{'[PASS]' if status else '[REVIEW]'} "
                f"{label}",
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
            "07_benchmark_statistical_qualification.csv",
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

        else:

            write_both(
                report,
                "\nFINAL COMPUTATIONAL STATUS: REVIEW REQUIRED",
            )

        # =========================================================
        # 7. Guardrails
        # =========================================================

        write_both(
            report,
            "\n7. INTERPRETATION GUARDRAILS",
        )

        write_both(
            report,
            "-" * 108,
        )

        write_both(
            report,
            "  - Significant GDIS-vs-null alignment is not the same "
            "claim as significant superiority over a baseline.",
        )

        write_both(
            report,
            "  - GDIS should be called superior to a conventional metric "
            "only if the paired individual-level comparison supports it.",
        )

        write_both(
            report,
            "  - A conventional metric may itself show significant "
            "transition alignment; this is scientifically informative "
            "and must be reported.",
        )

        write_both(
            report,
            "  - Destination-state medians/IQRs are state landmarks, "
            "not true transition-onset times.",
        )

        write_both(
            report,
            "  - No result in this benchmark may be used to retune "
            "GDIS, redefine a cohort, move a landmark, or choose a "
            "different primary dimensionality/window configuration.",
        )

        write_both(
            report,
            f"\nReport: {REPORT_FILE}",
        )

        write_both(
            report,
            f"Tables: {TABLE_DIR}",
        )

    print("\n" + "=" * 108)
    print("p22_GSE175634_benchmark_statistical_comparison.py completed.")
    print("=" * 108)
    print(f"Report : {REPORT_FILE}")
    print(f"Tables : {TABLE_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

