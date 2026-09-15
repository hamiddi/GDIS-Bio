#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================
"""
p11_statistical_alignment_null_validation.py

Statistical/null validation of the frozen GDIS-Bio benchmark.

Purpose
-------
p10 showed that GDIS had the smallest descriptive peak-localization error
relative to the pre-frozen median NEUROG3-early landmark. However, p10 was
descriptive.

This script adds statistical alignment tests WITHOUT recomputing GDIS,
changing windows, changing PCA dimensions, or changing biological landmarks.

Null model
----------
For each metric profile within each branch x differentiation group, the
metric values are circularly shifted relative to the fixed pseudotime /
biological metadata.

Why circular shifts?
--------------------
A circular shift preserves:
    - the complete metric-value distribution
    - the smooth/autocorrelated profile shape
    - local ordering among neighboring metric values

while breaking:
    - the alignment between the metric profile and the biological trajectory

This provides a conservative trajectory-alignment null without treating
overlapping windows as independent observations.

Main tests
----------
1. Mean absolute peak-localization error relative to the frozen median
   NEUROG3-early landmark.

2. Number of the four analyses whose global metric peak falls in a
   NEUROG3-early-dominant window.

3. Number of analyses in which the metric increases from the frozen
   prog_nkx61 landmark to the frozen NEUROG3-early landmark.

4. Signed peak position relative to the NEUROG3-early landmark.
   This is especially useful for interpreting metrics such as GDIS
   transition energy that peak BEFORE the median state landmark.

5. Cross-replicate profile correlation null:
   one replicate is circularly shifted relative to the other.

6. Exact paired sign-flip comparison:
   GDIS peak-localization error versus each conventional baseline
   across the four frozen branch x differentiation analyses.

Multiple testing
----------------
Benjamini-Hochberg false-discovery-rate adjustment is reported within each
family of metric-level circular-shift tests.

IMPORTANT
---------
The NEUROG3-early median remains a state landmark, NOT a proven true
transition-onset time.

No result from this script is used to retune GDIS.
"""

from pathlib import Path
import sys
from itertools import product

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

P7_TABLE_DIR = (
    PROJECT_DIR
    / "results"
    / "p7_trajectory_assembly_validation"
    / "tables"
)

P10_TABLE_DIR = (
    PROJECT_DIR
    / "results"
    / "p10_conventional_ews_benchmark"
    / "tables"
)

PROFILE_FILE = (
    P10_TABLE_DIR
    / "01_all_metric_profiles.csv"
)

LANDMARK_FILE = (
    P7_TABLE_DIR
    / "07_primary_state_landmark_window_mapping.csv"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
    / "p11_statistical_alignment_null_validation"
)

TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR = RESULTS_DIR / "figures"

REPORT_FILE = (
    RESULTS_DIR
    / "p11_statistical_alignment_null_validation_report.txt"
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

METRICS = [
    "gdis",
    "transition_energy",
    "total_variance",
    "mean_lag1_autocorrelation",
    "gaussian_differential_entropy",
    "mean_step_distance",
]

BASELINES = [
    "total_variance",
    "mean_lag1_autocorrelation",
    "gaussian_differential_entropy",
    "mean_step_distance",
]

METRIC_LABELS = {
    "gdis": "GDIS",
    "transition_energy": "GDIS transition energy",
    "total_variance": "Total variance",
    "mean_lag1_autocorrelation":
        "Mean lag-1 pseudotemporal autocorrelation",
    "gaussian_differential_entropy":
        "Gaussian differential entropy",
    "mean_step_distance":
        "Mean pseudotemporal step distance",
}

RANDOM_SEED = 20260913
N_PERMUTATIONS = 20000
REPLICATE_GRID_POINTS = 250


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def write_both(report, text=""):
    """Print and write the same text."""
    print(text)
    report.write(text + "\n")
    report.flush()


def save_table(df, filename, index=True):
    """Save dataframe to p11 tables directory."""
    path = TABLE_DIR / filename
    df.to_csv(path, index=index)
    return path


def safe_pearson(x, y):
    """Pearson correlation with finite-value filtering."""
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


def bh_adjust(p_values):
    """
    Benjamini-Hochberg FDR adjustment.

    Returns adjusted p-values in the original order.
    """
    p = np.asarray(
        p_values,
        dtype=float,
    )

    n = len(p)

    order = np.argsort(p)
    ranked = p[order]

    adjusted_ranked = (
        ranked
        * n
        / np.arange(
            1,
            n + 1,
            dtype=float,
        )
    )

    # Enforce monotonicity from largest rank backward.
    adjusted_ranked = np.minimum.accumulate(
        adjusted_ranked[::-1]
    )[::-1]

    adjusted_ranked = np.clip(
        adjusted_ranked,
        0.0,
        1.0,
    )

    adjusted = np.empty(
        n,
        dtype=float,
    )

    adjusted[order] = adjusted_ranked

    return adjusted


def empirical_p_lower(
    null_values,
    observed,
):
    """
    Lower-tail empirical p-value:
    P(null <= observed).

    Includes +1 correction.
    """
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
            len(null_values)
            + 1
        )
    )


def empirical_p_upper(
    null_values,
    observed,
):
    """
    Upper-tail empirical p-value:
    P(null >= observed).

    Includes +1 correction.
    """
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
            len(null_values)
            + 1
        )
    )


def get_landmark(
    landmarks,
    branch,
    differentiation,
    state,
):
    subset = landmarks[
        (
            landmarks[
                "branch"
            ] == branch
        )
        & (
            landmarks[
                "differentiation"
            ] == differentiation
        )
        & (
            landmarks[
                "state"
            ] == state
        )
    ]

    if len(subset) != 1:
        raise ValueError(
            f"Expected one landmark for "
            f"branch={branch}, diff={differentiation}, "
            f"state={state}; found {len(subset)}."
        )

    return float(
        subset.iloc[0][
            "state_median_pseudotime"
        ]
    )


def nearest_index(
    parameters,
    target,
):
    """Index of parameter closest to target."""
    parameters = np.asarray(
        parameters,
        dtype=float,
    )

    return int(
        np.argmin(
            np.abs(
                parameters
                - float(target)
            )
        )
    )


def prepare_groups(
    profiles,
    landmarks,
):
    """
    Prepare fixed arrays and frozen landmark indices for each group.
    """
    prepared = {}

    for branch, differentiation in GROUPS:

        df = (
            profiles[
                (
                    profiles[
                        "branch"
                    ] == branch
                )
                & (
                    profiles[
                        "differentiation"
                    ] == differentiation
                )
            ]
            .sort_values(
                "parameter"
            )
            .reset_index(
                drop=True
            )
            .copy()
        )

        if df.empty:
            raise ValueError(
                f"No profile rows for branch={branch}, "
                f"diff={differentiation}."
            )

        parameters = df[
            "parameter"
        ].to_numpy(
            dtype=float
        )

        early_landmark = get_landmark(
            landmarks,
            branch,
            differentiation,
            "neurog3_early",
        )

        progenitor_landmark = get_landmark(
            landmarks,
            branch,
            differentiation,
            "prog_nkx61",
        )

        early_index = nearest_index(
            parameters,
            early_landmark,
        )

        progenitor_index = nearest_index(
            parameters,
            progenitor_landmark,
        )

        prepared[
            (
                branch,
                differentiation,
            )
        ] = {
            "df": df,
            "parameters":
                parameters,
            "early_landmark":
                early_landmark,
            "progenitor_landmark":
                progenitor_landmark,
            "early_index":
                early_index,
            "progenitor_index":
                progenitor_index,
            "dominant_state":
                df[
                    "dominant_state"
                ].astype(str).to_numpy(),
        }

    return prepared


def circular_shift_statistics(
    prepared,
    metric,
    rng,
    n_permutations,
):
    """
    Circular-shift null statistics for one metric.

    Returns
    -------
    observed : dict
    null : dict of ndarray
    """
    group_observed = []

    for branch, differentiation in GROUPS:

        item = prepared[
            (
                branch,
                differentiation,
            )
        ]

        df = item["df"]

        values = df[
            metric
        ].to_numpy(
            dtype=float
        )

        parameters = item[
            "parameters"
        ]

        peak_index = int(
            np.nanargmax(
                values
            )
        )

        peak_parameter = float(
            parameters[
                peak_index
            ]
        )

        localization_error = abs(
            peak_parameter
            - item[
                "early_landmark"
            ]
        )

        signed_offset = (
            peak_parameter
            - item[
                "early_landmark"
            ]
        )

        peak_is_early_state = (
            item[
                "dominant_state"
            ][peak_index]
            == "neurog3_early"
        )

        early_minus_prog = (
            float(
                values[
                    item[
                        "early_index"
                    ]
                ]
            )
            - float(
                values[
                    item[
                        "progenitor_index"
                    ]
                ]
            )
        )

        group_observed.append(
            {
                "localization_error":
                    localization_error,
                "signed_offset":
                    signed_offset,
                "peak_is_early_state":
                    peak_is_early_state,
                "early_minus_prog":
                    early_minus_prog,
            }
        )

    observed = {
        "mean_localization_error":
            float(
                np.mean(
                    [
                        x[
                            "localization_error"
                        ]
                        for x
                        in group_observed
                    ]
                )
            ),
        "mean_signed_peak_offset":
            float(
                np.mean(
                    [
                        x[
                            "signed_offset"
                        ]
                        for x
                        in group_observed
                    ]
                )
            ),
        "n_peak_in_early_state":
            int(
                np.sum(
                    [
                        x[
                            "peak_is_early_state"
                        ]
                        for x
                        in group_observed
                    ]
                )
            ),
        "n_positive_early_rise":
            int(
                np.sum(
                    [
                        x[
                            "early_minus_prog"
                        ]
                        > 0
                        for x
                        in group_observed
                    ]
                )
            ),
        "mean_early_minus_prog":
            float(
                np.mean(
                    [
                        x[
                            "early_minus_prog"
                        ]
                        for x
                        in group_observed
                    ]
                )
            ),
    }

    null_mean_error = np.empty(
        n_permutations,
        dtype=float,
    )

    null_mean_signed_offset = np.empty(
        n_permutations,
        dtype=float,
    )

    null_n_early_peak = np.empty(
        n_permutations,
        dtype=np.int16,
    )

    null_n_positive_rise = np.empty(
        n_permutations,
        dtype=np.int16,
    )

    null_mean_early_minus_prog = np.empty(
        n_permutations,
        dtype=float,
    )

    # Cache arrays because profiles are tiny and repeatedly reused.
    cache = {}

    for branch, differentiation in GROUPS:

        item = prepared[
            (
                branch,
                differentiation,
            )
        ]

        values = item[
            "df"
        ][
            metric
        ].to_numpy(
            dtype=float
        )

        cache[
            (
                branch,
                differentiation,
            )
        ] = {
            "values":
                values,
            "n":
                len(values),
            "parameters":
                item[
                    "parameters"
                ],
            "early_landmark":
                item[
                    "early_landmark"
                ],
            "dominant_state":
                item[
                    "dominant_state"
                ],
            "early_index":
                item[
                    "early_index"
                ],
            "progenitor_index":
                item[
                    "progenitor_index"
                ],
        }

    for perm in range(
        n_permutations
    ):

        errors = []
        signed_offsets = []
        early_peak_count = 0
        positive_rise_count = 0
        rises = []

        for branch, differentiation in GROUPS:

            item = cache[
                (
                    branch,
                    differentiation,
                )
            ]

            n = item["n"]

            # Exclude zero shift so every null iteration breaks
            # the observed alignment in every group.
            shift = int(
                rng.integers(
                    1,
                    n,
                )
            )

            shifted = np.roll(
                item[
                    "values"
                ],
                shift,
            )

            peak_index = int(
                np.nanargmax(
                    shifted
                )
            )

            peak_parameter = float(
                item[
                    "parameters"
                ][peak_index]
            )

            error = abs(
                peak_parameter
                - item[
                    "early_landmark"
                ]
            )

            signed = (
                peak_parameter
                - item[
                    "early_landmark"
                ]
            )

            early_peak = (
                item[
                    "dominant_state"
                ][peak_index]
                == "neurog3_early"
            )

            rise = (
                float(
                    shifted[
                        item[
                            "early_index"
                        ]
                    ]
                )
                - float(
                    shifted[
                        item[
                            "progenitor_index"
                        ]
                    ]
                )
            )

            errors.append(
                error
            )

            signed_offsets.append(
                signed
            )

            early_peak_count += int(
                early_peak
            )

            positive_rise_count += int(
                rise > 0
            )

            rises.append(
                rise
            )

        null_mean_error[
            perm
        ] = float(
            np.mean(errors)
        )

        null_mean_signed_offset[
            perm
        ] = float(
            np.mean(
                signed_offsets
            )
        )

        null_n_early_peak[
            perm
        ] = early_peak_count

        null_n_positive_rise[
            perm
        ] = positive_rise_count

        null_mean_early_minus_prog[
            perm
        ] = float(
            np.mean(rises)
        )

    null = {
        "mean_localization_error":
            null_mean_error,
        "mean_signed_peak_offset":
            null_mean_signed_offset,
        "n_peak_in_early_state":
            null_n_early_peak,
        "n_positive_early_rise":
            null_n_positive_rise,
        "mean_early_minus_prog":
            null_mean_early_minus_prog,
    }

    return observed, null


def replicate_correlation_null(
    prepared,
    metric,
    branch,
    rng,
    n_permutations,
):
    """
    Circularly shift Differentiation 2 metric values relative to its
    pseudotime positions and test the observed cross-replicate Pearson
    correlation.
    """
    a = prepared[
        (
            branch,
            1,
        )
    ]["df"]

    b = prepared[
        (
            branch,
            2,
        )
    ]["df"]

    lower = max(
        float(
            a[
                "parameter"
            ].min()
        ),
        float(
            b[
                "parameter"
            ].min()
        ),
    )

    upper = min(
        float(
            a[
                "parameter"
            ].max()
        ),
        float(
            b[
                "parameter"
            ].max()
        ),
    )

    grid = np.linspace(
        lower,
        upper,
        REPLICATE_GRID_POINTS,
    )

    a_values = np.interp(
        grid,
        a[
            "parameter"
        ],
        a[
            metric
        ],
    )

    b_values = b[
        metric
    ].to_numpy(
        dtype=float
    )

    b_parameters = b[
        "parameter"
    ].to_numpy(
        dtype=float
    )

    observed_b = np.interp(
        grid,
        b_parameters,
        b_values,
    )

    observed_corr = safe_pearson(
        a_values,
        observed_b,
    )

    null = np.empty(
        n_permutations,
        dtype=float,
    )

    n_b = len(
        b_values
    )

    for perm in range(
        n_permutations
    ):

        shift = int(
            rng.integers(
                1,
                n_b,
            )
        )

        shifted = np.roll(
            b_values,
            shift,
        )

        shifted_interp = np.interp(
            grid,
            b_parameters,
            shifted,
        )

        null[
            perm
        ] = safe_pearson(
            a_values,
            shifted_interp,
        )

    p_value = empirical_p_upper(
        null,
        observed_corr,
    )

    return {
        "observed_pearson":
            observed_corr,
        "null_mean":
            float(
                np.mean(null)
            ),
        "null_q025":
            float(
                np.quantile(
                    null,
                    0.025,
                )
            ),
        "null_q975":
            float(
                np.quantile(
                    null,
                    0.975,
                )
            ),
        "p_upper":
            p_value,
    }


def exact_sign_flip_test(
    differences,
):
    """
    Exact one-sided sign-flip test.

    differences = baseline_error - gdis_error

    Positive mean difference favors GDIS (smaller error).
    """
    differences = np.asarray(
        differences,
        dtype=float,
    )

    observed = float(
        np.mean(
            differences
        )
    )

    null_means = []

    for signs in product(
        [-1.0, 1.0],
        repeat=len(
            differences
        ),
    ):

        signed = (
            differences
            * np.asarray(
                signs,
                dtype=float,
            )
        )

        null_means.append(
            float(
                np.mean(
                    signed
                )
            )
        )

    null_means = np.asarray(
        null_means,
        dtype=float,
    )

    # One-sided: baseline error minus GDIS error > 0.
    p_value = float(
        np.mean(
            null_means
            >= observed
            - 1e-15
        )
    )

    return {
        "observed_mean_difference":
            observed,
        "p_one_sided":
            p_value,
        "n_exact_permutations":
            len(
                null_means
            ),
    }


def plot_null_localization(
    null_summary,
    output_path,
):
    """
    Plot observed localization error with null 95% interval.
    """
    df = null_summary.copy()

    x = np.arange(
        len(df)
    )

    observed = df[
        "observed_mean_localization_error"
    ].to_numpy(
        dtype=float
    )

    lower = df[
        "null_localization_q025"
    ].to_numpy(
        dtype=float
    )

    upper = df[
        "null_localization_q975"
    ].to_numpy(
        dtype=float
    )

    yerr = np.vstack(
        [
            observed
            - lower,
            upper
            - observed,
        ]
    )

    # Error bars may extend asymmetrically around observed even when
    # observed lies outside the central null interval. Matplotlib
    # requires nonnegative magnitudes.
    yerr = np.abs(
        yerr
    )

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.errorbar(
        x,
        observed,
        yerr=yerr,
        fmt="o",
        capsize=4,
    )

    ax.set_xticks(
        x,
        df[
            "metric_label"
        ],
        rotation=35,
        ha="right",
    )

    ax.set_ylabel(
        "Mean absolute peak-localization error"
    )

    ax.set_title(
        "Observed localization versus circular-shift null"
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

    if not PROFILE_FILE.exists():
        raise FileNotFoundError(
            f"Missing p10 metric profiles: "
            f"{PROFILE_FILE}"
        )

    if not LANDMARK_FILE.exists():
        raise FileNotFoundError(
            f"Missing frozen landmark table: "
            f"{LANDMARK_FILE}"
        )

    profiles = pd.read_csv(
        PROFILE_FILE
    )

    landmarks = pd.read_csv(
        LANDMARK_FILE
    )

    required_profile_columns = [
        "branch",
        "differentiation",
        "parameter",
        "dominant_state",
    ] + METRICS

    missing = [
        c
        for c in required_profile_columns
        if c not in profiles.columns
    ]

    if missing:
        raise ValueError(
            "Profile file missing columns: "
            + ", ".join(missing)
        )

    prepared = prepare_groups(
        profiles,
        landmarks,
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

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
            "GDIS-Bio Statistical Alignment and Null Validation",
        )

        write_both(
            report,
            "=" * 96,
        )

        write_both(
            report,
            f"\nRandom seed: "
            f"{RANDOM_SEED}",
        )

        write_both(
            report,
            f"Circular-shift permutations per metric: "
            f"{N_PERMUTATIONS:,}",
        )

        write_both(
            report,
            "Null model preserves each metric profile's values and "
            "neighboring shape while breaking alignment to biological "
            "pseudotime landmarks.",
        )

        write_both(
            report,
            "Overlapping windows are NOT treated as independent "
            "samples in the circular-shift tests.",
        )

        # =========================================================
        # 1. Circular-shift biological alignment nulls
        # =========================================================

        write_both(
            report,
            "\n1. CIRCULAR-SHIFT BIOLOGICAL ALIGNMENT NULL TESTS",
        )

        write_both(
            report,
            "-" * 96,
        )

        alignment_rows = []

        for metric in METRICS:

            observed, null = (
                circular_shift_statistics(
                    prepared=prepared,
                    metric=metric,
                    rng=rng,
                    n_permutations=
                        N_PERMUTATIONS,
                )
            )

            row = {
                "metric":
                    metric,
                "metric_label":
                    METRIC_LABELS[
                        metric
                    ],
                "observed_mean_localization_error":
                    observed[
                        "mean_localization_error"
                    ],
                "null_localization_mean":
                    float(
                        np.mean(
                            null[
                                "mean_localization_error"
                            ]
                        )
                    ),
                "null_localization_q025":
                    float(
                        np.quantile(
                            null[
                                "mean_localization_error"
                            ],
                            0.025,
                        )
                    ),
                "null_localization_q975":
                    float(
                        np.quantile(
                            null[
                                "mean_localization_error"
                            ],
                            0.975,
                        )
                    ),
                "p_localization_lower":
                    empirical_p_lower(
                        null[
                            "mean_localization_error"
                        ],
                        observed[
                            "mean_localization_error"
                        ],
                    ),
                "observed_n_peaks_in_neurog3_early":
                    observed[
                        "n_peak_in_early_state"
                    ],
                "null_mean_n_peaks_in_neurog3_early":
                    float(
                        np.mean(
                            null[
                                "n_peak_in_early_state"
                            ]
                        )
                    ),
                "p_early_peak_enrichment_upper":
                    empirical_p_upper(
                        null[
                            "n_peak_in_early_state"
                        ],
                        observed[
                            "n_peak_in_early_state"
                        ],
                    ),
                "observed_n_positive_progenitor_to_early_rise":
                    observed[
                        "n_positive_early_rise"
                    ],
                "null_mean_n_positive_rise":
                    float(
                        np.mean(
                            null[
                                "n_positive_early_rise"
                            ]
                        )
                    ),
                "p_positive_rise_count_upper":
                    empirical_p_upper(
                        null[
                            "n_positive_early_rise"
                        ],
                        observed[
                            "n_positive_early_rise"
                        ],
                    ),
                "observed_mean_progenitor_to_early_change":
                    observed[
                        "mean_early_minus_prog"
                    ],
                "null_mean_progenitor_to_early_change":
                    float(
                        np.mean(
                            null[
                                "mean_early_minus_prog"
                            ]
                        )
                    ),
                "p_mean_rise_upper":
                    empirical_p_upper(
                        null[
                            "mean_early_minus_prog"
                        ],
                        observed[
                            "mean_early_minus_prog"
                        ],
                    ),
                "observed_mean_signed_peak_offset":
                    observed[
                        "mean_signed_peak_offset"
                    ],
                "null_mean_signed_peak_offset":
                    float(
                        np.mean(
                            null[
                                "mean_signed_peak_offset"
                            ]
                        )
                    ),
                "p_signed_offset_lower":
                    empirical_p_lower(
                        null[
                            "mean_signed_peak_offset"
                        ],
                        observed[
                            "mean_signed_peak_offset"
                        ],
                    ),
            }

            alignment_rows.append(
                row
            )

            write_both(
                report,
                f"{METRIC_LABELS[metric]}: "
                f"observed mean localization error="
                f"{row['observed_mean_localization_error']:.6f}, "
                f"p={row['p_localization_lower']:.6g}; "
                f"early-state peaks="
                f"{row['observed_n_peaks_in_neurog3_early']}/4",
            )

        alignment = pd.DataFrame(
            alignment_rows
        )

        # FDR correction within each test family.
        p_families = [
            (
                "p_localization_lower",
                "q_localization_bh",
            ),
            (
                "p_early_peak_enrichment_upper",
                "q_early_peak_enrichment_bh",
            ),
            (
                "p_positive_rise_count_upper",
                "q_positive_rise_count_bh",
            ),
            (
                "p_mean_rise_upper",
                "q_mean_rise_bh",
            ),
            (
                "p_signed_offset_lower",
                "q_signed_offset_bh",
            ),
        ]

        for p_col, q_col in p_families:

            alignment[
                q_col
            ] = bh_adjust(
                alignment[
                    p_col
                ].to_numpy(
                    dtype=float
                )
            )

        save_table(
            alignment,
            "01_circular_shift_alignment_null_tests.csv",
            index=False,
        )

        write_both(
            report,
            "\nFDR-adjusted alignment summary:\n"
            + alignment[
                [
                    "metric_label",
                    "observed_mean_localization_error",
                    "p_localization_lower",
                    "q_localization_bh",
                    "observed_n_peaks_in_neurog3_early",
                    "p_early_peak_enrichment_upper",
                    "q_early_peak_enrichment_bh",
                    "observed_n_positive_progenitor_to_early_rise",
                    "p_mean_rise_upper",
                    "q_mean_rise_bh",
                    "observed_mean_signed_peak_offset",
                    "p_signed_offset_lower",
                    "q_signed_offset_bh",
                ]
            ].round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 2. Replicate correlation null
        # =========================================================

        write_both(
            report,
            "\n2. CROSS-REPLICATE CORRELATION NULL TESTS",
        )

        write_both(
            report,
            "-" * 96,
        )

        replicate_rows = []

        for metric in METRICS:

            for branch in [0, 1]:

                result = (
                    replicate_correlation_null(
                        prepared=prepared,
                        metric=metric,
                        branch=branch,
                        rng=rng,
                        n_permutations=
                            N_PERMUTATIONS,
                    )
                )

                replicate_rows.append(
                    {
                        "metric":
                            metric,
                        "metric_label":
                            METRIC_LABELS[
                                metric
                            ],
                        "branch":
                            branch,
                        "lineage":
                            BRANCH_LABELS[
                                branch
                            ],
                        "observed_pearson":
                            result[
                                "observed_pearson"
                            ],
                        "null_mean":
                            result[
                                "null_mean"
                            ],
                        "null_q025":
                            result[
                                "null_q025"
                            ],
                        "null_q975":
                            result[
                                "null_q975"
                            ],
                        "p_upper":
                            result[
                                "p_upper"
                            ],
                    }
                )

        replicate_table = pd.DataFrame(
            replicate_rows
        )

        # FDR across the 12 metric x branch replicate tests.
        replicate_table[
            "q_bh"
        ] = bh_adjust(
            replicate_table[
                "p_upper"
            ].to_numpy(
                dtype=float
            )
        )

        save_table(
            replicate_table,
            "02_replicate_correlation_null_tests.csv",
            index=False,
        )

        write_both(
            report,
            replicate_table.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 3. Exact paired GDIS-vs-baseline localization comparisons
        # =========================================================

        write_both(
            report,
            "\n3. EXACT PAIRED GDIS-vs-BASELINE LOCALIZATION TESTS",
        )

        write_both(
            report,
            "-" * 96,
        )

        # Compute group-specific absolute localization errors directly.
        error_rows = []

        for metric in METRICS:

            for branch, differentiation in GROUPS:

                item = prepared[
                    (
                        branch,
                        differentiation,
                    )
                ]

                values = item[
                    "df"
                ][metric].to_numpy(
                    dtype=float
                )

                peak_index = int(
                    np.nanargmax(
                        values
                    )
                )

                peak_parameter = float(
                    item[
                        "parameters"
                    ][peak_index]
                )

                error = abs(
                    peak_parameter
                    - item[
                        "early_landmark"
                    ]
                )

                error_rows.append(
                    {
                        "metric":
                            metric,
                        "branch":
                            branch,
                        "differentiation":
                            differentiation,
                        "absolute_localization_error":
                            error,
                    }
                )

        error_table = pd.DataFrame(
            error_rows
        )

        save_table(
            error_table,
            "03_group_localization_errors.csv",
            index=False,
        )

        paired_rows = []

        gdis_errors = (
            error_table[
                error_table[
                    "metric"
                ] == "gdis"
            ]
            .sort_values(
                [
                    "branch",
                    "differentiation",
                ]
            )[
                "absolute_localization_error"
            ]
            .to_numpy(
                dtype=float
            )
        )

        for baseline in BASELINES:

            baseline_errors = (
                error_table[
                    error_table[
                        "metric"
                    ] == baseline
                ]
                .sort_values(
                    [
                        "branch",
                        "differentiation",
                    ]
                )[
                    "absolute_localization_error"
                ]
                .to_numpy(
                    dtype=float
                )
            )

            differences = (
                baseline_errors
                - gdis_errors
            )

            result = exact_sign_flip_test(
                differences
            )

            paired_rows.append(
                {
                    "comparison":
                        (
                            "GDIS vs "
                            + METRIC_LABELS[
                                baseline
                            ]
                        ),
                    "baseline_metric":
                        baseline,
                    "mean_gdis_error":
                        float(
                            np.mean(
                                gdis_errors
                            )
                        ),
                    "mean_baseline_error":
                        float(
                            np.mean(
                                baseline_errors
                            )
                        ),
                    "mean_baseline_minus_gdis_error":
                        result[
                            "observed_mean_difference"
                        ],
                    "n_groups_gdis_strictly_better":
                        int(
                            np.sum(
                                gdis_errors
                                < baseline_errors
                            )
                        ),
                    "n_groups_tied":
                        int(
                            np.sum(
                                np.isclose(
                                    gdis_errors,
                                    baseline_errors,
                                    atol=1e-12,
                                    rtol=0.0,
                                )
                            )
                        ),
                    "n_groups_baseline_strictly_better":
                        int(
                            np.sum(
                                baseline_errors
                                < gdis_errors
                            )
                        ),
                    "exact_one_sided_p":
                        result[
                            "p_one_sided"
                        ],
                    "n_exact_sign_flip_permutations":
                        result[
                            "n_exact_permutations"
                        ],
                }
            )

        paired_table = pd.DataFrame(
            paired_rows
        )

        paired_table[
            "q_bh"
        ] = bh_adjust(
            paired_table[
                "exact_one_sided_p"
            ].to_numpy(
                dtype=float
            )
        )

        save_table(
            paired_table,
            "04_exact_gdis_vs_baseline_localization_tests.csv",
            index=False,
        )

        write_both(
            report,
            paired_table.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 4. Transition-energy lead-position test
        # =========================================================

        write_both(
            report,
            "\n4. SIGNED PEAK-POSITION INTERPRETATION",
        )

        write_both(
            report,
            "-" * 96,
        )

        signed_table = alignment[
            [
                "metric",
                "metric_label",
                "observed_mean_signed_peak_offset",
                "null_mean_signed_peak_offset",
                "p_signed_offset_lower",
                "q_signed_offset_bh",
            ]
        ].copy()

        signed_table[
            "interpretation"
        ] = np.where(
            signed_table[
                "observed_mean_signed_peak_offset"
            ] < 0,
            "peak precedes median NEUROG3-early landmark on average",
            "peak follows median NEUROG3-early landmark on average",
        )

        save_table(
            signed_table,
            "05_signed_peak_position_tests.csv",
            index=False,
        )

        write_both(
            report,
            signed_table.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 5. Figure
        # =========================================================

        write_both(
            report,
            "\n5. NULL-VALIDATION FIGURE",
        )

        write_both(
            report,
            "-" * 96,
        )

        plot_null_localization(
            alignment,
            FIGURE_DIR
            / "01_observed_vs_null_localization_error.png",
        )

        write_both(
            report,
            "Saved observed-versus-null localization figure.",
        )

        # =========================================================
        # 6. Final interpretation guardrails
        # =========================================================

        write_both(
            report,
            "\n6. FINAL INTERPRETATION",
        )

        write_both(
            report,
            "-" * 96,
        )

        gdis_row = alignment[
            alignment[
                "metric"
            ] == "gdis"
        ].iloc[0]

        write_both(
            report,
            f"GDIS circular-shift localization p-value: "
            f"{gdis_row['p_localization_lower']:.6g}",
        )

        write_both(
            report,
            f"GDIS localization BH q-value: "
            f"{gdis_row['q_localization_bh']:.6g}",
        )

        write_both(
            report,
            "\nThe exact paired comparisons use only the four "
            "independent branch x differentiation analyses and therefore "
            "have very limited resolution. A non-significant paired test "
            "must NOT be interpreted as evidence of equivalence.",
        )

        write_both(
            report,
            "\nThe circular-shift tests assess biological alignment "
            "relative to this trajectory while preserving each profile's "
            "internal structure. They do not replace independent-dataset "
            "validation.",
        )

        write_both(
            report,
            "\nThe median NEUROG3-early landmark is not the true "
            "transition-onset time. A statistically early signed peak "
            "therefore supports earlier positioning relative to the "
            "state median, not proven prospective prediction.",
        )

        write_both(
            report,
            "\nNo GDIS parameter or analysis choice was changed based "
            "on these null-test results.",
        )

        write_both(
            report,
            "\nRecommended next step:",
        )

        write_both(
            report,
            "Freeze the current GSE114412 result and move to an "
            "independent biological dataset for external validation.",
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
    print("p11_statistical_alignment_null_validation.py completed.")
    print("=" * 96)
    print(f"Report : {REPORT_FILE}")
    print(f"Tables : {TABLE_DIR}")
    print(f"Figures: {FIGURE_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

