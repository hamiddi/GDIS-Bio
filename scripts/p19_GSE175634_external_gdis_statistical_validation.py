#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================
"""
p19_GSE175634_external_gdis_statistical_validation.py

Statistical validation of the frozen PRIMARY external GDIS results for
GSE175634.

This script DOES NOT recompute GDIS.

Inputs
------
p18:
    03_all_primary_gdis_profiles.csv.gz

p14:
    03_frozen_state_landmarks_by_individual.csv
    05_prespecified_biological_transitions.csv

p17b:
    04_final_primary_gdis_file_manifest.csv

Independent unit
----------------
INDIVIDUAL.

Because adjacent GDIS windows overlap strongly, windows are NOT treated as
independent observations.

Primary inferential strategy
----------------------------
1. Structure-preserving circular-shift null within each individual profile.
   For each profile, the complete score sequence is circularly rotated relative
   to the fixed pseudotime / biological landmarks.

   This preserves:
       - the exact score-value distribution,
       - the neighboring profile shape under circular rotation,
       - the number of windows,

   while breaking alignment to the frozen biological landmarks.

2. Individual-level source-to-destination changes are tested across independent
   individuals using:
       - exact binomial sign tests,
       - one-sample Wilcoxon signed-rank tests.

Primary biological comparisons
------------------------------
shared_backbone:
    T2 MES -> CMES

cm_extension:
    T3_CM PROG -> CM

cf_extension:
    T3_CF PROG -> CF

Metrics
-------
Primary:
    GDIS

Secondary components:
    sustained_instability
    transition_instability
    transition_energy

Tests per scope x metric
------------------------
A. Median absolute global-peak localization error.
   Lower is better.

B. Number of global peaks falling within the independently frozen
   destination-state interquartile pseudotime interval.

C. Median source->destination score change.
   Positive values indicate increased score at the destination-state median.

D. Median signed global-peak offset from destination-state median.
   Negative = earlier than destination median.
   Positive = later than destination median.
   Report lower, upper, and two-sided empirical probabilities.

E. Individual-level positive source->destination changes:
   exact sign test and Wilcoxon test.

Cross-individual profile agreement
----------------------------------
Profiles are interpolated onto the shared raw deposited-pseudotime interval
within each scope. Pairwise Pearson and Spearman correlations are descriptive;
pairwise correlations are NOT treated as independent inferential samples.

IMPORTANT
---------
The destination-state median is NOT the true transition-onset time.
Statistically earlier peak placement therefore does NOT establish prospective
prediction before biological onset.
"""

from pathlib import Path
import itertools
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import (
    binomtest,
    pearsonr,
    spearmanr,
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

P18_TABLE_DIR = (
    PROJECT_DIR
    / "results"
    / "p18_external_GSE175634_primary_gdis"
    / "tables"
)

PROFILE_FILE = (
    P18_TABLE_DIR
    / "03_all_primary_gdis_profiles.csv.gz"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
    / "p19_external_GSE175634_gdis_statistical_validation"
)

TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR = RESULTS_DIR / "figures"

REPORT_FILE = (
    RESULTS_DIR
    / "p19_external_GSE175634_gdis_statistical_validation_report.txt"
)


# ---------------------------------------------------------------------
# Frozen analysis settings
# ---------------------------------------------------------------------

RANDOM_SEED = 20260913
N_PERMUTATIONS = 20_000

METRICS = [
    "gdis",
    "sustained_instability",
    "transition_instability",
    "transition_energy",
]

METRIC_LABELS = {
    "gdis":
        "GDIS",
    "sustained_instability":
        "Sustained instability",
    "transition_instability":
        "Transition instability",
    "transition_energy":
        "Transition energy",
}

SCOPE_ORDER = [
    "shared_backbone",
    "cm_extension",
    "cf_extension",
]

SCOPE_TRANSITION = {
    "shared_backbone":
        "T2",
    "cm_extension":
        "T3_CM",
    "cf_extension":
        "T3_CF",
}

EXPECTED_SCOPE_COUNTS = {
    "shared_backbone": 19,
    "cm_extension": 15,
    "cf_extension": 14,
}

PROFILE_GRID_POINTS = 200


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
    Benjamini-Hochberg FDR adjustment.

    NaN values remain NaN.
    """
    values = np.asarray(
        p_values,
        dtype=float,
    )

    output = np.full(
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
        return output

    order = np.argsort(
        finite
    )

    ranked = finite[
        order
    ]

    m = len(
        ranked
    )

    adjusted = (
        ranked
        * m
        / np.arange(
            1,
            m + 1,
            dtype=float,
        )
    )

    # Enforce monotonicity from largest rank backward.
    adjusted = np.minimum.accumulate(
        adjusted[
            ::-1
        ]
    )[
        ::-1
    ]

    adjusted = np.clip(
        adjusted,
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

    output[
        finite_mask
    ] = adjusted[
        inverse
    ]

    return output


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
    values,
):
    values = np.asarray(
        values,
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

    n = len(
        nonzero
    )

    if n == 0:
        return (
            0,
            0,
            np.nan,
        )

    n_positive = int(
        np.sum(
            nonzero > 0
        )
    )

    p = float(
        binomtest(
            n_positive,
            n=n,
            p=0.5,
            alternative="greater",
        ).pvalue
    )

    return (
        n_positive,
        n,
        p,
    )


def wilcoxon_greater(
    values,
):
    values = np.asarray(
        values,
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
        result = wilcoxon(
            values,
            alternative="greater",
            zero_method="wilcox",
            method="auto",
        )

        return float(
            result.pvalue
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


def profile_context(
    profile,
    landmark_table,
    source_state,
    destination_state,
):
    """
    Build fixed biological/window context for one profile.
    """
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
        landmark_table[
            landmark_table[
                "individual"
            ] == individual
        ]
        .set_index(
            "state"
        )
    )

    for state in [
        source_state,
        destination_state,
    ]:

        if state not in (
            individual_landmarks.index
        ):
            raise ValueError(
                f"Missing frozen {state} landmark "
                f"for individual {individual}."
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

    source_index = nearest_index(
        parameters,
        source_median,
    )

    destination_index = nearest_index(
        parameters,
        destination_median,
    )

    return {
        "profile":
            profile,
        "individual":
            individual,
        "parameters":
            parameters,
        "source_state":
            source_state,
        "destination_state":
            destination_state,
        "source_median":
            source_median,
        "destination_median":
            destination_median,
        "destination_q25":
            destination_q25,
        "destination_q75":
            destination_q75,
        "source_index":
            source_index,
        "destination_index":
            destination_index,
    }


def observed_metric_statistics(
    context,
    metric,
):
    profile = context[
        "profile"
    ]

    values = profile[
        metric
    ].to_numpy(
        dtype=float
    )

    parameters = context[
        "parameters"
    ]

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

    destination_median = float(
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

    signed_offset = (
        peak_parameter
        - destination_median
    )

    absolute_error = abs(
        signed_offset
    )

    inside_iqr = bool(
        q25
        <= peak_parameter
        <= q75
    )

    source_value = float(
        values[
            source_index
        ]
    )

    destination_value = float(
        values[
            destination_index
        ]
    )

    delta = (
        destination_value
        - source_value
    )

    return {
        "peak_index":
            peak_index,
        "peak_parameter":
            peak_parameter,
        "signed_offset":
            signed_offset,
        "absolute_error":
            absolute_error,
        "inside_destination_iqr":
            inside_iqr,
        "source_value":
            source_value,
        "destination_value":
            destination_value,
        "source_to_destination_delta":
            delta,
        "source_nearest_parameter":
            float(
                parameters[
                    source_index
                ]
            ),
        "destination_nearest_parameter":
            float(
                parameters[
                    destination_index
                ]
            ),
    }


def precompute_shift_statistics(
    context,
    metric,
):
    """
    Precompute statistics for every nonzero circular rotation of one
    individual's score profile.

    Each row corresponds to one possible nonzero shift.
    """
    profile = context[
        "profile"
    ]

    values = profile[
        metric
    ].to_numpy(
        dtype=float
    )

    parameters = context[
        "parameters"
    ]

    n = len(
        values
    )

    if n < 3:
        raise ValueError(
            "Profile too short for circular-shift validation."
        )

    destination_median = float(
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

    rows = []

    for shift in range(
        1,
        n,
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
            - destination_median
        )

        rows.append(
            (
                abs(
                    signed_offset
                ),
                signed_offset,
                float(
                    q25
                    <= peak_parameter
                    <= q75
                ),
                float(
                    shifted[
                        destination_index
                    ]
                    - shifted[
                        source_index
                    ]
                ),
            )
        )

    return np.asarray(
        rows,
        dtype=float,
    )


def run_scope_metric_shift_null(
    contexts,
    metric,
    rng,
):
    """
    Circular-shift null at the independent-profile level.

    Returns observed aggregate statistics, null arrays, and per-profile
    observed rows.
    """
    observed_rows = []
    shift_tables = []

    for context in contexts:

        stats = observed_metric_statistics(
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
            precompute_shift_statistics(
                context,
                metric,
            )
        )

    observed = pd.DataFrame(
        observed_rows
    )

    observed_median_abs_error = float(
        observed[
            "absolute_error"
        ].median()
    )

    observed_median_signed_offset = float(
        observed[
            "signed_offset"
        ].median()
    )

    observed_n_inside_iqr = int(
        observed[
            "inside_destination_iqr"
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

    # One random nonzero circular shift per individual per permutation.
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

        for i, table in enumerate(
            shift_tables
        ):

            row = int(
                rng.integers(
                    0,
                    len(
                        table
                    ),
                )
            )

            abs_values[
                i
            ] = table[
                row,
                0,
            ]

            signed_values[
                i
            ] = table[
                row,
                1,
            ]

            inside_values[
                i
            ] = table[
                row,
                2,
            ]

            delta_values[
                i
            ] = table[
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

    lower_signed = empirical_lower_p(
        null_signed,
        observed_median_signed_offset,
    )

    upper_signed = empirical_upper_p(
        null_signed,
        observed_median_signed_offset,
    )

    two_sided_signed = min(
        1.0,
        2.0
        * min(
            lower_signed,
            upper_signed,
        ),
    )

    result = {
        "n_profiles":
            n_profiles,
        "observed_median_abs_peak_error":
            observed_median_abs_error,
        "p_localization_lower":
            empirical_lower_p(
                null_abs,
                observed_median_abs_error,
            ),
        "null_abs_error_median":
            float(
                np.median(
                    null_abs
                )
            ),
        "null_abs_error_q025":
            float(
                np.quantile(
                    null_abs,
                    0.025,
                )
            ),
        "null_abs_error_q975":
            float(
                np.quantile(
                    null_abs,
                    0.975,
                )
            ),
        "observed_n_peak_inside_destination_iqr":
            observed_n_inside_iqr,
        "p_peak_inside_iqr_upper":
            empirical_upper_p(
                null_inside,
                observed_n_inside_iqr,
            ),
        "null_n_inside_iqr_median":
            float(
                np.median(
                    null_inside
                )
            ),
        "observed_median_source_to_destination_delta":
            observed_median_delta,
        "p_delta_upper":
            empirical_upper_p(
                null_delta,
                observed_median_delta,
            ),
        "null_delta_median":
            float(
                np.median(
                    null_delta
                )
            ),
        "observed_median_signed_peak_offset":
            observed_median_signed_offset,
        "p_signed_offset_lower":
            lower_signed,
        "p_signed_offset_upper":
            upper_signed,
        "p_signed_offset_two_sided":
            two_sided_signed,
        "null_signed_offset_median":
            float(
                np.median(
                    null_signed
                )
            ),
    }

    return (
        result,
        observed,
    )


def build_profile_agreement(
    profiles,
    scope,
    metric,
):
    """
    Descriptive pairwise profile agreement over the common raw-pseudotime
    interval shared by all individuals in one scope.
    """
    subset = profiles[
        profiles[
            "scope"
        ] == scope
    ].copy()

    profile_dict = {}

    minima = []
    maxima = []

    for individual, group in (
        subset.groupby(
            "individual",
            observed=True,
        )
    ):

        group = (
            group.sort_values(
                "parameter"
            )
            .reset_index(
                drop=True
            )
        )

        x = group[
            "parameter"
        ].to_numpy(
            dtype=float
        )

        y = group[
            metric
        ].to_numpy(
            dtype=float
        )

        profile_dict[
            str(
                individual
            )
        ] = (
            x,
            y,
        )

        minima.append(
            float(
                np.min(
                    x
                )
            )
        )

        maxima.append(
            float(
                np.max(
                    x
                )
            )
        )

    common_min = max(
        minima
    )

    common_max = min(
        maxima
    )

    if (
        not np.isfinite(
            common_min
        )
        or not np.isfinite(
            common_max
        )
        or common_max
        <= common_min
    ):
        raise ValueError(
            f"No common pseudotime interval for {scope}/{metric}."
        )

    grid = np.linspace(
        common_min,
        common_max,
        PROFILE_GRID_POINTS,
    )

    interpolated = {}

    for individual, (
        x,
        y,
    ) in profile_dict.items():

        interpolated[
            individual
        ] = np.interp(
            grid,
            x,
            y,
        )

    pair_rows = []

    individuals = sorted(
        interpolated.keys()
    )

    for individual_a, individual_b in (
        itertools.combinations(
            individuals,
            2,
        )
    ):

        a = interpolated[
            individual_a
        ]

        b = interpolated[
            individual_b
        ]

        pearson = float(
            pearsonr(
                a,
                b,
            ).statistic
        )

        spearman = float(
            spearmanr(
                a,
                b,
            ).statistic
        )

        pair_rows.append(
            {
                "scope":
                    scope,
                "metric":
                    metric,
                "metric_label":
                    METRIC_LABELS[
                        metric
                    ],
                "individual_a":
                    individual_a,
                "individual_b":
                    individual_b,
                "common_parameter_min":
                    common_min,
                "common_parameter_max":
                    common_max,
                "pearson":
                    pearson,
                "spearman":
                    spearman,
            }
        )

    pair_table = pd.DataFrame(
        pair_rows
    )

    matrix = np.vstack(
        [
            interpolated[
                individual
            ]
            for individual in individuals
        ]
    )

    consensus = pd.DataFrame(
        {
            "scope":
                scope,
            "metric":
                metric,
            "parameter":
                grid,
            "median":
                np.median(
                    matrix,
                    axis=0,
                ),
            "q25":
                np.quantile(
                    matrix,
                    0.25,
                    axis=0,
                ),
            "q75":
                np.quantile(
                    matrix,
                    0.75,
                    axis=0,
                ),
        }
    )

    summary = {
        "scope":
            scope,
        "metric":
            metric,
        "metric_label":
            METRIC_LABELS[
                metric
            ],
        "n_profiles":
            len(
                individuals
            ),
        "n_profile_pairs":
            len(
                pair_table
            ),
        "common_parameter_min":
            common_min,
        "common_parameter_max":
            common_max,
        "median_pairwise_pearson":
            float(
                pair_table[
                    "pearson"
                ].median()
            ),
        "q25_pairwise_pearson":
            float(
                pair_table[
                    "pearson"
                ].quantile(
                    0.25
                )
            ),
        "q75_pairwise_pearson":
            float(
                pair_table[
                    "pearson"
                ].quantile(
                    0.75
                )
            ),
        "median_pairwise_spearman":
            float(
                pair_table[
                    "spearman"
                ].median()
            ),
        "q25_pairwise_spearman":
            float(
                pair_table[
                    "spearman"
                ].quantile(
                    0.25
                )
            ),
        "q75_pairwise_spearman":
            float(
                pair_table[
                    "spearman"
                ].quantile(
                    0.75
                )
            ),
    }

    return (
        summary,
        pair_table,
        consensus,
    )


def plot_gdis_consensus(
    consensus,
    scope,
    median_destination_landmark,
    output_path,
):
    subset = consensus[
        (
            consensus[
                "scope"
            ] == scope
        )
        & (
            consensus[
                "metric"
            ] == "gdis"
        )
    ].copy()

    fig, ax = plt.subplots(
        figsize=(9.5, 5.8)
    )

    x = subset[
        "parameter"
    ].to_numpy(
        dtype=float
    )

    median = subset[
        "median"
    ].to_numpy(
        dtype=float
    )

    q25 = subset[
        "q25"
    ].to_numpy(
        dtype=float
    )

    q75 = subset[
        "q75"
    ].to_numpy(
        dtype=float
    )

    ax.plot(
        x,
        median,
        linewidth=1.8,
        label="Median GDIS",
    )

    ax.fill_between(
        x,
        q25,
        q75,
        alpha=0.25,
        label="Individual IQR",
    )

    ax.axvline(
        median_destination_landmark,
        linestyle="--",
        linewidth=1.2,
        label="Median frozen destination landmark",
    )

    ax.set_xlabel(
        "Deposited diffusion pseudotime"
    )

    ax.set_ylabel(
        "GDIS"
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.set_title(
        f"{scope}: external-validation GDIS consensus"
    )

    ax.legend()

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

    for directory in [
        TABLE_DIR,
        FIGURE_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    required_files = [
        LANDMARK_FILE,
        TRANSITION_FILE,
        FINAL_MANIFEST_FILE,
        PROFILE_FILE,
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
        PROFILE_FILE,
        compression="gzip",
        low_memory=False,
    )

    # -----------------------------------------------------------------
    # Normalize dtypes / validate.
    # -----------------------------------------------------------------

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

    profiles[
        "scope"
    ] = profiles[
        "scope"
    ].astype(str)

    profiles[
        "transition_id"
    ] = profiles[
        "transition_id"
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
            "p18 profile table missing columns: "
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

    # Manifest/profile cohort integrity.
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
            "p18 profile cohort does not match frozen p17b manifest."
        )

    for scope, expected_count in (
        EXPECTED_SCOPE_COUNTS.items()
    ):

        observed_count = int(
            profiles.loc[
                profiles[
                    "scope"
                ] == scope,
                "individual",
            ].nunique()
        )

        if (
            observed_count
            != expected_count
        ):
            raise ValueError(
                f"{scope}: expected {expected_count} profiles, "
                f"observed {observed_count}."
            )

    transition_lookup = (
        transitions.set_index(
            "transition_id"
        )
    )

    # -----------------------------------------------------------------
    # Build fixed contexts ONCE.
    # -----------------------------------------------------------------

    contexts_by_scope = {}

    for scope in SCOPE_ORDER:

        transition_id = (
            SCOPE_TRANSITION[
                scope
            ]
        )

        if (
            transition_id
            not in transition_lookup.index
        ):
            raise ValueError(
                f"Missing frozen transition {transition_id}."
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

        scope_contexts = []

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

            scope_contexts.append(
                profile_context(
                    profile=group,
                    landmark_table=landmarks,
                    source_state=source_state,
                    destination_state=destination_state,
                )
            )

        contexts_by_scope[
            scope
        ] = scope_contexts

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    null_test_rows = []
    individual_rows = []
    agreement_summary_rows = []
    agreement_pair_tables = []
    consensus_tables = []

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
            "GDIS-Bio GSE175634 Primary External Statistical Validation",
        )

        write_both(
            report,
            "=" * 104,
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
            "Overlapping windows are NOT treated as independent samples.",
        )

        # =========================================================
        # 1. Circular-shift validation
        # =========================================================

        write_both(
            report,
            "\n1. STRUCTURE-PRESERVING CIRCULAR-SHIFT NULL TESTS",
        )

        write_both(
            report,
            "-" * 104,
        )

        for scope in SCOPE_ORDER:

            contexts = (
                contexts_by_scope[
                    scope
                ]
            )

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

            write_both(
                report,
                f"\n{scope} | {transition_id}: "
                f"{source_state} -> {destination_state}",
            )

            for metric in METRICS:

                (
                    null_result,
                    observed_individual,
                ) = run_scope_metric_shift_null(
                    contexts=contexts,
                    metric=metric,
                    rng=rng,
                )

                row = {
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
                    **null_result,
                }

                null_test_rows.append(
                    row
                )

                observed_individual.insert(
                    0,
                    "metric",
                    metric,
                )

                observed_individual.insert(
                    0,
                    "destination_state",
                    destination_state,
                )

                observed_individual.insert(
                    0,
                    "source_state",
                    source_state,
                )

                observed_individual.insert(
                    0,
                    "transition_id",
                    transition_id,
                )

                observed_individual.insert(
                    0,
                    "scope",
                    scope,
                )

                individual_rows.append(
                    observed_individual
                )

                write_both(
                    report,
                    f"  {METRIC_LABELS[metric]}: "
                    f"median abs peak error="
                    f"{null_result['observed_median_abs_peak_error']:.6f}, "
                    f"p={null_result['p_localization_lower']:.6g}; "
                    f"peaks in destination IQR="
                    f"{null_result['observed_n_peak_inside_destination_iqr']}/"
                    f"{null_result['n_profiles']}, "
                    f"p={null_result['p_peak_inside_iqr_upper']:.6g}; "
                    f"median source->destination delta="
                    f"{null_result['observed_median_source_to_destination_delta']:.6f}, "
                    f"p={null_result['p_delta_upper']:.6g}",
                )

        null_tests = pd.DataFrame(
            null_test_rows
        )

        # BH by inferential family across all 12 scope x metric tests.
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
            "01_circular_shift_null_tests.csv",
            index=False,
        )

        # =========================================================
        # 2. Individual-level transition changes
        # =========================================================

        write_both(
            report,
            "\n2. INDEPENDENT-INDIVIDUAL SOURCE-to-DESTINATION TESTS",
        )

        write_both(
            report,
            "-" * 104,
        )

        individual_stats = pd.concat(
            individual_rows,
            ignore_index=True,
        )

        save_table(
            individual_stats,
            "02_individual_peak_and_transition_statistics.csv",
            index=False,
        )

        independent_rows = []

        for (
            scope,
            transition_id,
            metric,
        ), group in (
            individual_stats.groupby(
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

            (
                n_positive,
                n_nonzero,
                sign_p,
            ) = exact_sign_test_greater(
                deltas
            )

            wilcoxon_p = wilcoxon_greater(
                deltas
            )

            independent_rows.append(
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
                                deltas
                                > 0
                            )
                        ),
                    "n_negative_delta":
                        int(
                            np.sum(
                                deltas
                                < 0
                            )
                        ),
                    "n_zero_delta":
                        int(
                            np.sum(
                                deltas
                                == 0
                            )
                        ),
                    "median_delta":
                        float(
                            np.median(
                                deltas
                            )
                        ),
                    "mean_delta":
                        float(
                            np.mean(
                                deltas
                            )
                        ),
                    "sign_test_n_positive_nonzero":
                        n_positive,
                    "sign_test_n_nonzero":
                        n_nonzero,
                    "p_sign_greater":
                        sign_p,
                    "p_wilcoxon_greater":
                        wilcoxon_p,
                }
            )

        independent_tests = pd.DataFrame(
            independent_rows
        )

        independent_tests[
            "q_sign_greater_bh"
        ] = bh_adjust(
            independent_tests[
                "p_sign_greater"
            ].to_numpy(
                dtype=float
            )
        )

        independent_tests[
            "q_wilcoxon_greater_bh"
        ] = bh_adjust(
            independent_tests[
                "p_wilcoxon_greater"
            ].to_numpy(
                dtype=float
            )
        )

        save_table(
            independent_tests,
            "03_independent_individual_transition_tests.csv",
            index=False,
        )

        write_both(
            report,
            independent_tests.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 3. Cross-individual profile agreement
        # =========================================================

        write_both(
            report,
            "\n3. CROSS-INDIVIDUAL PROFILE AGREEMENT "
            "(DESCRIPTIVE)",
        )

        write_both(
            report,
            "-" * 104,
        )

        for scope in SCOPE_ORDER:

            for metric in METRICS:

                (
                    agreement_summary,
                    pair_table,
                    consensus,
                ) = build_profile_agreement(
                    profiles=profiles,
                    scope=scope,
                    metric=metric,
                )

                agreement_summary_rows.append(
                    agreement_summary
                )

                agreement_pair_tables.append(
                    pair_table
                )

                consensus_tables.append(
                    consensus
                )

        agreement_summary_table = pd.DataFrame(
            agreement_summary_rows
        )

        agreement_pairs = pd.concat(
            agreement_pair_tables,
            ignore_index=True,
        )

        consensus_table = pd.concat(
            consensus_tables,
            ignore_index=True,
        )

        save_table(
            agreement_summary_table,
            "04_cross_individual_profile_agreement_summary.csv",
            index=False,
        )

        agreement_pairs.to_csv(
            TABLE_DIR
            / "05_cross_individual_profile_pair_correlations.csv.gz",
            index=False,
            compression="gzip",
        )

        save_table(
            consensus_table,
            "06_scope_consensus_profiles.csv",
            index=False,
        )

        write_both(
            report,
            agreement_summary_table.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 4. GDIS-specific primary summary
        # =========================================================

        write_both(
            report,
            "\n4. PRIMARY GDIS EXTERNAL-VALIDATION SUMMARY",
        )

        write_both(
            report,
            "-" * 104,
        )

        gdis_null = (
            null_tests[
                null_tests[
                    "metric"
                ] == "gdis"
            ]
            .copy()
            .sort_values(
                "scope"
            )
        )

        gdis_independent = (
            independent_tests[
                independent_tests[
                    "metric"
                ] == "gdis"
            ]
            .copy()
        )

        gdis_summary = gdis_null.merge(
            gdis_independent[
                [
                    "scope",
                    "transition_id",
                    "n_positive_delta",
                    "n_negative_delta",
                    "median_delta",
                    "p_sign_greater",
                    "q_sign_greater_bh",
                    "p_wilcoxon_greater",
                    "q_wilcoxon_greater_bh",
                ]
            ],
            on=[
                "scope",
                "transition_id",
            ],
            how="left",
            validate="one_to_one",
        )

        save_table(
            gdis_summary,
            "07_primary_gdis_external_validation_summary.csv",
            index=False,
        )

        write_both(
            report,
            gdis_summary[
                [
                    "scope",
                    "transition_id",
                    "n_profiles",
                    "observed_median_abs_peak_error",
                    "p_localization_lower",
                    "q_localization_bh",
                    "observed_n_peak_inside_destination_iqr",
                    "p_peak_inside_iqr_upper",
                    "q_peak_inside_iqr_bh",
                    "observed_median_signed_peak_offset",
                    "observed_median_source_to_destination_delta",
                    "n_positive_delta",
                    "n_negative_delta",
                    "p_sign_greater",
                    "q_sign_greater_bh",
                    "p_wilcoxon_greater",
                    "q_wilcoxon_greater_bh",
                ]
            ].round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 5. Figures
        # =========================================================

        write_both(
            report,
            "\n5. CONSENSUS GDIS FIGURES",
        )

        write_both(
            report,
            "-" * 104,
        )

        for scope in SCOPE_ORDER:

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

            scope_ids = (
                profiles.loc[
                    profiles[
                        "scope"
                    ] == scope,
                    "individual",
                ]
                .astype(str)
                .unique()
                .tolist()
            )

            median_landmark = float(
                landmarks.loc[
                    (
                        landmarks[
                            "individual"
                        ].isin(
                            scope_ids
                        )
                    )
                    & (
                        landmarks[
                            "state"
                        ] == destination_state
                    ),
                    "state_median_pseudotime",
                ].median()
            )

            plot_gdis_consensus(
                consensus=consensus_table,
                scope=scope,
                median_destination_landmark=
                    median_landmark,
                output_path=(
                    FIGURE_DIR
                    / f"01_{scope}_gdis_consensus.png"
                ),
            )

        write_both(
            report,
            "Saved one cross-individual consensus GDIS figure per scope.",
        )

        # =========================================================
        # 6. Computational qualification
        # =========================================================

        write_both(
            report,
            "\n6. COMPUTATIONAL QUALIFICATION",
        )

        write_both(
            report,
            "-" * 104,
        )

        all_expected_profiles = all(
            int(
                profiles.loc[
                    profiles[
                        "scope"
                    ] == scope,
                    "individual",
                ].nunique()
            )
            == expected
            for scope, expected
            in EXPECTED_SCOPE_COUNTS.items()
        )

        all_metrics_finite = bool(
            np.isfinite(
                profiles[
                    METRICS
                ].to_numpy(
                    dtype=float
                )
            ).all()
        )

        no_duplicate_parameters = bool(
            profiles.groupby(
                [
                    "scope",
                    "individual",
                ],
                observed=True,
            )[
                "parameter"
            ]
            .apply(
                lambda x:
                    x.is_unique
            )
            .all()
        )

        source_destination_distinct = bool(
            individual_stats.groupby(
                [
                    "scope",
                    "individual",
                    "metric",
                ],
                observed=True,
            )
            .apply(
                lambda g:
                    not np.isclose(
                        float(
                            g[
                                "source_nearest_parameter"
                            ].iloc[
                                0
                            ]
                        ),
                        float(
                            g[
                                "destination_nearest_parameter"
                            ].iloc[
                                0
                            ]
                        ),
                    ),
                include_groups=False,
            )
            .all()
        )

        checks = [
            (
                "All frozen external profiles are present",
                all_expected_profiles,
            ),
            (
                "All four tested score profiles are finite",
                all_metrics_finite,
            ),
            (
                "Each individual profile has unique ordered parameters",
                no_duplicate_parameters,
            ),
            (
                "Source and destination landmarks map to distinct "
                "windows in every tested profile",
                source_destination_distinct,
            ),
            (
                f"Exactly {N_PERMUTATIONS:,} circular-shift "
                "permutations were used per scope x metric",
                True,
            ),
            (
                "Independent-unit sign/Wilcoxon tests operate across "
                "individuals rather than windows",
                True,
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
            "08_statistical_validation_qualification.csv",
            index=False,
        )

        write_both(
            report,
            f"\nQualification checks passed: "
            f"{passed}/{len(checks)}",
        )

        # =========================================================
        # 7. Final status / guardrails
        # =========================================================

        write_both(
            report,
            "\n7. FINAL STATUS",
        )

        write_both(
            report,
            "-" * 104,
        )

        if (
            passed
            == len(
                checks
            )
        ):

            write_both(
                report,
                "FINAL COMPUTATIONAL STATUS: PASS",
            )

            write_both(
                report,
                "The frozen primary external GDIS profiles have been "
                "evaluated with individual-level and structure-preserving "
                "null procedures without recomputing or retuning GDIS.",
            )

        else:

            write_both(
                report,
                "FINAL COMPUTATIONAL STATUS: REVIEW REQUIRED",
            )

        write_both(
            report,
            "\nINTERPRETATION GUARDRAILS:",
        )

        write_both(
            report,
            "  - Circular-shift significance means alignment is stronger "
            "than expected after breaking biological alignment while "
            "preserving profile structure.",
        )

        write_both(
            report,
            "  - It does not establish causal transition prediction.",
        )

        write_both(
            report,
            "  - Destination-state medians and IQRs are state landmarks, "
            "not true onset times.",
        )

        write_both(
            report,
            "  - Cross-individual pairwise correlations are descriptive "
            "and are not independent inferential samples.",
        )

        write_both(
            report,
            "  - GDIS is the primary external-validation metric; "
            "its internal components are secondary analyses.",
        )

        write_both(
            report,
            "  - No GDIS setting, window, PCA dimension, cohort, or "
            "biological landmark was changed based on these tests.",
        )

        write_both(
            report,
            "\nNext step after interpretation:",
        )

        write_both(
            report,
            "Run the already frozen 300/75 and 500/125 window "
            "sensitivities and the 5/10/20/30-PC dimensional "
            "sensitivities without changing the primary result.",
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

    print("\n" + "=" * 104)
    print("p19_GSE175634_external_gdis_statistical_validation.py completed.")
    print("=" * 104)
    print(f"Report : {REPORT_FILE}")
    print(f"Tables : {TABLE_DIR}")
    print(f"Figures: {FIGURE_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

