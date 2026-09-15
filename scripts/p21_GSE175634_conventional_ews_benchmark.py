#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================
"""
p21_GSE175634_conventional_ews_benchmark.py

Conventional early-warning / trajectory benchmark for the frozen PRIMARY
GSE175634 external-validation analysis.

This stage mirrors the conventional benchmark used in the GSE114412 discovery
dataset and does NOT recompute GDIS.

Frozen primary design
---------------------
State space:
    50 PCs

Window family:
    400 cells/window
    100-cell step
    75% overlap

Independent replicate unit:
    individual

Final p17b cohort:
    shared_backbone : 19
    cm_extension    : 15
    cf_extension    : 14
    TOTAL           : 48 profiles

Metrics
-------
1. GDIS
   Read unchanged from p18.

2. GDIS transition energy
   Read unchanged from p18.

3. Total variance
       V = sum_j Var(X_j)
   where sample variance uses ddof=1 across the 400 ordered cells.

4. Mean lag-1 pseudotemporal autocorrelation
   For each PC coordinate j:
       rho_j = Corr(X[0:-1,j], X[1:,j])
   The metric is the arithmetic mean of finite rho_j values.

5. Gaussian differential entropy
   For a d-dimensional Gaussian with sample covariance Sigma:
       H = 0.5 * [ d*(1 + ln(2*pi)) + ln(det(Sigma)) ]

   A scale-relative eigenvalue floor is used ONLY if numerical singularity
   prevents a positive finite log-determinant. With 400 observations and
   50 PCs this safeguard is not expected to be active routinely.

6. Mean pseudotemporal step distance
       D = mean_t ||X[t+1] - X[t]||_2

IMPORTANT
---------
Autocorrelation and step distance are pseudotemporal quantities because
adjacent rows are independently sampled cells ordered by deposited diffusion
pseudotime. They are not physical-time autocorrelation or velocity.

Biological landmarks
--------------------
The same p14/p17b source- and destination-state median pseudotimes are used
only after metric computation.

This script is DESCRIPTIVE.
Statistical null validation and paired GDIS-vs-baseline tests are reserved for
p22 so that the benchmark values are frozen before inferential comparison.
"""

from pathlib import Path
import itertools
import math
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


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

P18_PROFILE_FILE = (
    P18_TABLE_DIR
    / "03_all_primary_gdis_profiles.csv.gz"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
    / "p21_external_GSE175634_conventional_ews_benchmark"
)

TABLE_DIR = RESULTS_DIR / "tables"
DATA_DIR = RESULTS_DIR / "data"

REPORT_FILE = (
    RESULTS_DIR
    / "p21_external_GSE175634_conventional_ews_benchmark_report.txt"
)


# ---------------------------------------------------------------------
# Frozen analysis settings
# ---------------------------------------------------------------------

EXPECTED_SCOPE_COUNTS = {
    "shared_backbone": 19,
    "cm_extension": 15,
    "cf_extension": 14,
}

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

METRICS = list(
    METRIC_LABELS.keys()
)

CONVENTIONAL_METRICS = [
    "total_variance",
    "mean_lag1_autocorrelation",
    "gaussian_differential_entropy",
    "mean_step_distance",
]

EXPECTED_WINDOW_SIZE = 400
EXPECTED_DIMENSION = 50

PROFILE_GRID_POINTS = 200

# Numerical safeguard for entropy ONLY when covariance is not positive definite.
ENTROPY_EIGENVALUE_RELATIVE_FLOOR = 1e-12


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


def compute_total_variance(X):
    """
    Sum of sample variances across state-space coordinates.
    Equivalent to trace(sample covariance).
    """
    X = np.asarray(
        X,
        dtype=np.float64,
    )

    variances = np.var(
        X,
        axis=0,
        ddof=1,
    )

    return float(
        np.sum(
            variances
        )
    )


def compute_mean_lag1_autocorrelation(X):
    """
    Mean per-coordinate lag-1 Pearson autocorrelation along pseudotemporal
    row order.

    Coordinates with zero variance in either lagged vector are excluded.
    """
    X = np.asarray(
        X,
        dtype=np.float64,
    )

    correlations = []

    for j in range(
        X.shape[
            1
        ]
    ):

        x0 = X[
            :-1,
            j,
        ]

        x1 = X[
            1:,
            j,
        ]

        sd0 = float(
            np.std(
                x0,
                ddof=1,
            )
        )

        sd1 = float(
            np.std(
                x1,
                ddof=1,
            )
        )

        if (
            not np.isfinite(
                sd0
            )
            or not np.isfinite(
                sd1
            )
            or sd0 <= 0.0
            or sd1 <= 0.0
        ):
            continue

        r = float(
            np.corrcoef(
                x0,
                x1,
            )[
                0,
                1,
            ]
        )

        if np.isfinite(
            r
        ):
            correlations.append(
                r
            )

    if not correlations:
        return np.nan

    return float(
        np.mean(
            correlations
        )
    )


def compute_gaussian_differential_entropy(X):
    """
    Multivariate Gaussian differential entropy based on the sample covariance.

    H = 0.5 * [d * (1 + log(2*pi)) + log(det(Sigma))]

    Returns
    -------
    entropy : float
    used_eigenvalue_floor : bool
    """
    X = np.asarray(
        X,
        dtype=np.float64,
    )

    d = int(
        X.shape[
            1
        ]
    )

    covariance = np.cov(
        X,
        rowvar=False,
        ddof=1,
    )

    covariance = np.asarray(
        covariance,
        dtype=np.float64,
    )

    # Symmetrize against tiny floating-point asymmetry.
    covariance = 0.5 * (
        covariance
        + covariance.T
    )

    sign, logdet = np.linalg.slogdet(
        covariance
    )

    used_floor = False

    if (
        sign <= 0
        or not np.isfinite(
            logdet
        )
    ):

        used_floor = True

        eigenvalues = np.linalg.eigvalsh(
            covariance
        )

        scale = float(
            np.trace(
                covariance
            )
            / max(
                d,
                1,
            )
        )

        if (
            not np.isfinite(
                scale
            )
            or scale <= 0
        ):
            scale = 1.0

        floor = (
            ENTROPY_EIGENVALUE_RELATIVE_FLOOR
            * scale
        )

        eigenvalues = np.maximum(
            eigenvalues,
            floor,
        )

        logdet = float(
            np.sum(
                np.log(
                    eigenvalues
                )
            )
        )

    entropy = 0.5 * (
        d
        * (
            1.0
            + math.log(
                2.0
                * math.pi
            )
        )
        + float(
            logdet
        )
    )

    return (
        float(
            entropy
        ),
        used_floor,
    )


def compute_mean_step_distance(X):
    """
    Mean Euclidean distance between successive pseudotemporally ordered cells.
    """
    X = np.asarray(
        X,
        dtype=np.float64,
    )

    steps = np.diff(
        X,
        axis=0,
    )

    distances = np.linalg.norm(
        steps,
        axis=1,
    )

    return float(
        np.mean(
            distances
        )
    )


def compute_conventional_metrics(X):
    entropy, used_floor = (
        compute_gaussian_differential_entropy(
            X
        )
    )

    return {
        "total_variance":
            compute_total_variance(
                X
            ),
        "mean_lag1_autocorrelation":
            compute_mean_lag1_autocorrelation(
                X
            ),
        "gaussian_differential_entropy":
            entropy,
        "mean_step_distance":
            compute_mean_step_distance(
                X
            ),
        "entropy_used_eigenvalue_floor":
            bool(
                used_floor
            ),
    }


def nearest_parameter_index(
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


def normalized_change(
    values,
    source_value,
    destination_value,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    metric_range = float(
        np.max(
            values
        )
        - np.min(
            values
        )
    )

    if (
        not np.isfinite(
            metric_range
        )
        or metric_range <= 0
    ):
        return np.nan

    return float(
        (
            destination_value
            - source_value
        )
        / metric_range
    )


def interpolate_profile(
    parameter,
    values,
    grid,
):
    return np.interp(
        grid,
        parameter,
        values,
    )


def pairwise_profile_agreement(
    profile_table,
    scope,
    metric,
):
    subset = (
        profile_table[
            profile_table[
                "scope"
            ] == scope
        ]
        .copy()
    )

    profiles = {}

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

        profiles[
            str(
                individual
            )
        ] = (
            x,
            y,
        )

        minima.append(
            float(
                x.min()
            )
        )

        maxima.append(
            float(
                x.max()
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
        or common_max <= common_min
    ):
        raise ValueError(
            f"No common pseudotime interval for "
            f"{scope}/{metric}."
        )

    grid = np.linspace(
        common_min,
        common_max,
        PROFILE_GRID_POINTS,
    )

    interpolated = {
        individual:
            interpolate_profile(
                x,
                y,
                grid,
            )
        for individual, (
            x,
            y,
        ) in profiles.items()
    }

    rows = []

    individuals = sorted(
        interpolated.keys()
    )

    for a, b in itertools.combinations(
        individuals,
        2,
    ):

        ya = interpolated[
            a
        ]

        yb = interpolated[
            b
        ]

        rows.append(
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
                    a,
                "individual_b":
                    b,
                "pearson":
                    float(
                        pearsonr(
                            ya,
                            yb,
                        ).statistic
                    ),
                "spearman":
                    float(
                        spearmanr(
                            ya,
                            yb,
                        ).statistic
                    ),
                "mae":
                    float(
                        np.mean(
                            np.abs(
                                ya
                                - yb
                            )
                        )
                    ),
                "shared_pseudotime_min":
                    common_min,
                "shared_pseudotime_max":
                    common_max,
            }
        )

    return pd.DataFrame(
        rows
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
        LANDMARK_FILE,
        TRANSITION_FILE,
        FINAL_MANIFEST_FILE,
        P18_PROFILE_FILE,
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

    p18_profiles = pd.read_csv(
        P18_PROFILE_FILE,
        compression="gzip",
        low_memory=False,
    )

    # Normalize identifiers.
    for table in [
        landmarks,
        manifest,
        p18_profiles,
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

    manifest[
        "scope"
    ] = manifest[
        "scope"
    ].astype(str)

    p18_profiles[
        "scope"
    ] = p18_profiles[
        "scope"
    ].astype(str)

    transition_lookup = (
        transitions.set_index(
            "transition_id"
        )
    )

    # Frozen cohort integrity.
    for scope, expected in (
        EXPECTED_SCOPE_COUNTS.items()
    ):

        observed = int(
            manifest.loc[
                manifest[
                    "scope"
                ] == scope,
                "individual",
            ].nunique()
        )

        if observed != expected:
            raise ValueError(
                f"{scope}: expected {expected} frozen profiles, "
                f"observed {observed}."
            )

    p18_pairs = set(
        zip(
            p18_profiles[
                "scope"
            ].astype(str),
            p18_profiles[
                "individual"
            ].astype(str),
        )
    )

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

    if (
        p18_pairs
        != manifest_pairs
    ):
        raise ValueError(
            "p18 profile cohort does not match frozen p17b manifest."
        )

    profile_frames = []
    peak_rows = []
    change_rows = []
    entropy_floor_rows = []

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
            "GDIS-Bio GSE175634 Conventional Early-Warning Benchmark",
        )

        write_both(
            report,
            "=" * 106,
        )

        write_both(
            report,
            "\nFrozen analysis design:",
        )

        write_both(
            report,
            "  50 PCs; 400 cells/window; 100-cell step; 75% overlap",
        )

        write_both(
            report,
            "  Final p17b transition-evaluable cohort: 48 profiles",
        )

        write_both(
            report,
            "  GDIS values are read unchanged from p18.",
        )

        write_both(
            report,
            "\nConventional benchmark metrics:",
        )

        for metric in (
            CONVENTIONAL_METRICS
        ):

            write_both(
                report,
                f"  - "
                f"{METRIC_LABELS[metric]}",
            )

        write_both(
            report,
            "\nNOTE: autocorrelation and step distance are "
            "pseudotemporal, not physical-time, quantities.",
        )

        # =========================================================
        # 1. Compute conventional metrics
        # =========================================================

        write_both(
            report,
            "\n1. COMPUTE CONVENTIONAL METRICS ON FROZEN WINDOWS",
        )

        write_both(
            report,
            "-" * 106,
        )

        for row_number, row in (
            manifest.sort_values(
                [
                    "scope",
                    "individual",
                ]
            )
            .reset_index(
                drop=True
            )
            .iterrows()
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

            transition_id = str(
                row[
                    "transition_id"
                ]
            )

            destination_state = str(
                row[
                    "destination_state"
                ]
            )

            destination_landmark = float(
                row[
                    "destination_state_median_pseudotime"
                ]
            )

            source_state = str(
                transition_lookup.loc[
                    transition_id,
                    "source_state",
                ]
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

            if trajectories.shape[
                1:
            ] != (
                EXPECTED_WINDOW_SIZE,
                EXPECTED_DIMENSION,
            ):
                raise ValueError(
                    f"{scope}/{individual}: unexpected trajectory shape "
                    f"{trajectories.shape}."
                )

            if len(
                parameters
            ) != trajectories.shape[
                0
            ]:
                raise ValueError(
                    f"{scope}/{individual}: parameter/window mismatch."
                )

            if not np.all(
                np.diff(
                    parameters
                ) > 0
            ):
                raise ValueError(
                    f"{scope}/{individual}: parameters not strictly increasing."
                )

            p18_group = (
                p18_profiles[
                    (
                        p18_profiles[
                            "scope"
                        ] == scope
                    )
                    & (
                        p18_profiles[
                            "individual"
                        ] == individual
                    )
                ]
                .sort_values(
                    "parameter"
                )
                .reset_index(
                    drop=True
                )
            )

            if len(
                p18_group
            ) != len(
                parameters
            ):
                raise ValueError(
                    f"{scope}/{individual}: p18 profile/window mismatch."
                )

            if not np.allclose(
                p18_group[
                    "parameter"
                ].to_numpy(
                    dtype=float
                ),
                parameters,
                atol=1e-12,
                rtol=1e-10,
            ):
                raise ValueError(
                    f"{scope}/{individual}: p18 parameters differ "
                    "from frozen NPZ parameters."
                )

            benchmark = pd.DataFrame(
                {
                    "scope":
                        scope,
                    "individual":
                        individual,
                    "transition_id":
                        transition_id,
                    "source_state":
                        source_state,
                    "destination_state":
                        destination_state,
                    "destination_landmark":
                        destination_landmark,
                    "window_index":
                        np.arange(
                            len(
                                parameters
                            ),
                            dtype=int,
                        ),
                    "parameter":
                        parameters,
                    "gdis":
                        p18_group[
                            "gdis"
                        ].to_numpy(
                            dtype=float
                        ),
                    "transition_energy":
                        p18_group[
                            "transition_energy"
                        ].to_numpy(
                            dtype=float
                        ),
                }
            )

            entropy_floor_count = 0

            conventional_values = {
                metric: []
                for metric in (
                    CONVENTIONAL_METRICS
                )
            }

            for window_index in range(
                trajectories.shape[
                    0
                ]
            ):

                metrics = (
                    compute_conventional_metrics(
                        trajectories[
                            window_index
                        ]
                    )
                )

                for metric in (
                    CONVENTIONAL_METRICS
                ):

                    conventional_values[
                        metric
                    ].append(
                        metrics[
                            metric
                        ]
                    )

                entropy_floor_count += int(
                    metrics[
                        "entropy_used_eigenvalue_floor"
                    ]
                )

            for metric in (
                CONVENTIONAL_METRICS
            ):

                benchmark[
                    metric
                ] = np.asarray(
                    conventional_values[
                        metric
                    ],
                    dtype=float,
                )

            if not np.isfinite(
                benchmark[
                    METRICS
                ].to_numpy(
                    dtype=float
                )
            ).all():
                raise ValueError(
                    f"{scope}/{individual}: non-finite benchmark values."
                )

            profile_frames.append(
                benchmark
            )

            entropy_floor_rows.append(
                {
                    "scope":
                        scope,
                    "individual":
                        individual,
                    "n_windows":
                        int(
                            len(
                                benchmark
                            )
                        ),
                    "n_entropy_windows_using_eigenvalue_floor":
                        entropy_floor_count,
                }
            )

            # Frozen individual source/destination landmarks.
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

            source_landmark = float(
                individual_landmarks.loc[
                    source_state,
                    "state_median_pseudotime",
                ]
            )

            source_index = nearest_parameter_index(
                parameters,
                source_landmark,
            )

            destination_index = nearest_parameter_index(
                parameters,
                destination_landmark,
            )

            for metric in METRICS:

                values = benchmark[
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

                peak_rows.append(
                    {
                        "scope":
                            scope,
                        "individual":
                            individual,
                        "transition_id":
                            transition_id,
                        "metric":
                            metric,
                        "metric_label":
                            METRIC_LABELS[
                                metric
                            ],
                        "peak_parameter":
                            peak_parameter,
                        "destination_landmark":
                            destination_landmark,
                        "peak_minus_destination_landmark":
                            (
                                peak_parameter
                                - destination_landmark
                            ),
                        "absolute_peak_localization_error":
                            abs(
                                peak_parameter
                                - destination_landmark
                            ),
                        "peak_value":
                            float(
                                values[
                                    peak_index
                                ]
                            ),
                    }
                )

                change_rows.append(
                    {
                        "scope":
                            scope,
                        "individual":
                            individual,
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
                        "source_landmark":
                            source_landmark,
                        "destination_landmark":
                            destination_landmark,
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
                        "source_value":
                            source_value,
                        "destination_value":
                            destination_value,
                        "destination_minus_source":
                            delta,
                        "normalized_destination_minus_source":
                            normalized_change(
                                values,
                                source_value,
                                destination_value,
                            ),
                        "increases_to_destination":
                            bool(
                                delta > 0
                            ),
                    }
                )

            write_both(
                report,
                f"[{row_number + 1:02d}/{len(manifest):02d}] "
                f"{scope:16s} | individual {individual} | "
                f"{len(parameters):3d} windows",
            )

        profiles = pd.concat(
            profile_frames,
            ignore_index=True,
        )

        peaks = pd.DataFrame(
            peak_rows
        )

        changes = pd.DataFrame(
            change_rows
        )

        entropy_floor_table = pd.DataFrame(
            entropy_floor_rows
        )

        profiles.to_csv(
            DATA_DIR
            / "all_primary_benchmark_profiles.csv.gz",
            index=False,
            compression="gzip",
        )

        save_table(
            peaks,
            "01_global_peak_localization.csv",
            index=False,
        )

        save_table(
            changes,
            "02_source_to_destination_metric_change.csv",
            index=False,
        )

        save_table(
            entropy_floor_table,
            "03_entropy_numerical_safeguard_usage.csv",
            index=False,
        )

        # =========================================================
        # 2. Peak-localization summary
        # =========================================================

        write_both(
            report,
            "\n2. GLOBAL-PEAK LOCALIZATION SUMMARY",
        )

        write_both(
            report,
            "-" * 106,
        )

        localization_summary = (
            peaks.groupby(
                [
                    "scope",
                    "transition_id",
                    "metric",
                    "metric_label",
                ],
                observed=True,
            )
            .agg(
                n_profiles=(
                    "individual",
                    "nunique",
                ),
                median_absolute_peak_localization_error=(
                    "absolute_peak_localization_error",
                    "median",
                ),
                mean_absolute_peak_localization_error=(
                    "absolute_peak_localization_error",
                    "mean",
                ),
                median_signed_peak_offset=(
                    "peak_minus_destination_landmark",
                    "median",
                ),
                q25_absolute_peak_localization_error=(
                    "absolute_peak_localization_error",
                    lambda x:
                        x.quantile(
                            0.25
                        ),
                ),
                q75_absolute_peak_localization_error=(
                    "absolute_peak_localization_error",
                    lambda x:
                        x.quantile(
                            0.75
                        ),
                ),
            )
            .reset_index()
        )

        save_table(
            localization_summary,
            "04_scope_localization_summary.csv",
            index=False,
        )

        write_both(
            report,
            localization_summary.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 3. Source -> destination change summary
        # =========================================================

        write_both(
            report,
            "\n3. SOURCE-to-DESTINATION METRIC CHANGE SUMMARY",
        )

        write_both(
            report,
            "-" * 106,
        )

        change_summary = (
            changes.groupby(
                [
                    "scope",
                    "transition_id",
                    "metric",
                    "metric_label",
                ],
                observed=True,
            )
            .agg(
                n_profiles=(
                    "individual",
                    "nunique",
                ),
                n_positive_change=(
                    "increases_to_destination",
                    "sum",
                ),
                median_change=(
                    "destination_minus_source",
                    "median",
                ),
                mean_change=(
                    "destination_minus_source",
                    "mean",
                ),
                median_normalized_change=(
                    "normalized_destination_minus_source",
                    "median",
                ),
            )
            .reset_index()
        )

        save_table(
            change_summary,
            "05_scope_source_destination_change_summary.csv",
            index=False,
        )

        write_both(
            report,
            change_summary.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 4. Cross-individual profile agreement
        # =========================================================

        write_both(
            report,
            "\n4. CROSS-INDIVIDUAL PROFILE AGREEMENT "
            "(DESCRIPTIVE)",
        )

        write_both(
            report,
            "-" * 106,
        )

        pair_tables = []

        for scope in SCOPE_ORDER:

            for metric in METRICS:

                pair_tables.append(
                    pairwise_profile_agreement(
                        profiles,
                        scope,
                        metric,
                    )
                )

        pairwise = pd.concat(
            pair_tables,
            ignore_index=True,
        )

        pairwise.to_csv(
            TABLE_DIR
            / "06_pairwise_profile_agreement.csv.gz",
            index=False,
            compression="gzip",
        )

        agreement_summary = (
            pairwise.groupby(
                [
                    "scope",
                    "metric",
                    "metric_label",
                ],
                observed=True,
            )
            .agg(
                n_pairs=(
                    "individual_a",
                    "size",
                ),
                median_pearson=(
                    "pearson",
                    "median",
                ),
                q25_pearson=(
                    "pearson",
                    lambda x:
                        x.quantile(
                            0.25
                        ),
                ),
                q75_pearson=(
                    "pearson",
                    lambda x:
                        x.quantile(
                            0.75
                        ),
                ),
                median_spearman=(
                    "spearman",
                    "median",
                ),
                median_pairwise_mae=(
                    "mae",
                    "median",
                ),
            )
            .reset_index()
        )

        save_table(
            agreement_summary,
            "07_profile_agreement_summary.csv",
            index=False,
        )

        write_both(
            report,
            agreement_summary.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 5. Integrated descriptive benchmark
        # =========================================================

        write_both(
            report,
            "\n5. INTEGRATED DESCRIPTIVE BENCHMARK",
        )

        write_both(
            report,
            "-" * 106,
        )

        integrated = (
            localization_summary.merge(
                change_summary[
                    [
                        "scope",
                        "transition_id",
                        "metric",
                        "n_positive_change",
                        "median_normalized_change",
                    ]
                ],
                on=[
                    "scope",
                    "transition_id",
                    "metric",
                ],
                how="left",
                validate="one_to_one",
            )
            .merge(
                agreement_summary[
                    [
                        "scope",
                        "metric",
                        "median_pearson",
                        "median_spearman",
                    ]
                ],
                on=[
                    "scope",
                    "metric",
                ],
                how="left",
                validate="one_to_one",
            )
        )

        save_table(
            integrated,
            "08_integrated_descriptive_benchmark.csv",
            index=False,
        )

        write_both(
            report,
            integrated[
                [
                    "scope",
                    "transition_id",
                    "metric",
                    "metric_label",
                    "n_profiles",
                    "median_absolute_peak_localization_error",
                    "median_signed_peak_offset",
                    "n_positive_change",
                    "median_normalized_change",
                    "median_pearson",
                    "median_spearman",
                ]
            ].round(
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
            "\n6. BENCHMARK QUALIFICATION",
        )

        write_both(
            report,
            "-" * 106,
        )

        all_values_finite = bool(
            np.isfinite(
                profiles[
                    METRICS
                ].to_numpy(
                    dtype=float
                )
            ).all()
        )

        all_profiles_present = all(
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

        all_window_counts_match = True

        for (
            scope,
            individual,
        ), group in (
            profiles.groupby(
                [
                    "scope",
                    "individual",
                ],
                observed=True,
            )
        ):

            p18_n = len(
                p18_profiles[
                    (
                        p18_profiles[
                            "scope"
                        ] == scope
                    )
                    & (
                        p18_profiles[
                            "individual"
                        ] == individual
                    )
                ]
            )

            if len(
                group
            ) != p18_n:
                all_window_counts_match = False
                break

        entropy_floor_count = int(
            entropy_floor_table[
                "n_entropy_windows_using_eigenvalue_floor"
            ].sum()
        )

        checks = [
            (
                "All benchmark metric values are finite",
                all_values_finite,
            ),
            (
                "All 48 frozen scope x individual profiles are benchmarked",
                all_profiles_present,
            ),
            (
                "Every benchmark profile uses exactly the same windows "
                "as p18 GDIS",
                all_window_counts_match,
            ),
            (
                "GDIS and transition-energy values are reused from p18 "
                "rather than recomputed",
                True,
            ),
        ]

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
            f"\nEntropy covariance numerical safeguard used in "
            f"{entropy_floor_count} total windows.",
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

        write_both(
            report,
            "\nINTERPRETATION RULES:",
        )

        write_both(
            report,
            "  - Smaller peak-localization error means closer alignment "
            "to the independently frozen destination-state median.",
        )

        write_both(
            report,
            "  - No metric is considered superior from one descriptive "
            "criterion alone.",
        )

        write_both(
            report,
            "  - Destination-state medians are biological state landmarks, "
            "not proven transition-onset times.",
        )

        write_both(
            report,
            "  - Pairwise profile correlations are descriptive because "
            "pairs sharing individuals are not independent.",
        )

        write_both(
            report,
            "  - No benchmark result is used to retune GDIS.",
        )

        write_both(
            report,
            "\nNext step:",
        )

        write_both(
            report,
            "Apply structure-preserving circular-shift nulls and paired "
            "individual-level GDIS-vs-baseline localization tests to these "
            "now-frozen benchmark profiles.",
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

    print("\n" + "=" * 106)
    print("p21_GSE175634_conventional_ews_benchmark.py completed.")
    print("=" * 106)
    print(f"Report : {REPORT_FILE}")
    print(f"Tables : {TABLE_DIR}")
    print(f"Data   : {DATA_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

