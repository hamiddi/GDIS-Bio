#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================
"""
p10_conventional_ews_benchmark.py

Benchmark the frozen GDIS-Bio result against conventional
early-warning / transition metrics on exactly the same windows.

NO GDIS parameters, PCA dimensions, windows, trajectories, or biological
landmarks are changed in this script.

Dataset
-------
GSE114412 validated endocrine trajectory.

Frozen primary representation
-----------------------------
50 PCs
400 cells/window
100-cell step
75% overlap

Groups
------
Branch 0 / Differentiation 1
Branch 0 / Differentiation 2
Branch 1 / Differentiation 1
Branch 1 / Differentiation 2

Conventional benchmark metrics
------------------------------
For each frozen 400 x 50 pseudotemporal window:

1. Total variance
       trace(covariance)

2. Mean lag-1 pseudotemporal autocorrelation
       mean Pearson correlation between PC_k[i] and PC_k[i+1]
       across the 50 PC coordinate series.

3. Gaussian differential entropy
       H = 0.5 * [d(1 + log(2*pi)) + log(det(Sigma_reg))]

   with a tiny pre-specified covariance ridge for numerical stability.

4. Mean pseudotemporal step distance
       mean ||x[i+1] - x[i]||_2

Important interpretation
------------------------
"lag-1 autocorrelation" and "step distance" here are PSEUDOTEMPORAL,
not physical-time, quantities because the cells are independent cells
ordered along deposited pseudotime.

GDIS comparators
----------------
The already-computed p8 values are included without recomputation:

- GDIS
- transition energy

Benchmark questions
-------------------
A. Where does each metric reach its global maximum relative to the frozen
   median NEUROG3-early landmark?

B. Is the metric higher at NEUROG3-early than at prog_nkx61?

C. How reproducible is each metric profile between the two independent
   differentiations?

D. Does each metric peak in a NEUROG3-early-dominant window?

E. How consistent is the localization across both future lineages?

This is a descriptive benchmark, not a tuning exercise.
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

P7_DIR = (
    PROJECT_DIR
    / "results"
    / "p7_trajectory_assembly_validation"
)

P7_DATA_DIR = P7_DIR / "data"
P7_TABLE_DIR = P7_DIR / "tables"

P8_DATA_DIR = (
    PROJECT_DIR
    / "results"
    / "p8_gdis_primary_analysis"
    / "data"
)

PRIMARY_CONFIG = "primary_w400_s100"
N_PCS = 50

LANDMARK_FILE = (
    P7_TABLE_DIR
    / "07_primary_state_landmark_window_mapping.csv"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
    / "p10_conventional_ews_benchmark"
)

TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR = RESULTS_DIR / "figures"

REPORT_FILE = (
    RESULTS_DIR
    / "p10_conventional_ews_benchmark_report.txt"
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

PRIMARY_GDIS_FILES = {
    (0, 1): "primary_branch0_diff1_gdis.csv",
    (0, 2): "primary_branch0_diff2_gdis.csv",
    (1, 1): "primary_branch1_diff1_gdis.csv",
    (1, 2): "primary_branch1_diff2_gdis.csv",
}

# Conventional metrics used in the main benchmark.
BASELINE_METRICS = [
    "total_variance",
    "mean_lag1_autocorrelation",
    "gaussian_differential_entropy",
    "mean_step_distance",
]

COMPARATOR_METRICS = [
    "gdis",
    "transition_energy",
]

ALL_METRICS = (
    COMPARATOR_METRICS
    + BASELINE_METRICS
)

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

# Tiny fixed covariance ridge used only for numerical stability.
COVARIANCE_RIDGE_RELATIVE = 1e-6

CONSENSUS_GRID_POINTS = 250


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def write_both(report, text=""):
    """Print and write text."""
    print(text)
    report.write(text + "\n")
    report.flush()


def save_table(df, filename, index=True):
    """Save dataframe under tables directory."""
    path = TABLE_DIR / filename
    df.to_csv(path, index=index)
    return path


def group_label(branch, differentiation):
    return (
        f"branch{int(branch)}_"
        f"diff{int(differentiation)}"
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


def minmax(values):
    """
    Min-max scale to [0, 1] for visualization only.
    Raw metrics are used for all quantitative benchmark tables.
    """
    values = np.asarray(
        values,
        dtype=float,
    )

    lo = float(
        np.nanmin(values)
    )

    hi = float(
        np.nanmax(values)
    )

    if not np.isfinite(
        lo
    ) or not np.isfinite(
        hi
    ):
        return np.full_like(
            values,
            np.nan,
            dtype=float,
        )

    if hi <= lo:
        return np.zeros_like(
            values,
            dtype=float,
        )

    return (
        values - lo
    ) / (
        hi - lo
    )


def load_trajectory_archive(
    branch,
    differentiation,
):
    label = group_label(
        branch,
        differentiation,
    )

    path = (
        P7_DATA_DIR
        / PRIMARY_CONFIG
        / f"{label}.npz"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing p7 archive: {path}"
        )

    data = np.load(
        path,
        allow_pickle=False,
    )

    trajectories = np.asarray(
        data["trajectories"],
        dtype=np.float64,
    )

    parameters = np.asarray(
        data["parameters"],
        dtype=np.float64,
    )

    return (
        path,
        trajectories,
        parameters,
    )


def load_window_metadata(
    branch,
    differentiation,
):
    label = group_label(
        branch,
        differentiation,
    )

    path = (
        P7_DATA_DIR
        / PRIMARY_CONFIG
        / f"{label}_windows.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing p7 window metadata: {path}"
        )

    return pd.read_csv(path)


def load_gdis_output(
    branch,
    differentiation,
):
    path = (
        P8_DATA_DIR
        / PRIMARY_GDIS_FILES[
            (branch, differentiation)
        ]
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing p8 GDIS output: {path}"
        )

    df = pd.read_csv(path)

    required = [
        "parameter",
        "gdis",
        "transition_energy",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path.name} missing columns: "
            + ", ".join(missing)
        )

    return (
        df[
            required
        ]
        .sort_values(
            "parameter"
        )
        .reset_index(drop=True)
    )


def mean_lag1_autocorrelation(X):
    """
    Mean lag-1 Pearson correlation across PC dimensions.

    X shape:
        n_ordered_cells x n_dimensions
    """
    values = []

    for j in range(
        X.shape[1]
    ):

        x0 = X[:-1, j]
        x1 = X[1:, j]

        corr = safe_pearson(
            x0,
            x1,
        )

        if np.isfinite(corr):
            values.append(corr)

    if not values:
        return np.nan

    return float(
        np.mean(values)
    )


def gaussian_differential_entropy(
    covariance,
):
    """
    Gaussian differential entropy using a tiny fixed relative ridge.

    H = 0.5 * [d*(1 + ln(2*pi)) + ln(det(Sigma_reg))]
    """
    covariance = np.asarray(
        covariance,
        dtype=float,
    )

    d = covariance.shape[0]

    diagonal = np.diag(
        covariance
    )

    positive_diag = diagonal[
        diagonal > 0
    ]

    if len(positive_diag):
        scale = float(
            np.median(
                positive_diag
            )
        )
    else:
        scale = 1.0

    ridge = (
        COVARIANCE_RIDGE_RELATIVE
        * max(
            scale,
            np.finfo(float).eps,
        )
    )

    covariance_reg = (
        covariance
        + ridge
        * np.eye(d)
    )

    sign, logdet = (
        np.linalg.slogdet(
            covariance_reg
        )
    )

    if sign <= 0:
        return np.nan

    entropy = 0.5 * (
        d
        * (
            1.0
            + np.log(
                2.0
                * np.pi
            )
        )
        + logdet
    )

    return float(entropy)


def compute_window_baselines(
    trajectory,
):
    """
    Compute all conventional metrics for one frozen trajectory window.
    """
    X = np.asarray(
        trajectory,
        dtype=np.float64,
    )

    if X.ndim != 2:
        raise ValueError(
            f"Expected 2D trajectory; got {X.shape}"
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "Non-finite values in trajectory."
        )

    covariance = np.cov(
        X,
        rowvar=False,
        ddof=1,
    )

    total_variance = float(
        np.trace(
            covariance
        )
    )

    lag1 = (
        mean_lag1_autocorrelation(
            X
        )
    )

    entropy = (
        gaussian_differential_entropy(
            covariance
        )
    )

    steps = np.linalg.norm(
        np.diff(
            X,
            axis=0,
        ),
        axis=1,
    )

    mean_step = float(
        np.mean(
            steps
        )
    )

    median_step = float(
        np.median(
            steps
        )
    )

    return {
        "total_variance":
            total_variance,
        "mean_lag1_autocorrelation":
            lag1,
        "gaussian_differential_entropy":
            entropy,
        "mean_step_distance":
            mean_step,
        "median_step_distance":
            median_step,
    }


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


def nearest_row(
    df,
    parameter,
):
    distances = np.abs(
        df["parameter"].to_numpy(
            dtype=float
        )
        - float(parameter)
    )

    pos = int(
        np.argmin(
            distances
        )
    )

    return (
        df.iloc[pos],
        float(
            distances[pos]
        ),
    )


def profile_agreement(
    a,
    b,
    metric,
):
    """
    Replicate profile agreement on shared actual-pseudotime interval.
    """
    lower = max(
        float(
            a["parameter"].min()
        ),
        float(
            b["parameter"].min()
        ),
    )

    upper = min(
        float(
            a["parameter"].max()
        ),
        float(
            b["parameter"].max()
        ),
    )

    grid = np.linspace(
        lower,
        upper,
        CONSENSUS_GRID_POINTS,
    )

    va = np.interp(
        grid,
        a["parameter"],
        a[metric],
    )

    vb = np.interp(
        grid,
        b["parameter"],
        b[metric],
    )

    return {
        "pearson":
            safe_pearson(
                va,
                vb,
            ),
        "spearman":
            safe_spearman(
                va,
                vb,
            ),
        "mae":
            float(
                np.mean(
                    np.abs(
                        va
                        - vb
                    )
                )
            ),
        "shared_min":
            lower,
        "shared_max":
            upper,
    }


def normalized_landmark_rise(
    profile,
    metric,
    progenitor_parameter,
    early_parameter,
):
    """
    Difference from progenitor to NEUROG3-early, normalized by the
    observed within-profile range of that metric.
    """
    prog_row, _ = nearest_row(
        profile,
        progenitor_parameter,
    )

    early_row, _ = nearest_row(
        profile,
        early_parameter,
    )

    delta = float(
        early_row[metric]
        - prog_row[metric]
    )

    values = profile[
        metric
    ].to_numpy(dtype=float)

    metric_range = float(
        np.max(values)
        - np.min(values)
    )

    normalized_delta = (
        delta / metric_range
        if metric_range > 0
        else np.nan
    )

    return (
        float(
            prog_row[metric]
        ),
        float(
            early_row[metric]
        ),
        delta,
        normalized_delta,
    )


def plot_standardized_profiles(
    profiles,
    branch,
    differentiation,
    early_landmark,
    output_path,
):
    """
    Overlay within-profile min-max scaled benchmark metrics.
    Scaling is for visualization only.
    """
    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    df = profiles[
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
    ].sort_values(
        "parameter"
    )

    for metric in ALL_METRICS:

        ax.plot(
            df[
                "parameter"
            ],
            minmax(
                df[metric]
            ),
            linewidth=1.5,
            label=METRIC_LABELS[
                metric
            ],
        )

    ax.axvline(
        early_landmark,
        linestyle="--",
        linewidth=1.2,
        label=(
            "Frozen NEUROG3-early "
            "median landmark"
        ),
    )

    ax.set_xlabel(
        "Deposited pseudotime"
    )

    ax.set_ylabel(
        "Within-profile scaled value"
    )

    ax.set_ylim(
        -0.03,
        1.03,
    )

    ax.set_title(
        f"Benchmark profiles: "
        f"{BRANCH_LABELS[branch]}, "
        f"Differentiation {differentiation}"
    )

    ax.legend(
        fontsize=8,
    )

    ax.grid(
        alpha=0.25
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

    if not LANDMARK_FILE.exists():
        raise FileNotFoundError(
            f"Missing landmark table: "
            f"{LANDMARK_FILE}"
        )

    landmarks = pd.read_csv(
        LANDMARK_FILE
    )

    all_profiles = []

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as report:

        write_both(
            report,
            "=" * 94,
        )

        write_both(
            report,
            "GDIS-Bio Benchmark Against Conventional "
            "Early-Warning / Transition Metrics",
        )

        write_both(
            report,
            "=" * 94,
        )

        write_both(
            report,
            "\nFrozen analysis design:",
        )

        write_both(
            report,
            "  50 PCs; 400 cells/window; "
            "100-cell step; 75% overlap",
        )

        write_both(
            report,
            "  No GDIS values are recomputed.",
        )

        write_both(
            report,
            "  No windows, landmarks, or hyperparameters "
            "are changed.",
        )

        write_both(
            report,
            "\nConventional benchmark metrics:",
        )

        for metric in BASELINE_METRICS:
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
        # 1. Compute baseline profiles
        # =========================================================

        write_both(
            report,
            "\n1. COMPUTE CONVENTIONAL METRICS ON FROZEN WINDOWS",
        )

        write_both(
            report,
            "-" * 94,
        )

        for branch, differentiation in GROUPS:

            (
                _,
                trajectories,
                parameters,
            ) = load_trajectory_archive(
                branch,
                differentiation,
            )

            window_meta = (
                load_window_metadata(
                    branch,
                    differentiation,
                )
            )

            gdis = load_gdis_output(
                branch,
                differentiation,
            )

            if len(
                trajectories
            ) != len(
                window_meta
            ):
                raise ValueError(
                    "Trajectory/window metadata length mismatch."
                )

            if len(
                trajectories
            ) != len(
                gdis
            ):
                raise ValueError(
                    "Trajectory/GDIS length mismatch."
                )

            meta_parameters = (
                window_meta[
                    "parameter_median_pseudotime"
                ].to_numpy(
                    dtype=float
                )
            )

            if not np.allclose(
                parameters,
                meta_parameters,
                atol=1e-12,
                rtol=1e-10,
            ):
                raise ValueError(
                    "NPZ parameters and window metadata "
                    "do not match."
                )

            if not np.allclose(
                parameters,
                gdis[
                    "parameter"
                ].to_numpy(
                    dtype=float
                ),
                atol=1e-12,
                rtol=1e-10,
            ):
                raise ValueError(
                    "NPZ parameters and p8 GDIS output "
                    "do not match."
                )

            metric_rows = []

            for i in range(
                trajectories.shape[0]
            ):

                row = (
                    compute_window_baselines(
                        trajectories[i]
                    )
                )

                row[
                    "window_index"
                ] = i

                row[
                    "parameter"
                ] = float(
                    parameters[i]
                )

                metric_rows.append(
                    row
                )

            baseline_df = (
                pd.DataFrame(
                    metric_rows
                )
                .sort_values(
                    "window_index"
                )
                .reset_index(
                    drop=True
                )
            )

            # Attach frozen GDIS comparators.
            baseline_df[
                "gdis"
            ] = gdis[
                "gdis"
            ].to_numpy(
                dtype=float
            )

            baseline_df[
                "transition_energy"
            ] = gdis[
                "transition_energy"
            ].to_numpy(
                dtype=float
            )

            # Attach frozen biological metadata.
            for col in [
                "median_day",
                "dominant_state",
                "dominant_state_fraction",
                "pseudotime_min",
                "pseudotime_max",
                "pseudotime_span",
            ]:
                if col in window_meta.columns:
                    baseline_df[
                        col
                    ] = window_meta[
                        col
                    ].to_numpy()

            baseline_df.insert(
                0,
                "differentiation",
                differentiation,
            )

            baseline_df.insert(
                0,
                "lineage",
                BRANCH_LABELS[
                    branch
                ],
            )

            baseline_df.insert(
                0,
                "branch",
                branch,
            )

            all_profiles.append(
                baseline_df
            )

            write_both(
                report,
                f"{group_label(branch, differentiation)}: "
                f"{len(baseline_df)} windows computed",
            )

        profiles = pd.concat(
            all_profiles,
            ignore_index=True,
        )

        save_table(
            profiles,
            "01_all_metric_profiles.csv",
            index=False,
        )

        # =========================================================
        # 2. Peak localization
        # =========================================================

        write_both(
            report,
            "\n2. GLOBAL-PEAK LOCALIZATION BENCHMARK",
        )

        write_both(
            report,
            "-" * 94,
        )

        peak_rows = []

        for branch, differentiation in GROUPS:

            subset = (
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
            )

            early_landmark = get_landmark(
                landmarks,
                branch,
                differentiation,
                "neurog3_early",
            )

            for metric in ALL_METRICS:

                values = subset[
                    metric
                ].to_numpy(
                    dtype=float
                )

                pos = int(
                    np.nanargmax(
                        values
                    )
                )

                row = subset.iloc[
                    pos
                ]

                peak_parameter = float(
                    row[
                        "parameter"
                    ]
                )

                peak_rows.append(
                    {
                        "branch":
                            branch,
                        "lineage":
                            BRANCH_LABELS[
                                branch
                            ],
                        "differentiation":
                            differentiation,
                        "metric":
                            metric,
                        "metric_label":
                            METRIC_LABELS[
                                metric
                            ],
                        "peak_parameter":
                            peak_parameter,
                        "neurog3_early_landmark":
                            early_landmark,
                        "peak_minus_early_landmark":
                            (
                                peak_parameter
                                - early_landmark
                            ),
                        "absolute_peak_localization_error":
                            abs(
                                peak_parameter
                                - early_landmark
                            ),
                        "peak_median_day":
                            (
                                float(
                                    row[
                                        "median_day"
                                    ]
                                )
                                if (
                                    "median_day"
                                    in row.index
                                )
                                else np.nan
                            ),
                        "peak_dominant_state":
                            (
                                str(
                                    row[
                                        "dominant_state"
                                    ]
                                )
                                if (
                                    "dominant_state"
                                    in row.index
                                )
                                else ""
                            ),
                        "peak_value":
                            float(
                                row[
                                    metric
                                ]
                            ),
                    }
                )

        peak_table = pd.DataFrame(
            peak_rows
        )

        save_table(
            peak_table,
            "02_peak_localization_benchmark.csv",
            index=False,
        )

        write_both(
            report,
            peak_table.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 3. Progenitor -> NEUROG3-early change
        # =========================================================

        write_both(
            report,
            "\n3. PROGENITOR -> NEUROG3-EARLY METRIC CHANGE",
        )

        write_both(
            report,
            "-" * 94,
        )

        rise_rows = []

        for branch, differentiation in GROUPS:

            subset = profiles[
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
            ].sort_values(
                "parameter"
            )

            prog_landmark = get_landmark(
                landmarks,
                branch,
                differentiation,
                "prog_nkx61",
            )

            early_landmark = get_landmark(
                landmarks,
                branch,
                differentiation,
                "neurog3_early",
            )

            for metric in ALL_METRICS:

                (
                    prog_value,
                    early_value,
                    delta,
                    normalized_delta,
                ) = normalized_landmark_rise(
                    subset,
                    metric,
                    prog_landmark,
                    early_landmark,
                )

                rise_rows.append(
                    {
                        "branch":
                            branch,
                        "lineage":
                            BRANCH_LABELS[
                                branch
                            ],
                        "differentiation":
                            differentiation,
                        "metric":
                            metric,
                        "metric_label":
                            METRIC_LABELS[
                                metric
                            ],
                        "progenitor_value":
                            prog_value,
                        "neurog3_early_value":
                            early_value,
                        "early_minus_progenitor":
                            delta,
                        "normalized_early_minus_progenitor":
                            normalized_delta,
                        "increases_to_neurog3_early":
                            bool(
                                delta > 0
                            ),
                    }
                )

        rise_table = pd.DataFrame(
            rise_rows
        )

        save_table(
            rise_table,
            "03_progenitor_to_early_change.csv",
            index=False,
        )

        write_both(
            report,
            rise_table.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 4. Replicate profile agreement
        # =========================================================

        write_both(
            report,
            "\n4. CROSS-REPLICATE PROFILE AGREEMENT",
        )

        write_both(
            report,
            "-" * 94,
        )

        agreement_rows = []

        for branch in [0, 1]:

            a = profiles[
                (
                    profiles[
                        "branch"
                    ] == branch
                )
                & (
                    profiles[
                        "differentiation"
                    ] == 1
                )
            ]

            b = profiles[
                (
                    profiles[
                        "branch"
                    ] == branch
                )
                & (
                    profiles[
                        "differentiation"
                    ] == 2
                )
            ]

            for metric in ALL_METRICS:

                agreement = (
                    profile_agreement(
                        a,
                        b,
                        metric,
                    )
                )

                agreement_rows.append(
                    {
                        "branch":
                            branch,
                        "lineage":
                            BRANCH_LABELS[
                                branch
                            ],
                        "metric":
                            metric,
                        "metric_label":
                            METRIC_LABELS[
                                metric
                            ],
                        "pearson":
                            agreement[
                                "pearson"
                            ],
                        "spearman":
                            agreement[
                                "spearman"
                            ],
                        "mae":
                            agreement[
                                "mae"
                            ],
                        "shared_pseudotime_min":
                            agreement[
                                "shared_min"
                            ],
                        "shared_pseudotime_max":
                            agreement[
                                "shared_max"
                            ],
                    }
                )

        agreement_table = (
            pd.DataFrame(
                agreement_rows
            )
        )

        save_table(
            agreement_table,
            "04_cross_replicate_metric_agreement.csv",
            index=False,
        )

        write_both(
            report,
            agreement_table.round(
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
            "-" * 94,
        )

        integrated_rows = []

        for metric in ALL_METRICS:

            peak_metric = (
                peak_table[
                    peak_table[
                        "metric"
                    ] == metric
                ]
            )

            rise_metric = (
                rise_table[
                    rise_table[
                        "metric"
                    ] == metric
                ]
            )

            agreement_metric = (
                agreement_table[
                    agreement_table[
                        "metric"
                    ] == metric
                ]
            )

            integrated_rows.append(
                {
                    "metric":
                        metric,
                    "metric_label":
                        METRIC_LABELS[
                            metric
                        ],
                    "mean_absolute_peak_localization_error":
                        float(
                            peak_metric[
                                "absolute_peak_localization_error"
                            ].mean()
                        ),
                    "median_absolute_peak_localization_error":
                        float(
                            peak_metric[
                                "absolute_peak_localization_error"
                            ].median()
                        ),
                    "n_of_4_peaks_in_neurog3_early_dominant_window":
                        int(
                            (
                                peak_metric[
                                    "peak_dominant_state"
                                ]
                                ==
                                "neurog3_early"
                            ).sum()
                        ),
                    "n_of_4_increasing_from_progenitor_to_neurog3_early":
                        int(
                            rise_metric[
                                "increases_to_neurog3_early"
                            ].sum()
                        ),
                    "mean_normalized_progenitor_to_early_change":
                        float(
                            rise_metric[
                                "normalized_early_minus_progenitor"
                            ].mean()
                        ),
                    "minimum_replicate_pearson":
                        float(
                            agreement_metric[
                                "pearson"
                            ].min()
                        ),
                    "mean_replicate_pearson":
                        float(
                            agreement_metric[
                                "pearson"
                            ].mean()
                        ),
                    "minimum_replicate_spearman":
                        float(
                            agreement_metric[
                                "spearman"
                            ].min()
                        ),
                    "mean_replicate_spearman":
                        float(
                            agreement_metric[
                                "spearman"
                            ].mean()
                        ),
                }
            )

        integrated = pd.DataFrame(
            integrated_rows
        )

        integrated = integrated.sort_values(
            [
                "mean_absolute_peak_localization_error",
                "minimum_replicate_pearson",
            ],
            ascending=[
                True,
                False,
            ],
        ).reset_index(
            drop=True
        )

        save_table(
            integrated,
            "05_integrated_benchmark_summary.csv",
            index=False,
        )

        write_both(
            report,
            integrated.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 6. Branch-consensus localization
        # =========================================================

        write_both(
            report,
            "\n6. BRANCH-CONSENSUS BENCHMARK",
        )

        write_both(
            report,
            "-" * 94,
        )

        consensus_rows = []

        for branch in [0, 1]:

            a = profiles[
                (
                    profiles[
                        "branch"
                    ] == branch
                )
                & (
                    profiles[
                        "differentiation"
                    ] == 1
                )
            ].sort_values(
                "parameter"
            )

            b = profiles[
                (
                    profiles[
                        "branch"
                    ] == branch
                )
                & (
                    profiles[
                        "differentiation"
                    ] == 2
                )
            ].sort_values(
                "parameter"
            )

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
                CONSENSUS_GRID_POINTS,
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

            for metric in ALL_METRICS:

                va = np.interp(
                    grid,
                    a[
                        "parameter"
                    ],
                    a[metric],
                )

                vb = np.interp(
                    grid,
                    b[
                        "parameter"
                    ],
                    b[metric],
                )

                consensus = (
                    va + vb
                ) / 2.0

                pos = int(
                    np.nanargmax(
                        consensus
                    )
                )

                peak = float(
                    grid[pos]
                )

                consensus_rows.append(
                    {
                        "branch":
                            branch,
                        "lineage":
                            BRANCH_LABELS[
                                branch
                            ],
                        "metric":
                            metric,
                        "metric_label":
                            METRIC_LABELS[
                                metric
                            ],
                        "consensus_peak_parameter":
                            peak,
                        "mean_neurog3_early_landmark":
                            mean_early_landmark,
                        "consensus_peak_minus_early_landmark":
                            (
                                peak
                                - mean_early_landmark
                            ),
                        "absolute_consensus_peak_localization_error":
                            abs(
                                peak
                                - mean_early_landmark
                            ),
                        "consensus_peak_value":
                            float(
                                consensus[
                                    pos
                                ]
                            ),
                    }
                )

        consensus_table = (
            pd.DataFrame(
                consensus_rows
            )
        )

        save_table(
            consensus_table,
            "06_branch_consensus_peak_benchmark.csv",
            index=False,
        )

        write_both(
            report,
            consensus_table.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 7. Visualization
        # =========================================================

        write_both(
            report,
            "\n7. BENCHMARK FIGURES",
        )

        write_both(
            report,
            "-" * 94,
        )

        figure_index = 1

        for branch, differentiation in GROUPS:

            early_landmark = get_landmark(
                landmarks,
                branch,
                differentiation,
                "neurog3_early",
            )

            plot_standardized_profiles(
                profiles=profiles,
                branch=branch,
                differentiation=
                    differentiation,
                early_landmark=
                    early_landmark,
                output_path=(
                    FIGURE_DIR
                    / (
                        f"{figure_index:02d}_"
                        f"benchmark_profiles_"
                        f"{group_label(branch, differentiation)}"
                        f".png"
                    )
                ),
            )

            figure_index += 1

        # Localization-error figure.
        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        plot_df = integrated.copy()

        ax.bar(
            plot_df[
                "metric_label"
            ],
            plot_df[
                "mean_absolute_peak_localization_error"
            ],
        )

        ax.set_ylabel(
            "Mean absolute peak distance from "
            "NEUROG3-early median landmark"
        )

        ax.set_title(
            "Transition-localization benchmark"
        )

        ax.tick_params(
            axis="x",
            rotation=35,
        )

        ax.grid(
            axis="y",
            alpha=0.25,
        )

        fig.tight_layout()

        fig.savefig(
            FIGURE_DIR
            / "05_peak_localization_error_summary.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        write_both(
            report,
            "Saved standardized profile figures and "
            "localization-error summary.",
        )

        # =========================================================
        # 8. Final status and interpretation guardrails
        # =========================================================

        write_both(
            report,
            "\n8. FINAL STATUS",
        )

        write_both(
            report,
            "-" * 94,
        )

        numeric_columns = [
            "total_variance",
            "mean_lag1_autocorrelation",
            "gaussian_differential_entropy",
            "mean_step_distance",
            "gdis",
            "transition_energy",
        ]

        finite_pass = bool(
            np.isfinite(
                profiles[
                    numeric_columns
                ].to_numpy(
                    dtype=float
                )
            ).all()
        )

        alignment_pass = bool(
            profiles.groupby(
                [
                    "branch",
                    "differentiation",
                ]
            ).size().shape[0]
            == 4
        )

        if finite_pass:
            write_both(
                report,
                "[PASS] All benchmark metric values are finite.",
            )
        else:
            write_both(
                report,
                "[CHECK] Non-finite benchmark metric values detected.",
            )

        if alignment_pass:
            write_both(
                report,
                "[PASS] All four frozen branch x differentiation "
                "profiles were benchmarked.",
            )
        else:
            write_both(
                report,
                "[CHECK] Group-profile alignment requires review.",
            )

        if (
            finite_pass
            and alignment_pass
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
            "  - Smaller peak-localization error means closer "
            "alignment to the frozen median NEUROG3-early landmark.",
        )

        write_both(
            report,
            "  - A conventional metric is not considered superior "
            "or inferior from any single criterion alone.",
        )

        write_both(
            report,
            "  - Replicate reproducibility, biological localization, "
            "and progenitor-to-early change should be interpreted together.",
        )

        write_both(
            report,
            "  - The NEUROG3-early median is a frozen state landmark, "
            "not a proven true transition-onset time.",
        )

        write_both(
            report,
            "  - No benchmark result is used here to retune GDIS.",
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

    print("\n" + "=" * 94)
    print("p10_conventional_ews_benchmark.py completed.")
    print("=" * 94)
    print(f"Report : {REPORT_FILE}")
    print(f"Tables : {TABLE_DIR}")
    print(f"Figures: {FIGURE_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

