#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================
"""
p8_gdis_primary_analysis.py

First GDIS-Bio calculation for GSE114412.

This script uses the frozen trajectory inputs created by p7 and the
reference pyGDIS implementation WITHOUT changing the GDIS formulation.

Primary analysis
----------------
State space:
    50 PCs

Windowing:
    400 cells/window
    100-cell step
    75% overlap

Groups:
    Branch 0 / Differentiation 1  -> SC-EC replicate 1
    Branch 0 / Differentiation 2  -> SC-EC replicate 2
    Branch 1 / Differentiation 1  -> SC-beta replicate 1
    Branch 1 / Differentiation 2  -> SC-beta replicate 2

GDIS mode
---------
Data-only mode:
    no analytical Jacobian is supplied.

No biological critical value is supplied to pyGDIS.
The reference implementation therefore resolves the transition-localization
center from the maximum transition-energy estimate. This choice is recorded
and then compared AFTERWARD with biological landmarks that were frozen in p7.

This avoids defining the biological transition from the GDIS result itself.

Pre-specified sensitivity analyses
----------------------------------
A. Window sensitivity at 50 PCs:
       300/75
       400/100   [PRIMARY]
       500/125

B. PCA dimensionality sensitivity using the PRIMARY 400/100 windows:
       5, 10, 20, 30, 50 PCs

C. Reference transition-weight sensitivity:
       0.00, 0.18, 0.25, 0.50, 0.75, 1.00

IMPORTANT INTERPRETATION
------------------------
These trajectories are pseudotime-ordered sequences of independent cells.
They are NOT longitudinal recordings of the same cell, and pseudotime is NOT
physical time.

This script does NOT use universal GDIS thresholds and does NOT label windows
as "stable", "unstable", or "chaotic" from score magnitude alone.

Requirements
------------
numpy
pandas
scipy
matplotlib
pygdis >= 1.0.0

Install pyGDIS if necessary:
    python -m pip install pygdis
"""

from pathlib import Path
import sys
import platform
from dataclasses import asdict
from importlib.metadata import version, PackageNotFoundError

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

try:
    from gdis import (
        GDIS,
        GDISConfig,
        transition_weight_sensitivity,
    )
except ImportError as exc:
    raise SystemExit(
        "\nERROR: pyGDIS is not installed in this Python environment.\n"
        "Install the reference package with:\n\n"
        "    python -m pip install pygdis\n\n"
        "Then rerun this script.\n"
    ) from exc


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

LANDMARK_FILE = (
    P7_TABLE_DIR
    / "07_primary_state_landmark_window_mapping.csv"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
    / "p8_gdis_primary_analysis"
)

TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR = RESULTS_DIR / "figures"
DATA_DIR = RESULTS_DIR / "data"

REPORT_FILE = (
    RESULTS_DIR
    / "p8_gdis_primary_analysis_report.txt"
)

PRIMARY_CONFIG = "primary_w400_s100"
WINDOW_CONFIGS = [
    "sensitivity_w300_s75",
    "primary_w400_s100",
    "sensitivity_w500_s125",
]

PRIMARY_DIM = 50
DIMENSION_SENSITIVITY = [5, 10, 20, 30, 50]

TRANSITION_WEIGHTS = (
    0.0,
    0.18,
    0.25,
    0.50,
    0.75,
    1.0,
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

EXPECTED_PY_GDIS_VERSION = "1.0.0"

# The line above intentionally cannot contain a space in a Python variable
# name. It is replaced immediately below during file generation.


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def write_both(report, text=""):
    """Print and write text."""
    print(text)
    report.write(text + "\n")
    report.flush()


def save_table(df, filename, index=True):
    """Save a dataframe as CSV."""
    path = TABLE_DIR / filename
    df.to_csv(path, index=index)
    return path


def group_label(branch, differentiation):
    """Compact group identifier."""
    return f"branch{int(branch)}_diff{int(differentiation)}"


def safe_spearman(x, y):
    """Spearman correlation with finite-value filtering."""
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


def load_npz(config_name, branch, differentiation):
    """Load one p7 trajectory archive."""
    label = group_label(
        branch,
        differentiation,
    )

    path = (
        P7_DATA_DIR
        / config_name
        / f"{label}.npz"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing p7 trajectory archive: {path}"
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

    return path, trajectories, parameters


def load_window_metadata(
    config_name,
    branch,
    differentiation,
):
    """Load matching p7 window metadata."""
    label = group_label(
        branch,
        differentiation,
    )

    path = (
        P7_DATA_DIR
        / config_name
        / f"{label}_windows.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing p7 window metadata: {path}"
        )

    df = pd.read_csv(path)

    return path, df


def validate_input_pair(
    trajectories,
    parameters,
    metadata,
    n_dim,
):
    """Validate one trajectory family before running pyGDIS."""
    if trajectories.ndim != 3:
        raise ValueError(
            f"Expected 3D trajectory array; got "
            f"{trajectories.shape}"
        )

    if trajectories.shape[0] != len(parameters):
        raise ValueError(
            "Number of trajectories does not match "
            "number of parameters."
        )

    if trajectories.shape[0] != len(metadata):
        raise ValueError(
            "Number of trajectories does not match "
            "window metadata rows."
        )

    if trajectories.shape[2] < n_dim:
        raise ValueError(
            f"Requested {n_dim} PCs, but trajectory "
            f"archive contains only {trajectories.shape[2]}."
        )

    if not np.isfinite(
        trajectories[:, :, :n_dim]
    ).all():
        raise ValueError(
            "Non-finite trajectory coordinates detected."
        )

    if not np.isfinite(parameters).all():
        raise ValueError(
            "Non-finite parameters detected."
        )

    if np.any(np.diff(parameters) <= 0):
        raise ValueError(
            "pyGDIS requires unique strictly increasing parameters."
        )

    meta_parameters = metadata[
        "parameter_median_pseudotime"
    ].to_numpy(dtype=float)

    if not np.allclose(
        parameters,
        meta_parameters,
        atol=1e-12,
        rtol=1e-10,
    ):
        raise ValueError(
            "NPZ parameters do not exactly match "
            "p7 window metadata."
        )


def run_gdis(
    trajectories,
    parameters,
    n_dim,
):
    """
    Run the unchanged pyGDIS reference formulation in data-only mode.
    """
    trajectories_for_gdis = [
        np.asarray(
            trajectory[:, :n_dim],
            dtype=np.float64,
        )
        for trajectory in trajectories
    ]

    estimator = GDIS()

    result = estimator.fit_transform(
        trajectories_for_gdis,
        parameters,
    )

    return result


def attach_window_metadata(result_df, window_df):
    """
    Attach p7 biological/window metadata to pyGDIS output.

    pyGDIS sorts on parameter, so p7 metadata is sorted the same way.
    """
    result_df = (
        result_df
        .sort_values("parameter")
        .reset_index(drop=True)
    )

    window_sorted = (
        window_df
        .sort_values(
            "parameter_median_pseudotime"
        )
        .reset_index(drop=True)
    )

    if len(result_df) != len(window_sorted):
        raise ValueError(
            "GDIS output and window metadata have "
            "different lengths."
        )

    if not np.allclose(
        result_df["parameter"],
        window_sorted[
            "parameter_median_pseudotime"
        ],
        atol=1e-12,
        rtol=1e-10,
    ):
        raise ValueError(
            "GDIS parameters and window metadata "
            "do not align."
        )

    metadata_columns = [
        c
        for c in window_sorted.columns
        if c
        not in {
            "parameter_median_pseudotime",
        }
    ]

    combined = pd.concat(
        [
            result_df,
            window_sorted[
                metadata_columns
            ],
        ],
        axis=1,
    )

    return combined


def nearest_landmark(
    value,
    landmark_subset,
):
    """Find nearest frozen biological state landmark."""
    if landmark_subset.empty:
        return {
            "state": None,
            "state_median_pseudotime": np.nan,
            "distance": np.nan,
        }

    distances = np.abs(
        landmark_subset[
            "state_median_pseudotime"
        ].to_numpy(dtype=float)
        - float(value)
    )

    pos = int(
        np.argmin(distances)
    )

    row = landmark_subset.iloc[pos]

    return {
        "state": str(row["state"]),
        "state_median_pseudotime":
            float(
                row[
                    "state_median_pseudotime"
                ]
            ),
        "distance":
            float(distances[pos]),
    }


def summarize_primary_group(
    result,
    combined,
    branch,
    differentiation,
    landmarks,
):
    """Create one summary row for a primary GDIS run."""
    gdis_values = combined[
        "gdis"
    ].to_numpy(dtype=float)

    sustained = combined[
        "sustained_instability"
    ].to_numpy(dtype=float)

    transition_energy_values = combined[
        "transition_energy"
    ].to_numpy(dtype=float)

    transition_instability = combined[
        "transition_instability"
    ].to_numpy(dtype=float)

    peak_gdis_pos = int(
        np.argmax(gdis_values)
    )

    peak_sustained_pos = int(
        np.argmax(sustained)
    )

    peak_energy_pos = int(
        np.argmax(
            transition_energy_values
        )
    )

    peak_transition_pos = int(
        np.argmax(
            transition_instability
        )
    )

    critical_value = float(
        result.metadata[
            "critical_value"
        ]
    )

    lm = landmarks[
        (landmarks["branch"] == branch)
        & (
            landmarks[
                "differentiation"
            ] == differentiation
        )
    ].copy()

    peak_gdis_landmark = (
        nearest_landmark(
            combined.iloc[
                peak_gdis_pos
            ]["parameter"],
            lm,
        )
    )

    critical_landmark = (
        nearest_landmark(
            critical_value,
            lm,
        )
    )

    return {
        "branch": int(branch),
        "lineage": BRANCH_LABELS[
            int(branch)
        ],
        "differentiation":
            int(differentiation),
        "n_windows": int(
            len(combined)
        ),
        "gdis_min":
            float(np.min(gdis_values)),
        "gdis_median":
            float(np.median(gdis_values)),
        "gdis_mean":
            float(np.mean(gdis_values)),
        "gdis_max":
            float(np.max(gdis_values)),
        "gdis_peak_window_index":
            int(
                combined.iloc[
                    peak_gdis_pos
                ]["window_index"]
            ),
        "gdis_peak_parameter":
            float(
                combined.iloc[
                    peak_gdis_pos
                ]["parameter"]
            ),
        "gdis_peak_median_day":
            float(
                combined.iloc[
                    peak_gdis_pos
                ]["median_day"]
            ),
        "gdis_peak_dominant_state":
            str(
                combined.iloc[
                    peak_gdis_pos
                ]["dominant_state"]
            ),
        "gdis_peak_nearest_landmark":
            peak_gdis_landmark[
                "state"
            ],
        "gdis_peak_landmark_distance":
            peak_gdis_landmark[
                "distance"
            ],
        "sustained_peak_parameter":
            float(
                combined.iloc[
                    peak_sustained_pos
                ]["parameter"]
            ),
        "transition_energy_peak_parameter":
            float(
                combined.iloc[
                    peak_energy_pos
                ]["parameter"]
            ),
        "transition_instability_peak_parameter":
            float(
                combined.iloc[
                    peak_transition_pos
                ]["parameter"]
            ),
        "resolved_critical_value":
            critical_value,
        "critical_value_source":
            str(
                result.metadata[
                    "critical_value_source"
                ]
            ),
        "critical_nearest_landmark":
            critical_landmark[
                "state"
            ],
        "critical_landmark_distance":
            critical_landmark[
                "distance"
            ],
        "transition_weight":
            float(
                result.metadata[
                    "transition_weight"
                ]
            ),
        "spearman_gdis_vs_pseudotime":
            safe_spearman(
                combined["parameter"],
                combined["gdis"],
            ),
    }


def profile_agreement(
    a,
    b,
    value_column,
    n_grid=200,
):
    """
    Compare two profiles on their shared actual-pseudotime range.
    """
    a = (
        a.sort_values("parameter")
        .drop_duplicates(
            "parameter",
            keep="first",
        )
    )

    b = (
        b.sort_values("parameter")
        .drop_duplicates(
            "parameter",
            keep="first",
        )
    )

    lower = max(
        float(a["parameter"].min()),
        float(b["parameter"].min()),
    )

    upper = min(
        float(a["parameter"].max()),
        float(b["parameter"].max()),
    )

    if upper <= lower:
        return {
            "pearson": np.nan,
            "spearman": np.nan,
            "mae": np.nan,
            "shared_min": lower,
            "shared_max": upper,
        }

    grid = np.linspace(
        lower,
        upper,
        n_grid,
    )

    va = np.interp(
        grid,
        a["parameter"],
        a[value_column],
    )

    vb = np.interp(
        grid,
        b["parameter"],
        b[value_column],
    )

    return {
        "pearson":
            safe_pearson(va, vb),
        "spearman":
            safe_spearman(va, vb),
        "mae":
            float(
                np.mean(
                    np.abs(
                        va - vb
                    )
                )
            ),
        "shared_min":
            lower,
        "shared_max":
            upper,
    }


def compare_to_primary_profile(
    primary_df,
    alternative_df,
):
    """Compare sensitivity profile to frozen primary profile."""
    agreement = profile_agreement(
        primary_df,
        alternative_df,
        "gdis",
    )

    primary_peak = float(
        primary_df.loc[
            primary_df[
                "gdis"
            ].idxmax(),
            "parameter",
        ]
    )

    alternative_peak = float(
        alternative_df.loc[
            alternative_df[
                "gdis"
            ].idxmax(),
            "parameter",
        ]
    )

    agreement[
        "primary_peak_parameter"
    ] = primary_peak

    agreement[
        "alternative_peak_parameter"
    ] = alternative_peak

    agreement[
        "absolute_peak_parameter_shift"
    ] = abs(
        primary_peak
        - alternative_peak
    )

    return agreement


def plot_branch_replicates(
    result_tables,
    branch,
    output_path,
):
    """Overlay primary GDIS profiles for two differentiations."""
    fig, ax = plt.subplots(
        figsize=(9, 5.8)
    )

    for differentiation in [1, 2]:

        key = (
            PRIMARY_CONFIG,
            PRIMARY_DIM,
            branch,
            differentiation,
        )

        table = result_tables[key]

        ax.plot(
            table["parameter"],
            table["gdis"],
            marker="o",
            markersize=3,
            linewidth=1.7,
            label=(
                f"Differentiation "
                f"{differentiation}"
            ),
        )

    ax.set_xlabel(
        "Median deposited pseudotime"
    )

    ax.set_ylabel(
        "GDIS"
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.set_title(
        f"Primary GDIS profile: "
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


def plot_group_components(
    table,
    branch,
    differentiation,
    output_path,
):
    """Plot key pyGDIS output channels for one primary group."""
    fig, ax = plt.subplots(
        figsize=(9, 5.8)
    )

    ax.plot(
        table["parameter"],
        table["gdis"],
        linewidth=1.8,
        label="GDIS",
    )

    ax.plot(
        table["parameter"],
        table[
            "sustained_instability"
        ],
        linewidth=1.5,
        label="Sustained instability",
    )

    ax.plot(
        table["parameter"],
        table[
            "transition_instability"
        ],
        linewidth=1.5,
        label="Transition instability",
    )

    ax.set_xlabel(
        "Median deposited pseudotime"
    )

    ax.set_ylabel(
        "Bounded score"
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.set_title(
        f"pyGDIS components: "
        f"{BRANCH_LABELS[branch]}, "
        f"Differentiation {differentiation}"
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

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Package version
    # -------------------------------------------------------------

    try:
        pygdis_version = version(
            "pygdis"
        )
    except PackageNotFoundError:
        pygdis_version = "unknown"

    landmarks = pd.read_csv(
        LANDMARK_FILE
    )

    required_landmark_columns = [
        "branch",
        "differentiation",
        "state",
        "state_median_pseudotime",
    ]

    missing = [
        c
        for c in required_landmark_columns
        if c not in landmarks.columns
    ]

    if missing:
        raise ValueError(
            "Landmark table is missing: "
            + ", ".join(missing)
        )

    # Reference configuration object for provenance.
    reference_config = GDISConfig()
    reference_config_dict = asdict(
        reference_config
    )

    result_tables = {}
    result_objects = {}

    primary_summary_rows = []
    computational_check_rows = []

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
            "GDIS-Bio Primary Analysis",
        )

        write_both(
            report,
            "Dataset: GSE114412 validated endocrine trajectory",
        )

        write_both(
            report,
            "=" * 92,
        )

        write_both(
            report,
            "\nSCIENTIFIC INTERPRETATION:",
        )

        write_both(
            report,
            "The input trajectories are pseudotime-ordered "
            "sequences of independent single cells, not "
            "longitudinal observations of the same cell.",
        )

        write_both(
            report,
            "No universal GDIS magnitude threshold is used.",
        )

        # =========================================================
        # 1. pyGDIS reference provenance
        # =========================================================

        write_both(
            report,
            "\n1. pyGDIS REFERENCE IMPLEMENTATION",
        )

        write_both(
            report,
            "-" * 92,
        )

        write_both(
            report,
            f"Python: {platform.python_version()}",
        )

        write_both(
            report,
            f"pyGDIS package version: "
            f"{pygdis_version}",
        )

        write_both(
            report,
            "Mode: data-only "
            "(no analytical Jacobian)",
        )

        write_both(
            report,
            "Biological critical value supplied to pyGDIS: NO",
        )

        write_both(
            report,
            "Primary state space: 50 PCs",
        )

        write_both(
            report,
            "Primary windowing: "
            "400 cells/window, 100-cell step",
        )

        write_both(
            report,
            "\nReference GDIS configuration:",
        )

        for key, value in (
            reference_config_dict.items()
        ):
            write_both(
                report,
                f"  {key}: {value}",
            )

        # =========================================================
        # 2. Primary GDIS calculation
        # =========================================================

        write_both(
            report,
            "\n2. PRIMARY GDIS CALCULATION",
        )

        write_both(
            report,
            "-" * 92,
        )

        for branch, differentiation in GROUPS:

            label = group_label(
                branch,
                differentiation,
            )

            (
                npz_path,
                trajectories,
                parameters,
            ) = load_npz(
                PRIMARY_CONFIG,
                branch,
                differentiation,
            )

            (
                metadata_path,
                window_metadata,
            ) = load_window_metadata(
                PRIMARY_CONFIG,
                branch,
                differentiation,
            )

            validate_input_pair(
                trajectories,
                parameters,
                window_metadata,
                PRIMARY_DIM,
            )

            result = run_gdis(
                trajectories,
                parameters,
                PRIMARY_DIM,
            )

            result_df = (
                result.to_dataframe()
            )

            combined = (
                attach_window_metadata(
                    result_df,
                    window_metadata,
                )
            )

            # p7 window metadata already contains branch and
            # differentiation. Verify those values rather than attempting
            # to insert duplicate columns.
            if "branch" in combined.columns:
                observed_branches = set(
                    pd.to_numeric(
                        combined["branch"],
                        errors="raise",
                    ).astype(int).unique()
                )

                if observed_branches != {int(branch)}:
                    raise ValueError(
                        f"Branch metadata mismatch for "
                        f"{group_label(branch, differentiation)}: "
                        f"{observed_branches}"
                    )
            else:
                combined.insert(
                    0,
                    "branch",
                    int(branch),
                )

            if "differentiation" in combined.columns:
                observed_differentiations = set(
                    pd.to_numeric(
                        combined["differentiation"],
                        errors="raise",
                    ).astype(int).unique()
                )

                if observed_differentiations != {
                    int(differentiation)
                }:
                    raise ValueError(
                        f"Differentiation metadata mismatch for "
                        f"{group_label(branch, differentiation)}: "
                        f"{observed_differentiations}"
                    )
            else:
                combined.insert(
                    0,
                    "differentiation",
                    int(differentiation),
                )

            # These analysis-provenance columns are not present in p7
            # metadata and are safe to add.
            if "analysis_config" not in combined.columns:
                combined.insert(
                    0,
                    "analysis_config",
                    PRIMARY_CONFIG,
                )
            else:
                combined["analysis_config"] = PRIMARY_CONFIG

            if "analysis_dimension" not in combined.columns:
                combined.insert(
                    1,
                    "analysis_dimension",
                    PRIMARY_DIM,
                )
            else:
                combined["analysis_dimension"] = PRIMARY_DIM

            result_tables[
                (
                    PRIMARY_CONFIG,
                    PRIMARY_DIM,
                    branch,
                    differentiation,
                )
            ] = combined

            result_objects[
                (
                    branch,
                    differentiation,
                )
            ] = result

            output_path = (
                DATA_DIR
                / f"primary_{label}_gdis.csv"
            )

            combined.to_csv(
                output_path,
                index=False,
            )

            summary_row = (
                summarize_primary_group(
                    result=result,
                    combined=combined,
                    branch=branch,
                    differentiation=
                        differentiation,
                    landmarks=landmarks,
                )
            )

            primary_summary_rows.append(
                summary_row
            )

            # Computational validity only.
            gdis_values = combined[
                "gdis"
            ].to_numpy(dtype=float)

            all_numeric = combined.select_dtypes(
                include=[np.number]
            ).to_numpy(dtype=float)

            check_row = {
                "branch": branch,
                "differentiation":
                    differentiation,
                "n_windows":
                    len(combined),
                "finite_numeric_outputs":
                    bool(
                        np.isfinite(
                            all_numeric
                        ).all()
                    ),
                "gdis_bounded_0_1":
                    bool(
                        np.all(
                            gdis_values >= 0.0
                        )
                        and np.all(
                            gdis_values < 1.0
                        )
                    ),
                "parameters_strictly_increasing":
                    bool(
                        np.all(
                            np.diff(
                                combined[
                                    "parameter"
                                ]
                            ) > 0
                        )
                    ),
                "critical_source_data_driven":
                    (
                        result.metadata[
                            "critical_value_source"
                        ]
                        ==
                        "data_driven_transition_energy_peak"
                    ),
            }

            computational_check_rows.append(
                check_row
            )

            write_both(
                report,
                f"{label}: "
                f"{len(combined)} windows | "
                f"GDIS range "
                f"{combined['gdis'].min():.6f}"
                f"-{combined['gdis'].max():.6f} | "
                f"peak p="
                f"{summary_row['gdis_peak_parameter']:.6f} | "
                f"transition center="
                f"{summary_row['resolved_critical_value']:.6f}",
            )

        primary_summary = pd.DataFrame(
            primary_summary_rows
        )

        save_table(
            primary_summary,
            "01_primary_gdis_summary.csv",
            index=False,
        )

        write_both(
            report,
            "\nPrimary summary:\n"
            + primary_summary.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 3. Frozen biological-landmark comparison
        # =========================================================

        write_both(
            report,
            "\n3. GDIS VALUES AT PRE-FROZEN BIOLOGICAL LANDMARKS",
        )

        write_both(
            report,
            "-" * 92,
        )

        landmark_rows = []

        for branch, differentiation in GROUPS:

            table = result_tables[
                (
                    PRIMARY_CONFIG,
                    PRIMARY_DIM,
                    branch,
                    differentiation,
                )
            ]

            lm = landmarks[
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
            ]

            for _, landmark in (
                lm.iterrows()
            ):

                target = float(
                    landmark[
                        "state_median_pseudotime"
                    ]
                )

                distances = np.abs(
                    table[
                        "parameter"
                    ].to_numpy(dtype=float)
                    - target
                )

                pos = int(
                    np.argmin(distances)
                )

                row = table.iloc[pos]

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
                        "state":
                            landmark[
                                "state"
                            ],
                        "state_median_pseudotime":
                            target,
                        "nearest_window_parameter":
                            float(
                                row[
                                    "parameter"
                                ]
                            ),
                        "absolute_parameter_difference":
                            float(
                                distances[
                                    pos
                                ]
                            ),
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

        landmark_gdis = pd.DataFrame(
            landmark_rows
        )

        save_table(
            landmark_gdis,
            "02_gdis_at_frozen_biological_landmarks.csv",
            index=False,
        )

        write_both(
            report,
            landmark_gdis.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 4. Replicate agreement
        # =========================================================

        write_both(
            report,
            "\n4. PRIMARY GDIS CROSS-REPLICATE AGREEMENT",
        )

        write_both(
            report,
            "-" * 92,
        )

        replicate_rows = []

        for branch in [0, 1]:

            a = result_tables[
                (
                    PRIMARY_CONFIG,
                    PRIMARY_DIM,
                    branch,
                    1,
                )
            ]

            b = result_tables[
                (
                    PRIMARY_CONFIG,
                    PRIMARY_DIM,
                    branch,
                    2,
                )
            ]

            for metric in [
                "gdis",
                "sustained_instability",
                "transition_instability",
                "transition_energy",
            ]:

                agreement = (
                    profile_agreement(
                        a,
                        b,
                        metric,
                    )
                )

                replicate_rows.append(
                    {
                        "branch":
                            branch,
                        "lineage":
                            BRANCH_LABELS[
                                branch
                            ],
                        "metric":
                            metric,
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

        replicate_agreement = (
            pd.DataFrame(
                replicate_rows
            )
        )

        save_table(
            replicate_agreement,
            "03_primary_cross_replicate_profile_agreement.csv",
            index=False,
        )

        write_both(
            report,
            replicate_agreement.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 5. Window sensitivity
        # =========================================================

        write_both(
            report,
            "\n5. PRE-SPECIFIED WINDOW-SIZE SENSITIVITY",
        )

        write_both(
            report,
            "-" * 92,
        )

        window_sensitivity_rows = []

        for config_name in WINDOW_CONFIGS:

            for branch, differentiation in GROUPS:

                (
                    _,
                    trajectories,
                    parameters,
                ) = load_npz(
                    config_name,
                    branch,
                    differentiation,
                )

                (
                    _,
                    metadata,
                ) = load_window_metadata(
                    config_name,
                    branch,
                    differentiation,
                )

                validate_input_pair(
                    trajectories,
                    parameters,
                    metadata,
                    PRIMARY_DIM,
                )

                result = run_gdis(
                    trajectories,
                    parameters,
                    PRIMARY_DIM,
                )

                table = attach_window_metadata(
                    result.to_dataframe(),
                    metadata,
                )

                result_tables[
                    (
                        config_name,
                        PRIMARY_DIM,
                        branch,
                        differentiation,
                    )
                ] = table

                primary = result_tables[
                    (
                        PRIMARY_CONFIG,
                        PRIMARY_DIM,
                        branch,
                        differentiation,
                    )
                ]

                agreement = (
                    compare_to_primary_profile(
                        primary,
                        table,
                    )
                )

                window_sensitivity_rows.append(
                    {
                        "config":
                            config_name,
                        "branch":
                            branch,
                        "lineage":
                            BRANCH_LABELS[
                                branch
                            ],
                        "differentiation":
                            differentiation,
                        "n_windows":
                            len(table),
                        "profile_pearson_vs_primary":
                            agreement[
                                "pearson"
                            ],
                        "profile_spearman_vs_primary":
                            agreement[
                                "spearman"
                            ],
                        "profile_mae_vs_primary":
                            agreement[
                                "mae"
                            ],
                        "primary_peak_parameter":
                            agreement[
                                "primary_peak_parameter"
                            ],
                        "alternative_peak_parameter":
                            agreement[
                                "alternative_peak_parameter"
                            ],
                        "absolute_peak_parameter_shift":
                            agreement[
                                "absolute_peak_parameter_shift"
                            ],
                    }
                )

        window_sensitivity = pd.DataFrame(
            window_sensitivity_rows
        )

        save_table(
            window_sensitivity,
            "04_window_size_sensitivity.csv",
            index=False,
        )

        write_both(
            report,
            window_sensitivity.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 6. PCA dimensionality sensitivity
        # =========================================================

        write_both(
            report,
            "\n6. PRE-SPECIFIED PCA-DIMENSION SENSITIVITY",
        )

        write_both(
            report,
            "-" * 92,
        )

        dimension_rows = []

        for n_dim in DIMENSION_SENSITIVITY:

            for branch, differentiation in GROUPS:

                (
                    _,
                    trajectories,
                    parameters,
                ) = load_npz(
                    PRIMARY_CONFIG,
                    branch,
                    differentiation,
                )

                (
                    _,
                    metadata,
                ) = load_window_metadata(
                    PRIMARY_CONFIG,
                    branch,
                    differentiation,
                )

                validate_input_pair(
                    trajectories,
                    parameters,
                    metadata,
                    n_dim,
                )

                result = run_gdis(
                    trajectories,
                    parameters,
                    n_dim,
                )

                table = attach_window_metadata(
                    result.to_dataframe(),
                    metadata,
                )

                result_tables[
                    (
                        PRIMARY_CONFIG,
                        n_dim,
                        branch,
                        differentiation,
                    )
                ] = table

                primary = result_tables[
                    (
                        PRIMARY_CONFIG,
                        PRIMARY_DIM,
                        branch,
                        differentiation,
                    )
                ]

                agreement = (
                    compare_to_primary_profile(
                        primary,
                        table,
                    )
                )

                dimension_rows.append(
                    {
                        "dimension":
                            n_dim,
                        "branch":
                            branch,
                        "lineage":
                            BRANCH_LABELS[
                                branch
                            ],
                        "differentiation":
                            differentiation,
                        "profile_pearson_vs_50PC":
                            agreement[
                                "pearson"
                            ],
                        "profile_spearman_vs_50PC":
                            agreement[
                                "spearman"
                            ],
                        "profile_mae_vs_50PC":
                            agreement[
                                "mae"
                            ],
                        "primary_50PC_peak_parameter":
                            agreement[
                                "primary_peak_parameter"
                            ],
                        "dimension_peak_parameter":
                            agreement[
                                "alternative_peak_parameter"
                            ],
                        "absolute_peak_parameter_shift":
                            agreement[
                                "absolute_peak_parameter_shift"
                            ],
                    }
                )

        dimension_sensitivity = (
            pd.DataFrame(
                dimension_rows
            )
        )

        save_table(
            dimension_sensitivity,
            "05_pca_dimension_sensitivity.csv",
            index=False,
        )

        write_both(
            report,
            dimension_sensitivity.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 7. Transition-weight sensitivity
        # =========================================================

        write_both(
            report,
            "\n7. REFERENCE TRANSITION-WEIGHT SENSITIVITY",
        )

        write_both(
            report,
            "-" * 92,
        )

        transition_weight_tables = []

        for branch, differentiation in GROUPS:

            result = result_objects[
                (
                    branch,
                    differentiation,
                )
            ]

            sensitivity = (
                transition_weight_sensitivity(
                    result,
                    weights=
                        TRANSITION_WEIGHTS,
                )
            )

            sensitivity.insert(
                0,
                "differentiation",
                differentiation,
            )

            sensitivity.insert(
                0,
                "branch",
                branch,
            )

            sensitivity.insert(
                1,
                "lineage",
                BRANCH_LABELS[
                    branch
                ],
            )

            transition_weight_tables.append(
                sensitivity
            )

        transition_weight_table = (
            pd.concat(
                transition_weight_tables,
                ignore_index=True,
            )
        )

        save_table(
            transition_weight_table,
            "06_transition_weight_sensitivity.csv",
            index=False,
        )

        transition_weight_summary = (
            transition_weight_table
            .groupby(
                [
                    "branch",
                    "lineage",
                    "differentiation",
                    "transition_weight",
                ],
                observed=True,
            )
            .agg(
                max_gdis=(
                    "gdis",
                    "max",
                ),
                mean_gdis=(
                    "gdis",
                    "mean",
                ),
                median_gdis=(
                    "gdis",
                    "median",
                ),
            )
            .reset_index()
        )

        save_table(
            transition_weight_summary,
            "07_transition_weight_summary.csv",
            index=False,
        )

        write_both(
            report,
            transition_weight_summary.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 8. Computational qualification
        # =========================================================

        write_both(
            report,
            "\n8. COMPUTATIONAL QUALIFICATION",
        )

        write_both(
            report,
            "-" * 92,
        )

        computational_checks = (
            pd.DataFrame(
                computational_check_rows
            )
        )

        save_table(
            computational_checks,
            "08_primary_computational_checks.csv",
            index=False,
        )

        write_both(
            report,
            computational_checks.to_string(
                index=False
            ),
        )

        check_columns = [
            "finite_numeric_outputs",
            "gdis_bounded_0_1",
            "parameters_strictly_increasing",
            "critical_source_data_driven",
        ]

        all_computational_checks_pass = bool(
            computational_checks[
                check_columns
            ].all().all()
        )

        # =========================================================
        # 9. Figures
        # =========================================================

        write_both(
            report,
            "\n9. PRIMARY FIGURES",
        )

        write_both(
            report,
            "-" * 92,
        )

        plot_branch_replicates(
            result_tables,
            branch=0,
            output_path=(
                FIGURE_DIR
                / "01_primary_gdis_sc_ec_replicates.png"
            ),
        )

        plot_branch_replicates(
            result_tables,
            branch=1,
            output_path=(
                FIGURE_DIR
                / "02_primary_gdis_sc_beta_replicates.png"
            ),
        )

        figure_counter = 3

        for branch, differentiation in GROUPS:

            table = result_tables[
                (
                    PRIMARY_CONFIG,
                    PRIMARY_DIM,
                    branch,
                    differentiation,
                )
            ]

            plot_group_components(
                table=table,
                branch=branch,
                differentiation=
                    differentiation,
                output_path=(
                    FIGURE_DIR
                    / (
                        f"{figure_counter:02d}_"
                        f"components_"
                        f"{group_label(branch, differentiation)}"
                        f".png"
                    )
                ),
            )

            figure_counter += 1

        write_both(
            report,
            "Saved replicate GDIS profiles and "
            "per-group component profiles.",
        )

        # =========================================================
        # 10. Final status
        # =========================================================

        write_both(
            report,
            "\n10. FINAL STATUS",
        )

        write_both(
            report,
            "-" * 92,
        )

        if all_computational_checks_pass:

            write_both(
                report,
                "FINAL COMPUTATIONAL STATUS: PASS",
            )

            write_both(
                report,
                "The first GDIS-Bio calculation completed using "
                "the frozen primary inputs and the unchanged "
                "reference pyGDIS formulation.",
            )

        else:

            write_both(
                report,
                "FINAL COMPUTATIONAL STATUS: REVIEW REQUIRED",
            )

        write_both(
            report,
            "\nIMPORTANT:",
        )

        write_both(
            report,
            "This script does not declare a biological transition "
            "merely because GDIS is high or maximal.",
        )

        write_both(
            report,
            "Biological interpretation must compare the GDIS, "
            "sustained-instability, and transition-energy profiles "
            "with the pre-frozen experimental-day and cell-state "
            "landmarks.",
        )

        write_both(
            report,
            "No parameter, dimensionality, window size, transition "
            "weight, or biological landmark was optimized based on "
            "the observed primary GDIS curve.",
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

        write_both(
            report,
            f"Data:    {DATA_DIR}",
        )

    print("\n" + "=" * 92)
    print("p8_gdis_primary_analysis.py completed.")
    print("=" * 92)
    print(f"Report : {REPORT_FILE}")
    print(f"Tables : {TABLE_DIR}")
    print(f"Figures: {FIGURE_DIR}")
    print(f"Data   : {DATA_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

