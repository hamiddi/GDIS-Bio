#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================
"""
p18_GSE175634_primary_external_gdis.py

Primary external-validation GDIS analysis for GSE175634.

This is the FIRST GDIS calculation performed on the external dataset.

Frozen design inherited from p14-p17b
-------------------------------------
Primary state space:
    50 PCs

Primary windows:
    400 cells/window
    100-cell step
    75% overlap

Final transition-evaluable replicate cohort:
    shared_backbone : 19 individuals
    cm_extension    : 15 individuals
    cf_extension    : 14 individuals
    TOTAL           : 48 independent scope x individual profiles

pyGDIS mode
------------
Data-only:
    no analytical Jacobian supplied

Critical localization:
    NO biological critical_value supplied

Therefore pyGDIS uses its reference data-driven transition-energy peak for
transition localization.

IMPORTANT
---------
This script does NOT:
    - tune GDIS
    - change windows
    - change PCA dimensions
    - move biological landmarks
    - use biological landmarks as pyGDIS critical values
    - exclude profiles based on GDIS results
    - declare external validation success/failure

Biological landmarks are attached only AFTER GDIS calculation for descriptive
comparison.

Primary outputs
---------------
For each of the 48 frozen profiles:
    - complete GDIS dataframe merged with frozen window metadata
    - peak locations for:
        GDIS
        sustained instability
        transition instability
        transition energy
    - signed and absolute offsets relative to the pre-specified destination
      state median pseudotime
    - score values at the window nearest the frozen destination landmark

Aggregate outputs:
    - one primary summary row per profile
    - per-scope descriptive summaries
    - transition-weight sensitivity tables (rescoring only; no descriptor rerun)
    - scope-level profile figures

Reference pyGDIS package:
    version 1.0.0
"""

from pathlib import Path
import json
import platform
import sys
from importlib.metadata import version as package_version, PackageNotFoundError

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from gdis import (
        GDIS,
        GDISConfig,
        transition_weight_sensitivity,
    )
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

FINAL_GROUP_COUNT_FILE = (
    P17B_TABLE_DIR
    / "06_final_group_counts.csv"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
    / "p18_external_GSE175634_primary_gdis"
)

DATA_DIR = RESULTS_DIR / "data"
TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR = RESULTS_DIR / "figures"

REPORT_FILE = (
    RESULTS_DIR
    / "p18_external_GSE175634_primary_gdis_report.txt"
)


# ---------------------------------------------------------------------
# Frozen pyGDIS reference settings
# ---------------------------------------------------------------------

EXPECTED_PYGDIS_VERSION = "1.0.0"

EXPECTED_CONFIG = {
    "alpha_j": 0.42,
    "alpha_s": 0.33,
    "alpha_a": 0.25,
    "k_j": 3.0,
    "k_s": 2.6,
    "k_a": 2.4,
    "hill_gamma": 0.72,
    "hill_c": 0.22,
    "complexity_gain": 0.055,
    "temporal_gain": 0.055,
    "temporal_threshold": 0.60,
    "transition_weight": 0.18,
    "transition_width_fraction": 0.09,
    "critical_value": None,
    "smoothing_window": 9,
    "smoothing_polynomial_order": 3,
    "max_sustained": 0.985,
}

TRANSITION_WEIGHTS = (
    0.0,
    0.18,
    0.25,
    0.50,
    0.75,
    1.0,
)

EXPECTED_SCOPE_COUNTS = {
    "shared_backbone": 19,
    "cm_extension": 15,
    "cf_extension": 14,
}

EXPECTED_TOTAL_PROFILES = 48

EXPECTED_WINDOW_SIZE = 400
EXPECTED_N_PCS = 50

CRITICAL_SOURCE_EXPECTED = (
    "data_driven_transition_energy_peak"
)


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


def safe_package_version():
    try:
        return package_version("pygdis")
    except PackageNotFoundError:
        return "UNKNOWN"


def verify_reference_config():
    """
    Verify that installed GDISConfig defaults match the frozen reference.
    """
    config = GDISConfig()

    rows = []
    all_match = True

    for name, expected in EXPECTED_CONFIG.items():

        if not hasattr(config, name):
            rows.append(
                {
                    "parameter": name,
                    "expected": expected,
                    "observed": "<missing>",
                    "match": False,
                }
            )
            all_match = False
            continue

        observed = getattr(
            config,
            name,
        )

        if expected is None:
            match = (
                observed is None
            )
        elif isinstance(
            expected,
            float,
        ):
            match = bool(
                np.isclose(
                    float(observed),
                    expected,
                    rtol=0.0,
                    atol=1e-12,
                )
            )
        else:
            match = (
                observed == expected
            )

        rows.append(
            {
                "parameter": name,
                "expected": expected,
                "observed": observed,
                "match": match,
            }
        )

        if not match:
            all_match = False

    return (
        config,
        pd.DataFrame(rows),
        all_match,
    )


def json_safe(value):
    """
    Convert nested metadata into JSON-serializable Python values.
    """
    if isinstance(
        value,
        dict,
    ):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            json_safe(v)
            for v in value
        ]

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        Path,
    ):
        return str(value)

    return value


def get_component(
    result,
    name,
):
    if name not in result.components:
        raise KeyError(
            f"pyGDIS result is missing component '{name}'."
        )

    values = np.asarray(
        result.components[
            name
        ],
        dtype=float,
    )

    return values


def peak_record(
    merged,
    column,
    prefix,
):
    values = pd.to_numeric(
        merged[
            column
        ],
        errors="raise",
    ).to_numpy(
        dtype=float
    )

    index = int(
        np.nanargmax(
            values
        )
    )

    row = merged.iloc[
        index
    ]

    return {
        f"{prefix}_peak_window_index":
            int(
                row[
                    "window_index"
                ]
            ),
        f"{prefix}_peak_parameter":
            float(
                row[
                    "parameter"
                ]
            ),
        f"{prefix}_peak_value":
            float(
                row[
                    column
                ]
            ),
        f"{prefix}_peak_median_day":
            (
                float(
                    row[
                        "median_day_numeric"
                    ]
                )
                if (
                    "median_day_numeric"
                    in row.index
                )
                else np.nan
            ),
        f"{prefix}_peak_dominant_state":
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
    }


def nearest_landmark_record(
    merged,
    landmark,
):
    parameters = merged[
        "parameter"
    ].to_numpy(
        dtype=float
    )

    position = int(
        np.argmin(
            np.abs(
                parameters
                - float(
                    landmark
                )
            )
        )
    )

    row = merged.iloc[
        position
    ]

    output = {
        "landmark_nearest_window_index":
            int(
                row[
                    "window_index"
                ]
            ),
        "landmark_nearest_window_parameter":
            float(
                row[
                    "parameter"
                ]
            ),
        "landmark_nearest_parameter_error":
            float(
                abs(
                    float(
                        row[
                            "parameter"
                        ]
                    )
                    - float(
                        landmark
                    )
                )
            ),
        "landmark_window_median_day":
            (
                float(
                    row[
                        "median_day_numeric"
                    ]
                )
                if (
                    "median_day_numeric"
                    in row.index
                )
                else np.nan
            ),
        "landmark_window_dominant_state":
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
    }

    value_columns = [
        "gdis",
        "potential",
        "sustained_instability",
        "transition_instability",
        "transition_energy",
        "transition_base",
        "core",
    ]

    for column in value_columns:
        if column in row.index:
            output[
                f"{column}_at_landmark_window"
            ] = float(
                row[
                    column
                ]
            )

    return output


def merge_gdis_with_window_metadata(
    result_df,
    window_meta,
):
    """
    pyGDIS internally sorts parameters. The frozen inputs are already
    strictly increasing, but merge by parameter after a high-precision
    equality check rather than assuming row position silently.
    """
    result_df = (
        result_df.sort_values(
            "parameter"
        )
        .reset_index(
            drop=True
        )
        .copy()
    )

    window_meta = (
        window_meta.sort_values(
            "parameter_median_pseudotime"
        )
        .reset_index(
            drop=True
        )
        .copy()
    )

    if len(
        result_df
    ) != len(
        window_meta
    ):
        raise ValueError(
            "GDIS result length does not match window metadata."
        )

    result_parameters = result_df[
        "parameter"
    ].to_numpy(
        dtype=float
    )

    metadata_parameters = window_meta[
        "parameter_median_pseudotime"
    ].to_numpy(
        dtype=float
    )

    if not np.allclose(
        result_parameters,
        metadata_parameters,
        atol=1e-12,
        rtol=1e-10,
    ):
        maximum_difference = float(
            np.max(
                np.abs(
                    result_parameters
                    - metadata_parameters
                )
            )
        )

        raise ValueError(
            "GDIS parameters do not align with frozen window metadata. "
            f"Max absolute difference={maximum_difference:.12g}"
        )

    # Avoid duplicate parameter column.
    metadata_copy = window_meta.drop(
        columns=[
            "parameter_median_pseudotime",
        ]
    )

    merged = pd.concat(
        [
            result_df,
            metadata_copy,
        ],
        axis=1,
    )

    return merged


def plot_scope_profiles(
    all_profiles,
    scope,
    destination_state,
    output_path,
):
    """
    Plot all individual GDIS profiles for one scope in raw deposited
    pseudotime. Individual frozen destination landmarks are shown as points
    at their nearest GDIS window, rather than as one pooled vertical line.
    """
    subset = all_profiles[
        all_profiles[
            "scope"
        ] == scope
    ].copy()

    fig, ax = plt.subplots(
        figsize=(10.5, 6)
    )

    for individual, group in (
        subset.groupby(
            "individual",
            observed=True,
        )
    ):

        group = group.sort_values(
            "parameter"
        )

        ax.plot(
            group[
                "parameter"
            ],
            group[
                "gdis"
            ],
            linewidth=1.0,
            alpha=0.55,
        )

        landmark = float(
            group[
                "frozen_destination_landmark"
            ].iloc[
                0
            ]
        )

        nearest_position = int(
            np.argmin(
                np.abs(
                    group[
                        "parameter"
                    ].to_numpy(
                        dtype=float
                    )
                    - landmark
                )
            )
        )

        nearest = group.iloc[
            nearest_position
        ]

        ax.scatter(
            [
                nearest[
                    "parameter"
                ]
            ],
            [
                nearest[
                    "gdis"
                ]
            ],
            s=12,
            alpha=0.65,
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
        f"{scope}: primary external GDIS profiles\n"
        f"points mark nearest windows to frozen {destination_state} "
        f"median landmarks"
    )

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
        DATA_DIR,
        TABLE_DIR,
        FIGURE_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    required_files = [
        TRANSITION_FILE,
        FINAL_MANIFEST_FILE,
        FINAL_GROUP_COUNT_FILE,
    ]

    missing = [
        str(path)
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

    # -----------------------------------------------------------------
    # Verify installed pyGDIS BEFORE external calculation.
    # -----------------------------------------------------------------

    installed_pygdis = (
        safe_package_version()
    )

    (
        reference_config,
        config_check_table,
        config_defaults_match,
    ) = verify_reference_config()

    version_match = (
        installed_pygdis
        == EXPECTED_PYGDIS_VERSION
    )

    if not version_match:
        raise RuntimeError(
            "Installed pyGDIS version does not match the frozen "
            f"external-validation version. Expected "
            f"{EXPECTED_PYGDIS_VERSION}, observed {installed_pygdis}."
        )

    if not config_defaults_match:
        mismatch = config_check_table[
            ~config_check_table[
                "match"
            ]
        ]

        raise RuntimeError(
            "Installed pyGDIS default configuration differs from "
            "the frozen reference:\n"
            + mismatch.to_string(
                index=False
            )
        )

    manifest = pd.read_csv(
        FINAL_MANIFEST_FILE,
        low_memory=False,
    )

    final_counts = pd.read_csv(
        FINAL_GROUP_COUNT_FILE,
        low_memory=False,
    )

    transitions = pd.read_csv(
        TRANSITION_FILE,
        low_memory=False,
    )

    manifest[
        "individual"
    ] = manifest[
        "individual"
    ].astype(str)

    final_counts[
        "scope"
    ] = final_counts[
        "scope"
    ].astype(str)

    transitions[
        "transition_id"
    ] = transitions[
        "transition_id"
    ].astype(str)

    # -----------------------------------------------------------------
    # Frozen cohort integrity.
    # -----------------------------------------------------------------

    if len(
        manifest
    ) != EXPECTED_TOTAL_PROFILES:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_PROFILES} final primary profiles, "
            f"found {len(manifest)}."
        )

    observed_scope_counts = (
        manifest.groupby(
            "scope",
            observed=True,
        )[
            "individual"
        ]
        .nunique()
        .to_dict()
    )

    for scope, expected in (
        EXPECTED_SCOPE_COUNTS.items()
    ):

        observed = int(
            observed_scope_counts.get(
                scope,
                0,
            )
        )

        if observed != expected:
            raise ValueError(
                f"Frozen scope count mismatch for {scope}: "
                f"expected {expected}, observed {observed}."
            )

    if not manifest[
        "referenced_files_exist"
    ].astype(bool).all():
        raise FileNotFoundError(
            "Final manifest contains missing trajectory files."
        )

    # -----------------------------------------------------------------
    # Main calculation.
    # -----------------------------------------------------------------

    summary_rows = []
    all_profile_frames = []
    sensitivity_frames = []
    metadata_rows = []
    computational_rows = []

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as report:

        write_both(
            report,
            "=" * 102,
        )

        write_both(
            report,
            "GDIS-Bio Primary External Validation GDIS Analysis",
        )

        write_both(
            report,
            "Dataset: GSE175634",
        )

        write_both(
            report,
            "=" * 102,
        )

        write_both(
            report,
            "\nTHIS IS THE FIRST GDIS CALCULATION ON GSE175634.",
        )

        write_both(
            report,
            "Biological transition landmarks are NOT supplied to pyGDIS.",
        )

        write_both(
            report,
            "Data-only mode is used: no analytical Jacobian.",
        )

        # =========================================================
        # 1. Software / frozen configuration
        # =========================================================

        write_both(
            report,
            "\n1. SOFTWARE AND FROZEN pyGDIS CONFIGURATION",
        )

        write_both(
            report,
            "-" * 102,
        )

        write_both(
            report,
            f"Python: {platform.python_version()}",
        )

        write_both(
            report,
            f"pyGDIS: {installed_pygdis}",
        )

        write_both(
            report,
            "Analytical Jacobian supplied: NO",
        )

        write_both(
            report,
            "Biological critical_value supplied: NO",
        )

        write_both(
            report,
            f"Frozen profiles: {len(manifest)}",
        )

        write_both(
            report,
            "Frozen scope counts: "
            + str(
                observed_scope_counts
            ),
        )

        save_table(
            config_check_table,
            "01_pygdis_reference_config_verification.csv",
            index=False,
        )

        write_both(
            report,
            "\nReference configuration verification:\n"
            + config_check_table.to_string(
                index=False
            ),
        )

        # =========================================================
        # 2. Calculate 48 frozen primary profiles
        # =========================================================

        write_both(
            report,
            "\n2. PRIMARY EXTERNAL GDIS CALCULATION",
        )

        write_both(
            report,
            "-" * 102,
        )

        for row_number, manifest_row in (
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
                manifest_row[
                    "scope"
                ]
            )

            individual = str(
                manifest_row[
                    "individual"
                ]
            )

            transition_id = str(
                manifest_row[
                    "transition_id"
                ]
            )

            destination_state = str(
                manifest_row[
                    "destination_state"
                ]
            )

            frozen_landmark = float(
                manifest_row[
                    "destination_state_median_pseudotime"
                ]
            )

            npz_path = Path(
                str(
                    manifest_row[
                        "npz_path"
                    ]
                )
            )

            window_meta_path = Path(
                str(
                    manifest_row[
                        "window_metadata_path"
                    ]
                )
            )

            if not npz_path.exists():
                raise FileNotFoundError(
                    f"Missing trajectory file: {npz_path}"
                )

            if not window_meta_path.exists():
                raise FileNotFoundError(
                    f"Missing window metadata: {window_meta_path}"
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

            window_meta = pd.read_csv(
                window_meta_path,
                low_memory=False,
            )

            # ------------------------------------------------------
            # Input qualification before calling pyGDIS.
            # ------------------------------------------------------

            if trajectories.ndim != 3:
                raise ValueError(
                    f"{scope}/{individual}: expected 3D trajectory "
                    f"array; got {trajectories.shape}."
                )

            if trajectories.shape[
                1
            ] != EXPECTED_WINDOW_SIZE:
                raise ValueError(
                    f"{scope}/{individual}: expected window size "
                    f"{EXPECTED_WINDOW_SIZE}; got "
                    f"{trajectories.shape[1]}."
                )

            if trajectories.shape[
                2
            ] != EXPECTED_N_PCS:
                raise ValueError(
                    f"{scope}/{individual}: expected "
                    f"{EXPECTED_N_PCS} PCs; got "
                    f"{trajectories.shape[2]}."
                )

            if len(
                parameters
            ) != trajectories.shape[
                0
            ]:
                raise ValueError(
                    f"{scope}/{individual}: parameter/window count "
                    "mismatch."
                )

            if len(
                window_meta
            ) != len(
                parameters
            ):
                raise ValueError(
                    f"{scope}/{individual}: metadata/window count "
                    "mismatch."
                )

            if not np.isfinite(
                trajectories
            ).all():
                raise ValueError(
                    f"{scope}/{individual}: non-finite trajectory data."
                )

            if not np.isfinite(
                parameters
            ).all():
                raise ValueError(
                    f"{scope}/{individual}: non-finite parameters."
                )

            increments = np.diff(
                parameters
            )

            if not np.all(
                increments > 0
            ):
                raise ValueError(
                    f"{scope}/{individual}: parameters are not "
                    "strictly increasing."
                )

            metadata_parameters = pd.to_numeric(
                window_meta[
                    "parameter_median_pseudotime"
                ],
                errors="raise",
            ).to_numpy(
                dtype=float
            )

            if not np.allclose(
                parameters,
                metadata_parameters,
                atol=1e-12,
                rtol=1e-10,
            ):
                raise ValueError(
                    f"{scope}/{individual}: NPZ parameter values "
                    "do not match window metadata."
                )

            # ------------------------------------------------------
            # UNCHANGED reference pyGDIS calculation.
            # ------------------------------------------------------

            result = GDIS().fit_transform(
                trajectories,
                parameters,
            )

            result_df = result.to_dataframe()

            # Required reference outputs.
            required_result_columns = [
                "parameter",
                "gdis",
                "potential",
                "sustained_instability",
                "transition_instability",
            ]

            missing_result_columns = [
                column
                for column in required_result_columns
                if column
                not in result_df.columns
            ]

            if missing_result_columns:
                raise ValueError(
                    f"{scope}/{individual}: pyGDIS result missing "
                    f"columns {missing_result_columns}."
                )

            # Explicitly attach components, overwriting only if needed
            # with the package's actual component arrays.
            for component_name in [
                "jacobian_raw",
                "stretching_raw",
                "expansion_raw",
                "entropy_raw",
                "temporal_raw",
                "temporal_mean",
                "temporal_persistence",
                "jacobian_scaled",
                "stretching_scaled",
                "expansion_scaled",
                "jacobian_saturated",
                "stretching_saturated",
                "expansion_saturated",
                "core",
                "complexity_factor",
                "temporal_factor",
                "transition_energy",
                "critical_window",
                "transition_base",
            ]:

                if component_name in (
                    result.components
                ):

                    values = get_component(
                        result,
                        component_name,
                    )

                    if len(
                        values
                    ) != len(
                        result_df
                    ):
                        raise ValueError(
                            f"{scope}/{individual}: component "
                            f"{component_name} length mismatch."
                        )

                    result_df[
                        component_name
                    ] = values

            if (
                "transition_energy"
                not in result_df.columns
            ):
                raise ValueError(
                    f"{scope}/{individual}: transition_energy "
                    "component is unavailable."
                )

            if (
                "transition_base"
                not in result_df.columns
            ):
                raise ValueError(
                    f"{scope}/{individual}: transition_base "
                    "component is unavailable."
                )

            merged = (
                merge_gdis_with_window_metadata(
                    result_df,
                    window_meta,
                )
            )

            # ------------------------------------------------------
            # Attach / verify frozen profile provenance.
            #
            # p17 window metadata already contains "scope" and
            # "individual".  Do not insert duplicate columns; instead
            # verify that the merged metadata agrees exactly with the
            # frozen p17b manifest.
            # ------------------------------------------------------

            if "individual" in merged.columns:

                observed_individuals = set(
                    merged[
                        "individual"
                    ].astype(str).unique().tolist()
                )

                if observed_individuals != {
                    individual
                }:
                    raise ValueError(
                        f"{scope}/{individual}: merged window metadata "
                        f"contains unexpected individual values "
                        f"{sorted(observed_individuals)}."
                    )

                merged[
                    "individual"
                ] = individual

            else:

                merged.insert(
                    0,
                    "individual",
                    individual,
                )

            if "scope" in merged.columns:

                observed_scopes = set(
                    merged[
                        "scope"
                    ].astype(str).unique().tolist()
                )

                if observed_scopes != {
                    scope
                }:
                    raise ValueError(
                        f"{scope}/{individual}: merged window metadata "
                        f"contains unexpected scope values "
                        f"{sorted(observed_scopes)}."
                    )

                merged[
                    "scope"
                ] = scope

            else:

                merged.insert(
                    0,
                    "scope",
                    scope,
                )

            # These fields are not expected in p17 window metadata.
            # If they are present for any reason, verify/overwrite them
            # deterministically rather than failing on duplicate insert.
            merged[
                "transition_id"
            ] = transition_id

            merged[
                "frozen_destination_state"
            ] = destination_state

            merged[
                "frozen_destination_landmark"
            ] = frozen_landmark

            # Put provenance columns first for readability.
            provenance_columns = [
                "scope",
                "individual",
                "transition_id",
                "frozen_destination_state",
                "frozen_destination_landmark",
            ]

            remaining_columns = [
                column
                for column in merged.columns
                if column not in provenance_columns
            ]

            merged = merged[
                provenance_columns
                + remaining_columns
            ]

            # ------------------------------------------------------
            # Computational validity checks.
            # ------------------------------------------------------

            gdis_values = merged[
                "gdis"
            ].to_numpy(
                dtype=float
            )

            all_finite = bool(
                np.isfinite(
                    merged[
                        [
                            "parameter",
                            "gdis",
                            "potential",
                            "sustained_instability",
                            "transition_instability",
                            "transition_energy",
                            "transition_base",
                        ]
                    ].to_numpy(
                        dtype=float
                    )
                ).all()
            )

            bounded = bool(
                np.all(
                    gdis_values >= 0.0
                )
                and np.all(
                    gdis_values < 1.0
                )
            )

            result_metadata = (
                dict(
                    result.metadata
                )
                if result.metadata
                is not None
                else {}
            )

            critical_source = str(
                result_metadata.get(
                    "critical_value_source",
                    "",
                )
            )

            critical_source_match = (
                critical_source
                == CRITICAL_SOURCE_EXPECTED
            )

            # ------------------------------------------------------
            # Peak summaries.
            # ------------------------------------------------------

            summary_row = {
                "scope":
                    scope,
                "individual":
                    individual,
                "transition_id":
                    transition_id,
                "destination_state":
                    destination_state,
                "frozen_destination_landmark":
                    frozen_landmark,
                "n_windows":
                    int(
                        len(
                            merged
                        )
                    ),
                "parameter_min":
                    float(
                        merged[
                            "parameter"
                        ].min()
                    ),
                "parameter_max":
                    float(
                        merged[
                            "parameter"
                        ].max()
                    ),
                "gdis_min":
                    float(
                        merged[
                            "gdis"
                        ].min()
                    ),
                "gdis_mean":
                    float(
                        merged[
                            "gdis"
                        ].mean()
                    ),
                "gdis_max":
                    float(
                        merged[
                            "gdis"
                        ].max()
                    ),
                "critical_value_source":
                    critical_source,
                "all_primary_outputs_finite":
                    all_finite,
                "gdis_bounded_0_1":
                    bounded,
            }

            summary_row.update(
                peak_record(
                    merged,
                    "gdis",
                    "gdis",
                )
            )

            summary_row.update(
                peak_record(
                    merged,
                    "sustained_instability",
                    "sustained",
                )
            )

            summary_row.update(
                peak_record(
                    merged,
                    "transition_instability",
                    "transition_instability",
                )
            )

            summary_row.update(
                peak_record(
                    merged,
                    "transition_energy",
                    "transition_energy",
                )
            )

            summary_row.update(
                nearest_landmark_record(
                    merged,
                    frozen_landmark,
                )
            )

            # Signed offsets: negative = peak precedes destination median.
            for prefix in [
                "gdis",
                "sustained",
                "transition_instability",
                "transition_energy",
            ]:

                parameter_key = (
                    f"{prefix}_peak_parameter"
                )

                peak_parameter = float(
                    summary_row[
                        parameter_key
                    ]
                )

                summary_row[
                    f"{prefix}_peak_minus_landmark"
                ] = (
                    peak_parameter
                    - frozen_landmark
                )

                summary_row[
                    f"{prefix}_absolute_peak_landmark_error"
                ] = abs(
                    peak_parameter
                    - frozen_landmark
                )

            summary_rows.append(
                summary_row
            )

            # ------------------------------------------------------
            # Save full profile.
            # ------------------------------------------------------

            stem = (
                f"{scope}_individual_{individual}"
            )

            profile_path = (
                DATA_DIR
                / f"{stem}_gdis.csv"
            )

            merged.to_csv(
                profile_path,
                index=False,
            )

            all_profile_frames.append(
                merged
            )

            # ------------------------------------------------------
            # Transition-weight sensitivity:
            # rescore existing result only, no descriptor rerun.
            # ------------------------------------------------------

            weight_table = (
                transition_weight_sensitivity(
                    result,
                    weights=TRANSITION_WEIGHTS,
                )
            )

            weight_table = (
                weight_table.copy()
            )

            weight_table.insert(
                0,
                "individual",
                individual,
            )

            weight_table.insert(
                0,
                "scope",
                scope,
            )

            weight_table.insert(
                2,
                "transition_id",
                transition_id,
            )

            sensitivity_frames.append(
                weight_table
            )

            # ------------------------------------------------------
            # Save pyGDIS metadata.
            # ------------------------------------------------------

            metadata_json = json.dumps(
                json_safe(
                    result_metadata
                ),
                sort_keys=True,
            )

            metadata_rows.append(
                {
                    "scope":
                        scope,
                    "individual":
                        individual,
                    "transition_id":
                        transition_id,
                    "critical_value_source":
                        critical_source,
                    "metadata_json":
                        metadata_json,
                }
            )

            computational_rows.append(
                {
                    "scope":
                        scope,
                    "individual":
                        individual,
                    "n_windows":
                        int(
                            len(
                                merged
                            )
                        ),
                    "trajectory_shape":
                        (
                            f"{trajectories.shape[0]}x"
                            f"{trajectories.shape[1]}x"
                            f"{trajectories.shape[2]}"
                        ),
                    "parameters_strictly_increasing":
                        bool(
                            np.all(
                                increments > 0
                            )
                        ),
                    "result_parameters_match_input":
                        bool(
                            np.allclose(
                                merged[
                                    "parameter"
                                ].to_numpy(
                                    dtype=float
                                ),
                                parameters,
                                atol=1e-12,
                                rtol=1e-10,
                            )
                        ),
                    "all_primary_outputs_finite":
                        all_finite,
                    "gdis_bounded_0_1":
                        bounded,
                    "critical_value_source":
                        critical_source,
                    "critical_value_source_match":
                        critical_source_match,
                }
            )

            write_both(
                report,
                f"[{row_number + 1:02d}/{len(manifest):02d}] "
                f"{scope:16s} | individual {individual} | "
                f"windows={len(merged):3d} | "
                f"GDIS max={merged['gdis'].max():.6f} at "
                f"p={summary_row['gdis_peak_parameter']:.6f} | "
                f"energy peak p="
                f"{summary_row['transition_energy_peak_parameter']:.6f} | "
                f"landmark={frozen_landmark:.6f}",
            )

        # =========================================================
        # 3. Aggregate exports
        # =========================================================

        write_both(
            report,
            "\n3. AGGREGATE PRIMARY OUTPUTS",
        )

        write_both(
            report,
            "-" * 102,
        )

        summary = (
            pd.DataFrame(
                summary_rows
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

        all_profiles = pd.concat(
            all_profile_frames,
            ignore_index=True,
        )

        weight_sensitivity = pd.concat(
            sensitivity_frames,
            ignore_index=True,
        )

        metadata_table = pd.DataFrame(
            metadata_rows
        )

        computational_table = pd.DataFrame(
            computational_rows
        )

        save_table(
            summary,
            "02_primary_profile_summary.csv",
            index=False,
        )

        all_profiles.to_csv(
            TABLE_DIR
            / "03_all_primary_gdis_profiles.csv.gz",
            index=False,
            compression="gzip",
        )

        save_table(
            weight_sensitivity,
            "04_transition_weight_sensitivity.csv",
            index=False,
        )

        save_table(
            metadata_table,
            "05_pygdis_result_metadata.csv",
            index=False,
        )

        save_table(
            computational_table,
            "06_computational_qualification_by_profile.csv",
            index=False,
        )

        # Scope-level descriptive summary only.
        scope_summary = (
            summary.groupby(
                [
                    "scope",
                    "transition_id",
                    "destination_state",
                ],
                observed=True,
            )
            .agg(
                n_profiles=(
                    "individual",
                    "nunique",
                ),
                median_gdis_max=(
                    "gdis_max",
                    "median",
                ),
                q25_gdis_max=(
                    "gdis_max",
                    lambda x: x.quantile(
                        0.25
                    ),
                ),
                q75_gdis_max=(
                    "gdis_max",
                    lambda x: x.quantile(
                        0.75
                    ),
                ),
                median_gdis_peak_minus_landmark=(
                    "gdis_peak_minus_landmark",
                    "median",
                ),
                median_transition_energy_peak_minus_landmark=(
                    "transition_energy_peak_minus_landmark",
                    "median",
                ),
                median_abs_gdis_peak_landmark_error=(
                    "gdis_absolute_peak_landmark_error",
                    "median",
                ),
                median_abs_transition_energy_peak_landmark_error=(
                    "transition_energy_absolute_peak_landmark_error",
                    "median",
                ),
            )
            .reset_index()
        )

        save_table(
            scope_summary,
            "07_scope_descriptive_summary.csv",
            index=False,
        )

        write_both(
            report,
            "\nScope-level descriptive summary:\n"
            + scope_summary.round(
                6
            ).to_string(
                index=False
            ),
        )

        # =========================================================
        # 4. Figures
        # =========================================================

        write_both(
            report,
            "\n4. PRIMARY PROFILE FIGURES",
        )

        write_both(
            report,
            "-" * 102,
        )

        scope_destination = (
            manifest.groupby(
                "scope",
                observed=True,
            )[
                "destination_state"
            ]
            .first()
            .to_dict()
        )

        for scope in [
            "shared_backbone",
            "cm_extension",
            "cf_extension",
        ]:

            plot_scope_profiles(
                all_profiles=all_profiles,
                scope=scope,
                destination_state=str(
                    scope_destination[
                        scope
                    ]
                ),
                output_path=(
                    FIGURE_DIR
                    / f"01_{scope}_primary_gdis_profiles.png"
                ),
            )

        write_both(
            report,
            "Saved one raw-pseudotime profile figure per scope.",
        )

        # =========================================================
        # 5. Computational qualification
        # =========================================================

        write_both(
            report,
            "\n5. COMPUTATIONAL QUALIFICATION",
        )

        write_both(
            report,
            "-" * 102,
        )

        all_finite_pass = bool(
            computational_table[
                "all_primary_outputs_finite"
            ].all()
        )

        all_bounded_pass = bool(
            computational_table[
                "gdis_bounded_0_1"
            ].all()
        )

        all_parameter_pass = bool(
            computational_table[
                "parameters_strictly_increasing"
            ].all()
        )

        all_parameter_match_pass = bool(
            computational_table[
                "result_parameters_match_input"
            ].all()
        )

        all_critical_source_pass = bool(
            computational_table[
                "critical_value_source_match"
            ].all()
        )

        expected_shape_pass = bool(
            computational_table[
                "trajectory_shape"
            ].str.endswith(
                f"x{EXPECTED_WINDOW_SIZE}x{EXPECTED_N_PCS}"
            ).all()
        )

        expected_scope_count_pass = (
            len(
                summary
            )
            == EXPECTED_TOTAL_PROFILES
            and all(
                int(
                    summary.loc[
                        summary[
                            "scope"
                        ] == scope,
                        "individual",
                    ].nunique()
                )
                == expected
                for scope, expected
                in EXPECTED_SCOPE_COUNTS.items()
            )
        )

        checks = [
            (
                f"Installed pyGDIS version is exactly "
                f"{EXPECTED_PYGDIS_VERSION}",
                version_match,
            ),
            (
                "Installed GDISConfig defaults match frozen reference",
                config_defaults_match,
            ),
            (
                "All 48 frozen primary profiles were analyzed",
                expected_scope_count_pass,
            ),
            (
                "All trajectory arrays are n_windows x 400 x 50",
                expected_shape_pass,
            ),
            (
                "All input parameters are strictly increasing",
                all_parameter_pass,
            ),
            (
                "All pyGDIS result parameters match frozen inputs",
                all_parameter_match_pass,
            ),
            (
                "All primary GDIS outputs are finite",
                all_finite_pass,
            ),
            (
                "All GDIS values satisfy 0 <= GDIS < 1",
                all_bounded_pass,
            ),
            (
                "Every profile records data-driven transition-energy "
                "critical localization",
                all_critical_source_pass,
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
            "08_primary_gdis_computational_qualification.csv",
            index=False,
        )

        write_both(
            report,
            f"\nComputational checks passed: "
            f"{passed}/{len(checks)}",
        )

        # =========================================================
        # 6. Final status / interpretation guardrails
        # =========================================================

        write_both(
            report,
            "\n6. FINAL STATUS",
        )

        write_both(
            report,
            "-" * 102,
        )

        if passed == len(
            checks
        ):

            write_both(
                report,
                "FINAL COMPUTATIONAL STATUS: PASS",
            )

            write_both(
                report,
                "All 48 frozen primary external-validation GDIS "
                "profiles were computed with the unchanged reference "
                "implementation.",
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
            "  - This script does NOT declare external validation success.",
        )

        write_both(
            report,
            "  - GDIS peak, sustained-instability peak, transition-"
            "instability peak, and transition-energy peak are distinct.",
        )

        write_both(
            report,
            "  - A negative peak-minus-landmark value means the peak "
            "precedes the destination-state MEDIAN landmark, not proven "
            "biological onset.",
        )

        write_both(
            report,
            "  - Biological landmarks were attached only after the "
            "unsupervised/data-driven GDIS calculation.",
        )

        write_both(
            report,
            "  - Overlapping windows are statistically dependent and "
            "must not be treated as independent observations.",
        )

        write_both(
            report,
            "  - Individual is the independent replication unit.",
        )

        write_both(
            report,
            "  - No GDIS parameter was selected or changed based on "
            "GSE175634 results.",
        )

        write_both(
            report,
            "\nNext step:",
        )

        write_both(
            report,
            "Evaluate transition localization, across-individual "
            "consistency, null alignment, and transferred sensitivity "
            "analyses without retuning GDIS.",
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

    print("\n" + "=" * 102)
    print("p18_GSE175634_primary_external_gdis.py completed.")
    print("=" * 102)
    print(f"Report : {REPORT_FILE}")
    print(f"Tables : {TABLE_DIR}")
    print(f"Figures: {FIGURE_DIR}")
    print(f"Data   : {DATA_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

