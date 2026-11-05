#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""
P21-A: Frozen GSE175634 GDIS reference reproduction.

Read-only audit:
- Load original p17 trajectory arrays.
- Recalculate GDIS with installed pyGDIS.
- Compare recalculated scores to archived p18 results.
- Fail on missing data, invalid parameters, or mismatch.

Does not overwrite any reference artifacts.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from gdis import GDIS, GDISConfig

ROOT = Path(__file__).resolve().parent.parent

MANIFEST = (
    ROOT / "results"
    / "p17b_external_GSE175634_transition_evaluability_freeze"
    / "tables"
    / "04_final_primary_gdis_file_manifest.csv"
)

REFERENCE_DIR = (
    ROOT / "results"
    / "p18_external_GSE175634_primary_gdis"
    / "data"
)

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


def resolve_path(value):
    """
    Resolve archived paths against the current project.

    Preserve the frozen manifest unchanged.
    """
    original = Path(str(value))

    if original.is_file():
        return original

    parts = original.parts

    if "results" in parts:
        index = parts.index("results")
        relocated = ROOT.joinpath(*parts[index:])

        if relocated.is_file():
            return relocated

    raise FileNotFoundError(
        f"Archived file unavailable: {original}"
    )


def check_configuration():
    config = GDISConfig()

    for key, expected in EXPECTED_CONFIG.items():
        observed = getattr(config, key)

        if expected is None:
            valid = observed is None
        elif isinstance(expected, float):
            valid = bool(np.isclose(
                observed, expected, atol=1e-12, rtol=0
            ))
        else:
            valid = observed == expected

        if not valid:
            raise RuntimeError(
                f"Configuration mismatch: {key}: "
                f"observed={observed}, expected={expected}"
            )

    print("REFERENCE CONFIGURATION: PASS")


def audit_row(row):
    scope = str(row["scope"])
    individual = str(row["individual"])

    npz_path = resolve_path(row["npz_path"])
    reference_path = (
        REFERENCE_DIR / f"{npz_path.stem}_gdis.csv"
    )

    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)

    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)

    with np.load(npz_path, allow_pickle=False) as archive:
        trajectories = np.asarray(
            archive["trajectories"], dtype=np.float64
        )
        parameters = np.asarray(
            archive["parameters"], dtype=np.float64
        )

    if trajectories.ndim != 3:
        raise ValueError("Invalid trajectory dimensions")

    if trajectories.shape[1:] != (400, 50):
        raise ValueError(
            f"Unexpected trajectory shape: {trajectories.shape}"
        )

    if len(parameters) != len(trajectories):
        raise ValueError("Parameter/window count mismatch")

    if not np.isfinite(trajectories).all():
        raise ValueError("Nonfinite trajectory values")

    if not np.isfinite(parameters).all():
        raise ValueError("Nonfinite parameters")

    if not np.all(np.diff(parameters) > 0):
        raise ValueError("Parameters not strictly increasing")

    recalculated = GDIS().fit_transform(
        trajectories, parameters
    ).to_dataframe()

    archived = pd.read_csv(reference_path)

    required = ["parameter", "gdis"]

    for column in required:
        if column not in recalculated.columns:
            raise ValueError(
                f"Missing recalculated column: {column}"
            )
        if column not in archived.columns:
            raise ValueError(
                f"Missing archived column: {column}"
            )

    if len(recalculated) != len(archived):
        raise ValueError("Reference profile length mismatch")

    parameter_error = np.max(np.abs(
        recalculated["parameter"].to_numpy(dtype=float)
        - archived["parameter"].to_numpy(dtype=float)
    ))

    if not np.allclose(
        recalculated["parameter"].to_numpy(dtype=float),
        archived["parameter"].to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise RuntimeError(
            f"Reference parameter mismatch: {parameter_error}"
        )

    new_scores = recalculated["gdis"].to_numpy(dtype=float)
    old_scores = archived["gdis"].to_numpy(dtype=float)

    if not (
        np.isfinite(new_scores).all()
        and np.isfinite(old_scores).all()
    ):
        raise ValueError("Nonfinite GDIS scores")

    maximum_error = float(np.max(np.abs(
        new_scores - old_scores
    )))

    passed = bool(np.allclose(
        new_scores, old_scores,
        rtol=1e-10,
        atol=1e-10,
    ))

    print(
        f"{scope}/{individual}: "
        f"windows={len(parameters)}, "
        f"max_abs_error={maximum_error:.6e}, "
        f"status={'PASS' if passed else 'FAIL'}"
    )

    if not passed:
        raise RuntimeError(
            f"Reference reproduction failed: {scope}/{individual}"
        )

    return maximum_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all",
        action="store_true",
        help="Audit all frozen reference profiles",
    )
    args = parser.parse_args()

    check_configuration()

    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)

    manifest = pd.read_csv(
        MANIFEST,
        dtype={"individual": str},
    )

    if len(manifest) != 48:
        raise RuntimeError(
            f"Expected 48 profiles, found {len(manifest)}"
        )

    if manifest.duplicated(
        ["scope", "individual"]
    ).any():
        raise RuntimeError("Duplicate reference profiles")

    if not args.all:
        manifest = manifest.iloc[:1]

    errors = []

    for _, row in manifest.iterrows():
        errors.append(audit_row(row))

    print()
    print("P25-A REFERENCE AUDIT: PASS")
    print("Profiles checked:", len(errors))
    print("Maximum absolute error:", max(errors))


if __name__ == "__main__":
    main()
