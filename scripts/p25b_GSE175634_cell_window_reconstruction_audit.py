#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""P25-B: Read-only cell-to-window reference reconstruction audit."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

STATE_FILE = (
    ROOT / "results"
    / "p15_external_GSE175634_preprocessing_state_space"
    / "data/external_state_space_pca_50.csv.gz"
)

MANIFEST_FILE = (
    ROOT / "results"
    / "p17b_external_GSE175634_transition_evaluability_freeze"
    / "tables/04_final_primary_gdis_file_manifest.csv"
)

STATES = {
    "shared_backbone": ["IPSC", "MES", "CMES", "PROG"],
    "cm_extension": ["IPSC", "MES", "CMES", "PROG", "CM"],
    "cf_extension": ["IPSC", "MES", "CMES", "PROG", "CF"],
}

PC_COLUMNS = [f"PC{i}" for i in range(1, 51)]


def resolve_path(value):
    original = Path(str(value))

    if original.is_file():
        return original

    parts = original.parts

    if "results" in parts:
        index = parts.index("results")
        relocated = ROOT.joinpath(*parts[index:])

        if relocated.is_file():
            return relocated

    raise FileNotFoundError(original)


def load_state_space():
    if not STATE_FILE.is_file():
        raise FileNotFoundError(STATE_FILE)

    columns = [
        "cell", "individual", "type", "dpt_pseudotime"
    ] + PC_COLUMNS

    df = pd.read_csv(
        STATE_FILE,
        compression="gzip",
        usecols=columns,
        dtype={
            "cell": str,
            "individual": str,
            "type": str,
        },
        low_memory=False,
    )

    df["dpt_pseudotime"] = pd.to_numeric(
        df["dpt_pseudotime"],
        errors="raise",
    )

    df[PC_COLUMNS] = df[PC_COLUMNS].astype(np.float32)

    if df["cell"].duplicated().any():
        raise RuntimeError("Duplicate cell identifiers")

    if not np.isfinite(
        df["dpt_pseudotime"].to_numpy()
    ).all():
        raise RuntimeError("Nonfinite pseudotime")

    print("STATE SPACE LOADED:", len(df), "cells")

    return df


def audit_profile(row, state_space):
    scope = str(row["scope"])
    individual = str(row["individual"])

    expected_states = STATES[scope]

    group = state_space.loc[
        (state_space["individual"] == individual)
        & (state_space["type"].isin(expected_states))
    ].copy()

    if group.empty:
        raise RuntimeError(
            f"Empty cell group: {scope}/{individual}"
        )

    missing = set(expected_states) - set(group["type"])

    if missing:
        raise RuntimeError(
            f"{scope}/{individual}: missing states {missing}"
        )

    ordered = group.sort_values(
        ["dpt_pseudotime", "cell"],
        kind="mergesort",
    ).reset_index(drop=True)

    X = ordered[PC_COLUMNS].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    pseudo = ordered["dpt_pseudotime"].to_numpy(
        dtype=np.float64,
    )

    window_size = 400
    step_size = 100

    starts = range(
        0,
        len(ordered) - window_size + 1,
        step_size,
    )

    reconstructed = np.stack(
        [
            X[start:start + window_size]
            for start in starts
        ],
        axis=0,
    )

    parameters = np.asarray(
        [
            np.median(pseudo[start:start + window_size])
            for start in starts
        ],
        dtype=np.float64,
    )

    npz_path = resolve_path(row["npz_path"])

    with np.load(npz_path, allow_pickle=False) as archive:
        reference = np.asarray(
            archive["trajectories"]
        )
        reference_parameters = np.asarray(
            archive["parameters"]
        )

    if reconstructed.shape != reference.shape:
        raise RuntimeError(
            f"{scope}/{individual}: shape mismatch: "
            f"{reconstructed.shape} vs {reference.shape}"
        )

    if len(parameters) != len(reference_parameters):
        raise RuntimeError("Parameter count mismatch")

    matrix_error = float(np.max(np.abs(
        reconstructed.astype(np.float64)
        - reference.astype(np.float64)
    )))

    parameter_error = float(np.max(np.abs(
        parameters - reference_parameters
    )))

    matrix_exact = bool(np.array_equal(
        reconstructed, reference
    ))

    parameters_match = bool(np.allclose(
        parameters,
        reference_parameters,
        atol=1e-12,
        rtol=0,
    ))

    if not matrix_exact or not parameters_match:
        raise RuntimeError(
            f"FAIL {scope}/{individual}: "
            f"matrix_error={matrix_error:.6e}, "
            f"parameter_error={parameter_error:.6e}"
        )

    print(
        f"PASS {scope}/{individual}: "
        f"cells={len(ordered)}, "
        f"windows={len(parameters)}, "
        f"matrix_error={matrix_error:.3e}, "
        f"parameter_error={parameter_error:.3e}"
    )

    return matrix_error, parameter_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    manifest = pd.read_csv(
        MANIFEST_FILE,
        dtype={"individual": str},
    )

    if len(manifest) != 48:
        raise RuntimeError(
            f"Expected 48 profiles, found {len(manifest)}"
        )

    if manifest.duplicated(
        ["scope", "individual"]
    ).any():
        raise RuntimeError("Duplicate manifest profiles")

    if not args.all:
        manifest = manifest.iloc[:1]

    state_space = load_state_space()

    errors = []

    for _, row in manifest.iterrows():
        errors.append(audit_profile(row, state_space))

    print()
    print("P25-B RECONSTRUCTION AUDIT: PASS")
    print("Profiles checked:", len(errors))
    print("Maximum matrix error:", max(x[0] for x in errors))
    print("Maximum parameter error:", max(x[1] for x in errors))


if __name__ == "__main__":
    main()
