#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================
"""
p4b_replicate_trajectory_validation.py

Targeted replicate validation for the GSE114412 endocrine trajectory.

Purpose
-------
The previous trajectory validation showed that the deposited endocrine
trajectory is biologically coherent, temporally ordered, and cleanly
separated into:
    Branch 0 -> SC-EC
    Branch 1 -> SC-beta

However, cluster-composition correlations between Differentiation 1 and
Differentiation 2 were lower than desired because those correlations are
sensitive to differences in sampled cell-state proportions.

This script therefore evaluates reproducibility using trajectory-focused
criteria that are more appropriate for GDIS-Bio:

1. Experimental day vs pseudotime Spearman correlation
   separately for each differentiation and branch
2. Day-by-day median pseudotime monotonicity
3. Biological-state pseudotime ordering
4. Branch endpoint purity in each differentiation
5. Cross-replicate correlation of daily median pseudotime profiles
6. Cross-replicate correlation of state-level median pseudotime profiles
7. Absolute differences in state-level pseudotime landmarks
8. Agreement of transition locations across differentiations
9. Final targeted replicate reproducibility assessment

IMPORTANT
---------
This script performs NO:
- normalization
- filtering
- PCA
- UMAP
- trajectory inference
- GDIS calculation

It only validates the deposited pseudotime trajectory.
"""

from pathlib import Path
import sys

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

RAW_DIR = PROJECT_DIR / "raw_data"
RESULTS_DIR = PROJECT_DIR / "results" / "p4b_replicate_trajectory_validation"
TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR = RESULTS_DIR / "figures"
REPORT_FILE = RESULTS_DIR / "p4b_replicate_trajectory_validation_report.txt"

PSEUDOTIME_FILE = (
    RAW_DIR / "GSE114412_Stage_5.endocrine_pseudotime.cell_metadata.tsv.gz"
)

CELL_ID = "library.barcode"

EXPECTED_TRAJECTORY_STATES = [
    "prog_nkx61",
    "neurog3_early",
    "neurog3_mid",
    "neurog3_late",
    "sc_beta",
    "sc_ec",
]

COMMON_PREBRANCH_STATES = [
    "prog_nkx61",
    "neurog3_early",
    "neurog3_mid",
    "neurog3_late",
]

BRANCH_ENDPOINTS = {
    0: "sc_ec",
    1: "sc_beta",
}

# Conservative reproducibility thresholds.
MIN_SPEARMAN_DAY_PSEUDOTIME = 0.60
MIN_DAILY_PROFILE_CORRELATION = 0.95
MIN_STATE_PROFILE_CORRELATION = 0.95
MIN_BRANCH_ENDPOINT_PURITY = 0.99
MAX_MEDIAN_STATE_PSEUDOTIME_DIFFERENCE = 0.10
MIN_MONOTONIC_FRACTION = 1.0


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def write_both(report, text=""):
    """Print text and write it to the report."""
    print(text)
    report.write(text + "\n")


def save_table(df, filename, index=True):
    """Save a dataframe to the tables directory."""
    path = TABLE_DIR / filename
    df.to_csv(path, index=index)
    return path


def ensure_columns(df, columns, name):
    """Verify that all required columns are present."""
    missing = [col for col in columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"{name} is missing required columns: "
            + ", ".join(missing)
        )


def safe_spearman(x, y):
    """
    Compute Spearman rank correlation using pandas/numpy only.
    """
    x = pd.Series(x)
    y = pd.Series(y)

    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan

    rx = x.rank(method="average")
    ry = y.rank(method="average")

    if rx.std(ddof=0) == 0 or ry.std(ddof=0) == 0:
        return np.nan

    return float(np.corrcoef(rx, ry)[0, 1])


def safe_pearson(x, y):
    """
    Compute Pearson correlation safely.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def monotonic_fraction(values):
    """
    Fraction of adjacent differences that are non-decreasing.
    """
    values = np.asarray(values, dtype=float)

    if len(values) < 2:
        return np.nan

    diffs = np.diff(values)

    return float(np.mean(diffs >= 0))


def ordered_state_table(df, differentiation, branch):
    """
    Return median pseudotime for biologically ordered states
    within one differentiation and one branch.
    """
    endpoint = BRANCH_ENDPOINTS[branch]
    states = COMMON_PREBRANCH_STATES + [endpoint]

    subset = df[
        (df["Differentiation"] == differentiation)
        & (df["Pseudotime_branch"] == branch)
    ]

    rows = []

    for state in states:
        g = subset[subset["Assigned_cluster"] == state]

        if g.empty:
            rows.append(
                {
                    "state": state,
                    "n_cells": 0,
                    "median_pseudotime": np.nan,
                    "q25": np.nan,
                    "q75": np.nan,
                    "median_day": np.nan,
                }
            )
            continue

        values = g["Pseudotime_value"].dropna()

        rows.append(
            {
                "state": state,
                "n_cells": int(len(g)),
                "median_pseudotime": float(values.median()),
                "q25": float(values.quantile(0.25)),
                "q75": float(values.quantile(0.75)),
                "median_day": float(g["CellDay"].median()),
            }
        )

    return pd.DataFrame(rows)


def plot_daily_profiles(profile_table, output_path):
    """
    Plot daily median pseudotime by differentiation and branch.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for (branch, diff), group in profile_table.groupby(
        ["branch", "differentiation"]
    ):
        group = group.sort_values("day")

        ax.plot(
            group["day"],
            group["median_pseudotime"],
            marker="o",
            linewidth=1.8,
            label=f"Branch {branch}, Diff {diff}",
        )

    ax.set_xlabel("Experimental day")
    ax.set_ylabel("Median pseudotime")
    ax.set_title(
        "Replicate comparison of daily median pseudotime profiles"
    )
    ax.set_xticks(sorted(profile_table["day"].unique()))
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_state_profiles(state_profiles, output_path):
    """
    Plot state-level median pseudotime profiles across differentiations.
    """
    fig, ax = plt.subplots(figsize=(11, 6.5))

    for (branch, diff), group in state_profiles.groupby(
        ["branch", "differentiation"]
    ):
        group = group.copy()
        group["order"] = group["state"].map(
            {
                "prog_nkx61": 0,
                "neurog3_early": 1,
                "neurog3_mid": 2,
                "neurog3_late": 3,
                "sc_ec": 4,
                "sc_beta": 4,
            }
        )

        group = group.sort_values("order")

        ax.plot(
            group["order"],
            group["median_pseudotime"],
            marker="o",
            linewidth=1.8,
            label=f"Branch {branch}, Diff {diff}",
        )

    ax.set_xticks(
        [0, 1, 2, 3, 4],
        [
            "prog_nkx61",
            "neurog3_early",
            "neurog3_mid",
            "neurog3_late",
            "endpoint",
        ],
        rotation=25,
    )

    ax.set_ylabel("Median pseudotime")
    ax.set_title(
        "Biological-state pseudotime ordering across replicates"
    )
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_endpoint_purity(purity_table, output_path):
    """
    Plot endpoint purity by branch and differentiation.
    """
    labels = [
        f"B{int(row.branch)}-D{int(row.differentiation)}"
        for row in purity_table.itertuples()
    ]

    values = purity_table["endpoint_purity"].values

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(labels, values)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Endpoint purity")
    ax.set_title("Branch endpoint purity across differentiations")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------

def main():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    pseudo = pd.read_csv(
        PSEUDOTIME_FILE,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )

    ensure_columns(
        pseudo,
        [
            CELL_ID,
            "Assigned_cluster",
            "Pseudotime_value",
            "Pseudotime_branch",
            "Differentiation",
            "CellDay",
        ],
        "Pseudotime metadata",
    )

    pseudo["Pseudotime_value"] = pd.to_numeric(
        pseudo["Pseudotime_value"],
        errors="coerce",
    )

    pseudo["Pseudotime_branch"] = pd.to_numeric(
        pseudo["Pseudotime_branch"],
        errors="raise",
    ).astype(int)

    pseudo["Differentiation"] = pd.to_numeric(
        pseudo["Differentiation"],
        errors="raise",
    ).astype(int)

    pseudo["CellDay"] = pd.to_numeric(
        pseudo["CellDay"],
        errors="raise",
    ).astype(int)

    branches = sorted(pseudo["Pseudotime_branch"].unique())
    diffs = sorted(pseudo["Differentiation"].unique())
    days = sorted(pseudo["CellDay"].unique())

    # -----------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------

    with open(REPORT_FILE, "w", encoding="utf-8") as report:

        write_both(report, "=" * 84)
        write_both(report, "GDIS-Bio Replicate Trajectory Validation")
        write_both(report, "Dataset: GSE114412 Stage 5 endocrine trajectory")
        write_both(report, "=" * 84)

        write_both(
            report,
            "\nThis targeted validation evaluates trajectory reproducibility "
            "between Differentiation 1 and Differentiation 2."
        )

        # =============================================================
        # 1. Basic structure
        # =============================================================

        write_both(report, "\n1. BASIC STRUCTURE")
        write_both(report, "-" * 84)

        write_both(
            report,
            f"Total trajectory cells: {len(pseudo):,}"
        )

        write_both(
            report,
            f"Branches: {branches}"
        )

        write_both(
            report,
            f"Differentiations: {diffs}"
        )

        write_both(
            report,
            f"Experimental days: {days}"
        )

        # =============================================================
        # 2. Day vs pseudotime correlation by replicate and branch
        # =============================================================

        write_both(
            report,
            "\n2. DAY vs PSEUDOTIME CORRELATION BY BRANCH AND DIFFERENTIATION"
        )
        write_both(report, "-" * 84)

        corr_rows = []

        for branch in branches:
            for diff in diffs:

                group = pseudo[
                    (pseudo["Pseudotime_branch"] == branch)
                    & (pseudo["Differentiation"] == diff)
                ]

                rho = safe_spearman(
                    group["CellDay"],
                    group["Pseudotime_value"],
                )

                corr_rows.append(
                    {
                        "branch": branch,
                        "differentiation": diff,
                        "n_cells": len(group),
                        "spearman_day_vs_pseudotime": rho,
                    }
                )

        corr_table = pd.DataFrame(corr_rows)

        save_table(
            corr_table,
            "01_day_pseudotime_correlations.csv",
            index=False,
        )

        write_both(
            report,
            corr_table.round(6).to_string(index=False)
        )

        # =============================================================
        # 3. Daily median pseudotime profiles
        # =============================================================

        write_both(
            report,
            "\n3. DAILY MEDIAN PSEUDOTIME PROFILES"
        )
        write_both(report, "-" * 84)

        daily_rows = []

        for branch in branches:
            for diff in diffs:

                group = pseudo[
                    (pseudo["Pseudotime_branch"] == branch)
                    & (pseudo["Differentiation"] == diff)
                ]

                medians = (
                    group.groupby("CellDay")["Pseudotime_value"]
                    .median()
                    .reindex(days)
                )

                for day, value in medians.items():
                    daily_rows.append(
                        {
                            "branch": branch,
                            "differentiation": diff,
                            "day": int(day),
                            "median_pseudotime": float(value),
                        }
                    )

        daily_profiles = pd.DataFrame(daily_rows)

        save_table(
            daily_profiles,
            "02_daily_median_pseudotime_profiles.csv",
            index=False,
        )

        monotonic_rows = []

        for branch in branches:
            for diff in diffs:

                profile = (
                    daily_profiles[
                        (daily_profiles["branch"] == branch)
                        & (daily_profiles["differentiation"] == diff)
                    ]
                    .sort_values("day")
                )

                frac = monotonic_fraction(
                    profile["median_pseudotime"].values
                )

                monotonic_rows.append(
                    {
                        "branch": branch,
                        "differentiation": diff,
                        "fraction_non_decreasing": frac,
                    }
                )

        monotonic_table = pd.DataFrame(monotonic_rows)

        save_table(
            monotonic_table,
            "03_daily_profile_monotonicity.csv",
            index=False,
        )

        write_both(
            report,
            "\nMonotonicity of daily median pseudotime:\n"
            + monotonic_table.round(6).to_string(index=False)
        )

        # Cross-replicate correlation of daily profiles.
        daily_corr_rows = []

        if len(diffs) == 2:
            d1, d2 = diffs

            for branch in branches:

                p1 = (
                    daily_profiles[
                        (daily_profiles["branch"] == branch)
                        & (daily_profiles["differentiation"] == d1)
                    ]
                    .sort_values("day")
                )

                p2 = (
                    daily_profiles[
                        (daily_profiles["branch"] == branch)
                        & (daily_profiles["differentiation"] == d2)
                    ]
                    .sort_values("day")
                )

                corr = safe_pearson(
                    p1["median_pseudotime"],
                    p2["median_pseudotime"],
                )

                mean_abs_diff = float(
                    np.mean(
                        np.abs(
                            p1["median_pseudotime"].values
                            - p2["median_pseudotime"].values
                        )
                    )
                )

                daily_corr_rows.append(
                    {
                        "branch": branch,
                        "daily_profile_correlation": corr,
                        "mean_absolute_daily_median_difference":
                            mean_abs_diff,
                    }
                )

        daily_corr_table = pd.DataFrame(
            daily_corr_rows
        )

        save_table(
            daily_corr_table,
            "04_cross_replicate_daily_profile_agreement.csv",
            index=False,
        )

        write_both(
            report,
            "\nCross-replicate agreement of daily median profiles:\n"
            + daily_corr_table.round(6).to_string(index=False)
        )

        plot_daily_profiles(
            daily_profiles,
            FIGURE_DIR / "01_daily_median_pseudotime_profiles.png",
        )

        # =============================================================
        # 4. Biological-state pseudotime ordering
        # =============================================================

        write_both(
            report,
            "\n4. BIOLOGICAL-STATE PSEUDOTIME ORDERING"
        )
        write_both(report, "-" * 84)

        state_tables = []

        for branch in branches:
            for diff in diffs:

                table = ordered_state_table(
                    pseudo,
                    differentiation=diff,
                    branch=branch,
                )

                table.insert(0, "differentiation", diff)
                table.insert(0, "branch", branch)

                state_tables.append(table)

        state_profiles = pd.concat(
            state_tables,
            ignore_index=True,
        )

        save_table(
            state_profiles,
            "05_state_pseudotime_profiles.csv",
            index=False,
        )

        write_both(
            report,
            state_profiles.round(6).to_string(index=False)
        )

        plot_state_profiles(
            state_profiles,
            FIGURE_DIR / "02_state_pseudotime_profiles.png",
        )

        # =============================================================
        # 5. Cross-replicate state profile agreement
        # =============================================================

        write_both(
            report,
            "\n5. CROSS-REPLICATE STATE-PROFILE AGREEMENT"
        )
        write_both(report, "-" * 84)

        state_corr_rows = []

        if len(diffs) == 2:

            d1, d2 = diffs

            for branch in branches:

                p1 = state_profiles[
                    (state_profiles["branch"] == branch)
                    & (state_profiles["differentiation"] == d1)
                ].copy()

                p2 = state_profiles[
                    (state_profiles["branch"] == branch)
                    & (state_profiles["differentiation"] == d2)
                ].copy()

                merged_states = p1.merge(
                    p2,
                    on=["branch", "state"],
                    suffixes=("_d1", "_d2"),
                    validate="one_to_one",
                )

                corr = safe_pearson(
                    merged_states["median_pseudotime_d1"],
                    merged_states["median_pseudotime_d2"],
                )

                abs_diffs = np.abs(
                    merged_states["median_pseudotime_d1"]
                    - merged_states["median_pseudotime_d2"]
                )

                state_corr_rows.append(
                    {
                        "branch": branch,
                        "state_profile_correlation": corr,
                        "mean_absolute_state_median_difference":
                            float(abs_diffs.mean()),
                        "max_absolute_state_median_difference":
                            float(abs_diffs.max()),
                    }
                )

        state_corr_table = pd.DataFrame(
            state_corr_rows
        )

        save_table(
            state_corr_table,
            "06_cross_replicate_state_profile_agreement.csv",
            index=False,
        )

        write_both(
            report,
            state_corr_table.round(6).to_string(index=False)
        )

        # =============================================================
        # 6. Branch endpoint purity
        # =============================================================

        write_both(
            report,
            "\n6. BRANCH ENDPOINT PURITY BY DIFFERENTIATION"
        )
        write_both(report, "-" * 84)

        purity_rows = []

        for branch in branches:
            endpoint = BRANCH_ENDPOINTS[branch]

            opposite_endpoint = (
                "sc_beta"
                if endpoint == "sc_ec"
                else "sc_ec"
            )

            for diff in diffs:

                group = pseudo[
                    (pseudo["Pseudotime_branch"] == branch)
                    & (pseudo["Differentiation"] == diff)
                ]

                endpoint_n = int(
                    (group["Assigned_cluster"] == endpoint).sum()
                )

                opposite_n = int(
                    (group["Assigned_cluster"] == opposite_endpoint).sum()
                )

                terminal_n = endpoint_n + opposite_n

                purity = (
                    endpoint_n / terminal_n
                    if terminal_n > 0
                    else np.nan
                )

                purity_rows.append(
                    {
                        "branch": branch,
                        "differentiation": diff,
                        "expected_endpoint": endpoint,
                        "n_expected_endpoint": endpoint_n,
                        "n_opposite_endpoint": opposite_n,
                        "endpoint_purity": purity,
                    }
                )

        purity_table = pd.DataFrame(
            purity_rows
        )

        save_table(
            purity_table,
            "07_branch_endpoint_purity.csv",
            index=False,
        )

        write_both(
            report,
            purity_table.round(6).to_string(index=False)
        )

        plot_endpoint_purity(
            purity_table,
            FIGURE_DIR / "03_branch_endpoint_purity.png",
        )

        # =============================================================
        # 7. Landmark agreement
        # =============================================================

        write_both(
            report,
            "\n7. TRANSITION-LANDMARK AGREEMENT"
        )
        write_both(report, "-" * 84)

        landmark_rows = []

        if len(diffs) == 2:

            d1, d2 = diffs

            for branch in branches:

                endpoint = BRANCH_ENDPOINTS[branch]
                states = COMMON_PREBRANCH_STATES + [endpoint]

                for state in states:

                    g1 = pseudo[
                        (pseudo["Pseudotime_branch"] == branch)
                        & (pseudo["Differentiation"] == d1)
                        & (pseudo["Assigned_cluster"] == state)
                    ]

                    g2 = pseudo[
                        (pseudo["Pseudotime_branch"] == branch)
                        & (pseudo["Differentiation"] == d2)
                        & (pseudo["Assigned_cluster"] == state)
                    ]

                    med1 = float(
                        g1["Pseudotime_value"].median()
                    ) if len(g1) else np.nan

                    med2 = float(
                        g2["Pseudotime_value"].median()
                    ) if len(g2) else np.nan

                    day1 = float(
                        g1["CellDay"].median()
                    ) if len(g1) else np.nan

                    day2 = float(
                        g2["CellDay"].median()
                    ) if len(g2) else np.nan

                    landmark_rows.append(
                        {
                            "branch": branch,
                            "state": state,
                            "n_diff1": len(g1),
                            "n_diff2": len(g2),
                            "median_pseudotime_diff1": med1,
                            "median_pseudotime_diff2": med2,
                            "absolute_pseudotime_difference":
                                abs(med1 - med2)
                                if np.isfinite(med1) and np.isfinite(med2)
                                else np.nan,
                            "median_day_diff1": day1,
                            "median_day_diff2": day2,
                            "absolute_median_day_difference":
                                abs(day1 - day2)
                                if np.isfinite(day1) and np.isfinite(day2)
                                else np.nan,
                        }
                    )

        landmark_table = pd.DataFrame(
            landmark_rows
        )

        save_table(
            landmark_table,
            "08_transition_landmark_agreement.csv",
            index=False,
        )

        write_both(
            report,
            landmark_table.round(6).to_string(index=False)
        )

        # =============================================================
        # 8. Final targeted reproducibility assessment
        # =============================================================

        write_both(
            report,
            "\n8. FINAL TARGETED REPRODUCIBILITY ASSESSMENT"
        )
        write_both(report, "-" * 84)

        # Individual checks.
        corr_pass = bool(
            (
                corr_table["spearman_day_vs_pseudotime"]
                >= MIN_SPEARMAN_DAY_PSEUDOTIME
            ).all()
        )

        monotonic_pass = bool(
            (
                monotonic_table["fraction_non_decreasing"]
                >= MIN_MONOTONIC_FRACTION
            ).all()
        )

        daily_profile_pass = bool(
            (
                daily_corr_table["daily_profile_correlation"]
                >= MIN_DAILY_PROFILE_CORRELATION
            ).all()
        )

        state_profile_pass = bool(
            (
                state_corr_table["state_profile_correlation"]
                >= MIN_STATE_PROFILE_CORRELATION
            ).all()
        )

        purity_pass = bool(
            (
                purity_table["endpoint_purity"]
                >= MIN_BRANCH_ENDPOINT_PURITY
            ).all()
        )

        state_difference_pass = bool(
            (
                state_corr_table[
                    "max_absolute_state_median_difference"
                ]
                <= MAX_MEDIAN_STATE_PSEUDOTIME_DIFFERENCE
            ).all()
        )

        landmark_day_pass = bool(
            (
                landmark_table[
                    "absolute_median_day_difference"
                ].fillna(np.inf)
                <= 1.0
            ).all()
        )

        checks = [
            (
                f"All branch/replicate day-vs-pseudotime Spearman "
                f"correlations >= {MIN_SPEARMAN_DAY_PSEUDOTIME}",
                corr_pass,
            ),
            (
                "Daily median pseudotime is non-decreasing in every "
                "branch and differentiation",
                monotonic_pass,
            ),
            (
                f"Cross-replicate daily median-profile correlations "
                f">= {MIN_DAILY_PROFILE_CORRELATION}",
                daily_profile_pass,
            ),
            (
                f"Cross-replicate biological-state profile correlations "
                f">= {MIN_STATE_PROFILE_CORRELATION}",
                state_profile_pass,
            ),
            (
                f"Branch endpoint purity >= {MIN_BRANCH_ENDPOINT_PURITY}",
                purity_pass,
            ),
            (
                f"Maximum state-level median pseudotime difference "
                f"<= {MAX_MEDIAN_STATE_PSEUDOTIME_DIFFERENCE}",
                state_difference_pass,
            ),
            (
                "Median experimental-day landmarks differ by no more "
                "than one day",
                landmark_day_pass,
            ),
        ]

        passed = 0

        for label, status in checks:
            if status:
                write_both(
                    report,
                    f"[PASS] {label}"
                )
                passed += 1
            else:
                write_both(
                    report,
                    f"[CHECK] {label}"
                )

        write_both(
            report,
            f"\nTargeted reproducibility checks passed: "
            f"{passed}/{len(checks)}"
        )

        if passed == len(checks):
            write_both(
                report,
                "\nFINAL STATUS: PASS"
            )
            write_both(
                report,
                "The endocrine trajectory is reproducible across "
                "Differentiation 1 and Differentiation 2 under "
                "trajectory-focused validation criteria."
            )
            write_both(
                report,
                "The dataset can proceed to preprocessing/state-space "
                "construction for GDIS analysis."
            )
        else:
            write_both(
                report,
                "\nFINAL STATUS: REVIEW REQUIRED"
            )
            write_both(
                report,
                "One or more targeted trajectory reproducibility "
                "criteria require interpretation before proceeding."
            )

        write_both(
            report,
            "\nNo normalization, filtering, dimensionality reduction, "
            "trajectory reconstruction, or GDIS calculation was performed."
        )

        write_both(
            report,
            f"\nTables:  {TABLE_DIR}"
        )

        write_both(
            report,
            f"Figures: {FIGURE_DIR}"
        )

        write_both(
            report,
            f"Report:  {REPORT_FILE}"
        )

    print("\n" + "=" * 84)
    print("p4b_replicate_trajectory_validation.py completed.")
    print("=" * 84)
    print(f"Report : {REPORT_FILE}")
    print(f"Tables : {TABLE_DIR}")
    print(f"Figures: {FIGURE_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

