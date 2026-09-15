#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================
"""
p4_endocrine_trajectory_validation.py

Validate the deposited endocrine trajectory for GSE114412 Stage 5
before any new normalization, trajectory reconstruction, or GDIS analysis.

Purpose
-------
This script validates whether the 18,099-cell endocrine pseudotime subset
is suitable as the PRIMARY trajectory for GDIS-Bio.

It evaluates:

1. CellWeek (main metadata) vs CellDay (pseudotime metadata)
2. Pseudotime range and distribution by experimental day
3. Pseudotime distribution by biological cluster
4. Pseudotime branch composition
5. Branch enrichment for SC-beta and SC-EC outcomes
6. Branch-specific temporal progression
7. Differentiation 1 vs 2 consistency within branches
8. Monotonic association between pseudotime and experimental day
9. Transition landmarks along pseudotime
10. Final suitability assessment for downstream GDIS analysis

IMPORTANT
---------
This script performs NO:
- normalization
- filtering
- PCA
- UMAP
- trajectory inference
- GDIS calculation

It evaluates only the deposited metadata and trajectory annotations.
"""

from pathlib import Path
import sys
import math

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

RESULTS_DIR = PROJECT_DIR / "results" / "p4_endocrine_trajectory_validation"
TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR = RESULTS_DIR / "figures"
REPORT_FILE = RESULTS_DIR / "p4_endocrine_trajectory_validation_report.txt"

MAIN_METADATA_FILE = (
    RAW_DIR / "GSE114412_Stage_5.all.cell_metadata.tsv.gz"
)

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

BETA_STATE = "sc_beta"
EC_STATE = "sc_ec"


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def write_both(report, text=""):
    """Print text and write it to the report."""
    print(text)
    report.write(text + "\n")


def save_table(df, filename, index=True):
    """Save a dataframe as CSV."""
    path = TABLE_DIR / filename
    df.to_csv(path, index=index)
    return path


def ensure_columns(df, columns, name):
    """Ensure required columns are available."""
    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"{name} is missing required columns: "
            + ", ".join(missing)
        )


def safe_spearman(x, y):
    """
    Compute Spearman correlation using pandas ranks only.

    Avoids requiring scipy for this validation script.
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


def quantile_summary(grouped, value_column):
    """Return min, quartiles, median, mean, max, and count."""
    rows = []

    for key, group in grouped:
        values = pd.to_numeric(
            group[value_column],
            errors="coerce"
        ).dropna()

        if len(values) == 0:
            continue

        q = values.quantile([0.25, 0.50, 0.75])

        if isinstance(key, tuple):
            row = {f"group_{i+1}": v for i, v in enumerate(key)}
        else:
            row = {"group": key}

        row.update(
            {
                "n_cells": int(len(values)),
                "min": float(values.min()),
                "q25": float(q.loc[0.25]),
                "median": float(q.loc[0.50]),
                "mean": float(values.mean()),
                "q75": float(q.loc[0.75]),
                "max": float(values.max()),
            }
        )

        rows.append(row)

    return pd.DataFrame(rows)


def plot_pseudotime_by_day(df, output_path):
    """Boxplot of pseudotime by experimental day."""
    days = sorted(df["CellDay"].unique())

    data = [
        df.loc[df["CellDay"] == day, "Pseudotime_value"].dropna().values
        for day in days
    ]

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.boxplot(
        data,
        tick_labels=[str(day) for day in days],
        showfliers=False,
    )

    ax.set_xlabel("Experimental day")
    ax.set_ylabel("Deposited pseudotime")
    ax.set_title("Endocrine pseudotime distribution by experimental day")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pseudotime_by_cluster(df, output_path):
    """Boxplot of pseudotime by biological cluster."""
    states = [
        state for state in EXPECTED_TRAJECTORY_STATES
        if state in set(df["Assigned_cluster"])
    ]

    data = [
        df.loc[
            df["Assigned_cluster"] == state,
            "Pseudotime_value"
        ].dropna().values
        for state in states
    ]

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.boxplot(
        data,
        tick_labels=states,
        showfliers=False,
    )

    ax.set_xlabel("Biological state")
    ax.set_ylabel("Deposited pseudotime")
    ax.set_title("Pseudotime distribution across endocrine trajectory states")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_branch_composition(branch_props, output_path):
    """Stacked bar plot of branch composition."""
    fig, ax = plt.subplots(figsize=(9, 6))

    branch_props.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        width=0.75,
    )

    ax.set_xlabel("Pseudotime branch")
    ax.set_ylabel("Fraction of branch cells")
    ax.set_title("Biological-state composition of pseudotime branches")
    ax.legend(
        title="Assigned cluster",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_branch_day_distribution(branch_day_props, output_path):
    """Grouped line plot of branch membership across days."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for branch in branch_day_props.columns:
        ax.plot(
            branch_day_props.index,
            branch_day_props[branch],
            marker="o",
            linewidth=1.8,
            label=f"Branch {branch}",
        )

    ax.set_xlabel("Experimental day")
    ax.set_ylabel("Fraction of cells")
    ax.set_title("Pseudotime branch composition across experimental days")
    ax.set_xticks(branch_day_props.index)
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_branch_fate_enrichment(fate_table, output_path):
    """Bar plot for beta/ec enrichment by branch."""
    plot_df = fate_table[
        ["sc_beta_fraction", "sc_ec_fraction"]
    ].copy()

    fig, ax = plt.subplots(figsize=(8, 5.5))

    plot_df.plot(kind="bar", ax=ax)

    ax.set_xlabel("Pseudotime branch")
    ax.set_ylabel("Fraction of branch cells")
    ax.set_title("SC-beta and SC-EC enrichment by pseudotime branch")
    ax.legend(["SC-beta", "SC-EC"])
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

    # -----------------------------------------------------------------
    # Load metadata
    # -----------------------------------------------------------------

    main_meta = pd.read_csv(
        MAIN_METADATA_FILE,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )

    pseudo = pd.read_csv(
        PSEUDOTIME_FILE,
        sep="\t",
        compression="gzip",
        low_memory=False,
    )

    ensure_columns(
        main_meta,
        [
            CELL_ID,
            "Assigned_cluster",
            "Assigned_subcluster",
            "Differentiation",
            "CellWeek",
            "Lib_prep_batch",
        ],
        "Main metadata",
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
            "Lib_prep_batch",
        ],
        "Pseudotime metadata",
    )

    main_meta["CellWeek"] = pd.to_numeric(
        main_meta["CellWeek"],
        errors="raise",
    ).astype(int)

    pseudo["CellDay"] = pd.to_numeric(
        pseudo["CellDay"],
        errors="raise",
    ).astype(int)

    pseudo["Differentiation"] = pd.to_numeric(
        pseudo["Differentiation"],
        errors="raise",
    ).astype(int)

    pseudo["Pseudotime_value"] = pd.to_numeric(
        pseudo["Pseudotime_value"],
        errors="coerce",
    )

    # Normalize branch labels while preserving deposited values.
    pseudo["Pseudotime_branch"] = pd.to_numeric(
        pseudo["Pseudotime_branch"],
        errors="raise",
    ).astype(int)

    # Merge the main experimental-day annotation.
    merged = pseudo.merge(
        main_meta[
            [
                CELL_ID,
                "CellWeek",
                "Assigned_cluster",
                "Assigned_subcluster",
                "Differentiation",
            ]
        ].rename(
            columns={
                "Assigned_cluster": "Main_Assigned_cluster",
                "Assigned_subcluster": "Main_Assigned_subcluster",
                "Differentiation": "Main_Differentiation",
            }
        ),
        on=CELL_ID,
        how="left",
        validate="one_to_one",
    )

    # -----------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------

    with open(REPORT_FILE, "w", encoding="utf-8") as report:

        write_both(report, "=" * 82)
        write_both(report, "GDIS-Bio Endocrine Trajectory Validation")
        write_both(report, "Dataset: GSE114412 Stage 5")
        write_both(report, "=" * 82)

        # =============================================================
        # 1. Basic structure
        # =============================================================

        write_both(report, "\n1. ENDOCRINE TRAJECTORY STRUCTURE")
        write_both(report, "-" * 82)

        write_both(
            report,
            f"Trajectory cells: {len(pseudo):,}"
        )

        write_both(
            report,
            f"Experimental days: "
            f"{sorted(pseudo['CellDay'].unique().tolist())}"
        )

        write_both(
            report,
            f"Branches: "
            f"{sorted(pseudo['Pseudotime_branch'].unique().tolist())}"
        )

        write_both(
            report,
            f"Differentiations: "
            f"{sorted(pseudo['Differentiation'].unique().tolist())}"
        )

        cluster_counts = (
            pseudo["Assigned_cluster"]
            .value_counts()
            .sort_values(ascending=False)
        )

        write_both(report, "\nTrajectory cluster counts:")

        for cluster, count in cluster_counts.items():
            write_both(
                report,
                f"  {cluster:25s} {count:8,d}"
            )

        # =============================================================
        # 2. CellDay vs CellWeek
        # =============================================================

        write_both(
            report,
            "\n2. EXPERIMENTAL TIME AGREEMENT: CellDay vs CellWeek"
        )
        write_both(report, "-" * 82)

        missing_week = int(merged["CellWeek"].isna().sum())

        merged["day_match"] = (
            merged["CellDay"] == merged["CellWeek"]
        )

        exact_matches = int(merged["day_match"].sum())
        match_fraction = exact_matches / len(merged)

        write_both(
            report,
            f"Cells missing CellWeek after merge: {missing_week:,}"
        )

        write_both(
            report,
            f"Exact CellDay == CellWeek matches: "
            f"{exact_matches:,}/{len(merged):,} "
            f"({match_fraction * 100:.2f}%)"
        )

        day_cross = pd.crosstab(
            merged["CellDay"],
            merged["CellWeek"],
        )

        save_table(
            day_cross,
            "01_cellday_vs_cellweek.csv",
            index=True,
        )

        write_both(
            report,
            "\nCellDay x CellWeek cross-tabulation:\n"
            + day_cross.to_string()
        )

        # =============================================================
        # 3. Annotation consistency
        # =============================================================

        write_both(report, "\n3. ANNOTATION CONSISTENCY")
        write_both(report, "-" * 82)

        cluster_match = (
            merged["Assigned_cluster"]
            == merged["Main_Assigned_cluster"]
        )

        diff_match = (
            merged["Differentiation"]
            == merged["Main_Differentiation"]
        )

        write_both(
            report,
            f"Cluster annotation agreement: "
            f"{cluster_match.mean() * 100:.2f}%"
        )

        write_both(
            report,
            f"Differentiation annotation agreement: "
            f"{diff_match.mean() * 100:.2f}%"
        )

        # =============================================================
        # 4. Pseudotime by day
        # =============================================================

        write_both(report, "\n4. PSEUDOTIME BY EXPERIMENTAL DAY")
        write_both(report, "-" * 82)

        day_summary = quantile_summary(
            pseudo.groupby("CellDay"),
            "Pseudotime_value",
        ).rename(columns={"group": "day"})

        save_table(
            day_summary,
            "02_pseudotime_summary_by_day.csv",
            index=False,
        )

        write_both(
            report,
            day_summary.round(6).to_string(index=False)
        )

        rho_day = safe_spearman(
            pseudo["CellDay"],
            pseudo["Pseudotime_value"],
        )

        write_both(
            report,
            f"\nSpearman correlation "
            f"(experimental day vs pseudotime): {rho_day:.6f}"
        )

        plot_pseudotime_by_day(
            pseudo,
            FIGURE_DIR / "01_pseudotime_by_day.png",
        )

        # =============================================================
        # 5. Pseudotime by biological state
        # =============================================================

        write_both(
            report,
            "\n5. PSEUDOTIME BY BIOLOGICAL STATE"
        )
        write_both(report, "-" * 82)

        cluster_summary = quantile_summary(
            pseudo.groupby("Assigned_cluster"),
            "Pseudotime_value",
        ).rename(columns={"group": "cluster"})

        expected_order = {
            state: i
            for i, state in enumerate(EXPECTED_TRAJECTORY_STATES)
        }

        cluster_summary["expected_order"] = (
            cluster_summary["cluster"]
            .map(expected_order)
            .fillna(999)
        )

        cluster_summary = (
            cluster_summary
            .sort_values(
                ["expected_order", "median"]
            )
            .drop(columns=["expected_order"])
            .reset_index(drop=True)
        )

        save_table(
            cluster_summary,
            "03_pseudotime_summary_by_cluster.csv",
            index=False,
        )

        write_both(
            report,
            cluster_summary.round(6).to_string(index=False)
        )

        plot_pseudotime_by_cluster(
            pseudo,
            FIGURE_DIR / "02_pseudotime_by_cluster.png",
        )

        # =============================================================
        # 6. Branch composition
        # =============================================================

        write_both(report, "\n6. BRANCH COMPOSITION")
        write_both(report, "-" * 82)

        branch_counts = pd.crosstab(
            pseudo["Pseudotime_branch"],
            pseudo["Assigned_cluster"],
        )

        branch_props = branch_counts.div(
            branch_counts.sum(axis=1),
            axis=0,
        )

        save_table(
            branch_counts,
            "04_branch_cluster_counts.csv",
            index=True,
        )

        save_table(
            branch_props,
            "05_branch_cluster_proportions.csv",
            index=True,
        )

        write_both(
            report,
            "\nBranch cluster counts:\n"
            + branch_counts.to_string()
        )

        write_both(
            report,
            "\nBranch cluster proportions:\n"
            + branch_props.round(4).to_string()
        )

        plot_branch_composition(
            branch_props,
            FIGURE_DIR / "03_branch_cluster_composition.png",
        )

        # =============================================================
        # 7. SC-beta vs SC-EC fate enrichment
        # =============================================================

        write_both(
            report,
            "\n7. SC-BETA vs SC-EC ENRICHMENT BY BRANCH"
        )
        write_both(report, "-" * 82)

        fate_rows = []

        for branch, group in pseudo.groupby("Pseudotime_branch"):

            n = len(group)

            beta_n = int(
                (group["Assigned_cluster"] == BETA_STATE).sum()
            )

            ec_n = int(
                (group["Assigned_cluster"] == EC_STATE).sum()
            )

            endocrine_outcome_n = beta_n + ec_n

            beta_fraction = beta_n / n if n else np.nan
            ec_fraction = ec_n / n if n else np.nan

            if endocrine_outcome_n > 0:
                beta_within_outcome = beta_n / endocrine_outcome_n
                ec_within_outcome = ec_n / endocrine_outcome_n
            else:
                beta_within_outcome = np.nan
                ec_within_outcome = np.nan

            fate_rows.append(
                {
                    "branch": branch,
                    "n_cells": n,
                    "sc_beta_n": beta_n,
                    "sc_ec_n": ec_n,
                    "sc_beta_fraction": beta_fraction,
                    "sc_ec_fraction": ec_fraction,
                    "beta_fraction_within_beta_ec": beta_within_outcome,
                    "ec_fraction_within_beta_ec": ec_within_outcome,
                }
            )

        fate_table = (
            pd.DataFrame(fate_rows)
            .set_index("branch")
            .sort_index()
        )

        save_table(
            fate_table,
            "06_branch_beta_ec_enrichment.csv",
            index=True,
        )

        write_both(
            report,
            fate_table.round(4).to_string()
        )

        plot_branch_fate_enrichment(
            fate_table,
            FIGURE_DIR / "04_branch_beta_ec_enrichment.png",
        )

        # =============================================================
        # 8. Branch distribution by day
        # =============================================================

        write_both(
            report,
            "\n8. BRANCH DISTRIBUTION BY EXPERIMENTAL DAY"
        )
        write_both(report, "-" * 82)

        branch_day_counts = pd.crosstab(
            pseudo["CellDay"],
            pseudo["Pseudotime_branch"],
        )

        branch_day_props = branch_day_counts.div(
            branch_day_counts.sum(axis=1),
            axis=0,
        )

        save_table(
            branch_day_counts,
            "07_branch_counts_by_day.csv",
            index=True,
        )

        save_table(
            branch_day_props,
            "08_branch_proportions_by_day.csv",
            index=True,
        )

        write_both(
            report,
            "\nBranch counts by day:\n"
            + branch_day_counts.to_string()
        )

        plot_branch_day_distribution(
            branch_day_props,
            FIGURE_DIR / "05_branch_distribution_by_day.png",
        )

        # =============================================================
        # 9. Branch x differentiation reproducibility
        # =============================================================

        write_both(
            report,
            "\n9. BRANCH x DIFFERENTIATION REPRODUCIBILITY"
        )
        write_both(report, "-" * 82)

        branch_diff_counts = pd.crosstab(
            pseudo["Pseudotime_branch"],
            pseudo["Differentiation"],
        )

        branch_diff_props = branch_diff_counts.div(
            branch_diff_counts.sum(axis=1),
            axis=0,
        )

        save_table(
            branch_diff_counts,
            "09_branch_differentiation_counts.csv",
            index=True,
        )

        save_table(
            branch_diff_props,
            "10_branch_differentiation_proportions.csv",
            index=True,
        )

        write_both(
            report,
            "\nBranch x differentiation counts:\n"
            + branch_diff_counts.to_string()
        )

        # Detailed cluster composition by branch and differentiation.
        branch_diff_cluster = pd.crosstab(
            [
                pseudo["Pseudotime_branch"],
                pseudo["Differentiation"],
            ],
            pseudo["Assigned_cluster"],
        )

        save_table(
            branch_diff_cluster,
            "11_branch_differentiation_cluster_counts.csv",
            index=True,
        )

        # Compare cluster composition between differentiations within branch.
        reproducibility_rows = []

        branches = sorted(
            pseudo["Pseudotime_branch"].unique()
        )

        diffs = sorted(
            pseudo["Differentiation"].unique()
        )

        if len(diffs) == 2:

            d1, d2 = diffs

            for branch in branches:

                g1 = pseudo[
                    (pseudo["Pseudotime_branch"] == branch)
                    & (pseudo["Differentiation"] == d1)
                ]

                g2 = pseudo[
                    (pseudo["Pseudotime_branch"] == branch)
                    & (pseudo["Differentiation"] == d2)
                ]

                p1 = (
                    g1["Assigned_cluster"]
                    .value_counts(normalize=True)
                )

                p2 = (
                    g2["Assigned_cluster"]
                    .value_counts(normalize=True)
                )

                states = sorted(
                    set(p1.index).union(p2.index)
                )

                v1 = np.array(
                    [p1.get(s, 0.0) for s in states],
                    dtype=float,
                )

                v2 = np.array(
                    [p2.get(s, 0.0) for s in states],
                    dtype=float,
                )

                if (
                    len(states) >= 2
                    and np.std(v1) > 0
                    and np.std(v2) > 0
                ):
                    corr = float(
                        np.corrcoef(v1, v2)[0, 1]
                    )
                else:
                    corr = np.nan

                l1 = float(
                    np.abs(v1 - v2).sum()
                )

                reproducibility_rows.append(
                    {
                        "branch": branch,
                        "n_diff1": len(g1),
                        "n_diff2": len(g2),
                        "cluster_composition_correlation": corr,
                        "cluster_composition_L1_distance": l1,
                    }
                )

        reproducibility = pd.DataFrame(
            reproducibility_rows
        )

        save_table(
            reproducibility,
            "12_branch_reproducibility.csv",
            index=False,
        )

        if not reproducibility.empty:
            write_both(
                report,
                "\nWithin-branch differentiation agreement:\n"
                + reproducibility.round(4).to_string(index=False)
            )

        # =============================================================
        # 10. Pseudotime monotonicity within branches
        # =============================================================

        write_both(
            report,
            "\n10. PSEUDOTIME MONOTONICITY WITHIN BRANCHES"
        )
        write_both(report, "-" * 82)

        monotonic_rows = []

        for branch, group in pseudo.groupby(
            "Pseudotime_branch"
        ):

            rho = safe_spearman(
                group["CellDay"],
                group["Pseudotime_value"],
            )

            medians = (
                group.groupby("CellDay")["Pseudotime_value"]
                .median()
                .sort_index()
            )

            diffs_median = np.diff(
                medians.values
            )

            fraction_non_decreasing = (
                np.mean(diffs_median >= 0)
                if len(diffs_median) > 0
                else np.nan
            )

            monotonic_rows.append(
                {
                    "branch": branch,
                    "n_cells": len(group),
                    "spearman_day_vs_pseudotime": rho,
                    "fraction_day_to_day_median_non_decreasing":
                        fraction_non_decreasing,
                }
            )

        monotonic = pd.DataFrame(
            monotonic_rows
        )

        save_table(
            monotonic,
            "13_branch_pseudotime_monotonicity.csv",
            index=False,
        )

        write_both(
            report,
            monotonic.round(6).to_string(index=False)
        )

        # =============================================================
        # 11. Transition landmarks
        # =============================================================

        write_both(
            report,
            "\n11. BIOLOGICAL TRANSITION LANDMARKS ALONG PSEUDOTIME"
        )
        write_both(report, "-" * 82)

        landmark_rows = []

        for state in EXPECTED_TRAJECTORY_STATES:

            group = pseudo[
                pseudo["Assigned_cluster"] == state
            ]

            if group.empty:
                continue

            values = group[
                "Pseudotime_value"
            ].dropna()

            days_state = group["CellDay"]

            landmark_rows.append(
                {
                    "state": state,
                    "n_cells": len(group),
                    "pseudotime_q10": float(values.quantile(0.10)),
                    "pseudotime_q25": float(values.quantile(0.25)),
                    "pseudotime_median": float(values.median()),
                    "pseudotime_q75": float(values.quantile(0.75)),
                    "pseudotime_q90": float(values.quantile(0.90)),
                    "median_experimental_day": float(days_state.median()),
                    "first_day": int(days_state.min()),
                    "last_day": int(days_state.max()),
                }
            )

        landmarks = pd.DataFrame(
            landmark_rows
        )

        if not landmarks.empty:

            landmarks["expected_order"] = (
                landmarks["state"]
                .map({
                    s: i
                    for i, s in enumerate(
                        EXPECTED_TRAJECTORY_STATES
                    )
                })
            )

            landmarks = (
                landmarks.sort_values(
                    "expected_order"
                )
                .drop(columns="expected_order")
                .reset_index(drop=True)
            )

        save_table(
            landmarks,
            "14_transition_landmarks.csv",
            index=False,
        )

        write_both(
            report,
            landmarks.round(6).to_string(index=False)
        )

        # =============================================================
        # 12. Final suitability checks
        # =============================================================

        write_both(
            report,
            "\n12. FINAL TRAJECTORY SUITABILITY ASSESSMENT"
        )
        write_both(report, "-" * 82)

        expected_present = set(
            EXPECTED_TRAJECTORY_STATES
        ).issubset(
            set(pseudo["Assigned_cluster"])
        )

        fate_separation = False

        if len(fate_table) >= 2:
            beta_dominant_branch = (
                fate_table["sc_beta_fraction"].idxmax()
            )
            ec_dominant_branch = (
                fate_table["sc_ec_fraction"].idxmax()
            )

            fate_separation = (
                beta_dominant_branch
                != ec_dominant_branch
            )

        replicate_pass = True

        if not reproducibility.empty:
            replicate_pass = bool(
                (
                    reproducibility[
                        "cluster_composition_correlation"
                    ] >= 0.90
                ).all()
            )

        branch_monotonic_pass = bool(
            (
                monotonic[
                    "spearman_day_vs_pseudotime"
                ] > 0
            ).all()
        )

        checks = [
            (
                "All 18,099 pseudotime cells map to main metadata",
                missing_week == 0,
            ),
            (
                "CellDay agrees exactly with CellWeek",
                match_fraction == 1.0,
            ),
            (
                "Cluster annotations agree with main metadata",
                cluster_match.mean() == 1.0,
            ),
            (
                "Differentiation annotations agree with main metadata",
                diff_match.mean() == 1.0,
            ),
            (
                "Expected endocrine trajectory states are present",
                expected_present,
            ),
            (
                "Pseudotime is positively associated with experimental day",
                rho_day > 0,
            ),
            (
                "Pseudotime is positively associated with day in both branches",
                branch_monotonic_pass,
            ),
            (
                "Beta- and EC-associated endpoints separate by branch",
                fate_separation,
            ),
            (
                "Branch composition is reproducible across differentiations",
                replicate_pass,
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
            f"\nAutomated checks passed: "
            f"{passed}/{len(checks)}"
        )

        if passed == len(checks):
            write_both(
                report,
                "\nFINAL STATUS: PASS"
            )
            write_both(
                report,
                "The deposited 18,099-cell endocrine trajectory "
                "is suitable for use as the primary GDIS-Bio trajectory."
            )
        else:
            write_both(
                report,
                "\nFINAL STATUS: REVIEW REQUIRED"
            )
            write_both(
                report,
                "One or more trajectory properties require interpretation "
                "before using this subset as the primary GDIS trajectory."
            )

        write_both(
            report,
            "\nNo normalization, filtering, dimensionality reduction, "
            "trajectory inference, or GDIS calculation was performed."
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

    print("\n" + "=" * 82)
    print("p4_endocrine_trajectory_validation.py completed.")
    print("=" * 82)
    print(f"Report : {REPORT_FILE}")
    print(f"Tables : {TABLE_DIR}")
    print(f"Figures: {FIGURE_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

