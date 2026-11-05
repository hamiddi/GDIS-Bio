#!/usr/bin/env python3
# =============================================================================
# Paper: GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing
#        Transcriptional State Transitions in Single-Cell Trajectories
# Authors: Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash
# =============================================================================

"""Create Supplementary Figure S8 and supplementary tables from completed P25 outputs."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

BASE = (
    ROOT / "results"
    / "p25_GSE175634_pseudotime_ordering_robustness"
)

TABLES = BASE / "tables"
ALT = BASE / "alternative_reconstruction_cohort"

FIGURE_DIR = BASE / "figures"
SUPP_DIR = BASE / "supplementary_outputs"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
SUPP_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Load source files
# ---------------------------------------------------------------------
bootstrap = pd.read_csv(TABLES / "05_individual_bootstrap_ci.csv")
individual_local = pd.read_csv(TABLES / "04_individual_level_ordering_summary.csv")
geodesic_qual = pd.read_csv(ALT / "01_geodesic_qualification_full.csv", dtype={"individual": str})
geodesic_violation = pd.read_csv(ALT / "03_state_order_violation_summary.csv")
paired_scope = pd.read_csv(TABLES / "09_geodesic_paired_scope_statistics.csv")
paired_individual = pd.read_csv(TABLES / "08_geodesic_paired_individual_results.csv", dtype={"individual": str})

# Normalize booleans if needed
for col in ["qualified", "reference_within_iqr", "geodesic_within_iqr"]:
    if col in paired_individual.columns:
        paired_individual[col] = (
            paired_individual[col].astype(str).str.lower().map({"true": True, "false": False})
        )

if "qualified" in geodesic_qual.columns:
    geodesic_qual["qualified"] = (
        geodesic_qual["qualified"].astype(str).str.lower().map({"true": True, "false": False})
    )

scope_order = ["shared_backbone", "cm_extension", "cf_extension"]
scope_labels = {
    "shared_backbone": "Shared backbone",
    "cm_extension": "CM extension",
    "cf_extension": "CF extension",
}

metric_labels = {
    "median_profile_r": "Median profile correlation",
    "median_abs_peak_shift": "Median absolute peak shift",
    "median_window_overlap": "Median window overlap",
    "peak_change_fraction": "Peak-change fraction",
    "median_delta_distance": "Median paired delta distance",
    "median_abs_delta_distance": "Median absolute delta distance",
}

# ---------------------------------------------------------------------
# Supplementary tables
# ---------------------------------------------------------------------

# Auxiliary table A: local ordering perturbation robustness summary
selected_metrics = [
    "median_profile_r",
    "median_abs_peak_shift",
    "median_window_overlap",
    "peak_change_fraction",
]

local_rows = []
for metric in selected_metrics:
    subset = bootstrap[bootstrap["metric"] == metric].copy()
    subset = subset.rename(columns={
        "individual_level_median": f"{metric}_median",
        "bootstrap_ci_lower": f"{metric}_ci_lower",
        "bootstrap_ci_upper": f"{metric}_ci_upper",
    })
    keep = ["scope", "sigma_rank",
            f"{metric}_median",
            f"{metric}_ci_lower",
            f"{metric}_ci_upper"]
    local_rows.append(subset[keep])

supp_s7 = local_rows[0]
for df in local_rows[1:]:
    supp_s7 = supp_s7.merge(df, on=["scope", "sigma_rank"], how="outer")

supp_s7["scope_label"] = supp_s7["scope"].map(scope_labels)
supp_s7 = supp_s7.sort_values(
    by=["scope", "sigma_rank"],
    key=lambda s: s.map({k: i for i, k in enumerate(scope_order)}) if s.name == "scope" else s
)

supp_s7.to_csv(
    SUPP_DIR / "P25_auxiliary_local_ordering_robustness.csv",
    index=False
)

# Auxiliary table B: geodesic reconstruction qualification profile-level results
supp_s8 = geodesic_qual.copy()
supp_s8["scope_label"] = supp_s8["scope"].map(scope_labels)
supp_s8 = supp_s8.sort_values(
    by=["scope", "individual"],
    key=lambda s: s.map({k: i for i, k in enumerate(scope_order)}) if s.name == "scope" else s
)

supp_s8.to_csv(
    SUPP_DIR / "P25_auxiliary_geodesic_reconstruction_qualification.csv",
    index=False
)

# Auxiliary table C: state-order violation summary
supp_s9 = geodesic_violation.copy()
supp_s9["scope_label"] = supp_s9["scope"].map(scope_labels)
supp_s9 = supp_s9.sort_values(
    by=["scope", "transition"],
    key=lambda s: s.map({k: i for i, k in enumerate(scope_order)}) if s.name == "scope" else s
)

supp_s9.to_csv(
    SUPP_DIR / "P25_auxiliary_state_order_violation_summary.csv",
    index=False
)

# Auxiliary table D: paired geodesic vs deposited statistics
supp_s10 = paired_scope.copy()
supp_s10["scope_label"] = supp_s10["scope"].map(scope_labels)
supp_s10 = supp_s10.sort_values(
    by=["scope", "qualification"],
    key=lambda s: s.map({k: i for i, k in enumerate(scope_order)}) if s.name == "scope" else s
)

supp_s10.to_csv(
    SUPP_DIR / "P25_auxiliary_geodesic_paired_statistics.csv",
    index=False
)

# Excel workbook with all supplementary tables
with pd.ExcelWriter(SUPP_DIR / "P25_auxiliary_tables.xlsx") as writer:
    supp_s7.to_excel(writer, sheet_name="Local_ordering", index=False)
    supp_s8.to_excel(writer, sheet_name="Reconstruction", index=False)
    supp_s9.to_excel(writer, sheet_name="State_violations", index=False)
    supp_s10.to_excel(writer, sheet_name="Paired_statistics", index=False)

# ---------------------------------------------------------------------
# Supplementary Figure S8
# ---------------------------------------------------------------------
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)

# Panel A: local ordering profile correlation
ax1 = fig.add_subplot(gs[0, 0])
metric = "median_profile_r"
for scope in scope_order:
    sub = bootstrap[
        (bootstrap["scope"] == scope) &
        (bootstrap["metric"] == metric)
    ].sort_values("sigma_rank")
    x = sub["sigma_rank"].to_numpy(dtype=float)
    y = sub["individual_level_median"].to_numpy(dtype=float)
    yerr_low = y - sub["bootstrap_ci_lower"].to_numpy(dtype=float)
    yerr_high = sub["bootstrap_ci_upper"].to_numpy(dtype=float) - y
    ax1.errorbar(
        x, y,
        yerr=np.vstack([yerr_low, yerr_high]),
        marker="o",
        linewidth=1.5,
        capsize=3,
        label=scope_labels[scope]
    )
ax1.set_title("A. Local-ordering profile stability")
ax1.set_xlabel("Gaussian rank-noise SD")
ax1.set_ylabel("Median profile correlation")
ax1.set_xticks([10, 50, 100])
ax1.set_ylim(0.84, 1.00)
ax1.grid(alpha=0.25)
ax1.legend(fontsize=8, loc="lower left")

# Panel B: local ordering peak displacement
ax2 = fig.add_subplot(gs[0, 1])
metric = "median_abs_peak_shift"
for scope in scope_order:
    sub = bootstrap[
        (bootstrap["scope"] == scope) &
        (bootstrap["metric"] == metric)
    ].sort_values("sigma_rank")
    x = sub["sigma_rank"].to_numpy(dtype=float)
    y = sub["individual_level_median"].to_numpy(dtype=float)
    yerr_low = y - sub["bootstrap_ci_lower"].to_numpy(dtype=float)
    yerr_high = sub["bootstrap_ci_upper"].to_numpy(dtype=float) - y
    ax2.errorbar(
        x, y,
        yerr=np.vstack([yerr_low, yerr_high]),
        marker="o",
        linewidth=1.5,
        capsize=3,
        label=scope_labels[scope]
    )
ax2.set_title("B. Local-ordering peak displacement")
ax2.set_xlabel("Gaussian rank-noise SD")
ax2.set_ylabel("Median absolute peak shift")
ax2.set_xticks([10, 50, 100])
ax2.grid(alpha=0.25)

# Panel C: geodesic qualification rate
ax3 = fig.add_subplot(gs[1, 0])
qual_summary = (
    geodesic_qual.groupby("scope", as_index=False)
    .agg(
        n_profiles=("individual", "size"),
        n_qualified=("qualified", "sum")
    )
)
qual_summary["qualification_rate"] = (
    qual_summary["n_qualified"] / qual_summary["n_profiles"] * 100.0
)
qual_summary["scope_label"] = qual_summary["scope"].map(scope_labels)
qual_summary = qual_summary.set_index("scope").loc[scope_order].reset_index()

bars = ax3.bar(
    np.arange(len(qual_summary)),
    qual_summary["qualification_rate"].to_numpy(dtype=float)
)
ax3.set_title("C. Graph-geodesic qualification rate")
ax3.set_ylabel("Qualified trajectories (%)")
ax3.set_xticks(np.arange(len(qual_summary)))
ax3.set_xticklabels(qual_summary["scope_label"], rotation=15, ha="right")
ax3.set_ylim(0, 100)
ax3.grid(axis="y", alpha=0.25)
for rect, value in zip(bars, qual_summary["qualification_rate"]):
    ax3.text(
        rect.get_x() + rect.get_width() / 2,
        rect.get_height() + 2,
        f"{value:.1f}%",
        ha="center",
        va="bottom",
        fontsize=9
    )

# Panel D: paired geodesic minus deposited distance by scope
ax4 = fig.add_subplot(gs[1, 1])
all_summary = paired_scope[paired_scope["qualification"] == "all"].copy()
all_summary["scope_label"] = all_summary["scope"].map(scope_labels)
all_summary = all_summary.set_index("scope").loc[scope_order].reset_index()

positions = np.arange(1, len(scope_order) + 1)

for i, scope in enumerate(scope_order, start=1):
    vals = paired_individual.loc[
        paired_individual["scope"] == scope,
        "delta_distance"
    ].to_numpy(dtype=float)

    jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else np.array([0.0])
    ax4.scatter(
        np.full(len(vals), i) + jitter,
        vals,
        alpha=0.7,
        s=25
    )

    row = all_summary[all_summary["scope"] == scope].iloc[0]
    median = float(row["median_paired_delta"])
    low = float(row["delta_ci_low"])
    high = float(row["delta_ci_high"])

    ax4.errorbar(
        i, median,
        yerr=np.array([[median - low], [high - median]]),
        fmt="o",
        capsize=4,
        markersize=7,
        linewidth=1.5
    )

ax4.axhline(0.0, linestyle="--", linewidth=1.0)
ax4.set_title("D. Paired localization change under geodesic ordering")
ax4.set_ylabel("Geodesic − deposited distance")
ax4.set_xticks(positions)
ax4.set_xticklabels([scope_labels[s] for s in scope_order], rotation=15, ha="right")
ax4.grid(alpha=0.25)

fig.suptitle(
    "Supplementary Figure S8. Sensitivity of GDIS to pseudotemporal ordering\n"
    "External GSE175634 cardiac validation dataset",
    fontsize=14
)

fig.subplots_adjust(
    left=0.09,
    right=0.97,
    bottom=0.10,
    top=0.88,
    hspace=0.40,
    wspace=0.30,
)

png_path = FIGURE_DIR / "Supplementary_Figure_S8.png"
pdf_path = FIGURE_DIR / "Supplementary_Figure_S8.pdf"
tiff_path = FIGURE_DIR / "Supplementary_Figure_S8.tiff"

fig.savefig(png_path, dpi=600, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")

# High-resolution TIFF omitted from default pipeline to avoid very large outputs.
# Publication TIFF can be exported from the PDF manually when required.
plt.close(fig)

# ---------------------------------------------------------------------
# Caption file
# ---------------------------------------------------------------------
caption = """Supplementary Figure S8. Sensitivity of GDIS to pseudotemporal ordering in the external GSE175634 cardiac dataset.
(A) Individual-level median GDIS profile correlation under Gaussian rank perturbations (rank-noise standard deviations 10, 50, and 100) with bootstrap 95% confidence intervals.
(B) Individual-level median absolute GDIS peak displacement under the same perturbation analysis.
(C) Fraction of graph-geodesic reconstructions satisfying the prespecified strict biological state-ordering criterion in each trajectory scope.
(D) Paired changes in GDIS peak-to-landmark distance under graph-geodesic versus deposited ordering across individual trajectories; points denote individual trajectories, and error bars denote bootstrap 95% confidence intervals for the scope-specific medians. Positive values indicate increased peak-to-landmark distance under graph-geodesic ordering.
"""

with open(SUPP_DIR / "Supplementary_Figure_S8_caption.txt", "w") as f:
    f.write(caption)

print("Supplementary Figure S8 and supplementary tables created.")
print("Figure PNG :", png_path)
print("Figure PDF :", pdf_path)
print("Figure TIFF:", tiff_path)
print("Supplementary directory:", SUPP_DIR)
