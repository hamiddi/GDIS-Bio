#!/usr/bin/env python3
"""Ordered GDIS-Bio pipeline driver.

This driver is intentionally transparent: it executes the archived scientific
scripts as separate Python processes and does not modify their analytical logic.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
STATE_DIR = ROOT / "results" / "pipeline_state"

CORE = [
    "p1_download_data.py",
    "p2_preflight_data.py",
    "p3_dataset_characterization.py",
    "p4_endocrine_trajectory_validation.py",
    "p4b_replicate_trajectory_validation.py",
    "p5_preprocessing_state_space.py",
    "p6_state_space_sensitivity.py",
    "p6b_reference_geometry_validation.py",
    "p7_trajectory_assembly_validation.py",
    "p8_gdis_primary_analysis.py",
    "p9_gdis_result_validation.py",
    "p10_conventional_ews_benchmark.py",
    "p11_statistical_alignment_null_validation.py",
    "p12_download_preflight_GSE175634.py",
    "p13_GSE175634_trajectory_characterization.py",
    "p14_GSE175634_external_validation_design_freeze.py",
    "p15_GSE175634_preprocessing_state_space.py",
    "p16_GSE175634_state_space_geometry_validation.py",
    "p17_GSE175634_trajectory_window_assembly.py",
    "p17b_GSE175634_transition_evaluability_freeze.py",
    "p18_GSE175634_primary_external_gdis.py",
    "p19_GSE175634_external_gdis_statistical_validation.py",
    "p20_GSE175634_external_gdis_sensitivity.py",
    "p21_GSE175634_conventional_ews_benchmark.py",
    "p22_GSE175634_benchmark_statistical_comparison.py",
    "p23_GSE175634_external_validation_evidence_freeze.py",
]

REVIEW = [
    "p24_GSE175634_component_ablation.py",
    "p25_GSE175634_pseudotime_ordering_robustness.py",
    "p25b_GSE175634_cell_window_reconstruction_audit.py",
    "p25c_GSE175634_ordering_perturbation_pilot.py",
    "p25d_GSE175634_ordering_robustness_full.py",
    "p25e_GSE175634_ordering_statistical_analysis.py",
    "p25f_GSE175634_alternative_pseudotime.py",
    "p25g_GSE175634_graph_geodesic_pilot.py",
    "p25h_GSE175634_alternative_ordering_gdis.py",
    "p25i_GSE175634_geodesic_cohort_audit.py",
    "p25j_GSE175634_geodesic_gdis_comparison.py",
    "p25k_GSE175634_geodesic_paired_statistics.py",
    "p25l_make_supplementary_figure_s7_and_tables.py",
    "p25m_make_figure_s8_and_auxiliary_tables.py",
    "p26_comment6_blocked_null_audit.py",
    "p27_reviewer2_comment2_noncircular_landmark_null.py",
    "p28_reviewer2_comment5_pancreatic_iqr.py",
    "p29_reviewer2_comment6_window_span_audit.py",
]


def stage_id(filename: str) -> str:
    return filename.split("_", 1)[0].lower()


def stages_for_phase(phase: str) -> list[str]:
    if phase == "core":
        return CORE.copy()
    if phase == "review":
        return REVIEW.copy()
    return CORE + REVIEW


def validate_files(stages: list[str]) -> list[str]:
    missing = [name for name in stages if not (SCRIPTS / name).is_file()]
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the GDIS-Bio scientific pipeline in stage order.")
    parser.add_argument("--phase", choices=["core", "review", "all"], default="all")
    parser.add_argument("--list", action="store_true", help="List selected stages and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--check", action="store_true", help="Check that selected script files exist and exit.")
    parser.add_argument("--resume", action="store_true", help="Skip stages with successful-stage markers.")
    parser.add_argument("--only", help="Run one stage ID, e.g. p27 or p25i.")
    args = parser.parse_args()

    stages = stages_for_phase(args.phase)

    if args.only:
        wanted = args.only.lower()
        stages = [name for name in stages if stage_id(name) == wanted]
        if not stages:
            parser.error(f"stage {args.only!r} is not in phase {args.phase!r}")

    missing = validate_files(stages)
    if missing:
        print("Missing required scripts:", file=sys.stderr)
        for name in missing:
            print(f"  {name}", file=sys.stderr)
        return 1

    if args.list:
        for i, name in enumerate(stages, 1):
            print(f"{i:02d}. {stage_id(name):6s} {name}")
        return 0

    if args.check:
        print(f"All {len(stages)} selected stage scripts are present.")
        return 0

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    python_bin = os.environ.get("PYTHON_BIN", sys.executable)

    for i, name in enumerate(stages, 1):
        sid = stage_id(name)
        marker = STATE_DIR / f"{sid}.success"
        cmd = [python_bin, str(SCRIPTS / name)]

        if args.resume and marker.exists():
            print(f"[{i}/{len(stages)}] SKIP {sid}: success marker exists")
            continue

        print(f"[{i}/{len(stages)}] {sid}: {' '.join(cmd)}", flush=True)

        if args.dry_run:
            continue

        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print(f"Pipeline stopped at {sid} ({name}) with exit code {result.returncode}.", file=sys.stderr)
            return result.returncode

        marker.write_text(name + "\n", encoding="utf-8")

    if args.dry_run:
        print("Dry run complete.")
    else:
        print("Selected GDIS-Bio pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
