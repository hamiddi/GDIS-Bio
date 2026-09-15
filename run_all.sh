#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[GDIS-Bio] repository: $ROOT"
"$PYTHON_BIN" tools/check_environment.py

echo "[GDIS-Bio] running p1_download_data.py"
"$PYTHON_BIN" "scripts/p1_download_data.py"

echo "[GDIS-Bio] running p2_preflight_data.py"
"$PYTHON_BIN" "scripts/p2_preflight_data.py"

echo "[GDIS-Bio] running p3_dataset_characterization.py"
"$PYTHON_BIN" "scripts/p3_dataset_characterization.py"

echo "[GDIS-Bio] running p4_endocrine_trajectory_validation.py"
"$PYTHON_BIN" "scripts/p4_endocrine_trajectory_validation.py"

echo "[GDIS-Bio] running p4b_replicate_trajectory_validation.py"
"$PYTHON_BIN" "scripts/p4b_replicate_trajectory_validation.py"

echo "[GDIS-Bio] running p5_preprocessing_state_space.py"
"$PYTHON_BIN" "scripts/p5_preprocessing_state_space.py"

echo "[GDIS-Bio] running p6_state_space_sensitivity.py"
"$PYTHON_BIN" "scripts/p6_state_space_sensitivity.py"

echo "[GDIS-Bio] running p6b_reference_geometry_validation.py"
"$PYTHON_BIN" "scripts/p6b_reference_geometry_validation.py"

echo "[GDIS-Bio] running p7_trajectory_assembly_validation.py"
"$PYTHON_BIN" "scripts/p7_trajectory_assembly_validation.py"

echo "[GDIS-Bio] running p8_gdis_primary_analysis.py"
"$PYTHON_BIN" "scripts/p8_gdis_primary_analysis.py"

echo "[GDIS-Bio] running p9_gdis_result_validation.py"
"$PYTHON_BIN" "scripts/p9_gdis_result_validation.py"

echo "[GDIS-Bio] running p10_conventional_ews_benchmark.py"
"$PYTHON_BIN" "scripts/p10_conventional_ews_benchmark.py"

echo "[GDIS-Bio] running p11_statistical_alignment_null_validation.py"
"$PYTHON_BIN" "scripts/p11_statistical_alignment_null_validation.py"

echo "[GDIS-Bio] running p12_download_preflight_GSE175634.py"
"$PYTHON_BIN" "scripts/p12_download_preflight_GSE175634.py"

echo "[GDIS-Bio] running p13_GSE175634_trajectory_characterization.py"
"$PYTHON_BIN" "scripts/p13_GSE175634_trajectory_characterization.py"

echo "[GDIS-Bio] running p14_GSE175634_external_validation_design_freeze.py"
"$PYTHON_BIN" "scripts/p14_GSE175634_external_validation_design_freeze.py"

echo "[GDIS-Bio] running p15_GSE175634_preprocessing_state_space.py"
"$PYTHON_BIN" "scripts/p15_GSE175634_preprocessing_state_space.py"

echo "[GDIS-Bio] running p16_GSE175634_state_space_geometry_validation.py"
"$PYTHON_BIN" "scripts/p16_GSE175634_state_space_geometry_validation.py"

echo "[GDIS-Bio] running p17_GSE175634_trajectory_window_assembly.py"
"$PYTHON_BIN" "scripts/p17_GSE175634_trajectory_window_assembly.py"

echo "[GDIS-Bio] running p17b_GSE175634_transition_evaluability_freeze.py"
"$PYTHON_BIN" "scripts/p17b_GSE175634_transition_evaluability_freeze.py"

echo "[GDIS-Bio] running p18_GSE175634_primary_external_gdis.py"
"$PYTHON_BIN" "scripts/p18_GSE175634_primary_external_gdis.py"

echo "[GDIS-Bio] running p19_GSE175634_external_gdis_statistical_validation.py"
"$PYTHON_BIN" "scripts/p19_GSE175634_external_gdis_statistical_validation.py"

echo "[GDIS-Bio] running p20_GSE175634_external_gdis_sensitivity.py"
"$PYTHON_BIN" "scripts/p20_GSE175634_external_gdis_sensitivity.py"

echo "[GDIS-Bio] running p21_GSE175634_conventional_ews_benchmark.py"
"$PYTHON_BIN" "scripts/p21_GSE175634_conventional_ews_benchmark.py"

echo "[GDIS-Bio] running p22_GSE175634_benchmark_statistical_comparison.py"
"$PYTHON_BIN" "scripts/p22_GSE175634_benchmark_statistical_comparison.py"

echo "[GDIS-Bio] running p23_GSE175634_external_validation_evidence_freeze.py"
"$PYTHON_BIN" "scripts/p23_GSE175634_external_validation_evidence_freeze.py"

echo "[GDIS-Bio] full pipeline completed successfully."
