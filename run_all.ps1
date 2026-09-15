$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Python = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }
Write-Host "[GDIS-Bio] repository: $Root"
& $Python "tools/check_environment.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[GDIS-Bio] running p1_download_data.py"
& $Python "scripts/p1_download_data.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[GDIS-Bio] running p2_preflight_data.py"
& $Python "scripts/p2_preflight_data.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[GDIS-Bio] running p3_dataset_characterization.py"
& $Python "scripts/p3_dataset_characterization.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[GDIS-Bio] running p4_endocrine_trajectory_validation.py"
& $Python "scripts/p4_endocrine_trajectory_validation.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[GDIS-Bio] running p4b_replicate_trajectory_validation.py"
& $Python "scripts/p4b_replicate_trajectory_validation.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[GDIS-Bio] running p5_preprocessing_state_space.py"
& $Python "scripts/p5_preprocessing_state_space.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[GDIS-Bio] running p6_state_space_sensitivity.py"
& $Python "scripts/p6_state_space_sensitivity.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[GDIS-Bio] running p6b_reference_geometry_validation.py"
& $Python "scripts/p6b_reference_geometry_validation.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[GDIS-Bio] running p7_trajectory_assembly_validation.py"
& $Python "scripts/p7_trajectory_assembly_validation.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[GDIS-Bio] running p8_gdis_primary_analysis.py"
& $Python "scripts/p8_gdis_primary_analysis.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[GDIS-Bio] running p9_gdis_result_validation.py"
& $Python "scripts/p9_gdis_result_validation.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[GDIS-Bio] running p10_conventional_ews_benchmark.py"
& $Python "scripts/p10_conventional_ews_benchmark.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[GDIS-Bio] running p11_statistical_alignment_null_validation.py"
& $Python "scripts/p11_statistical_alignment_null_validation.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[GDIS-Bio] running p12_download_preflight_GSE175634.py"
& $Python "scripts/p12_download_preflight_GSE175634.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[GDIS-Bio] running p13_GSE175634_trajectory_characterization.py"
& $Python "scripts/p13_GSE175634_trajectory_characterization.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[GDIS-Bio] running p14_GSE175634_external_validation_design_freeze.py"
& $Python "scripts/p14_GSE175634_external_validation_design_freeze.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[GDIS-Bio] running p15_GSE175634_preprocessing_state_space.py"
& $Python "scripts/p15_GSE175634_preprocessing_state_space.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[GDIS-Bio] running p16_GSE175634_state_space_geometry_validation.py"
& $Python "scripts/p16_GSE175634_state_space_geometry_validation.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[GDIS-Bio] running p17_GSE175634_trajectory_window_assembly.py"
& $Python "scripts/p17_GSE175634_trajectory_window_assembly.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[GDIS-Bio] running p17b_GSE175634_transition_evaluability_freeze.py"
& $Python "scripts/p17b_GSE175634_transition_evaluability_freeze.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[GDIS-Bio] running p18_GSE175634_primary_external_gdis.py"
& $Python "scripts/p18_GSE175634_primary_external_gdis.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[GDIS-Bio] running p19_GSE175634_external_gdis_statistical_validation.py"
& $Python "scripts/p19_GSE175634_external_gdis_statistical_validation.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[GDIS-Bio] running p20_GSE175634_external_gdis_sensitivity.py"
& $Python "scripts/p20_GSE175634_external_gdis_sensitivity.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[GDIS-Bio] running p21_GSE175634_conventional_ews_benchmark.py"
& $Python "scripts/p21_GSE175634_conventional_ews_benchmark.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[GDIS-Bio] running p22_GSE175634_benchmark_statistical_comparison.py"
& $Python "scripts/p22_GSE175634_benchmark_statistical_comparison.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[GDIS-Bio] running p23_GSE175634_external_validation_evidence_freeze.py"
& $Python "scripts/p23_GSE175634_external_validation_evidence_freeze.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[GDIS-Bio] full pipeline completed successfully."
