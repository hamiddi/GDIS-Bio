# Statistical validation and benchmarking

The pipeline deliberately separates profile construction from inferential units. Overlapping sliding windows are dependent and are **not** treated as independent replicates. Inference is performed at the trajectory or biological-individual level.

## Discovery (GSE114412)

- `p10_conventional_ews_benchmark.py` evaluates GDIS against total variance, Gaussian differential entropy, mean pseudotemporal step distance, lag-1 pseudotemporal autocorrelation, and the internal GDIS transition-energy component on matched windows.
- `p11_statistical_alignment_null_validation.py` performs 20,000 structure-preserving circular shifts (seed `20260913`) to test alignment with the independently frozen NEUROG3-early landmark and applies Benjamini-Hochberg FDR correction.
- Small-n paired comparisons across the four branch × differentiation trajectories are interpreted conservatively.

## External validation (GSE175634)

- `p19_GSE175634_external_gdis_statistical_validation.py` evaluates frozen GDIS localization and source-to-destination changes across independent individuals.
- `p21_GSE175634_conventional_ews_benchmark.py` computes the conventional benchmark metrics using the frozen external windows.
- `p22_GSE175634_benchmark_statistical_comparison.py` performs individual-level GDIS-versus-comparator comparisons, including sign tests, paired Wilcoxon tests, circular-shift localization tests, and FDR adjustment.
- `p23_GSE175634_external_validation_evidence_freeze.py` consolidates the final evidence without recomputing GDIS or adding new inference.

## Interpretation

Peak-to-landmark distance measures localization relative to independently defined biological reference regions; it is not an error against a known physical transition time. Pseudotime is an ordered developmental coordinate, not physical time.
