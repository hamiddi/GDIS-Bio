# Results

The analysis scripts write generated outputs into this directory. Generated outputs are ignored by Git by default.

`reference_outputs/` contains the compact tables and reports from the frozen manuscript analysis. These files are included as verification targets and are **not** inputs to the end-to-end pipeline. A fresh run should regenerate corresponding `results/p*` directories alongside `reference_outputs/`.

Large intermediate matrices (`.h5ad`, large PCA arrays, window arrays, etc.) are intentionally excluded from GitHub. The complete original results tree was approximately 1.49 GB; see [`../docs/original_results_manifest.csv`](../docs/original_results_manifest.csv).
