# Dataset descriptions

## Discovery dataset: GSE114412

Human stem-cell pancreatic differentiation. The reproducibility workflow uses the Stage-5 processed counts, all-cell metadata, and deposited endocrine pseudotime metadata. The discovery analysis evaluates the endocrine trajectories leading toward SC-EC and SC-β states across two independent differentiations.

Expected files are downloaded by `scripts/p1_download_data.py`.

## External validation dataset: GSE175634

Human iPSC-to-cardiac differentiation with multiple individuals and experimental days. The frozen external-validation design evaluates the shared MES→CMES transition (T2) and terminal PROG→CM (T3-CM) and PROG→CF (T3-CF) transitions. Early shared-backbone cells are not retrospectively assigned to terminal fates.

Expected files are downloaded and checked by `scripts/p12_download_preflight_GSE175634.py`.

## Data-handling principle

Raw public data are not distributed in this repository. Download, integrity/preflight, preprocessing, and all downstream derived outputs are produced by the scripts in the order documented in `reproducibility_workflow.md`.
