# GDIS-Bio

## A reproducible framework for localizing transcriptional state transitions in single-cell trajectories

**GDIS-Bio** adapts the **Generalized Dynamical Instability Score (GDIS)** to pseudotemporally ordered single-cell RNA-seq trajectories. The workflow integrates local sensitivity, trajectory stretching, state-space expansion, nonlinear complexity, and persistence to characterize transcriptional state-space reorganization and to evaluate whether GDIS peaks align with independently annotated biological transitions.

**Authors:** Hamid Ismail, Ahmed Harb, Basem William, and Marwan Bikdash  
**Repository release:** v1.1.0  
**Reference GDIS software:** [pyGDIS v1.0.0](https://pypi.org/project/pygdis/)  
**License:** [MIT](LICENSE)

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](environment.yml)
[![pyGDIS 1.0.0](https://img.shields.io/badge/pyGDIS-1.0.0-6f42c1.svg)](https://pypi.org/project/pygdis/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Workflow](https://img.shields.io/badge/workflow-P1%E2%80%93P29-informational.svg)](#analysis-pipeline)

---

## Contents

- [Overview](#overview)
- [Scientific scope and interpretation](#scientific-scope-and-interpretation)
- [Study design and datasets](#study-design-and-datasets)
- [Analysis pipeline](#analysis-pipeline)
- [P25 ordering-robustness pipeline](#p25-ordering-robustness-pipeline)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Reproducibility and validation](#reproducibility-and-validation)
- [Key results represented in this repository](#key-results-represented-in-this-repository)
- [Repository structure](#repository-structure)
- [Data availability](#data-availability)
- [Citation](#citation)
- [References](#references)
- [License and support](#license-and-support)

---

## Overview

GDIS-Bio is a research-analysis repository for studying **where major transcriptional reorganizations occur along single-cell developmental trajectories**.

The analysis has two principal biological settings:

1. **Pancreatic differentiation — discovery analysis**  
   Human stem-cell differentiation toward pancreatic endocrine states using GEO accession **GSE114412**.

2. **Cardiac differentiation — external validation**  
   Human iPSC-to-cardiomyocyte differentiation using GEO accession **GSE175634**, evaluated across independent individuals and prespecified developmental scopes.

The repository contains the scientific scripts used for:

- public-data acquisition and preflight checks;
- state-space construction and validation;
- pseudotemporal trajectory assembly;
- primary GDIS analysis;
- conventional comparator analysis;
- landmark-localization statistics;
- external validation;
- sensitivity analyses;
- component ablation;
- ordering-robustness analyses;
- reviewer-requested audits;
- publication-oriented supplementary outputs.

### What GDIS-Bio is designed to evaluate

GDIS-Bio evaluates whether local changes in the geometry and organization of transcriptomic state space produce peaks that align with **independently annotated biological landmarks**.

It does **not** assume that a GDIS peak is automatically a mechanistic transition point.

---

## Scientific scope and interpretation

Several distinctions are essential for interpreting the analyses correctly.

> **Pseudotime is not physical time.**  
> Pseudotime orders different cells along an inferred developmental trajectory. It is not longitudinal tracking of the same cell.

> **Peak localization is not proof of a bifurcation.**  
> Proximity between a GDIS maximum and an annotated state supports positional alignment under the tested analysis assumptions. It does not by itself establish a dynamical bifurcation, causal instability, or exact biological transition onset.

> **Overlapping windows are not independent replicates.**  
> Statistical inference is performed at the appropriate biological unit—trajectory, differentiation experiment, or individual—not at the level of overlapping sliding windows.

> **GDIS is complementary to simpler statistics.**  
> Conventional measures can equal or outperform GDIS for some transitions. GDIS should therefore be interpreted as an integrative representation of local dynamical structure rather than a universally superior localization statistic.

These principles are reflected throughout the analysis and reviewer-response stages.

---

## Study design and datasets

| Property | Design |
|---|---|
| Discovery dataset | [GSE114412](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE114412) |
| Discovery system | Human stem-cell pancreatic differentiation |
| Discovery biological replication | Two independent differentiations |
| Discovery trajectories | SC-EC and SC-β branches |
| External dataset | [GSE175634](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE175634) |
| External system | Human iPSC-to-cardiomyocyte differentiation |
| External cohort | 19 shared-backbone, 15 CM-extension, 14 CF-extension individual×scope profiles |
| Primary state space | 50 principal components |
| Primary sliding window | 400 cells |
| Primary step size | 100 cells |
| Prespecified PC sensitivities | 5, 10, 20, and 30 PCs |
| Prespecified window sensitivities | 300/75 and 500/125 cell/step configurations |
| GDIS implementation | pyGDIS v1.0.0 |
| Main conventional comparators | Total variance, Gaussian differential entropy, mean pseudotemporal step distance, lag-1 pseudotemporal autocorrelation |
| Internal GDIS comparator | Transition energy |

### Discovery dataset

The pancreatic discovery analysis is based on the dataset reported by **Veres et al. (2019)**, who profiled human in-vitro β-cell differentiation at single-cell resolution. The corresponding GEO accession is **GSE114412**.

### External validation dataset

The cardiac external validation analysis uses **GSE175634**, reported by **Elorbany et al. (2022)**. The study collected single-cell RNA-seq data across seven differentiation time points from 19 human cell lines and reconstructed bifurcating cardiac developmental trajectories.

---

## Analysis pipeline

The scientific scripts are organized as ordered stages.

| Stage | Purpose |
|---|---|
| P1–P4b | Pancreatic data acquisition, preflight checks, characterization, and trajectory validation |
| P5–P7 | Preprocessing, PCA/state-space validation, and trajectory-window assembly |
| P8–P11 | Primary pancreatic GDIS, result validation, conventional benchmarking, and null analysis |
| P12–P17b | Cardiac acquisition, design freeze, preprocessing, state-space geometry, and trajectory assembly |
| P18–P23 | External GDIS analysis, individual-level statistics, sensitivity, benchmarking, and evidence freeze |
| P24 | Controlled component ablation |
| P25 + lettered stages | Pseudotime-order robustness and alternative graph-geodesic reconstruction |
| P26 | Dependence-aware coordinated shift sensitivity |
| P27 | Endpoint-aware non-circular landmark null |
| P28 | Annotated-state IQR localization analysis |
| P29 | Fixed-cell-window pseudotime-span audit |

Detailed background is available in:

- [`docs/reproducibility_workflow.md`](docs/reproducibility_workflow.md)
- [`docs/reviewer_analyses.md`](docs/reviewer_analyses.md)
- [`docs/statistical_tests.md`](docs/statistical_tests.md)
- [`docs/manuscript_mapping.md`](docs/manuscript_mapping.md)
- [`docs/data_provenance.md`](docs/data_provenance.md)

### Core scripts

Representative stages include:

```bash
python scripts/p1_download_data.py
python scripts/p5_preprocessing_state_space.py
python scripts/p8_gdis_primary_analysis.py
python scripts/p12_download_preflight_GSE175634.py
python scripts/p18_GSE175634_primary_external_gdis.py
python scripts/p24_GSE175634_component_ablation.py
```

Individual stages depend on outputs from earlier stages. They should not be assumed to be independent standalone commands.

---

## P25 ordering-robustness pipeline

P25 evaluates whether the conclusions are robust to changes in cell ordering and alternative reconstruction of developmental progression.

The retained P25 sequence is:

```text
p25_GSE175634_pseudotime_ordering_robustness.py
p25b_GSE175634_cell_window_reconstruction_audit.py
p25c_GSE175634_ordering_perturbation_pilot.py
p25d_GSE175634_ordering_robustness_full.py
p25e_GSE175634_ordering_statistical_analysis.py
p25f_GSE175634_alternative_pseudotime.py
p25g_GSE175634_graph_geodesic_pilot.py
p25h_GSE175634_alternative_ordering_gdis.py
p25i_GSE175634_geodesic_cohort_audit.py
p25j_GSE175634_geodesic_gdis_comparison.py
p25k_GSE175634_geodesic_paired_statistics.py
p25l_make_supplementary_figure_s7_and_tables.py
p25m_make_figure_s8_and_auxiliary_tables.py
```

### P25 pipeline validation

A repository-level validator is included:

```bash
bash scripts/p25_pipeline_validate.sh --smoke
```

The smoke mode always checks:

- required P25 files;
- Python dependencies;
- syntax compilation;
- obvious temporary/backup artifacts.

If the generated P15 state-space file and P17b frozen manifest are present, it also runs one real-data P25-I reconstruction and checks that no computational failure occurs. A clean source-only GitHub clone is not expected to contain those generated upstream files, so their absence is reported as informational rather than as a validation failure.

For integrity checking of an already materialized full P25 output tree:

```bash
bash scripts/p25_pipeline_validate.sh --verify
```

When the full P25-I output is present, `--verify` checks the 48-profile cohort numerically. In a source-only clone without generated results, it completes the repository/code checks and reports that numerical verification was skipped.

The validated P25-I cohort contains:

- **48** individual×scope profiles;
- **19** shared-backbone profiles;
- **15** CM-extension profiles;
- **14** CF-extension profiles.

The current validated graph-geodesic audit produced no `computation_error` records.

### Interpretation of the alternative geodesic ordering

The strict state-order qualification criterion is intentionally conservative. A profile can retain strong global rank agreement with the deposited pseudotime while failing strict monotonic ordering of every state median.

The P25 workflow therefore distinguishes:

- global concordance;
- local state-order violations;
- branch-extension preservation;
- paired GDIS comparisons under alternative ordering.

Unqualified reconstructions are retained rather than discarded.

---

## Installation

### Recommended environment

The reference environment uses Python 3.11.

```bash
conda env create -f environment.yml
conda activate gdis_bio
```

The environment currently specifies:

```text
Python      3.11.16
NumPy       2.4.6
pandas      2.3.3
SciPy       1.17.1
Scanpy      1.11.5
AnnData     0.12.19
pyGDIS      1.0.0
```

Additional dependencies include:

- matplotlib
- scikit-learn
- openpyxl
- requests

### Pip alternative

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a complete scientific rerun, Conda is recommended because the workflow includes a substantial scientific Python stack.

---

## Quick start

Clone the repository and create the environment:

```bash
git clone https://github.com/hamiddi/GDIS-Bio.git
cd GDIS-Bio

conda env create -f environment.yml
conda activate gdis_bio
```

Check that the scientific scripts compile:

```bash
PYTHONPYCACHEPREFIX=/tmp/gdis_bio_pycache python -m compileall -q scripts
```

Validate the P25 pipeline:

```bash
bash scripts/p25_pipeline_validate.sh --smoke
bash scripts/p25_pipeline_validate.sh --verify
```

### Pipeline driver

The repository includes a transparent pipeline driver that executes the archived scientific scripts as separate Python processes without changing their analytical logic.

```bash
# Inspect all stages
python tools/pipeline.py --list

# Confirm that all stage scripts are present
python tools/pipeline.py --phase all --check

# Preview commands without executing the scientific analyses
python tools/pipeline.py --phase all --dry-run

# Original P1–P23 workflow
python tools/pipeline.py --phase core

# Reviewer-response P24–P29 workflow
python tools/pipeline.py --phase review

# Complete workflow
python tools/pipeline.py --phase all
```

The historical `run_all.sh` / `run_all.ps1` launchers execute the P1–P23 core workflow. `run_pipeline.sh` / `run_pipeline.ps1` invoke `tools/pipeline.py`.

Do not commit:

```text
__pycache__/
*.pyc
*.pyo
*.log
*.tmp
*.bak
```

The supplied `.gitignore` already excludes these file classes.

---

## Reproducibility and validation

GDIS-Bio distinguishes several levels of reproducibility.

### 1. Syntax and environment validation

At minimum:

```bash
PYTHONPYCACHEPREFIX=/tmp/gdis_bio_pycache python -m compileall -q scripts
bash scripts/p25_pipeline_validate.sh --smoke
```

### 2. Existing-output integrity validation

For the P25 alternative-ordering result tree:

```bash
bash scripts/p25_pipeline_validate.sh --verify
```

### 3. Scientific rerun

A full scientific rerun requires the public GEO inputs and regeneration of upstream intermediate files in stage order.

The large raw datasets and many intermediate matrices are intentionally not versioned in GitHub.

### 4. Release validation

Before creating a public release or archival DOI:

1. validate the environment;
2. compile all scientific scripts;
3. confirm that required public inputs are accessible;
4. rerun the intended pipeline stages;
5. compare generated numerical outputs with the reference evidence;
6. record software versions and the Git commit identifier;
7. create the release tag only after validation is complete.

---

## Key results represented in this repository

The following values summarize analyses represented by the repository and accompanying manuscript materials. They should be interpreted under the limitations described above.

### Pancreatic discovery

Across two independent pancreatic differentiations and both lineage branches, GDIS maxima localized near the independently annotated NEUROG3-early state.

The descriptive mean absolute peak-to-landmark distance was:

- **GDIS:** 0.0112
- **Total variance:** 0.0269

The small number of independent differentiation experiments does not support a claim of universal superiority.

### External cardiac validation

Using frozen GDIS settings, median peak-to-landmark distances were:

| Transition | GDIS median absolute landmark distance |
|---|---:|
| MES → CMES | 0.0376 |
| PROG → CM | 0.0954 |
| PROG → CF | 0.0405 |

Conventional metrics equaled or outperformed GDIS for some cardiac transitions.

### Component ablation

The controlled P24 component-ablation analysis retained the frozen external-validation design. None of the 18 component-removal comparisons was significant after Benjamini-Hochberg correction; the minimum adjusted q-value was **0.4455**.

### Ordering robustness

P25 evaluates perturbations of the deposited ordering and alternative pseudotime/geodesic reconstruction. The strict graph-geodesic state-order audit qualified **26 of 48** individual×scope profiles.

This binary criterion is deliberately more stringent than global rank concordance; the repository retains failed strict-order profiles for transparent downstream comparison.

---

## Repository structure

The intended GitHub release is organized as follows:

```text
GDIS-Bio/
├── README.md
├── CITATION.cff
├── LICENSE
├── CHANGELOG.md
├── environment.yml
├── requirements.txt
├── scripts/
│   ├── p1_...py
│   ├── ...
│   ├── p25_...py
│   ├── p25b_...py
│   ├── ...
│   ├── p25m_...py
│   └── p25_pipeline_validate.sh
├── tools/
│   ├── check_environment.py
│   └── pipeline.py
├── docs/
│   ├── dataset_descriptions.md
│   ├── reproducibility_workflow.md
│   ├── statistical_tests.md
│   ├── reviewer_analyses.md
│   ├── manuscript_mapping.md
│   └── data_provenance.md
├── raw_data/
├── results/
├── manuscript_assets/
│   ├── figures/
│   └── tables/
├── run_all.sh
├── run_all.ps1
├── run_pipeline.sh
└── run_pipeline.ps1
```

The top-level launchers and the `tools/` utilities are included in this repository and are checked during release preparation.

---

## Data availability

The analyses use publicly available datasets.

### Pancreatic differentiation

- GEO accession: **GSE114412**
- Study: Veres et al., *Nature* (2019)
- GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE114412
- DOI: https://doi.org/10.1038/s41586-019-1168-5

### Cardiac differentiation

- GEO accession: **GSE175634**
- Study: Elorbany et al., *PLoS Genetics* (2022)
- GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE175634
- DOI: https://doi.org/10.1371/journal.pgen.1009666

Large raw data files, generated `.h5ad` objects, and many intermediate state-space/window matrices are not intended to be stored directly in GitHub.

---

## Citation

### Citing GDIS-Bio

If you use this repository, please cite the accompanying GDIS-Bio manuscript and the versioned software release.

Until a final journal DOI and archival release DOI are available, use the metadata in [`CITATION.cff`](CITATION.cff).

Suggested software citation:

> Ismail H, Harb A, William B, Bikdash M. **GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing Transcriptional State Transitions in Single-Cell Trajectories.** Version 1.1.0. GitHub repository, 2026. https://github.com/hamiddi/GDIS-Bio

Suggested BibTeX:

```bibtex
@software{ismail_gdisbio_2026,
  author  = {Ismail, Hamid and Harb, Ahmed and William, Basem and Bikdash, Marwan},
  title   = {GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing Transcriptional State Transitions in Single-Cell Trajectories},
  year    = {2026},
  version = {1.1.0},
  url     = {https://github.com/hamiddi/GDIS-Bio}
}
```

When a journal DOI and/or Zenodo DOI becomes available, update both this section and `CITATION.cff`.

### Citing pyGDIS

The GDIS calculations use **pyGDIS v1.0.0**.

Until the companion methodological article has final publication metadata, cite the software repository:

> Ismail HD, Harb A, Bikdash M. **pyGDIS: Generalized Dynamical Instability Score.** Version 1.0.0. 2026. https://github.com/hamiddi/pygdis

See also:

- PyPI: https://pypi.org/project/pygdis/
- Source: https://github.com/hamiddi/pygdis

### Citing the source datasets

Please also cite the original biological datasets when using analyses derived from them.

**Pancreatic dataset**

> Veres A, Faust AL, Bushnell HL, et al. Charting cellular identity during human in vitro β-cell differentiation. *Nature*. 2019;569:368–373. https://doi.org/10.1038/s41586-019-1168-5

**Cardiac dataset**

> Elorbany R, Popp JM, Rhodes K, et al. Single-cell sequencing reveals lineage-specific dynamic genetic regulation of gene expression during human cardiomyocyte differentiation. *PLoS Genetics*. 2022;18(1):e1009666. https://doi.org/10.1371/journal.pgen.1009666

---

## References

1. **Veres A, Faust AL, Bushnell HL, et al.** Charting cellular identity during human in vitro β-cell differentiation. *Nature*. 2019;569:368–373. doi:10.1038/s41586-019-1168-5.

2. **Elorbany R, Popp JM, Rhodes K, et al.** Single-cell sequencing reveals lineage-specific dynamic genetic regulation of gene expression during human cardiomyocyte differentiation. *PLoS Genetics*. 2022;18(1):e1009666. doi:10.1371/journal.pgen.1009666.

3. **Wolf FA, Angerer P, Theis FJ.** SCANPY: large-scale single-cell gene expression data analysis. *Genome Biology*. 2018;19:15. doi:10.1186/s13059-017-1382-0.

4. **Ismail HD, Harb A, Bikdash M.** pyGDIS: Generalized Dynamical Instability Score. Version 1.0.0. 2026. https://github.com/hamiddi/pygdis.

5. **Ismail H, Harb A, William B, Bikdash M.** GDIS-Bio: A Generalized Dynamical Instability Framework for Localizing Transcriptional State Transitions in Single-Cell Trajectories. Version 1.1.0. 2026. https://github.com/hamiddi/GDIS-Bio.

---

## Documentation

Additional repository documentation:

- [`docs/dataset_descriptions.md`](docs/dataset_descriptions.md) — datasets, accessions, and biological context
- [`docs/reproducibility_workflow.md`](docs/reproducibility_workflow.md) — stage ordering and dependencies
- [`docs/statistical_tests.md`](docs/statistical_tests.md) — statistical tests and units of inference
- [`docs/reviewer_analyses.md`](docs/reviewer_analyses.md) — P24–P29 reviewer-response analyses
- [`docs/manuscript_mapping.md`](docs/manuscript_mapping.md) — code-to-figure/table mapping
- [`docs/data_provenance.md`](docs/data_provenance.md) — provenance of distributed artifacts
- [`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md) — release preparation checklist

---

## License and support

GDIS-Bio is distributed under the [MIT License](LICENSE).

For reproducibility problems, open an issue in the GitHub repository and include:

- script/stage name;
- operating system;
- Python/Conda environment;
- relevant package versions;
- command used;
- error traceback or log.

Do not include passwords, access tokens, private credentials, or identifiable/private data in issue reports.

---

## Acknowledgment

If this repository contributes to a publication, presentation, educational resource, or derivative analysis, please cite **GDIS-Bio**, **pyGDIS**, and the relevant source dataset(s).
