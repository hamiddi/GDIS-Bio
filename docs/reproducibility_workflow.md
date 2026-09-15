# Reproducibility workflow

The repository preserves the execution order used for the manuscript. Design and biological-landmark freezing occurs **before** GDIS calculation in both discovery and external-validation stages.

| Stage | Purpose |
|---|---|
| `p1` | Download GSE114412 Stage-5 public files |
| `p2` | Preflight raw discovery data |
| `p3` | Characterize discovery dataset |
| `p4` | Validate deposited endocrine trajectory |
| `p4b` | Validate differentiation replicates |
| `p5` | Preprocess expression data and construct the primary state space |
| `p6` | Evaluate PCA/state-space dimensionality sensitivity |
| `p6b` | Validate the 50-PC reference geometry |
| `p7` | Assemble and validate pseudotemporal sliding windows |
| `p8` | Run the first/frozen discovery GDIS analysis |
| `p9` | Validate discovery GDIS localization and replicate agreement |
| `p10` | Benchmark conventional transition-associated metrics |
| `p11` | Run discovery statistical/null validation |
| `p12` | Download and preflight GSE175634 |
| `p13` | Characterize external trajectories |
| `p14` | Freeze external cohort, landmarks, and design before GDIS |
| `p15` | Preprocess external data and construct the common state space |
| `p16` | Validate external state-space geometry before GDIS |
| `p17` | Assemble external trajectory windows without GDIS |
| `p17b` | Freeze final evaluable external GDIS groups |
| `p18` | Run the first/frozen external GDIS analysis |
| `p19` | Validate external GDIS statistically |
| `p20` | Run prespecified external GDIS sensitivity analyses |
| `p21` | Benchmark conventional metrics externally |
| `p22` | Run paired/statistical external benchmark comparison |
| `p23` | Freeze and consolidate final external evidence |

## One-command execution

Linux/macOS:
```bash
bash run_all.sh
```

Windows PowerShell:
```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1
```

The full run downloads >1 GB of compressed public data and creates large intermediate state-space and trajectory files. An HPC or high-memory workstation is recommended for the complete external-validation workflow.

## Frozen-analysis safeguards

- Biological landmarks are defined independently of GDIS.
- No dataset-specific retuning is performed for external validation.
- Primary state space: 50 PCs; lower-dimensional alternatives are sensitivity analyses.
- Primary windows: 400 cells with a 100-cell step; 300/75 and 500/125 are prespecified sensitivities.
- `pyGDIS==1.0.0` is used for the primary GDIS calculations.
- No universal biological GDIS threshold is imposed (`critical_value=None`).
- Overlapping windows are not treated as independent inferential replicates.
- Pseudotime ordering is not interpreted as physical-time tracking of the same cells.
