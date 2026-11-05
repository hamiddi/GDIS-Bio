# Publication checklist — v1.1.0 candidate

## Before pushing

- [ ] Compare this package to current GitHub default branch and review all diffs before merging.
- [ ] Ensure original P1–P23 algorithms and their frozen reference files are unchanged (this archive preserves them as received).
- [ ] Confirm no protected/identifying datasets, passwords, absolute local metadata, or oversized artifacts were introduced.
- [ ] Verify P25m regenerated Figure S8 is visually identical in meaning to the submitted Figure S8.
- [ ] Confirm curated manuscript Tables S15–S19 match the new P25 evidence; the P25 auxiliary tables are not a direct substitute.
- [ ] Confirm Figure S7 component ablation publication artwork is supplied and maps to P24 results.
- [ ] Compare manuscript v5 Code Availability text with the actual GitHub release and DOI.
- [ ] Attempt a clean full rerun on a suitable Linux/HPC environment if claiming full independent reproducibility. Otherwise disclose that reference-result comparisons are provided and note missing generated intermediates.

## Offline commands

```bash
python tools/verify_reference.py
python tools/check_repository.py
python tools/pipeline.py --phase all --dry-run
python tools/pipeline.py --phase review --check  # requires upstream files; missing is expected in a clean checkout
```

## Safe Git workflow (run after personally reviewing code/results)

```bash
# Work in a clone of the real repository, on a separate branch
git checkout -b manuscript-v5-reproducibility
# Copy files from this candidate into the clone, preserving historical files
git status --short
git diff --stat
git add -A
git commit -m "Add manuscript v5 reproducibility analyses P24-P29"
# Push/merge only after human review and HPC rerun where feasible
```

## Archive and cite

After the commit is merged, tag a release such as `v1.1.0`, obtain the exact commit SHA, create a **new version** in the existing Zenodo record, and use the *actual* new version DOI in the manuscript. Do not replace or edit the original v1.0.0 archive. The pyGDIS library dependency remains pinned to pyGDIS v1.0.0.
