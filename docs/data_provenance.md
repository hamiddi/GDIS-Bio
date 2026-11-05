# Data provenance, inputs and privacy

- **GSE114412**: deposited pancreatic endocrine scRNA-seq, including differentiations and terminal branches, downloaded by P1.
- **GSE175634**: human iPSC-derived cardiac differentiation with independent individuals, downloaded/preflighted by P12.
- The frozen primary state space is 50 PCs; primary windows are 400 cells with step 100.
- Raw GEO files and large `.h5ad` / window matrices are excluded from this distribution, not from the required computation.
- Running `python tools/pipeline.py --phase all` requires internet for downloads and sufficient RAM and disk. The selected archived reference outputs alone cannot replace source `.h5ad` and window arrays.
- The review archives contain selected final CSVs and summary records; they were generated on an external research system and were not regenerated in this assembly session.
- Original private/local absolute paths in the P28 manifest were redacted for public release.
- Potentially identifying data and credentials should not be committed. Use the supplied `.gitignore`; review all untracked files with `git status --short` before pushing.
- For a transparent exact release, record the Git commit SHA, environment solver lock (if available), and an archived DOI **after** publication; do not cite a DOI for a release that has not yet been made.
