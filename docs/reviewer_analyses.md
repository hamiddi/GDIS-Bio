# Reviewer-response analyses (P24–P29)

This study uses **cross-sectional scRNA-seq**; pseudotime is **not physical time**. GDIS maxima do not demonstrate a bifurcation, transition onset, or causal instability. Frozen analysis parameters (50 PCs; 400-cell windows at 100-cell steps; pyGDIS 1.0.0) were not retuned for external data.

| Family | Analysis | v5 evidence | Statistical/interpretive limit |
|---|---|---|---|
| P24 | Controlled cardiac component ablation | Supplementary Figure S7 | 18 comparisons; none significant after BH (minimum q=0.4455); some removals improve localization |
| P25 | Cell-rank perturbations, alternative DPT and geodesic ordering, paired robustness | Supplementary Figure S8; Tables S15–S19 | Individuals are independent only within analysis scope; rankings change with ordering; unqualified geodesic reconstructions are retained |
| P26 | Coordinated fractional circular shift and differentiation-level tests | Table S8 | Two independent pancreatic differentiation experiments; circular shifts introduce an artificial endpoint join |
| P27 | Endpoint-aware, non-circular random-landmark null | Table S9 | Hypothetical reference-position exchangeability is not a model of real biology |
| P28 | NEUROG3-early state IQR sensitivity | Table S10 | 4 dependent trajectories; exploratory post-review interval; IQR is not an onset interval |
| P29 | Fixed-cell window pseudotime-span audit | Table S20 | Descriptive audit; NOT a fixed-pseudotime-width recomputation |

## Reference headline values

- P24: T2 full-GDIS median absolute landmark error 0.0376; T3-CM 0.0954; T3-CF 0.0405; 18 ablation comparisons and min adjusted q=0.4455.
- P25: 48 cardiac individual×scope profiles (19 shared, 15 CM, 14 CF); 1,440 rank-noise perturbed profiles; 26/48 alternative geodesic reconstructions meet the strict state-order criterion.
- P26: mean observed GDIS landmark error 0.011175, coordinated-shift null mean 0.264562, empirical p=0.001050, adjusted q=0.006300. Differentiation-level comparative p-values 0.25–0.50.
- P27: null mean 0.318243; interval 0.048099–0.630220; empirical p=0.001650, q=0.009900.
- P28: GDIS maximum within state IQR in 4/4 trajectories, compared with total variance 2/4, transition energy 1/4, others 0/4.
- P29: 167 pancreatic and 4,317 cardiac primary windows, 400 cells per window.

## Preservation and provenance

The uploaded P24–P29 outputs are stored as read-only, inspectable **reference artifacts** in `results/reference_outputs/reviewer_analyses/`; they are not a claim that this packaged repository has been rerun. Some original P25 artifacts used an older supplementary numbering; these are kept only as explicitly named **legacy artifacts**. The updated figure-generation script is `p25m_make_figure_s8_and_auxiliary_tables.py`. Its four auxiliary tables **do not substitute** for the manuscript's final Tables S15–S19 (different grouping). The original `p25l...py` is preserved as historical code but excluded from the default pipeline.

The P28 public manifest has local absolute user paths redacted. See `docs/data_provenance.md`.
