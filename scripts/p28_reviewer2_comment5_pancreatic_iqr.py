#!/usr/bin/env python3
"""P28: post-review pancreatic NEUROG3-early IQR localization sensitivity.

READ ONLY: frozen P10 profiles, P11 peak errors, and original biological landmarks.
Results go to results/p28_reviewer2_comment5_pancreatic_iqr/.

If the frozen landmark table already contains q25/q75, use them. Otherwise provide
--cells PATH to a cell-level metadata CSV/TSV/parquet/h5ad with branch,
differentiation, state, and deposited pseudotime columns. Cell-derived medians
must reproduce the frozen discovery landmark medians; otherwise abort.

A median state coordinate / IQR is NOT a known transition onset / onset interval.
IQR analysis is post-review complementary, not retrospectively co-primary.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

METRICS = ["gdis", "transition_energy", "total_variance",
           "mean_lag1_autocorrelation", "gaussian_differential_entropy",
           "mean_step_distance"]
GROUPS = [(d,b) for d in (1,2) for b in (0,1)]
QUARTILE_PAIRS = [
    ('state_q25_pseudotime','state_q75_pseudotime'),
    ('state_pseudotime_q25','state_pseudotime_q75'),
    ('state_pseudotime_25th','state_pseudotime_75th'),
    ('state_pseudotime_q1','state_pseudotime_q3'),
    ('state_pseudotime_p25','state_pseudotime_p75'),
    ('state_pseudotime_25','state_pseudotime_75'),
    ('state_q1_pseudotime','state_q3_pseudotime'),
    ('q25','q75'), ('q1','q3'), ('p25','p75'),
]

def load_cells(path):
    suffix=path.suffix.lower()
    if suffix=='.csv': return pd.read_csv(path)
    if suffix in ('.tsv','.txt'): return pd.read_csv(path,sep='\t')
    if suffix=='.parquet': return pd.read_parquet(path)
    if suffix=='.h5ad':
        import anndata as ad
        obj=ad.read_h5ad(path,backed='r')
        try: return obj.obs.copy().reset_index()
        finally: obj.file.close()
    raise ValueError('Unsupported cells file. Supply .csv, .tsv, .parquet, or .h5ad')

def canonicalize_cells(df,args):
    cols={str(c).lower():c for c in df.columns}
    defaults={
        'branch':('branch','pseudotime_branch','endo_branch','pseudotimebranch'),
        'differentiation':('differentiation','differentiation_replicate','replicate','diff','differentiationreplicate'),
        'state':('state','cell_state','celltype','cell_type','celltype_label','assigned_state'),
        'pseudotime':('pseudotime','pseudo_time','endo_pseudotime','deposited_pseudotime','parameter','dpt_pseudotime'),
    }
    chosen={}
    for field,possibilities in defaults.items():
        explicit=getattr(args,field+'_column')
        if explicit:
            if explicit not in df.columns: raise ValueError(f'Column {explicit!r} missing; columns={list(df.columns)}')
            chosen[field]=explicit
        else:
            cand=next((cols[k] for k in possibilities if k in cols),None)
            if cand is None: raise ValueError(f'Cannot identify {field} column. Use --{field}-column. Found: {list(df.columns)}')
            chosen[field]=cand
    x=df[list(chosen.values())].copy()
    x.columns=list(chosen.keys())
    # Exact state label after simple whitespace/case normalization only.
    x['state']=x.state.astype(str).str.strip().str.lower()
    x=x[x.state.eq('neurog3_early')].copy()
    x['pseudotime']=pd.to_numeric(x.pseudotime,errors='coerce')
    x['branch']=pd.to_numeric(x.branch,errors='coerce')
    x['differentiation']=pd.to_numeric(x.differentiation,errors='coerce')
    x=x.dropna(subset=['branch','differentiation','pseudotime'])
    if x.empty: raise ValueError('No NEUROG3-early cells with valid group and pseudotime metadata.')
    return x

def validate_quartiles(iqr,landmarks, tol):
    rec=[]
    for d,b in GROUPS:
        q=iqr[(iqr.differentiation==d)&(iqr.branch==b)]
        landmark=landmarks[(landmarks.differentiation==d)&(landmarks.branch==b)&
                           (landmarks.state.astype(str).str.strip().str.lower()=='neurog3_early')]
        if len(q)!=1 or len(landmark)!=1:
            raise ValueError(f'Expected exactly one IQR and frozen landmark for {(d,b)}; found {len(q)},{len(landmark)}')
        row=q.iloc[0]
        actual=float(landmark.iloc[0].state_median_pseudotime)
        median=float(row.state_median_pseudotime)
        lo=float(row.state_q25_pseudotime); hi=float(row.state_q75_pseudotime)
        if not np.isfinite([lo,median,hi]).all() or not lo<=median<=hi:
            raise ValueError(f'Invalid IQR or median for {(d,b)}: {lo}, {median}, {hi}')
        if not np.isclose(median,actual,rtol=0,atol=tol):
            raise ValueError(f'FROZEN MEDIAN AUDIT FAILED for {(d,b)}: IQR metadata median {median:.12g} != original {actual:.12g}; check source cell population and deduplication.')
        rec.append(dict(differentiation=d,branch=b,n_state_cells=int(row.n_state_cells) if 'n_state_cells' in row and pd.notna(row.n_state_cells) else np.nan,
                        state_q25_pseudotime=lo,state_median_pseudotime=actual,state_q75_pseudotime=hi,
                        iqr_width=hi-lo))
    return pd.DataFrame(rec)

def main():
    p=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--cells',type=Path,help='Original cell-level endocrine pseudotime metadata (required if frozen landmark table lacks IQR columns)')
    p.add_argument('--branch-column');p.add_argument('--differentiation-column')
    p.add_argument('--state-column');p.add_argument('--pseudotime-column')
    p.add_argument('--median-tolerance',type=float,default=1e-7)
    a=p.parse_args()
    root=a.root.resolve()
    profile_path=root/'results/p10_conventional_ews_benchmark/tables/01_all_metric_profiles.csv'
    archived_path=root/'results/p11_statistical_alignment_null_validation/tables/03_group_localization_errors.csv'
    refs=list((root/'results').rglob('07_primary_state_landmark_window_mapping.csv'))
    if len(refs)!=1: raise RuntimeError(f'Expected exactly one frozen state landmark file, found {len(refs)}: {refs}')
    for f in (profile_path,archived_path,refs[0]):
        if not f.is_file(): raise FileNotFoundError(f)
    profiles=pd.read_csv(profile_path); archived=pd.read_csv(archived_path); landmarks=pd.read_csv(refs[0]);
    required_profile=['branch','differentiation','parameter']+METRICS
    for c in required_profile:
        if c not in profiles: raise ValueError(f'Missing {c} in {profile_path}')
    for c in ['differentiation','branch','state','state_median_pseudotime']:
        if c not in landmarks: raise ValueError(f'Missing {c} in {refs[0]}')
    quartile_columns=next(((x,y) for x,y in QUARTILE_PAIRS if x in landmarks and y in landmarks),None)
    if a.cells is not None:
        raw=load_cells(a.cells.expanduser().resolve())
        x=canonicalize_cells(raw,a)
        rows=[]
        for (d,b),g in x.groupby(['differentiation','branch']):
            v=g.pseudotime.to_numpy(float)
            q25,med,q75=np.quantile(v,[.25,.50,.75])
            rows.append(dict(differentiation=d,branch=b,n_state_cells=len(v),state_q25_pseudotime=q25,state_median_pseudotime=med,state_q75_pseudotime=q75))
        iqr_raw=pd.DataFrame(rows)
        origin=f'cell-level {a.cells}'
    elif quartile_columns:
        q25,q75=quartile_columns
        iqr_raw=(landmarks[landmarks.state.astype(str).str.strip().str.lower().eq('neurog3_early')]
                 .rename(columns={q25:'state_q25_pseudotime',q75:'state_q75_pseudotime'}))
        origin=f'archived landmark table quartiles {quartile_columns}'
    else:
        print('IQR QUARTILES ARE NOT PRESENT IN THE ARCHIVED LANDMARK TABLE.')
        print('Frozen landmark columns:',list(landmarks.columns))
        print('Provide an original per-cell metadata file with --cells PATH (not a window-level profile).')
        print('Inspect candidate metadata paths without modifying the frozen results:')
        print("  find data results -type f \\( -iname '*pseudotime*.csv' -o -iname '*metadata*.csv' -o -iname '*pseudotime*.tsv' -o -iname '*.h5ad' \\) | head -80")
        raise SystemExit(2)
    interval=validate_quartiles(iqr_raw,landmarks,a.median_tolerance)
    rows=[]
    for d,b in GROUPS:
        g=profiles[(profiles.differentiation==d)&(profiles.branch==b)].sort_values('parameter',kind='stable')
        I=interval[(interval.differentiation==d)&(interval.branch==b)].iloc[0]
        if len(g)<3: raise ValueError(f'Incomplete profile for {(d,b)}')
        x=g.parameter.to_numpy(float)
        if not np.all(np.isfinite(x)) or np.any(np.diff(x)<0): raise ValueError(f'Invalid frozen coordinates for {(d,b)}')
        if not x[0]<=I.state_median_pseudotime<=x[-1]: raise ValueError(f'Frozen median not in analysis support for {(d,b)}')
        for metric in METRICS:
            v=g[metric].to_numpy(float)
            if not np.all(np.isfinite(v)): raise ValueError(f'Non-finite {metric} for {(d,b)}')
            idx=int(np.argmax(v)); peak=float(x[idx]); median=float(I.state_median_pseudotime)
            q1=float(I.state_q25_pseudotime); q3=float(I.state_q75_pseudotime)
            median_error=abs(peak-median)
            prior=archived[(archived.differentiation==d)&(archived.branch==b)&(archived.metric==metric)]
            if len(prior)!=1: raise ValueError(f'Missing frozen P11 peak error for {(d,b,metric)}')
            ref_error=float(prior.iloc[0].absolute_localization_error)
            if not np.isclose(median_error,ref_error,rtol=0,atol=1e-8):
                raise ValueError(f'P11 AUDIT FAILED for {(d,b,metric)}: {median_error:.12g} vs {ref_error:.12g}')
            interval_error=max(q1-peak,0.0,peak-q3)
            # Floating-point boundary tolerance is numerical only, not a widened biological IQR.
            inside=(q1-1e-10)<=peak<=(q3+1e-10)
            if inside and interval_error<=1e-10: interval_error=0.0
            rows.append(dict(differentiation=d,branch=b,metric=metric,peak_parameter=peak,
                             state_q25=q1,state_median=median,state_q75=q3,iqr_width=q3-q1,
                             within_iqr=int(inside),distance_to_iqr=float(interval_error),
                             distance_to_median=median_error,peak_position=('inside' if inside else 'before' if peak<q1 else 'after')))
    details=pd.DataFrame(rows)
    if len(details)!=24: raise ValueError(f'Expected 24 metric-by-trajectory rows, found {len(details)}')
    summaries=[]
    for metric,g in details.groupby('metric',sort=False):
        summaries.append(dict(metric=metric,n_trajectories=len(g),n_in_iqr=int(g.within_iqr.sum()),
                              proportion_in_iqr=float(g.within_iqr.mean()),
                              mean_distance_to_iqr=float(g.distance_to_iqr.mean()),
                              median_distance_to_iqr=float(g.distance_to_iqr.median()),
                              mean_distance_to_median=float(g.distance_to_median.mean()),
                              n_before=int((g.peak_position=='before').sum()),
                              n_after=int((g.peak_position=='after').sum())))
    summary=pd.DataFrame(summaries)
    # Branches within differentiation share cells: summarize without pretending 4 independent replicates.
    by_diff=(details.groupby(['differentiation','metric'],sort=False)
             .agg(branches_inside=('within_iqr','sum'),mean_distance_to_iqr=('distance_to_iqr','mean'),
                  mean_distance_to_median=('distance_to_median','mean')).reset_index())
    dest=root/'results/p28_reviewer2_comment5_pancreatic_iqr'
    dest.mkdir(parents=True,exist_ok=True)
    interval.to_csv(dest/'01_neurog3_early_state_iqr_by_trajectory.csv',index=False,float_format='%.12g')
    details.to_csv(dest/'02_metric_peak_iqr_detail_24.csv',index=False,float_format='%.12g')
    summary.to_csv(dest/'03_metric_iqr_summary_6.csv',index=False,float_format='%.12g')
    by_diff.to_csv(dest/'04_differentiation_block_summary_12.csv',index=False,float_format='%.12g')
    (dest/'README_INTERPRETATION.txt').write_text(
        'P28 POST-REVIEW SECONDARY SENSITIVITY ANALYSIS; not a retrospectively prespecified co-primary endpoint.\n'
        'NEUROG3-early q25/q75 computed from cell-level annotated state pseudotime or archived cell-derived quantiles.\n'
        'IQR denotes the middle 50% of NEUROG3-early annotated cells in pseudotime, NOT biological onset interval.\n'
        'Frozen P10 peaks and all 24 original P11 median localization errors are audited exactly.\n'
        'Four branch x differentiation trajectories arise from TWO independent differentiation experiments.\n'
        'IQR inclusion and distances are descriptive; NO inferential p-values or independent-replication claims.\n'
        'A broad reference interval can yield inclusion by chance; width-adjusted null needed for significance.\n'
        f'IQR source: {origin}\n'
        f'Median identity tolerance (absolute): {a.median_tolerance}.\n')
    (dest/'manifest.json').write_text(json.dumps({'analysis':'P28','source_profiles':str(profile_path),
        'source_archived_peaks':str(archived_path),'source_frozen_landmark':str(refs[0]),
        'source_iqr':origin,'metric_count':6,'trajectory_count':4,'audited_error_count':24},indent=2))
    print('\n=== P28 NEUROG3-EARLY IQR ===')
    print('Frozen median and P11 peak audit: PASS (24/24 metric-by-trajectory peak errors)')
    print('Quartile source:',origin)
    print('\nANNOTATED NEUROG3-EARLY STATE INTERVALS:')
    print(interval.to_string(index=False,float_format=lambda z:f'{z:.6f}'))
    print('\nMETRIC SUMMARY (DESCRIPTIVE, NOT 4 INDEPENDENT REPLICATES):')
    print(summary.to_string(index=False,float_format=lambda z:f'{z:.6f}'))
    print('\nDETAILS:',dest/'02_metric_peak_iqr_detail_24.csv')
    print('OUTPUT:',dest)

if __name__=='__main__': main()

