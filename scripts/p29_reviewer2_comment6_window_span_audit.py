#!/usr/bin/env python3
"""P29 reviewer-2 comment 6: read-only audit of frozen fixed-cell pseudotime spans.
Run from GDIS_Bio project root with pandas/numpy. Produces descriptive tables only.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.cwd()
OUT = ROOT / 'results/p29_reviewer2_comment6_window_span_audit'
PAN = ROOT / 'results/p7_trajectory_assembly_validation/tables/04_primary_window_metadata.csv'
CARD_DIR = ROOT / 'results/p17_external_GSE175634_trajectory_window_assembly/data/primary_w400_s100'
CARD_PEAKS = ROOT / 'results/p18_external_GSE175634_primary_gdis/tables/02_primary_profile_summary.csv'
REQUIRED = ['window_index','n_cells','parameter_median_pseudotime','pseudotime_min','pseudotime_max','pseudotime_span']

def check(df, source):
    miss=[c for c in REQUIRED if c not in df]
    if miss: raise RuntimeError(f'Missing columns in {source}: {miss}')
    for c in REQUIRED: df[c]=pd.to_numeric(df[c], errors='raise')
    if df.empty or df[REQUIRED].isna().any().any(): raise RuntimeError(f'Invalid numeric values in {source}')
    if not (df.n_cells==400).all(): raise RuntimeError(f'Non-400 cell window in {source}')
    e=(df.pseudotime_max-df.pseudotime_min-df.pseudotime_span).abs().max()
    if e>1e-8: raise RuntimeError(f'Span mismatch in {source}: {e}')
    if (df.pseudotime_span<0).any(): raise RuntimeError(f'Negative span in {source}')
    if source == 'pancreatic':
        keys = ['differentiation', 'branch', 'window_index']
    else:
        keys = ['scope', 'individual', 'window_index']
    if df.duplicated(subset=keys).any():
        raise RuntimeError(f'Duplicate window index within a trajectory in {source}')
    if 'start_rank' in df and 'end_rank_exclusive' in df:
        if not ((df.end_rank_exclusive-df.start_rank)==400).all():
            raise RuntimeError(f'Window rank size mismatch in {source}')
    return df

def summarize(df, group_cols):
    rows=[]
    for key,g in df.groupby(group_cols, dropna=False, sort=True):
        key=key if isinstance(key,tuple) else (key,)
        v=g.pseudotime_span.to_numpy(dtype=float)
        row=dict(zip(group_cols,key))
        row.update(n_windows=len(v),span_min=np.min(v),span_q25=np.quantile(v,.25),
                   span_median=np.median(v),span_q75=np.quantile(v,.75),
                   span_max=np.max(v),span_mean=np.mean(v),span_sd=np.std(v,ddof=1) if len(v)>1 else np.nan,
                   max_to_min_ratio=(np.max(v)/np.min(v) if np.min(v)>0 else np.nan))
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    if not PAN.is_file(): raise SystemExit(f'Missing: {PAN}')
    if not CARD_DIR.is_dir(): raise SystemExit(f'Missing: {CARD_DIR}')
    if not CARD_PEAKS.is_file(): raise SystemExit(f'Missing: {CARD_PEAKS}')
    p=check(pd.read_csv(PAN),'pancreatic')
    if len(p)!=167: raise RuntimeError(f'Expected 167 pancreatic windows, got {len(p)}')
    if p.groupby(['differentiation','branch']).ngroups!=4: raise RuntimeError('Expected 4 pancreatic trajectories')
    p=p.assign(dataset='pancreatic',scope='endocrine',individual='',trajectory_id=p.differentiation.astype(str)+'_'+p.branch.astype(str))
    pc=[]
    for path in sorted(CARD_DIR.glob('*/*_windows.csv')):
        d=check(pd.read_csv(path),str(path))
        if d.scope.nunique()!=1 or d.individual.nunique()!=1: raise RuntimeError(f'Mixed cohort file {path}')
        d['source_file']=str(path.relative_to(ROOT))
        pc.append(d)
    if not pc: raise RuntimeError('No primary cardiac window files')
    c=pd.concat(pc,ignore_index=True)
    c['individual']=c.individual.astype(str)
    c['trajectory_id']=c.scope.astype(str)+'_'+c.individual
    c=c.assign(dataset='cardiac',branch=np.nan,differentiation=np.nan)
    peaks=pd.read_csv(CARD_PEAKS)
    peaks['individual']=peaks.individual.astype(str)
    keys=['scope','individual']
    if peaks.duplicated(keys).any() or c.duplicated(keys+['window_index']).any():
        raise RuntimeError('Duplicate cardiac individual/scope or window index')
    if len(peaks)!=48:
        raise RuntimeError(f'Expected 48 frozen P18 peak records, got {len(peaks)}')
    seen=set(map(tuple,c[keys].drop_duplicates().to_numpy()))
    expected=set(map(tuple,peaks[keys].drop_duplicates().to_numpy()))
    missing=expected-seen
    excluded=seen-expected
    if missing:
        raise RuntimeError(f'Frozen P18 profiles missing from P17 windows: {sorted(missing)}')
    expected_excluded = {('cm_extension', '19108'), ('cf_extension', '19108')}
    if excluded != expected_excluded:
        raise RuntimeError(
            f'Unexpected pre-evaluability exclusions: found {sorted(excluded)}, '
            f'expected {sorted(expected_excluded)}. Check cohort identifiers before proceeding.'
        )
    # P17 is a pre-evaluability archive and includes profiles omitted from P18.
    # Audit only frozen P18 profiles; preserve the original P17 files unchanged.
    excluded_df=c[c[keys].apply(tuple,axis=1).isin(excluded)].copy()
    c=c.merge(peaks[keys],on=keys,how='inner',validate='many_to_one')
    if c.groupby(keys).ngroups!=48:
        raise RuntimeError('Frozen P18 cohort selection did not yield 48 profiles')
    print('Archived P17 profiles excluded from frozen P18 analysis:', sorted(excluded))
    # Check each profile length against frozen summary.
    sizes=c.groupby(keys).size().rename('actual_n_windows').reset_index()
    lengths=peaks[keys+['n_windows']].merge(sizes,on=keys,validate='one_to_one',indicator=True)
    if not (lengths['_merge']=='both').all(): raise RuntimeError('Missing cardiac window counts')
    if not (lengths.n_windows==lengths.actual_n_windows).all():
        bad=lengths.loc[lengths.n_windows!=lengths.actual_n_windows, keys+['n_windows','actual_n_windows']]
        raise RuntimeError(f'Window count mismatches: {bad.to_dict(orient="records")}')
    # GDIS peak window spans: no GDIS recalculation.
    pk=peaks[keys+['gdis_peak_window_index','gdis_peak_parameter','frozen_destination_landmark']].copy()
    pk['gdis_peak_window_index']=pd.to_numeric(pk.gdis_peak_window_index)
    window_lookup=c[keys+['window_index','pseudotime_span','parameter_median_pseudotime']]
    a=pk.merge(window_lookup,left_on=keys+['gdis_peak_window_index'],right_on=keys+['window_index'],how='left',validate='one_to_one')
    if a.pseudotime_span.isna().any():
        raise RuntimeError(f'Missing cardiac peak windows: {a.loc[a.pseudotime_span.isna(), keys].to_dict(orient="records")}')
    if (a.gdis_peak_parameter-a.parameter_median_pseudotime).abs().max()>1e-8: raise RuntimeError('Peak parameter mismatch')
    med=c.groupby(keys).pseudotime_span.median().rename('profile_median_window_span').reset_index()
    a=a.merge(med,on=keys,validate='one_to_one')
    a['peak_to_profile_median_span_ratio']=a.pseudotime_span/a.profile_median_window_span
    a=a.rename(columns={'pseudotime_span':'gdis_peak_window_span'})
    # Optional correlation only descriptive, windows overlap and are not independent.
    rows=[]
    for (scope,individual),g in c.groupby(keys):
        q=g[['parameter_median_pseudotime','pseudotime_span']].corr(method='spearman').iloc[0,1]
        rows.append(dict(scope=scope,individual=individual,span_vs_pseudotime_spearman=q))
    correlations=pd.DataFrame(rows)
    OUT.mkdir(parents=True,exist_ok=True)
    p.to_csv(OUT/'01_pancreatic_windows_167.csv',index=False)
    c.to_csv(OUT/'02_cardiac_primary_windows.csv',index=False)
    excluded_df.to_csv(OUT/'00_cardiac_pre_evaluability_excluded_windows.csv',index=False)
    pa=summarize(p,['differentiation','branch'])
    ca=summarize(c,keys)
    scope=summarize(c,['scope'])
    pa.to_csv(OUT/'03_pancreatic_trajectory_span_summary.csv',index=False)
    ca.to_csv(OUT/'04_cardiac_individual_scope_span_summary.csv',index=False)
    scope.to_csv(OUT/'05_cardiac_scope_pooled_span_descriptive.csv',index=False)
    a.to_csv(OUT/'06_cardiac_gdis_peak_window_span.csv',index=False)
    correlations.to_csv(OUT/'07_span_vs_pseudotime_position_by_profile.csv',index=False)
    print('\n=== P29 FROZEN WINDOW SPAN AUDIT ===')
    print('Pancreatic: 4 trajectories / 167 windows; Cardiac: 48 individual-scope profiles /',len(c),'windows')
    print('PASS: pseudotime-span arithmetic, 400-cell window lengths, cardiac cohort and peak indexing')
    print('\nPANCREATIC BY TRAJECTORY:')
    print(pa.to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    print('\nCARDIAC BY SCOPE (pooled windows; descriptive only):')
    print(scope.to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    print('\nCARDIAC PEAK WINDOW SPAN / WITHIN-PROFILE MEDIAN SPAN:')
    print(a.groupby('scope').peak_to_profile_median_span_ratio.agg(['count','median','min','max']).to_string(float_format=lambda x:f'{x:.4f}'))
    print('\nNOTE: The comparison is observational; no fixed-pseudotime-width reanalysis or causal inference was performed.')
    print('NOTE: Overlapping windows, shared cardiac backbones, and pancreatic branches are NOT independent samples.')
    print('OUTPUT:',OUT)

if __name__=='__main__': main()

