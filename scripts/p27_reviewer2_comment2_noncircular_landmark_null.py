#!/usr/bin/env python3
"""P27: endpoint-aware, non-circular *landmark-position* sensitivity null.

Keeps observed GDIS/benchmark profiles and their maxima FIXED. Within each
independent pancreatic differentiation, draws ONE uniform fractional position
and maps it to each branch's observed pseudotime support. Thus paired branches
have coordinated hypothetical landmark positions, without circular wraparound.

This is a sensitivity null for landmark-location specificity, NOT a replacement
for the P11 circular-shift null and NOT a biological model of landmark formation.
Original P7/P10/P11 outputs are read only. New results saved under P27.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

METRICS = ["gdis", "transition_energy", "total_variance",
           "mean_lag1_autocorrelation", "gaussian_differential_entropy",
           "mean_step_distance"]
GROUPS = [(d,b) for d in (1,2) for b in (0,1)]

def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = np.empty(len(p), float)
    ranked = p[order] * len(p) / np.arange(1, len(p)+1)
    q[order] = np.minimum.accumulate(ranked[::-1])[::-1].clip(0,1)
    return q

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path.cwd())
    ap.add_argument('--n-permutations',type=int,default=20000)
    ap.add_argument('--seed',type=int,default=20261103)
    args=ap.parse_args()
    if args.n_permutations < 100: raise ValueError('Use at least 100 draws')
    root=args.root.resolve()
    src=root/'results/p10_conventional_ews_benchmark/tables/01_all_metric_profiles.csv'
    refs=list((root/'results').rglob('07_primary_state_landmark_window_mapping.csv'))
    if len(refs)!=1: raise ValueError(f'Expected one landmark file, found {refs}')
    archived_path=root/'results/p11_statistical_alignment_null_validation/tables/03_group_localization_errors.csv'
    for f in (src,refs[0],archived_path):
        if not f.is_file(): raise FileNotFoundError(f)
    df=pd.read_csv(src); landmarks=pd.read_csv(refs[0]); archived=pd.read_csv(archived_path)
    for col in ['branch','differentiation','parameter']+METRICS:
        if col not in df: raise ValueError('Missing metric column '+col)
    for col in ['branch','differentiation','state','state_median_pseudotime']:
        if col not in landmarks: raise ValueError('Missing landmark column '+col)
    records=[]
    for d,b in GROUPS:
        g=(df[(df['differentiation']==d)&(df['branch']==b)]
           .sort_values('parameter',kind='stable').reset_index(drop=True))
        if len(g)<3: raise ValueError(f'Insufficient windows for differentiation {d}, branch {b}')
        x=g.parameter.to_numpy(float)
        if not np.all(np.isfinite(x)) or np.any(np.diff(x)<0) or x[-1]<=x[0]:
            raise ValueError(f'Invalid pseudotime support: {(d,b)}')
        hits=landmarks[(landmarks.differentiation==d)&(landmarks.branch==b)&
                       (landmarks.state=='neurog3_early')]
        if len(hits)!=1: raise ValueError(f'Expected one NEUROG3-early landmark for {(d,b)}')
        landmark=float(hits.iloc[0].state_median_pseudotime)
        if not x[0]<=landmark<=x[-1]:
            raise ValueError(f'Landmark outside observed window support: {(d,b)}')
        for metric in METRICS:
            v=g[metric].to_numpy(float)
            if not np.all(np.isfinite(v)): raise ValueError(f'Non-finite metric: {(d,b,metric)}')
            peak=float(x[int(np.argmax(v))]); error=abs(peak-landmark)
            a=archived[(archived.differentiation==d)&(archived.branch==b)&
                       (archived.metric==metric)]
            if len(a)!=1: raise ValueError(f'P11 reference mismatch: {(d,b,metric)}')
            frozen=float(a.iloc[0].absolute_localization_error)
            if not np.isclose(error,frozen,atol=1e-8,rtol=0):
                raise ValueError(f'P11 frozen audit FAILED: {(d,b,metric)} computed {error}, stored {frozen}')
            records.append(dict(differentiation=d,branch=b,metric=metric,
                                peak_parameter=peak,landmark=landmark,
                                pseudotime_min=float(x[0]),pseudotime_max=float(x[-1]),
                                observed_error=error))
    tab=pd.DataFrame(records)
    rng=np.random.default_rng(args.seed)
    # Draw a shared hypothetical landmark percentile for each differentiation;
    # each branch maps the percentile onto its original, unwrapped coordinate range.
    u={d:rng.uniform(0.0,1.0,size=args.n_permutations) for d in (1,2)}
    outputs=[]; draws=[]
    for metric in METRICS:
        subset=tab[tab.metric==metric]
        null=np.zeros(args.n_permutations,float)
        for r in subset.itertuples(index=False):
            pseudo_landmark=r.pseudotime_min+(r.pseudotime_max-r.pseudotime_min)*u[r.differentiation]
            null+=np.abs(r.peak_parameter-pseudo_landmark)/4.0
        observed=float(subset.observed_error.mean())
        p=(1+np.count_nonzero(null <= observed+1e-12))/(args.n_permutations+1)
        outputs.append(dict(metric=metric,observed_mean_error=observed,
                            null_mean=float(np.mean(null)),null_q025=float(np.quantile(null,.025)),
                            null_q975=float(np.quantile(null,.975)),p_lower=float(p),
                            n_draws=args.n_permutations))
        draws.append(pd.DataFrame({'metric':metric,'draw':np.arange(args.n_permutations),
                                   'null_mean_error':null}))
    out=pd.DataFrame(outputs)
    out['q_bh_6_metrics']=bh(out.p_lower.to_numpy())
    dest=root/'results/p27_reviewer2_comment2_noncircular_landmark_null'
    dest.mkdir(parents=True,exist_ok=True)
    tab.to_csv(dest/'01_verified_frozen_peaks.csv',index=False)
    out.to_csv(dest/'02_endpoint_aware_landmark_null.csv',index=False)
    pd.concat(draws,ignore_index=True).to_csv(dest/'03_null_draws.csv',index=False)
    (dest/'README_INTERPRETATION.txt').write_text(
        'P27 SECOND NULL: endpoint-aware landmark-position randomization.\n'
        'For each iteration, draw one independent uniform fractional pseudotime position per\n'
        'differentiation experiment; map the SAME fraction to both branch-specific supports.\n'
        'All observed profiles, maxima, and natural endpoint ordering remain unchanged.\n'
        'This avoids artificial endpoint wraparound and coordinates paired branches.\n'
        'H0: the prespecified landmark is exchangeable with a position drawn uniformly\n'
        'from each frozen profile pseudotime support, coordinated within differentiation.\n'
        'This exchangeability assumption is strong and is NOT a mechanistic/developmental\n'
        'generative model; significance does NOT establish biological instability.\n'
        'The null differs from P11/P26 (which reposition score profiles). P-values are\n'
        'not interchangeable across null models; compare conclusions, not magnitudes.\n'
        'Two differentiation blocks are not sufficient for population generalization.\n'
        'BH correction applies separately across six metric-level tests.\n'
        f'Permutation draws: {args.n_permutations}; seed: {args.seed}.\n')
    print('\n=== P27 ENDPOINT-AWARE LANDMARK-POSITION NULL ===')
    print(out.to_string(index=False,float_format=lambda z:f'{z:.6f}'))
    print('\nVerified all 24 metric-by-trajectory P11 peak errors.')
    print('OUTPUT:',dest)

if __name__=='__main__': main()

