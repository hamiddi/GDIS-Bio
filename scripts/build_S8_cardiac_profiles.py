#!/usr/bin/env python3
"""Publication-oriented Supplementary Figure S8 (compact layout and readable footer) (frozen P18 cardiac results).

Read-only inputs: P18 individual GDIS CSVs and P18 primary summary.
Outputs: S8A/B/C 3-column individual-profile figures, S8D signed-offset
summary, and a strict, reproducible peak audit. No GDIS recalculation.

Run on Hyperion:
  python build_S8_cardiac_profiles_v5_compact.py --root "$PWD"
Optional: --out /path/to/new/output --dpi 600 --font-scale 1.4
"""
import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd

SCOPES = {
    'shared_backbone': ('A', 'MES → CMES (T2)', 19, 'CMES'),
    'cm_extension': ('B', 'PROG → CM (T3-CM)', 15, 'CM'),
    'cf_extension': ('C', 'PROG → CF (T3-CF)', 14, 'CF'),
}
STATE_COLORS = {
    'IPSC': '#756bb1', 'MES': '#3182bd', 'CMES': '#41ab8d',
    'PROG': '#dda73a', 'CM': '#c96982', 'CF': '#a86d50',
}
GDIS_COLOR = '#243849'
PEAK_COLOR = '#ca3e4e'
LANDMARK_COLOR = '#c67c25'


def extract_one(path, summaries):
    df = pd.read_csv(path, dtype={'individual': str})
    required = {'scope', 'individual', 'parameter', 'gdis', 'window_index',
                'frozen_destination_landmark', 'frozen_destination_state',
                'dominant_state'}
    missing = required.difference(df.columns)
    if missing or df.empty:
        raise ValueError(f'{path}: empty or missing columns {sorted(missing)}')
    scope, individual = str(df.scope.iloc[0]), str(df.individual.iloc[0])
    if scope not in SCOPES:
        raise ValueError(f'Unexpected scope {scope}: {path.name}')
    if not df.scope.eq(scope).all() or not df.individual.astype(str).eq(individual).all():
        raise ValueError(f'Mixed scope/individual in {path.name}')
    df = df.sort_values('window_index', kind='stable').reset_index(drop=True)
    x = df.parameter.to_numpy(dtype=float)
    y = df.gdis.to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(np.diff(x) < -1e-10):
        raise ValueError(f'Nonfinite or nonmonotonic data: {scope}/{individual}')
    if (y < -1e-8).any() or (y > 1 + 1e-8).any():
        raise ValueError(f'GDIS outside [0,1]: {scope}/{individual}')
    if df.window_index.duplicated().any():
        raise ValueError(f'Duplicate window indices: {scope}/{individual}')

    peak = df.iloc[int(np.argmax(y))]
    landmark = float(df.frozen_destination_landmark.iloc[0])
    dest = str(df.frozen_destination_state.iloc[0])
    if not np.isfinite(landmark) or not np.allclose(
        df.frozen_destination_landmark.astype(float).to_numpy(), landmark, rtol=0, atol=1e-10
    ):
        raise ValueError(f'Nonconstant landmark: {scope}/{individual}')
    if dest != SCOPES[scope][3] or not df.frozen_destination_state.astype(str).eq(dest).all():
        raise ValueError(f'Destination inconsistency: {scope}/{individual}')

    match = summaries[(summaries.scope.astype(str) == scope) &
                      (summaries.individual.astype(str) == individual)]
    if len(match) != 1:
        raise ValueError(f'Expected one reference summary for {scope}/{individual}, got {len(match)}')
    reference = match.iloc[0]
    comparisons = {
        'peak parameter': (float(peak.parameter), float(reference.gdis_peak_parameter)),
        'peak value': (float(peak.gdis), float(reference.gdis_peak_value)),
        'landmark': (landmark, float(reference.frozen_destination_landmark)),
        'absolute distance': (abs(float(peak.parameter) - landmark),
                              float(reference.gdis_absolute_peak_landmark_error)),
    }
    for metric, (value, expected) in comparisons.items():
        if not np.isclose(value, expected, atol=1e-8, rtol=0):
            raise ValueError(f'{scope}/{individual}: {metric}: {value} != {expected}')
    if 'n_windows' in reference and int(reference.n_windows) != len(df):
        raise ValueError(f'{scope}/{individual}: window count mismatch')

    # State fractions come from the frozen 400-cell windows, not inferred colors.
    active_states = [s for s in STATE_COLORS if f'fraction_{s}' in df.columns]
    if dest not in active_states:
        raise ValueError(f'{scope}/{individual}: destination fraction is missing')
    fractions = df[[f'fraction_{s}' for s in active_states]].to_numpy(float)
    if not np.isfinite(fractions).all() or (fractions < -1e-7).any():
        raise ValueError(f'Invalid state fractions: {scope}/{individual}')
    if not np.allclose(fractions.sum(axis=1), 1.0, atol=0.003, rtol=0):
        raise ValueError(f'State fractions do not total 1: {scope}/{individual}')
    fractions = np.maximum(fractions, 0)
    fractions = fractions / fractions.sum(axis=1, keepdims=True)

    signed = float(peak.parameter) - landmark
    row = {
        'scope': scope, 'individual': individual, 'n_windows': len(df),
        'destination_state': dest, 'landmark_DPT': landmark,
        'peak_DPT': float(peak.parameter), 'peak_GDIS': float(peak.gdis),
        'signed_peak_minus_landmark_DPT': signed,
        'abs_peak_to_landmark_DPT': abs(signed),
        'peak_dominant_state': str(peak.dominant_state),
        'peak_relative_to_landmark': ('before' if signed < 0 else 'after' if signed > 0 else 'equal'),
        'source_file': str(path),
    }
    return df, peak, landmark, active_states, fractions, row


def center_edges(x):
    """Edges between window-center DPTs; no claim of cell-level intervals."""
    if len(x) == 1:
        return np.array([x[0] - .002, x[0] + .002])
    mid = (x[:-1] + x[1:]) / 2
    edges = np.r_[x[0] - (mid[0] - x[0]), mid, x[-1] + (x[-1] - mid[-1])]
    return edges


def plot_composition(ax, x, states, fractions):
    """Compact true per-window state fractions, with DPT-width-aware bins.

    The raster is located in the bottom 12% of the plot, not smoothed.
    Each column represents one sliding window centered at its archived DPT.
    """
    nrows = 48
    v = (np.arange(nrows) + .5) / nrows
    # index of cumulative state reaching the vertical bin midpoint
    which = np.sum(v[:, None, None] > np.cumsum(fractions, axis=1)[None, :, :], axis=2)
    which = np.minimum(which, len(states) - 1)
    rgba = np.empty((nrows, len(x), 4), float)
    palette = np.array([mcolors.to_rgba(STATE_COLORS[s]) for s in states])
    rgba[:] = palette[which]
    edges = center_edges(x)
    transform = blended_transform_factory(ax.transData, ax.transAxes)
    ax.imshow(rgba, origin='lower', interpolation='nearest', aspect='auto',
              extent=(edges[0], edges[-1], .015, .105),
              transform=transform, zorder=3, clip_on=True)
    ax.plot([0, 1], [.105, .105], transform=ax.transAxes,
            color='#7e8790', lw=.55, alpha=.5, zorder=4)


def draw_profile_set(scope, profiles, out, dpi, font_scale=1.3):
    panel, title, n_expected, dest = SCOPES[scope]
    keys = sorted([key for key in profiles if key[0] == scope], key=lambda k: k[1])
    ncols = 3
    nrows = math.ceil(len(keys) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(17.5, 3.05 * nrows + 2.4),
                             sharey=True, squeeze=False)
    fig.suptitle(f'Supplementary Figure S8{panel}  |  {title}  |  n = {n_expected}',
                 x=.065, y=.987, ha='left', fontsize=17 * font_scale, fontweight='bold')

    # Shared x-axis range within each scope, containing every profile and landmark.
    x_max = max(max(float(profiles[k][0].parameter.max()), profiles[k][2]) for k in keys)
    x_max = min(1.0, x_max + .035)
    x_min = min(0., min(float(profiles[k][0].parameter.min()) for k in keys) - .01)
    for i, (ax, key) in enumerate(zip(axes.flat, keys)):
        df, peak, landmark, states, fractions, record = profiles[key]
        x = df.parameter.to_numpy(float)
        y = df.gdis.to_numpy(float)
        ax.plot(x, y, color=GDIS_COLOR, lw=1.55, zorder=2)
        ax.axvline(landmark, color=LANDMARK_COLOR, ls='--', lw=1.25, zorder=4)
        ax.scatter([float(peak.parameter)], [float(peak.gdis)], s=35,
                   color=PEAK_COLOR, edgecolor='white', lw=.55, zorder=5)
        plot_composition(ax, x, states, fractions)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, 1.02)
        ax.set_title(f'Individual {key[1]}  |  ΔDPT = {record["signed_peak_minus_landmark_DPT"]:+.3f}',
                     fontsize=10.0 * font_scale, pad=8)
        ax.grid(axis='y', color='#6c737a', alpha=.15, lw=.65)
        ax.tick_params(axis='both', labelsize=8.5 * font_scale)
        ax.set_yticks([0,.25,.5,.75,1.0])
        # Label the last occupied row and all columns for partially empty grids.
        if i // ncols == nrows - 1:
            ax.set_xlabel('Deposited DPT', fontsize=9.0 * font_scale, labelpad=5)
        if i % ncols == 0:
            ax.set_ylabel('GDIS', fontsize=9.0 * font_scale, labelpad=5)
    for ax in axes.flat[len(keys):]:
        ax.axis('off')

    shown_states = [s for s in STATE_COLORS if s in (["IPSC", "MES", "CMES", "PROG", dest])]
    handles = [
        Line2D([0], [0], color=GDIS_COLOR, lw=1.6, label='GDIS'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=PEAK_COLOR,
               markeredgecolor='white', markersize=7, label='Global maximum'),
        Line2D([0], [0], color=LANDMARK_COLOR, ls='--', lw=1.5, label='Frozen landmark'),
    ] + [Patch(facecolor=STATE_COLORS[s], label=s) for s in shown_states]
    # Put explanatory text and state legend in their OWN dedicated footer axis.
    # This prevents any overlap with the lowest row of subplot labels and gives
    # the legend a large, readable size in exported TIFF/PNG/PDF figures.
    footer = fig.add_axes([.07, .018, .86, .145], frameon=False)
    footer.set_xlim(0, 1)
    footer.set_ylim(0, 1)
    footer.set_axis_off()
    footer.text(
        .5, .91,
        'WINDOW CELL-STATE COMPOSITION (proportions): each colored strip represents overlapping 400-cell windows.',
        ha='center', va='top', fontsize=11.5 * font_scale,
        color='#28333d', fontweight='medium',
    )
    footer.text(
        .5, .70,
        'Signed ΔDPT = GDIS peak position − frozen destination-state median; '
        'negative values indicate earlier maxima. No smoothing.',
        ha='center', va='top', fontsize=10.5 * font_scale,
        color='#28333d',
    )
    footer.legend(
        handles=handles, loc='lower center', bbox_to_anchor=(.5, .005),
        ncol=4, frameon=False, fontsize=11.5 * font_scale,
        handlelength=2.25, handletextpad=.75, labelspacing=1.25,
        columnspacing=2.2, borderaxespad=0,
    )
    # Compact footer-to-plot gap without colliding with the final x-axis labels.
    fig.subplots_adjust(left=.074, right=.985, top=.926, bottom=.183,
                        hspace=.67, wspace=.24)
    stem = f'Supplementary_Figure_S8{panel}_{scope}'
    save_fig(fig, out / stem, dpi)
    plt.close(fig)
    print(f'Generated {stem} (.png, .tiff, .pdf)')


def draw_summary(audit, out, dpi):
    fig, ax = plt.subplots(figsize=(10.4, 5.1))
    scope_order = list(SCOPES)
    labels = ['MES → CMES\n(n = 19)', 'PROG → CM\n(n = 15)', 'PROG → CF\n(n = 14)']
    point_colors = ['#2868a3', '#9a5a30', '#428567']
    rng = np.random.default_rng(8128)  # visual jitter only, reproducible
    for idx, scope in enumerate(scope_order):
        df = audit[audit.scope == scope].sort_values('individual')
        signed = df.signed_peak_minus_landmark_DPT.to_numpy(float)
        jitter = rng.uniform(-.15,.15,size=len(signed))
        ax.scatter(signed, np.full(len(signed), idx)+jitter, s=39,
                   facecolor=point_colors[idx], edgecolor='white', lw=.5, alpha=.83,
                   zorder=2)
        median = float(np.median(signed))
        ax.scatter([median], [idx], s=130, marker='D', facecolor='black',
                   edgecolor='white', lw=.7, zorder=4)
        before = int((signed < 0).sum())
        after = int((signed > 0).sum())
        label_x = max(.34, float(np.max(audit.signed_peak_minus_landmark_DPT))+ .035)
        ax.text(label_x, idx, f'{before} before / {after} after', va='center',
                ha='left', fontsize=10)
    ax.axvline(0, color=LANDMARK_COLOR, lw=1.6, ls='--', zorder=1)
    ax.set_yticks(range(3), labels)
    ax.invert_yaxis()
    ax.set_xlim(min(-.37, float(audit.signed_peak_minus_landmark_DPT.min())-.05),
                max(.60, label_x + .20))
    ax.set_xlabel('Signed displacement: GDIS maximum − destination landmark (deposited DPT units)',
                  fontsize=10)
    ax.set_title('Supplementary Figure S8D  |  Signed GDIS peak offsets',
                 loc='left', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', color='#7e858c', alpha=.15)
    ax.tick_params(axis='y', length=0, labelsize=11)
    fig.text(.19, .105,
             'Each dot: one individual–scope trajectory; diamond: scope median.\n'
             'Negative offsets indicate maxima before the frozen landmark.',
             fontsize=9, color='#45515c')
    fig.subplots_adjust(left=.19, right=.97, top=.82, bottom=.29)
    save_fig(fig, out / 'Supplementary_Figure_S8D_signed_peak_offsets', dpi)
    plt.close(fig)
    print('Generated Supplementary_Figure_S8D_signed_peak_offsets (.png, .tiff, .pdf)')


def save_fig(fig, stem, dpi):
    fig.savefig(stem.with_suffix('.png'), dpi=dpi, facecolor='white')
    fig.savefig(stem.with_suffix('.tiff'), dpi=dpi, facecolor='white',
                pil_kwargs={'compression': 'tiff_lzw'})
    fig.savefig(stem.with_suffix('.pdf'), facecolor='white')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--font-scale', type=float, default=1.3,
                        help='Multiplier for S8A/B/C text sizes (default: 1.3)')
    args = parser.parse_args()
    if args.dpi < 72 or args.dpi > 1200:
        parser.error('--dpi must be between 72 and 1200')
    if not (0.8 <= args.font_scale <= 1.8):
        parser.error('--font-scale must be between 0.8 and 1.8')
    root = args.root.expanduser().resolve()
    out = (args.out or root/'results'/'reviewer1_comment5_S8_v5_compact').expanduser().resolve()
    p18 = root/'results'/'p18_external_GSE175634_primary_gdis'
    data_dir = p18/'data'
    summary_file = p18/'tables'/'02_primary_profile_summary.csv'
    if not data_dir.is_dir() or not summary_file.is_file():
        raise FileNotFoundError(f'Frozen P18 files not found under {p18}')
    summaries = pd.read_csv(summary_file, dtype={'individual': str})
    files = sorted(data_dir.glob('*_individual_*_gdis.csv'))
    if len(files) != 48:
        raise RuntimeError(f'Expected exactly 48 archived profiles; found {len(files)}')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42})
    profiles, audit_records = {}, []
    for path in files:
        df, peak, landmark, states, fractions, row = extract_one(path, summaries)
        key = row['scope'], row['individual']
        if key in profiles:
            raise ValueError(f'Duplicate scope/individual: {key}')
        profiles[key] = df, peak, landmark, states, fractions, row
        audit_records.append(row)
    for scope, (_, _, count, _) in SCOPES.items():
        actual = sum(k[0] == scope for k in profiles)
        if actual != count:
            raise ValueError(f'{scope}: expected {count}, found {actual}')
    audit = pd.DataFrame(audit_records).sort_values(['scope', 'individual'])
    out.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out/'S8_profile_peak_audit_v5.csv', index=False)
    for scope in SCOPES:
        draw_profile_set(scope, profiles, out, args.dpi, args.font_scale)
    draw_summary(audit, out, args.dpi)
    print('\n--- Verified frozen-P18 audit ---')
    for scope in SCOPES:
        a = audit[audit.scope == scope]
        signed = a.signed_peak_minus_landmark_DPT
        print(f'{scope}: n={len(a)}; median |ΔDPT|={a.abs_peak_to_landmark_DPT.median():.4f}; '
              f'median ΔDPT={signed.median():+.4f}; before={(signed<0).sum()}; after={(signed>0).sum()}')
    print('SUCCESS: 48/48 frozen profiles verified, no recalculation or filtering.')
    print('Output directory:', out)


if __name__ == '__main__':
    main()

