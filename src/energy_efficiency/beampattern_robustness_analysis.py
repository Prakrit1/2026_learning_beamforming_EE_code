"""
Quantifies the "narrow EE beam vs. broad regular-SAC beam" robustness
question directly from ALREADY-SAVED beampattern data -- no new Monte Carlo
needed, just re-reading the two existing gzips:
  - EE_beampattern_comparison_N16K3_aod0.05        (EE vs. 100W "regular" SAC)
  - EE_beampattern_across_error_models_N16K3        (EE aod=0.0/0.025/0.05 vs MMSE)

Background (see handoff): CSIT error here perturbs cos(AoD), not the angle
itself (additive_error_on_aod is hardcoded to 0 in los_channel_model.py).
At our operating angles (~1.33-1.81 rad, sin(theta)~0.97-1.0), an error
bound of `b` on cos(AoD) corresponds to roughly Delta_theta ~ b / sin(theta)
radians of equivalent angular offset. error=0.025/0.05/0.10 -> ~1.5/3/6 deg.

For every realization x model x user in both gzips, this computes:
  - peak_gain, peak_angle: where the beam's main lobe actually points
  - true_angle: the user's real AoD (dotted line in the existing plots)
  - offset_deg = peak_angle - true_angle: how far off-target the peak is
    (this ALREADY reflects one realized CSIT error draw baked into the
    precoder -- these gzips were generated at CSIT error=0.05)
  - coverage_ratio = gain(true_angle) / peak_gain: fraction of the beam's
    own peak gain that actually lands on the true user position. 1.0 =
    perfectly aimed; near 0 = beam peak sits somewhere the user isn't.
  - beamwidth_3db_deg: angular width of the main lobe above half its peak
    (interpolated at the crossing points) -- literally "how forgiving is
    this beam to being slightly off-target".

Aggregates per model (mean/median across all realizations x users) and
correlates each realization's mean coverage_ratio with that realization's
actual achieved sum_rate (already stored in the gzip), to test directly:
does a wider/more forgiving beam actually buy higher sum rate when the
precoder's aim is imperfect?
"""
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False

import gzip
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.config.config import Config
from src.config.config_plotting import PlotConfig

REPO_ROOT = Path('/home/parajuli/repos/2025_learning_beamforming_rsma_code')

EXPERIMENTS = {
    'ee_vs_regular': {
        'gzip_path': Path(REPO_ROOT, 'outputs', 'metrics', 'EE_beampattern_comparison_N16K3_aod0.05', 'beam_patterns', 'beam_patterns.gzip'),
        'models': ['regular_schemeI', 'EE_dinkelbach_aod0.05'],
        'labels': {'regular_schemeI': 'SAC (100 W)', 'EE_dinkelbach_aod0.05': 'SAC (Energy-Efficient)'},
        'colors': {'regular_schemeI': 'blue', 'EE_dinkelbach_aod0.05': 'green'},
    },
    'across_error_models': {
        'gzip_path': Path(REPO_ROOT, 'outputs', 'metrics', 'EE_beampattern_across_error_models_N16K3', 'beam_patterns', 'beam_patterns.gzip'),
        'models': ['mmse', 'EE_aod0.0', 'EE_aod0.025', 'EE_aod0.05'],
        'labels': {'mmse': 'MMSE', 'EE_aod0.0': 'EE (err=0.0)', 'EE_aod0.025': 'EE (err=0.025)', 'EE_aod0.05': 'EE (err=0.05)'},
        'colors': {'mmse': 'gold', 'EE_aod0.0': 'green', 'EE_aod0.025': 'blue', 'EE_aod0.05': 'magenta'},
    },
    'fairness_comparison': {
        'gzip_path': Path(REPO_ROOT, 'outputs', 'metrics', 'EE_beampattern_fairness_comparison_N16K3_aod0.05', 'beam_patterns', 'beam_patterns.gzip'),
        'models': ['EE_aod0.05_baseline', 'EE_aod0.05_fair0.5', 'EE_aod0.05_fair1.5', 'EE_aod0.05_fair3.0'],
        'labels': {
            'EE_aod0.05_baseline': 'EE aod0.05 (no fairness)',
            'EE_aod0.05_fair0.5': 'EE aod0.05 (fairness=0.5)',
            'EE_aod0.05_fair1.5': 'EE aod0.05 (fairness=1.5)',
            'EE_aod0.05_fair3.0': 'EE aod0.05 (fairness=3.0)',
        },
        'colors': {
            'EE_aod0.05_baseline': 'magenta',
            'EE_aod0.05_fair0.5': 'blue',
            'EE_aod0.05_fair1.5': 'green',
            'EE_aod0.05_fair3.0': 'gold',
        },
    },
}

ERROR_BOUNDS_COS_AOD = [0.025, 0.05, 0.10]


def half_max_beamwidth(angle_sweep_range, gain_curve, peak_idx):
    """Angular width (rad) of the contiguous region around peak_idx staying above half the peak gain, edges linearly interpolated."""
    peak_gain = gain_curve[peak_idx]
    half = peak_gain / 2.0

    # walk left from peak until dropping below half
    left_idx = peak_idx
    while left_idx > 0 and gain_curve[left_idx - 1] >= half:
        left_idx -= 1
    if left_idx > 0:
        # interpolate crossing between left_idx-1 and left_idx
        g0, g1 = gain_curve[left_idx - 1], gain_curve[left_idx]
        a0, a1 = angle_sweep_range[left_idx - 1], angle_sweep_range[left_idx]
        frac = (half - g0) / (g1 - g0) if g1 != g0 else 0.0
        left_edge = a0 + frac * (a1 - a0)
    else:
        left_edge = angle_sweep_range[0]

    right_idx = peak_idx
    n = len(gain_curve)
    while right_idx < n - 1 and gain_curve[right_idx + 1] >= half:
        right_idx += 1
    if right_idx < n - 1:
        g0, g1 = gain_curve[right_idx], gain_curve[right_idx + 1]
        a0, a1 = angle_sweep_range[right_idx], angle_sweep_range[right_idx + 1]
        frac = (half - g0) / (g1 - g0) if g1 != g0 else 0.0
        right_edge = a0 + frac * (a1 - a0)
    else:
        right_edge = angle_sweep_range[-1]

    return right_edge - left_edge


def analyze_experiment(name, spec):
    with gzip.open(spec['gzip_path'], 'rb') as f:
        angle_sweep_range, data = pickle.load(f)

    num_realizations = len(data)
    rows = []  # one row per (realization, model, user)

    for r_idx, entry in enumerate(data):
        true_angles = entry['user_positions'][0]  # single-satellite
        for model in spec['models']:
            gains_all_users = entry[model]['power_gains']  # (num_users, num_angles, num_sats)
            sum_rate = entry[model]['sum_rate']
            num_users = gains_all_users.shape[0]
            for user_id in range(num_users):
                gain_curve = gains_all_users[user_id, :, 0]
                true_angle = true_angles[user_id]

                peak_idx = int(np.argmax(gain_curve))
                peak_gain = gain_curve[peak_idx]
                peak_angle = angle_sweep_range[peak_idx]

                gain_at_true = np.interp(true_angle, angle_sweep_range, gain_curve)
                coverage_ratio = gain_at_true / peak_gain if peak_gain > 0 else np.nan

                offset_deg = np.degrees(peak_angle - true_angle)
                beamwidth_deg = np.degrees(half_max_beamwidth(angle_sweep_range, gain_curve, peak_idx))

                rows.append({
                    'realization': r_idx,
                    'model': model,
                    'user_id': user_id,
                    'true_angle': true_angle,
                    'peak_gain': peak_gain,
                    'offset_deg': offset_deg,
                    'coverage_ratio': coverage_ratio,
                    'beamwidth_deg': beamwidth_deg,
                    'sum_rate': sum_rate,
                    'equivalent_deg_per_0.05': np.degrees(0.05 / np.sin(true_angle)),
                })

    return rows, num_realizations


def print_summary(name, rows):
    print(f'\n=== {name} ===')
    models = sorted(set(row['model'] for row in rows))
    header = f"{'model':<20} {'mean_beamwidth_deg':>19} {'mean_|offset|_deg':>18} {'mean_coverage':>14} {'median_coverage':>16} {'frac_coverage<0.5':>18}"
    print(header)
    for model in models:
        model_rows = [r for r in rows if r['model'] == model]
        beamwidths = np.array([r['beamwidth_deg'] for r in model_rows])
        offsets = np.abs(np.array([r['offset_deg'] for r in model_rows]))
        coverages = np.array([r['coverage_ratio'] for r in model_rows])
        coverages_valid = coverages[~np.isnan(coverages)]
        frac_low_coverage = np.mean(coverages_valid < 0.5)
        print(f"{model:<20} {np.mean(beamwidths):>19.2f} {np.mean(offsets):>18.2f} {np.mean(coverages_valid):>14.3f} {np.median(coverages_valid):>16.3f} {frac_low_coverage:>18.2f}")

    # per-realization: does mean coverage_ratio (avg across users) predict sum_rate?
    print('\nPer-realization mean-coverage vs. sum_rate correlation (Pearson r):')
    for model in models:
        model_rows = [r for r in rows if r['model'] == model]
        realizations = sorted(set(r['realization'] for r in model_rows))
        mean_cov = []
        sum_rates = []
        for real_id in realizations:
            real_rows = [r for r in model_rows if r['realization'] == real_id]
            cov = np.array([r['coverage_ratio'] for r in real_rows])
            cov = cov[~np.isnan(cov)]
            if len(cov) == 0:
                continue
            mean_cov.append(np.mean(cov))
            sum_rates.append(real_rows[0]['sum_rate'])
        if len(mean_cov) > 2:
            corr = np.corrcoef(mean_cov, sum_rates)[0, 1]
        else:
            corr = float('nan')
        print(f'  {model:<20} r={corr:.3f}  (n={len(mean_cov)} realizations)')


def plot_experiment(name, spec, rows, plot_cfg):
    models = spec['models']

    fig, axes = plt.subplots(1, 2, figsize=(1.8 * plot_cfg.textwidth, 0.45 * plot_cfg.textwidth))

    # left: beamwidth distribution (bar of means + errorbars) per model
    ax = axes[0]
    means = []
    stds = []
    for model in models:
        model_rows = [r for r in rows if r['model'] == model]
        bw = np.array([r['beamwidth_deg'] for r in model_rows])
        means.append(np.mean(bw))
        stds.append(np.std(bw))
    x = np.arange(len(models))
    ax.bar(x, means, yerr=stds, color=[plot_cfg.cp2[spec['colors'][m]] for m in models], capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([spec['labels'][m] for m in models], rotation=20, ha='right', fontsize=7)
    ax.set_ylabel('-3dB beamwidth [deg]')
    ax.set_title('Main-lobe width')

    # right: per-realization mean coverage_ratio vs sum_rate scatter
    ax = axes[1]
    for model in models:
        model_rows = [r for r in rows if r['model'] == model]
        realizations = sorted(set(r['realization'] for r in model_rows))
        mean_cov = []
        sum_rates = []
        for real_id in realizations:
            real_rows = [r for r in model_rows if r['realization'] == real_id]
            cov = np.array([r['coverage_ratio'] for r in real_rows])
            cov = cov[~np.isnan(cov)]
            if len(cov) == 0:
                continue
            mean_cov.append(np.mean(cov))
            sum_rates.append(real_rows[0]['sum_rate'])
        ax.scatter(mean_cov, sum_rates, label=spec['labels'][model], color=plot_cfg.cp2[spec['colors'][model]], s=20)
    ax.set_xlabel('Mean coverage ratio (gain at true angle / peak gain)')
    ax.set_ylabel('Sum rate [bps/Hz]')
    ax.set_title('Coverage ratio vs. achieved sum rate')
    ax.legend(fontsize=7)

    fig.tight_layout()
    pdf_path = Path(plot_cfg.plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)
    out = Path(pdf_path, f'beampattern_robustness_{name}.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=300, transparent=True)
    print(f'Saved: {out}')


if __name__ == '__main__':
    cfg = Config()
    cfg.show_plots = False
    plot_cfg = PlotConfig()
    matplotlib.rcParams['text.usetex'] = False  # PlotConfig() resets this on construction

    for name, spec in EXPERIMENTS.items():
        rows, num_realizations = analyze_experiment(name, spec)
        print_summary(name, rows)
        plot_experiment(name, spec, rows, plot_cfg)

    print('\nFor reference, equivalent angular offset from a cos(AoD) error bound, at our operating angles (~1.33-1.81 rad):')
    for b in ERROR_BOUNDS_COS_AOD:
        lo = np.degrees(b / np.sin(1.81))
        hi = np.degrees(b / np.sin(1.33))
        print(f'  error={b}: ~{lo:.2f}-{hi:.2f} deg')
