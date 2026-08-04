"""
Rate AND power (same Monte Carlo samples, same convention as
rate_power_detlam.py) for the four ema0.3+warmstart detlam retrains
(2026-07-30 session, jobs 155291-94) -- aod=0.0 and aod=0.05, with and
without fairness=1.5. All nadir, 3GPP Set-1 defaults (30 dBi / 75 W).
aod=0.025 excluded (never retried).

MMSE and the pre-fix (rollout-lambda) SAC nadir curve are reused from
rate_power_3gpp_pilots.gzip. The four ORIGINAL (pre-ema) detlam curves are
reused from rate_power_detlam.gzip (produced by rate_power_detlam.py in the
prior session) -- none of these three are re-simulated, only the four new
checkpoints are.
"""
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False

import gzip
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config.config_plotting import PlotConfig
from src.data.calc_sum_rate import calc_sum_rate
from src.data.calc_tx_power_distribution import calc_tx_power_distribution
from src.data.satellite_manager import SatelliteManager
from src.data.user_manager import UserManager
from src.utils.get_precoding import get_precoding_learned_clip_only
from src.utils.load_model import load_model
from src.utils.update_sim import update_sim

error_sweep_range = np.linspace(0, 0.10, 11)
monte_carlo_iterations = 10000

PREFIX_GZIP_REL = ('EE_3gpp_pilots_rate_power_sweep', 'rate_power_3gpp_pilots.gzip')
OLD_DETLAM_GZIP_REL = ('EE_detlam_rate_power_sweep', 'rate_power_detlam.gzip')

CHECKPOINTS = [
    {
        'key': 'ema_ws_aod0.0',
        'label': 'SAC detlam aod0.0 (ema+ws)',
        'training_name': 'EE_dinkelbach_adaptive_detlam512_ema0.3_N16K3_satg30_p75_eta0.6_rawpow_warmstart',
    },
    {
        'key': 'ema_ws_aod0.05',
        'label': 'SAC detlam aod0.05 (ema+ws)',
        'training_name': 'EE_dinkelbach_adaptive_aod0.05_detlam512_ema0.3_N16K3_satg30_p75_eta0.6_rawpow_warmstart',
    },
    {
        'key': 'ema_ws_aod0.0_fair1.5',
        'label': 'SAC detlam aod0.0, fair1.5 (ema+ws)',
        'training_name': 'EE_dinkelbach_adaptive_detlam512_ema0.3_N16K3_satg30_p75_eta0.6_rawpow_fair1.5_warmstart',
    },
    {
        'key': 'ema_ws_aod0.05_fair1.5',
        'label': 'SAC detlam aod0.05, fair1.5 (ema+ws)',
        'training_name': 'EE_dinkelbach_adaptive_aod0.05_detlam512_ema0.3_N16K3_satg30_p75_eta0.6_rawpow_fair1.5_warmstart',
    },
]

OLD_DETLAM_KEYS = ['detlam_aod0.0', 'detlam_aod0.05', 'detlam_aod0.0_fair1.5', 'detlam_aod0.05_fair1.5']

COLORS = {
    'prefix_nadir': '#888888',
    'mmse_nadir': '#caa023',
    'detlam_aod0.0': '#a8d5b0',
    'detlam_aod0.05': '#a9bde0',
    'detlam_aod0.0_fair1.5': '#f9d38a',
    'detlam_aod0.05_fair1.5': '#c4b8dc',
    'ema_ws_aod0.0': '#307b3b',
    'ema_ws_aod0.05': '#254796',
    'ema_ws_aod0.0_fair1.5': '#f7a600',
    'ema_ws_aod0.05_fair1.5': '#3b296a',
}


def make_config():
    for var in ['EE_SAT_GAIN_DBI', 'EE_POWER_BUDGET_WATT', 'EE_TARGET_ELEVATION_DEG']:
        os.environ.pop(var, None)
    from src.config.config import Config
    cfg = Config()
    cfg.show_plots = False
    return cfg


def get_best_model_path(trained_models_path, training_name):
    """Session-aware checkpoint selection -- see my_evaluation.py's identical function for the full rationale."""
    base_path = Path(trained_models_path, training_name, 'base')
    checkpoints = [p for p in base_path.iterdir() if p.is_dir() and 'full_snap' in p.name]
    if not checkpoints:
        raise FileNotFoundError(f'No checkpoints found under {base_path}')

    checkpoints_by_time = sorted(checkpoints, key=lambda p: os.path.getmtime(p))

    max_gap_seconds = 90 * 60
    session_start_idx = len(checkpoints_by_time) - 1
    for i in range(len(checkpoints_by_time) - 1, 0, -1):
        gap = os.path.getmtime(checkpoints_by_time[i]) - os.path.getmtime(checkpoints_by_time[i - 1])
        if gap > max_gap_seconds:
            session_start_idx = i
            break
    else:
        session_start_idx = 0

    same_session_checkpoints = checkpoints_by_time[session_start_idx:]
    best = sorted(same_session_checkpoints, key=lambda p: float(p.name.split('_')[-1]))[-1]
    return best


def run_rate_power_sweep(cfg, label, get_precoder_func):
    satellite_manager = SatelliteManager(config=cfg)
    user_manager = UserManager(config=cfg)

    error_param = 'additive_error_on_cosine_of_aod'
    initial_error_config = cfg.config_error_model.error_rng_parametrizations[error_param]['args'].copy()

    mean_rate = np.zeros(len(error_sweep_range))
    mean_power = np.zeros(len(error_sweep_range))
    std_rate = np.zeros(len(error_sweep_range))
    std_power = np.zeros(len(error_sweep_range))

    for error_idx, error_value in enumerate(error_sweep_range):
        cfg.config_error_model.error_rng_parametrizations[error_param]['args']['low'] = -error_value
        cfg.config_error_model.error_rng_parametrizations[error_param]['args']['high'] = error_value

        rate_samples = np.zeros(monte_carlo_iterations)
        power_samples = np.zeros(monte_carlo_iterations)

        for iter_idx in range(monte_carlo_iterations):
            update_sim(cfg, satellite_manager, user_manager)
            w_precoder = get_precoder_func(cfg, user_manager, satellite_manager)
            rate_samples[iter_idx] = calc_sum_rate(
                channel_state=satellite_manager.channel_state_information,
                w_precoder=w_precoder,
                noise_power_watt=cfg.noise_power_watt,
            )
            power_samples[iter_idx] = calc_tx_power_distribution(w_precoder=w_precoder).sum()

        mean_rate[error_idx] = rate_samples.mean()
        std_rate[error_idx] = rate_samples.std()
        mean_power[error_idx] = power_samples.mean()
        std_power[error_idx] = power_samples.std()
        print(f'[{label}] error={error_value:.2f}: rate={mean_rate[error_idx]:.4f} bps/Hz, '
              f'power={mean_power[error_idx]:.2f} W ({100 * mean_power[error_idx] / cfg.power_constraint_watt:.1f}% of budget)')

    cfg.config_error_model.error_rng_parametrizations[error_param]['args'] = initial_error_config

    return {
        'power_budget': cfg.power_constraint_watt,
        'mean_rate': mean_rate, 'std_rate': std_rate,
        'mean_power': mean_power, 'std_power': std_power,
    }


if __name__ == '__main__':
    cfg0 = make_config()

    prefix_gzip = Path(cfg0.output_metrics_path, *PREFIX_GZIP_REL)
    with gzip.open(prefix_gzip, 'rb') as file:
        prefix_data = pickle.load(file)
    if not np.allclose(prefix_data['error_sweep_range'], error_sweep_range):
        raise ValueError(f'error grid mismatch between {prefix_gzip} and this sweep')

    old_detlam_gzip = Path(cfg0.output_metrics_path, *OLD_DETLAM_GZIP_REL)
    with gzip.open(old_detlam_gzip, 'rb') as file:
        old_detlam_data = pickle.load(file)
    if not np.allclose(old_detlam_data['error_sweep_range'], error_sweep_range):
        raise ValueError(f'error grid mismatch between {old_detlam_gzip} and this sweep')

    results = {
        'prefix_nadir': dict(prefix_data['results']['3gpp_nadir'], label='SAC pre-fix (rollout-lambda, aod0.05)'),
        'mmse_nadir': dict(prefix_data['results']['mmse_nadir'], label='MMSE nadir'),
    }
    for key in OLD_DETLAM_KEYS:
        results[key] = dict(old_detlam_data['results'][key])
        results[key]['label'] = results[key]['label'] + ' (pre-ema)'
    print(f'[reuse] loaded prefix_nadir + mmse_nadir from {prefix_gzip} (not re-simulated)')
    print(f'[reuse] loaded 4 pre-ema detlam checkpoints from {old_detlam_gzip} (not re-simulated)')

    out_path = Path(cfg0.output_metrics_path, 'EE_detlam_ema_warmstart_rate_power_sweep')
    out_path.mkdir(parents=True, exist_ok=True)
    gzip_path = Path(out_path, 'rate_power_detlam_ema_warmstart.gzip')

    for ckpt in CHECKPOINTS:
        cfg = make_config()
        cfg.config_learner.training_name = ckpt['training_name']
        model_path = get_best_model_path(cfg.trained_models_path, ckpt['training_name'])
        print(f"[{ckpt['key']}] checkpoint: {model_path}")

        precoder_network, norm_factors = load_model(model_path)
        if norm_factors != {}:
            cfg.config_learner.get_state_args['norm_state'] = True
        results[ckpt['key']] = run_rate_power_sweep(
            cfg, ckpt['label'],
            lambda c, um, sm: get_precoding_learned_clip_only(c, um, sm, norm_factors, precoder_network),
        )
        results[ckpt['key']]['label'] = ckpt['label']

    with gzip.open(gzip_path, 'wb') as file:
        pickle.dump({'error_sweep_range': error_sweep_range, 'results': results}, file=file)
    print(f'Saved: {gzip_path}')

    # ---- power savings summary -------------------------------------------
    print('\n=== Power savings summary (mean over full error sweep) ===')
    budget = results['mmse_nadir']['power_budget']
    prefix_mean_power = results['prefix_nadir']['mean_power'].mean()
    for key, data in results.items():
        if key in ('mmse_nadir', 'prefix_nadir'):
            continue
        mp = data['mean_power'].mean()
        pct_of_budget = 100 * mp / budget
        savings_vs_budget = 100 * (1 - mp / budget)
        savings_vs_prefix = 100 * (1 - mp / prefix_mean_power)
        print(f"{data['label']:38s}  mean power={mp:6.2f} W  "
              f"({pct_of_budget:5.1f}% of {budget:.0f} W budget, "
              f"{savings_vs_budget:5.1f}% saved vs budget, "
              f"{savings_vs_prefix:+5.1f}% vs pre-fix's {prefix_mean_power:.1f} W)")

    # ---- old-vs-new (ema+warmstart) direct comparison ----------------------
    print('\n=== Pre-ema vs. ema+warmstart: mean rate and power (full sweep mean) ===')
    pairs = [
        ('detlam_aod0.0', 'ema_ws_aod0.0'),
        ('detlam_aod0.05', 'ema_ws_aod0.05'),
        ('detlam_aod0.0_fair1.5', 'ema_ws_aod0.0_fair1.5'),
        ('detlam_aod0.05_fair1.5', 'ema_ws_aod0.05_fair1.5'),
    ]
    for old_key, new_key in pairs:
        old_rate = results[old_key]['mean_rate'].mean()
        new_rate = results[new_key]['mean_rate'].mean()
        old_power = results[old_key]['mean_power'].mean()
        new_power = results[new_key]['mean_power'].mean()
        print(f'{old_key:24s}  rate {old_rate:.3f} -> {new_rate:.3f} ({100*(new_rate/old_rate-1):+.1f}%)  '
              f'power {old_power:.2f} W -> {new_power:.2f} W ({100*(new_power/old_power-1):+.1f}%)')

    # ---- plot ---------------------------------------------------------------
    plot_cfg = PlotConfig()
    matplotlib.rcParams['text.usetex'] = False  # PlotConfig() resets this on construction

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    plot_order = ['prefix_nadir', 'mmse_nadir'] + OLD_DETLAM_KEYS + [c['key'] for c in CHECKPOINTS]

    ax = axes[0]
    for key in plot_order:
        data = results[key]
        ax.errorbar(error_sweep_range, data['mean_rate'], yerr=data['std_rate'], marker='o',
                    color=COLORS[key], ecolor=COLORS[key], elinewidth=1, capsize=3, linewidth=1.5,
                    markersize=5, label=data['label'])
    ax.set_xlabel('Error Bound')
    ax.set_ylabel('Rate R [bps/Hz]')
    ax.set_title('Rate vs. CSIT error', fontsize=11)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=5.5, loc='best')

    ax2 = axes[1]
    for key in plot_order:
        data = results[key]
        power_pct = 100 * data['mean_power'] / data['power_budget']
        power_pct_std = 100 * data['std_power'] / data['power_budget']
        ax2.errorbar(error_sweep_range, power_pct, yerr=power_pct_std, marker='o',
                     color=COLORS[key], ecolor=COLORS[key], elinewidth=1, capsize=3, linewidth=1.5,
                     markersize=5, label=data['label'])
    ax2.axhline(100, color='#d03b3b', linestyle='--', linewidth=1.5, label='Power budget')
    ax2.set_xlabel('Error Bound')
    ax2.set_ylabel('Mean transmit power [% of own budget]')
    ax2.set_title('Power vs. CSIT error', fontsize=11)
    ax2.grid(True, alpha=0.25, linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=5.5, loc='best')

    ax3 = axes[2]
    for key in plot_order:
        data = results[key]
        power_pct = 100 * data['mean_power'] / data['power_budget']
        ax3.plot(power_pct, data['mean_rate'], color=COLORS[key], linewidth=1, zorder=1)
        ax3.scatter(power_pct, data['mean_rate'], color=COLORS[key], s=60, zorder=5,
                    edgecolor='white', linewidth=0.6, label=data['label'])
        for x, y, e in zip(power_pct, data['mean_rate'], error_sweep_range):
            ax3.annotate(f'{e:.2f}', (x, y), textcoords='offset points', xytext=(5, 4), fontsize=6.5)
    ax3.axvline(100, color='#d03b3b', linestyle='--', linewidth=1.2, label='Power budget')
    ax3.set_xlabel('Mean transmit power [% of own budget]')
    ax3.set_ylabel('Rate R [bps/Hz]')
    ax3.set_title('Rate vs. power (labels = error level)', fontsize=11)
    ax3.grid(True, alpha=0.25, linewidth=0.5)
    ax3.set_axisbelow(True)
    ax3.legend(fontsize=5.5, loc='best')

    fig.suptitle('Ema0.3+warmstart detlam retrains vs. pre-ema detlam, pre-fix SAC and MMSE, nadir 30 dBi / 75 W', fontsize=12, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    pdf_path = Path(plot_cfg.plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)
    out = Path(pdf_path, 'rate_power_detlam_ema_warmstart.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=300, transparent=True)
    print(f'Saved: {out}')
    plt.close(fig)
