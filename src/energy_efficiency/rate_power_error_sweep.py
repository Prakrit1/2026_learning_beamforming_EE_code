"""
Rate AND power, measured TOGETHER from the same Monte Carlo samples, swept
across CSIT error -- for the three raw-power-fixed Dinkelbach checkpoints
trained at Delta-epsilon_aod=0.0, 0.025, and 0.05 (jobs 147505/148538/147506,
see handoff_prompt_EE_evaluation.txt's "Finding 3"), completing the paper's
own three-point (0.0/0.025/0.05) training match.

Every other evaluation script this project has only ever measured ONE of
rate or power at a time: my_evaluation.py sweeps error and measures rate
only; tx_power_distribution.py measures power only, at a single fixed
zero-error scenario. Neither can answer "how do rate and power trade off
against each other as CSIT error increases" for a single policy, because
that requires both metrics from the SAME simulated samples at EACH error
level. This script does that directly (one Monte Carlo loop, both metrics
computed from the same w_precoder per sample), producing:
1. rate-vs-error and power-vs-error (two panels, so no dual-axis chart)
2. the actual rate-vs-power trajectory, with error level as the implicit
   parameter along each checkpoint's curve

Uses the SAME error_sweep_range convention as my_evaluation.py (0 to 0.5,
11 points) so the rate-vs-error panel is directly comparable to every
other rate curve produced this week.
"""
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False

import gzip
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config.config import Config
from src.config.config_plotting import PlotConfig
from src.data.calc_sum_rate import calc_sum_rate
from src.data.calc_tx_power_distribution import calc_tx_power_distribution
from src.data.satellite_manager import SatelliteManager
from src.data.user_manager import UserManager
from src.utils.get_precoding import get_precoding_learned_clip_only
from src.utils.load_model import load_model
from src.utils.update_sim import update_sim

# Narrowed 2026-07-14 at user's request to match the reference paper's
# actual evaluation range (Fig. 3/4 only sweep up to Delta-epsilon_aod~
# 0.10-0.12), consistent with the same change in my_evaluation.py.
error_sweep_range = np.linspace(0, 0.10, 11)
monte_carlo_iterations = 10000


def get_best_model_path(trained_models_path, training_name):
    """Session-aware checkpoint selection -- see my_evaluation.py's identical function for the full rationale."""
    import os

    base_path = Path(trained_models_path, training_name, 'base')
    checkpoints = [p for p in base_path.iterdir() if p.is_dir() and 'full_snap' in p.name]
    if not checkpoints:
        raise FileNotFoundError(f'No checkpoints found under {base_path}')

    checkpoints_by_time = sorted(checkpoints, key=lambda p: os.path.getmtime(p))

    max_gap_minutes = 90
    max_gap_seconds = max_gap_minutes * 60

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


def run_rate_power_sweep(cfg, model_path, label):
    precoder_network, norm_factors = load_model(model_path)
    if norm_factors != {}:
        cfg.config_learner.get_state_args['norm_state'] = True

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
            w_precoder = get_precoding_learned_clip_only(cfg, user_manager, satellite_manager, norm_factors, precoder_network)
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

    return mean_rate, std_rate, mean_power, std_power


if __name__ == '__main__':
    models_to_check = [
        ('EE_dinkelbach_adaptive_lwin5000_N16K3_eta0.6_rawpow', 'error=0.0 (training)'),
        ('EE_dinkelbach_adaptive_aod0.025_lwin5000_N16K3_eta0.6_rawpow', 'error=0.025 (training)'),
        ('EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_eta0.6_rawpow', 'error=0.05 (training)'),
    ]
    colors = ['#307b3b', '#c9a227', '#1baf7a']  # green, gold, aqua -- matches Dinkelbach-family convention elsewhere

    results = {}
    power_budget = None
    for training_name, label in models_to_check:
        cfg = Config()
        cfg.show_plots = False
        power_budget = cfg.power_constraint_watt
        cfg.config_learner.training_name = training_name
        model_path = get_best_model_path(cfg.trained_models_path, training_name)
        mean_rate, std_rate, mean_power, std_power = run_rate_power_sweep(cfg, model_path, label)
        results[label] = {
            'mean_rate': mean_rate, 'std_rate': std_rate,
            'mean_power': mean_power, 'std_power': std_power,
        }

    # persist raw results so the plot can be regenerated without rerunning Monte Carlo
    out_path = Path(Config().output_metrics_path, 'EE_dinkelbach_adaptive_rate_power_sweep')
    out_path.mkdir(parents=True, exist_ok=True)
    with gzip.open(Path(out_path, 'rate_power_error_sweep.gzip'), 'wb') as file:
        pickle.dump({'error_sweep_range': error_sweep_range, 'power_budget': power_budget, 'results': results}, file=file)
    print(f"Saved: {Path(out_path, 'rate_power_error_sweep.gzip')}")

    plot_cfg = PlotConfig()
    matplotlib.rcParams['text.usetex'] = False  # PlotConfig() resets this on construction

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # ---- Panel 1: rate vs error ----
    ax = axes[0]
    for (label, data), color in zip(results.items(), colors):
        ax.errorbar(error_sweep_range, data['mean_rate'], yerr=data['std_rate'], marker='o',
                    color=color, ecolor=color, elinewidth=1, capsize=3, linewidth=1.5, markersize=5, label=label)
    ax.set_xlabel('Error Bound')
    ax.set_ylabel('Rate R [bps/Hz]')
    ax.set_title('Rate vs. CSIT error', fontsize=11)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc='best')

    # ---- Panel 2: power vs error ----
    ax2 = axes[1]
    for (label, data), color in zip(results.items(), colors):
        ax2.errorbar(error_sweep_range, data['mean_power'], yerr=data['std_power'], marker='o',
                     color=color, ecolor=color, elinewidth=1, capsize=3, linewidth=1.5, markersize=5, label=label)
    ax2.axhline(power_budget, color='#d03b3b', linestyle='--', linewidth=1.5, label='Power budget')
    ax2.set_xlabel('Error Bound')
    ax2.set_ylabel('Mean total transmit power [W]')
    ax2.set_title('Power vs. CSIT error', fontsize=11)
    ax2.grid(True, alpha=0.25, linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=8, loc='best')

    # ---- Panel 3: rate vs power, error as the implicit trajectory parameter ----
    ax3 = axes[2]
    for (label, data), color in zip(results.items(), colors):
        ax3.plot(data['mean_power'], data['mean_rate'], color=color, linewidth=1, zorder=1)
        ax3.scatter(data['mean_power'], data['mean_rate'], color=color, s=60, zorder=5,
                    edgecolor='white', linewidth=0.6, label=label)
        for x, y, e in zip(data['mean_power'], data['mean_rate'], error_sweep_range):
            ax3.annotate(f'{e:.2f}', (x, y), textcoords='offset points', xytext=(5, 4), fontsize=6.5)
    ax3.axvline(power_budget, color='#d03b3b', linestyle='--', linewidth=1.2, label='Power budget')
    ax3.set_xlabel('Mean total transmit power [W]')
    ax3.set_ylabel('Rate R [bps/Hz]')
    ax3.set_title('Rate vs. power (labels = error level)', fontsize=11)
    ax3.grid(True, alpha=0.25, linewidth=0.5)
    ax3.set_axisbelow(True)
    ax3.legend(fontsize=8, loc='best')

    fig.suptitle('Raw-power-fixed Dinkelbach: error=0.0 vs 0.025 vs 0.05 training', fontsize=13, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    pdf_path = Path(plot_cfg.plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)
    out = Path(pdf_path, 'rate_power_error_sweep.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=300, transparent=True)
    print(f'Saved: {out}')
    plt.close(fig)
