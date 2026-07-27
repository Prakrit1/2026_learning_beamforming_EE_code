"""
Rate, power, AND Jain's fairness, measured TOGETHER from the same Monte
Carlo samples, swept across CSIT error -- for the aod=0.05 baseline
(no fairness term) vs. all three trial fairness weights (0.5/1.5/3.0,
jobs 152699/152700/152701 -- see handoff_prompt_EE_evaluation.txt / this
session's fairness-reward work). Answers whether the fairness benefit
keeps growing with weight or starts trading rate away.

Directly tests what the fairness term was added for: does it raise
calc_jain_fairness (the SAME function now added to the training reward,
reused here unchanged) at a given CSIT error, and at what rate/power cost?
Same pattern as rate_power_error_sweep.py (one MC loop, all metrics from
the same w_precoder per sample) with a third metric appended.
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

from src.config.config import Config
from src.config.config_plotting import PlotConfig
from src.data.calc_sum_rate import calc_sum_rate
from src.data.calc_fairness import calc_jain_fairness
from src.data.calc_tx_power_distribution import calc_tx_power_distribution
from src.data.satellite_manager import SatelliteManager
from src.data.user_manager import UserManager
from src.utils.get_precoding import get_precoding_learned_clip_only
from src.utils.load_model import load_model
from src.utils.update_sim import update_sim

error_sweep_range = np.linspace(0, 0.10, 11)
monte_carlo_iterations = 10000


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


def run_sweep(cfg, model_path, label):
    precoder_network, norm_factors = load_model(model_path)
    if norm_factors != {}:
        cfg.config_learner.get_state_args['norm_state'] = True

    satellite_manager = SatelliteManager(config=cfg)
    user_manager = UserManager(config=cfg)

    error_param = 'additive_error_on_cosine_of_aod'
    initial_error_config = cfg.config_error_model.error_rng_parametrizations[error_param]['args'].copy()

    mean_rate = np.zeros(len(error_sweep_range))
    mean_power = np.zeros(len(error_sweep_range))
    mean_fairness = np.zeros(len(error_sweep_range))
    std_rate = np.zeros(len(error_sweep_range))
    std_power = np.zeros(len(error_sweep_range))
    std_fairness = np.zeros(len(error_sweep_range))

    for error_idx, error_value in enumerate(error_sweep_range):
        cfg.config_error_model.error_rng_parametrizations[error_param]['args']['low'] = -error_value
        cfg.config_error_model.error_rng_parametrizations[error_param]['args']['high'] = error_value

        rate_samples = np.zeros(monte_carlo_iterations)
        power_samples = np.zeros(monte_carlo_iterations)
        fairness_samples = np.zeros(monte_carlo_iterations)

        for iter_idx in range(monte_carlo_iterations):
            update_sim(cfg, satellite_manager, user_manager)
            w_precoder = get_precoding_learned_clip_only(cfg, user_manager, satellite_manager, norm_factors, precoder_network)
            rate_samples[iter_idx] = calc_sum_rate(
                channel_state=satellite_manager.channel_state_information,
                w_precoder=w_precoder,
                noise_power_watt=cfg.noise_power_watt,
            )
            power_samples[iter_idx] = calc_tx_power_distribution(w_precoder=w_precoder).sum()
            fairness_samples[iter_idx] = calc_jain_fairness(
                channel_state=satellite_manager.channel_state_information,
                w_precoder=w_precoder,
                noise_power_watt=cfg.noise_power_watt,
            )

        mean_rate[error_idx] = rate_samples.mean()
        std_rate[error_idx] = rate_samples.std()
        mean_power[error_idx] = power_samples.mean()
        std_power[error_idx] = power_samples.std()
        mean_fairness[error_idx] = fairness_samples.mean()
        std_fairness[error_idx] = fairness_samples.std()
        print(f'[{label}] error={error_value:.2f}: rate={mean_rate[error_idx]:.4f} bps/Hz, '
              f'power={mean_power[error_idx]:.2f} W ({100 * mean_power[error_idx] / cfg.power_constraint_watt:.1f}% of budget), '
              f'jain_fairness={mean_fairness[error_idx]:.4f}')

    cfg.config_error_model.error_rng_parametrizations[error_param]['args'] = initial_error_config

    return mean_rate, std_rate, mean_power, std_power, mean_fairness, std_fairness


if __name__ == '__main__':
    models_to_check = [
        ('EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_eta0.6_rawpow', 'baseline (no fairness)'),
        ('EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_eta0.6_rawpow_fair0.5', 'fairness=0.5'),
        ('EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_eta0.6_rawpow_fair1.5', 'fairness=1.5'),
        ('EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_eta0.6_rawpow_fair3.0', 'fairness=3.0'),
    ]
    colors = ['#1baf7a', '#f4a261', '#e76f51', '#9d0208']  # aqua (baseline), then a light->dark ramp for increasing fairness weight

    results = {}
    power_budget = None
    for training_name, label in models_to_check:
        cfg = Config()
        cfg.show_plots = False
        power_budget = cfg.power_constraint_watt
        cfg.config_learner.training_name = training_name
        model_path = get_best_model_path(cfg.trained_models_path, training_name)
        print(f'[{label}] -> {model_path}')
        mean_rate, std_rate, mean_power, std_power, mean_fairness, std_fairness = run_sweep(cfg, model_path, label)
        results[label] = {
            'mean_rate': mean_rate, 'std_rate': std_rate,
            'mean_power': mean_power, 'std_power': std_power,
            'mean_fairness': mean_fairness, 'std_fairness': std_fairness,
        }

    out_path = Path(Config().output_metrics_path, 'EE_fairness_evaluation_aod0.05')
    out_path.mkdir(parents=True, exist_ok=True)
    with gzip.open(Path(out_path, 'fairness_evaluation.gzip'), 'wb') as file:
        pickle.dump({'error_sweep_range': error_sweep_range, 'power_budget': power_budget, 'results': results}, file=file)
    print(f"Saved: {Path(out_path, 'fairness_evaluation.gzip')}")

    plot_cfg = PlotConfig()
    matplotlib.rcParams['text.usetex'] = False  # PlotConfig() resets this on construction

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

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

    ax3 = axes[2]
    for (label, data), color in zip(results.items(), colors):
        ax3.errorbar(error_sweep_range, data['mean_fairness'], yerr=data['std_fairness'], marker='o',
                     color=color, ecolor=color, elinewidth=1, capsize=3, linewidth=1.5, markersize=5, label=label)
    ax3.set_xlabel('Error Bound')
    ax3.set_ylabel("Jain's fairness index")
    ax3.set_ylim(1/3, 1.0)  # K=3 users: Jain's index ranges (1/3, 1]
    ax3.set_title('Fairness vs. CSIT error', fontsize=11)
    ax3.grid(True, alpha=0.25, linewidth=0.5)
    ax3.set_axisbelow(True)
    ax3.legend(fontsize=8, loc='best')

    fig.suptitle('aod=0.05 Dinkelbach: baseline vs. fairness={0.5,1.5,3.0} reward term', fontsize=13, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    pdf_path = Path(plot_cfg.plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)
    out = Path(pdf_path, 'fairness_evaluation_aod0.05_fair_all.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=300, transparent=True)
    print(f'Saved: {out}')
    plt.close(fig)
