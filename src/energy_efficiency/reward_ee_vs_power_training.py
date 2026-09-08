"""Per-episode reward AND EE vs power used, from the REAL training logs.

Every training episode logged (in training_error_learned_full.gzip):
  - mean_reward_per_episode        -- the reward the model actually produced
  - episode_mean_power_dinkelbach  -- mean power that episode, NORMALIZED by
                                      the budget (x budget); *75 = DC-draw W
  - episode_mean_rate_dinkelbach   -- mean sum rate that episode
  - lambda_ee_per_episode          -- adaptive lambda at that episode

We plot each episode as a real (power, reward) and (power, EE) point and bin-
average over power, so power is on the x-axis. Nothing is recomputed from a
forced-power sweep -- these are the values the policy generated during
training.

IMPORTANT (see EE_sac.py:510-511): the Dinkelbach reward R - lambda*P trends
to 0 as lambda converges, so the reward-vs-power panel collapses toward 0 near
the converged power -- it reflects the training/lambda trajectory, not a
power-optimization landscape. EE = rate/power (the checkpoint score) is the
meaningful power-vs-performance curve; both panels are shown so this is visible.

Run:  python3 src/energy_efficiency/reward_ee_vs_power_training.py
Saves reports/figures/{pdf,jpg,png}/reward_ee_vs_power_training.*
"""
import gzip
import pickle
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt

from src.config.config import Config
from src.config.config_plotting import PlotConfig

TRAINING_NAME = 'EE_dinkelbach_adaptive_lwin5000_N16K3_satg30_p75_eta0.6_rawpow'
N_BINS = 18


def binned_mean(x, y, n_bins):
    edges = np.linspace(np.nanmin(x), np.nanmax(x), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = (x >= edges[b]) & (x < edges[b + 1] if b < n_bins - 1 else x <= edges[b + 1])
        if np.any(sel):
            means[b] = np.nanmean(y[sel])
    ok = np.isfinite(means)
    return centers[ok], means[ok]


if __name__ == '__main__':
    cfg = Config()
    cfg.show_plots = False
    plot_cfg = PlotConfig()
    matplotlib.rcParams['text.usetex'] = False  # compute nodes have no latex

    metrics_path = Path(cfg.output_metrics_path, TRAINING_NAME, 'base', 'training_error_learned_full.gzip')
    if not metrics_path.exists():
        raise FileNotFoundError(f'{metrics_path} not found -- this is the training metrics log.')
    with gzip.open(metrics_path, 'rb') as file:
        m = pickle.load(file)

    reward = np.asarray(m['mean_reward_per_episode'], dtype=float)
    power_norm = np.asarray(m['episode_mean_power_dinkelbach'], dtype=float)
    rate = np.asarray(m['episode_mean_rate_dinkelbach'], dtype=float)
    lam = np.asarray(m['lambda_ee_per_episode'], dtype=float)

    # keep only episodes where the Dinkelbach metrics were actually recorded
    valid = np.isfinite(power_norm) & np.isfinite(reward) & np.isfinite(rate) & (power_norm > 0)
    reward, power_norm, rate = reward[valid], power_norm[valid], rate[valid]
    episode_idx = np.arange(len(m['mean_reward_per_episode']))[valid]

    # normalized power (x budget) -> total consumed DC watts; EE in bits/s/Hz/W
    power_watt = power_norm * cfg.power_constraint_watt
    ee = rate / power_watt

    print(f'episodes with data: {valid.sum()} / {valid.size}')
    print(f'power (DC) range: {power_watt.min():.1f} - {power_watt.max():.1f} W')
    print(f'reward range: {reward.min():.4f} - {reward.max():.4f} (last {reward[-1]:.4f})')
    print(f'EE range: {ee.min():.5f} - {ee.max():.5f}; EE max at power={power_watt[int(np.argmax(ee))]:.1f} W')

    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.85
    fig, (ax_r, ax_e) = plt.subplots(2, 1, sharex=True, figsize=(plot_width, plot_height))

    # scatter colored by training episode (shows the trajectory / lambda drift)
    sc = ax_r.scatter(power_watt, reward, c=episode_idx, cmap='viridis', s=10, alpha=0.5, zorder=2)
    bx, by = binned_mean(power_watt, reward, N_BINS)
    ax_r.plot(bx, by, color='black', linewidth=1.8, zorder=3, label='binned mean')
    ax_r.axhline(0.0, color='0.6', linewidth=0.8, linestyle=':')
    ax_r.set_ylabel('Reward  (per episode)', fontsize=12)
    ax_r.grid(True, alpha=0.2, linewidth=0.5)
    ax_r.legend(fontsize=9, frameon=False, loc='best')

    ax_e.scatter(power_watt, ee, c=episode_idx, cmap='viridis', s=10, alpha=0.5, zorder=2)
    ex, ey = binned_mean(power_watt, ee, N_BINS)
    ax_e.plot(ex, ey, color='black', linewidth=1.8, zorder=3, label='binned mean')
    ax_e.set_ylabel('EE = rate / power\n[bits/s/Hz/W]', fontsize=12)
    ax_e.set_xlabel('Total consumed power [W]  (= normalized power x 75)', fontsize=12)
    ax_e.grid(True, alpha=0.2, linewidth=0.5)
    ax_e.legend(fontsize=9, frameon=False, loc='best')

    cbar = fig.colorbar(sc, ax=[ax_r, ax_e], location='right', fraction=0.04, pad=0.02)
    cbar.set_label('training episode', fontsize=10)

    for subdir, dpi, transparent in [('pdf', 300, True), ('jpg', 200, False), ('png', 200, True)]:
        target = Path(plot_cfg.plots_parent_path, subdir)
        target.mkdir(parents=True, exist_ok=True)
        out = Path(target, f'reward_ee_vs_power_training.{subdir}')
        fig.savefig(out, bbox_inches='tight', dpi=dpi, transparent=transparent)
        print(f'Saved: {out}')

    plt.close(fig)
