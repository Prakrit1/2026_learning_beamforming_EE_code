"""EE vs transmit power from the REAL training logs -- not a forced-power sweep.

During training the policy actually used a range of transmit powers (SAC
exploration + the policy evolving), and at each episode it achieved a real
mean rate. Every episode is therefore a genuine (transmit power, EE) point.
We plot those points and bin-average over transmit power. Nothing is forced or
recomputed: rate and power are read straight from
training_error_learned_full.gzip.

Units: episode_mean_power_dinkelbach is the mean TOTAL consumed power that
episode, normalized by the budget (x budget). Inverting the known power model
gives the radiated transmit power for the x-axis:
    P_total = power_norm * P_max
    P_tx    = (P_total - N*P_circuit) * eta_PA
and EE = rate / P_total  [bits/s/Hz/W].

Caveats (honest): the x-range only covers powers the policy actually explored,
and EE at a given power mixes early (poor) and late (converged) policies -- the
points are colored by episode so that trajectory is visible; the binned mean
averages over it.

Run:  python3 src/energy_efficiency/ee_vs_transmit_power_from_training.py
Saves reports/figures/{pdf,jpg,png}/ee_vs_transmit_power_from_training.*
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
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        hi = x <= edges[b + 1] if b == n_bins - 1 else x < edges[b + 1]
        sel = (x >= edges[b]) & hi
        counts[b] = int(np.sum(sel))
        if counts[b] > 0:
            means[b] = np.nanmean(y[sel])
    ok = np.isfinite(means)
    return centers[ok], means[ok], counts[ok]


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

    power_norm = np.asarray(m['episode_mean_power_dinkelbach'], dtype=float)
    rate = np.asarray(m['episode_mean_rate_dinkelbach'], dtype=float)

    valid = np.isfinite(power_norm) & np.isfinite(rate) & (power_norm > 0)
    power_norm, rate = power_norm[valid], rate[valid]
    episode_idx = np.arange(len(m['episode_mean_power_dinkelbach']))[valid]

    circuit_watt = cfg.sat_nr * cfg.sat_ant_nr * cfg.circuit_power_watt
    p_total = power_norm * cfg.power_constraint_watt              # consumed DC watts
    p_tx = (p_total - circuit_watt) * cfg.pa_efficiency           # radiated transmit watts
    ee = rate / p_total                                          # bits/s/Hz/W

    print(f'episodes with data: {valid.sum()} / {valid.size}')
    print(f'transmit power explored: {p_tx.min():.2f} - {p_tx.max():.2f} W')
    print(f'EE range: {ee.min():.5f} - {ee.max():.5f}; EE max at P_tx={p_tx[int(np.argmax(ee))]:.2f} W')
    bx, by, bc = binned_mean(p_tx, ee, N_BINS)
    print('binned EE vs P_tx (center W, mean EE, n):')
    for cx, cy, cn in zip(bx, by, bc):
        print(f'  {cx:7.2f}  {cy:.5f}  ({cn})')

    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.6
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))

    sc = ax.scatter(p_tx, ee, c=episode_idx, cmap='viridis', s=12, alpha=0.5, zorder=2)
    ax.plot(bx, by, color='black', linewidth=1.8, zorder=3, label='binned mean')

    ax.set_xlabel(r'Transmit power $P_{\mathrm{tx}}$ [W]', fontsize=13)
    ax.set_ylabel('Energy efficiency [bits/s/Hz/W]', fontsize=13)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, frameon=False, loc='best')

    cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label('training episode', fontsize=10)
    fig.tight_layout()

    for subdir, dpi, transparent in [('pdf', 300, True), ('jpg', 200, False), ('png', 200, True)]:
        target = Path(plot_cfg.plots_parent_path, subdir)
        target.mkdir(parents=True, exist_ok=True)
        out = Path(target, f'ee_vs_transmit_power_from_training.{subdir}')
        fig.savefig(out, bbox_inches='tight', dpi=dpi, transparent=transparent)
        print(f'Saved: {out}')

    plt.close(fig)
