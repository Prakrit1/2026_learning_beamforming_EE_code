import os
import sys

# Guard against leftover shell env vars from other ablation runs (elevation/
# gain/budget sweeps) silently changing this system's configuration -- must be
# the same system as the operating points cached in the triplet.
os.environ.pop('EE_SAT_GAIN_DBI', None)
os.environ.pop('EE_POWER_BUDGET_WATT', None)
os.environ.pop('EE_TARGET_ELEVATION_DEG', None)

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

"""
Power-vs-rate trade-off, learned policy vs RM, in the dual-axis layout used by
Ha et al. (Fig. 6): transmit power on the LEFT axis, sum rate on the RIGHT
axis, one bar group per method. It shows both quantities at once --

  - RM (full power): transmits the full 75 W budget for its rate,
  - learned (proposed): transmits far less power while keeping most of the
    rate,

so the figure reads directly as "the learned policy saved this much power and
gave up this much rate relative to full-power RM". Both operating points come
straight from the cached rate_power_triplet.gzip ('sac_aod0.0' = learned,
'sac_aod0.0_fullpower' = RM) -- no new simulation.

Saves reports/figures/{pdf,jpg,png}/ee_power_rate_tradeoff_sac_error{X}.*
"""

CSIT_ERROR_BOUND = float(sys.argv[sys.argv.index('--error') + 1]) if '--error' in sys.argv else 0.0


def total_power_watt(cfg, transmit_power_watt):
    return transmit_power_watt / cfg.pa_efficiency + cfg.sat_nr * cfg.sat_ant_nr * cfg.circuit_power_watt


if __name__ == '__main__':
    cfg = Config()
    cfg.show_plots = False
    plot_cfg = PlotConfig()
    # PlotConfig() re-enables text.usetex; compute nodes have no latex binary,
    # so re-assert it off or savefig crashes silently under sbatch.
    matplotlib.rcParams['text.usetex'] = False

    triplet_gzip = Path(cfg.output_metrics_path, 'EE_lwin5000_3gpp_triplet', 'rate_power_triplet.gzip')
    if not triplet_gzip.exists():
        raise FileNotFoundError(f'{triplet_gzip} not found -- run plotting_scenario.py first.')
    with gzip.open(triplet_gzip, 'rb') as file:
        triplet = pickle.load(file)
    idx = int(np.argmin(np.abs(triplet['error_sweep_range'] - CSIT_ERROR_BOUND)))

    # learned (clip-only, ~35 W) and RM (same policy rescaled to full 75 W)
    p_learned = float(triplet['results']['sac_aod0.0']['mean_power'][idx])
    r_learned = float(triplet['results']['sac_aod0.0']['mean_rate'][idx])
    p_rm = float(triplet['results']['sac_aod0.0_fullpower']['mean_power'][idx])
    r_rm = float(triplet['results']['sac_aod0.0_fullpower']['mean_rate'][idx])

    power_saved_pct = 100.0 * (1.0 - p_learned / p_rm)
    rate_lost_pct = 100.0 * (1.0 - r_learned / r_rm)
    ee_learned = r_learned / total_power_watt(cfg, p_learned)
    ee_rm = r_rm / total_power_watt(cfg, p_rm)
    print(f'learned: P_tx={p_learned:.1f} W (total {total_power_watt(cfg, p_learned):.1f} W), '
          f'rate={r_learned:.2f} bits/s/Hz, EE={ee_learned:.4f}')
    print(f'RM     : P_tx={p_rm:.1f} W (total {total_power_watt(cfg, p_rm):.1f} W), '
          f'rate={r_rm:.2f} bits/s/Hz, EE={ee_rm:.4f}')
    print(f'-> learned saves {power_saved_pct:.0f}% transmit power for {rate_lost_pct:.0f}% less rate '
          f'(EE +{100 * (ee_learned / ee_rm - 1):.0f}%)')

    # ---- dual-axis bar figure (power left, rate right) --------------------
    power_color = plot_cfg.cp2['blue']
    rate_color = plot_cfg.cp2['green']

    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.6

    fig, ax_p = plt.subplots(figsize=(plot_width, plot_height))
    ax_r = ax_p.twinx()

    x = np.array([0.0, 1.0])          # learned, RM
    bar_w = 0.36
    powers = [p_learned, p_rm]
    rates = [r_learned, r_rm]

    bars_p = ax_p.bar(x - bar_w / 2, powers, width=bar_w, color=power_color,
                      edgecolor='black', linewidth=0.5, label='Transmit power', zorder=3)
    bars_r = ax_r.bar(x + bar_w / 2, rates, width=bar_w, color=rate_color,
                      edgecolor='black', linewidth=0.5, label='Sum rate', zorder=3)

    # value labels; on the learned bars also show the delta vs RM
    ax_p.text(bars_p[0].get_x() + bar_w / 2, powers[0], f'{powers[0]:.0f} W\n(-{power_saved_pct:.0f}%)',
              ha='center', va='bottom', fontsize=9.5, color=power_color)
    ax_p.text(bars_p[1].get_x() + bar_w / 2, powers[1], f'{powers[1]:.0f} W',
              ha='center', va='bottom', fontsize=9.5, color=power_color)
    ax_r.text(bars_r[0].get_x() + bar_w / 2, rates[0], f'{rates[0]:.1f}\n(-{rate_lost_pct:.0f}%)',
              ha='center', va='bottom', fontsize=9.5, color=rate_color)
    ax_r.text(bars_r[1].get_x() + bar_w / 2, rates[1], f'{rates[1]:.1f}',
              ha='center', va='bottom', fontsize=9.5, color=rate_color)

    ax_p.set_xticks(x)
    ax_p.set_xticklabels(['EE', 'RM'], fontsize=12)
    ax_p.set_ylabel('Transmit power [W]', fontsize=13, color=power_color)
    ax_r.set_ylabel('Sum rate [bits/s/Hz]', fontsize=13, color=rate_color)
    ax_p.tick_params(axis='y', labelcolor=power_color)
    ax_r.tick_params(axis='y', labelcolor=rate_color)

    ax_p.set_ylim(0, max(powers) * 1.25)
    ax_r.set_ylim(0, max(rates) * 1.25)
    ax_p.set_xlim(-0.6, 1.6)
    ax_p.set_axisbelow(True)
    ax_p.grid(True, axis='y', alpha=0.2, linewidth=0.5)

    fig.legend(handles=[bars_p, bars_r], loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=2, fontsize=11, frameon=False, columnspacing=1.8, handletextpad=0.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    for subdir, dpi, transparent in [('pdf', 300, True), ('jpg', 200, False), ('png', 200, True)]:
        target = Path(plot_cfg.plots_parent_path, subdir)
        target.mkdir(parents=True, exist_ok=True)
        out = Path(target, f'ee_power_rate_tradeoff_sac_error{CSIT_ERROR_BOUND:g}.{subdir}')
        fig.savefig(out, bbox_inches='tight', dpi=dpi, transparent=transparent)
        print(f'Saved: {out}')

    plt.close(fig)
