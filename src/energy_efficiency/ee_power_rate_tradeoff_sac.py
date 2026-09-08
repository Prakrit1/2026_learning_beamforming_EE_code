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

    # Group by QUANTITY: the two power bars (EE, RM) together on the left axis,
    # the two rate bars (EE, RM) together on the right axis. Within each group
    # EE is solid and RM is hatched, so the same scheme reads across groups.
    from matplotlib.patches import Patch

    bar_w = 0.55
    x_power = np.array([0.0, 1.4])        # EE, RM  -- power group (left axis, blue)
    x_rate = np.array([2.8, 4.2])         # EE, RM  -- rate group (right axis, green)
    powers = [p_learned, p_rm]
    rates = [r_learned, r_rm]

    bars_p = ax_p.bar(x_power, powers, width=bar_w, color=power_color,
                      edgecolor='black', linewidth=0.6, zorder=3)
    bars_r = ax_r.bar(x_rate, rates, width=bar_w, color=rate_color,
                      edgecolor='black', linewidth=0.6, zorder=3)
    bars_p[1].set_hatch('//')             # RM power
    bars_r[1].set_hatch('//')             # RM rate

    # Decrement arrows (no on-bar values): a dashed guide at the EE level and a
    # double-headed arrow just right of each RM bar, labelled with the percent
    # drop -- so "power down X%, rate down only Y%" reads off the notation.
    def decrement_arrow(ax, x_ee, x_rm, y_ee, y_rm, pct, word):
        arrow_x = x_rm - bar_w / 2 - 0.12          # just left of the RM bar
        ax.hlines(y_ee, x_ee, arrow_x, colors='0.45', linestyles='--', linewidth=1.0, zorder=4)
        ax.annotate('', xy=(arrow_x, y_ee), xytext=(arrow_x, y_rm),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=1.5), zorder=6)
        ax.text(arrow_x - 0.10, 0.5 * (y_ee + y_rm), rf'$\approx {pct:.0f}\%$' + f'\n{word}',
                ha='right', va='center', fontsize=11, zorder=6)

    decrement_arrow(ax_p, x_power[0], x_power[1], p_learned, p_rm, power_saved_pct, 'power saved')
    decrement_arrow(ax_r, x_rate[0], x_rate[1], r_learned, r_rm, rate_lost_pct, 'rate loss')

    # per-bar EE/RM labels only (group identity is given by the left/right axes)
    ax_p.set_xticks([x_power[0], x_power[1], x_rate[0], x_rate[1]])
    ax_p.set_xticklabels(['EE', 'RM', 'EE', 'RM'], fontsize=11)

    ax_p.set_ylabel('Transmit power [W]', fontsize=13, color=power_color)
    ax_r.set_ylabel('Sum rate [bits/s/Hz]', fontsize=13, color=rate_color)
    ax_p.tick_params(axis='y', labelcolor=power_color)
    ax_r.tick_params(axis='y', labelcolor=rate_color)

    ax_p.set_ylim(0, max(powers) * 1.30)
    ax_r.set_ylim(0, max(rates) * 1.30)
    ax_p.set_xlim(-0.8, 5.0)
    ax_p.set_axisbelow(True)
    ax_p.grid(True, axis='y', alpha=0.2, linewidth=0.5)

    # legend in the trained^ / evaluated-P style of the triplet figures
    trained_watt = int(round(cfg.power_constraint_watt))
    ee_eval_watt = int(round(p_learned))
    rm_eval_watt = int(round(p_rm))
    legend_handles = [
        Patch(facecolor='0.75', edgecolor='black',
              label=rf'EE$^{{{trained_watt}}}$, $P={ee_eval_watt}$ W'),
        Patch(facecolor='0.75', edgecolor='black', hatch='//',
              label=rf'RM$^{{{trained_watt}}}$, $P={rm_eval_watt}$ W'),
    ]
    ax_p.legend(handles=legend_handles, loc='upper center', ncol=2, fontsize=10,
                frameon=False, columnspacing=1.6, handletextpad=0.5)
    fig.tight_layout()

    for subdir, dpi, transparent in [('pdf', 300, True), ('jpg', 200, False), ('png', 200, True)]:
        target = Path(plot_cfg.plots_parent_path, subdir)
        target.mkdir(parents=True, exist_ok=True)
        out = Path(target, f'ee_power_rate_tradeoff_sac_error{CSIT_ERROR_BOUND:g}.{subdir}')
        fig.savefig(out, bbox_inches='tight', dpi=dpi, transparent=transparent)
        print(f'Saved: {out}')

    plt.close(fig)
