"""Combines two previously-separate results into one figure so they read as
one consistent story instead of two seemingly contradicting numbers:

1. The EE-vs-fixed-power sweep (ee_vs_transmit_power_sweep_sac.py): take the
   deployed EE-SAC's own raw (pre-clip) beam DIRECTION per channel
   realization, force it through power levels 1-75 W, and measure EE at
   each -- shows the curve peaks at ~12.4 W for that beam shape.
2. The deployed EE-SAC's own actual (clip-only, un-forced) operating point
   at CSIT error 0 (~34.8 W, ~9 bit/s/Hz), from
   outputs/metrics/EE_lwin5000_3gpp_triplet/rate_power_triplet.gzip
   (produced by plotting_scenario.py / used by error_sweep_training_triplet.py).

Plotting only 12.4 W without also showing where the deployed policy actually
sits (and where the 75 W full-power baseline sits) reads as a contradiction
to a reader who only sees "we save power, using 35 W" elsewhere in the
paper. Putting all three reference points (theoretical per-direction
maximizer, deployed operating point, full budget) on one curve removes that
ambiguity: 35 W is still a real power saving relative to 75 W, even though
it is not the curve's peak.

Prerequisites (must already exist -- this script does no new Monte Carlo
simulation):
    outputs/metrics/EE_vs_transmit_power/ee_vs_power_sweep_sac_error0.gzip
        (run ee_vs_transmit_power_sweep_sac.py --error 0 first if missing)
    outputs/metrics/EE_vs_transmit_power/ee_vs_power_sweep_error0.gzip
        (run ee_vs_transmit_power_sweep.py --error 0 first if missing)
    outputs/metrics/EE_lwin5000_3gpp_triplet/rate_power_triplet.gzip
        (run plotting_scenario.py first if missing)

Saves reports/figures/{pdf,jpg,png}/ee_operating_point_consistency.*
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

CSIT_ERROR_BOUND = 0.0  # matches the sweep gzip and the triplet's error_sweep_range[0]


def total_power_watt(cfg, transmit_power_watt):
    return transmit_power_watt / cfg.pa_efficiency + cfg.sat_nr * cfg.sat_ant_nr * cfg.circuit_power_watt


if __name__ == '__main__':
    cfg = Config()
    plot_cfg = PlotConfig()

    metrics_path = Path(cfg.output_metrics_path)

    # 1) EE-vs-fixed-power sweep curves (SAC direction + MMSE direction)
    sac_sweep_path = Path(metrics_path, 'EE_vs_transmit_power', f'ee_vs_power_sweep_sac_error{CSIT_ERROR_BOUND:g}.gzip')
    with gzip.open(sac_sweep_path, 'rb') as file:
        sac_sweep = pickle.load(file)

    mmse_sweep_path = Path(metrics_path, 'EE_vs_transmit_power', f'ee_vs_power_sweep_error{CSIT_ERROR_BOUND:g}.gzip')
    mmse_sweep = None
    if mmse_sweep_path.exists():
        with gzip.open(mmse_sweep_path, 'rb') as file:
            mmse_sweep = pickle.load(file)

    power_sweep_watt = sac_sweep['power_sweep_watt']
    ee_sweep = sac_sweep['ee']
    argmax_idx = int(np.argmax(ee_sweep))
    ee_peak_power = power_sweep_watt[argmax_idx]
    ee_peak = ee_sweep[argmax_idx]

    # 2) deployed EE-SAC's own actual (clip-only) operating point at error=0
    triplet_path = Path(metrics_path, 'EE_lwin5000_3gpp_triplet', 'rate_power_triplet.gzip')
    with gzip.open(triplet_path, 'rb') as file:
        triplet = pickle.load(file)
    deployed_rate = triplet['results']['sac_aod0.0']['mean_rate'][0]
    deployed_power = triplet['results']['sac_aod0.0']['mean_power'][0]
    deployed_ee = deployed_rate / total_power_watt(cfg, deployed_power)

    power_budget = sac_sweep.get('power_budget', cfg.power_constraint_watt)

    # --- plot ---
    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.62
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))

    ax.plot(power_sweep_watt, ee_sweep, color=plot_cfg.cp2['green'], marker='o', markersize=4,
            linewidth=1.5, label='EE-SAC direction, swept over fixed power')
    if mmse_sweep is not None:
        ax.plot(mmse_sweep['power_sweep_watt'], mmse_sweep['ee'], color=plot_cfg.cp2['gold'], marker='x',
                markersize=4, linewidth=1.3, linestyle='--', label='MMSE direction, swept over fixed power')

    ax.axvline(power_budget, color=plot_cfg.cp2['black'], linestyle=':', linewidth=1.2,
               label=f'power budget ({power_budget:.0f} W)')
    ax.axvline(ee_peak_power, color=plot_cfg.cp2['magenta'], linestyle='--', linewidth=1.3,
               label=f'per-direction EE maximizer ({ee_peak_power:.1f} W)')

    ax.scatter([deployed_power], [deployed_ee], color=plot_cfg.cp2['blue'], marker='*', s=140,
               zorder=5, label=f'deployed EE-SAC operating point ({deployed_power:.1f} W)')
    ax.annotate(
        f'{deployed_power:.1f} W\n{deployed_ee:.4f} bps/Hz/W',
        xy=(deployed_power, deployed_ee), xytext=(8, -14), textcoords='offset points',
        fontsize=7, color=plot_cfg.cp2['blue'], fontweight='bold',
    )

    ax.set_xlabel('Transmit power [W]')
    ax.set_ylabel('EE [bps/Hz/W]')
    ax.set_title(f'EE vs. transmit power, CSIT error bound={CSIT_ERROR_BOUND:g}', fontsize=9)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, loc='upper right')
    fig.tight_layout(pad=0.4)

    pdf_path = Path(plot_cfg.plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(pdf_path, 'ee_operating_point_consistency.pdf'), bbox_inches='tight', dpi=300, transparent=True)

    jpg_path = Path(plot_cfg.plots_parent_path, 'jpg')
    jpg_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(jpg_path, 'ee_operating_point_consistency.jpg'), bbox_inches='tight', dpi=200)

    png_path = Path(plot_cfg.plots_parent_path, 'png')
    png_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(png_path, 'ee_operating_point_consistency.png'), bbox_inches='tight', dpi=200, transparent=True)

    print(f'EE maximizer: {ee_peak_power:.2f} W, EE={ee_peak:.5f} bps/Hz/W')
    print(f'Deployed operating point: {deployed_power:.2f} W, EE={deployed_ee:.5f} bps/Hz/W '
          f'({100 * deployed_ee / ee_peak:.0f}% of peak)')
    plt.close(fig)
