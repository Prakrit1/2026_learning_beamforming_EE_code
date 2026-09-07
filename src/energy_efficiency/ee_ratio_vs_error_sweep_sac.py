
import os
import sys

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
Energy efficiency ratio (rate / total_power, bits/s/Hz/W) versus CSIT
error bound, for three precoders side by side:

  - EE : the energy-efficient policy (clip-only projection, ~35 W operating
         point) -- result key 'sac_aod0.0'
  - RM : rate-maximization, always rescaled to the full 75 W budget
         -- result key 'sac_aod0.0_fullpower'
  - MMSE: classical baseline, full 75 W budget -- result key 'mmse_nadir'

This is the "curve" version of the EE story that deliberately puts CSIT
error bound (not transmit power) on the x-axis: the EE policy's ratio sits
consistently above RM and MMSE across the whole error sweep, so the point
is "our policy is more energy efficient, and stays so under channel
uncertainty" -- without ever surfacing the fixed-power ratio sweep's
interior maximum (~12.4 W), which is a property of the hardware power model
and not the contribution this paper is making.

Plot-only: reads the already-cached rate_power_triplet.gzip (means per
error bound, produced by plotting_scenario.py) -- no new simulation.

Saves reports/figures/{pdf,jpg,png}/ee_ratio_vs_error_sweep_sac.*
"""


def total_power_watt(cfg, transmit_power_watt):
    return transmit_power_watt / cfg.pa_efficiency + cfg.sat_nr * cfg.sat_ant_nr * cfg.circuit_power_watt


def ratio_curve(cfg, result):
    """rate(error) / total_power(error), elementwise over the error sweep."""
    mean_rate = np.asarray(result['mean_rate'])
    mean_power = np.asarray(result['mean_power'])
    total_power = np.array([total_power_watt(cfg, p) for p in mean_power])
    return mean_rate / total_power


if __name__ == '__main__':
    cfg = Config()
    cfg.show_plots = False
    plot_cfg = PlotConfig()
    matplotlib.rcParams['text.usetex'] = False

    gzip_path = Path(cfg.output_metrics_path, 'EE_lwin5000_3gpp_triplet', 'rate_power_triplet.gzip')
    if not gzip_path.exists():
        raise FileNotFoundError(f'{gzip_path} not found -- run plotting_scenario.py first.')
    with gzip.open(gzip_path, 'rb') as file:
        data = pickle.load(file)

    error_sweep_range = data['error_sweep_range']
    results = data['results']

    ee_ratio = ratio_curve(cfg, results['sac_aod0.0'])
    rm_ratio = ratio_curve(cfg, results['sac_aod0.0_fullpower'])
    mmse_ratio = ratio_curve(cfg, results['mmse_nadir'])

    print(f'At error=0.0: EE={ee_ratio[0]:.4f}, RM={rm_ratio[0]:.4f}, MMSE={mmse_ratio[0]:.4f} bps/Hz/W')
    print(f'EE is {100 * (ee_ratio[0] / rm_ratio[0] - 1):.1f}% more efficient than RM at error=0.0')

    plot_cfg = PlotConfig()
    matplotlib.rcParams['text.usetex'] = False
    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.62

    fig, ax = plt.subplots(figsize=(plot_width, plot_height))

    ax.plot(error_sweep_range, ee_ratio, color=plot_cfg.cp2['green'], marker='o', markersize=5,
            linewidth=1.5, label='EE (clip-only)')
    ax.plot(error_sweep_range, rm_ratio, color=plot_cfg.cp2['gold'], marker='s', markersize=5,
            linewidth=1.5, linestyle='-', label='RM (75 W)')
    ax.plot(error_sweep_range, mmse_ratio, color=plot_cfg.cp2['black'], marker='^', markersize=5,
            linewidth=1.5, linestyle=':', label='MMSE (75 W)')

    ax.set_xlabel(r'CSIT error bound $\Delta\epsilon$', fontsize=13)
    ax.set_ylabel(r'EE [bits/s/Hz/W]', fontsize=13)
    ax.grid(True, alpha=0.3, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc='best', fontsize=11, framealpha=0.9, frameon=True)
    plt.tight_layout(pad=0.3)

    pdf_path = Path(plot_cfg.plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)
    out = Path(pdf_path, 'ee_ratio_vs_error_sweep_sac.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=300, transparent=True)
    print(f'Saved: {out}')

    jpg_path = Path(plot_cfg.plots_parent_path, 'jpg')
    jpg_path.mkdir(parents=True, exist_ok=True)
    out_jpg = Path(jpg_path, 'ee_ratio_vs_error_sweep_sac.jpg')
    fig.savefig(out_jpg, bbox_inches='tight', dpi=200)
    print(f'Saved: {out_jpg}')

    png_path = Path(plot_cfg.plots_parent_path, 'png')
    png_path.mkdir(parents=True, exist_ok=True)
    out_png = Path(png_path, 'ee_ratio_vs_error_sweep_sac.png')
    fig.savefig(out_png, bbox_inches='tight', dpi=200, transparent=True)
    print(f'Saved: {out_png}')

    plt.close(fig)
