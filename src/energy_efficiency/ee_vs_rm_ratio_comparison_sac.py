
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
from src.energy_efficiency.plotting_scenario import CHECKPOINTS, get_best_model_path

"""
Two-bar comparison of the classical EE ratio (rate/total_power, bps/Hz/W)
at exactly two single operating points, not a swept curve:

  - EE: the deployed policy's own actual operating point (~35 W), the one
    its real trained lambda_ee (~3.47, see dinkelbach_lambda_ee.txt) drives
    it to via the Dinkelbach reward -- NOT the theoretical ratio optimum
    (~12.4 W, see ee_vs_power_budget_sweep_sac.py/ee_vs_transmit_power_sweep_sac.py).
  - RM: rate-maximization, always rescaled to the full 75 W training budget.

Sidesteps the "why isn't EE at its own ratio optimum" question entirely --
this figure doesn't claim EE is ratio-optimal, only that the policy the
reward actually produces beats the naive full-power baseline in bits/Joule
while retaining most of the throughput. Both numbers come straight from
the already-cached rate_power_triplet.gzip (plotting_scenario.py's
'sac_aod0.0' and 'sac_aod0.0_fullpower' results) -- no new simulation.

Saves reports/figures/{pdf,jpg,png}/ee_vs_rm_ratio_comparison_sac_error{X}.*
"""

CSIT_ERROR_BOUND = float(sys.argv[sys.argv.index('--error') + 1]) if '--error' in sys.argv else 0.0
TRAINING_NAME = CHECKPOINTS['aod0.0']


def total_power_watt(cfg, transmit_power_watt):
    return transmit_power_watt / cfg.pa_efficiency + cfg.sat_nr * cfg.sat_ant_nr * cfg.circuit_power_watt


if __name__ == '__main__':
    cfg = Config()
    cfg.show_plots = False
    plot_cfg = PlotConfig()
    matplotlib.rcParams['text.usetex'] = False

    # Real trained lambda_ee, read purely for the plot's annotation --
    # not used in the ratio computation below (the ratio doesn't depend
    # on lambda at all, only on this checkpoint's own measured rate/power).
    checkpoint_path = get_best_model_path(cfg.trained_models_path, TRAINING_NAME)
    lambda_ee_path = Path(checkpoint_path, 'config', 'dinkelbach_lambda_ee.txt')
    if not lambda_ee_path.exists():
        raise FileNotFoundError(
            f'{lambda_ee_path} not found -- this checkpoint predates the lambda_ee-saving '
            f'instrumentation in EE_sac.py.'
        )
    with open(lambda_ee_path, 'r') as file:
        LAMBDA_EE = float(file.read().strip())
    print(f'[ee_vs_rm_ratio_comparison_sac] checkpoint: {checkpoint_path}, lambda_ee={LAMBDA_EE:.4f}')

    triplet_gzip = Path(cfg.output_metrics_path, 'EE_lwin5000_3gpp_triplet', 'rate_power_triplet.gzip')
    if not triplet_gzip.exists():
        raise FileNotFoundError(f'{triplet_gzip} not found -- run plotting_scenario.py first.')
    with gzip.open(triplet_gzip, 'rb') as file:
        triplet_data = pickle.load(file)
    error_idx = int(np.argmin(np.abs(triplet_data['error_sweep_range'] - CSIT_ERROR_BOUND)))

    ee_rate = float(triplet_data['results']['sac_aod0.0']['mean_rate'][error_idx])
    ee_power = float(triplet_data['results']['sac_aod0.0']['mean_power'][error_idx])
    ee_total_power = total_power_watt(cfg, ee_power)
    ee_ratio = ee_rate / ee_total_power

    rm_rate = float(triplet_data['results']['sac_aod0.0_fullpower']['mean_rate'][error_idx])
    rm_power = float(triplet_data['results']['sac_aod0.0_fullpower']['mean_power'][error_idx])
    rm_total_power = total_power_watt(cfg, rm_power)
    rm_ratio = rm_rate / rm_total_power

    print(f'EE (deployed, lambda_ee={LAMBDA_EE:.2f}): radiated={ee_power:.2f} W, '
          f'total_power={ee_total_power:.2f} W, rate={ee_rate:.4f} bps/Hz, ratio={ee_ratio:.4f} bps/Hz/W')
    print(f'RM (full power): radiated={rm_power:.2f} W, total_power={rm_total_power:.2f} W, '
          f'rate={rm_rate:.4f} bps/Hz, ratio={rm_ratio:.4f} bps/Hz/W')
    print(f'EE is {100 * (ee_ratio / rm_ratio - 1):.1f}% more efficient than RM at full power')

    plot_width = 0.6 * plot_cfg.textwidth
    plot_height = plot_width * 0.9

    fig, ax = plt.subplots(figsize=(plot_width, plot_height))

    labels = [fr'EE, $\lambda_{{\mathrm{{ee}}}} = {LAMBDA_EE:.2f}$' + f'\n(${ee_power:.0f}$ W radiated)', 'RM\n(75 W radiated)']
    ratios = [ee_ratio, rm_ratio]
    colors = [plot_cfg.cp2['green'], plot_cfg.cp2['gold']]

    bars = ax.bar(labels, ratios, color=colors, width=0.6)
    for bar, ratio in zip(bars, ratios):
        ax.annotate(f'{ratio:.3f}', xy=(bar.get_x() + bar.get_width() / 2, ratio),
                    xytext=(0, 4), textcoords='offset points', ha='center', fontsize=11)

    ax.set_ylabel(r'EE [bits/s/Hz/W]', fontsize=13)
    ax.grid(True, axis='y', alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()

    pdf_path = Path(plot_cfg.plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)
    out = Path(pdf_path, f'ee_vs_rm_ratio_comparison_sac_error{CSIT_ERROR_BOUND:g}.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=300, transparent=True)
    print(f'Saved: {out}')

    jpg_path = Path(plot_cfg.plots_parent_path, 'jpg')
    jpg_path.mkdir(parents=True, exist_ok=True)
    out_jpg = Path(jpg_path, f'ee_vs_rm_ratio_comparison_sac_error{CSIT_ERROR_BOUND:g}.jpg')
    fig.savefig(out_jpg, bbox_inches='tight', dpi=200)
    print(f'Saved: {out_jpg}')

    png_path = Path(plot_cfg.plots_parent_path, 'png')
    png_path.mkdir(parents=True, exist_ok=True)
    out_png = Path(png_path, f'ee_vs_rm_ratio_comparison_sac_error{CSIT_ERROR_BOUND:g}.png')
    fig.savefig(out_png, bbox_inches='tight', dpi=200, transparent=True)
    print(f'Saved: {out_png}')

    plt.close(fig)
