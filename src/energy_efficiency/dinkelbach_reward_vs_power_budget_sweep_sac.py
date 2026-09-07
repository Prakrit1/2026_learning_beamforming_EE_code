
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
The classical ratio EE = rate/total_power (ee_vs_transmit_power_sweep_sac.py,
ee_vs_power_budget_sweep_sac.py) is NOT what the SAC-EE policy is actually
trained to maximize -- training uses the Dinkelbach reward

    reward(P) = sumrate(P) - lambda_ee * (total_power_watt(P) / TRAINING_BUDGET_WATT)

(EE_sac.py:330, note power is normalized by the *training-time* budget,
not raw watts -- see total_power_dinkelbach there). This script re-plots
the already-cached adaptive clip-only sweep
(ee_vs_power_budget_sweep_sac_error{X}.gzip) against THIS objective
instead, to check whether the policy's own ~35 W operating point is
actually where its true training reward peaks (unlike the ratio, which
peaks at ~12.4 W regardless of allocation strategy -- see the chat
discussion this script came out of).

lambda_ee is NOT recoverable exactly: it's only ever printed to the
training log ("Dinkelbach lambda updated to X", EE_sac.py:533), never
saved into the checkpoint file, and the actual training log for this
checkpoint no longer exists (confirmed by searching full git history,
including the d5669cf..0530fd3 window when slurm logs were briefly
tracked -- job 156358's log was never committed). LAMBDA_EE_ESTIMATE
below is therefore an approximation, back-derived from the fixed-point
condition Dinkelbach's EMA update converges to (lambda_ee ~= rate_ema /
power_ema at steady state), using this checkpoint's own known converged
operating averages at the 75 W training budget (rate_power_triplet.gzip,
'sac_aod0.0', error=0). If you find the actual logged value, hardcode it
here instead and note where it came from.

Saves reports/figures/{pdf,jpg,png}/dinkelbach_reward_vs_power_budget_sac_error{X}.*
"""

CSIT_ERROR_BOUND = float(sys.argv[sys.argv.index('--error') + 1]) if '--error' in sys.argv else 0.0


def total_power_watt(cfg, transmit_power_watt):
    return transmit_power_watt / cfg.pa_efficiency + cfg.sat_nr * cfg.sat_ant_nr * cfg.circuit_power_watt


if __name__ == '__main__':
    cfg = Config()
    cfg.show_plots = False
    plot_cfg = PlotConfig()
    matplotlib.rcParams['text.usetex'] = False

    TRAINING_BUDGET_WATT = cfg.power_constraint_watt  # 75 W -- the fixed normalizer, not the swept budget below

    metrics_path = Path(cfg.output_metrics_path, 'EE_vs_transmit_power')
    budget_sweep_gzip = Path(metrics_path, f'ee_vs_power_budget_sweep_sac_error{CSIT_ERROR_BOUND:g}.gzip')
    if not budget_sweep_gzip.exists():
        raise FileNotFoundError(
            f'{budget_sweep_gzip} not found -- run ee_vs_power_budget_sweep_sac.py '
            f'(--error {CSIT_ERROR_BOUND:g}) first, this script only re-plots its cached sweep.'
        )
    with gzip.open(budget_sweep_gzip, 'rb') as file:
        sweep_data = pickle.load(file)
    budget_sweep_watt = sweep_data['budget_sweep_watt']
    mean_rate = sweep_data['mean_rate']
    mean_power = sweep_data['mean_power']

    # lambda_ee estimate: back out from this checkpoint's own converged
    # operating point at the training budget (rate_power_triplet.gzip's
    # error-sweep results, evaluated at the same CSIT_ERROR_BOUND).
    triplet_gzip = Path(cfg.output_metrics_path, 'EE_lwin5000_3gpp_triplet', 'rate_power_triplet.gzip')
    if not triplet_gzip.exists():
        raise FileNotFoundError(
            f'{triplet_gzip} not found -- run plotting_scenario.py first, '
            f'needed here to estimate lambda_ee from the deployed operating point.'
        )
    with gzip.open(triplet_gzip, 'rb') as file:
        triplet_data = pickle.load(file)
    error_idx = int(np.argmin(np.abs(triplet_data['error_sweep_range'] - CSIT_ERROR_BOUND)))
    deployed_rate = triplet_data['results']['sac_aod0.0']['mean_rate'][error_idx]
    deployed_power = triplet_data['results']['sac_aod0.0']['mean_power'][error_idx]
    deployed_total_power_normalized = total_power_watt(cfg, deployed_power) / TRAINING_BUDGET_WATT
    LAMBDA_EE_ESTIMATE = deployed_rate / deployed_total_power_normalized
    print(f'lambda_ee estimate: {LAMBDA_EE_ESTIMATE:.4f} '
          f'(back-derived from deployed point: rate={deployed_rate:.4f} bps/Hz, '
          f'power={deployed_power:.2f} W, training_budget={TRAINING_BUDGET_WATT:.0f} W -- '
          f'NOT the exact logged training-time value, see module docstring)')

    total_power_normalized = np.array([total_power_watt(cfg, p) for p in mean_power]) / TRAINING_BUDGET_WATT
    dinkelbach_reward = mean_rate - LAMBDA_EE_ESTIMATE * total_power_normalized

    argmax_idx = int(np.argmax(dinkelbach_reward))
    print(f'Dinkelbach-reward maximizer over the swept budget: budget={budget_sweep_watt[argmax_idx]:.2f} W, '
          f'achieved mean_power={mean_power[argmax_idx]:.2f} W, reward={dinkelbach_reward[argmax_idx]:.4f}, '
          f'rate={mean_rate[argmax_idx]:.4f} bps/Hz')
    print(f'At the full training-time budget (75 W): achieved mean_power={mean_power[-1]:.2f} W, '
          f'reward={dinkelbach_reward[-1]:.4f}, rate={mean_rate[-1]:.4f} bps/Hz')

    pdf_path = Path(plot_cfg.plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)

    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.62

    fig, ax = plt.subplots(figsize=(plot_width, plot_height))

    ax.plot(budget_sweep_watt, dinkelbach_reward, color=plot_cfg.cp2['green'], marker='o', markersize=4,
            linewidth=1.5, label=r'Dinkelbach reward, $\lambda_{\mathrm{ee}} \approx$' + f'{LAMBDA_EE_ESTIMATE:.2f}')
    ax.axvline(budget_sweep_watt[argmax_idx], color='gray', linestyle='-.', linewidth=1.3,
               label=fr'$B^\star \approx {budget_sweep_watt[argmax_idx]:.0f}$ W')
    ax.plot(budget_sweep_watt[-1], dinkelbach_reward[-1], color=plot_cfg.cp2['gold'], marker='*', markersize=12,
            linestyle='none', zorder=5,
            label=fr'Deployed (75 W budget), $\bar P \approx {mean_power[-1]:.0f}$ W')

    ax.set_xlabel(r'Available power budget $B$ [W]', fontsize=13)
    ax.set_ylabel(r'Dinkelbach reward [bps/Hz]', fontsize=13)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1),
               ncol=1, fontsize=11, frameon=False, columnspacing=1.4,
               handletextpad=0.5)
    fig.tight_layout(rect=(0, 0, 1, 0.82))

    out = Path(pdf_path, f'dinkelbach_reward_vs_power_budget_sac_error{CSIT_ERROR_BOUND:g}.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=300, transparent=True)
    print(f'Saved: {out}')

    jpg_path = Path(plot_cfg.plots_parent_path, 'jpg')
    jpg_path.mkdir(parents=True, exist_ok=True)
    out_jpg = Path(jpg_path, f'dinkelbach_reward_vs_power_budget_sac_error{CSIT_ERROR_BOUND:g}.jpg')
    fig.savefig(out_jpg, bbox_inches='tight', dpi=200)
    print(f'Saved: {out_jpg}')

    png_path = Path(plot_cfg.plots_parent_path, 'png')
    png_path.mkdir(parents=True, exist_ok=True)
    out_png = Path(png_path, f'dinkelbach_reward_vs_power_budget_sac_error{CSIT_ERROR_BOUND:g}.png')
    fig.savefig(out_png, bbox_inches='tight', dpi=200, transparent=True)
    print(f'Saved: {out_png}')

    plt.close(fig)
