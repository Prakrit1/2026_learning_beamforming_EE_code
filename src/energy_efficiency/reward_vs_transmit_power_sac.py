import os
import sys

# Guard against leftover shell env vars from other ablation runs (elevation/
# gain/budget sweeps) silently changing this system's configuration -- must be
# the same system as the cached sweep and the checkpoint's lambda_ee.
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
Training-reward vs transmit-power curve for the deployed SAC policy.

The Dinkelbach reward the policy is actually trained on (EE_sac.py) is

    reward(P) = R(P) - lambda_ee * P_total(P) / P_max ,

i.e. sum rate minus a LINEAR power price. Concave rate minus a linear term is
concave with a single interior MAXIMUM, and that maximum sits at the policy's
own operating power (~35 W) -- because the trained lambda_ee places it there
(the first-order condition is R'(P) = lambda_ee / (eta_PA * P_max)).

This is deliberately NOT the energy-efficiency curve: EE = R/P_total peaks at
~12.4 W, but the REWARD peaks at ~35 W. Plotting the reward makes explicit
that the policy maximizes R - lambda*P (peak at 35 W), not R/P (peak at
12.4 W) -- which is exactly why the operating point is where it is. (Caveat:
the reward peaking at the operating point is true by construction; this figure
explains the operating point, it does not independently prove it optimal.)

Reads the cached rate(P) sweep from ee_vs_transmit_power_sweep_sac.py and
lambda_ee from the checkpoint (or --lambda X). No new simulation.

Saves reports/figures/{pdf,jpg,png}/reward_vs_transmit_power_sac_error{X}.*
"""

CSIT_ERROR_BOUND = float(sys.argv[sys.argv.index('--error') + 1]) if '--error' in sys.argv else 0.0
LAMBDA_OVERRIDE = float(sys.argv[sys.argv.index('--lambda') + 1]) if '--lambda' in sys.argv else None
TRAINING_NAME = CHECKPOINTS['aod0.0']


def total_power_watt(cfg, transmit_power_watt):
    return transmit_power_watt / cfg.pa_efficiency + cfg.sat_nr * cfg.sat_ant_nr * cfg.circuit_power_watt


def resolve_lambda_ee(cfg):
    if LAMBDA_OVERRIDE is not None:
        print(f'[reward_vs_transmit_power] using --lambda override: {LAMBDA_OVERRIDE}')
        return LAMBDA_OVERRIDE
    checkpoint_path = get_best_model_path(cfg.trained_models_path, TRAINING_NAME)
    lambda_path = Path(checkpoint_path, 'config', 'dinkelbach_lambda_ee.txt')
    if not lambda_path.exists():
        raise FileNotFoundError(
            f'{lambda_path} not found -- this checkpoint predates the lambda_ee-saving '
            f'instrumentation. Pass --lambda X (your trained value, ~3.47) to plot anyway.'
        )
    with open(lambda_path, 'r') as file:
        lam = float(file.read().strip())
    print(f'[reward_vs_transmit_power] checkpoint: {checkpoint_path}, lambda_ee={lam:.4f}')
    return lam


if __name__ == '__main__':
    cfg = Config()
    cfg.show_plots = False
    plot_cfg = PlotConfig()
    # PlotConfig() re-enables text.usetex; compute nodes have no latex binary,
    # so re-assert it off or savefig crashes silently under sbatch.
    matplotlib.rcParams['text.usetex'] = False

    sweep_gzip = Path(cfg.output_metrics_path, 'EE_vs_transmit_power',
                      f'ee_vs_power_sweep_sac_error{CSIT_ERROR_BOUND:g}.gzip')
    if not sweep_gzip.exists():
        raise FileNotFoundError(
            f'{sweep_gzip} not found -- run ee_vs_transmit_power_sweep_sac.py first '
            f'to generate the rate(P) sweep this reward curve is built from.'
        )
    with gzip.open(sweep_gzip, 'rb') as file:
        sweep = pickle.load(file)
    power_sweep_watt = sweep['power_sweep_watt']
    mean_rate = sweep['mean_rate']
    total_power = sweep['total_power_watt']
    ee = sweep['ee']

    lambda_ee = resolve_lambda_ee(cfg)
    p_max = cfg.power_constraint_watt

    # the exact per-step Dinkelbach reward, up to the constant reward coefficient
    # (a uniform y-scale that changes neither the shape nor the peak location).
    reward = mean_rate - lambda_ee * total_power / p_max

    reward_opt_idx = int(np.argmax(reward))
    ee_opt_idx = int(np.argmax(ee))
    print(f'reward peak at P_tx={power_sweep_watt[reward_opt_idx]:.1f} W '
          f'(rate={mean_rate[reward_opt_idx]:.3f}); '
          f'EE peak at P_tx={power_sweep_watt[ee_opt_idx]:.1f} W')

    # deployed operating point (mean transmit power) from the cached triplet
    triplet_gzip = Path(cfg.output_metrics_path, 'EE_lwin5000_3gpp_triplet', 'rate_power_triplet.gzip')
    P_deployed = None
    if triplet_gzip.exists():
        with gzip.open(triplet_gzip, 'rb') as file:
            triplet = pickle.load(file)
        idx = int(np.argmin(np.abs(triplet['error_sweep_range'] - CSIT_ERROR_BOUND)))
        P_deployed = float(triplet['results']['sac_aod0.0']['mean_power'][idx])

    # ---- figure -----------------------------------------------------------
    reward_color = plot_cfg.cp2['green']

    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.6
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))

    ax.plot(power_sweep_watt, reward, color=reward_color, linewidth=2.0,
            label='Reward', zorder=3)

    # reward maximum = the policy's operating point
    ax.scatter([power_sweep_watt[reward_opt_idx]], [reward[reward_opt_idx]],
               marker='*', s=200, color=plot_cfg.cp2['magenta'], edgecolor='black',
               linewidth=0.6, zorder=5)
    ax.annotate(f'reward max\n{power_sweep_watt[reward_opt_idx]:.0f} W',
                xy=(power_sweep_watt[reward_opt_idx], reward[reward_opt_idx]),
                xytext=(power_sweep_watt[reward_opt_idx] + 4, reward[reward_opt_idx]),
                fontsize=9.5, ha='left', va='center',
                arrowprops=dict(arrowstyle='-', color='0.4', lw=0.8))

    # deployed operating point, if it differs visibly from the argmax
    if P_deployed is not None:
        ax.axvline(P_deployed, color='0.7', linestyle=':', linewidth=1.0, zorder=1)

    # EE optimum for contrast: reward peaks well to its right
    ax.axvline(power_sweep_watt[ee_opt_idx], color='0.6', linestyle='-.', linewidth=1.2, zorder=1)
    ax.annotate(f'EE optimum\n{power_sweep_watt[ee_opt_idx]:.0f} W',
                xy=(power_sweep_watt[ee_opt_idx], float(np.min(reward))),
                xytext=(power_sweep_watt[ee_opt_idx] + 1.5, float(np.min(reward))),
                fontsize=9, ha='left', va='bottom', color='0.35')

    # RM (full power) reference point on the reward curve
    ax.scatter([power_sweep_watt[-1]], [reward[-1]], marker='o', s=55,
               facecolor='white', edgecolor='black', linewidth=1.0, zorder=5)
    ax.annotate(f'RM: {power_sweep_watt[-1]:.0f} W',
                xy=(power_sweep_watt[-1], reward[-1]),
                xytext=(power_sweep_watt[-1] - 3, reward[-1]),
                fontsize=9.5, ha='right', va='center',
                arrowprops=dict(arrowstyle='-', color='0.4', lw=0.8))

    ax.set_xlabel(r'Transmit power $P_{\mathrm{tx}}$ [W]', fontsize=13)
    ax.set_ylabel(r'Reward  $R-\lambda_{\mathrm{ee}}\,P_{\mathrm{tot}}/P_{\max}$ [bits/s/Hz]',
                  fontsize=12)
    ax.set_xlim(0, 78)
    ax.grid(True, axis='y', alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()

    for subdir, dpi, transparent in [('pdf', 300, True), ('jpg', 200, False), ('png', 200, True)]:
        target = Path(plot_cfg.plots_parent_path, subdir)
        target.mkdir(parents=True, exist_ok=True)
        out = Path(target, f'reward_vs_transmit_power_sac_error{CSIT_ERROR_BOUND:g}.{subdir}')
        fig.savefig(out, bbox_inches='tight', dpi=dpi, transparent=transparent)
        print(f'Saved: {out}')

    plt.close(fig)
