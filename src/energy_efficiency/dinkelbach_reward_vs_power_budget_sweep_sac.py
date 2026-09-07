
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
The classical ratio EE = rate/total_power (ee_vs_transmit_power_sweep_sac.py,
ee_vs_power_budget_sweep_sac.py) is NOT what the SAC-EE policy is actually
trained to maximize -- training uses the Dinkelbach reward

    reward(P) = sumrate(P) - lambda_ee * (total_power_watt(P) / TRAINING_BUDGET_WATT)

(EE_sac.py:330, note power is normalized by the *training-time* budget,
not raw watts -- see total_power_dinkelbach there). This script re-plots
the already-cached adaptive clip-only sweep
(ee_vs_power_budget_sweep_sac_error{X}.gzip) against THIS objective
instead.

lambda_ee is now read from the ACTUAL training run instead of estimated:
EE_sac.py was instrumented (this session) to save dinkelbach_lambda_ee.txt
next to every checkpoint it saves, and the aod=0.0 checkpoint was
retrained specifically to capture this. Earlier versions of this script
had to back-derive lambda_ee (first from the deployed ~35 W point, which
produced an invalid ~9.15 that peaked at an arbitrary ~18 W; then from
the ratio's own peak via Dinkelbach's theorem, which is only guaranteed
to reproduce the ratio's 12.4 W peak, not reveal anything new) because
the original checkpoint's training log was lost. Neither estimate is
needed anymore now that the real value is saved.

IMPORTANT: this script's rate/power sweep data (from
ee_vs_power_budget_sweep_sac_error{X}.gzip) MUST come from the SAME
checkpoint as the lambda_ee it's paired with, or the reward computed
here is meaningless. get_best_model_path is session-aware (picks the
best-scoring checkpoint from the most recent training session, see
plotting_scenario.py), so after retraining, re-run
ee_vs_power_budget_sweep_sac.slurm again FIRST to regenerate that gzip
against the new checkpoint before running this script.

Also plots "overall transmit power" (P_total = P/eta_PA + P_circuit,
the paper's \\ptotal, i.e. actual DC draw including circuit power and PA
inefficiency -- not just the raw radiated/swept budget) on a second axis
against the same x, so the reward's shape can be read directly against
how much power it actually costs.

Saves reports/figures/{pdf,jpg,png}/dinkelbach_reward_vs_power_budget_sac_error{X}.*
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

    TRAINING_BUDGET_WATT = cfg.power_constraint_watt  # 75 W -- the fixed normalizer, not the swept budget below

    checkpoint_path = get_best_model_path(cfg.trained_models_path, TRAINING_NAME)
    lambda_ee_path = Path(checkpoint_path, 'config', 'dinkelbach_lambda_ee.txt')
    if not lambda_ee_path.exists():
        raise FileNotFoundError(
            f'{lambda_ee_path} not found -- this checkpoint predates the lambda_ee-saving '
            f'instrumentation in EE_sac.py. Retrain (or point get_best_model_path at a '
            f'checkpoint saved after that change) before running this script.'
        )
    with open(lambda_ee_path, 'r') as file:
        LAMBDA_EE = float(file.read().strip())
    print(f'[dinkelbach_reward_vs_power_budget_sweep_sac] checkpoint: {checkpoint_path}')
    print(f'Real trained lambda_ee: {LAMBDA_EE:.4f} (read from {lambda_ee_path})')

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

    if sweep_data.get('checkpoint') != str(checkpoint_path):
        print(
            f'[dinkelbach_reward_vs_power_budget_sweep_sac] WARNING: cached sweep gzip was '
            f'generated from checkpoint {sweep_data.get("checkpoint")}, but lambda_ee was just '
            f'read from a different checkpoint ({checkpoint_path}). Re-run '
            f'ee_vs_power_budget_sweep_sac.py first so both come from the same checkpoint -- '
            f'proceeding anyway, but the reward below may not be meaningful.'
        )

    total_power_watt_arr = np.array([total_power_watt(cfg, p) for p in mean_power])
    total_power_normalized = total_power_watt_arr / TRAINING_BUDGET_WATT

    dinkelbach_reward = mean_rate - LAMBDA_EE * total_power_normalized

    argmax_idx = int(np.argmax(dinkelbach_reward))
    print(f'Dinkelbach-reward maximizer over the swept budget: budget={budget_sweep_watt[argmax_idx]:.2f} W, '
          f'achieved mean_power={mean_power[argmax_idx]:.2f} W, '
          f'overall (total) power={total_power_watt_arr[argmax_idx]:.2f} W, '
          f'reward={dinkelbach_reward[argmax_idx]:.4f}, rate={mean_rate[argmax_idx]:.4f} bps/Hz')
    print(f'At the full training-time budget (75 W): achieved mean_power={mean_power[-1]:.2f} W, '
          f'overall (total) power={total_power_watt_arr[-1]:.2f} W, '
          f'reward={dinkelbach_reward[-1]:.4f}, rate={mean_rate[-1]:.4f} bps/Hz')

    pdf_path = Path(plot_cfg.plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)

    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.62

    fig, ax = plt.subplots(figsize=(plot_width, plot_height))

    # Both curves in bps/Hz on one shared axis -- no Watts here, so the
    # vertical gap between them is directly meaningful: it equals exactly
    # the power-penalty term lambda_ee * total_power_normalized(B), not an
    # artifact of two differently-scaled axes.
    line_rate, = ax.plot(
        budget_sweep_watt, mean_rate, color=plot_cfg.cp2['blue'], marker='s', markersize=4,
        linestyle='--', linewidth=1.5, label='Sum rate (no power penalty)',
    )
    line_reward, = ax.plot(
        budget_sweep_watt, dinkelbach_reward, color=plot_cfg.cp2['green'], marker='o', markersize=4,
        linewidth=1.5, label=fr'Dinkelbach reward, $\lambda_{{\mathrm{{ee}}}} = {LAMBDA_EE:.2f}$ (trained)',
    )
    vline = ax.axvline(budget_sweep_watt[argmax_idx], color='gray', linestyle='-.', linewidth=1.3,
                        label=fr'$B^\star \approx {budget_sweep_watt[argmax_idx]:.0f}$ W')

    ax.set_xlabel(r'Available power budget $B$ [W]', fontsize=13)
    ax.set_ylabel(r'bps/Hz', fontsize=13)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    handles = [line_rate, line_reward, vline]
    labels = [h.get_label() for h in handles]
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.12),
               ncol=1, fontsize=11, frameon=False, columnspacing=1.4,
               handletextpad=0.5)
    fig.tight_layout(rect=(0, 0, 1, 0.80))

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
