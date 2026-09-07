
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
Plots the classical EE ratio (rate / total_power, bits/s/Hz/W) for the
adaptive clip-only policy, swept over available power budget, against a
flat full-power ("RM") baseline reference line -- both sides computed as
each one's own rate divided by its own total power, so the whole figure
is in consistent bps/Hz/W units throughout.

This script started out plotting the actual Dinkelbach training reward
(sumrate - lambda_ee * normalized_power, EE_sac.py:330) instead of the
ratio, since the ratio isn't what the policy is trained to maximize.
That's why it still reads the real trained lambda_ee (see below) and
prints it for reference. But per the chat discussion this went through:
dividing the reward by power a second time to get per-Watt units double-
counts power in a non-standard way, whereas dividing each side's own
rate by its own power (the ratio) is the clean, dimensionally consistent
choice for a bps/Hz/W axis -- so that's what's actually plotted now.

lambda_ee is read from the ACTUAL training run instead of estimated:
EE_sac.py was instrumented (this session) to save dinkelbach_lambda_ee.txt
next to every checkpoint it saves, and the aod=0.0 checkpoint was
retrained specifically to capture this. Earlier versions of this script
had to back-derive lambda_ee (first from the deployed ~35 W point, which
produced an invalid ~9.15 that peaked at an arbitrary ~18 W; then from
the ratio's own peak via Dinkelbach's theorem) because the original
checkpoint's training log was lost.

IMPORTANT: this script's rate/power sweep data (from
ee_vs_power_budget_sweep_sac_error{X}.gzip) MUST come from the SAME
checkpoint as the lambda_ee it's paired with (checked below, prints a
warning if not). get_best_model_path is session-aware (picks the
best-scoring checkpoint from the most recent training session, see
plotting_scenario.py), so after retraining, re-run
ee_vs_power_budget_sweep_sac.slurm again FIRST to regenerate that gzip
against the new checkpoint before running this script.

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

    # Real trained lambda_ee is still read above and printed below for
    # reference/logging, but the plotted quantities here are the classical
    # ratio (rate / total power) for both curves -- not the Dinkelbach
    # reward -- per the chat discussion: dividing the reward by power a
    # second time double-counts power in a non-standard way, whereas
    # dividing each side's own rate by its own power is a clean, dimensionally
    # consistent bps/Hz/W quantity for both the EE curve and the RM baseline.
    ee_ratio_arr = mean_rate / total_power_watt_arr

    peak_idx = int(np.argmax(ee_ratio_arr))
    print(f'EE ratio (rate/total_power) peaks at: budget={budget_sweep_watt[peak_idx]:.2f} W, '
          f'achieved mean_power={mean_power[peak_idx]:.2f} W, '
          f'overall (total) power={total_power_watt_arr[peak_idx]:.2f} W, '
          f'EE={ee_ratio_arr[peak_idx]:.4f} bps/Hz/W, rate={mean_rate[peak_idx]:.4f} bps/Hz')
    print(f'At the full training-time budget (75 W): achieved mean_power={mean_power[-1]:.2f} W, '
          f'overall (total) power={total_power_watt_arr[-1]:.2f} W, '
          f'EE={ee_ratio_arr[-1]:.4f} bps/Hz/W, rate={mean_rate[-1]:.4f} bps/Hz')

    # Full-power ("RM") baseline: this checkpoint's own precoding direction,
    # always rescaled to the 75 W training budget -- a flat reference,
    # independent of the swept budget on the x-axis here. Its own ratio
    # (rate / its own total power) is likewise a flat constant.
    triplet_gzip = Path(cfg.output_metrics_path, 'EE_lwin5000_3gpp_triplet', 'rate_power_triplet.gzip')
    if not triplet_gzip.exists():
        raise FileNotFoundError(
            f'{triplet_gzip} not found -- run plotting_scenario.py first, needed here for the '
            f'full-power (RM) baseline reference line.'
        )
    with gzip.open(triplet_gzip, 'rb') as file:
        triplet_data = pickle.load(file)
    error_idx = int(np.argmin(np.abs(triplet_data['error_sweep_range'] - CSIT_ERROR_BOUND)))
    rm_full_power_rate = float(triplet_data['results']['sac_aod0.0_fullpower']['mean_rate'][error_idx])
    rm_full_power_power = float(triplet_data['results']['sac_aod0.0_fullpower']['mean_power'][error_idx])
    rm_ratio = rm_full_power_rate / total_power_watt(cfg, rm_full_power_power)
    print(f'Full-power (RM) baseline: rate={rm_full_power_rate:.4f} bps/Hz, '
          f'EE={rm_ratio:.4f} bps/Hz/W (flat, always at 75 W)')

    pdf_path = Path(plot_cfg.plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)

    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.62

    fig, ax = plt.subplots(figsize=(plot_width, plot_height))

    # Both curves are the classical ratio (rate / total power), each side
    # using its own rate and power -- true bps/Hz/W throughout, no mixing
    # of a power-penalized reward with a raw rate. The flat RM baseline is
    # a true constant (doesn't depend on the swept budget); the shaded gap
    # to the EE ratio curve is the "EE gets most of the value for far less
    # power" argument, now in consistent per-Watt units.
    line_rm, = ax.plot(
        budget_sweep_watt, np.full_like(budget_sweep_watt, rm_ratio),
        color=plot_cfg.cp2['blue'], linestyle='--', linewidth=1.5,
        label='RM baseline (always 75 W)',
    )
    line_ee, = ax.plot(
        budget_sweep_watt, ee_ratio_arr, color=plot_cfg.cp2['green'], marker='o', markersize=4,
        linewidth=1.5, label='EE (rate / total power)',
    )
    ax.fill_between(
        budget_sweep_watt, ee_ratio_arr, rm_ratio,
        color=plot_cfg.cp2['gold'], alpha=0.15, zorder=0,
    )
    vline = ax.axvline(budget_sweep_watt[peak_idx], color='gray', linestyle='-.', linewidth=1.3,
                        label=fr'$B^\star \approx {budget_sweep_watt[peak_idx]:.0f}$ W (EE peak)')

    ax.set_xlabel(r'Available power budget $B$ [W]', fontsize=13)
    ax.set_ylabel(r'EE [bits/s/Hz/W]', fontsize=13)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    handles = [line_rm, line_ee, vline]
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
