"""
Plot the raw-ratio-reward-vs-Dinkelbach-adaptive-lambda ablation, using the
gzip produced by ratio_reward_ablation_eval.py (run that first).

Three figures, sharing the same aod-key/color/matched-vs-standalone
structure (MMSE full budget as context, aod=0.0/aod=0.05 as matched
ratio-reward-vs-lambda pairs -- same color, solid/dashed -- and aod=0.035
standalone, no matched lambda checkpoint exists at that error bound):
  1. error_sweep_ratio_reward_ablation_sumrate.pdf -- rate vs. error.
  2. error_sweep_ratio_reward_ablation_power.pdf -- transmit power (% of own
     budget) vs. error. The ratio-reward checkpoints were trained with the
     always-rescale-to-budget precoder (see the eval script's docstring), so
     their curves are expected to sit at ~100% throughout -- this figure
     confirms that empirically and shows how much less power the lambda
     checkpoints draw by comparison.
  3. error_sweep_ratio_reward_ablation_ee.pdf -- energy efficiency
     (bps/Hz/W) vs. error, EE = mean_rate / total_power_watt using the same
     total_power_watt = transmit_power/pa_efficiency + circuit_power formula
     as eq:ptotal / EE_sac.py's Dinkelbach branch / ee_vs_transmit_power_sweep.py.
     Also plots each ratio-reward checkpoint's own matched-power MMSE
     counterpart (thin dotted) -- the fair "does SAC's beamforming direction
     itself buy more EE than MMSE at the same power" comparison.

Console summary: rate delta (ratio-reward minus lambda) per matched aod
bound, across the whole error sweep -- does raw-ratio-reward come anywhere
close to the working-line Dinkelbach mechanism, or does the pricing term
matter? Plus each checkpoint's mean power (% of budget) per error point.
"""
import gzip
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt

from src.config.config import Config
from src.config.config_plotting import PlotConfig

MATCHED_AOD_KEYS = ['aod0.0', 'aod0.05']
STANDALONE_AOD_KEYS = ['aod0.035']
COLOR_NAMES = {'aod0.0': 'blue1', 'aod0.05': 'green1', 'aod0.035': 'orange1'}


def total_power_watt(cfg, transmit_power_watt):
    """Same formula as eq:ptotal / EE_sac.py's Dinkelbach branch / ee_vs_transmit_power_sweep.py."""
    return transmit_power_watt / cfg.pa_efficiency + cfg.sat_nr * cfg.sat_ant_nr * cfg.circuit_power_watt


def plot_matched_and_standalone(ax, error_sweep_range, results, palette, plot_cfg, get_y):
    """Shared MMSE-context + matched-pair + standalone-curve plotting loop,
    used identically for the rate/power/EE figures below (only get_y differs)."""
    mmse = results['mmse_nadir']
    ax.plot(error_sweep_range, get_y(mmse), color=plot_cfg.cp2['gold'],
            marker='v', linestyle='-', linewidth=1.5, markersize=5, label='MMSE (full budget)')

    for aod_key in MATCHED_AOD_KEYS:
        color = palette[COLOR_NAMES[aod_key]]
        ratio = results[f'sac_ratio_{aod_key}']
        lambda_sac = results[f'sac_lambda_{aod_key}']
        ax.plot(error_sweep_range, get_y(ratio), color=color, marker='o',
                linestyle='-', linewidth=1.5, markersize=5, label=ratio['label'])
        ax.plot(error_sweep_range, get_y(lambda_sac), color=color, marker='s',
                linestyle='--', linewidth=1.2, markersize=5, label=lambda_sac['label'])

    for aod_key in STANDALONE_AOD_KEYS:
        color = palette[COLOR_NAMES[aod_key]]
        ratio = results[f'sac_ratio_{aod_key}']
        ax.plot(error_sweep_range, get_y(ratio), color=color, marker='o',
                linestyle='-', linewidth=1.5, markersize=5,
                label=f"{ratio['label']} (no matched lambda ckpt)")


def finish_and_save(fig, ax, ylabel, out_path):
    ax.set_xlabel('Error Bound')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(
        loc='best', ncols=1, fontsize=6, framealpha=0.9, frameon=True,
        handlelength=1.6, labelspacing=0.3, borderpad=0.3, handletextpad=0.5,
    )
    plt.tight_layout(pad=0.2)
    plt.savefig(out_path, bbox_inches='tight', dpi=800, transparent=True)
    plt.close(fig)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    cfg = Config()
    plot_cfg = PlotConfig()
    matplotlib.rcParams['text.usetex'] = False  # override PlotConfig's reset
    palette = {**plot_cfg.cp2, **plot_cfg.cp3}  # COLOR_NAMES draws from both

    gzip_path = Path(cfg.output_metrics_path, 'EE_ratio_reward_ablation', 'ratio_reward_ablation.gzip')
    with gzip.open(gzip_path, 'rb') as file:
        data = pickle.load(file)
    error_sweep_range = data['error_sweep_range']
    results = data['results']

    pdf_path = Path(plot_cfg.plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)

    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.6

    # =========================================================================
    # Figure 1: rate vs. error
    # =========================================================================
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    plot_matched_and_standalone(ax, error_sweep_range, results, palette, plot_cfg, get_y=lambda r: r['mean_rate'])
    finish_and_save(fig, ax, 'Rate R [bps/Hz]', Path(pdf_path, 'error_sweep_ratio_reward_ablation_sumrate.pdf'))

    # =========================================================================
    # Figure 2: transmit power (% of own budget) vs. error
    # =========================================================================
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    plot_matched_and_standalone(
        ax, error_sweep_range, results, palette, plot_cfg,
        get_y=lambda r: 100 * r['mean_power'] / r['power_budget'],
    )
    ax.axhline(100, color='#d03b3b', linestyle=':', linewidth=1.3, label='Power budget')
    finish_and_save(
        fig, ax, 'Mean transmit power [% of own budget]',
        Path(pdf_path, 'error_sweep_ratio_reward_ablation_power.pdf'),
    )

    # =========================================================================
    # Figure 3: energy efficiency (bps/Hz/W) vs. error
    # =========================================================================
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    plot_matched_and_standalone(
        ax, error_sweep_range, results, palette, plot_cfg,
        get_y=lambda r: r['mean_rate'] / total_power_watt(cfg, r['mean_power']),
    )
    for aod_key in MATCHED_AOD_KEYS + STANDALONE_AOD_KEYS:
        color = palette[COLOR_NAMES[aod_key]]
        matched_mmse = results[f'mmse_matched_ratio_{aod_key}']
        ee = matched_mmse['mean_rate'] / total_power_watt(cfg, matched_mmse['mean_power'])
        ax.plot(error_sweep_range, ee, color=color, marker='x', linestyle=':', linewidth=1.0,
                markersize=4, label=matched_mmse['label'])
    finish_and_save(fig, ax, 'EE [bps/Hz/W]', Path(pdf_path, 'error_sweep_ratio_reward_ablation_ee.pdf'))

    # ---- console summary: how far off is raw-ratio-reward, where matched? ---
    for aod_key in MATCHED_AOD_KEYS:
        delta = results[f'sac_ratio_{aod_key}']['mean_rate'] - results[f'sac_lambda_{aod_key}']['mean_rate']
        print(f'[{aod_key}] rate delta ratio-reward minus lambda (per error point, bps/Hz): '
              f'{", ".join(f"{d:+.3f}" for d in delta)} '
              f'(mean {delta.mean():+.3f})')
    print('[aod0.035] no matched lambda checkpoint -- evaluated standalone, see gzip for raw numbers.')

    # ---- console summary: how much power is each checkpoint actually using? ----
    for aod_key in MATCHED_AOD_KEYS + STANDALONE_AOD_KEYS:
        ratio = results[f'sac_ratio_{aod_key}']
        ratio_power_pct = 100 * ratio['mean_power'] / ratio['power_budget']
        print(f'[ratio_{aod_key}] mean power (% of budget) per error point: '
              f'{", ".join(f"{p:.1f}" for p in ratio_power_pct)}')
    for aod_key in MATCHED_AOD_KEYS:
        lambda_sac = results[f'sac_lambda_{aod_key}']
        lambda_power_pct = 100 * lambda_sac['mean_power'] / lambda_sac['power_budget']
        print(f'[lambda_{aod_key}] mean power (% of budget) per error point: '
              f'{", ".join(f"{p:.1f}" for p in lambda_power_pct)}')
