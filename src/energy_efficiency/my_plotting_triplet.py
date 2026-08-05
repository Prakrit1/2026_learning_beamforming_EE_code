"""
Plot the lwin5000/3GPP-native nadir triplet (aod=0.0/0.025/0.05) against
MMSE, using the gzip produced by rate_power_triplet.py (run that first).

Two figures:
  1. error_sweep_triplet_matched_power_sumrate.pdf -- the "parajuli.pdf"
     replacement called for in the handoff's "what to do next" item 1: a
     compact-legend rate-vs-error figure with full-budget MMSE as context
     plus, for each of the three checkpoints, its own SAC curve against
     MMSE rescaled to that checkpoint's own measured power (the fair
     equal-power comparison this project has used since 2026-07-28).
  2. rate_power_triplet.pdf -- 3-panel diagnostic (rate vs error, power vs
     error as % of budget, rate-vs-power trajectory), same layout as
     rate_power_3gpp_pilots.py, extended to all three checkpoints.
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

AOD_KEYS = ['aod0.0', 'aod0.025', 'aod0.05']
AOD_COLOR_NAMES = {'aod0.0': 'green', 'aod0.025': 'blue', 'aod0.05': 'magenta'}


if __name__ == '__main__':
    cfg = Config()
    plot_cfg = PlotConfig()
    matplotlib.rcParams['text.usetex'] = False  # override PlotConfig's reset

    gzip_path = Path(cfg.output_metrics_path, 'EE_lwin5000_3gpp_triplet', 'rate_power_triplet.gzip')
    with gzip.open(gzip_path, 'rb') as file:
        data = pickle.load(file)
    error_sweep_range = data['error_sweep_range']
    results = data['results']

    pdf_path = Path(plot_cfg.plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)

    # =====================================================================
    # Figure 1: compact matched-power comparison (parajuli.pdf replacement)
    # =====================================================================
    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.6

    fig, ax = plt.subplots(figsize=(plot_width, plot_height))

    mmse = results['mmse_nadir']
    ax.plot(error_sweep_range, mmse['mean_rate'], color=plot_cfg.cp2['gold'],
            marker='v', linestyle='-', linewidth=1.5, markersize=5, label='MMSE (full budget)')

    for aod_key in AOD_KEYS:
        color = plot_cfg.cp2[AOD_COLOR_NAMES[aod_key]]
        sac = results[f'sac_{aod_key}']
        matched = results[f'mmse_matched_{aod_key}']
        err_label = aod_key.replace('aod', '')
        ax.plot(error_sweep_range, sac['mean_rate'], color=color, marker='o',
                linestyle='-', linewidth=1.5, markersize=5, label=f'SAC (train err {err_label})')
        ax.plot(error_sweep_range, matched['mean_rate'], color=color, marker='x',
                linestyle='--', linewidth=1.2, markersize=5, label=f'MMSE eq.pow ({err_label})')

    ax.set_xlabel('Error Bound')
    ax.set_ylabel('Rate R [bps/Hz]')
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(
        loc='upper right',
        ncols=1,
        fontsize=6,
        framealpha=0.9,
        frameon=True,
        handlelength=1.6,
        labelspacing=0.3,
        borderpad=0.3,
        handletextpad=0.5,
    )
    plt.tight_layout(pad=0.2)
    out1 = Path(pdf_path, 'error_sweep_triplet_matched_power_sumrate.pdf')
    plt.savefig(out1, bbox_inches='tight', dpi=800, transparent=True)
    plt.close(fig)
    print(f'Saved: {out1}')

    # =====================================================================
    # Figure 2: 3-panel diagnostic (rate / power / rate-vs-power)
    # =====================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    plot_series = {'mmse_nadir': mmse}
    colors = {'mmse_nadir': plot_cfg.cp2['gold']}
    for aod_key in AOD_KEYS:
        plot_series[f'sac_{aod_key}'] = results[f'sac_{aod_key}']
        colors[f'sac_{aod_key}'] = plot_cfg.cp2[AOD_COLOR_NAMES[aod_key]]

    # ---- Panel 1: rate vs error ----
    ax0 = axes[0]
    for key, series in plot_series.items():
        ax0.errorbar(error_sweep_range, series['mean_rate'], yerr=series['std_rate'], marker='o',
                     color=colors[key], ecolor=colors[key], elinewidth=1, capsize=3, linewidth=1.5,
                     markersize=5, label=series['label'])
    ax0.set_xlabel('Error Bound')
    ax0.set_ylabel('Rate R [bps/Hz]')
    ax0.set_title('Rate vs. CSIT error', fontsize=11)
    ax0.grid(True, alpha=0.25, linewidth=0.5)
    ax0.set_axisbelow(True)
    ax0.legend(fontsize=8, loc='best')

    # ---- Panel 2: power (% of own budget) vs error ----
    ax1 = axes[1]
    for key, series in plot_series.items():
        power_pct = 100 * series['mean_power'] / series['power_budget']
        power_pct_std = 100 * series['std_power'] / series['power_budget']
        ax1.errorbar(error_sweep_range, power_pct, yerr=power_pct_std, marker='o',
                     color=colors[key], ecolor=colors[key], elinewidth=1, capsize=3, linewidth=1.5,
                     markersize=5, label=series['label'])
    ax1.axhline(100, color='#d03b3b', linestyle='--', linewidth=1.5, label='Power budget')
    ax1.set_xlabel('Error Bound')
    ax1.set_ylabel('Mean transmit power [% of own budget]')
    ax1.set_title('Power vs. CSIT error', fontsize=11)
    ax1.grid(True, alpha=0.25, linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=8, loc='best')

    # ---- Panel 3: rate vs power (% of own budget), error as trajectory parameter ----
    ax2 = axes[2]
    for key, series in plot_series.items():
        power_pct = 100 * series['mean_power'] / series['power_budget']
        ax2.plot(power_pct, series['mean_rate'], color=colors[key], linewidth=1, zorder=1)
        ax2.scatter(power_pct, series['mean_rate'], color=colors[key], s=60, zorder=5,
                    edgecolor='white', linewidth=0.6, label=series['label'])
        for x, y, e in zip(power_pct, series['mean_rate'], error_sweep_range):
            ax2.annotate(f'{e:.2f}', (x, y), textcoords='offset points', xytext=(5, 4), fontsize=6.5)
    ax2.axvline(100, color='#d03b3b', linestyle='--', linewidth=1.2, label='Power budget')
    ax2.set_xlabel('Mean transmit power [% of own budget]')
    ax2.set_ylabel('Rate R [bps/Hz]')
    ax2.set_title('Rate vs. power (labels = error level)', fontsize=11)
    ax2.grid(True, alpha=0.25, linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=8, loc='best')

    fig.suptitle('lwin5000/3GPP Set-1 nadir triplet (30 dBi, 75 W) vs. MMSE', fontsize=13, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out2 = Path(pdf_path, 'rate_power_triplet.pdf')
    fig.savefig(out2, bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)
    print(f'Saved: {out2}')
