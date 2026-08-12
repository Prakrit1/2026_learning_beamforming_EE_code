"""
Base rate-vs-error-bound plotting code, shared across EE scenario scripts
(analogous to plot_beam_patterns.py for beampatterns): reads an already-
computed error-sweep results dict (produced by a scenario script such as
plotting_scenario.py) and draws one figure -- full-budget MMSE as context,
plus for each named checkpoint its own SAC curve against MMSE rescaled to
that checkpoint's own measured power at every error point (the "MMSE with
SAC powers" / matched-power comparison this project has used since
2026-07-28).

Scenario-agnostic: takes the checkpoint keys, colors, and results dict as
arguments rather than hardcoding a specific scenario's checkpoints.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt

from src.config.config_plotting import PlotConfig


def plot_rate_error_sweep(
        error_sweep_range,
        results: dict,
        aod_keys: list,
        color_names: dict,
        width,
        height,
        plots_parent_path,
        name: str,
) -> None:
    plot_cfg = PlotConfig()
    matplotlib.rcParams['text.usetex'] = False  # override PlotConfig's reset
    palette = {**plot_cfg.cp2, **plot_cfg.cp3}  # color_names draws from both

    fig, ax = plt.subplots(figsize=(width, height))

    mmse = results['mmse_nadir']
    ax.plot(error_sweep_range, mmse['mean_rate'], color=plot_cfg.cp2['gold'],
            marker='v', linestyle='-', linewidth=1.5, markersize=5, label='MMSE (full budget)')

    for aod_key in aod_keys:
        color = palette[color_names[aod_key]]
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

    pdf_path = Path(plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)
    out = Path(pdf_path, f'{name}.pdf')
    plt.savefig(out, bbox_inches='tight', dpi=800, transparent=True)
    plt.close(fig)
    print(f'Saved: {out}')
