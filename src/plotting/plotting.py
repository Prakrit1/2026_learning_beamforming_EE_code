"""
Base rate-vs-error-bound plotting code, shared across EE scenario scripts
(analogous to plot_beam_patterns.py for beampatterns): reads an already-
computed error-sweep results dict (produced by a scenario script such as
plotting_scenario.py) and draws one rate-vs-error figure from an explicit
list of curves.

Scenario-agnostic: the caller decides exactly which result keys to draw,
in what order, with what label/color/marker/linestyle -- this file has no
hardcoded assumption about which combination of MMSE/SAC curves belongs on
a given figure.
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
        curves: list,
        width,
        height,
        plots_parent_path,
        name: str,
) -> None:
    """
    curves: list of dicts, each describing one line on the figure:
        {
            'result_key': key into `results` (e.g. 'sac_aod0.0', 'mmse_matched_aod0.0'),
            'label': legend label,
            'color': matplotlib color,
            'marker': marker style (default 'o'),
            'linestyle': line style (default '-'),
        }
    """
    matplotlib.rcParams['text.usetex'] = False  # override PlotConfig's reset

    fig, ax = plt.subplots(figsize=(width, height))

    for curve in curves:
        series = results[curve['result_key']]
        ax.plot(
            error_sweep_range, series['mean_rate'],
            color=curve['color'],
            marker=curve.get('marker', 'o'),
            linestyle=curve.get('linestyle', '-'),
            linewidth=1.5, markersize=5,
            label=curve['label'],
        )

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
    print(f'Saved: {out}')

    # JPG alongside the PDF for quick viewing/sharing -- PDF stays the one
    # referenced from the LaTeX paper (\includegraphics{...pdf}), this is
    # not a replacement for it. No transparent=True: JPG has no alpha
    # channel, matplotlib just fills with white.
    jpg_path = Path(plots_parent_path, 'jpg')
    jpg_path.mkdir(parents=True, exist_ok=True)
    out_jpg = Path(jpg_path, f'{name}.jpg')
    plt.savefig(out_jpg, bbox_inches='tight', dpi=200)
    print(f'Saved: {out_jpg}')

    plt.close(fig)
