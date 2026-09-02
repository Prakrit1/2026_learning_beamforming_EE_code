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
        annotate_power: bool = False,
        legend_ncols: int = 1,
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

    annotate_power: if True, label each curve at its error=0 point with its
    measured power (W and % of budget), reading 'mean_power'/'power_budget'
    from that curve's results entry -- opt-in since not every figure using
    this function wants it (e.g. the MMSE/SAC full-vs-matched-power figure
    already encodes power in which curves are drawn, not via annotation).

    legend_ncols: number of legend columns (default 1); use 2 for a compact
    box when curve labels are short, matching the reference paper's Fig. 3 style.
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
        if annotate_power and 'mean_power' in series and 'power_budget' in series:
            power_watt = series['mean_power'][0]
            power_pct = 100 * power_watt / series['power_budget']
            ax.annotate(
                f'{power_watt:.1f} W ({power_pct:.0f}%)',
                xy=(error_sweep_range[0], series['mean_rate'][0]),
                xytext=(6, 6), textcoords='offset points',
                fontsize=7, color=curve['color'], fontweight='bold',
            )

    ax.set_xlabel('Error Bound (Δε)')
    ax.set_ylabel('Rate R [bps/Hz]')
    ax.grid(True, alpha=0.5, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(
        loc='upper right',
        ncols=legend_ncols,
        fontsize=9,
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

    # PNG with a transparent background, for dropping the figure onto other
    # content (slides/docs) where the background needs to show through --
    # JPG can't do this (no alpha channel), PNG can.
    png_path = Path(plots_parent_path, 'png')
    png_path.mkdir(parents=True, exist_ok=True)
    out_png = Path(png_path, f'{name}.png')
    plt.savefig(out_png, bbox_inches='tight', dpi=200, transparent=True)
    print(f'Saved: {out_png}')

    plt.close(fig)
