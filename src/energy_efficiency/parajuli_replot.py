"""
Replot of parajuli.pdf (see parajuli_matched_power_check.py for the Monte
Carlo generation) with two user-requested style changes:
  1. smaller legend
  2. wider figure (x-axis stretched out) for easier reading

Pure replot -- reads the already-computed gzips, no Monte Carlo recompute.
"""
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False

from pathlib import Path

import matplotlib.pyplot as plt

from src.config.config import Config
from src.config.config_plotting import PlotConfig
import src.plotting.plot_error_sweep_testing_graph as plot_module
from src.plotting.plot_error_sweep_testing_graph import plot_error_sweep_testing_graph

CLASSICAL_TRAINING_NAME = 'classical_baselines_N16K3_eta0.6'
SAC_AOD0_TRAINING_NAME = 'EE_dinkelbach_adaptive_lwin5000_N16K3_eta0.6_rawpow'
SAC_AOD05_TRAINING_NAME = 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_eta0.6_rawpow'
SAC_POWER_AOD0_WATT = 55.94  # from parajuli_matched_power_check.py's fresh measurement


def _save_figures_pdf_only(plots_parent_path, plot_name, padding=0):
    pdf_path = Path(plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)
    out = Path(pdf_path, f'{plot_name}.pdf')
    plt.savefig(out, bbox_inches='tight', pad_inches=padding, dpi=800, transparent=True)
    print(f'Saved: {out}')


plot_module.save_figures = _save_figures_pdf_only


if __name__ == '__main__':
    cfg = Config()
    cfg.show_plots = False

    plot_cfg = PlotConfig()
    matplotlib.rcParams['text.usetex'] = False  # PlotConfig() resets this on construction

    # wider + flatter than the original 0.99*textwidth square-ish figure --
    # stretches the x-axis (11 error points) out so points aren't crowded
    plot_width = 2.4 * plot_cfg.textwidth
    plot_height = plot_width * 0.42

    data_paths = [
        Path(cfg.output_metrics_path, CLASSICAL_TRAINING_NAME, 'error_sweep', 'testing_mmse_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path, CLASSICAL_TRAINING_NAME, 'error_sweep', 'testing_mmse_matched_parajuli_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path, SAC_AOD0_TRAINING_NAME, 'error_sweep', 'testing_learned_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path, SAC_AOD05_TRAINING_NAME, 'error_sweep', 'testing_learned_sweep_0.0_0.1.gzip'),
    ]
    legend = [
        'MMSE (100W)',
        f'MMSE ({SAC_POWER_AOD0_WATT:.1f}W, matched to SAC err=0.0)',
        'SAC (error=0.0)',
        'SAC (error=0.05)',
    ]
    colors = [
        plot_cfg.cp2['gold'],
        '#8a7000',
        '#307b3b',
        '#1baf7a',
    ]
    markerstyles = ['v', '^', 'o', 's']
    linestyles = [':', '-.', '-', '--']

    plot_error_sweep_testing_graph(
        paths=data_paths,
        metric='sumrate',
        name='parajuli',
        width=plot_width,
        height=plot_height,
        legend=legend,
        colors=colors,
        markerstyle=markerstyles,
        linestyles=linestyles,
        plots_parent_path=plot_cfg.plots_parent_path,
    )
    ax = plt.gca()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    # smaller legend: fontsize 10 -> 7, tighter padding/spacing
    ax.legend(
        legend, loc='upper right', ncols=1, fontsize=7,
        framealpha=1.0, frameon=True, handlelength=1.5,
        labelspacing=0.3, borderpad=0.4,
    )
    ax.set_xlabel('Error Bound')
    ax.set_ylabel('Rate R [bps/Hz]')
    plt.tight_layout(pad=0.2)

    pdf_out = Path(plot_cfg.plots_parent_path, 'pdf', 'parajuli.pdf')
    plt.savefig(pdf_out, bbox_inches='tight', dpi=800, transparent=True)
    print(f'Saved: {pdf_out}')
