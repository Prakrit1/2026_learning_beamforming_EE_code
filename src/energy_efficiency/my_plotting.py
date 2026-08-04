"""Plot error sweep (sum rate): current 3GPP models + previous-result benchmark.

Trimmed 2026-07-28 (user request: only the curves needed to understand the
current results, plus a benchmark of the previous result). The removed
figure blocks (Scheme-I-vs-Dinkelbach normalization comparison, N8K2
reproduction, tuned-LR-vs-default) are in git history if ever needed again.
"""
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
from pathlib import Path
from src.config.config import Config
from src.config.config_plotting import PlotConfig
import src.plotting.plot_error_sweep_testing_graph as plot_module
from src.plotting.plot_error_sweep_testing_graph import plot_error_sweep_testing_graph
matplotlib.rcParams['text.usetex'] = False  # override PlotConfig
def _save_figures_pdf_only(plots_parent_path, plot_name, padding=0):
    pdf_path = Path(plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)
    out = Path(pdf_path, f'{plot_name}.pdf')
    plt.savefig(out, bbox_inches='tight', pad_inches=padding, dpi=800, transparent=True)
    print(f'Saved: {out}')
plot_module.save_figures = _save_figures_pdf_only
if __name__ == '__main__':
    cfg = Config()
    plot_cfg = PlotConfig()
    matplotlib.rcParams['text.usetex'] = False
    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.6  # slightly taller to fit legend

    # -----------------------------------------------------------------
    # CURRENT RESULT: 3GPP Set-1 nadir, error=0.0 training bound (job
    # 156358, lwin5000, natively trained at 30 dBi/75 W) vs the existing
    # elev30 model (job 153656, still error=0.05 -- no aod=0.0 elev30
    # checkpoint exists) vs MMSE evaluated in each geometry at the same
    # system parameters, plus MMSE rescaled to the nadir SAC's own
    # measured power ("eq.pow", ~28.6-34.9 W of 75 W). Gzips produced by
    # my_evaluation.py's matching blocks -- run that first if these files
    # are missing.
    # -----------------------------------------------------------------
    gpp_nadir_folder = 'EE_dinkelbach_adaptive_lwin5000_N16K3_satg30_p75_eta0.6_rawpow'
    gpp_elev30_folder = 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_satg30_p75_elev30_eta0.6_rawpow'
    gpp_data_paths = [
        Path(cfg.output_metrics_path, gpp_nadir_folder, 'error_sweep', 'testing_mmse_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path, gpp_nadir_folder, 'error_sweep', 'testing_mmse_sacpower_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path, gpp_nadir_folder, 'error_sweep', 'testing_learned_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path, gpp_elev30_folder, 'error_sweep', 'testing_mmse_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path, gpp_elev30_folder, 'error_sweep', 'testing_learned_sweep_0.0_0.1.gzip'),
    ]
    # 'MMSE nadir eq.pow' = MMSE rescaled per error level to the transmit
    # power SAC nadir (train err 0.0) actually used -- the equal-power
    # comparison; plain 'MMSE nadir' spends the full 75 W budget.
    gpp_legend = [
        'MMSE nadir',
        'MMSE nadir eq.pow',
        'SAC nadir (train err 0.0)',
        'MMSE elev30',
        'SAC elev30 (train err 0.05)',
    ]
    gpp_colors = [
        plot_cfg.cp2['gold'],
        plot_cfg.cp2['magenta'],
        plot_cfg.cp2['green'],
        plot_cfg.cp2['gold'],
        plot_cfg.cp2['blue'],
    ]
    gpp_markerstyles = ['v', 'x', 'o', 'v', '^']
    gpp_linestyles = ['-', '-', '-', '--', '--']
    plot_error_sweep_testing_graph(
        paths=gpp_data_paths,
        metric='sumrate',
        name='error_sweep_3gpp_models',
        width=plot_width,
        height=plot_height,
        legend=gpp_legend,
        colors=gpp_colors,
        markerstyle=gpp_markerstyles,
        linestyles=gpp_linestyles,
        plots_parent_path=plot_cfg.plots_parent_path,
    )
    ax = plt.gca()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.legend(
        gpp_legend,
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
    ax.set_xlabel('Error Bound')
    ax.set_ylabel('Rate R [bps/Hz]')
    plt.tight_layout(pad=0.2)
    plt.savefig(
        Path(plot_cfg.plots_parent_path, 'pdf', 'error_sweep_3gpp_models_sumrate.pdf'),
        bbox_inches='tight', dpi=800, transparent=True,
    )
    print('Saved: error_sweep_3gpp_models_sumrate.pdf')

    # -----------------------------------------------------------------
    # BENCHMARK (previous result, old system: 20 dBi / 100 W): the
    # raw-power-fixed Dinkelbach checkpoints trained at error 0.0/0.025/
    # 0.05 vs MMSE, all evaluated in THAT system -- plotted from the
    # existing gzips on disk (generated pre-2026-07-27 in the old-system
    # configuration; NOT re-simulated). Kept as the reference the new
    # 3GPP curves are judged against.
    # -----------------------------------------------------------------
    error_bound_data_paths = [
        Path(cfg.output_metrics_path, 'full_EE_aod0.5_N16K3_eta0.6', 'error_sweep', 'testing_mmse_sweep_0.0_0.5.gzip'),
        Path(cfg.output_metrics_path, 'EE_dinkelbach_adaptive_lwin5000_N16K3_eta0.6_rawpow', 'error_sweep', 'testing_learned_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path, 'EE_dinkelbach_adaptive_aod0.025_lwin5000_N16K3_eta0.6_rawpow', 'error_sweep', 'testing_learned_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path, 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_eta0.6_rawpow', 'error_sweep', 'testing_learned_sweep_0.0_0.1.gzip'),
    ]
    # short labels ("old system" context lives in the figure name, not every
    # legend row) + compact legend, same reasoning as the 3GPP figure above
    error_bound_legend = [
        'MMSE',
        'SAC err=0.0',
        'SAC err=0.025',
        'SAC err=0.05',
    ]
    error_bound_colors = [
        plot_cfg.cp2['gold'],
        plot_cfg.cp2['green'],
        plot_cfg.cp2['blue'],
        plot_cfg.cp2['magenta'],
    ]
    error_bound_markerstyles = ['v', 'o', '^', 's']
    error_bound_linestyles = ['-', '-', '-.', '--']
    plot_error_sweep_testing_graph(
        paths=error_bound_data_paths,
        metric='sumrate',
        name='error_sweep_benchmark_old_system',
        width=plot_width,
        height=plot_height,
        legend=error_bound_legend,
        colors=error_bound_colors,
        markerstyle=error_bound_markerstyles,
        linestyles=error_bound_linestyles,
        plots_parent_path=plot_cfg.plots_parent_path,
    )
    ax = plt.gca()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.legend(
        error_bound_legend,
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
    ax.set_xlabel('Error Bound')
    ax.set_ylabel('Rate R [bps/Hz]')
    # zoom to the range the SAC checkpoints were evaluated over (the MMSE
    # reference gzip continues to 0.5, out of frame -- expected, not a bug)
    ax.set_xlim(-0.005, 0.105)
    plt.tight_layout(pad=0.2)
    plt.savefig(
        Path(plot_cfg.plots_parent_path, 'pdf', 'error_sweep_benchmark_old_system_sumrate.pdf'),
        bbox_inches='tight', dpi=800, transparent=True,
    )
    print('Saved: error_sweep_benchmark_old_system_sumrate.pdf')
