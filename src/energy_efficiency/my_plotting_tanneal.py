"""Plot error sweep (sum rate): the target_entropy-annealing checkpoint
(job 155683, 2026-07-31) vs. MMSE, the pre-fix (rollout-lambda, non-annealed)
3GPP nadir checkpoint, and all four detlam (deterministic-lambda) checkpoints
from the 2026-07-29 fix -- so the entropy-anneal approach can be judged
directly against both prior attempts at the over-transmission problem, not
just against MMSE/pre-fix in isolation. Run my_evaluation_tanneal.py first if
the tanneal gzip is missing (detlam/pre-fix/MMSE gzips already exist).
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
    plot_height = plot_width * 0.6

    # MMSE + pre-fix (rollout-lambda, non-annealed) SAC nadir: existing gzips, not re-run.
    rollout_nadir_folder = 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_satg30_p75_eta0.6_rawpow'
    # detlam checkpoints (2026-07-29 fix), aod0.025 excluded (diverged).
    detlam_aod0_folder = 'EE_dinkelbach_adaptive_detlam512_N16K3_satg30_p75_eta0.6_rawpow'
    detlam_aod05_folder = 'EE_dinkelbach_adaptive_aod0.05_detlam512_N16K3_satg30_p75_eta0.6_rawpow'
    detlam_aod0_fair15_folder = 'EE_dinkelbach_adaptive_detlam512_N16K3_satg30_p75_eta0.6_rawpow_fair1.5'
    detlam_aod05_fair15_folder = 'EE_dinkelbach_adaptive_aod0.05_detlam512_N16K3_satg30_p75_eta0.6_rawpow_fair1.5'
    # new: entropy-anneal checkpoint (job 155683)
    tanneal_folder = 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_satg30_p75_eta0.6_tanneal1to0_rawpow'

    data_paths = [
        Path(cfg.output_metrics_path, rollout_nadir_folder, 'error_sweep', 'testing_mmse_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path, rollout_nadir_folder, 'error_sweep', 'testing_learned_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path, detlam_aod0_folder, 'error_sweep', 'testing_learned_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path, detlam_aod05_folder, 'error_sweep', 'testing_learned_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path, detlam_aod0_fair15_folder, 'error_sweep', 'testing_learned_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path, detlam_aod05_fair15_folder, 'error_sweep', 'testing_learned_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path, tanneal_folder, 'error_sweep', 'testing_learned_sweep_0.0_0.1.gzip'),
    ]
    legend = [
        'MMSE nadir',
        'SAC nadir (pre-fix, rollout-lambda)',
        'SAC nadir (detlam, aod0.0)',
        'SAC nadir (detlam, aod0.05)',
        'SAC nadir (detlam, aod0.0, fair1.5)',
        'SAC nadir (detlam, aod0.05, fair1.5)',
        'SAC nadir (entropy-anneal, aod0.05)',
    ]
    colors = [
        plot_cfg.cp2['gold'],
        plot_cfg.cp2['magenta'],
        plot_cfg.cp2['green'],
        plot_cfg.cp2['blue'],
        plot_cfg.cp3['orange1'],
        plot_cfg.cp3['purple1'],
        '#d03b3b',
    ]
    markerstyles = ['v', 'x', 'o', 's', '^', 'd', 'P']
    linestyles = ['-', '--', '-', '-', '--', '--', '-']

    plot_error_sweep_testing_graph(
        paths=data_paths,
        metric='sumrate',
        name='error_sweep_tanneal_vs_detlam_prefix',
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
    ax.legend(
        legend,
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
        Path(plot_cfg.plots_parent_path, 'pdf', 'error_sweep_tanneal_vs_detlam_prefix_sumrate.pdf'),
        bbox_inches='tight', dpi=800, transparent=True,
    )
    print('Saved: error_sweep_tanneal_vs_detlam_prefix_sumrate.pdf')
