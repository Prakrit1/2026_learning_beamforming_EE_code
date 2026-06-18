"""Plot error sweep (sum rate): MMSE vs SAC (full_EE) vs SAC (simplified_EE)
   vs SAC (full_EE_without_normalization) vs SAC (simplified_EE_without_normalization)."""
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

# Patch save_figures used inside plot_error_sweep_testing_graph to skip pgf/eps (no LaTeX on server)
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

    # -----------------------------------------------------------------
    # Plot 1: normalized variants (existing) -> error_sweep_ee_vs_mmse
    # -----------------------------------------------------------------
    data_paths = [
        Path(cfg.output_metrics_path, 'full_EE', 'error_sweep', 'testing_mmse_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path, 'full_EE', 'error_sweep', 'testing_learned_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path, 'simplified_EE', 'error_sweep', 'testing_learned_sweep_0.0_0.1.gzip'),
    ]
    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.42
    plot_error_sweep_testing_graph(
        paths=data_paths,
        metric='sumrate',
        name='error_sweep_ee_vs_mmse',
        width=plot_width,
        height=plot_height,
        legend=['MMSE', 'SAC (full EE)', 'SAC (simplified EE)'],
        colors=[plot_cfg.cp2['gold'], plot_cfg.cp2['blue'], plot_cfg.cp2['magenta']],
        markerstyle=['v', 'o', 's'],
        linestyles=['-', '-', '-'],
        plots_parent_path=plot_cfg.plots_parent_path,
    )
    ax = plt.gca()
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    ax.legend(
        ['MMSE', 'SAC (full EE)', 'SAC (simplified EE)'],
        loc='upper right',
        ncols=1,
        fontsize=7,
        framealpha=1.0,
    )
    plt.tight_layout(pad=0.2)
    plt.savefig(
        Path(plot_cfg.plots_parent_path, 'pdf', 'error_sweep_ee_vs_mmse_sumrate.pdf'),
        bbox_inches='tight', dpi=800, transparent=True,
    )

    
