"""Plot error sweep (sum rate): MMSE vs SAC energy efficiency model."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from src.config.config import Config
from src.config.config_plotting import PlotConfig
from src.analysis.plotting.plot_error_sweep import plot_error_sweep_testing_graph
import src.plotting.plot_error_sweep_testing_graph as _plot_module

# Patch save_figures in the plot module's own namespace (where it was imported to)
def _save_figures_no_latex(plots_parent_path, plot_name, padding):
    pdf_path = Path(plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(pdf_path, f'{plot_name}.pdf'), bbox_inches='tight', pad_inches=padding, dpi=800, transparent=True)
    print(f'Saved: {plot_name}.pdf')

_plot_module.save_figures = _save_figures_no_latex  # patch where it actually lives

if __name__ == '__main__':
    cfg = Config()
    plot_cfg = PlotConfig()
    plt.rc('text', usetex=False)  # override PlotConfig's usetex=True

    data_paths = [
        Path(cfg.output_metrics_path, 'test', 'error_sweep', 'testing_mmse_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path, 'test', 'error_sweep', 'testing_learned_sweep_0.0_0.1.gzip'),
    ]

    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.42

    plot_error_sweep_testing_graph(
        paths=data_paths,
        metric='sumrate',
        name='error_sweep_ee_vs_mmse',
        width=plot_width,
        height=plot_height,
        legend=['MMSE', 'SAC (EE)'],
        colors=[plot_cfg.cp2['gold'], plot_cfg.cp2['blue']],
        markerstyle=['v', 'o'],
        linestyles=['-', '-'],
        plots_parent_path=plot_cfg.plots_parent_path,
    )
