"""Plot error sweep (sum rate): MMSE vs SAC energy efficiency model."""
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False

import matplotlib.pyplot as plt
from pathlib import Path

from src.config.config import Config
from src.config.config_plotting import PlotConfig
from src.analysis.plotting.plot_error_sweep import plot_error_sweep_testing_graph

matplotlib.rcParams['text.usetex'] = False  # override PlotConfig


def save_pdf(plots_parent_path, plot_name, padding=0):
    from pathlib import Path
    pdf_path = Path(plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)
    out = Path(pdf_path, f'{plot_name}.pdf')
    plt.savefig(out, bbox_inches='tight', pad_inches=padding, dpi=800, transparent=True)
    print(f'Saved: {out}')


if __name__ == '__main__':
    cfg = Config()
    plot_cfg = PlotConfig()
    matplotlib.rcParams['text.usetex'] = False

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

    save_pdf(plots_parent_path=plot_cfg.plots_parent_path, plot_name='error_sweep_ee_vs_mmse_sumrate')
    plt.show()
