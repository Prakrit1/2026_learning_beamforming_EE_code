"""Plot error sweep (sum rate): all models in one figure."""
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
    # All models in one figure
    #
    # NOTE: the MMSE sweep gzip is duplicated per training_name folder by
    # the evaluation script (since test_mmse_precoder_error_sweep is called
    # once per config/training_name in the eval script). Both MMSE files
    # below should contain numerically identical results (MMSE doesn't
    # depend on which SAC reward was used) -- only one MMSE curve is plotted
    # here, taken from the 'energy_efficiency' folder. If you want to sanity
    # check the two MMSE runs agree, diff the two gzip files directly rather
    # than re-plotting both.
    #
    # 'energy_efficiency' here is Scheme I -- the paper's always-rescale-to-
    # budget normalization (Eq. 26 style). Total transmit power for this
    # model is, by construction, pinned at the power budget for every
    # sample; this plot shows its RATE behaviour only, not power.
    #
    # 'energy_efficiency_no_normalization_fixed' is Scheme II/III -- clip-
    # only, genuine inequality constraint, free to transmit below budget.
    # Whether it actually does so is answered by the tx_power_histogram
    # script, not by this rate-vs-error plot.
    #
    # NOTE: training_name strings below ('energy_efficiency',
    # 'energy_efficiency_no_normalization_fixed') must match exactly what
    # was used in the corresponding evaluation script run -- these are the
    # same folder names used there.
    # -----------------------------------------------------------------
    data_paths = [
        Path(cfg.output_metrics_path, 'full_EE', 'error_sweep', 'testing_mmse_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path, 'full_EE', 'error_sweep', 'testing_learned_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path, 'Energy_efficiency_without_normalization_fix', 'error_sweep', 'testing_learned_sweep_0.0_0.1.gzip'),
    ]
    legend = [
        'MMSE',
        'SAC (EE, paper normalization)',
        'SAC (EE, no normalization fix)',
    ]
    colors = [
        plot_cfg.cp2['gold'],
        plot_cfg.cp2['blue'],
        plot_cfg.cp2['green'],
    ]
    markerstyles = ['v', 'o', '^']
    linestyles = ['-', '-', '--']
    plot_error_sweep_testing_graph(
        paths=data_paths,
        metric='sumrate',
        name='error_sweep_normalization_comparison',
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
        fontsize=7,
        framealpha=1.0,
        frameon=True,
    )
    ax.set_xlabel('Error Bound')
    ax.set_ylabel('Rate R [bps/Hz]')
    plt.tight_layout(pad=0.2)
    plt.savefig(
        Path(plot_cfg.plots_parent_path, 'pdf', 'error_sweep_normalization_comparison_sumrate.pdf'),
        bbox_inches='tight', dpi=800, transparent=True,
    )
    print('Saved: error_sweep_normalization_comparison_sumrate.pdf')
