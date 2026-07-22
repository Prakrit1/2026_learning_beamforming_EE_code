"""
Rate vs. CSIT error for MMSE, MRT, and ZF (all three classical baselines,
not just MMSE), on the SAME error_sweep_range as the two active SAC
checkpoints (np.linspace(0, 0.10, 11), matching the paper's own evaluation
range) -- combined into a single 6-curve comparison plot: MMSE (100W), MRT
(100W), ZF (100W), MMSE (56W, power-matched to SAC), SAC (error=0.0-trained),
SAC (error=0.05-trained).

MMSE (56W) reruns the same sweep with cfg.mmse_args['power_constraint_watt']
overridden to 56 W (what SAC error=0.0 actually uses at zero error) instead
of the full 100 W budget -- the fair, power-matched comparison across the
whole error range, not just the zero-error point (see
classical_baselines_zero_error.py for that single-point version).

Uses test_precoder_error_sweep directly (not the higher-level
test_mmse_precoder_error_sweep wrapper) for all classical precoders, so
they go through the identical code path and are saved in a consistent,
directly comparable way.
"""
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config.config import Config
from src.config.config_plotting import PlotConfig
from src.analysis.helpers.test_precoder_error_sweep import test_precoder_error_sweep
from src.data.calc_sum_rate import calc_sum_rate
from src.utils.get_precoding import get_precoding_mmse, get_precoding_mrc, get_precoding_zero_forcing
import src.plotting.plot_error_sweep_testing_graph as plot_module
from src.plotting.plot_error_sweep_testing_graph import plot_error_sweep_testing_graph

# same range as my_evaluation.py / rate_power_error_sweep.py for the two
# active SAC checkpoints -- keeps all 5 curves on the same x-axis resolution
error_sweep_range = np.linspace(0, 0.10, 11)
monte_carlo_iterations = 10000

CLASSICAL_TRAINING_NAME = 'classical_baselines_N16K3_eta0.6'


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
    cfg.config_learner.training_name = CLASSICAL_TRAINING_NAME

    classical_precoders = [
        (get_precoding_mmse, 'mmse'),
        (get_precoding_mrc, 'mrt'),
        (get_precoding_zero_forcing, 'zf'),
    ]

    for get_precoder_func, precoder_name in classical_precoders:
        test_precoder_error_sweep(
            config=cfg,
            error_sweep_parameter='additive_error_on_cosine_of_aod',
            error_sweep_range=error_sweep_range,
            precoder_name=precoder_name,
            monte_carlo_iterations=monte_carlo_iterations,
            get_precoder_func=lambda c, u, s, f=get_precoder_func: f(c, u, s),
            calc_reward_funcs=[calc_sum_rate],
        )

    # power-matched MMSE: same 56 W SAC (error=0.0) actually uses, swept
    # across the same error range. cfg.mmse_args is a copied-by-value dict
    # baked at Config() construction, so it must be overridden directly --
    # reassigning cfg.power_constraint_watt alone would not propagate here.
    cfg.mmse_args['power_constraint_watt'] = 56.0
    test_precoder_error_sweep(
        config=cfg,
        error_sweep_parameter='additive_error_on_cosine_of_aod',
        error_sweep_range=error_sweep_range,
        precoder_name='mmse_56w',
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=lambda c, u, s: get_precoding_mmse(c, u, s),
        calc_reward_funcs=[calc_sum_rate],
    )

    # ---- combined 6-curve plot ----
    plot_cfg = PlotConfig()
    matplotlib.rcParams['text.usetex'] = False  # PlotConfig() resets this on construction
    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.6

    data_paths = [
        Path(cfg.output_metrics_path, CLASSICAL_TRAINING_NAME, 'error_sweep', 'testing_mmse_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path, CLASSICAL_TRAINING_NAME, 'error_sweep', 'testing_mrt_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path, CLASSICAL_TRAINING_NAME, 'error_sweep', 'testing_zf_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path, CLASSICAL_TRAINING_NAME, 'error_sweep', 'testing_mmse_56w_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path, 'EE_dinkelbach_adaptive_lwin5000_N16K3_eta0.6_rawpow', 'error_sweep', 'testing_learned_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path, 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_eta0.6_rawpow', 'error_sweep', 'testing_learned_sweep_0.0_0.1.gzip'),
    ]
    # MMSE (56W) placed right before the SAC curves -- it's the power-matched
    # comparison for SAC (error=0.0), same convention as the bar chart in
    # classical_baselines_zero_error.py.
    legend = ['MMSE', 'MRT', 'ZF', 'MMSE (56W)', 'SAC (error=0.0)', 'SAC (error=0.05)']
    colors = [
        plot_cfg.cp2['gold'],
        '#7a4fbf',  # purple, matches classical_baselines_zero_error.py's convention
        '#c9622a',  # burnt orange, same
        '#8a7000',  # dark gold, ties visually back to MMSE while staying distinct
        '#307b3b',  # green, matches SAC-family convention elsewhere
        '#1baf7a',  # aqua
    ]
    markerstyles = ['v', 'D', 'x', '^', 'o', 's']
    linestyles = [':', ':', ':', '-.', '-', '--']

    plot_error_sweep_testing_graph(
        paths=data_paths,
        metric='sumrate',
        name='error_sweep_classical_vs_sac',
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
    ax.legend(legend, loc='upper right', ncols=1, fontsize=7, framealpha=1.0, frameon=True)
    ax.set_xlabel('Error Bound')
    ax.set_ylabel('Rate R [bps/Hz]')
    plt.tight_layout(pad=0.2)
    plt.savefig(
        Path(plot_cfg.plots_parent_path, 'pdf', 'error_sweep_classical_vs_sac_sumrate.pdf'),
        bbox_inches='tight', dpi=800, transparent=True,
    )
    print('Saved: error_sweep_classical_vs_sac_sumrate.pdf')
