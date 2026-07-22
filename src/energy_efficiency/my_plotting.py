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
    # 'energy_efficiency' (training_name 'full_EE_aod0.5_N16K3_eta0.6') is
    # Scheme I -- the paper's always-rescale-to-budget normalization (Eq. 26
    # style). Total transmit power for this model is, by construction,
    # pinned at the power budget for every sample; this plot shows its RATE
    # behaviour only, not power.
    #
    # 'energy_efficiency_dinkelbach_adaptive' (training_name
    # 'EE_dinkelbach_adaptive_aod0.5_lwin5000_N16K3_eta0.6') is clip-only,
    # genuine inequality constraint, free to transmit below budget, reward =
    # rate - lambda*power instead of a direct ratio. Whether it actually
    # transmits below budget is answered by the tx_power_histogram script,
    # not by this rate-vs-error plot.
    #
    # 'EE_dinkelbach_adaptive_aod0.5_block1_N16K3_eta0.6' (job 144104) is the
    # SAME Dinkelbach reward/clip-only precoder as the lwin5000 model above,
    # with only the lambda-update mechanism changed: lambda is held fixed for
    # a whole episode and updated once from that episode's fresh mean
    # rate/power (EE_DINKELBACH_BLOCK_EPISODES=1), instead of a continuous
    # per-step EMA. Plotted alongside the lwin5000 curve to test whether that
    # change visibly reduces the flat, under-MMSE-ceiling error-sweep shape
    # (see docs/EE_formulation.tex Section 6's rebuttal for why this was
    # suspected to be the dominant cause, not entropy regularization or
    # under-capacity).
    #
    # Both EMA/block models were trained at a fixed Delta-epsilon_aod=0.5 (see
    # EE_TRAIN_ERROR_BOUND in EE_sac.py), N=16/K=3 (matching the paper's
    # Fig. 3 scenario), and pa_efficiency=0.60 (matching Ha et al.
    # 2023/2024) -- see EE_sac.py __main__'s suffix logic. my_evaluation.py's
    # error_sweep_range goes up to 0.5 -- hence the '_0.0_0.5.gzip' filenames
    # below.
    #
    # NOTE: training_name strings in the paths below must match exactly what
    # was used in the corresponding evaluation script run -- these are the
    # same folder names used there. Previously pointed at 'full_EE_aod0.5' /
    # 'EE_dinkelbach_adaptive_aod0.5' (N=8/K=2, eta=0.35) -- updated to match
    # my_evaluation.py after the system-size/efficiency retrain.
    # -----------------------------------------------------------------
    data_paths = [
        Path(cfg.output_metrics_path, 'full_EE_aod0.5_N16K3_eta0.6', 'error_sweep', 'testing_mmse_sweep_0.0_0.5.gzip'),
        Path(cfg.output_metrics_path, 'full_EE_aod0.5_N16K3_eta0.6', 'error_sweep', 'testing_learned_sweep_0.0_0.5.gzip'),
        Path(cfg.output_metrics_path, 'EE_dinkelbach_adaptive_aod0.5_lwin5000_N16K3_eta0.6', 'error_sweep', 'testing_learned_sweep_0.0_0.5.gzip'),
        Path(cfg.output_metrics_path, 'EE_dinkelbach_adaptive_aod0.5_block1_N16K3_eta0.6', 'error_sweep', 'testing_learned_sweep_0.0_0.5.gzip'),
    ]
    legend = [
        'MMSE',
        'SAC (EE, paper normalization)',
        'SAC (EE, Dinkelbach, EMA lambda)',
        'SAC (EE, Dinkelbach, block lambda)',
    ]
    colors = [
        plot_cfg.cp2['gold'],
        plot_cfg.cp2['blue'],
        plot_cfg.cp2['green'],
        plot_cfg.cp2['magenta'],
    ]
    markerstyles = ['v', 'o', '^', 's']
    linestyles = ['-', '-', '--', '--']
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

    # -----------------------------------------------------------------
    # N8K2 paper-reproduction sanity check -- job 145427 (see
    # handoff_prompt_EE_evaluation.txt's "Jobs submitted Friday" #1 and
    # "What to do Monday" item 4). Does TODAY's code + current
    # hyperparameters (LR retuned to 4.2e-5/8.8e-6 at some point after the
    # old N8K2 data below was generated -- see EE_full_aod0.5_N8K2.slurm's
    # comment) still reach near-MMSE-ceiling performance at the OLDER
    # N=8/K=2 system size, matching what the old data on disk shows? The old
    # data (outputs/metrics/full_EE_aod0.5/, eta=0.35) predates this week's
    # eta=0.60/pa_efficiency retune and the N16K3 switch entirely -- it's
    # the healthy baseline this investigation kept referring back to (SAC
    # reaches ~84% of MMSE's zero-error ceiling and BEATS MMSE past
    # error~0.1, "exactly the paper's own robustness story").
    #
    # Both MMSE curves (old and new) are plotted rather than assuming they
    # agree: MMSE's precoder computation doesn't depend on pa_efficiency
    # (eta only scales DC power draw, not the precoder itself) so old vs new
    # MMSE should closely overlap -- if they visibly diverge, something
    # about the error model or system config silently changed between the
    # two runs, and that's worth knowing before trusting the SAC comparison.
    # -----------------------------------------------------------------
    n8k2_data_paths = [
        Path(cfg.output_metrics_path, 'full_EE_aod0.5', 'error_sweep', 'testing_mmse_sweep_0.0_0.5.gzip'),
        Path(cfg.output_metrics_path, 'full_EE_aod0.5_N8K2_eta0.6', 'error_sweep', 'testing_mmse_sweep_0.0_0.5.gzip'),
        Path(cfg.output_metrics_path, 'full_EE_aod0.5', 'error_sweep', 'testing_learned_sweep_0.0_0.5.gzip'),
        Path(cfg.output_metrics_path, 'full_EE_aod0.5_N8K2_eta0.6', 'error_sweep', 'testing_learned_sweep_0.0_0.5.gzip'),
    ]
    n8k2_legend = [
        'MMSE (old N8K2 data)',
        'MMSE (N8K2 repro, current code)',
        'SAC (old N8K2 data)',
        'SAC (N8K2 repro, current code)',
    ]
    n8k2_colors = [
        plot_cfg.cp2['gold'],
        plot_cfg.cp2['gold'],
        plot_cfg.cp2['blue'],
        plot_cfg.cp2['green'],
    ]
    n8k2_markerstyles = ['v', 'v', 'o', '^']
    n8k2_linestyles = [':', '-', '--', '-']
    plot_error_sweep_testing_graph(
        paths=n8k2_data_paths,
        metric='sumrate',
        name='error_sweep_n8k2_reproduction',
        width=plot_width,
        height=plot_height,
        legend=n8k2_legend,
        colors=n8k2_colors,
        markerstyle=n8k2_markerstyles,
        linestyles=n8k2_linestyles,
        plots_parent_path=plot_cfg.plots_parent_path,
    )
    ax = plt.gca()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.legend(
        n8k2_legend,
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
        Path(plot_cfg.plots_parent_path, 'pdf', 'error_sweep_n8k2_reproduction_sumrate.pdf'),
        bbox_inches='tight', dpi=800, transparent=True,
    )
    print('Saved: error_sweep_n8k2_reproduction_sumrate.pdf')

    # -----------------------------------------------------------------
    # Raw-power-fixed Dinkelbach reward (EE_sac.py's dead-gradient fix, see
    # handoff_prompt_EE_evaluation.txt's "Finding 1"), trained at
    # Delta-epsilon_aod=0.0, 0.025, and 0.05 -- jobs 147505/148538/147506,
    # evaluated by job 148543. Tests "Finding 3": every training run before
    # this used EE_TRAIN_ERROR_BOUND=0.5, 10x the reference paper's largest
    # training value (Fig. 3: 0.0/0.025/0.05) -- this is now the full
    # three-point match. Compare directly against MMSE -- all three
    # checkpoints reach a real fraction of MMSE's ceiling (unlike every
    # 0.5-trained checkpoint this project has produced), and higher training
    # error trades lower zero-error rate for more graceful degradation: the
    # 0.025-trained curve crosses above the 0.0-trained curve around
    # error~0.035-0.04, then is itself overtaken by the 0.05-trained curve
    # further out -- a clean monotonic ordering, the qualitative "robust"
    # shape the paper's own Fig. 3 shows.
    #
    # error_sweep_range for these three SAC curves was trimmed to
    # np.linspace(0, 0.10, 11) (matching the paper's own evaluation range,
    # see my_evaluation.py) -- hence the '_0.0_0.1.gzip' filenames below,
    # unlike the '_0.0_0.5.gzip' MMSE reference reused from the plot above (a
    # superset that happens to include these points; not re-evaluated
    # separately since MMSE doesn't depend on which SAC checkpoint it's
    # being compared against).
    # -----------------------------------------------------------------
    error_bound_data_paths = [
        Path(cfg.output_metrics_path, 'full_EE_aod0.5_N16K3_eta0.6', 'error_sweep', 'testing_mmse_sweep_0.0_0.5.gzip'),
        Path(cfg.output_metrics_path, 'EE_dinkelbach_adaptive_lwin5000_N16K3_eta0.6_rawpow', 'error_sweep', 'testing_learned_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path, 'EE_dinkelbach_adaptive_aod0.025_lwin5000_N16K3_eta0.6_rawpow', 'error_sweep', 'testing_learned_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path, 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_eta0.6_rawpow', 'error_sweep', 'testing_learned_sweep_0.0_0.1.gzip'),
    ]
    error_bound_legend = [
        'MMSE',
        'SAC (Dinkelbach, raw-power fix, trained error=0.0)',
        'SAC (Dinkelbach, raw-power fix, trained error=0.025)',
        'SAC (Dinkelbach, raw-power fix, trained error=0.05)',
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
        name='error_sweep_rawpow_error_bound_comparison',
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
        fontsize=7,
        framealpha=1.0,
        frameon=True,
    )
    ax.set_xlabel('Error Bound')
    ax.set_ylabel('Rate R [bps/Hz]')
    # zoom to the range these SAC checkpoints were actually evaluated over
    # (MMSE's reference curve continues further right, out of frame, since
    # it's reused from the wider 0-0.5 sweep -- that's expected, not a bug)
    ax.set_xlim(-0.005, 0.105)
    plt.tight_layout(pad=0.2)
    plt.savefig(
        Path(plot_cfg.plots_parent_path, 'pdf', 'error_sweep_rawpow_error_bound_comparison_sumrate.pdf'),
        bbox_inches='tight', dpi=800, transparent=True,
    )
    print('Saved: error_sweep_rawpow_error_bound_comparison_sumrate.pdf')

    # -----------------------------------------------------------------
    # Default-LR vs. tuned-LR, all three error bounds (jobs
    # 152303/152304/152305 tuned-LR checkpoints vs. the default-LR
    # checkpoints plotted above). Tuned LRs came from the Optuna searches
    # (147529/147530/147531, OOM-killed at 65-67/100 trials each -- see
    # handoff), which optimize mean reward over only 20% of the full episode
    # budget per trial; whether that transfers to the full ~13,000-episode
    # training run is exactly what this comparison checks. Same color per
    # error bound, solid=default-LR / dashed=tuned-LR, so the two LR
    # choices for a given error bound are visually paired.
    # -----------------------------------------------------------------
    lrtuned_data_paths = [
        Path(cfg.output_metrics_path, 'EE_dinkelbach_adaptive_lwin5000_N16K3_eta0.6_rawpow', 'error_sweep', 'testing_learned_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path, 'EE_dinkelbach_adaptive_lwin5000_N16K3_eta0.6_rawpow_lrc4.14e-05_lra2.50e-07', 'error_sweep', 'testing_learned_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path, 'EE_dinkelbach_adaptive_aod0.025_lwin5000_N16K3_eta0.6_rawpow', 'error_sweep', 'testing_learned_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path, 'EE_dinkelbach_adaptive_aod0.025_lwin5000_N16K3_eta0.6_rawpow_lrc1.82e-06_lra1.17e-06', 'error_sweep', 'testing_learned_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path, 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_eta0.6_rawpow', 'error_sweep', 'testing_learned_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path, 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_eta0.6_rawpow_lrc2.50e-06_lra7.22e-05', 'error_sweep', 'testing_learned_sweep_0.0_0.1.gzip'),
    ]
    lrtuned_legend = [
        'SAC (error=0.0, default LR)',
        'SAC (error=0.0, tuned LR)',
        'SAC (error=0.025, default LR)',
        'SAC (error=0.025, tuned LR)',
        'SAC (error=0.05, default LR)',
        'SAC (error=0.05, tuned LR)',
    ]
    lrtuned_colors = [
        plot_cfg.cp2['green'], plot_cfg.cp2['green'],
        plot_cfg.cp2['blue'], plot_cfg.cp2['blue'],
        plot_cfg.cp2['magenta'], plot_cfg.cp2['magenta'],
    ]
    lrtuned_markerstyles = ['o', 'o', '^', '^', 's', 's']
    lrtuned_linestyles = ['-', '--', '-', '--', '-', '--']
    plot_error_sweep_testing_graph(
        paths=lrtuned_data_paths,
        metric='sumrate',
        name='error_sweep_lrtuned_vs_default_comparison',
        width=plot_width,
        height=plot_height,
        legend=lrtuned_legend,
        colors=lrtuned_colors,
        markerstyle=lrtuned_markerstyles,
        linestyles=lrtuned_linestyles,
        plots_parent_path=plot_cfg.plots_parent_path,
    )
    ax = plt.gca()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.legend(
        lrtuned_legend,
        loc='upper right',
        ncols=1,
        fontsize=7,
        framealpha=1.0,
        frameon=True,
    )
    ax.set_xlabel('Error Bound')
    ax.set_ylabel('Rate R [bps/Hz]')
    ax.set_xlim(-0.005, 0.105)
    plt.tight_layout(pad=0.2)
    plt.savefig(
        Path(plot_cfg.plots_parent_path, 'pdf', 'error_sweep_lrtuned_vs_default_comparison_sumrate.pdf'),
        bbox_inches='tight', dpi=800, transparent=True,
    )
    print('Saved: error_sweep_lrtuned_vs_default_comparison_sumrate.pdf')
