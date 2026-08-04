"""
Verification / sanity-check script (user-requested, 2026-08-03).

Purpose: confirm the OLD-system SAC checkpoints (error=0.0 and error=0.05
training bounds, EE_dinkelbach_adaptive_lwin5000_N16K3_eta0.6_rawpow and
..._aod0.05_lwin5000_N16K3_eta0.6_rawpow) still draw the expected power under
THIS repo's current EE_sac.py / get_precoding code path, then reproduce the
2025 repo's error_sweep_classical_vs_sac_4curve_sumrate.pdf using data
computed entirely in this (2026) repo -- with the matched-power MMSE curve
built from a FRESH measurement of SAC(error=0.0)'s power, not a hardcoded
constant.

IMPORTANT: these two checkpoints were trained under the OLD system
parameters (20 dBi / 100 W) -- config.py's defaults are now the 3GPP values
(30 dBi / 75 W), so EE_SAT_GAIN_DBI/EE_POWER_BUDGET_WATT are force-set below
before Config() is ever constructed. See handoff_prompt_EE_evaluation.txt's
"CRITICAL eval footgun" note.

Rate curves for the two SAC checkpoints and the full-budget MMSE curve reuse
already-computed gzips on disk (both repos received the same models/ sync,
and these gzips already exist in outputs/metrics/ here) -- only the new
matched-power MMSE curve requires fresh Monte Carlo.
"""
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False

import os
# Force old-system parameters for this whole script -- these two checkpoints
# predate the 3GPP defaults introduced 2026-07-27.
os.environ['EE_SAT_GAIN_DBI'] = '20'
os.environ['EE_POWER_BUDGET_WATT'] = '100'
os.environ.pop('EE_TARGET_ELEVATION_DEG', None)

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.config.config import Config
from src.config.config_plotting import PlotConfig
from src.data.calc_sum_rate import calc_sum_rate
from src.data.calc_tx_power_distribution import calc_tx_power_distribution
from src.data.satellite_manager import SatelliteManager
from src.data.user_manager import UserManager
from src.utils.get_precoding import get_precoding_learned_clip_only, get_precoding_mmse
from src.utils.load_model import load_model
from src.utils.update_sim import update_sim
from src.analysis.helpers.test_precoder_error_sweep import test_precoder_error_sweep
import src.plotting.plot_error_sweep_testing_graph as plot_module
from src.plotting.plot_error_sweep_testing_graph import plot_error_sweep_testing_graph

ERROR_PARAM = 'additive_error_on_cosine_of_aod'
ERROR_SWEEP_RANGE = np.linspace(0, 0.10, 11)
MONTE_CARLO_ITERATIONS = 10000

CLASSICAL_TRAINING_NAME = 'classical_baselines_N16K3_eta0.6'
SAC_AOD0_TRAINING_NAME = 'EE_dinkelbach_adaptive_lwin5000_N16K3_eta0.6_rawpow'
SAC_AOD05_TRAINING_NAME = 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_eta0.6_rawpow'


def _save_figures_pdf_only(plots_parent_path, plot_name, padding=0):
    pdf_path = Path(plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)
    out = Path(pdf_path, f'{plot_name}.pdf')
    plt.savefig(out, bbox_inches='tight', pad_inches=padding, dpi=800, transparent=True)
    print(f'Saved: {out}')


plot_module.save_figures = _save_figures_pdf_only


def get_best_model_path(trained_models_path, training_name):
    """Session-aware checkpoint selection -- mirrors my_evaluation.py's identical function."""
    base_path = Path(trained_models_path, training_name, 'base')
    checkpoints = [p for p in base_path.iterdir() if p.is_dir() and 'full_snap' in p.name]
    if not checkpoints:
        raise FileNotFoundError(f'No checkpoints found under {base_path}')
    checkpoints_by_time = sorted(checkpoints, key=lambda p: os.path.getmtime(p))
    max_gap_seconds = 90 * 60
    session_start_idx = 0
    for i in range(len(checkpoints_by_time) - 1, 0, -1):
        gap = os.path.getmtime(checkpoints_by_time[i]) - os.path.getmtime(checkpoints_by_time[i - 1])
        if gap > max_gap_seconds:
            session_start_idx = i
            break
    same_session_checkpoints = checkpoints_by_time[session_start_idx:]
    best = sorted(same_session_checkpoints, key=lambda p: float(p.name.split('_')[-1]))[-1]
    return best


def measure_sac_power_and_rate(training_name, error_bound, label):
    """Fresh Monte Carlo measurement of one checkpoint's mean transmit power
    and rate at a single CSIT error level, using the clipped (physically
    transmittable) precoder -- same convention as rate_power_error_sweep.py."""
    cfg = Config()
    cfg.show_plots = False
    cfg.config_learner.training_name = training_name
    model_path = get_best_model_path(cfg.trained_models_path, training_name)
    precoder_network, norm_factors = load_model(model_path)
    if norm_factors != {}:
        cfg.config_learner.get_state_args['norm_state'] = True

    satellite_manager = SatelliteManager(config=cfg)
    user_manager = UserManager(config=cfg)

    cfg.config_error_model.error_rng_parametrizations[ERROR_PARAM]['args']['low'] = -error_bound
    cfg.config_error_model.error_rng_parametrizations[ERROR_PARAM]['args']['high'] = error_bound

    power_samples = np.zeros(MONTE_CARLO_ITERATIONS)
    rate_samples = np.zeros(MONTE_CARLO_ITERATIONS)
    for i in range(MONTE_CARLO_ITERATIONS):
        update_sim(cfg, satellite_manager, user_manager)
        w = get_precoding_learned_clip_only(cfg, user_manager, satellite_manager, norm_factors, precoder_network)
        power_samples[i] = calc_tx_power_distribution(w_precoder=w).sum()
        rate_samples[i] = calc_sum_rate(
            channel_state=satellite_manager.channel_state_information,
            w_precoder=w,
            noise_power_watt=cfg.noise_power_watt,
        )

    mean_power = power_samples.mean()
    mean_rate = rate_samples.mean()
    print(
        f'[{label}] model_path={model_path.name}, error={error_bound}: '
        f'power={mean_power:.2f} W ({100 * mean_power / cfg.power_constraint_watt:.1f}% of '
        f'{cfg.power_constraint_watt:.0f} W budget), rate={mean_rate:.4f} bps/Hz, '
        f'std_power={power_samples.std():.2f} W'
    )
    return mean_power, mean_rate


if __name__ == '__main__':
    print(f'=== Fresh SAC power measurement, {MONTE_CARLO_ITERATIONS} MC iterations each ===')
    sac_power_aod0, sac_rate_aod0 = measure_sac_power_and_rate(
        SAC_AOD0_TRAINING_NAME, error_bound=0.0, label='SAC error=0.0 checkpoint @ error=0.0',
    )
    sac_power_aod05, sac_rate_aod05 = measure_sac_power_and_rate(
        SAC_AOD05_TRAINING_NAME, error_bound=0.05, label='SAC error=0.05 checkpoint @ error=0.05',
    )

    print()
    print(f'=== Matched-power MMSE sweep: power_constraint_watt={sac_power_aod0:.2f} W '
          f'(SAC error=0.0 checkpoint\'s own measured power at error=0), '
          f'{len(ERROR_SWEEP_RANGE)} error points x {MONTE_CARLO_ITERATIONS} MC ===')
    cfg = Config()
    cfg.show_plots = False
    cfg.config_learner.training_name = CLASSICAL_TRAINING_NAME
    cfg.mmse_args['power_constraint_watt'] = float(sac_power_aod0)
    test_precoder_error_sweep(
        config=cfg,
        error_sweep_parameter=ERROR_PARAM,
        error_sweep_range=ERROR_SWEEP_RANGE,
        precoder_name='mmse_matched_parajuli',
        monte_carlo_iterations=MONTE_CARLO_ITERATIONS,
        get_precoder_func=lambda c, u, s: get_precoding_mmse(c, u, s),
        calc_reward_funcs=[calc_sum_rate],
    )

    # ---- combined 4-curve plot ----
    plot_cfg = PlotConfig()
    matplotlib.rcParams['text.usetex'] = False  # PlotConfig() resets this on construction
    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.8

    data_paths = [
        Path(cfg.output_metrics_path, CLASSICAL_TRAINING_NAME, 'error_sweep', 'testing_mmse_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path, CLASSICAL_TRAINING_NAME, 'error_sweep', 'testing_mmse_matched_parajuli_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path, SAC_AOD0_TRAINING_NAME, 'error_sweep', 'testing_learned_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path, SAC_AOD05_TRAINING_NAME, 'error_sweep', 'testing_learned_sweep_0.0_0.1.gzip'),
    ]
    legend = [
        'MMSE (100W)',
        f'MMSE ({sac_power_aod0:.1f}W, matched to SAC err=0.0)',
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
    ax.legend(legend, loc='upper right', ncols=1, fontsize=10, framealpha=1.0, frameon=True)
    ax.set_xlabel('Error Bound')
    ax.set_ylabel('Rate R [bps/Hz]')
    plt.tight_layout(pad=0.2)

    pdf_out = Path(plot_cfg.plots_parent_path, 'pdf', 'parajuli.pdf')
    plt.savefig(pdf_out, bbox_inches='tight', dpi=800, transparent=True)
    print(f'Saved: {pdf_out}')

    print()
    print('=== Summary ===')
    print(f'SAC (error=0.0 checkpoint) power @ error=0.0:  {sac_power_aod0:.2f} W ({100*sac_power_aod0/100:.1f}% of 100W budget), rate {sac_rate_aod0:.4f} bps/Hz')
    print(f'SAC (error=0.05 checkpoint) power @ error=0.05: {sac_power_aod05:.2f} W ({100*sac_power_aod05/100:.1f}% of 100W budget), rate {sac_rate_aod05:.4f} bps/Hz')
