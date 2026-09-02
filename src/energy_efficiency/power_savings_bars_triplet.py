
import os
import sys

os.environ.pop('EE_SAT_GAIN_DBI', None)
os.environ.pop('EE_POWER_BUDGET_WATT', None)
os.environ.pop('EE_TARGET_ELEVATION_DEG', None)

import gzip
import pickle
from pathlib import Path

import numpy as np

from src.config.config import Config
from src.config.config_plotting import PlotConfig
from src.data.calc_tx_power_distribution import calc_tx_power_distribution
from src.data.satellite_manager import SatelliteManager
from src.data.user_manager import UserManager
from src.utils.get_precoding import get_precoding_learned, get_precoding_learned_clip_only, get_precoding_mmse
from src.utils.load_model import load_model
from src.utils.update_sim import update_sim
from src.energy_efficiency.plotting_scenario import CHECKPOINTS, get_best_model_path
from src.energy_efficiency.tx_power_distribution import plot_power_savings_bars

PLOT_ONLY = '--plot-only' in sys.argv

eval_error_bound = 0.0  # "zero eval error" -- matches the archived reference figures
monte_carlo_iterations = 10000


def run_power_samples(cfg, label, get_precoder_func):
    """Per-user transmit power for monte_carlo_iterations draws at a fixed CSIT error bound.
    Returns an array of shape (monte_carlo_iterations, user_nr), the input plot_power_savings_bars expects."""
    satellite_manager = SatelliteManager(config=cfg)
    user_manager = UserManager(config=cfg)

    error_param = 'additive_error_on_cosine_of_aod'
    cfg.config_error_model.error_rng_parametrizations[error_param]['args']['low'] = -eval_error_bound
    cfg.config_error_model.error_rng_parametrizations[error_param]['args']['high'] = eval_error_bound

    samples = np.zeros((monte_carlo_iterations, cfg.user_nr))
    for iter_idx in range(monte_carlo_iterations):
        update_sim(cfg, satellite_manager, user_manager)
        w_precoder = get_precoder_func(cfg, user_manager, satellite_manager)
        samples[iter_idx, :] = calc_tx_power_distribution(w_precoder=w_precoder)
        if iter_idx % 1000 == 0:
            print(f'[{label}] {iter_idx}/{monte_carlo_iterations}')

    total_power = samples.sum(axis=1)
    print(f'[{label}] mean total power: {total_power.mean():.2f} W '
          f'({100 * total_power.mean() / cfg.power_constraint_watt:.1f}% of budget)')
    return samples


def run_matched_power_mmse_samples(cfg, label, target_mean_power):
    """Per-user transmit power for MMSE rescaled to target_mean_power -- the fair
    equal-power comparison against SAC's own measured power. Same rescaling as
    plotting_scenario.py's run_matched_power_mmse_sweep, but a single fixed
    target (this figure is one CSIT error point, not a sweep)."""
    satellite_manager = SatelliteManager(config=cfg)
    user_manager = UserManager(config=cfg)

    error_param = 'additive_error_on_cosine_of_aod'
    cfg.config_error_model.error_rng_parametrizations[error_param]['args']['low'] = -eval_error_bound
    cfg.config_error_model.error_rng_parametrizations[error_param]['args']['high'] = eval_error_bound

    samples = np.zeros((monte_carlo_iterations, cfg.user_nr))
    for iter_idx in range(monte_carlo_iterations):
        update_sim(cfg, satellite_manager, user_manager)
        w_mmse = get_precoding_mmse(cfg, user_manager, satellite_manager)
        current_power = np.real(np.trace(np.matmul(w_mmse.conj().T, w_mmse)))
        w_precoder = w_mmse * np.sqrt(target_mean_power / current_power)
        samples[iter_idx, :] = calc_tx_power_distribution(w_precoder=w_precoder)
        if iter_idx % 1000 == 0:
            print(f'[{label}] {iter_idx}/{monte_carlo_iterations}')

    total_power = samples.sum(axis=1)
    print(f'[{label}] mean total power: {total_power.mean():.2f} W (target {target_mean_power:.2f} W)')
    return samples


if __name__ == '__main__':
    cfg = Config()
    cfg.show_plots = False
    print(f'[system] sat_gain_dBi={cfg.sat_gain_dBi}, budget={cfg.power_constraint_watt} W, '
          f'eval error bound={eval_error_bound}')

    out_path = Path(cfg.output_metrics_path, 'EE_lwin5000_3gpp_triplet')
    out_path.mkdir(parents=True, exist_ok=True)
    gzip_path = Path(out_path, 'power_savings_bars.gzip')

    if not PLOT_ONLY:
        samples_dict = {}
        power_budget = cfg.power_constraint_watt

        # MMSE full budget -- system-only, no checkpoint dependency.
        mmse_full_label = 'MMSE (75 W budget)'
        samples_dict[mmse_full_label] = run_power_samples(cfg, mmse_full_label, get_precoding_mmse)

        # aod0.0 checkpoint, evaluated both ways: always-rescale-to-budget
        # ("full power") and clip-only ("energy-efficient") -- same trained
        # network, matching plotting_scenario.py's error-sweep figure.
        training_name = CHECKPOINTS['aod0.0']
        cfg.config_learner.training_name = training_name
        model_path = get_best_model_path(cfg.trained_models_path, training_name)
        print(f'[aod0.0] checkpoint: {model_path}')

        precoder_network, norm_factors = load_model(model_path)
        if norm_factors != {}:
            cfg.config_learner.get_state_args['norm_state'] = True

        sac_full_label = 'SAC (75 W budget)'
        samples_dict[sac_full_label] = run_power_samples(
            cfg, sac_full_label,
            lambda c, um, sm: get_precoding_learned(c, um, sm, norm_factors, precoder_network),
        )

        sac_ee_label = 'SAC (Δε = 0.0, energy-efficient)'
        samples_dict[sac_ee_label] = run_power_samples(
            cfg, sac_ee_label,
            lambda c, um, sm: get_precoding_learned_clip_only(c, um, sm, norm_factors, precoder_network),
        )

        # MMSE rescaled to SAC-EE's own measured mean power -- the fair
        # equal-power comparison, same pairing as the error-sweep figure.
        sac_ee_mean_power = samples_dict[sac_ee_label].sum(axis=1).mean()
        mmse_matched_label = 'MMSE (equal power, Δε = 0.0)'
        samples_dict[mmse_matched_label] = run_matched_power_mmse_samples(
            cfg, mmse_matched_label, sac_ee_mean_power,
        )

        with gzip.open(gzip_path, 'wb') as file:
            pickle.dump({'power_budget': power_budget, 'samples_dict': samples_dict}, file=file)
        print(f'Saved: {gzip_path}')

    # ---- plot (reads the gzip just saved above, or an existing one if --plot-only) ----
    with gzip.open(gzip_path, 'rb') as file:
        data = pickle.load(file)

    plot_cfg = PlotConfig()

    # keys must match the strings run_power_samples()/run_matched_power_mmse_samples()
    # used when building samples_dict (and thus the cached gzip); the display
    # labels shown on the chart are remapped separately below so relabeling
    # here never requires regenerating cached data.
    ordered_labels = [
        'SAC (75 W budget)',
        'MMSE (75 W budget)',
        'SAC (Δε = 0.0, energy-efficient)',
        'MMSE (equal power, Δε = 0.0)',
    ]
    display_labels = {
        'SAC (75 W budget)': 'SAC (75 W budget)',
        'MMSE (75 W budget)': 'MMSE (75 W budget)',
        'SAC (Δε = 0.0, energy-efficient)': 'EE (Δε = 0.0)',
        'MMSE (equal power, Δε = 0.0)': 'MMSE (equal power, Δε = 0.0)',
    }
    ordered_samples_dict = {
        display_labels[label]: data['samples_dict'][label] for label in ordered_labels
    }

    bar_colors = [
        plot_cfg.cp2['gold'],
        plot_cfg.cp2['black'],
        plot_cfg.cp2['green'],
        plot_cfg.cp2['black'],
    ]

    plot_power_savings_bars(
        samples_dict=ordered_samples_dict,
        power_budget=data['power_budget'],
        plots_parent_path=plot_cfg.plots_parent_path,
        name='power_savings_bars',
        model_colors=bar_colors,
        title='transmit power used vs. budget',
    )
