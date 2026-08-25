"""
"How much power is each precoder actually using?" for the current
lwin5000/3GPP nadir triplet (aod=0.0 job 156358, aod=0.025, aod=0.05 job
153655) plus the aod=0.0 checkpoint's full-power (always-rescale) variant --
the same four SAC evaluations as plotting_scenario.py's error-sweep figure,
here condensed to a single CSIT error point (0.0, "zero eval error") so the
loading-bar comparison isn't muddied by an error axis.

Reuses plot_power_savings_bars() from tx_power_distribution.py -- see that
function's docstring for the visual design (gray track = power budget,
colored fill = mean power used, "saves N W (X%)" annotation) -- rather than
reimplementing it. That module's own __main__ block targets older
checkpoints that no longer exist on disk (full_EE_aod0.5_N16K3_eta0.6's
Dinkelbach/no-normalization comparison, an N8K2 repro, etc.); this script
is the equivalent for the current triplet, matching the reference archived
figures reports/figures/pdf_archive/power_savings_bars_error_bound_0_0.05.pdf
and power_savings_bars_error_bound_0_0.025_0.05.pdf.

Also reuses plotting_scenario.py's CHECKPOINTS mapping and session-aware
get_best_model_path() rather than redefining them.

Saves one gzip: outputs/metrics/EE_lwin5000_3gpp_triplet/power_savings_bars.gzip
Pass --plot-only to skip the Monte Carlo run and just re-plot an existing gzip.
"""
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
from src.utils.get_precoding import get_precoding_learned, get_precoding_learned_clip_only
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
        power_budget = None

        for aod_key, training_name in CHECKPOINTS.items():
            cfg.config_learner.training_name = training_name
            model_path = get_best_model_path(cfg.trained_models_path, training_name)
            print(f'[{aod_key}] checkpoint: {model_path}')

            precoder_network, norm_factors = load_model(model_path)
            if norm_factors != {}:
                cfg.config_learner.get_state_args['norm_state'] = True

            power_budget = cfg.power_constraint_watt
            label = f'SAC (train err {aod_key.replace("aod", "")}, energy-efficient)'
            samples_dict[label] = run_power_samples(
                cfg, label,
                lambda c, um, sm: get_precoding_learned_clip_only(c, um, sm, norm_factors, precoder_network),
            )

            # aod0.0's checkpoint is additionally evaluated with the always-
            # rescale-to-budget precoder -- same trained network, forced to
            # spend the full 75W budget -- as the "no savings" reference bar,
            # matching the full-power curve in the error-sweep figure.
            if aod_key == 'aod0.0':
                fullpower_label = 'SAC (train err 0.0, full power)'
                samples_dict[fullpower_label] = run_power_samples(
                    cfg, fullpower_label,
                    lambda c, um, sm: get_precoding_learned(c, um, sm, norm_factors, precoder_network),
                )

        with gzip.open(gzip_path, 'wb') as file:
            pickle.dump({'power_budget': power_budget, 'samples_dict': samples_dict}, file=file)
        print(f'Saved: {gzip_path}')

    # ---- plot (reads the gzip just saved above, or an existing one if --plot-only) ----
    with gzip.open(gzip_path, 'rb') as file:
        data = pickle.load(file)

    plot_cfg = PlotConfig()

    # Reorder so the full-power reference bar is drawn first (top row), then
    # the three energy-efficient bars in increasing-training-error order --
    # plot_power_savings_bars draws samples_dict's first entry at the top.
    ordered_labels = [
        'SAC (train err 0.0, full power)',
        'SAC (train err 0.0, energy-efficient)',
        'SAC (train err 0.025, energy-efficient)',
        'SAC (train err 0.05, energy-efficient)',
    ]
    ordered_samples_dict = {label: data['samples_dict'][label] for label in ordered_labels}

    # gold for the full-power reference bar, then green/blue/magenta for the
    # energy-efficient 0.0/0.025/0.05 triplet -- same palette convention as
    # plotting_scenario.py's error-sweep figure, for visual consistency
    # across the paper's figure set.
    bar_colors = [
        plot_cfg.cp2['gold'],
        plot_cfg.cp2['green'],
        plot_cfg.cp2['blue'],
        plot_cfg.cp2['magenta'],
    ]

    plot_power_savings_bars(
        samples_dict=ordered_samples_dict,
        power_budget=data['power_budget'],
        plots_parent_path=plot_cfg.plots_parent_path,
        name='power_savings_bars_3gpp_triplet',
        model_colors=bar_colors,
        title='3GPP nadir triplet: transmit power used vs. budget (zero eval error)',
    )
