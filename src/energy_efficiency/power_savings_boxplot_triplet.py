
import os
import sys

os.environ.pop('EE_SAT_GAIN_DBI', None)
os.environ.pop('EE_POWER_BUDGET_WATT', None)
os.environ.pop('EE_TARGET_ELEVATION_DEG', None)

import gzip
import pickle
from pathlib import Path

from src.config.config import Config
from src.config.config_plotting import PlotConfig
from src.energy_efficiency.power_savings_bars_triplet import (
    run_power_samples,
    run_matched_power_mmse_samples,
)
from src.energy_efficiency.tx_power_distribution import plot_power_savings_comparison
from src.energy_efficiency.plotting_scenario import CHECKPOINTS, get_best_model_path
from src.utils.get_precoding import get_precoding_learned, get_precoding_learned_clip_only, get_precoding_mmse
from src.utils.load_model import load_model

PLOT_ONLY = '--plot-only' in sys.argv

# Same samples_dict shape (monte_carlo_iterations, user_nr per label) and
# same cache path as power_savings_bars_triplet.py's Figure 2 -- whichever
# of the two scripts runs first fills the shared gzip, the other just reads
# it with --plot-only. Keeps the boxplot (this figure) and the bar chart
# (Figure 2) numerically identical instead of drawing from two independent
# Monte Carlo runs.
if __name__ == '__main__':
    cfg = Config()
    cfg.show_plots = False
    print(f'[system] sat_gain_dBi={cfg.sat_gain_dBi}, budget={cfg.power_constraint_watt} W')

    out_path = Path(cfg.output_metrics_path, 'EE_lwin5000_3gpp_triplet')
    out_path.mkdir(parents=True, exist_ok=True)
    gzip_path = Path(out_path, 'power_savings_bars.gzip')

    if not PLOT_ONLY and not gzip_path.exists():
        samples_dict = {}
        power_budget = cfg.power_constraint_watt

        mmse_full_label = 'MMSE (75 W budget)'
        samples_dict[mmse_full_label] = run_power_samples(cfg, mmse_full_label, get_precoding_mmse)

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

        sac_ee_mean_power = samples_dict[sac_ee_label].sum(axis=1).mean()
        mmse_matched_label = 'MMSE (equal power, Δε = 0.0)'
        samples_dict[mmse_matched_label] = run_matched_power_mmse_samples(
            cfg, mmse_matched_label, sac_ee_mean_power,
        )

        with gzip.open(gzip_path, 'wb') as file:
            pickle.dump({'power_budget': power_budget, 'samples_dict': samples_dict}, file=file)
        print(f'Saved: {gzip_path}')
    elif not PLOT_ONLY:
        print(f'[power_savings_boxplot_triplet] {gzip_path} already exists (likely from '
              f'power_savings_bars_triplet.py) -- reusing it instead of re-running Monte Carlo.')

    with gzip.open(gzip_path, 'rb') as file:
        data = pickle.load(file)

    plot_cfg = PlotConfig()

    # Same MMSE/RM/EE notation as plotting_scenario.py's error-sweep figure
    # and power_savings_bars_triplet.py's bar chart.
    ee_train_watt = round(data['samples_dict']['SAC (Δε = 0.0, energy-efficient)'].sum(axis=1).mean())
    mmse_matched_watt = round(data['samples_dict']['MMSE (equal power, Δε = 0.0)'].sum(axis=1).mean())
    display_labels = {
        'MMSE (75 W budget)': 'MMSE',
        'SAC (75 W budget)': 'RM',
        'SAC (Δε = 0.0, energy-efficient)': f'EE, $P={ee_train_watt}$ W',
        'MMSE (equal power, Δε = 0.0)': f'MMSE, $P={mmse_matched_watt}$ W',
    }
    ordered_labels = [
        'MMSE (75 W budget)',
        'SAC (75 W budget)',
        'SAC (Δε = 0.0, energy-efficient)',
        'MMSE (equal power, Δε = 0.0)',
    ]
    ordered_samples_dict = {
        display_labels[label]: data['samples_dict'][label] for label in ordered_labels
    }

    model_colors = [
        plot_cfg.cp2['black'],
        plot_cfg.cp2['gold'],
        plot_cfg.cp2['green'],
        plot_cfg.cp2['black'],
    ]

    plot_power_savings_comparison(
        samples_dict=ordered_samples_dict,
        power_budget=data['power_budget'],
        plots_parent_path=plot_cfg.plots_parent_path,
        name='power_savings_boxplot',
        model_colors=model_colors,
    )
