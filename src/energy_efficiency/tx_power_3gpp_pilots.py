"""
Transmit-power distribution for the two 3GPP Set-1 pilot checkpoints
(jobs 153655 nadir / 153656 elev30) against the CURRENT 75 W budget --
the tx_power_distribution.py-style "is power being saved?" figure, redone
for the current system.

Everything here runs at the current config defaults (30 dBi / 75 W; the
elev30 checkpoint additionally sets EE_TARGET_ELEVATION_DEG=30 to place
users in the geometry it was trained for). Nothing is evaluated at the
old 20 dBi / 100 W parameters -- the previous result's benchmark figure
(power_savings_comparison_error_bound_0_0.025_0.05.pdf, old checkpoints
vs their own 100 W budget) is kept in reports/figures/pdf/ for
side-by-side comparison.

Scenario matches the rest of the pilot evaluation: config.py's default
user distribution (100 km mean, +-50 km roam) and the active error-model
config (zero CSIT error), same convention as tx_power_distribution.py's
error-bound block. Reuses that script's plot functions unchanged.
"""
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False

import os
from pathlib import Path

import numpy as np

from src.config.config_plotting import PlotConfig
from src.data.calc_tx_power_distribution import calc_tx_power_distribution
from src.data.satellite_manager import SatelliteManager
from src.data.user_manager import UserManager
from src.energy_efficiency.tx_power_distribution import (
    plot_power_savings_bars,
    plot_power_savings_comparison,
)
from src.utils.get_precoding import get_precoding_learned_clip_only
from src.utils.load_model import load_model
from src.utils.update_sim import update_sim

MONTE_CARLO_ITERATIONS = 10000

SYSTEM_ENV_VARS = ['EE_SAT_GAIN_DBI', 'EE_POWER_BUDGET_WATT', 'EE_TARGET_ELEVATION_DEG']

SYSTEMS = [
    {
        'label': 'SAC (EE, 3GPP Set-1, nadir)',
        'training_name': 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_satg30_p75_eta0.6_rawpow',
        'env': {},
    },
    {
        'label': 'SAC (EE, 3GPP Set-1, elev 30 deg)',
        'training_name': 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_satg30_p75_elev30_eta0.6_rawpow',
        'env': {'EE_TARGET_ELEVATION_DEG': '30'},
    },
]

COLORS = ['#307b3b', '#1b6faf']  # matches rate_power_3gpp_pilots.py's SAC curves


def make_config(env: dict):
    """Construct a Config under exactly the given system env overrides (all others cleared)."""
    for var in SYSTEM_ENV_VARS:
        os.environ.pop(var, None)
    os.environ.update(env)
    from src.config.config import Config  # env vars are read in Config.__init__, after the overrides above
    cfg = Config()
    cfg.show_plots = False
    return cfg


def get_best_model_path(trained_models_path, training_name):
    """Session-aware checkpoint selection -- see my_evaluation.py's identical function for the full rationale."""
    base_path = Path(trained_models_path, training_name, 'base')
    checkpoints = [p for p in base_path.iterdir() if p.is_dir() and 'full_snap' in p.name]
    if not checkpoints:
        raise FileNotFoundError(f'No checkpoints found under {base_path}')

    checkpoints_by_time = sorted(checkpoints, key=lambda p: os.path.getmtime(p))

    max_gap_seconds = 90 * 60
    session_start_idx = len(checkpoints_by_time) - 1
    for i in range(len(checkpoints_by_time) - 1, 0, -1):
        gap = os.path.getmtime(checkpoints_by_time[i]) - os.path.getmtime(checkpoints_by_time[i - 1])
        if gap > max_gap_seconds:
            session_start_idx = i
            break
    else:
        session_start_idx = 0

    same_session_checkpoints = checkpoints_by_time[session_start_idx:]
    best = sorted(same_session_checkpoints, key=lambda p: float(p.name.split('_')[-1]))[-1]
    return best


def run_monte_carlo(cfg, model_path, label):
    precoder_network, norm_factors = load_model(model_path)
    if norm_factors != {}:
        cfg.config_learner.get_state_args['norm_state'] = True

    satellite_manager = SatelliteManager(config=cfg)
    user_manager = UserManager(config=cfg)

    samples = np.zeros((MONTE_CARLO_ITERATIONS, cfg.user_nr))
    for iter_idx in range(MONTE_CARLO_ITERATIONS):
        update_sim(cfg, satellite_manager, user_manager)
        w_precoder = get_precoding_learned_clip_only(cfg, user_manager, satellite_manager, norm_factors, precoder_network)
        samples[iter_idx, :] = calc_tx_power_distribution(w_precoder=w_precoder)
        if iter_idx % 1000 == 0:
            print(f'[{label}] {iter_idx}/{MONTE_CARLO_ITERATIONS}')
    return samples


if __name__ == '__main__':
    samples_dict = {}
    power_budget = None

    for system in SYSTEMS:
        cfg = make_config(system['env'])
        cfg.config_learner.training_name = system['training_name']
        power_budget = cfg.power_constraint_watt
        model_path = get_best_model_path(cfg.trained_models_path, system['training_name'])
        print(f"[{system['label']}] checkpoint: {model_path}")
        print(f"[{system['label']}] sat_gain_dBi={cfg.sat_gain_dBi}, budget={cfg.power_constraint_watt} W, "
              f"user_center_aod_earth_deg={cfg.user_center_aod_earth_deg:.2f}")
        samples_dict[system['label']] = run_monte_carlo(cfg, model_path, system['label'])

    plot_cfg = PlotConfig()
    matplotlib.rcParams['text.usetex'] = False  # PlotConfig() resets this on construction

    plot_power_savings_comparison(
        samples_dict=samples_dict,
        power_budget=power_budget,
        plots_parent_path=plot_cfg.plots_parent_path,
        name='power_savings_comparison_3gpp_pilots',
        model_colors=COLORS,
    )
    plot_power_savings_bars(
        samples_dict=samples_dict,
        power_budget=power_budget,
        plots_parent_path=plot_cfg.plots_parent_path,
        name='power_savings_bars_3gpp_pilots',
        model_colors=COLORS,
        title='3GPP Set-1 pilots: power used vs. the 75 W budget',
    )
