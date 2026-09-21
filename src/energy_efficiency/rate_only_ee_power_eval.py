"""
RM (rate-only) TRAINED at the energy-efficient power and EVALUATED at that same
power -- the train-power == eval-power case:
    * Delta-eps=0.025: SAC_rateonly ..._p38  trained at 38 W, evaluated at 38 W
    * Delta-eps=0.05 : SAC_rateonly ..._p40  trained at 40 W, evaluated at 40 W
Each RM is run clip-only, so it transmits at its OWN native (~38/40 W) power --
no downscaling from 75 W, unlike the triplet's p75 RM. This isolates the
power-scaling effect: if the p75 RM sat below EE only because a 75 W-trained beam
was scaled down, then this train==eval RM should sit ON EE; if it still sits
below, the rate-only policy is genuinely weaker than EE at that operating point.

Non-destructive: reads EE / matched-MMSE curves from the existing triplet gzip
(outputs/metrics/EE_lwin5000_3gpp_triplet/rate_power_triplet.gzip) and writes its
own gzip + figure. Does NOT touch the triplet outputs. Run AFTER plotting_scenario
has produced the triplet gzip.
"""
import gzip
import os
import pickle
from pathlib import Path

import matplotlib
import numpy as np

from src.config.config import Config
from src.config.config_plotting import PlotConfig
from src.data.calc_sum_rate import calc_sum_rate
from src.data.calc_tx_power_distribution import calc_tx_power_distribution
from src.data.satellite_manager import SatelliteManager
from src.data.user_manager import UserManager
from src.plotting.plotting import plot_rate_error_sweep
from src.utils.get_precoding import get_precoding_learned_clip_only
from src.utils.load_model import load_model
from src.utils.update_sim import update_sim

error_sweep_range = np.linspace(0, 0.10, 11)
monte_carlo_iterations = 10000

# rate-only checkpoints trained AT the energy-efficient power (p38 / p40)
RM_EE_POWER_CHECKPOINTS = {
    'aod0.025': 'SAC_rateonly_aod0.025_N16K3_satg30_p38_eta0.6_rawpow',
    'aod0.05': 'SAC_rateonly_aod0.05_N16K3_satg30_p40_eta0.6_rawpow',
}


def get_best_model_path(trained_models_path, training_name):
    """Session-aware checkpoint selection -- identical to plotting_scenario.py's."""
    base_path = Path(trained_models_path, training_name, 'base')
    checkpoints = [p for p in base_path.iterdir() if p.is_dir() and 'full_snap' in p.name]
    if not checkpoints:
        raise FileNotFoundError(f'No checkpoints found under {base_path}')
    checkpoints_by_time = sorted(checkpoints, key=lambda p: os.path.getmtime(p))
    max_gap_seconds = 90 * 60
    session_start_idx = len(checkpoints_by_time) - 1
    for i in range(len(checkpoints_by_time) - 1, 0, -1):
        if os.path.getmtime(checkpoints_by_time[i]) - os.path.getmtime(checkpoints_by_time[i - 1]) > max_gap_seconds:
            session_start_idx = i
            break
    else:
        session_start_idx = 0
    same_session = checkpoints_by_time[session_start_idx:]
    return sorted(same_session, key=lambda p: float(p.name.split('_')[-1]))[-1]


def run_clip_only_sweep(cfg, label, get_precoder_func):
    """Native-power (clip-only) rate+power sweep -- mirrors plotting_scenario.run_rate_power_sweep."""
    satellite_manager = SatelliteManager(config=cfg)
    user_manager = UserManager(config=cfg)
    error_param = 'additive_error_on_cosine_of_aod'
    initial = cfg.config_error_model.error_rng_parametrizations[error_param]['args'].copy()

    mean_rate = np.zeros(len(error_sweep_range))
    mean_power = np.zeros(len(error_sweep_range))
    std_rate = np.zeros(len(error_sweep_range))
    std_power = np.zeros(len(error_sweep_range))

    for idx, error_value in enumerate(error_sweep_range):
        cfg.config_error_model.error_rng_parametrizations[error_param]['args']['low'] = -error_value
        cfg.config_error_model.error_rng_parametrizations[error_param]['args']['high'] = error_value
        rate_samples = np.zeros(monte_carlo_iterations)
        power_samples = np.zeros(monte_carlo_iterations)
        for j in range(monte_carlo_iterations):
            update_sim(cfg, satellite_manager, user_manager)
            w_precoder = get_precoder_func(cfg, user_manager, satellite_manager)
            rate_samples[j] = calc_sum_rate(
                channel_state=satellite_manager.channel_state_information,
                w_precoder=w_precoder, noise_power_watt=cfg.noise_power_watt,
            )
            power_samples[j] = calc_tx_power_distribution(w_precoder=w_precoder).sum()
        mean_rate[idx], std_rate[idx] = rate_samples.mean(), rate_samples.std()
        mean_power[idx], std_power[idx] = power_samples.mean(), power_samples.std()
        print(f'[{label}] error={error_value:.2f}: rate={mean_rate[idx]:.4f} bps/Hz, '
              f'power={mean_power[idx]:.2f} W')

    cfg.config_error_model.error_rng_parametrizations[error_param]['args'] = initial
    return {'power_budget': cfg.power_constraint_watt, 'mean_rate': mean_rate, 'std_rate': std_rate,
            'mean_power': mean_power, 'std_power': std_power}


if __name__ == '__main__':
    matplotlib.use('Agg')
    cfg = Config()
    cfg.show_plots = False
    print(f'[system] sat_gain_dBi={cfg.sat_gain_dBi}, budget={cfg.power_constraint_watt} W')

    _repo_root = Path(__file__).resolve().parents[2]
    triplet_gzip = Path(_repo_root, 'outputs', 'metrics', 'EE_lwin5000_3gpp_triplet', 'rate_power_triplet.gzip')
    with gzip.open(triplet_gzip, 'rb') as file:
        tri = pickle.load(file)['results']

    results = {}
    for aod_key, training_name in RM_EE_POWER_CHECKPOINTS.items():
        cfg.config_learner.training_name = training_name
        model_path = get_best_model_path(cfg.trained_models_path, training_name)
        print(f'[{aod_key}] RM checkpoint (trained at EE power): {model_path}')
        network, norm_factors = load_model(model_path)
        cfg.config_learner.get_state_args['norm_state'] = (norm_factors != {})

        rm = run_clip_only_sweep(
            cfg, f'RM (trained+eval at EE power, {aod_key})',
            lambda c, um, sm: get_precoding_learned_clip_only(c, um, sm, norm_factors, network),
        )
        rm['training_name'] = training_name
        rm['checkpoint'] = str(model_path)
        results[f'rm_eepower_{aod_key}'] = rm

        # side-by-side vs the EE curve (and matched-MMSE ceiling) from the triplet gzip
        ee = tri[f'sac_{aod_key}']
        mm = tri.get(f'mmse_matched_{aod_key}')
        print(f'[{aod_key}] error : ' + ' '.join(f'{e:6.2f}' for e in error_sweep_range))
        print(f'[{aod_key}] RM    : ' + ' '.join(f'{v:6.2f}' for v in rm['mean_rate'])
              + f'  (@ {rm["mean_power"][0]:.1f} W)')
        print(f'[{aod_key}] EE    : ' + ' '.join(f'{v:6.2f}' for v in ee['mean_rate'])
              + f'  (@ {ee["mean_power"][0]:.1f} W)')
        if mm is not None:
            print(f'[{aod_key}] MMSEm : ' + ' '.join(f'{v:6.2f}' for v in mm['mean_rate']))

    out_dir = Path(_repo_root, 'outputs', 'metrics', 'EE_rateonly_ee_power')
    out_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(Path(out_dir, 'rate_only_ee_power.gzip'), 'wb') as file:
        pickle.dump({'error_sweep_range': error_sweep_range, 'results': results,
                     'ee_ref': {k: tri[k] for k in ('sac_aod0.025', 'sac_aod0.05',
                                                    'mmse_matched_aod0.025', 'mmse_matched_aod0.05')}},
                    file=file)
    print(f'Saved: {Path(out_dir, "rate_only_ee_power.gzip")}')

    # ---- figure: EE (solid) vs RM trained+evaluated at the SAME EE power (dashed) ----
    plot_cfg = PlotConfig()
    matplotlib.rcParams['text.usetex'] = False  # re-assert after PlotConfig() -- no latex on compute nodes
    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.6

    combined = dict(results)
    combined.update({k: tri[k] for k in ('sac_aod0.025', 'sac_aod0.05')})
    rm38 = round(results['rm_eepower_aod0.025']['mean_power'][0])
    rm40 = round(results['rm_eepower_aod0.05']['mean_power'][0])
    curves = [
        {'result_key': 'sac_aod0.025', 'label': f'EE, Δε=0.025, $P={rm38}$ W',
         'color': plot_cfg.cp2['blue'], 'marker': 'o', 'linestyle': '-', 'markevery': (0, 2)},
        {'result_key': 'rm_eepower_aod0.025', 'label': f'RM (train+eval @ {rm38} W), Δε=0.025',
         'color': plot_cfg.cp2['blue'], 'marker': 's', 'linestyle': '--', 'markevery': (1, 2)},
        {'result_key': 'sac_aod0.05', 'label': f'EE, Δε=0.05, $P={rm40}$ W',
         'color': plot_cfg.cp2['magenta'], 'marker': 'o', 'linestyle': '-', 'markevery': (0, 2)},
        {'result_key': 'rm_eepower_aod0.05', 'label': f'RM (train+eval @ {rm40} W), Δε=0.05',
         'color': plot_cfg.cp2['magenta'], 'marker': 's', 'linestyle': '--', 'markevery': (1, 2)},
    ]
    plot_rate_error_sweep(
        error_sweep_range=error_sweep_range, results=combined, curves=curves,
        width=plot_width, height=plot_height, plots_parent_path=plot_cfg.plots_parent_path,
        name='rate_only_ee_power', annotate_power=False,
        legend_ncols=2, legend_loc='lower center', legend_bbox_to_anchor=(0.5, 1.02), legend_fontsize=8,
    )
    print('Done.')
