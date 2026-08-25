"""
Rate AND power (same Monte Carlo samples per error point) for the
lwin5000/3GPP-native nadir triplet -- aod=0.0 (job 156358), aod=0.025
(job TBD, first-ever lwin5000/3GPP attempt at this bound), aod=0.05
(job 153655) -- all plain EMA-Dinkelbach SAC, 3GPP Set-1 params
(30 dBi / 75 W), nadir geometry. This is the "clean lwin5000-vs-lwin5000"
comparison called for in the handoff's "what to do next" item 1, replacing
the stale detlam-based parajuli.pdf.

For each checkpoint this script computes, over the same
error_sweep_range/monte_carlo_iterations used throughout this project:
  - the SAC (clip-only precoder, matching how these Dinkelbach-rawpow
    checkpoints are evaluated elsewhere in this repo) rate+power curve --
    referred to as "energy-efficient" below since the clip-only projection
    is what lets the precoder spend less than the full power budget.
  - a MATCHED-POWER MMSE curve: MMSE rescaled, per error level, to that
    checkpoint's OWN measured mean power -- the fair "does SAC's
    beamforming itself compete" comparison used throughout this project's
    history (my_evaluation.py, mmse_matched_power_detlam.py, etc).

The aod=0.0 checkpoint is additionally evaluated with the always-rescale-
to-budget precoder (get_precoding_learned, "full power"), i.e. the SAME
trained network forced to spend the full 75W budget instead of the
clip-only projection -- this isolates the effect of the power projection
itself from the effect of training at a different error bound.

A single FULL-BUDGET MMSE curve is also computed once (all three
checkpoints share the identical nadir/30dBi/75W system, so it is not
checkpoint-dependent) and stored alongside.

This is a "scenario" script: it generates the data for this specific
checkpoint set, then calls the shared plot_rate_error_sweep() in
src/plotting/plotting.py to draw the figure -- same simulate+plot-in-one-
script pattern as beampattern.py (generate_beampatterns()
+ plot_beam_patterns()). Pass --plot-only to skip the (expensive, GPU-bound)
simulation and just re-plot an already-saved gzip.

Current figure (see the `curves` list in __main__): full-budget MMSE,
SAC(train err 0.0) at full power, SAC(train err 0.0) energy-efficient with
its matched-power MMSE, and SAC(train err 0.025/0.05) energy-efficient --
six curves total. Edit `curves` to change what's drawn without re-running
the simulation (--plot-only), or edit CHECKPOINTS to change what gets
simulated in the first place.

Saves one gzip: outputs/metrics/EE_lwin5000_3gpp_triplet/rate_power_triplet.gzip
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
from src.data.calc_sum_rate import calc_sum_rate
from src.data.calc_tx_power_distribution import calc_tx_power_distribution
from src.data.satellite_manager import SatelliteManager
from src.data.user_manager import UserManager
from src.utils.get_precoding import get_precoding_learned, get_precoding_learned_clip_only, get_precoding_mmse
from src.utils.load_model import load_model
from src.utils.update_sim import update_sim
from src.plotting.plotting import plot_rate_error_sweep

PLOT_ONLY = '--plot-only' in sys.argv

error_sweep_range = np.linspace(0, 0.10, 11)
monte_carlo_iterations = 10000

# aod bound -> training_name, in the order the lwin5000/3GPP triplet was
# scripted (see commit 6c15a56 / handoff "2026-08-04 session"). aod0.025's
# checkpoint directory must be copied into models/ from the cluster before
# this script can run past that point -- it is not present in every sandbox.
CHECKPOINTS = {
    'aod0.0': 'EE_dinkelbach_adaptive_lwin5000_N16K3_satg30_p75_eta0.6_rawpow',
    'aod0.025': 'EE_dinkelbach_adaptive_aod0.025_lwin5000_N16K3_satg30_p75_eta0.6_rawpow',
    'aod0.05': 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_satg30_p75_eta0.6_rawpow',
}


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
    most_recent = checkpoints_by_time[-1]
    if best != most_recent:
        print(f'[get_best_model_path] NOTE: within the current session, '
              f'{best.name} has higher reward than the most recent save '
              f'{most_recent.name} -- using {best.name}.')
    return best


def run_rate_power_sweep(cfg, label, get_precoder_func):
    """get_precoder_func(cfg, user_manager, satellite_manager) -> w_precoder."""
    satellite_manager = SatelliteManager(config=cfg)
    user_manager = UserManager(config=cfg)

    error_param = 'additive_error_on_cosine_of_aod'
    initial_error_config = cfg.config_error_model.error_rng_parametrizations[error_param]['args'].copy()

    mean_rate = np.zeros(len(error_sweep_range))
    mean_power = np.zeros(len(error_sweep_range))
    std_rate = np.zeros(len(error_sweep_range))
    std_power = np.zeros(len(error_sweep_range))

    for error_idx, error_value in enumerate(error_sweep_range):
        cfg.config_error_model.error_rng_parametrizations[error_param]['args']['low'] = -error_value
        cfg.config_error_model.error_rng_parametrizations[error_param]['args']['high'] = error_value

        rate_samples = np.zeros(monte_carlo_iterations)
        power_samples = np.zeros(monte_carlo_iterations)

        for iter_idx in range(monte_carlo_iterations):
            update_sim(cfg, satellite_manager, user_manager)
            w_precoder = get_precoder_func(cfg, user_manager, satellite_manager)
            rate_samples[iter_idx] = calc_sum_rate(
                channel_state=satellite_manager.channel_state_information,
                w_precoder=w_precoder,
                noise_power_watt=cfg.noise_power_watt,
            )
            power_samples[iter_idx] = calc_tx_power_distribution(w_precoder=w_precoder).sum()

        mean_rate[error_idx] = rate_samples.mean()
        std_rate[error_idx] = rate_samples.std()
        mean_power[error_idx] = power_samples.mean()
        std_power[error_idx] = power_samples.std()
        print(f'[{label}] error={error_value:.2f}: rate={mean_rate[error_idx]:.4f} bps/Hz, '
              f'power={mean_power[error_idx]:.2f} W ({100 * mean_power[error_idx] / cfg.power_constraint_watt:.1f}% of budget)')

    cfg.config_error_model.error_rng_parametrizations[error_param]['args'] = initial_error_config

    return {
        'power_budget': cfg.power_constraint_watt,
        'mean_rate': mean_rate, 'std_rate': std_rate,
        'mean_power': mean_power, 'std_power': std_power,
    }


def run_matched_power_mmse_sweep(cfg, label, target_mean_power):
    """MMSE, rescaled per error level to target_mean_power[error_idx] -- the fair
    equal-power comparison against a specific checkpoint's own measured power."""
    satellite_manager = SatelliteManager(config=cfg)
    user_manager = UserManager(config=cfg)

    error_param = 'additive_error_on_cosine_of_aod'
    initial_error_config = cfg.config_error_model.error_rng_parametrizations[error_param]['args'].copy()

    mean_rate = np.zeros(len(error_sweep_range))
    mean_power = np.zeros(len(error_sweep_range))
    std_rate = np.zeros(len(error_sweep_range))
    std_power = np.zeros(len(error_sweep_range))

    for error_idx, error_value in enumerate(error_sweep_range):
        cfg.config_error_model.error_rng_parametrizations[error_param]['args']['low'] = -error_value
        cfg.config_error_model.error_rng_parametrizations[error_param]['args']['high'] = error_value

        rate_samples = np.zeros(monte_carlo_iterations)
        power_samples = np.zeros(monte_carlo_iterations)

        for iter_idx in range(monte_carlo_iterations):
            update_sim(cfg, satellite_manager, user_manager)
            w_mmse = get_precoding_mmse(cfg, user_manager, satellite_manager)
            current_power = np.real(np.trace(np.matmul(w_mmse.conj().T, w_mmse)))
            w_precoder = w_mmse * np.sqrt(target_mean_power[error_idx] / current_power)
            rate_samples[iter_idx] = calc_sum_rate(
                channel_state=satellite_manager.channel_state_information,
                w_precoder=w_precoder,
                noise_power_watt=cfg.noise_power_watt,
            )
            power_samples[iter_idx] = calc_tx_power_distribution(w_precoder=w_precoder).sum()

        mean_rate[error_idx] = rate_samples.mean()
        std_rate[error_idx] = rate_samples.std()
        mean_power[error_idx] = power_samples.mean()
        std_power[error_idx] = power_samples.std()
        print(f'[{label}] error={error_value:.2f}: rate={mean_rate[error_idx]:.4f} bps/Hz, '
              f'power={mean_power[error_idx]:.2f} W (target {target_mean_power[error_idx]:.2f} W)')

    cfg.config_error_model.error_rng_parametrizations[error_param]['args'] = initial_error_config

    return {
        'power_budget': cfg.power_constraint_watt,
        'mean_rate': mean_rate, 'std_rate': std_rate,
        'mean_power': mean_power, 'std_power': std_power,
    }


if __name__ == '__main__':
    cfg = Config()
    cfg.show_plots = False
    print(f'[system] sat_gain_dBi={cfg.sat_gain_dBi}, budget={cfg.power_constraint_watt} W, '
          f'user_center_aod_earth_deg={cfg.user_center_aod_earth_deg:.2f}')

    out_path = Path(cfg.output_metrics_path, 'EE_lwin5000_3gpp_triplet')
    out_path.mkdir(parents=True, exist_ok=True)
    gzip_path = Path(out_path, 'rate_power_triplet.gzip')

    if not PLOT_ONLY:
        results = {}

        # ---- shared full-budget MMSE curve (system-only, same for all 3) ------
        cfg.config_learner.training_name = 'EE_lwin5000_3gpp_triplet'
        results['mmse_nadir'] = run_rate_power_sweep(cfg, 'MMSE (3GPP Set-1, nadir, full budget)', get_precoding_mmse)
        results['mmse_nadir']['label'] = 'MMSE (full budget, 75 W)'

        # ---- per-checkpoint SAC (clip-only, "energy-efficient") + matched-power MMSE ----
        for aod_key, training_name in CHECKPOINTS.items():
            cfg.config_learner.training_name = training_name
            model_path = get_best_model_path(cfg.trained_models_path, training_name)
            print(f'[{aod_key}] checkpoint: {model_path}')

            precoder_network, norm_factors = load_model(model_path)
            if norm_factors != {}:
                cfg.config_learner.get_state_args['norm_state'] = True

            sac_result = run_rate_power_sweep(
                cfg, f'SAC ({aod_key}, energy-efficient)',
                lambda c, um, sm: get_precoding_learned_clip_only(c, um, sm, norm_factors, precoder_network),
            )
            sac_result['label'] = f'SAC (train err {aod_key.replace("aod", "")}, energy-efficient)'
            sac_result['training_name'] = training_name
            sac_result['checkpoint'] = str(model_path)
            results[f'sac_{aod_key}'] = sac_result

            # aod0.0's checkpoint is additionally evaluated with the always-
            # rescale-to-budget precoder -- same trained network, forced to
            # spend the full 75W budget, to isolate the effect of the power
            # projection from the effect of the training error bound.
            if aod_key == 'aod0.0':
                fullpower_result = run_rate_power_sweep(
                    cfg, f'SAC ({aod_key}, full power)',
                    lambda c, um, sm: get_precoding_learned(c, um, sm, norm_factors, precoder_network),
                )
                fullpower_result['label'] = 'SAC (train err 0.0, full power)'
                fullpower_result['training_name'] = training_name
                fullpower_result['checkpoint'] = str(model_path)
                results[f'sac_{aod_key}_fullpower'] = fullpower_result

            matched_mmse_result = run_matched_power_mmse_sweep(
                cfg, f'MMSE matched-power ({aod_key})', sac_result['mean_power'],
            )
            matched_mmse_result['label'] = f'MMSE (equal power to SAC EE, err {aod_key.replace("aod", "")})'
            results[f'mmse_matched_{aod_key}'] = matched_mmse_result

        with gzip.open(gzip_path, 'wb') as file:
            pickle.dump({'error_sweep_range': error_sweep_range, 'results': results}, file=file)
        print(f'Saved: {gzip_path}')

    # ---- plot (reads the gzip just saved above, or an existing one if --plot-only) ----
    with gzip.open(gzip_path, 'rb') as file:
        data = pickle.load(file)

    plot_cfg = PlotConfig()
    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.6

    # Six curves: full-budget MMSE and full-power SAC as the two "spends the
    # whole 75W budget" references (black/gold, distinct markers), the
    # aod0.0 energy-efficient SAC paired by color with its matched-power
    # MMSE (green, solid vs dashed) so the two are visually linked, and the
    # aod0.025/aod0.05 energy-efficient SAC curves in their own colors.
    curves = [
        {'result_key': 'mmse_nadir', 'label': 'MMSE (full budget, 75 W)',
         'color': plot_cfg.cp2['black'], 'marker': '^', 'linestyle': ':'},
        {'result_key': 'sac_aod0.0_fullpower', 'label': 'SAC (train err 0.0, full power)',
         'color': plot_cfg.cp2['gold'], 'marker': 's', 'linestyle': '-'},
        {'result_key': 'sac_aod0.0', 'label': 'SAC (train err 0.0, energy-efficient)',
         'color': plot_cfg.cp2['green'], 'marker': 'o', 'linestyle': '-'},
        {'result_key': 'mmse_matched_aod0.0', 'label': 'MMSE (equal power to SAC EE, err 0.0)',
         'color': plot_cfg.cp2['green'], 'marker': 'x', 'linestyle': '--'},
        {'result_key': 'sac_aod0.025', 'label': 'SAC (train err 0.025, energy-efficient)',
         'color': plot_cfg.cp2['blue'], 'marker': 'o', 'linestyle': '-'},
        {'result_key': 'sac_aod0.05', 'label': 'SAC (train err 0.05, energy-efficient)',
         'color': plot_cfg.cp2['magenta'], 'marker': 'o', 'linestyle': '-'},
    ]

    plot_rate_error_sweep(
        error_sweep_range=data['error_sweep_range'],
        results=data['results'],
        curves=curves,
        width=plot_width,
        height=plot_height,
        plots_parent_path=plot_cfg.plots_parent_path,
        name='error_sweep_sumrate',
    )
