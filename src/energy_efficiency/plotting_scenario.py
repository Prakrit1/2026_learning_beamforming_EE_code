
import os
import sys

os.environ.pop('EE_SAT_GAIN_DBI', None)
os.environ.pop('EE_POWER_BUDGET_WATT', None)
os.environ.pop('EE_TARGET_ELEVATION_DEG', None)

import gzip
import pickle
import re
from pathlib import Path

import numpy as np

from src.config.config_plotting import PlotConfig
from src.plotting.plotting import plot_rate_error_sweep

PLOT_ONLY = '--plot-only' in sys.argv

error_sweep_range = np.linspace(0, 0.10, 11)
monte_carlo_iterations = 10000

CHECKPOINTS = {
    'aod0.0': 'EE_dinkelbach_adaptive_lwin5000_N16K3_satg30_p75_eta0.6_rawpow',
    'aod0.025': 'EE_dinkelbach_adaptive_aod0.025_lwin5000_N16K3_satg30_p75_eta0.6_rawpow',
    'aod0.05': 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_satg30_p75_eta0.6_rawpow',
}

# Genuine rate-only (RM) baseline for the error_sweep_sumrate figure --
# EE_REWARD_MODE=sum_rate_only, same 3GPP Set-1 system params (see
# SAC_rateonly_satg30_p75_nadir.slurm). Evaluated both at its native full 75 W
# budget (rm_fullpower) and matched to the EE policy's own measured per-error
# power (rm_35w, equal-power RM-vs-EE comparison, ~35 W at Delta-eps=0).
RM_TRAINING_NAME = 'SAC_rateonly_N16K3_satg30_p75_eta0.6_rawpow'

# Per-error genuine RM checkpoints for the error_sweep_training_triplet figure:
# each RM model is trained at ITS OWN error bound, at the power budget closest
# to the EE policy's operating power at that bound (p35 at Delta-eps=0 where EE
# runs ~35 W; p75 at 0.025/0.05), so the triplet compares EE^{Delta-eps} against
# an RM policy genuinely trained at the same Delta-eps -- not one Delta-eps=0 RM
# checkpoint re-matched three times. See SAC_rateonly_*_nadir.slurm.
RM_CHECKPOINTS = {
    'aod0.0': 'SAC_rateonly_N16K3_satg30_p35_eta0.6_rawpow',
    'aod0.025': 'SAC_rateonly_aod0.025_N16K3_satg30_p75_eta0.6_rawpow',
    'aod0.05': 'SAC_rateonly_aod0.05_N16K3_satg30_p75_eta0.6_rawpow',
}


def rm_train_watt(training_name):
    """Training-time power budget encoded in an RM checkpoint name (e.g.
    '..._p35_...' -> 35), used for the RM legend superscript."""
    match = re.search(r'_p(\d+)_', training_name)
    return int(match.group(1)) if match else None


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


def run_matched_power_learned_sweep(cfg, label, get_raw_precoder_func, target_mean_power):
    """A learned policy's raw precoder, rescaled per error level to
    target_mean_power[error_idx] -- the fair equal-power comparison against a
    reference (EE) curve whose measured power is target_mean_power. Mirrors
    run_matched_power_mmse_sweep exactly, but for a learned raw precoder
    instead of MMSE, so RM is compared to EE at EE's OWN power at every error
    point (not a fixed watt, which would let RM outspend EE as EE's clip-only
    power drifts down with error)."""
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
            w_raw = get_raw_precoder_func(cfg, user_manager, satellite_manager)
            current_power = np.real(np.trace(np.matmul(w_raw.conj().T, w_raw)))
            w_precoder = w_raw * np.sqrt(target_mean_power[error_idx] / current_power)
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
    _repo_root = Path(__file__).resolve().parents[2]
    out_path = Path(_repo_root, 'outputs', 'metrics', 'EE_lwin5000_3gpp_triplet')
    out_path.mkdir(parents=True, exist_ok=True)
    gzip_path = Path(out_path, 'rate_power_triplet.gzip')

    if not PLOT_ONLY:
        from src.config.config import Config
        from src.data.calc_sum_rate import calc_sum_rate
        from src.data.calc_tx_power_distribution import calc_tx_power_distribution
        from src.data.satellite_manager import SatelliteManager
        from src.data.user_manager import UserManager
        from src.utils.get_precoding import (
            get_precoding_learned,
            get_precoding_learned_clip_only,
            get_precoding_learned_no_norm,
            get_precoding_mmse,
        )
        from src.utils.load_model import load_model
        from src.utils.update_sim import update_sim

        cfg = Config()
        cfg.show_plots = False
        print(f'[system] sat_gain_dBi={cfg.sat_gain_dBi}, budget={cfg.power_constraint_watt} W, '
              f'user_center_aod_earth_deg={cfg.user_center_aod_earth_deg:.2f}')

        results = {}

        # ---- shared full-budget MMSE curve (system-only, same for all 3) ------
        cfg.config_learner.training_name = 'EE_lwin5000_3gpp_triplet'
        results['mmse_nadir'] = run_rate_power_sweep(cfg, 'MMSE (3GPP Set-1, nadir, full budget)', get_precoding_mmse)
        results['mmse_nadir']['label'] = 'MMSE (75 W budget)'

        # ---- per-checkpoint SAC (clip-only, "energy-efficient") + matched-power MMSE ----
        for aod_key, training_name in CHECKPOINTS.items():
            cfg.config_learner.training_name = training_name
            model_path = get_best_model_path(cfg.trained_models_path, training_name)
            print(f'[{aod_key}] checkpoint: {model_path}')

            precoder_network, norm_factors = load_model(model_path)
            if norm_factors != {}:
                cfg.config_learner.get_state_args['norm_state'] = True


            delta_eps = aod_key.replace('aod', '')
            sac_result = run_rate_power_sweep(
                cfg, f'SAC (Δε = {delta_eps}, energy-efficient)',
                lambda c, um, sm: get_precoding_learned_clip_only(c, um, sm, norm_factors, precoder_network),
            )
            sac_result['label'] = f'SAC (Δε = {delta_eps}, energy-efficient)'
            sac_result['training_name'] = training_name
            sac_result['checkpoint'] = str(model_path)
            results[f'sac_{aod_key}'] = sac_result


            if aod_key == 'aod0.0':
                fullpower_result = run_rate_power_sweep(
                    cfg, f'SAC ({aod_key}, full power)',
                    lambda c, um, sm: get_precoding_learned(c, um, sm, norm_factors, precoder_network),
                )
                fullpower_result['label'] = 'SAC (75 W budget)'
                fullpower_result['training_name'] = training_name
                fullpower_result['checkpoint'] = str(model_path)
                results[f'sac_{aod_key}_fullpower'] = fullpower_result

            matched_mmse_result = run_matched_power_mmse_sweep(
                cfg, f'MMSE matched-power ({aod_key})', sac_result['mean_power'],
            )
            matched_mmse_result['label'] = f'MMSE (equal power, Δε = {delta_eps})'
            results[f'mmse_matched_{aod_key}'] = matched_mmse_result

        # ---- RM baseline for error_sweep_sumrate: single Delta-eps=0 RM ------
        # checkpoint evaluated at its native full 75 W budget (rm_fullpower) AND
        # matched to the EE policy's OWN measured per-error power (rm_35w), so RM
        # and EE are compared at equal transmit power at every error point. A
        # fixed 35 W would let RM outspend EE at high error (EE's clip-only power
        # drifts down to ~28 W), a spurious RM-beats-EE crossover that is purely
        # a power gap. This block only feeds error_sweep_sumrate; the triplet's
        # per-error RM curves are computed separately below.
        try:
            cfg.config_learner.training_name = RM_TRAINING_NAME
            rm_model_path = get_best_model_path(cfg.trained_models_path, RM_TRAINING_NAME)
            print(f'[RM] checkpoint: {rm_model_path}')
            rm_network, rm_norm_factors = load_model(rm_model_path)
            cfg.config_learner.get_state_args['norm_state'] = (rm_norm_factors != {})

            rm_full = run_rate_power_sweep(
                cfg, 'RM (rate-only, 75 W budget)',
                lambda c, um, sm: get_precoding_learned(c, um, sm, rm_norm_factors, rm_network),
            )
            rm_full['label'] = 'RM (75 W budget)'
            rm_full['training_name'] = RM_TRAINING_NAME
            rm_full['checkpoint'] = str(rm_model_path)
            results['rm_fullpower'] = rm_full

            rm_35w = run_matched_power_learned_sweep(
                cfg, 'RM (rate-only, matched to EE power, aod0.0)',
                lambda c, um, sm: get_precoding_learned_no_norm(c, um, sm, rm_norm_factors, rm_network),
                results['sac_aod0.0']['mean_power'],
            )
            rm_35w['label'] = 'RM (equal power to EE, aod0.0)'
            rm_35w['training_name'] = RM_TRAINING_NAME
            rm_35w['checkpoint'] = str(rm_model_path)
            results['rm_35w'] = rm_35w
        except FileNotFoundError:
            print(f'[warn] RM checkpoint {RM_TRAINING_NAME!r} not found under '
                  f'{cfg.trained_models_path} -- error_sweep_sumrate will fall back '
                  f'to the sac_aod0.0_fullpower placeholder for RM. Sync the '
                  f'SAC_rateonly checkpoint into models/ and rerun.')

        # ---- per-error genuine RM curves for error_sweep_training_triplet -----
        # Each RM model is trained at its OWN Delta-eps (RM_CHECKPOINTS) and
        # evaluated at its OWN energy-efficient power -- clip-only inference, the
        # identical evaluation the EE checkpoints get above -- so every
        # RM^{Delta-eps} curve is drawn at the same energy-efficient operating
        # regime as its EE^{Delta-eps} counterpart (it is NOT rescaled to the EE
        # policy's measured power; each policy keeps the power it naturally emits).
        for aod_key, rm_training_name in RM_CHECKPOINTS.items():
            try:
                cfg.config_learner.training_name = rm_training_name
                rm_model_path = get_best_model_path(cfg.trained_models_path, rm_training_name)
                print(f'[RM {aod_key}] checkpoint: {rm_model_path}')
                rm_network, rm_norm_factors = load_model(rm_model_path)
                cfg.config_learner.get_state_args['norm_state'] = (rm_norm_factors != {})

                rm_m = run_rate_power_sweep(
                    cfg, f'RM (rate-only, energy-efficient power, {aod_key})',
                    lambda c, um, sm: get_precoding_learned_clip_only(c, um, sm, rm_norm_factors, rm_network),
                )
                rm_m['label'] = f'RM (energy-efficient power, {aod_key})'
                rm_m['training_name'] = rm_training_name
                rm_m['checkpoint'] = str(rm_model_path)
                rm_m['train_power_budget'] = rm_train_watt(rm_training_name)
                results[f'rm_matched_{aod_key}'] = rm_m
            except FileNotFoundError:
                print(f'[warn] per-error RM checkpoint {rm_training_name!r} not found '
                      f'under {cfg.trained_models_path} -- error_sweep_training_triplet '
                      f'will be missing its rm_matched_{aod_key} curve. Sync it into '
                      f'models/ and rerun.')

        with gzip.open(gzip_path, 'wb') as file:
            pickle.dump({'error_sweep_range': error_sweep_range, 'results': results}, file=file)
        print(f'Saved: {gzip_path}')

    # ---- plot (reads the gzip just saved above, or an existing one if --plot-only) ----
    with gzip.open(gzip_path, 'rb') as file:
        data = pickle.load(file)

    plot_cfg = PlotConfig()
    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.6

    # Legend text uses measured watts (rounded), not fixed placeholders.
    # trained_watt is the fixed training-time power budget (cfg.power_constraint_watt) --
    # the same for every curve here, since the EE reward's subtraction/lambda term shapes
    # what power the policy converges to *using*, not the budget it was trained under.
    trained_watt = round(data['results']['mmse_nadir']['power_budget'])
    mmse_eval_watt = round(data['results']['mmse_nadir']['mean_power'][0])
    ee_eval_watt = round(data['results']['sac_aod0.0']['mean_power'][0])
    mmse_matched_watt = round(data['results']['mmse_matched_aod0.0']['mean_power'][0])

    # Use the genuine rate-only (RM) checkpoint's curves if they were computed;
    # otherwise fall back to the sac_aod0.0_fullpower placeholder for RM.
    real_rm = 'rm_fullpower' in data['results'] and 'rm_35w' in data['results']

    curves = [
        {'result_key': 'mmse_nadir', 'label': f'MMSE$^{{{trained_watt}}}$, $P={mmse_eval_watt}$ W',
         'color': plot_cfg.cp2['black'], 'marker': '^', 'linestyle': ':', 'markevery': (0, 3)},
    ]

    if real_rm:
        # RM (gold) as two curves of the same policy: its native 75 W budget
        # (solid) and matched to the EE policy's own per-error power (dashed,
        # ~35 W at Δε=0), so RM-vs-EE is read off at EQUAL transmit power at
        # every error point.
        rm75_eval_watt = round(data['results']['rm_fullpower']['mean_power'][0])
        rm35_eval_watt = round(data['results']['rm_35w']['mean_power'][0])
        curves += [
            {'result_key': 'rm_fullpower', 'label': f'RM$^{{{trained_watt}}}$, $P={rm75_eval_watt}$ W',
             'color': plot_cfg.cp2['gold'], 'marker': 's', 'linestyle': '-', 'markevery': (1, 3)},
            {'result_key': 'rm_35w', 'label': f'RM$^{{{trained_watt}}}$, $P={rm35_eval_watt}$ W',
             'color': plot_cfg.cp2['gold'], 'marker': 's', 'linestyle': '--', 'markevery': (0, 3), 'marker_dx': 0.0025},
        ]
    else:
        # placeholder RM: reuse the EE (Δε=0.0) checkpoint's full-power inference
        # until the real SAC_rateonly checkpoint is synced into models/.
        rm_eval_watt = round(data['results']['sac_aod0.0_fullpower']['mean_power'][0])
        curves.append(
            {'result_key': 'sac_aod0.0_fullpower', 'label': f'RM$^{{{trained_watt}}}$, $P={rm_eval_watt}$ W',
             'color': plot_cfg.cp2['gold'], 'marker': 's', 'linestyle': '-', 'markevery': (1, 3)})

    curves += [
        {'result_key': 'sac_aod0.0', 'label': f'EE$^{{{trained_watt}}}$, $P={ee_eval_watt}$ W',
         'color': plot_cfg.cp2['green'], 'marker': 'o', 'linestyle': '-'},
        {'result_key': 'mmse_matched_aod0.0', 'label': f'MMSE$^{{{trained_watt}}}$, $P={mmse_matched_watt}$ W',
         'color': plot_cfg.cp2['black'], 'marker': 'x', 'linestyle': '--'},
    ]

    plot_rate_error_sweep(
        error_sweep_range=data['error_sweep_range'],
        results=data['results'],
        curves=curves,
        width=plot_width,
        height=plot_height,
        plots_parent_path=plot_cfg.plots_parent_path,
        name='error_sweep_sumrate',
        legend_ncols=3,
        legend_loc='lower center',
        legend_bbox_to_anchor=(0.5, 1.02),
        legend_fontsize=9,
    )
