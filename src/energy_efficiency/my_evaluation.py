"test script to evaluate learned performance of SAC"
import gzip
import os
import pickle

import numpy as np
from src.config.config import Config
from pathlib import Path
from src.analysis.helpers.test_learned_precoder import test_sac_precoder_clip_only_error_sweep
from src.analysis.helpers.test_precoder_error_sweep import test_precoder_error_sweep
from src.data.calc_sum_rate import calc_sum_rate
from src.utils.get_precoding import get_precoding_mmse


error_sweep_range = np.linspace(0, 0.10, 11)
monte_carlo_iterations = 10000

def get_best_model_path(trained_models_path, training_name):

    import os

    base_path = Path(trained_models_path, training_name, 'base')
    checkpoints = [p for p in base_path.iterdir() if p.is_dir() and 'full_snap' in p.name]
    if not checkpoints:
        raise FileNotFoundError(f'No checkpoints found under {base_path}')

    checkpoints_by_time = sorted(checkpoints, key=lambda p: os.path.getmtime(p))

    # max allowed gap between consecutive checkpoints within one continuous
    # session, in minutes. Checkpoints are typically saved every time a new
    # high score is hit, which can be frequent early in training and sparse
    # later -- adjust if your training produces longer natural gaps between
    # improvements within a single legitimate run.
    max_gap_minutes = 90
    max_gap_seconds = max_gap_minutes * 60

    # walk backward from the most recent checkpoint, stop at the first gap
    # larger than max_gap_seconds
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

    excluded = len(checkpoints) - len(same_session_checkpoints)
    if excluded > 0:
        excluded_names = [p.name for p in checkpoints_by_time[:session_start_idx]]
        print(f'[get_best_model_path] WARNING: excluded {excluded} checkpoint(s) '
              f'from an earlier session (gap > {max_gap_minutes} min detected): '
              f'{excluded_names}. If this is unexpected, check {base_path} manually.')

    return best


# ---------------------------------------------------------------------------
# Current checkpoint set: the two 3GPP Set-1 models (jobs 153655 nadir /
# 153656 elev30, trained 2026-07-27), evaluated at the CURRENT config
# defaults (30 dBi / 75 W). The elev30 block sets EE_TARGET_ELEVATION_DEG=30
# so users are placed in the geometry that checkpoint was trained for. MMSE
# is evaluated alongside in each geometry as the reference -- every MMSE
# gzip from before 2026-07-27 was generated at the old 20 dBi / 100 W link
# budget and cannot be reused at the current parameters.
#
# The pre-2026-07-27 checkpoint evaluations that used to live in this file
# (aod=0.0/0.025/0.05 + lr-tuned variants) were REMOVED, not merely skipped:
# their correct results are already on disk under
# outputs/metrics/<training_name>/error_sweep/ and serve as the benchmark
# curves in my_plotting.py; re-running them against the current 3GPP
# defaults would overwrite those gzips with numbers from a link budget those
# models were never trained for. See git history for the removed blocks if
# they ever need re-running (they then need the EE_SAT_GAIN_DBI=20 /
# EE_POWER_BUDGET_WATT=100 environment pins to reproduce their own system).
# ---------------------------------------------------------------------------

# ---- 3GPP Set-1, nadir, error=0.0 training bound --------------------------
# Points at job 156358's checkpoint (aod=0.0, lwin5000, natively trained at
# 30 dBi/75 W) -- the checkpoint that fills the "clean lwin5000-vs-lwin5000"
# gap noted in the 2026-08-03 handoff section, replacing the previous
# aod=0.05 nadir block. If re-pointing this file at a different training
# bound again, double check EVERY block below that reads training_name --
# it is easy to change this one and miss a downstream dependency (see the
# matched-power block removed just below, which broke exactly this way).
os.environ.pop('EE_SAT_GAIN_DBI', None)
os.environ.pop('EE_POWER_BUDGET_WATT', None)
os.environ.pop('EE_TARGET_ELEVATION_DEG', None)

NADIR_TRAINING_NAME = 'EE_dinkelbach_adaptive_lwin5000_N16K3_satg30_p75_eta0.6_rawpow'

config_3gpp_nadir = Config()
config_3gpp_nadir.config_learner.training_name = NADIR_TRAINING_NAME
print(f'[3gpp_nadir] sat_gain_dBi={config_3gpp_nadir.sat_gain_dBi}, '
      f'budget={config_3gpp_nadir.power_constraint_watt} W, '
      f'user_center_aod_earth_deg={config_3gpp_nadir.user_center_aod_earth_deg:.2f}')
model_path_3gpp_nadir = get_best_model_path(
    config_3gpp_nadir.trained_models_path, NADIR_TRAINING_NAME
)
test_sac_precoder_clip_only_error_sweep(
    config=config_3gpp_nadir,
    model_path=model_path_3gpp_nadir,
    error_sweep_parameter='additive_error_on_cosine_of_aod',
    error_sweep_range=error_sweep_range,
    monte_carlo_iterations=monte_carlo_iterations,
    metrics=['sumrate']
)
# MMSE reference, same geometry/system; its gzip lands in this model's folder
test_precoder_error_sweep(
    config=config_3gpp_nadir,
    error_sweep_parameter='additive_error_on_cosine_of_aod',
    error_sweep_range=error_sweep_range,
    precoder_name='mmse',
    monte_carlo_iterations=monte_carlo_iterations,
    get_precoder_func=get_precoding_mmse,
    calc_reward_funcs=[calc_sum_rate],
)

# MMSE at MATCHED power: rescaled per error level to the transmit power the
# aod=0.0 nadir SAC actually used there (measured means from
# rate_power_3gpp_pilots.py's 2026-08-04 rerun, ~28.6-34.9 W of the 75 W
# budget -- see that script's '3gpp_nadir' entry, now correctly the aod=0.0
# checkpoint's own power after being re-run against it). Full-budget MMSE
# above shows what 75 W buys; only this equal-power version isolates
# whether SAC's beamforming itself competes.
_rate_power_gzip = Path(config_3gpp_nadir.output_metrics_path,
                        'EE_3gpp_pilots_rate_power_sweep', 'rate_power_3gpp_pilots.gzip')
with gzip.open(_rate_power_gzip, 'rb') as _file:
    _rate_power_data = pickle.load(_file)
if not np.allclose(_rate_power_data['error_sweep_range'], error_sweep_range):
    raise ValueError(f'error grid mismatch between {_rate_power_gzip} and this sweep')
_sac_nadir_power_by_error = {
    round(float(err), 6): float(power)
    for err, power in zip(_rate_power_data['error_sweep_range'],
                          _rate_power_data['results']['3gpp_nadir']['mean_power'])
}

def get_precoding_mmse_sac_power(cfg, user_manager, satellite_manager):
    w_mmse = get_precoding_mmse(cfg, user_manager, satellite_manager)
    current_error = cfg.config_error_model.error_rng_parametrizations[
        'additive_error_on_cosine_of_aod']['args']['high']
    target_power = _sac_nadir_power_by_error[round(float(current_error), 6)]
    current_power = np.real(np.trace(np.matmul(w_mmse.conj().T, w_mmse)))
    return w_mmse * np.sqrt(target_power / current_power)

test_precoder_error_sweep(
    config=config_3gpp_nadir,
    error_sweep_parameter='additive_error_on_cosine_of_aod',
    error_sweep_range=error_sweep_range,
    precoder_name='mmse_sacpower',
    monte_carlo_iterations=monte_carlo_iterations,
    get_precoder_func=get_precoding_mmse_sac_power,
    calc_reward_funcs=[calc_sum_rate],
)


# ---- 3GPP Set-1, target elevation 30 deg ----------------------------------
os.environ['EE_TARGET_ELEVATION_DEG'] = '30'

config_3gpp_elev30 = Config()
config_3gpp_elev30.config_learner.training_name = 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_satg30_p75_elev30_eta0.6_rawpow'
print(f'[3gpp_elev30] sat_gain_dBi={config_3gpp_elev30.sat_gain_dBi}, '
      f'budget={config_3gpp_elev30.power_constraint_watt} W, '
      f'user_center_aod_earth_deg={config_3gpp_elev30.user_center_aod_earth_deg:.2f}')
model_path_3gpp_elev30 = get_best_model_path(
    config_3gpp_elev30.trained_models_path, 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_satg30_p75_elev30_eta0.6_rawpow'
)
test_sac_precoder_clip_only_error_sweep(
    config=config_3gpp_elev30,
    model_path=model_path_3gpp_elev30,
    error_sweep_parameter='additive_error_on_cosine_of_aod',
    error_sweep_range=error_sweep_range,
    monte_carlo_iterations=monte_carlo_iterations,
    metrics=['sumrate']
)
test_precoder_error_sweep(
    config=config_3gpp_elev30,
    error_sweep_parameter='additive_error_on_cosine_of_aod',
    error_sweep_range=error_sweep_range,
    precoder_name='mmse',
    monte_carlo_iterations=monte_carlo_iterations,
    get_precoder_func=get_precoding_mmse,
    calc_reward_funcs=[calc_sum_rate],
)

os.environ.pop('EE_TARGET_ELEVATION_DEG', None)
