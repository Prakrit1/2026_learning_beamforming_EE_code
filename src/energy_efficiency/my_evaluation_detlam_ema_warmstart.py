"""test script to evaluate learned performance of SAC: the four
ema0.3+warmstart detlam retrains (jobs 155291-94, 2026-07-30 session),
launched to address the undertraining finding in the original four detlam
checkpoints (154760/762/783/785). Same four aod/fairness combinations
(aod0.0/aod0.05, with/without fairness=1.5), all nadir, 3GPP Set-1
defaults (30 dBi / 75 W). aod=0.025 excluded throughout (diverged to NaN).

MMSE reference is NOT re-run: unchanged system, reused from the existing
3GPP nadir MMSE gzip already on disk.
"""
import os
from pathlib import Path

import numpy as np

from src.config.config import Config
from src.analysis.helpers.test_learned_precoder import test_sac_precoder_clip_only_error_sweep


error_sweep_range = np.linspace(0, 0.10, 11)
monte_carlo_iterations = 10000


def get_best_model_path(trained_models_path, training_name):

    base_path = Path(trained_models_path, training_name, 'base')
    checkpoints = [p for p in base_path.iterdir() if p.is_dir() and 'full_snap' in p.name]
    if not checkpoints:
        raise FileNotFoundError(f'No checkpoints found under {base_path}')

    checkpoints_by_time = sorted(checkpoints, key=lambda p: os.path.getmtime(p))

    max_gap_minutes = 90
    max_gap_seconds = max_gap_minutes * 60

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


# ---- 3GPP Set-1, nadir, no elevation override -----------------------------
os.environ.pop('EE_SAT_GAIN_DBI', None)
os.environ.pop('EE_POWER_BUDGET_WATT', None)
os.environ.pop('EE_TARGET_ELEVATION_DEG', None)

detlam_ema_warmstart_training_names = [
    'EE_dinkelbach_adaptive_detlam512_ema0.3_N16K3_satg30_p75_eta0.6_rawpow_warmstart',
    'EE_dinkelbach_adaptive_aod0.05_detlam512_ema0.3_N16K3_satg30_p75_eta0.6_rawpow_warmstart',
    'EE_dinkelbach_adaptive_detlam512_ema0.3_N16K3_satg30_p75_eta0.6_rawpow_fair1.5_warmstart',
    'EE_dinkelbach_adaptive_aod0.05_detlam512_ema0.3_N16K3_satg30_p75_eta0.6_rawpow_fair1.5_warmstart',
]

for training_name in detlam_ema_warmstart_training_names:
    config = Config()
    config.config_learner.training_name = training_name
    print(f'[{training_name}] sat_gain_dBi={config.sat_gain_dBi}, '
          f'budget={config.power_constraint_watt} W, '
          f'user_center_aod_earth_deg={config.user_center_aod_earth_deg:.2f}')
    model_path = get_best_model_path(config.trained_models_path, training_name)
    test_sac_precoder_clip_only_error_sweep(
        config=config,
        model_path=model_path,
        error_sweep_parameter='additive_error_on_cosine_of_aod',
        error_sweep_range=error_sweep_range,
        monte_carlo_iterations=monte_carlo_iterations,
        metrics=['sumrate'],
    )
