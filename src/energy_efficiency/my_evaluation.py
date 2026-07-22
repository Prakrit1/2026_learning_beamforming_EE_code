"test script to evaluate learned performance of SAC"
import numpy as np
from src.config.config import Config
from pathlib import Path
from src.analysis.helpers.test_learned_precoder import test_sac_precoder_clip_only_error_sweep


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
config_rawpow_aod0 = Config()
config_rawpow_aod0.config_learner.training_name = 'EE_dinkelbach_adaptive_lwin5000_N16K3_eta0.6_rawpow'
model_path_rawpow_aod0 = get_best_model_path(
    config_rawpow_aod0.trained_models_path, 'EE_dinkelbach_adaptive_lwin5000_N16K3_eta0.6_rawpow'
)
test_sac_precoder_clip_only_error_sweep(
    config=config_rawpow_aod0,
    model_path=model_path_rawpow_aod0,
    error_sweep_parameter='additive_error_on_cosine_of_aod',
    error_sweep_range=error_sweep_range,
    monte_carlo_iterations=monte_carlo_iterations,
    metrics=['sumrate']
)


# ---------------------------------------------------------------------------
config_rawpow_aod005 = Config()
config_rawpow_aod005.config_learner.training_name = 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_eta0.6_rawpow'
model_path_rawpow_aod005 = get_best_model_path(
    config_rawpow_aod005.trained_models_path, 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_eta0.6_rawpow'
)
test_sac_precoder_clip_only_error_sweep(
    config=config_rawpow_aod005,
    model_path=model_path_rawpow_aod005,
    error_sweep_parameter='additive_error_on_cosine_of_aod',
    error_sweep_range=error_sweep_range,
    monte_carlo_iterations=monte_carlo_iterations,
    metrics=['sumrate']
)


# ---------------------------------------------------------------------------
config_rawpow_aod0025 = Config()
config_rawpow_aod0025.config_learner.training_name = 'EE_dinkelbach_adaptive_aod0.025_lwin5000_N16K3_eta0.6_rawpow'
model_path_rawpow_aod0025 = get_best_model_path(
    config_rawpow_aod0025.trained_models_path, 'EE_dinkelbach_adaptive_aod0.025_lwin5000_N16K3_eta0.6_rawpow'
)
test_sac_precoder_clip_only_error_sweep(
    config=config_rawpow_aod0025,
    model_path=model_path_rawpow_aod0025,
    error_sweep_parameter='additive_error_on_cosine_of_aod',
    error_sweep_range=error_sweep_range,
    monte_carlo_iterations=monte_carlo_iterations,
    metrics=['sumrate']
)


# ---------------------------------------------------------------------------
# Tuned-LR checkpoints (jobs 152303/152304/152305), using each error bound's
# best lr_critic/lr_actor found by the Optuna searches (147529/147530/147531,
# OOM-killed at 65-67/100 trials -- see handoff). training_name suffix
# _lrc..._lra... distinguishes these from the default-LR checkpoints above.
config_lrtuned_aod0 = Config()
config_lrtuned_aod0.config_learner.training_name = 'EE_dinkelbach_adaptive_lwin5000_N16K3_eta0.6_rawpow_lrc4.14e-05_lra2.50e-07'
model_path_lrtuned_aod0 = get_best_model_path(
    config_lrtuned_aod0.trained_models_path, 'EE_dinkelbach_adaptive_lwin5000_N16K3_eta0.6_rawpow_lrc4.14e-05_lra2.50e-07'
)
test_sac_precoder_clip_only_error_sweep(
    config=config_lrtuned_aod0,
    model_path=model_path_lrtuned_aod0,
    error_sweep_parameter='additive_error_on_cosine_of_aod',
    error_sweep_range=error_sweep_range,
    monte_carlo_iterations=monte_carlo_iterations,
    metrics=['sumrate']
)


# ---------------------------------------------------------------------------
config_lrtuned_aod005 = Config()
config_lrtuned_aod005.config_learner.training_name = 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_eta0.6_rawpow_lrc2.50e-06_lra7.22e-05'
model_path_lrtuned_aod005 = get_best_model_path(
    config_lrtuned_aod005.trained_models_path, 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_eta0.6_rawpow_lrc2.50e-06_lra7.22e-05'
)
test_sac_precoder_clip_only_error_sweep(
    config=config_lrtuned_aod005,
    model_path=model_path_lrtuned_aod005,
    error_sweep_parameter='additive_error_on_cosine_of_aod',
    error_sweep_range=error_sweep_range,
    monte_carlo_iterations=monte_carlo_iterations,
    metrics=['sumrate']
)


# ---------------------------------------------------------------------------
# CAUTION: this checkpoint's entropy_scale_alpha diverged during training
# (bounded ~0-1 for the first ~8500 episodes, then exploded exponentially to
# ~1e19 by episode ~11000, still ~1e17 at the end -- see job 152304's log).
# Training completed without a NaN crash (so the existing RuntimeError
# divergence guard never triggered), but the actor's loss was almost
# certainly dominated by the entropy term for the back half of training --
# expect this checkpoint to perform poorly/erratically below.
config_lrtuned_aod0025 = Config()
config_lrtuned_aod0025.config_learner.training_name = 'EE_dinkelbach_adaptive_aod0.025_lwin5000_N16K3_eta0.6_rawpow_lrc1.82e-06_lra1.17e-06'
model_path_lrtuned_aod0025 = get_best_model_path(
    config_lrtuned_aod0025.trained_models_path, 'EE_dinkelbach_adaptive_aod0.025_lwin5000_N16K3_eta0.6_rawpow_lrc1.82e-06_lra1.17e-06'
)
test_sac_precoder_clip_only_error_sweep(
    config=config_lrtuned_aod0025,
    model_path=model_path_lrtuned_aod0025,
    error_sweep_parameter='additive_error_on_cosine_of_aod',
    error_sweep_range=error_sweep_range,
    monte_carlo_iterations=monte_carlo_iterations,
    metrics=['sumrate']
)
