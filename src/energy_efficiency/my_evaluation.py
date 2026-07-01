"test script to evaluate learned performance of SAC"
import numpy as np
from src.config.config import Config
from pathlib import Path
from src.analysis.helpers.test_mmse_precoder import test_mmse_precoder_error_sweep
from src.analysis.helpers.test_learned_precoder import test_sac_precoder_error_sweep, test_sac_precoder_clip_only_error_sweep

error_sweep_range = np.arange(0, 0.11, 0.01)
monte_carlo_iterations = 10000

def get_best_model_path(trained_models_path, training_name):
    """
    Return the checkpoint with the highest mean reward, but restricted to
    the most recent training SESSION in this folder.

    IMPORTANT: a folder can accumulate checkpoints from multiple, unrelated
    training runs over time if the same training_name is reused (or if an
    earlier experiment's folder was never cleaned up). Sorting purely by the
    reward value embedded in the filename -- as the original version of this
    function did -- will silently pick whichever checkpoint has the highest
    reward NUMBER, even if it is weeks old and from a completely different
    reward formula or codebase version.

    This version identifies session boundaries by GAPS between consecutive
    checkpoints (sorted by time), not by a fixed window measured from the
    most recent checkpoint. A fixed window can incorrectly span multiple
    distinct sessions if they happen to fall within e.g. 24h of each other
    on the same day (this was verified against a real case: checkpoints
    from 09:26-09:35 and a later, actually-current run starting at 17:16
    the same day both fell within a naive 24h window of the final
    checkpoint, incorrectly merging two unrelated sessions). Instead: sort
    all checkpoints by time, walk backward from the most recent one, and
    stop as soon as a gap larger than `max_gap_minutes` is found between two
    consecutive checkpoints. Everything from the most recent checkpoint back
    to (but not including) that gap is considered the current session.
    """
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
# energy_efficiency reward (Scheme I -- paper's always-rescale-to-budget
# normalization) -> saved under training_name 'full_EE'
# (confirmed via filesystem ls and training log: capital EE, not lowercase)
# -> saves eval output to outputs/metrics/full_EE/error_sweep/
# ---------------------------------------------------------------------------
config = Config()
config.config_learner.training_name = 'full_EE'
model_path_energy_efficiency = get_best_model_path(config.trained_models_path, 'full_EE')
test_sac_precoder_error_sweep(
    config=config,
    model_path=model_path_energy_efficiency,
    error_sweep_parameter='additive_error_on_cosine_of_aod',
    error_sweep_range=error_sweep_range,
    monte_carlo_iterations=monte_carlo_iterations,
    metrics=['sumrate']
)
test_mmse_precoder_error_sweep(
    config=config,
    error_sweep_parameter='additive_error_on_cosine_of_aod',
    error_sweep_range=error_sweep_range,
    monte_carlo_iterations=monte_carlo_iterations,
    metrics=['sumrate']
)

# ---------------------------------------------------------------------------
# energy_efficiency_no_normalization_fixed reward (Scheme II/III -- clip-
# only, genuine inequality constraint, can transmit below budget)
# -> saved under training_name 'Energy_efficiency without normalization fix'
#    (confirmed directly from the training log: this exact string, spaces
#    and capitalization included, is what config.config_learner.training_name
#    was set to for this run)
# -> saves eval output to outputs/metrics/Energy_efficiency without normalization fix/error_sweep/
# ---------------------------------------------------------------------------
config2 = Config()
config2.config_learner.training_name = 'Energy_efficiency_without_normalization_fix'
model_path_no_norm_fixed = get_best_model_path(
    config2.trained_models_path, 'Energy_efficiency_without_normalization_fix'
)
# Prakrit added this for unnormalized (clip-only) power evaluation -- previously
# this called test_sac_precoder_error_sweep, which always rescales the
# precoder to the power budget and hid this model's real (below-budget) power
# behavior
test_sac_precoder_clip_only_error_sweep(
    config=config2,
    model_path=model_path_no_norm_fixed,
    error_sweep_parameter='additive_error_on_cosine_of_aod',
    error_sweep_range=error_sweep_range,
    monte_carlo_iterations=monte_carlo_iterations,
    metrics=['sumrate']
)
test_mmse_precoder_error_sweep(
    config=config2,
    error_sweep_parameter='additive_error_on_cosine_of_aod',
    error_sweep_range=error_sweep_range,
    monte_carlo_iterations=monte_carlo_iterations,
    metrics=['sumrate']
)

# ---------------------------------------------------------------------------
# Reminder: this script only evaluates SUM RATE over the CSIT error sweep.
# It does NOT report transmit power, so it cannot tell you whether
# 'energy_efficiency_no_normalization_fixed' actually used less than the
# full power budget -- that is what the tx_power_histogram diagnostic script
# (with the pinned-fraction check) is for, run separately. Rate and power
# are two different, complementary results; this script gives you the
# former, matching the style of the source paper's Fig. 3/4, while the
# histogram script gives you the latter, matching your original Image 2/3.
# ---------------------------------------------------------------------------
