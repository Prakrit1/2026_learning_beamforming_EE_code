"""
Diagnostic: is the clip masking the reward gradient's effect on power?

The fixed-lambda power-distribution check (tx_power_distribution.py) showed
that POST-clip transmit power stays pinned near the budget across every
lambda from 0.0 to 5.0 -- only 1-2% of samples land below budget even at the
most aggressive penalty tested. That's the number that's actually
transmitted, but it doesn't say whether the underlying policy is doing
*anything* differently as lambda increases, because clip_only precoding
(src/utils/norm_precoder.py's clip_precoder_to_power_budget) rescales down
to exactly the budget whenever the RAW network output exceeds it -- so if
the raw output is almost always over budget regardless of lambda, the clip
alone could produce an identical flat post-clip curve even if the raw
(pre-clip) output is responding to lambda in some way that never survives
the clip.

This script measures the RAW (pre-clip) trace/power for the same 6
fixed-lambda checkpoints (jobs 145428-145433), computed from the exact same
raw network sample the clip would have acted on (not a second independent
forward pass), and reports:
- mean raw (pre-clip) power, vs. the budget
- fraction of samples where the raw output is already at/under budget (i.e.
  the clip would NOT trigger)
- for comparison, the already-known post-clip numbers (recomputed here from
  the same raw samples via clip_precoder_to_power_budget, not re-simulated)

No training involved -- this only runs the already-trained checkpoints
forward, so it's cheap enough to run interactively.
"""
from pathlib import Path

import numpy as np

from src.config.config import Config
from src.utils.get_precoding import get_precoding_learned_no_norm
from src.utils.load_model import load_model
from src.data.calc_tx_power_distribution import calc_tx_power_distribution
from src.data.satellite_manager import SatelliteManager
from src.data.user_manager import UserManager
from src.utils.update_sim import update_sim
from src.utils.norm_precoder import clip_precoder_to_power_budget


def get_best_model_path(trained_models_path, training_name):
    """Session-aware checkpoint selection -- see my_evaluation.py's identical function for the full rationale."""
    import os

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
    return best


def run_preclip_diagnostic(cfg, model_path, label, monte_carlo_iterations=10000):
    precoder_network, norm_factors = load_model(model_path)
    if norm_factors != {}:
        cfg.config_learner.get_state_args['norm_state'] = True

    cfg.user_distribution_mode = 'uniform'
    cfg.user_dist_average = 25000
    cfg.user_dist_bound = 0

    satellite_manager = SatelliteManager(config=cfg)
    user_manager = UserManager(config=cfg)

    raw_power_samples = np.zeros(monte_carlo_iterations)
    clipped_power_samples = np.zeros(monte_carlo_iterations)

    for iter_idx in range(monte_carlo_iterations):
        update_sim(cfg, satellite_manager, user_manager)
        w_raw = get_precoding_learned_no_norm(cfg, user_manager, satellite_manager, norm_factors, precoder_network)
        raw_power_samples[iter_idx] = calc_tx_power_distribution(w_precoder=w_raw).sum()

        # clip the SAME raw sample directly (not a second network forward
        # pass), so pre/post-clip come from identical raw output
        w_clipped = clip_precoder_to_power_budget(
            precoding_matrix=w_raw,
            power_constraint_watt=cfg.power_constraint_watt,
            per_satellite=True,
            sat_nr=cfg.sat_nr,
            sat_ant_nr=cfg.sat_ant_nr,
        )
        clipped_power_samples[iter_idx] = calc_tx_power_distribution(w_precoder=w_clipped).sum()

        if iter_idx % 2000 == 0:
            print(f'[{label}] {iter_idx}/{monte_carlo_iterations}')

    return raw_power_samples, clipped_power_samples


if __name__ == '__main__':
    lambda_values = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]

    print('\n' + '=' * 70)
    print('PRE-CLIP vs POST-CLIP POWER DIAGNOSTIC (fixed-lambda sweep)')
    print('=' * 70)

    for lv in lambda_values:
        training_name = f'EE_dinkelbach_adaptive_aod0.5_lambdafixed{lv}_N16K3_eta0.6'
        cfg = Config()
        cfg.show_plots = False
        cfg.config_learner.training_name = training_name
        power_budget = cfg.power_constraint_watt

        model_path = get_best_model_path(cfg.trained_models_path, training_name)
        raw_power, clipped_power = run_preclip_diagnostic(cfg, model_path, label=f'lambda={lv}')

        frac_raw_under_budget = (raw_power <= power_budget).mean()
        frac_clipped_under_budget = (clipped_power < power_budget - 0.5).mean()

        print(f'\nlambda={lv}:')
        print(f'  RAW (pre-clip)  power: mean={raw_power.mean():.2f} W '
              f'({100 * raw_power.mean() / power_budget:.1f}% of budget), '
              f'std={raw_power.std():.2f} W, '
              f'min={raw_power.min():.2f} W, max={raw_power.max():.2f} W')
        print(f'  Fraction of samples where RAW output is already <= budget '
              f'(clip would NOT trigger): {100 * frac_raw_under_budget:.1f}%')
        print(f'  CLIPPED (post-clip) power: mean={clipped_power.mean():.2f} W '
              f'({100 * clipped_power.mean() / power_budget:.1f}% of budget), '
              f'below budget (>0.5W margin): {100 * frac_clipped_under_budget:.1f}%')

    print('\n' + '=' * 70)
