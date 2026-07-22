"""
Smoke test for the action_bound_mode='tanh' override (EE_sac.py's __main__):
1. Confirms cfg.config_learner.algorithm_args['action_bound_mode'] actually
   reads 'tanh' (not silently left at the config.py default of None) --
   algorithm_args is a one-time shallow copy taken in _post_init(), same
   trap as target_entropy/hidden_layer_units before it.
2. Runs 3 tiny episodes to confirm nothing crashes with bounded actions.
3. Prints the RAW (pre-clip) precoder power for a few forward passes AFTER
   those tiny episodes, to see whether tanh-bounding the raw action alone
   already brings raw power into a saner range relative to the 100W budget,
   or whether it's merely now bounded-but-still-far-off (in which case a
   sensible fixed scale factor on top of the tanh output may also be
   needed). This is diagnostic only -- 3 episodes is nowhere near enough to
   see what a full 13,000-episode run converges to, this just checks the
   override works and nothing explodes.

Headless (Agg backend, show_plots=False) and CPU-only, matching the
established smoke-test convention -- must run via sbatch per this session's
policy, not directly in an interactive shell.
"""
import matplotlib
matplotlib.use('Agg')

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import numpy as np

from src.config.config import Config
from src.models.EE_sac import train_sac_energy_effiency
from src.utils.get_precoding import get_precoding_learned_no_norm
from src.utils.load_model import load_model
from src.data.calc_tx_power_distribution import calc_tx_power_distribution
from src.data.satellite_manager import SatelliteManager
from src.data.user_manager import UserManager
from src.utils.update_sim import update_sim


def get_best_model_path(trained_models_path, training_name):
    from pathlib import Path
    base_path = Path(trained_models_path, training_name, 'base')
    checkpoints = [p for p in base_path.iterdir() if p.is_dir() and 'full_snap' in p.name]
    checkpoints_by_time = sorted(checkpoints, key=lambda p: os.path.getmtime(p))
    return checkpoints_by_time[-1]


print('=== SMOKE TEST: action_bound_mode=tanh ===', flush=True)

cfg = Config()
cfg.show_plots = False
cfg.config_error_model.error_rng_parametrizations['additive_error_on_cosine_of_aod']['args']['low'] = -0.5
cfg.config_error_model.error_rng_parametrizations['additive_error_on_cosine_of_aod']['args']['high'] = 0.5

cfg.config_learner.action_bound_mode = 'tanh'
cfg.config_learner.algorithm_args['action_bound_mode'] = 'tanh'

cfg.config_learner.reward = {'energy_efficiency': 1.0}
training_name = 'smoke_test_action_bound_tanh'
cfg.config_learner.training_name = training_name
cfg.config_learner.training_episodes = 3
cfg.config_learner.training_steps_per_episode = 50
cfg.config_learner.num_parallel_envs = 4
cfg.config_learner.get_state_norm_factors_iterations = 200
cfg.verbosity = 0

print('algorithm_args action_bound_mode:', cfg.config_learner.algorithm_args['action_bound_mode'], flush=True)

train_sac_energy_effiency(config=cfg)
print('=== SMOKE TEST action_bound_mode=tanh: TRAINING PASSED (no crash) ===', flush=True)

# ---- diagnostic: raw (pre-clip) power after tanh bounding ----
print('\n=== Checking raw (pre-clip) precoder power after tanh bounding ===', flush=True)
model_path = get_best_model_path(cfg.trained_models_path, training_name)
precoder_network, norm_factors = load_model(model_path)
if norm_factors != {}:
    cfg.config_learner.get_state_args['norm_state'] = True

cfg.user_distribution_mode = 'uniform'
cfg.user_dist_average = 25000
cfg.user_dist_bound = 0

satellite_manager = SatelliteManager(config=cfg)
user_manager = UserManager(config=cfg)

n_samples = 500
raw_power_samples = np.zeros(n_samples)
for i in range(n_samples):
    update_sim(cfg, satellite_manager, user_manager)
    w_raw = get_precoding_learned_no_norm(cfg, user_manager, satellite_manager, norm_factors, precoder_network)
    raw_power_samples[i] = calc_tx_power_distribution(w_precoder=w_raw).sum()

power_budget = cfg.power_constraint_watt
print(f'Power budget: {power_budget} W', flush=True)
print(f'RAW (pre-clip) power over {n_samples} samples (untrained, 3-episode policy, tanh-bounded actions):', flush=True)
print(f'  mean={raw_power_samples.mean():.2f} W ({100*raw_power_samples.mean()/power_budget:.1f}% of budget)', flush=True)
print(f'  min={raw_power_samples.min():.2f} W, max={raw_power_samples.max():.2f} W, std={raw_power_samples.std():.2f} W', flush=True)
print('=== SMOKE TEST action_bound_mode=tanh: DIAGNOSTIC PASSED ===', flush=True)

# cleanup: this is a throwaway 3-episode checkpoint, not a real experiment
import shutil
from pathlib import Path
shutil.rmtree(Path(cfg.trained_models_path, training_name), ignore_errors=True)
shutil.rmtree(Path(cfg.output_metrics_path, training_name), ignore_errors=True)
print('Cleaned up smoke-test checkpoint/metrics folders.', flush=True)
