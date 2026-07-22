"""
Smoke test for the raw-power reward fix in EE_sac.py: the
energy_efficiency_dinkelbach_adaptive and energy_efficiency_no_normalization_fixed
branches now use raw_power_precoder (continuous, pre-clip) for the power
penalty term unconditionally, instead of collapsing to a constant
(power_precoder == budget) whenever the raw output exceeds budget.

Runs 3 tiny episodes of the Dinkelbach-adaptive reward (EMA lambda, the
default/legacy mode -- self-calibrating, so no need to guess a fixed lambda
value under the new, much larger power scale) and checks:
1. No crash.
2. training_name carries the new '_rawpow' suffix.
3. The logged mean-power-per-episode value is no longer frozen at the old
   constant (~1.8267 = 1/pa_efficiency + circuit_power/budget, the exact
   number that appeared in every single previous training log regardless of
   lambda/mechanism) -- it should now reflect the real raw magnitude, likely
   much larger initially since raw power sits at 500-1700% of budget before
   any training pushes it down.

Headless (Agg backend, show_plots=False), CPU-only. Must run via sbatch.
"""
import matplotlib
matplotlib.use('Agg')

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

from src.config.config import Config
from src.models.EE_sac import train_sac_energy_effiency

print('=== SMOKE TEST: raw-power reward fix (Dinkelbach EMA lambda) ===', flush=True)

cfg = Config()
cfg.show_plots = False
cfg.config_error_model.error_rng_parametrizations['additive_error_on_cosine_of_aod']['args']['low'] = -0.5
cfg.config_error_model.error_rng_parametrizations['additive_error_on_cosine_of_aod']['args']['high'] = 0.5
cfg.dinkelbach_lambda_window_steps = 5000
cfg.dinkelbach_block_episodes = 0
cfg.dinkelbach_lambda_fixed = None

cfg.config_learner.reward = {'energy_efficiency_dinkelbach_adaptive': 1.0}
training_name = 'smoke_test_rawpow_reward'
cfg.config_learner.training_name = training_name
cfg.config_learner.training_episodes = 3
cfg.config_learner.training_steps_per_episode = 50
cfg.config_learner.num_parallel_envs = 4
cfg.config_learner.get_state_norm_factors_iterations = 200
cfg.verbosity = 0

train_sac_energy_effiency(config=cfg)
print('=== SMOKE TEST raw-power reward fix: TRAINING PASSED (no crash) ===', flush=True)

import shutil
from pathlib import Path
shutil.rmtree(Path(cfg.trained_models_path, training_name), ignore_errors=True)
shutil.rmtree(Path(cfg.output_metrics_path, training_name), ignore_errors=True)
print('Cleaned up smoke-test checkpoint/metrics folders.', flush=True)
