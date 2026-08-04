"""
Smoke test for the power-conditioned reward mode (EE_REWARD_MODE=
energy_efficiency_power_conditioned in EE_sac.py's __main__, backed by the
power_cond_* block in train_sac_energy_effiency).

Design recap: instead of pricing power inside the reward (Dinkelbach's lambda),
the power BUDGET is sampled fresh per transition and fed to the actor as an
extra state input; the precoder is always rescaled to exactly that budget
(Scheme I's always-rescale logic, generalized from a hardcoded constant to a
variable input), and the reward is pure rate (no power term). The idea is to
learn the rate-maximizing frontier r*(H, P) in one network, then pick the
EE-optimal P via a cheap offline search afterwards instead of a live
concurrently-trained multiplier.

This test checks, in order:
1. No crash -- in particular, that network_args['size_state'] being bumped by
   +1 (for the appended power feature) doesn't break actor/critic construction
   or the get_action_batch() forward pass (a shape mismatch here would raise,
   not silently misbehave, so "no crash" is a real check).
2. The per-episode "Power-conditioned: episode mean sampled budget ..." log
   line shows a sampled budget inside [power_cond_min_watt, power_cond_max_watt]
   and a plausible (nonzero, non-huge) mean rate -- printed for manual
   eyeballing since 4 tiny episodes isn't enough to see real learning, only
   that the mechanism produces sane numbers.

Headless (Agg), CPU-only, matching the established smoke-test convention --
must run via sbatch, not interactively.
"""
import matplotlib
matplotlib.use('Agg')

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

from src.config.config import Config
from src.models.EE_sac import train_sac_energy_effiency

print('=== SMOKE TEST: power-conditioned reward mode ===', flush=True)

cfg = Config()
cfg.show_plots = False

cfg.config_learner.reward = {'energy_efficiency_power_conditioned': 1.0}
training_name = 'smoke_test_power_conditioned'
cfg.config_learner.training_name = training_name

# large enough that training_minimum_experiences (1024) is crossed partway
# through episode 0, same sizing as smoke_test_target_entropy_anneal.py
cfg.config_learner.training_episodes = 4
cfg.config_learner.training_steps_per_episode = 1200
cfg.config_learner.num_parallel_envs = 4
cfg.config_learner.get_state_norm_factors_iterations = 200
cfg.verbosity = 0

# bypass EE_sac.py's __main__ env-var parsing and set power-conditioning
# config directly, same pattern smoke_test_action_bound.py uses for
# action_bound_mode
cfg.power_cond_enabled = True
cfg.power_cond_min_watt = 1.0
cfg.power_cond_max_watt = cfg.power_constraint_watt  # 75 W under 3GPP defaults
cfg.power_cond_sampling = 'log_uniform'
size_state_before = cfg.config_learner.algorithm_args['network_args']['size_state']
cfg.config_learner.algorithm_args['network_args']['size_state'] += 1

print(f'power_cond range: [{cfg.power_cond_min_watt}, {cfg.power_cond_max_watt}] W, '
      f'sampling={cfg.power_cond_sampling}', flush=True)
print(f'network_args size_state: {size_state_before} -> '
      f'{cfg.config_learner.algorithm_args["network_args"]["size_state"]} (+1 for power input)', flush=True)

train_sac_energy_effiency(config=cfg)
print('=== SMOKE TEST power-conditioned: TRAINING PASSED (no crash) ===', flush=True)
print('Check the per-episode "Power-conditioned: episode mean sampled budget ..." log lines above:', flush=True)
print('  sampled budget should land inside [1.0, 75.0] W, mean rate should be a plausible', flush=True)
print('  finite bps/Hz-scale number (roughly in the single digits to low tens, not 0 or huge)', flush=True)

# cleanup: this is a throwaway checkpoint, not a real experiment
import shutil
from pathlib import Path
shutil.rmtree(Path(cfg.trained_models_path, training_name), ignore_errors=True)
shutil.rmtree(Path(cfg.output_metrics_path, training_name), ignore_errors=True)
print('Cleaned up smoke-test checkpoint/metrics folders.', flush=True)
