"""
Smoke test for target_entropy annealing (EE_TARGET_ENTROPY_ANNEAL_END and
friends in EE_sac.py's __main__, backed by SoftActorCritic.set_target_entropy()
in soft_actor_critic.py).

This exists to catch one specific, easy-to-get-silently-wrong failure mode:
train_graph() is @tf.function-traced and reads self.target_entropy inside the
graph. If target_entropy were still a plain Python float (as it was before
this change), mutating it between episodes would have NO effect on the
already-traced graph -- the old value gets baked in as a constant at first
trace, and training would silently keep using target_entropy from episode 0
forever, even though the annealed value logged each episode looks correct.
Making target_entropy a tf.Variable (mirroring how log_entropy_scale_alpha
already is one) and updating it via .assign() is supposed to fix this -- this
test verifies the fix actually took effect, not just that nothing crashed.

Design: anneal target_entropy from 1.0 (config default) to -50.0 over just 2
episodes, then hold at -50.0. -50.0 is deliberately extreme relative to any
realistic policy log-prob-density value, so if the mutation is taking effect,
entropy_scale_alpha (auto-tuned via alpha_loss = -log_alpha * mean(logprob +
target_entropy)) should crash sharply toward 0 as soon as episode 1 starts
training against it. If the retrace bug were still present, alpha would keep
drifting as if target_entropy were still 1.0 for the entire run, decoupled
from the logged (but ineffective) target_entropy value.

Uses reward_mode='energy_efficiency' (Scheme I) deliberately -- this test is
about the entropy mechanism, not the Dinkelbach lambda mechanism, so the
fewer moving parts the better. Headless (Agg), CPU-only, matching the
established smoke-test convention -- must run via sbatch, not interactively.
"""
import matplotlib
matplotlib.use('Agg')

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

from src.config.config import Config
from src.models.EE_sac import train_sac_energy_effiency

print('=== SMOKE TEST: target_entropy annealing ===', flush=True)

cfg = Config()
cfg.show_plots = False

cfg.config_learner.reward = {'energy_efficiency': 1.0}
training_name = 'smoke_test_target_entropy_anneal'
cfg.config_learner.training_name = training_name

# large enough that training_minimum_experiences (1024) is crossed partway
# through episode 0, so entropy training (and thus the anneal) is actually
# exercised within these few episodes, not just buffer warm-up
cfg.config_learner.training_episodes = 4
cfg.config_learner.training_steps_per_episode = 1200
cfg.config_learner.num_parallel_envs = 4
cfg.config_learner.get_state_norm_factors_iterations = 200
cfg.verbosity = 0

# bypass EE_sac.py's __main__ env-var parsing and set the anneal config
# directly, same pattern smoke_test_action_bound.py uses for action_bound_mode
cfg.target_entropy_anneal_start = cfg.config_learner.algorithm_args['target_entropy']  # 1.0
cfg.target_entropy_anneal_end = -50.0
cfg.target_entropy_anneal_episodes = 2

print(f'target_entropy schedule: {cfg.target_entropy_anneal_start} -> '
      f'{cfg.target_entropy_anneal_end} over {cfg.target_entropy_anneal_episodes} episodes '
      f'(then held), total {cfg.config_learner.training_episodes} episodes', flush=True)

train_sac_energy_effiency(config=cfg)
print('=== SMOKE TEST target_entropy anneal: TRAINING PASSED (no crash) ===', flush=True)
print('Check the per-episode "target_entropy: ..." and "entropy_scale_alpha: ..." log lines above:', flush=True)
print('  episode 0 should show target_entropy=1.0000, episodes 1-3 should show target_entropy=-50.0000', flush=True)
print('  entropy_scale_alpha should start crashing toward 0 once target_entropy=-50.0000 takes effect', flush=True)
print('  (if entropy_scale_alpha instead keeps drifting as if nothing changed, the tf.Variable fix did not take)', flush=True)

# cleanup: this is a throwaway checkpoint, not a real experiment
import shutil
from pathlib import Path
shutil.rmtree(Path(cfg.trained_models_path, training_name), ignore_errors=True)
shutil.rmtree(Path(cfg.output_metrics_path, training_name), ignore_errors=True)
print('Cleaned up smoke-test checkpoint/metrics folders.', flush=True)
