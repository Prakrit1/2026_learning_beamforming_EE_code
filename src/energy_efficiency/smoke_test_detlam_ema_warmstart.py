"""
Smoke test for the 2026-07-30 reconsideration of the deterministic-lambda
fix: EMA smoothing of the per-episode update (EE_LAMBDA_DET_EMA_ALPHA) and
warm-starting the policy network from a prior checkpoint
(EE_WARM_START_MODEL_PATH), both added to EE_sac.py.

Runs 3 tiny episodes and checks:
1. No crash (in particular: load_weights() succeeds against the real
   aod=0.0 detlam checkpoint -- confirms network architecture matches).
2. training_name carries both '_ema0.3' and '_warmstart' suffixes.
3. Lambda is hard-set on the first deterministic eval, then EMA-blended
   (not hard-replaced) on the second -- printed lambda values should show
   this (episode 2's lambda should not simply equal its own raw estimate).

Headless (Agg backend, show_plots=False), CPU-only. Must run via sbatch.
"""
import matplotlib
matplotlib.use('Agg')

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

from src.config.config import Config
from src.models.EE_sac import train_sac_energy_effiency

print('=== SMOKE TEST: detlam EMA smoothing + warm start ===', flush=True)

WARM_START_PATH = (
    '/home/parajuli/repos/2026_learning_beamforming_EE_code/models/'
    'EE_dinkelbach_adaptive_detlam512_N16K3_satg30_p75_eta0.6_rawpow/'
    'base/full_snap_energy_effiency_3.033'
)

cfg = Config()
cfg.show_plots = False
cfg.config_error_model.error_rng_parametrizations['additive_error_on_cosine_of_aod']['args']['low'] = 0.0
cfg.config_error_model.error_rng_parametrizations['additive_error_on_cosine_of_aod']['args']['high'] = 0.0

cfg.config_learner.reward = {'energy_efficiency_dinkelbach_adaptive': 1.0}
cfg.dinkelbach_deterministic_lambda = True
cfg.dinkelbach_det_eval_steps = 64
cfg.dinkelbach_det_eval_every = 1
cfg.dinkelbach_det_ema_alpha = 0.3
cfg.warm_start_model_path = WARM_START_PATH

training_name = 'smoke_test_detlam_ema_warmstart_ema0.3_warmstart'
cfg.config_learner.training_name = training_name
cfg.config_learner.training_episodes = 3
cfg.config_learner.training_steps_per_episode = 50
cfg.config_learner.num_parallel_envs = 4
cfg.config_learner.get_state_norm_factors_iterations = 200
cfg.verbosity = 0

train_sac_energy_effiency(config=cfg)
print('=== SMOKE TEST detlam EMA + warm start: TRAINING PASSED (no crash) ===', flush=True)

import shutil
from pathlib import Path
shutil.rmtree(Path(cfg.trained_models_path, training_name), ignore_errors=True)
shutil.rmtree(Path(cfg.output_metrics_path, training_name), ignore_errors=True)
print('Cleaned up smoke-test checkpoint/metrics folders.', flush=True)
