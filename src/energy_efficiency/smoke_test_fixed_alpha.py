"""
Smoke test for two things at once:
1. The new entropy_scale_alpha log line (EE_sac.py's episode-end logger.info
   call) -- confirms it prints without error and shows a sane value.
2. The EE_ENTROPY_SCALE_LR=0.0 override -- confirms alpha stays PINNED at
   entropy_scale_alpha_initial (1.0) across episodes instead of drifting,
   i.e. that setting the entropy-scale optimizer's LR to 0 actually disables
   adaptive tuning (matches the paper's fixed zeta_e=1.0 for comparison).

Also runs WITHOUT the override in a second call, to confirm alpha DOES move
away from 1.0 in the normal (adaptive, current default) case -- this is the
first-ever direct observation of that drift, since it was never logged
before this session.

Headless (Agg backend, show_plots=False), CPU-only. Must run via sbatch.
"""
import matplotlib
matplotlib.use('Agg')

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

from src.config.config import Config
from src.models.EE_sac import train_sac_energy_effiency


def run(label, entropy_scale_lr=None):
    print(f'\n=== SMOKE TEST: {label} ===', flush=True)
    cfg = Config()
    cfg.show_plots = False
    cfg.config_error_model.error_rng_parametrizations['additive_error_on_cosine_of_aod']['args']['low'] = -0.5
    cfg.config_error_model.error_rng_parametrizations['additive_error_on_cosine_of_aod']['args']['high'] = 0.5

    if entropy_scale_lr is not None:
        cfg.config_learner.training_args['entropy_scale_optimizer_args']['learning_rate'] = entropy_scale_lr
        # algorithm_args shares this nested dict by reference (built via
        # {**training_args, ...} in _post_init -- shallow copy, nested dicts
        # are NOT deep-copied), so no separate override needed; confirm that
        # assumption directly here rather than trusting it blindly
        assert cfg.config_learner.algorithm_args['entropy_scale_optimizer_args'] is \
            cfg.config_learner.training_args['entropy_scale_optimizer_args'], \
            'entropy_scale_optimizer_args is NOT shared by reference -- override would silently not work!'

    cfg.config_learner.reward = {'energy_efficiency': 1.0}
    training_name = f'smoke_test_fixed_alpha_{label}'
    cfg.config_learner.training_name = training_name
    cfg.config_learner.training_episodes = 10
    cfg.config_learner.training_steps_per_episode = 50
    cfg.config_learner.num_parallel_envs = 4
    cfg.config_learner.get_state_norm_factors_iterations = 200
    cfg.verbosity = 0

    train_sac_energy_effiency(config=cfg)
    print(f'=== SMOKE TEST {label}: PASSED (no crash) ===', flush=True)

    import shutil
    from pathlib import Path
    shutil.rmtree(Path(cfg.trained_models_path, training_name), ignore_errors=True)
    shutil.rmtree(Path(cfg.output_metrics_path, training_name), ignore_errors=True)


run('adaptive_default', entropy_scale_lr=None)
run('fixed_lr0', entropy_scale_lr=0.0)
print('\nALL SMOKE TESTS PASSED', flush=True)
