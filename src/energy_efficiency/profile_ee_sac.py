from pathlib import Path
from sys import path as sys_path
project_root_path = Path(Path(__file__).parent, '..', '..')
sys_path.append(str(project_root_path.resolve()))

from src.config.config import Config
from src.models.EE_sac import train_sac_energy_effiency

cfg = Config()
cfg.config_learner.training_name = 'profile_test_do_not_use'
cfg.config_learner.reward = {'energy_efficiency_no_normalization_fixed': 1.0}

# 30 episodes (not 3): a short run is dominated by one-time tf.function
# retracing (esp. inside sac.train()), which swamps the profile and hides
# the steady-state per-call cost we actually want to measure. 30 episodes
# amortizes that fixed cost enough to see steady-state numbers, while still
# finishing in a few minutes.
cfg.config_learner.training_episodes = 30
cfg.config_learner.training_steps_per_episode = 600

cfg.profile = True
cfg.show_plots = False
cfg.verbosity = 1

train_sac_energy_effiency(config=cfg)
