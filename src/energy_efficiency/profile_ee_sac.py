from pathlib import Path
from sys import path as sys_path
project_root_path = Path(Path(__file__).parent, '..', '..')
sys_path.append(str(project_root_path.resolve()))

from src.config.config import Config
from src.models.EE_sac import train_sac_energy_effiency

cfg = Config()
cfg.config_learner.training_name = 'profile_test_do_not_use'
cfg.config_learner.reward = {'energy_efficiency_no_normalization_fixed': 1.0}

# small enough to run in a few minutes, large enough to clear
# training_minimum_experiences (1000) and trigger several train() calls
# (every 10 steps) so train_graph shows up meaningfully in the profile
cfg.config_learner.training_episodes = 3
cfg.config_learner.training_steps_per_episode = 600

cfg.profile = True
cfg.show_plots = False
cfg.verbosity = 1

train_sac_energy_effiency(config=cfg)
