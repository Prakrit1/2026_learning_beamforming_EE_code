from pathlib import Path
from sys import path as sys_path
project_root_path = Path(Path(__file__).parent, '..', '..')
sys_path.append(str(project_root_path.resolve()))

import os

import optuna
import numpy as np

from src.config.config import Config
from src.models.EE_sac import train_sac_energy_effiency

# Same knobs as EE_sac.py's __main__, so this search targets whichever EE
# scenario is currently under investigation (defaults match the N16K3/eta0.6
# Dinkelbach run flagged as possibly under-tuned for its problem size).
EE_REWARD_MODE = os.environ.get('EE_REWARD_MODE', 'energy_efficiency_dinkelbach_adaptive')
EE_TRAIN_ERROR_BOUND = float(os.environ.get('EE_TRAIN_ERROR_BOUND', 0.5))
EE_DINKELBACH_LAMBDA_WINDOW_STEPS = int(os.environ.get('EE_DINKELBACH_LAMBDA_WINDOW_STEPS', 5000))
EE_PA_EFFICIENCY = os.environ.get('EE_PA_EFFICIENCY')  # unset -> config.py default


def build_config() -> Config:
    """Mirror EE_sac.py's __main__ config setup so a trial trains under the
    same reward/error/system-size scenario as the real sbatch runs, instead
    of whatever ConfigSACLearner's hardcoded defaults happen to be."""

    config = Config()
    config.show_plots = False
    config.verbosity = 0

    config.config_error_model.error_rng_parametrizations['additive_error_on_cosine_of_aod']['args']['low'] = -EE_TRAIN_ERROR_BOUND
    config.config_error_model.error_rng_parametrizations['additive_error_on_cosine_of_aod']['args']['high'] = EE_TRAIN_ERROR_BOUND

    if EE_PA_EFFICIENCY is not None:
        config.pa_efficiency = float(EE_PA_EFFICIENCY)

    config.dinkelbach_lambda_window_steps = EE_DINKELBACH_LAMBDA_WINDOW_STEPS

    if EE_REWARD_MODE == 'energy_efficiency':
        config.config_learner.reward = {'energy_efficiency': 1.0}
    elif EE_REWARD_MODE == 'energy_efficiency_no_normalization_fixed':
        config.config_learner.reward = {'energy_efficiency_no_normalization_fixed': 1.0}
    elif EE_REWARD_MODE == 'energy_efficiency_dinkelbach_adaptive':
        config.config_learner.reward = {'energy_efficiency_dinkelbach_adaptive': 1.0}
    else:
        raise ValueError(f'Unknown EE_REWARD_MODE: {EE_REWARD_MODE!r}')

    # 'optuna_' prefix keeps trial checkpoints out of the real training_name
    # folders (e.g. 'EE_dinkelbach_adaptive_aod0.5_lwin5000_N16K3_eta0.6') --
    # those are read by EE_sac.py/my_evaluation.py's get_best_model_path
    # session-gap heuristic and must not accumulate unrelated trial snapshots.
    config.config_learner.training_name = (
        f'optuna_{EE_REWARD_MODE}_aod{EE_TRAIN_ERROR_BOUND}'
        f'_N{config.sat_tot_ant_nr}K{config.user_nr}_eta{config.pa_efficiency}'
    )

    return config


def objective(trial):

    config = build_config()

    # Suggest new values for this trial (same log-uniform search space as
    # optuna_train.py's base-model search)
    lr_critic = trial.suggest_float(name='lr_critic', low=-6, high=-4)
    lr_actor = trial.suggest_float(name='lr_actor', low=-7, high=-4)

    lr_critic = 10 ** lr_critic
    lr_actor = 10 ** lr_actor

    config.config_learner.training_episodes = int(0.2 * config.config_learner.training_episodes)  # % of training budget

    config.config_learner.network_args['policy_network_optimizer_args']['learning_rate'] = lr_actor
    config.config_learner.network_args['value_network_optimizer_args']['learning_rate'] = lr_critic

    # A single trial diverging to NaN (train_sac_energy_effiency raises
    # RuntimeError rather than continue training on NaN, see EE_sac.py) used
    # to crash the ENTIRE multi-day, 100-trial study -- confirmed this
    # actually happened (job 143462, aborted at trial 61 with 60 good trials
    # already done). With no persistent storage backing the study either
    # (see __main__ below), all 60 were lost outright. Catching it here and
    # reporting the trial as FAILED lets Optuna simply move on to the next
    # trial instead of losing everything already completed.
    try:
        _, metrics = train_sac_energy_effiency(
            config=config,
            optuna_trial=trial,
        )
    except RuntimeError as e:
        print(f'[optuna] Trial {trial.number} failed (not pruned by design -- a genuine '
              f'divergence, not just poor performance): {e}')
        raise optuna.TrialPruned()

    window_length = 10
    window_low = max(len(metrics['mean_reward_per_episode']) - window_length, 0)

    return np.nanmean(metrics['mean_reward_per_episode'][window_low:])


if __name__ == '__main__':
    # Persistent SQLite storage: without this, the whole study lives only in
    # this process's memory -- if the job is killed (OOM, node failure, or a
    # trial-crashing exception that somehow still escapes the try/except
    # above), every completed trial's result is gone, not just the one that
    # failed. This actually happened once already (see objective()'s
    # comment). storage_path is unique per (reward mode, error bound)
    # combination so the three parallel searches (error 0.0/0.025/0.05)
    # never share a database file. load_if_exists=True makes a resubmitted
    # job resume an interrupted study instead of starting over from trial 0.
    optuna_studies_dir = Path(project_root_path, 'outputs', 'optuna_studies')
    optuna_studies_dir.mkdir(parents=True, exist_ok=True)
    study_name = f'{EE_REWARD_MODE}_aod{EE_TRAIN_ERROR_BOUND}'
    storage_path = f'sqlite:///{Path(optuna_studies_dir, study_name)}.db'

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(),  # default: TPESampler
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=2,  # don't prune until n number of trials
            n_warmup_steps=2000,  # don't prune until n number of steps within a trial
            n_min_trials=3,  # minimum number of trials to prune, ensure at least n trials survive
            interval_steps=10,  # number of steps between pruning checks
        )
    )

    # resume-aware: if this is a rerun of an interrupted study, only run the
    # remaining trials, not another full 100
    remaining_trials = max(0, 100 - len(study.trials))
    print(f'[optuna] study "{study_name}": {len(study.trials)} trial(s) already recorded, '
          f'running {remaining_trials} more.')

    study.optimize(
        func=objective,
        n_trials=remaining_trials,
        gc_after_trial=True,  # garbage collection
    )
    print(study.best_trial)
    print(study.best_trials)
    # print(study.best_params)
