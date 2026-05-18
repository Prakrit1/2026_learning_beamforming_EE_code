from pathlib import Path
from sys import path as sys_path

project_root_path = Path(Path(__file__).parent, '..', '..')
sys_path.append(str(project_root_path.resolve()))

from datetime import datetime
from shutil import (
    copytree,
    rmtree,
)
import gzip
import pickle

import numpy as np
from matplotlib.pyplot import show as plt_show
import optuna

import src
from src.config.config import (
    Config,
)
from src.data.satellite_manager import (
    SatelliteManager,
)
from src.data.user_manager import (
    UserManager,
)
from src.models.algorithms.soft_actor_critic import (
    SoftActorCritic,
)
from src.models.helpers.get_state_norm_factors import (
    get_state_norm_factors,
)
from src.data.calc_sum_rate import (
    calc_sum_rate,
)
from src.data.calc_fairness import (
    calc_jain_fairness,
)
from src.data.precoder.mmse_precoder import (
    mmse_precoder_normalized,
    mmse_precoder_user_specific_normalized,
)
from src.data.precoder.mrc_precoder import (
    mrc_precoder_user_specific_normalized,
)
from src.data.precoder.regularized_zero_forcing import (
    regularized_zero_forcing_precoder_user_specific_normalized,
    regularized_zero_forcing_precoder_user_specific_normalized_without_inversion,
)
from src.data.precoder.rate_splitting import rate_splitting_no_norm
from src.utils.real_complex_vector_reshaping import (
    real_vector_to_half_complex_vector,
    complex_vector_to_double_real_vector,
    rad_and_phase_to_complex_vector,
)
from src.utils.norm_precoder import (
    norm_precoder,
)
from src.data.channel.get_steering_vec import (
    get_steering_vecs
)
from src.utils.plot_sweep import (
    plot_sweep,
)
from src.utils.profiling import (
    start_profiling,
    end_profiling,
)
from src.utils.progress_printer import (
    progress_printer,
)
from src.utils.update_sim import (
    update_sim,
)


def train_sac_reduced_based_on_aod(
        config: 'src.config.config.Config',
        optuna_trial: optuna.Trial or None = None,
) -> Path:
    """Train a reduced Soft Actor Critic precoder according to the config.

    Reduced means:
    - no RSMA common stream
    - SAC only controls private-user power allocation
    - for RZF additionally controls/selects the regularization factor
    """

    def progress_print(to_log: bool = False) -> None:
        progress = (
                (training_episode_id * config.config_learner.training_steps_per_episode + training_step_id + 1)
                / (config.config_learner.training_episodes * config.config_learner.training_steps_per_episode)
        )
        if not to_log:
            progress_printer(progress=progress, real_time_start=real_time_start)
        else:
            progress_printer(progress=progress, real_time_start=real_time_start, logger=logger)

    def save_model_checkpoint(extra):

        name = f''
        if extra is not None:
            name += f'sac_reduced_snap_{extra:.3f}'

        checkpoint_path = Path(
            config.trained_models_path,
            config.config_learner.training_name,
            'base',
            name,
        )

        logger.info(f'Saved model checkpoint at mean reward {extra:.3f}')

        sac.networks['policy'][0]['primary'].save(Path(checkpoint_path, 'model'))

        # save config
        config.save(Path(checkpoint_path, 'config'))

        # save norm dict
        with gzip.open(Path(checkpoint_path, 'config', 'norm_dict.gzip'), 'wb') as file:
            pickle.dump(norm_dict, file)

        # clean model checkpoints
        for high_score_prior_id, high_score_prior in enumerate(reversed(high_scores)):
            if high_score > 1.05 * high_score_prior or high_score_prior_id > 3:
                name = f'sac_reduced_snap_{high_score_prior:.3f}'

                prior_checkpoint_path = Path(
                    config.trained_models_path,
                    config.config_learner.training_name,
                    'base',
                    name
                )
                rmtree(path=prior_checkpoint_path, ignore_errors=True)
                high_scores.remove(high_score_prior)

        return checkpoint_path

    def save_results():

        name = f'training_error_sac_reduced.gzip'

        results_path = Path(config.output_metrics_path, config.config_learner.training_name, 'base')
        results_path.mkdir(parents=True, exist_ok=True)

        with gzip.open(Path(results_path, name), 'wb') as file:
            pickle.dump(metrics, file=file)

    logger = config.logger.getChild(__name__)

    # Number of SAC actions:
    # RZF: 1 action for regularization selection + user_nr power actions
    # MRT/MMSE: only user_nr power actions
    if config.private_part_precoding_style == 'RZF':
        config.config_learner.algorithm_args['network_args']['num_actions'] = 1 + config.user_nr
    elif config.private_part_precoding_style in ('MRT', 'MMSE'):
        config.config_learner.algorithm_args['network_args']['num_actions'] = config.user_nr
    else:
        raise ValueError(f"Unknown private part precoding mode={config.private_part_precoding_style}")

    satellite_manager = SatelliteManager(config=config)
    user_manager = UserManager(config=config)
    sac = SoftActorCritic(rng=config.rng, **config.config_learner.algorithm_args)

    norm_dict = get_state_norm_factors(
        config=config,
        satellite_manager=satellite_manager,
        user_manager=user_manager,
    )

    logger.info('State normalization factors found')
    logger.info(norm_dict)

    metrics: dict = {
        'mean_sum_rate_per_episode': -np.inf * np.ones(config.config_learner.training_episodes)
    }

    high_score = -np.inf
    high_scores = []

    real_time_start = datetime.now()

    profiler = None
    if config.profile:
        profiler = start_profiling()

    step_experience: dict = {
        'state': 0,
        'action': 0,
        'reward': 0,
        'next_state': 0,
    }

    for training_episode_id in range(config.config_learner.training_episodes):

        episode_metrics: dict = {
            'sum_rate_per_step': -np.inf * np.ones(config.config_learner.training_steps_per_episode),
            'mean_log_prob_density': np.inf * np.ones(config.config_learner.training_steps_per_episode),
            'value_loss': -np.inf * np.ones(config.config_learner.training_steps_per_episode),
        }

        # reset/update simulation for new episode
        update_sim(config, satellite_manager, user_manager)

        state_next = config.config_learner.get_state(
            config=config,
            user_manager=user_manager,
            satellite_manager=satellite_manager,
            norm_factors=norm_dict['norm_factors'],
            **config.config_learner.get_state_args
        )

        for training_step_id in range(config.config_learner.training_steps_per_episode):

            simulation_step = (
                    training_episode_id * config.config_learner.training_steps_per_episode
                    + training_step_id
            )

            # determine state
            state_current = state_next
            step_experience['state'] = state_current

            # determine action based on state
            action = sac.get_action(state=state_current)
            step_experience['action'] = action

            # ------------------------------------------------------------
            # Interpret SAC action
            # ------------------------------------------------------------

            if config.private_part_precoding_style == 'RZF':
                mmse_scale = config.noise_power_watt * (
                        config.user_nr / config.power_constraint_watt
                )

                # map action[0] from [-1, 1] to [0, 1]
                x = (np.clip(action[0], -1, 1) + 1) / 2

                if x < 1 / 3:
                    option = 0
                elif x < 2 / 3:
                    option = 1
                else:
                    option = 2

                # 0 = ZF, 1 = MMSE-like, 10 = strong regularization / MRT-like
                factor_map = np.array([0, 1, 10])
                regularization_factor = factor_map[option] * mmse_scale

                # remaining actions are private-user power factors
                power_factors_private_users = action[1:config.user_nr + 1]

            elif config.private_part_precoding_style in ('MRT', 'MMSE'):

                # all actions are private-user power factors
                power_factors_private_users = action[0:config.user_nr]

            else:
                raise ValueError(f"Unknown private part precoding mode={config.private_part_precoding_style}")

            # ------------------------------------------------------------
            # Power normalization
            # ------------------------------------------------------------

            # Map SAC actions from [-1, 1] to positive relative weights [0, 1]
            power_factors_private_users_positive = (
                    np.clip(power_factors_private_users, -1, 1) + 1
            ) / 2

            # Ensure total private transmit power does not exceed power_constraint_watt
            sum_power_users = np.sum(power_factors_private_users_positive) + 1e-12

            # same logic as in your selected RSMA code
            power_scale = min(1, 1 / sum_power_users)

            power_factors_private_users_normalized = (
                    power_factors_private_users_positive
                    * power_scale
                    * config.power_constraint_watt
            )

            power_constraint_private_part = np.sum(power_factors_private_users_normalized)

            # ------------------------------------------------------------
            # Mask inactive users
            # ------------------------------------------------------------

            eps_power = 1e-4 * config.power_constraint_watt
            active_user_mask = power_factors_private_users_normalized > eps_power

            channel_matrix_private_effective = satellite_manager.erroneous_channel_state_information.copy()
            phase_aod_steering_to_users = satellite_manager.erroneous_phase_aod_steering_to_users

            steering_vectors = get_steering_vecs(satellite_manager, phase_aod_steering_to_users)

            channel_matrix_private_effective[~active_user_mask, :] = 0.0

            # ------------------------------------------------------------
            # Build private precoder
            # ------------------------------------------------------------

            if config.private_part_precoding_style == 'RZF':

                if not config.matrix_inversion_approximation:
                    private_part_precoding = regularized_zero_forcing_precoder_user_specific_normalized(
                        channel_matrix=steering_vectors,
                        regularization_factor=regularization_factor,
                        power_factors_users=power_factors_private_users_normalized,
                    )
                else:
                    private_part_precoding = regularized_zero_forcing_precoder_user_specific_normalized_without_inversion(
                        channel_matrix=steering_vectors,
                        regularization_factor=regularization_factor,
                        power_factors_users=power_factors_private_users_normalized,
                        order=config.matrix_inversion_approximation_order,
                    )

            elif config.private_part_precoding_style == 'MRT':

                private_part_precoding = mrc_precoder_user_specific_normalized(
                    channel_matrix=steering_vectors,
                    power_factors_users=power_factors_private_users_normalized,
                )

            elif config.private_part_precoding_style == 'MMSE':

                private_part_precoding = mmse_precoder_user_specific_normalized(
                    channel_matrix=steering_vectors,
                    noise_power_watt=config.noise_power_watt,
                    power_constraint_watt=power_constraint_private_part,
                    power_factors_users=power_factors_private_users_normalized,
                )

            else:
                raise ValueError(f"Unknown private part precoding mode={config.private_part_precoding_style}")

            w_precoder = private_part_precoding

            # ------------------------------------------------------------
            # Calculate reward
            # ------------------------------------------------------------

            reward = 0

            if 'sum_rate' in config.config_learner.reward:
                sum_rate_reward = calc_sum_rate(
                    channel_state=satellite_manager.channel_state_information,
                    w_precoder=w_precoder,
                    noise_power_watt=config.noise_power_watt,
                )
                reward += config.config_learner.reward['sum_rate'] * sum_rate_reward

            if 'fairness' in config.config_learner.reward:
                fairness_reward = calc_jain_fairness(
                    channel_state=satellite_manager.channel_state_information,
                    w_precoder=w_precoder,
                    noise_power_watt=config.noise_power_watt,
                )
                reward += config.config_learner.reward['fairness'] * fairness_reward

            if any(key not in ['sum_rate', 'fairness'] for key in config.config_learner.reward.keys()):
                raise ValueError("No valid reward provided")

            step_experience['reward'] = reward

            # ------------------------------------------------------------
            # Update simulation state
            # ------------------------------------------------------------

            update_sim(config, satellite_manager, user_manager)

            state_next = config.config_learner.get_state(
                config=config,
                user_manager=user_manager,
                satellite_manager=satellite_manager,
                norm_factors=norm_dict['norm_factors'],
                **config.config_learner.get_state_args
            )

            step_experience['next_state'] = state_next

            sac.add_experience(experience=step_experience)

            # ------------------------------------------------------------
            # Train SAC off-policy
            # ------------------------------------------------------------

            train_policy = config.config_learner.policy_training_criterion(
                simulation_step=simulation_step
            )
            train_value = config.config_learner.value_training_criterion(
                simulation_step=simulation_step
            )

            if train_value or train_policy:
                mean_log_prob_density, value_loss = sac.train(
                    toggle_train_value_networks=train_value,
                    toggle_train_policy_network=train_policy,
                    toggle_train_entropy_scale_alpha=True,
                )
            else:
                mean_log_prob_density = np.nan
                value_loss = np.nan

            # ------------------------------------------------------------
            # Log step results
            # ------------------------------------------------------------

            episode_metrics['sum_rate_per_step'][training_step_id] = reward
            episode_metrics['mean_log_prob_density'][training_step_id] = mean_log_prob_density
            episode_metrics['value_loss'][training_step_id] = value_loss

            if config.verbosity > 0:
                if training_step_id % 50 == 0:
                    progress_print()

        # ------------------------------------------------------------
        # Log episode results
        # ------------------------------------------------------------

        episode_mean_sum_rate = np.nanmean(episode_metrics['sum_rate_per_step'])
        metrics['mean_sum_rate_per_episode'][training_episode_id] = episode_mean_sum_rate

        # If doing optuna optimization: check trial results, stop early if bad
        if optuna_trial:
            window = 10
            lower_end = max(training_episode_id - window, 0)
            episode_result = np.nanmean(
                metrics['mean_sum_rate_per_episode'][lower_end:training_episode_id + 1]
            )

            optuna_trial.report(episode_result, training_episode_id)

            if optuna_trial.should_prune():
                raise optuna.TrialPruned()

        if config.verbosity > 0:
            print('\r', end='')  # clear console for logging results

        progress_print(to_log=True)

        logger.info(
            f'Episode {training_episode_id}:'
            f' Episode mean reward: {episode_mean_sum_rate:.4f}'
            f' std {np.nanstd(episode_metrics["sum_rate_per_step"]):.2f},'
            f' current exploration: {np.nanmean(episode_metrics["mean_log_prob_density"]):.2f},'
            f' value loss: {np.nanmean(episode_metrics["value_loss"]):.5f}'
        )

        # save network snapshot
        if episode_mean_sum_rate > high_score:
            high_score = episode_mean_sum_rate.copy()
            high_scores.append(high_score)
            best_model_path = save_model_checkpoint(episode_mean_sum_rate)

    # end compute performance profiling
    if profiler is not None:
        end_profiling(profiler)

    save_results()

    if config.show_plots:
        plot_sweep(
            range(config.config_learner.training_episodes),
            metrics['mean_sum_rate_per_episode'],
            'Training Episode',
            'Sum Rate',
        )
        plt_show()

    return best_model_path, metrics


if __name__ == '__main__':
    cfg = Config()
    train_sac_reduced_based_on_aod(config=cfg)