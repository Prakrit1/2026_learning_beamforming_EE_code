from pathlib import Path
from sys import path as sys_path
project_root_path = Path(Path(__file__).parent, '..', '..')
sys_path.append(str(project_root_path.resolve()))

from datetime import datetime
from shutil import (
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
from src.data.calc_fairness import(
    calc_jain_fairness
)
from src.data.precoder.mmse_precoder import (
    mmse_precoder_normalized,
)
from src.utils.real_complex_vector_reshaping import (
    real_vector_to_half_complex_vector,
    complex_vector_to_double_real_vector,
    rad_and_phase_to_complex_vector,
)
from src.utils.norm_precoder import (
    norm_precoder,
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

import os 
import tensorflow as tf 

 

gpus = tf.config.list_physical_devices('GPU') 
if gpus: 
    try: 
        for gpu in gpus: 
            tf.config.experimental.set_memory_growth(gpu, True) 
        print(f"Using GPU: {gpus}") 
    except RuntimeError as e: 
        print(f"GPU error: {e}") 
else: 
    print("No GPU found, running on CPU") 


def train_sac_energy_effiency(
        config: 'src.config.config.Config',
        optuna_trial: optuna.Trial or None = None,
) -> Path:
    """Train a Soft Actor Critic precoder according to the config."""

    def progress_print(to_log: bool = False) -> None:
        # PARALLEL PROCESSING: denominator uses transitions_per_episode (actual
        # transitions collected under batching), not the old
        # training_steps_per_episode config value directly -- see
        # steps_per_episode_batched below for why they can differ.
        progress = (
                (training_episode_id * transitions_per_episode + training_step_id + 1)
                / (config.config_learner.training_episodes * transitions_per_episode)
        )
        if not to_log:
            progress_printer(progress=progress, real_time_start=real_time_start)
        else:
            progress_printer(progress=progress, real_time_start=real_time_start, logger=logger)

    def add_mmse_experience():

        # this needs to use erroneous csi, otherwise the data distribution in buffer
        #  is changed significantly from reality, i.e., the learner gets too much confidence that
        #  the csi is reliable
        w_mmse = mmse_precoder_normalized(
            channel_matrix=satellite_manager.erroneous_channel_state_information,
            **config.mmse_args
        )
        reward_mmse = calc_sum_rate(
            channel_state=satellite_manager.channel_state_information,
            w_precoder=w_mmse,
            noise_power_watt=config.noise_power_watt,
        )
        if (reward_mmse > reward) or not config.config_learner.only_add_mmse_samples_with_greater_reward:
            mmse_experience = {
                'state': state_current,
                'action': complex_vector_to_double_real_vector(w_mmse.flatten()),
                'reward': reward_mmse,
                'next_state': state_next,
            }
            sac.add_experience(mmse_experience)

    def save_model_checkpoint(extra):

        name = f''
        if extra is not None:
            name += f'full_snap_energy_effiency_{extra:.3f}'
        checkpoint_path = Path(
            config.trained_models_path,
            config.config_learner.training_name,
            'base',
            name,
        )

        logger.info(f'Saved model checkpoint at mean reward {extra:.3f}')

        # CHECKPOINT SPEEDUP: save_weights() instead of the old full save().
        # Profiling showed checkpoint saving was ~34% of total training wall
        # time (~2.2s/save, 13 saves in just 30 episodes) -- model.save()
        # does a full Keras SavedModel export, which retraces serialization
        # signatures for every layer on every call. save_weights() skips that
        # entirely (measured ~15x faster on the save call alone) at the cost
        # of needing the architecture reconstructed from config before
        # loading -- see the matching load path in load_model.py.
        model_path = Path(checkpoint_path, 'model')
        rmtree(path=model_path, ignore_errors=True)
        model_path.mkdir(parents=True, exist_ok=True)
        sac.networks['policy'][0]['primary'].save_weights(Path(model_path, 'weights.weights.h5'))

        # save config
        config.save(Path(checkpoint_path, 'config'))

        # save norm dict
        with gzip.open(Path(checkpoint_path, 'config', 'norm_dict.gzip'), 'wb') as file:
            pickle.dump(norm_dict, file)

        # clean model checkpoints
        for high_score_prior_id, high_score_prior in enumerate(reversed(high_scores)):
            if high_score > 1.05 * high_score_prior or high_score_prior_id > 3:

                name = f'full_snap_energy_effiency_{high_score_prior:.3f}'

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

        name = f'training_error_learned_full.gzip'

        results_path = Path(config.output_metrics_path, config.config_learner.training_name, 'base')
        results_path.mkdir(parents=True, exist_ok=True)
        with gzip.open(Path(results_path, name), 'wb') as file:
            pickle.dump(metrics, file=file)

    logger = config.logger.getChild(__name__)

    config.config_learner.algorithm_args['network_args']['num_actions'] = 2 * config.sat_nr * config.sat_ant_nr * config.user_nr

    # norm factors are computed on a single dedicated env, before the parallel
    # training envs below are created
    satellite_manager = SatelliteManager(config=config)
    user_manager = UserManager(config=config)
    sac = SoftActorCritic(rng=config.rng, **config.config_learner.algorithm_args)

    norm_dict = get_state_norm_factors(config=config, satellite_manager=satellite_manager, user_manager=user_manager)
    logger.info('State normalization factors found')
    logger.info(norm_dict)

    # === PARALLEL PROCESSING START =============================================
    # Profiling a 14h training run showed >90% of wall time was GPU dispatch
    # overhead from calling sac.get_action() one sample at a time (13M times for
    # a full run) -- not simulation or gradient-step compute. Fix: step
    # num_parallel_envs independent simulation environments together each
    # iteration, batching ONLY the actor forward pass (get_action_batch) across
    # them. update_sim and the reward-branch calculations below remain
    # per-environment (looped, unchanged logic) since they were already cheap
    # (~0.5ms/call) and batching them would require much riskier changes to
    # SatelliteManager/UserManager internals for comparatively little payoff.
    # See num_parallel_envs in config_sac_learner.py and get_action_batch() in
    # soft_actor_critic.py.
    num_parallel_envs = config.config_learner.num_parallel_envs
    satellite_managers = [SatelliteManager(config=config) for _ in range(num_parallel_envs)]
    user_managers = [UserManager(config=config) for _ in range(num_parallel_envs)]
    state_next_list = [None] * num_parallel_envs

    steps_per_episode_batched = config.config_learner.training_steps_per_episode // num_parallel_envs
    transitions_per_episode = steps_per_episode_batched * num_parallel_envs
    logger.info(
        f'Running {num_parallel_envs} parallel envs, '
        f'{steps_per_episode_batched} batched steps/episode '
        f'({transitions_per_episode} transitions/episode, '
        f'vs. {config.config_learner.training_steps_per_episode} requested)'
    )

    metrics: dict = {
        'mean_reward_per_episode': -np.inf * np.ones(config.config_learner.training_episodes)
    }
    high_score = -np.inf
    high_scores = []

    real_time_start = datetime.now()

    profiler = None
    if config.profile:
        profiler = start_profiling()

    step_experience: dict = {'state': 0, 'action': 0, 'reward': 0, 'next_state': 0}

    for training_episode_id in range(config.config_learner.training_episodes):

        episode_metrics: dict = {
            'reward_per_step': np.nan * np.ones(transitions_per_episode),
            'mean_log_prob_density': np.nan * np.ones(transitions_per_episode),
            'value_loss': np.nan * np.ones(transitions_per_episode),
        }

        # PARALLEL PROCESSING: reset all parallel envs for new episode (was a
        # single update_sim/get_state call on one env; now one per env_idx)
        for env_idx in range(num_parallel_envs):
            update_sim(config, satellite_managers[env_idx], user_managers[env_idx])
            state_next_list[env_idx] = config.config_learner.get_state(
                config=config,
                user_manager=user_managers[env_idx],
                satellite_manager=satellite_managers[env_idx],
                norm_factors=norm_dict['norm_factors'],
                **config.config_learner.get_state_args
            )

        # PARALLEL PROCESSING: outer loop now iterates over batched GROUPS of
        # num_parallel_envs transitions, instead of one transition at a time
        for group_idx in range(steps_per_episode_batched):

            # PARALLEL PROCESSING: the single batched forward pass this whole
            # rewrite exists for -- replaces num_parallel_envs individual
            # sac.get_action() calls (batch=1 each) with one sac.get_action_batch()
            # call (batch=num_parallel_envs), removing per-call GPU dispatch
            # overhead as the dominant cost (see soft_actor_critic.py)
            states_batch = np.stack(state_next_list).astype('float32')
            actions_batch = sac.get_action_batch(states=states_batch)

            # PARALLEL PROCESSING: everything below mirrors the original
            # per-step body, now looped once per env within this batched group;
            # satellite_manager, user_manager, state_current, action are rebound
            # to this env_idx each iteration, so the unmodified
            # add_mmse_experience() closure below keeps working unchanged
            for env_idx in range(num_parallel_envs):

                # PARALLEL PROCESSING: flatten (group_idx, env_idx) back into a
                # single monotonic transition index, so simulation_step and the
                # train_policy/train_value_criterion thresholds below behave
                # identically to the original single-env per-transition counter
                training_step_id = group_idx * num_parallel_envs + env_idx
                simulation_step = training_episode_id * transitions_per_episode + training_step_id

                # PARALLEL PROCESSING: rebind to this env's own manager pair
                satellite_manager = satellite_managers[env_idx]
                user_manager = user_managers[env_idx]

                # determine state
                state_current = state_next_list[env_idx]
                step_experience['state'] = state_current

                # determine action based on state (already computed for the whole batch above)
                action = actions_batch[env_idx]
                step_experience['action'] = action

                # reshape to fit reward calculation
                w_precoder_vector = real_vector_to_half_complex_vector(action)
                # w_precoder_vector = rad_and_phase_to_complex_vector(action)
                w_precoder = w_precoder_vector.reshape((config.sat_nr*config.sat_ant_nr, config.user_nr))
                power_precoder = np.real(np.trace(np.matmul(w_precoder.conj().T, w_precoder)))

                # Pre-clip power, captured before the unconditional clip-down block below.
                # Needed for 'energy_efficiency_no_normalization_fixed' so that branch tests
                # an action space that can genuinely transmit below the budget, rather than
                # reading the same post-clip value every other branch reads (which is what
                # made the old 'energy_efficiency_without_normalization' branch not actually
                # test anything different from 'energy_efficiency').
                raw_power_precoder = power_precoder

                if power_precoder > config.power_constraint_watt:
                    power_precoder = config.power_constraint_watt
                    norm_factor = np.sqrt(power_precoder / np.trace(np.matmul(w_precoder.conj().T, w_precoder)))
                    normalized_precoder = norm_factor * w_precoder
                    w_precoder = normalized_precoder

                # w_precoder_normed = norm_precoder(precoding_matrix=w_precoder, power_constraint_watt=config.power_constraint_watt,
                #                                   per_satellite=True, sat_nr=config.sat_nr, sat_ant_nr=config.sat_ant_nr)

                # step simulation based on action, determine reward
                reward = 0
                #if 'sum_rate' in config.config_learner.reward:
                #    sum_rate_reward = calc_sum_rate(
                #        channel_state=satellite_manager.channel_state_information,
                #        w_precoder=w_precoder,
                #        noise_power_watt=config.noise_power_watt,
                #    )
                #    reward += config.config_learner.reward['sum_rate'] * sum_rate_reward
                    
                # if 'fairness' in config.config_learner.reward:
                #     fairness_reward = calc_jain_fairness(
                #         channel_state=satellite_manager.channel_state_information,
                #         w_precoder=w_precoder,
                #         noise_power_watt=config.noise_power_watt,
                #     )
                #     reward += config.config_learner.reward['fairness'] * fairness_reward
                
                if 'sum_rate_over_transmit_power' in config.config_learner.reward:
                    sum_rate_reward = calc_sum_rate(
                        channel_state=satellite_manager.channel_state_information,
                        w_precoder=w_precoder,
                        noise_power_watt=config.noise_power_watt,
                    )
                    if power_precoder < 1:
                        sum_rate_over_transmit_power = 0
                    else:
                        normalized_power = power_precoder / config.power_constraint_watt
                        sum_rate_over_transmit_power = sum_rate_reward / normalized_power
                    reward += config.config_learner.reward['sum_rate_over_transmit_power'] * sum_rate_over_transmit_power
                    
                    
                # Prakrit added this for calculation of EE
                # FIXED: this branch now implements genuine Scheme I -- the
                # source paper's "same as the learned precoders" Eq.(26)-style
                # normalization, which ALWAYS rescales raw output to land
                # exactly at the power budget, whether the raw output started
                # above or below it. Previously this branch reused the
                # clip-only `w_precoder`/`power_precoder` computed in the
                # unconditional block above -- which only ever rescales DOWN
                # when over budget and passes raw output through unchanged when
                # under budget. That made this branch mathematically identical
                # to 'energy_efficiency_no_normalization_fixed' whenever
                # raw_power_precoder <= power_constraint_watt, so the two
                # branches were not actually testing different normalization
                # schemes. This version always rescales by
                # sqrt(power_constraint_watt / raw_power_precoder), so
                # tr{W^H W} == power_constraint_watt unconditionally, and
                # d(power)/d(action) == 0 everywhere by construction -- the
                # formal Scheme I property.
                if 'energy_efficiency' in config.config_learner.reward:
                    if raw_power_precoder < 1e-9:
                        # degenerate all-zero raw output: nothing to rescale
                        # toward a nonzero target.
                        energy_efficiency = 0.0
                    else:
                        scheme_i_norm_factor = np.sqrt(config.power_constraint_watt / raw_power_precoder)
                        w_precoder_scheme_i = scheme_i_norm_factor * w_precoder_vector.reshape(
                            (config.sat_nr * config.sat_ant_nr, config.user_nr)
                        )
                        # by construction this is always == config.power_constraint_watt;
                        # kept as an explicit trace computation (rather than just writing
                        # config.power_constraint_watt directly) so a numerical assertion
                        # can be added during debugging if desired.
                        power_precoder_scheme_i = np.real(
                            np.trace(np.matmul(w_precoder_scheme_i.conj().T, w_precoder_scheme_i))
                        )

                        sum_rate_reward = calc_sum_rate(
                            channel_state=satellite_manager.channel_state_information,
                            w_precoder=w_precoder_scheme_i,
                            noise_power_watt=config.noise_power_watt,
                        )
                        # Normalize EVERYTHING by power_constraint_watt (unchanged from
                        # before -- this is a constant rescale of the denominator and
                        # does not affect the optimum, only gradient magnitude)
                        # Divide by pa_efficiency first: radiated power power_precoder_scheme_i
                        # is only a fraction eta of what the amplifier actually draws from the
                        # DC supply, so the amplifier's true power draw is
                        # power_precoder_scheme_i / eta, not power_precoder_scheme_i itself
                        # (Auer et al. 2011). Under Scheme I this term is constant (== budget /
                        # eta) since raw output is always rescaled to the full power budget, so
                        # this only rescales the reward's denominator uniformly here -- it
                        # matters once compared against the no-normalization branches below,
                        # where transmit power actually varies.
                        normalized_transmit = (power_precoder_scheme_i / config.pa_efficiency) / config.power_constraint_watt
                        normalized_circuit = (
                            config.sat_nr
                            * config.sat_ant_nr
                            * config.circuit_power_watt
                            / config.power_constraint_watt
                        )
                        total_power_normalized = normalized_transmit + normalized_circuit

                        if total_power_normalized < 1e-9:
                            energy_efficiency = 0.0
                        else:
                            energy_efficiency = sum_rate_reward / total_power_normalized

                    reward += config.config_learner.reward['energy_efficiency'] * energy_efficiency
                    
                    
                
                if 'sum_rate_without_normalization' in config.config_learner.reward:
                    sum_rate_reward = calc_sum_rate(
                        channel_state=satellite_manager.channel_state_information,
                        w_precoder=w_precoder,
                        noise_power_watt=config.noise_power_watt,
                    )
                    total_transmit_power = power_precoder

                    if total_transmit_power < 1e-9:
                        sum_rate_without_normalization = 0.0
                    else:
                        sum_rate_without_normalization = sum_rate_reward / total_transmit_power
                    reward += config.config_learner.reward['sum_rate_without_normalization'] * sum_rate_without_normalization


                if 'energy_efficiency_without_normalization' in config.config_learner.reward:
                    sum_rate_reward = calc_sum_rate(
                        channel_state=satellite_manager.channel_state_information,
                        w_precoder=w_precoder,
                        noise_power_watt=config.noise_power_watt,
                    )
                    # Amplifier draw, not radiated power: the PA only radiates a fraction
                    # pa_efficiency of what it draws, so an agent that wants to actually
                    # save DC power must cut transmit_power by more than a 1:1 ratio
                    # against circuit_power (Auer et al. 2011).
                    transmit_power = power_precoder / config.pa_efficiency
                    circuit_power = config.sat_nr * config.sat_ant_nr * config.circuit_power_watt
                    total_transmit_power = transmit_power + circuit_power

                    if total_transmit_power < 1e-9:
                        energy_efficiency_without_normalization = 0.0
                    else:
                        energy_efficiency_without_normalization = sum_rate_reward / total_transmit_power

                    reward += config.config_learner.reward['energy_efficiency_without_normalization'] * energy_efficiency_without_normalization
                    
                    
                if 'energy_efficiency_with_DDinkelbach_full' in config.config_learner.reward:
                    sum_rate_reward = calc_sum_rate(
                        channel_state=satellite_manager.channel_state_information,
                        w_precoder=w_precoder,
                        noise_power_watt=config.noise_power_watt,
                    )
                    # Same amplifier-draw correction as 'energy_efficiency_without_normalization'
                    # above -- divide radiated power by pa_efficiency to get true DC draw
                    # before adding the additive circuit power term (Auer et al. 2011).
                    transmit_power = np.real(power_precoder) / config.pa_efficiency
                    circuit_power = config.sat_nr * config.sat_ant_nr * config.circuit_power_watt
                    total_transmit_power = transmit_power + circuit_power
                    if total_transmit_power < 1e-9:
                        energy_efficiency_with_DDinkelbach_full = 0.0
                    else:
                        energy_efficiency_with_DDinkelbach_full = sum_rate_reward - 2 * total_transmit_power
                    reward += config.config_learner.reward['energy_efficiency_with_DDinkelbach_full'] * energy_efficiency_with_DDinkelbach_full

                if 'energy_efficiency_with_Dinkelbach_simp' in config.config_learner.reward:
                    sum_rate_reward = calc_sum_rate(
                        channel_state=satellite_manager.channel_state_information,
                        w_precoder=w_precoder,
                        noise_power_watt=config.noise_power_watt,
                    )
                    transmit_power = np.real(power_precoder)
                    if transmit_power < 1e-9:
                        energy_efficiency_with_Dinkelbach_simp = 0.0
                    else:
                        energy_efficiency_with_Dinkelbach_simp = sum_rate_reward / transmit_power
                    reward += config.config_learner.reward['energy_efficiency_with_Dinkelbach_simp'] * energy_efficiency_with_Dinkelbach_simp

                # New branch: properly isolated "without normalization" EE reward.
                # Unlike 'sum_rate_without_normalization' and
                # 'energy_efficiency_without_normalization' above -- which both read
                # the already-clipped `power_precoder`, so they only ever differ
                # from 'sum_rate_over_transmit_power'/'energy_efficiency' by a
                # constant rescale factor and never actually test a different
                # action space -- this branch uses `raw_power_precoder` (captured
                # before the unconditional clip-down block runs) and the
                # corresponding un-normalized precoder `w_precoder_raw`. This is
                # the only branch where the agent can genuinely receive reward for
                # transmitting at less than the full power budget, since both the
                # rate calculation and the power-in-denominator reflect the
                # actor's raw output rather than a budget-saturated rescaling of
                # it. Uses the same 1e-9 epsilon as the other non-buggy branches
                # (not the inconsistent '< 1' Watt floor used in
                # 'sum_rate_over_transmit_power' above).
                if 'energy_efficiency_no_normalization_fixed' in config.config_learner.reward:
                    if raw_power_precoder > config.power_constraint_watt:
                        # still must not exceed the physical power budget -- this
                        # is a hardware limit, not the "normalization" being
                        # tested here. Only the upward rescale-to-saturate is
                        # removed; the downward clip-on-violation is kept.
                        w_precoder_raw = w_precoder  # already clipped to budget above
                        power_for_ee = power_precoder  # == config.power_constraint_watt
                    else:
                        w_precoder_raw = w_precoder_vector.reshape(
                            (config.sat_nr * config.sat_ant_nr, config.user_nr)
                        )
                        power_for_ee = raw_power_precoder

                    sum_rate_reward_raw = calc_sum_rate(
                        channel_state=satellite_manager.channel_state_information,
                        w_precoder=w_precoder_raw,
                        noise_power_watt=config.noise_power_watt,
                    )
                    # Normalize by power_constraint_watt, same as the 'energy_efficiency'
                    # (Scheme I) branch above -- otherwise this branch's power-in-denominator
                    # is ~power_constraint_watt times larger (raw Watts vs. a budget-relative
                    # fraction), making its reward ~100x smaller in magnitude than
                    # 'energy_efficiency' for an equivalent sum rate. That scale mismatch
                    # starved the policy gradient against the fixed-magnitude L2 weight
                    # penalty (training_l2_norm_scale_value/policy), causing the critic to
                    # collapse onto the near-constant tiny reward within ~500 episodes and
                    # freezing the actor for the rest of training.
                    # Amplifier draw correction (Auer et al. 2011), same as the other EE
                    # branches -- matters most in THIS branch specifically, since
                    # power_for_ee is the one quantity across all EE branches that can
                    # genuinely fall below the power budget rather than being rescaled
                    # up to saturate it, so this is where power back-off actually gets
                    # rewarded/penalized through the eta factor.
                    normalized_power_for_ee = (power_for_ee / config.pa_efficiency) / config.power_constraint_watt
                    normalized_circuit_for_ee = (
                        config.sat_nr * config.sat_ant_nr * config.circuit_power_watt
                        / config.power_constraint_watt
                    )
                    total_power_for_ee = normalized_power_for_ee + normalized_circuit_for_ee

                    if total_power_for_ee < 1e-9:
                        energy_efficiency_no_normalization_fixed = 0.0
                    else:
                        energy_efficiency_no_normalization_fixed = sum_rate_reward_raw / total_power_for_ee

                    reward += (
                        config.config_learner.reward['energy_efficiency_no_normalization_fixed']
                        * energy_efficiency_no_normalization_fixed
                    )

                

                valid_reward_keys = [
                    'sum_rate',
                    'energy_efficiency',
                    'energy_efficiency_no_normalization_fixed',
                ]
                if any(key not in valid_reward_keys for key in config.config_learner.reward.keys()):
                    raise ValueError("No valid reward provided")
                    
                step_experience['reward'] = reward

                # optionally add the corresponding mmse precoder to the data set
                if config.rng.random() < config.config_learner.percentage_mmse_samples_added_to_exp_buffer:
                    add_mmse_experience()  # todo note: currently state_next saved in the mmse experience is not correct

                # update simulation state
                update_sim(config, satellite_manager, user_manager)

                # get new state
                state_next = config.config_learner.get_state(
                    config=config,
                    user_manager=user_manager,
                    satellite_manager=satellite_manager,
                    norm_factors=norm_dict['norm_factors'],
                    **config.config_learner.get_state_args
                )
                step_experience['next_state'] = state_next
                # PARALLEL PROCESSING: persist this env's next state so it feeds
                # the batched get_action_batch() call at the top of the next group
                state_next_list[env_idx] = state_next

                sac.add_experience(experience=step_experience)

                # train allocator off-policy
                train_policy = config.config_learner.policy_training_criterion(simulation_step=simulation_step)
                train_value = config.config_learner.value_training_criterion(simulation_step=simulation_step)

                if train_value or train_policy:
                    mean_log_prob_density, value_loss = sac.train(
                        toggle_train_value_networks=train_value,
                        toggle_train_policy_network=train_policy,
                        toggle_train_entropy_scale_alpha=True,
                    )
                else:
                    mean_log_prob_density = np.nan
                    value_loss = np.nan

                # log results
                episode_metrics['reward_per_step'][training_step_id] = reward
                episode_metrics['mean_log_prob_density'][training_step_id] = mean_log_prob_density
                episode_metrics['value_loss'][training_step_id] = value_loss

                if config.verbosity > 0:
                    if training_step_id % 50 == 0:
                        progress_print()

        # log episode results
        episode_mean_reward = np.nanmean(episode_metrics['reward_per_step'])
        metrics['mean_reward_per_episode'][training_episode_id] = episode_mean_reward

        # If doing optuna optimization: check trial results, stop early if bad
        if optuna_trial:
            window = 10
            lower_end = max(training_episode_id-window, 0)
            episode_result = np.nanmean(metrics['mean_reward_per_episode'][lower_end:training_episode_id+1])

            optuna_trial.report(episode_result, training_episode_id)
            if optuna_trial.should_prune():
                raise optuna.TrialPruned()

        if config.verbosity > 0:
            print('\r', end='')  # clear console for logging results
        progress_print(to_log=True)
        logger.info(
            f'Episode {training_episode_id}:'
            f' Episode mean reward: {episode_mean_reward:.4f}'
            f' std {np.nanstd(episode_metrics["reward_per_step"]):.2f},'
            f' current exploration: {np.nanmean(episode_metrics["mean_log_prob_density"]):.2f},'
            f' value loss: {np.nanmean(episode_metrics["value_loss"]):.5f}'
            # f' curr. lr: {sac.networks["policy"][0]["primary"].optimizer.learning_rate(sac.networks["policy"][0]["primary"].optimizer.iterations):.2E}'
        )
        logger.info(f'Starting training: name={config.config_learner.training_name}, reward={config.config_learner.reward}')

        # save network snapshot
        if episode_mean_reward > high_score:
            high_score = episode_mean_reward.copy()
            high_scores.append(high_score)
            best_model_path = save_model_checkpoint(episode_mean_reward)

        # end compute performance profiling
    if profiler is not None:
        end_profiling(profiler)

    save_results()

    if config.show_plots:
        plot_sweep(range(config.config_learner.training_episodes), metrics['mean_reward_per_episode'],
                   'Training Episode', 'Reward')
        plt_show()

    return best_model_path, metrics


if __name__ == '__main__':
    cfg = Config()

    # Allow selecting the reward variant via env var so two sbatch jobs can run
    # concurrently off the same unedited source (rather than hand-editing the
    # hardcoded reward/training_name in config_sac_learner.py before each
    # submission, which races if two jobs read the file at different times).
    # Falls back to whatever is hardcoded in config_sac_learner.py if unset.
    reward_mode = os.environ.get('EE_REWARD_MODE')
    if reward_mode == 'energy_efficiency':
        cfg.config_learner.reward = {'energy_efficiency': 1.0}  # Scheme I, full EE with circuit power
        cfg.config_learner.training_name = 'full_EE'
    elif reward_mode == 'energy_efficiency_no_normalization_fixed':
        cfg.config_learner.reward = {'energy_efficiency_no_normalization_fixed': 1.0}
        cfg.config_learner.training_name = 'EE_no_norm_fixed'
    elif reward_mode is not None:
        raise ValueError(f'Unknown EE_REWARD_MODE: {reward_mode!r}')

    train_sac_energy_effiency(config=cfg)
