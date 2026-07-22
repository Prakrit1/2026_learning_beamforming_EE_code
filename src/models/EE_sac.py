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
from src.data.calc_fairness import (
    calc_jain_fairness,
)
from src.data.precoder.mmse_precoder import (
    mmse_precoder_normalized,
)
from src.utils.real_complex_vector_reshaping import (
    real_vector_to_half_complex_vector,
    complex_vector_to_double_real_vector,
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
        # transitions_per_episode, not training_steps_per_episode: batching can change the count
        progress = (
                (training_episode_id * transitions_per_episode + training_step_id + 1)
                / (config.config_learner.training_episodes * transitions_per_episode)
        )
        if not to_log:
            progress_printer(progress=progress, real_time_start=real_time_start)
        else:
            progress_printer(progress=progress, real_time_start=real_time_start, logger=logger)

    def compute_mmse_action_and_reward():
        """Compute the MMSE precoder and its reward on the CURRENT (pre-transition)
        channel, so it lines up with state_current. This must run before
        update_sim() mutates satellite_manager/user_manager in place -- see the
        call site below for why the resulting experience is only assembled and
        added later, once state_next for this same transition is known."""

        # must use erroneous CSI, else the buffer distribution overstates CSI reliability
        w_mmse = mmse_precoder_normalized(
            channel_matrix=satellite_manager.erroneous_channel_state_information,
            **config.mmse_args
        )
        reward_mmse = calc_sum_rate(
            channel_state=satellite_manager.channel_state_information,
            w_precoder=w_mmse,
            noise_power_watt=config.noise_power_watt,
        )
        return w_mmse, reward_mmse

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

        # save_weights() is ~15x faster than model.save() (was ~34% of wall time);
        # requires reconstructing the architecture from config on load, see load_model.py
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

    # Batch the actor forward pass across num_parallel_envs envs (was >90% of wall
    # time as per-sample GPU dispatch overhead); sim stepping stays per-env/unbatched.
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
    # Only ever assigned once the run clears is_past_dinkelbach_warmup
    # (30-episode gate for the Dinkelbach-adaptive reward, see below) --
    # every real production run (13,000 episodes) clears this long before
    # returning, but a short run (e.g. a smoke test with training_episodes <
    # the warmup) would hit an UnboundLocalError on the return statement
    # without this default.
    best_model_path = None
    # Dinkelbach lambda: three mutually-exclusive update modes.
    # Legacy (default): EMA of rate/power updated every step (window in steps,
    # not episodes, via EE_DINKELBACH_LAMBDA_WINDOW_STEPS -- see __main__).
    # Block (EE_DINKELBACH_BLOCK_EPISODES > 0): lambda is held FIXED for an
    # entire block of N episodes while rate/power stats accumulate fresh (not
    # blended with prior-lambda history), then lambda is updated once at the
    # block boundary from that block's mean rate/power and the accumulators
    # reset. This matches the block/outer-loop alternation classical Dinkelbach
    # and its inexact extensions (Jin et al., arXiv:2312.10418) actually
    # require, instead of a continuous EMA drifting on the same timescale as
    # the actor/critic gradient updates -- see docs/EE_formulation.tex Section 6
    # for why the EMA's "fixed inner problem" assumption doesn't hold as
    # implemented.
    # Fixed (EE_DINKELBACH_LAMBDA_FIXED set): lambda is a constant for the
    # ENTIRE run, never updated at all -- takes priority over both modes above.
    # Empirically, block-update (job 144104) tracked the EMA control almost
    # exactly (see docs/EE_formulation.tex), which argues the update mechanism
    # isn't the bottleneck -- this mode tests something more basic first: does
    # the power penalty term have ANY effect on the learned policy at all, for
    # a given constant lambda, before worrying about how lambda evolves.
    dinkelbach_lambda_fixed = getattr(config, 'dinkelbach_lambda_fixed', None)
    lambda_ee = dinkelbach_lambda_fixed if dinkelbach_lambda_fixed is not None else 0.0
    dinkelbach_rate_ema = None
    dinkelbach_power_ema = None
    dinkelbach_lambda_window_steps = getattr(config, 'dinkelbach_lambda_window_steps', transitions_per_episode)
    dinkelbach_lambda_ema_alpha = 2.0 / (dinkelbach_lambda_window_steps + 1)
    dinkelbach_block_episodes = getattr(config, 'dinkelbach_block_episodes', 0)
    dinkelbach_block_rate_sum = 0.0
    dinkelbach_block_power_sum = 0.0
    dinkelbach_block_count = 0
    # warmup before the EMA/block estimate has enough samples to be a meaningful
    # rate/power ratio -- at least one full block, so checkpointing never fires
    # before lambda has been updated from its lambda=0 initialization even once.
    # Fixed mode needs no warmup at all (lambda is meaningful from episode 0).
    dinkelbach_high_score_warmup_episodes = (
        0 if dinkelbach_lambda_fixed is not None else max(30, dinkelbach_block_episodes)
    )

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
            # only used for reward 'energy_efficiency_dinkelbach_adaptive'
            'dinkelbach_rate_per_step': np.nan * np.ones(transitions_per_episode),
            'dinkelbach_power_per_step': np.nan * np.ones(transitions_per_episode),
            # only used when 'fairness' is also in config.config_learner.reward
            'fairness_per_step': np.nan * np.ones(transitions_per_episode),
        }

        for env_idx in range(num_parallel_envs):
            update_sim(config, satellite_managers[env_idx], user_managers[env_idx])
            state_next_list[env_idx] = config.config_learner.get_state(
                config=config,
                user_manager=user_managers[env_idx],
                satellite_manager=satellite_managers[env_idx],
                norm_factors=norm_dict['norm_factors'],
                **config.config_learner.get_state_args
            )

        for group_idx in range(steps_per_episode_batched):

            # one batched actor call across envs instead of num_parallel_envs single calls
            states_batch = np.stack(state_next_list).astype('float32')
            actions_batch = sac.get_action_batch(states=states_batch)

            for env_idx in range(num_parallel_envs):

                # flatten (group_idx, env_idx) into one monotonic transition index
                training_step_id = group_idx * num_parallel_envs + env_idx
                simulation_step = training_episode_id * transitions_per_episode + training_step_id

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
                w_precoder = w_precoder_vector.reshape((config.sat_nr*config.sat_ant_nr, config.user_nr))
                power_precoder = np.real(np.trace(np.matmul(w_precoder.conj().T, w_precoder)))

                # captured pre-clip so no-normalization branches can reward transmitting below budget
                raw_power_precoder = power_precoder

                if power_precoder > config.power_constraint_watt:
                    power_precoder = config.power_constraint_watt
                    norm_factor = np.sqrt(power_precoder / np.trace(np.matmul(w_precoder.conj().T, w_precoder)))
                    normalized_precoder = norm_factor * w_precoder
                    w_precoder = normalized_precoder

                # step simulation based on action, determine reward
                reward = 0

                # Scheme I (paper's Eq. 26 normalization): always rescale to exactly
                # power_constraint_watt, so tr{W^H W} is constant and d(power)/d(action) == 0.
                if 'energy_efficiency' in config.config_learner.reward:
                    if raw_power_precoder < 1e-9:
                        energy_efficiency = 0.0
                    else:
                        scheme_i_norm_factor = np.sqrt(config.power_constraint_watt / raw_power_precoder)
                        w_precoder_scheme_i = scheme_i_norm_factor * w_precoder_vector.reshape(
                            (config.sat_nr * config.sat_ant_nr, config.user_nr)
                        )
                        # always == power_constraint_watt by construction; kept explicit for debug asserts
                        power_precoder_scheme_i = np.real(
                            np.trace(np.matmul(w_precoder_scheme_i.conj().T, w_precoder_scheme_i))
                        )

                        sum_rate_reward = calc_sum_rate(
                            channel_state=satellite_manager.channel_state_information,
                            w_precoder=w_precoder_scheme_i,
                            noise_power_watt=config.noise_power_watt,
                        )
                        # /pa_efficiency: radiated power is only a fraction eta of actual DC draw (Auer et al. 2011)
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

                # Uses raw_power_precoder (pre-clip), so the agent can genuinely be
                # rewarded for transmitting below budget, unlike the Scheme I branch above.
                #
                # RATE term uses w_precoder -- the PHYSICALLY VALID, already-clipped
                # (if it needed clipping) precoder from above; the satellite cannot
                # actually transmit more than the power budget, so the achieved rate
                # must reflect what's actually transmittable.
                #
                # POWER term uses raw_power_precoder UNCONDITIONALLY, not a value that
                # collapses to the constant power_constraint_watt whenever the raw
                # output exceeds budget (the previous version's "if over budget" branch
                # substituted power_precoder, which IS exactly that constant post-clip).
                # A constant contributes zero gradient -- it can't distinguish "5% over
                # budget" from "1700% over budget", so the power penalty went dead
                # exactly in the region the raw network output actually lived in (see
                # preclip_power_diagnostic.py: the fixed-lambda checkpoints sat at
                # 480-1700% of budget on ~98-99.8% of samples). Using raw_power_precoder
                # here always gives a live, continuous gradient proportional to the
                # actual violation size, everywhere, while the rate term above stays
                # physically correct regardless.
                if 'energy_efficiency_no_normalization_fixed' in config.config_learner.reward:
                    w_precoder_raw = w_precoder
                    power_for_ee = raw_power_precoder

                    sum_rate_reward_raw = calc_sum_rate(
                        channel_state=satellite_manager.channel_state_information,
                        w_precoder=w_precoder_raw,
                        noise_power_watt=config.noise_power_watt,
                    )
                    # normalize by power_constraint_watt to match 'energy_efficiency' branch's
                    # reward magnitude -- without it the reward is ~100x smaller, too weak
                    # a gradient against the L2 weight penalty
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

                # Dinkelbach: reward = rate - lambda*power, lambda tracks achieved EE.
                # Avoids the raw ratio's blowup near power->0 while targeting the same
                # optimum as lambda converges. Same clip-only power path as
                # 'energy_efficiency_no_normalization_fixed' (agent is free to transmit
                # below budget). Power divided by power_constraint_watt below to keep
                # lambda O(1) -- raw Watts made the power penalty too weak relative to
                # rate's own noise for the agent to learn to reduce power.
                if 'energy_efficiency_dinkelbach_adaptive' in config.config_learner.reward:
                    # Same fix as 'energy_efficiency_no_normalization_fixed' above: rate
                    # term uses the physically-valid clipped w_precoder, power term uses
                    # raw_power_precoder UNCONDITIONALLY instead of collapsing to the
                    # constant power_precoder whenever over budget -- that constant was
                    # the actual reason the fixed-lambda sweep (jobs 145428-145433) showed
                    # zero power response to lambda: post-clip power stays pinned near
                    # budget regardless of lambda specifically because the reward's power
                    # term went flat/gradient-dead in exactly the region the raw output
                    # lived in (480-1700% of budget on ~98-99.8% of samples, see
                    # preclip_power_diagnostic.py).
                    w_precoder_dinkelbach = w_precoder
                    power_dinkelbach = raw_power_precoder

                    sum_rate_reward_dinkelbach = calc_sum_rate(
                        channel_state=satellite_manager.channel_state_information,
                        w_precoder=w_precoder_dinkelbach,
                        noise_power_watt=config.noise_power_watt,
                    )
                    # amplifier draw correction (Auer et al. 2011)
                    transmit_power_dinkelbach = power_dinkelbach / config.pa_efficiency
                    circuit_power_dinkelbach = config.sat_nr * config.sat_ant_nr * config.circuit_power_watt
                    total_power_dinkelbach_watt = transmit_power_dinkelbach + circuit_power_dinkelbach
                    # lambda_ee is expressed against this reward-scale value, not raw Watts
                    total_power_dinkelbach = total_power_dinkelbach_watt / config.power_constraint_watt

                    energy_efficiency_dinkelbach_adaptive = sum_rate_reward_dinkelbach - lambda_ee * total_power_dinkelbach
                    reward += (
                        config.config_learner.reward['energy_efficiency_dinkelbach_adaptive']
                        * energy_efficiency_dinkelbach_adaptive
                    )

                    episode_metrics['dinkelbach_rate_per_step'][training_step_id] = sum_rate_reward_dinkelbach
                    episode_metrics['dinkelbach_power_per_step'][training_step_id] = total_power_dinkelbach

                    # Optional fairness term, same calc_jain_fairness()/additive
                    # pattern already used in train_sac.py et al. -- reused
                    # as-is, not reimplemented. Evaluated on the same
                    # (physically-valid, clipped) w_precoder_dinkelbach/channel
                    # used for the rate term above, so it rewards the actual
                    # per-user rate spread the agent is being scored on.
                    if 'fairness' in config.config_learner.reward:
                        fairness_reward_dinkelbach = calc_jain_fairness(
                            channel_state=satellite_manager.channel_state_information,
                            w_precoder=w_precoder_dinkelbach,
                            noise_power_watt=config.noise_power_watt,
                        )
                        reward += config.config_learner.reward['fairness'] * fairness_reward_dinkelbach
                        episode_metrics['fairness_per_step'][training_step_id] = fairness_reward_dinkelbach

                    # this step's reward used lambda_ee from before this update
                    if dinkelbach_lambda_fixed is not None:
                        # fixed mode: lambda_ee never changes, nothing to accumulate
                        pass
                    elif dinkelbach_block_episodes > 0:
                        # block mode: accumulate only -- lambda_ee itself is not
                        # touched until the block boundary (end of episode loop)
                        dinkelbach_block_rate_sum += sum_rate_reward_dinkelbach
                        dinkelbach_block_power_sum += total_power_dinkelbach
                        dinkelbach_block_count += 1
                    else:
                        # legacy continuous per-step EMA
                        if dinkelbach_rate_ema is None:
                            dinkelbach_rate_ema = sum_rate_reward_dinkelbach
                            dinkelbach_power_ema = total_power_dinkelbach
                        else:
                            dinkelbach_rate_ema += dinkelbach_lambda_ema_alpha * (sum_rate_reward_dinkelbach - dinkelbach_rate_ema)
                            dinkelbach_power_ema += dinkelbach_lambda_ema_alpha * (total_power_dinkelbach - dinkelbach_power_ema)
                        if dinkelbach_power_ema > 1e-9:
                            lambda_ee = dinkelbach_rate_ema / dinkelbach_power_ema

                valid_reward_keys = [
                    'energy_efficiency',
                    'energy_efficiency_no_normalization_fixed',
                    'energy_efficiency_dinkelbach_adaptive',
                    'fairness',
                ]
                if any(key not in valid_reward_keys for key in config.config_learner.reward.keys()):
                    raise ValueError("No valid reward provided")

                # fail fast on divergence instead of silently training on NaN for hours
                if not np.isfinite(reward):
                    raise RuntimeError(
                        f'Non-finite reward ({reward}) at episode {training_episode_id}, '
                        f'step {training_step_id} (simulation_step {simulation_step}) -- '
                        f'training has diverged (lambda_ee={lambda_ee}). '
                        f'Aborting rather than continuing to train on NaN.'
                    )

                step_experience['reward'] = reward

                # snapshot the MMSE precoder/reward on the PRE-transition channel now
                # (before update_sim() advances satellite_manager/user_manager below),
                # but only assemble and add the experience once state_next for THIS
                # transition is known further down -- pairing it with a leftover
                # state_next from a previous env/step was the original bug here.
                add_mmse_sample = config.rng.random() < config.config_learner.percentage_mmse_samples_added_to_exp_buffer
                if add_mmse_sample:
                    w_mmse, reward_mmse = compute_mmse_action_and_reward()

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
                state_next_list[env_idx] = state_next

                sac.add_experience(experience=step_experience)

                # now that state_next is correct for this transition, add the MMSE
                # experience too (environment evolution is exogenous/action-independent,
                # so the same state_next is valid for the hypothetical "MMSE instead of
                # SAC" transition)
                if add_mmse_sample and (
                        reward_mmse > reward or not config.config_learner.only_add_mmse_samples_with_greater_reward
                ):
                    sac.add_experience(experience={
                        'state': state_current,
                        'action': complex_vector_to_double_real_vector(w_mmse.flatten()),
                        'reward': reward_mmse,
                        'next_state': state_next,
                    })

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
        # entropy_scale_alpha: never logged before this session -- also never
        # persisted in the saved checkpoint (save_model_checkpoint only saves
        # policy network weights), so this is the ONLY way to ever observe
        # how far this codebase's ADAPTIVE alpha (standard Haarnoja et al.
        # dual-gradient-descent entropy tuning against target_entropy) drifts
        # from its initial value over training. The reference paper (Schroder
        # et al., Table 4 "Entropy Scale zeta_e = 1.0") uses a FIXED entropy
        # scale for the entire run, no adaptive tuning at all -- this log
        # line is what lets that comparison actually be checked empirically
        # instead of assumed.
        current_entropy_scale_alpha = float(tf.exp(sac.log_entropy_scale_alpha).numpy())
        logger.info(
            f'Episode {training_episode_id}:'
            f' Episode mean reward: {episode_mean_reward:.4f}'
            f' std {np.nanstd(episode_metrics["reward_per_step"]):.2f},'
            f' current exploration: {np.nanmean(episode_metrics["mean_log_prob_density"]):.2f},'
            f' value loss: {np.nanmean(episode_metrics["value_loss"]):.5f},'
            f' entropy_scale_alpha: {current_entropy_scale_alpha:.4f}'
            # f' curr. lr: {sac.networks["policy"][0]["primary"].optimizer.learning_rate(sac.networks["policy"][0]["primary"].optimizer.iterations):.2E}'
        )
        logger.info(f'Starting training: name={config.config_learner.training_name}, reward={config.config_learner.reward}')

        # checkpoint_score for Dinkelbach must be the achieved EE ratio (mean rate/mean
        # power), not the raw reward: rate - lambda*power trends toward 0 as lambda
        # converges (Dinkelbach fixed point), so a running max over raw reward always
        # picks an early, undertrained episode instead. episode_mean_power below is in
        # reward-scale units (/power_constraint_watt), not raw Watts.
        checkpoint_score = episode_mean_reward
        if 'energy_efficiency_dinkelbach_adaptive' in config.config_learner.reward:
            episode_mean_rate = np.nanmean(episode_metrics['dinkelbach_rate_per_step'])
            episode_mean_power = np.nanmean(episode_metrics['dinkelbach_power_per_step'])

            if dinkelbach_lambda_fixed is not None:
                logger.info(
                    f'Dinkelbach lambda FIXED (constant for entire run) at {lambda_ee:.4f} '
                    f'(episode mean rate {episode_mean_rate:.4f}, '
                    f'mean power {episode_mean_power:.4f} x budget '
                    f'= {episode_mean_power * config.power_constraint_watt:.2f} W DC draw)'
                )
            elif dinkelbach_block_episodes > 0:
                is_block_boundary = (training_episode_id + 1) % dinkelbach_block_episodes == 0
                if is_block_boundary and dinkelbach_block_count > 0:
                    block_mean_rate = dinkelbach_block_rate_sum / dinkelbach_block_count
                    block_mean_power = dinkelbach_block_power_sum / dinkelbach_block_count
                    if block_mean_power > 1e-9:
                        lambda_ee = block_mean_rate / block_mean_power
                    logger.info(
                        f'Dinkelbach lambda updated (block boundary, '
                        f'{dinkelbach_block_episodes} ep) to {lambda_ee:.4f} '
                        f'(block mean rate {block_mean_rate:.4f}, '
                        f'block mean power {block_mean_power:.4f} x budget '
                        f'= {block_mean_power * config.power_constraint_watt:.2f} W DC draw)'
                    )
                    # reset for the next block -- fresh, non-blended batch,
                    # not carried-over history computed under a stale lambda
                    dinkelbach_block_rate_sum = 0.0
                    dinkelbach_block_power_sum = 0.0
                    dinkelbach_block_count = 0
                else:
                    logger.info(
                        f'Dinkelbach lambda held fixed at {lambda_ee:.4f} (within block; '
                        f'episode mean rate {episode_mean_rate:.4f}, '
                        f'mean power {episode_mean_power:.4f} x budget '
                        f'= {episode_mean_power * config.power_constraint_watt:.2f} W DC draw)'
                    )
            else:
                logger.info(
                    f'Dinkelbach lambda updated to {lambda_ee:.4f} '
                    f'(episode mean rate {episode_mean_rate:.4f}, '
                    f'mean power {episode_mean_power:.4f} x budget '
                    f'= {episode_mean_power * config.power_constraint_watt:.2f} W DC draw)'
                )
            checkpoint_score = episode_mean_rate / episode_mean_power if episode_mean_power > 1e-9 else -np.inf

        # save network snapshot
        is_past_dinkelbach_warmup = (
            'energy_efficiency_dinkelbach_adaptive' not in config.config_learner.reward
            or training_episode_id >= dinkelbach_high_score_warmup_episodes
        )
        if is_past_dinkelbach_warmup and checkpoint_score > high_score:
            high_score = checkpoint_score
            high_scores.append(high_score)
            best_model_path = save_model_checkpoint(checkpoint_score)

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

    # env var lets multiple sbatch jobs select a reward variant off the same source
    reward_mode = os.environ.get('EE_REWARD_MODE')

    # fixed CSIT error to train under (paper trains one model per fixed Delta-epsilon_aod
    # rather than only at zero error); defaults to 0.0 for unchanged folder names
    error_bound = float(os.environ.get('EE_TRAIN_ERROR_BOUND', 0.0))
    cfg.config_error_model.error_rng_parametrizations['additive_error_on_cosine_of_aod']['args']['low'] = -error_bound
    cfg.config_error_model.error_rng_parametrizations['additive_error_on_cosine_of_aod']['args']['high'] = error_bound
    error_suffix = f'_aod{error_bound}' if error_bound != 0.0 else ''

    # Dinkelbach lambda EMA window in steps, only used for the dinkelbach_adaptive reward.
    # A window of 200 (less than one 960-step episode) diverged to NaN around episode
    # ~1000; 5000 (wider than one episode) is smooth enough while still updating every step.
    dinkelbach_lambda_window_steps = int(os.environ.get('EE_DINKELBACH_LAMBDA_WINDOW_STEPS', 5000))
    cfg.dinkelbach_lambda_window_steps = dinkelbach_lambda_window_steps

    # Block-update mode: hold lambda fixed for N consecutive episodes (fresh
    # rate/power stats accumulated over exactly that block), then update once
    # from the block average and reset -- matches the block/outer-loop
    # alternation the classical Dinkelbach scheme and its inexact extensions
    # actually require, instead of a continuous per-step EMA drifting on the
    # same timescale as the actor/critic gradient updates. 0 (default)
    # preserves the legacy EMA behavior above; when > 0 this takes over and
    # EE_DINKELBACH_LAMBDA_WINDOW_STEPS is ignored.
    dinkelbach_block_episodes = int(os.environ.get('EE_DINKELBACH_BLOCK_EPISODES', 0))
    cfg.dinkelbach_block_episodes = dinkelbach_block_episodes

    # Fixed-lambda mode: hold lambda CONSTANT for the entire run, never updated
    # (takes priority over both EMA and block modes above). Tests something
    # more basic than either: does the power penalty term have any effect on
    # the learned policy at all, for a given constant lambda -- before trusting
    # any conclusion about how lambda should be updated over time. Unset
    # (default) preserves whichever of the two modes above is selected. Uses a
    # string sentinel (not a float env var defaulting to some number) because
    # 0.0 is itself a value worth testing and must be distinguishable from
    # "not set".
    _dinkelbach_lambda_fixed_str = os.environ.get('EE_DINKELBACH_LAMBDA_FIXED')
    dinkelbach_lambda_fixed = (
        float(_dinkelbach_lambda_fixed_str) if _dinkelbach_lambda_fixed_str is not None else None
    )
    cfg.dinkelbach_lambda_fixed = dinkelbach_lambda_fixed

    # PA efficiency override, for ablating eta_PA independently of system size (N, K)
    # -- see config.py's pa_efficiency comment. Defaults to the config.py value so
    # runs that don't set this env var are unaffected.
    cfg.pa_efficiency = float(os.environ.get('EE_PA_EFFICIENCY', cfg.pa_efficiency))

    # tag training_name with (N, K) so runs under different system sizes never share a folder
    system_suffix = f'_N{cfg.sat_tot_ant_nr}K{cfg.user_nr}'

    # tag training_name with pa_efficiency so checkpoints trained under different eta never mix
    eta_suffix = f'_eta{cfg.pa_efficiency}'

    # Probability [0.0, 1.0] per step of injecting the MMSE precoder (on the current
    # erroneous channel) into the replay buffer as an expert demonstration -- see
    # add_mmse_experience/compute_mmse_action_and_reward above. Testing whether bootstrapping
    # from MMSE helps SAC actually reach MMSE's near-optimal performance at low CSIT
    # error, instead of the flat, error-insensitive curve seen without it.
    mmse_inject_prob = float(os.environ.get('EE_MMSE_INJECT_PROB', 0.0))
    cfg.config_learner.percentage_mmse_samples_added_to_exp_buffer = mmse_inject_prob
    mmse_suffix = f'_mmse{mmse_inject_prob}' if mmse_inject_prob != 0.0 else ''

    # Target entropy override -- tests the "entropy floor" hypothesis: is
    # target_entropy=1.0 (config_sac_learner.py, fixed regardless of N/K;
    # its own comment "SAC heuristic impl. = product of action_space.shape"
    # suggests it was set for a much smaller action space and never rescaled)
    # acting as a floor that prevents the policy from ever sharpening toward
    # near-deterministic, near-optimal precoding at N16K3's much larger
    # ~96-dim action space? Lower target_entropy lets alpha shrink over
    # training (see soft_actor_critic.py's alpha_loss: alpha increases when
    # actual policy entropy exceeds target_entropy, decreases when it falls
    # below), permitting lower actual entropy -- i.e. less residual action
    # noise -- once the value function no longer needs it. Must override
    # BOTH training_args (for logging/consistency) and algorithm_args
    # directly: algorithm_args is built once in _post_init() as
    # {**training_args, ...}, a shallow copy of scalar values, so editing
    # training_args alone after Config() construction would silently not
    # take effect. Defaults to the config.py value (1.0, unset) so runs that
    # don't set this env var are unaffected.
    target_entropy_override = os.environ.get('EE_TARGET_ENTROPY')
    if target_entropy_override is not None:
        target_entropy_value = float(target_entropy_override)
        cfg.config_learner.training_args['target_entropy'] = target_entropy_value
        cfg.config_learner.algorithm_args['target_entropy'] = target_entropy_value
        entropy_suffix = f'_tent{target_entropy_value}'
    else:
        entropy_suffix = ''

    # Entropy-scale LR override -- tests the "adaptive-vs-fixed entropy
    # scale" hypothesis, which is more precisely targeted than the
    # target_entropy override above. The reference paper (Schroder et al.,
    # Table 4: "Entropy Scale zeta_e = 1.0") uses a FIXED entropy scale for
    # the ENTIRE training run (see the paper's Eq. 20: L_mu = L_mu,1 +
    # exp(zeta_e)*L_mu,2, a constant balancing factor) -- no adaptive tuning
    # against a target entropy at all, nowhere in the paper's SAC
    # description. This codebase's actual soft_actor_critic.py instead
    # implements standard Haarnoja et al. ADAPTIVE entropy tuning
    # (log_entropy_scale_alpha is a trainable variable, updated every step
    # via its own optimizer against target_entropy) -- a different algorithm
    # than what the paper validated, even though both start at alpha=1.0.
    # config_sac_learner.py's own comment on entropy_scale_optimizer_args
    # ('LR=0.0 -> No adaptive entropy scale -> manually tune initial entropy
    # scale') already flags that LR=0.0 recovers the paper's fixed-scale
    # behavior -- but the actual default is 1e-3 (adaptive ON), not 0.0.
    # Setting EE_ENTROPY_SCALE_LR=0.0 pins alpha at entropy_scale_alpha_initial
    # (1.0, matching the paper's zeta_e exactly) for the whole run.
    # entropy_scale_optimizer_args is a dict NESTED inside training_args, and
    # algorithm_args's shallow copy shares that nested dict BY REFERENCE
    # (unlike scalar values such as target_entropy) -- so mutating it via
    # training_args alone is sufficient, no separate algorithm_args override
    # needed here. Defaults to the config.py value (1e-3, unset) so runs that
    # don't set this env var are unaffected.
    entropy_scale_lr_override = os.environ.get('EE_ENTROPY_SCALE_LR')
    if entropy_scale_lr_override is not None:
        entropy_scale_lr_value = float(entropy_scale_lr_override)
        cfg.config_learner.training_args['entropy_scale_optimizer_args']['learning_rate'] = entropy_scale_lr_value
        entropy_scale_lr_suffix = f'_escalelr{entropy_scale_lr_value}'
    else:
        entropy_scale_lr_suffix = ''

    # Hidden layer width override -- tests the "network capacity" hypothesis:
    # is hidden_layer_units=[512,512,512,512] (config_sac_learner.py, fixed
    # regardless of N/K) under-capacity for N16K3's state/action space
    # relative to N8K2's much smaller one? Comma-separated list of ints,
    # applied to both the value and policy networks. cfg.config_learner.
    # network_args is the SAME dict object referenced inside algorithm_args
    # (_post_init does 'network_args': self.network_args, not a copy), so
    # editing it here correctly propagates without a separate algorithm_args
    # update. Defaults to unchanged so runs that don't set this env var are
    # unaffected.
    hidden_layer_units_override = os.environ.get('EE_HIDDEN_LAYER_UNITS')
    if hidden_layer_units_override is not None:
        hidden_layer_units_value = [int(u) for u in hidden_layer_units_override.split(',')]
        cfg.config_learner.network_args['value_network_args']['hidden_layer_units'] = hidden_layer_units_value
        cfg.config_learner.network_args['policy_network_args']['hidden_layer_units'] = hidden_layer_units_value
        capacity_suffix = f'_hidden{"x".join(str(u) for u in hidden_layer_units_value)}'
    else:
        capacity_suffix = ''

    # Actor/critic learning rate override -- lets a run use the per-error-bound
    # rates found by the Optuna searches (outputs/optuna_studies/
    # energy_efficiency_dinkelbach_adaptive_aod{0.0,0.025,0.05}.db, see
    # src/models/optuna_train_EE.py) instead of config_sac_learner.py's fixed
    # defaults (shared across all three error bounds even though the searches
    # found materially different optima for each). Both must be set together
    # (a partial override would silently retrain with a hybrid the search
    # never evaluated). Defaults to unset so runs that don't set these env
    # vars are unaffected.
    lr_critic_override = os.environ.get('EE_LR_CRITIC')
    lr_actor_override = os.environ.get('EE_LR_ACTOR')
    if lr_critic_override is not None or lr_actor_override is not None:
        assert lr_critic_override is not None and lr_actor_override is not None, (
            'EE_LR_CRITIC and EE_LR_ACTOR must be set together'
        )
        lr_critic_value = float(lr_critic_override)
        lr_actor_value = float(lr_actor_override)
        cfg.config_learner.network_args['value_network_optimizer_args']['learning_rate'] = lr_critic_value
        cfg.config_learner.network_args['policy_network_optimizer_args']['learning_rate'] = lr_actor_value
        lr_suffix = f'_lrc{lr_critic_value:.2e}_lra{lr_actor_value:.2e}'
    else:
        lr_suffix = ''

    # Optional Jain's-fairness reward term (calc_jain_fairness, reused as-is
    # from train_sac.py et al. -- see the 'fairness' branch inside the
    # energy_efficiency_dinkelbach_adaptive reward block above). Off by
    # default (weight 0.0 means the 'fairness' key is never added to
    # cfg.config_learner.reward, so unset runs are bit-identical to before).
    fairness_weight = float(os.environ.get('EE_FAIRNESS_WEIGHT', 0.0))
    fairness_suffix = f'_fair{fairness_weight}' if fairness_weight != 0.0 else ''

    # Action-bound-mode override -- tests the "unbounded actor output"
    # hypothesis: PolicyNetworkSoft's output_layer_means (network_models.py)
    # is a plain linear Dense layer with no squashing, and
    # action_bound_mode defaults to None (config_sac_learner.py) -- nothing
    # structurally keeps the raw precoder magnitude near the power budget's
    # scale. Combined with clip-only precoding (norm_precoder.py's
    # clip_precoder_to_power_budget) pinning both the TRANSMITTED power and
    # the REWARD's power term to the budget whenever the raw output exceeds
    # it -- true regardless of whether the raw output is 5x or 1700% over --
    # there is no gradient pulling the raw magnitude back down once it
    # drifts past the budget scale, and the pre-clip power diagnostic
    # (src/energy_efficiency/preclip_power_diagnostic.py) confirmed the
    # fixed-lambda checkpoints sit at 480-1700% of budget on average. This
    # is fully-implemented, tested infrastructure (see
    # src/models/helpers/bound_action.py's proper Haarnoja et al. tanh
    # log-prob correction) that has simply never been turned on for any EE
    # run. 'tanh' bounds actions to (-1, 1) per component BEFORE they become
    # precoder weights, giving the raw output a fixed, finite ceiling
    # instead of an unconstrained one. Defaults to the config.py value
    # (None, unset) so runs that don't set this env var are unaffected.
    action_bound_mode_override = os.environ.get('EE_ACTION_BOUND_MODE')
    if action_bound_mode_override is not None:
        cfg.config_learner.action_bound_mode = action_bound_mode_override
        cfg.config_learner.algorithm_args['action_bound_mode'] = action_bound_mode_override
        bound_suffix = f'_abound{action_bound_mode_override}'
    else:
        bound_suffix = ''

    # Marks checkpoints trained under the fixed reward computation (power
    # term uses raw_power_precoder unconditionally, see the
    # energy_efficiency_no_normalization_fixed/energy_efficiency_dinkelbach_adaptive
    # branches above) -- NOT an env var toggle, this is a permanent
    # correctness fix applied to both reward modes from now on, but existing
    # checkpoints under e.g. 'EE_dinkelbach_adaptive_aod0.5_lwin5000_N16K3_eta0.6'
    # were trained under the OLD (gradient-dead-once-over-budget) reward and
    # are not a fair comparison -- this suffix keeps new runs from silently
    # landing in the same folder and getting mixed in with those.
    rawpow_suffix = '_rawpow'

    if reward_mode == 'energy_efficiency':
        cfg.config_learner.reward = {'energy_efficiency': 1.0}  # Scheme I, full EE with circuit power
        cfg.config_learner.training_name = f'full_EE{error_suffix}{system_suffix}{eta_suffix}{mmse_suffix}{entropy_suffix}{entropy_scale_lr_suffix}{capacity_suffix}{bound_suffix}{lr_suffix}'
    elif reward_mode == 'energy_efficiency_no_normalization_fixed':
        cfg.config_learner.reward = {'energy_efficiency_no_normalization_fixed': 1.0}
        cfg.config_learner.training_name = f'EE_no_norm_fixed{error_suffix}{system_suffix}{eta_suffix}{mmse_suffix}{entropy_suffix}{entropy_scale_lr_suffix}{capacity_suffix}{bound_suffix}{rawpow_suffix}{lr_suffix}'
    elif reward_mode == 'energy_efficiency_dinkelbach_adaptive':
        cfg.config_learner.reward = {'energy_efficiency_dinkelbach_adaptive': 1.0}
        if fairness_weight != 0.0:
            cfg.config_learner.reward['fairness'] = fairness_weight
        # lambda_mode_suffix keeps EMA-window, block-update, and fixed-lambda
        # runs (and different block sizes/windows/lambda values) from ever
        # sharing a checkpoint folder
        if dinkelbach_lambda_fixed is not None:
            lambda_mode_suffix = f'_lambdafixed{dinkelbach_lambda_fixed}'
        elif dinkelbach_block_episodes > 0:
            lambda_mode_suffix = f'_block{dinkelbach_block_episodes}'
        else:
            lambda_mode_suffix = f'_lwin{dinkelbach_lambda_window_steps}'
        cfg.config_learner.training_name = f'EE_dinkelbach_adaptive{error_suffix}{lambda_mode_suffix}{system_suffix}{eta_suffix}{mmse_suffix}{entropy_suffix}{entropy_scale_lr_suffix}{capacity_suffix}{bound_suffix}{rawpow_suffix}{lr_suffix}{fairness_suffix}'
    elif reward_mode is not None:
        raise ValueError(f'Unknown EE_REWARD_MODE: {reward_mode!r}')

    train_sac_energy_effiency(config=cfg)
