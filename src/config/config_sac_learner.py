
import tensorflow as tf

from src.models.helpers.get_state import (
    get_state_erroneous_channel_state_information,
    get_state_aods,
)


class ConfigSACLearner:
    """Defines parameters to use when learning with Soft Actor Critic."""

    def __init__(
            self,
            sat_nr,
            sat_ant_nr,
            user_nr,
    ) -> None:

        self.training_name: str = (
             #'full_EE'
             'EE_no_norm_fixed'

        )
        # NOTE: for EE_sac.py runs (full_EE / EE_no_norm_fixed / dinkelbach
        # variants), self.training_name and self.reward below are IGNORED --
        # EE_sac.py's __main__ overwrites both based on the EE_REWARD_MODE
        # env var set in the .slurm file, so multiple sbatch jobs can run
        # concurrently off this one unedited file instead of racing on a
        # hand-edited default. See EE_sac.py's __main__ for the full list of
        # valid EE_REWARD_MODE values:
        #   energy_efficiency                        -> full_EE (Scheme I, always full power)
        #   energy_efficiency_no_normalization_fixed  -> EE_no_norm_fixed (clip-only, direct ratio reward)
        #   energy_efficiency_dinkelbach_adaptive     -> EE_dinkelbach_adaptive (clip-only, adaptive-lambda reward)
        # Also set EE_TRAIN_ERROR_BOUND (float, default 0.0) to train at a
        # fixed CSIT error bound; it appends '_aod{bound}' to the training
        # name automatically. The values below only apply when
        # EE_REWARD_MODE is unset (e.g. journal_training_sweeps/*.py, which
        # calls Config() directly without going through EE_sac.py).
        self.reward: dict = {
            #'energy_efficiency': 1.0,           # full EE with circuit power
            'energy_efficiency_no_normalization_fixed': 1.0,                    # energy efficiency without normalization
        }

        self.get_state = get_state_erroneous_channel_state_information
        self.get_state_args = {
            'csi_format': 'rad_phase',  # 'rad_phase', 'rad_phase_reduced', 'real_imag'
            'norm_state': True,  # !!HEURISTIC!!, this will break if you dramatically change the setup
        }
        self.get_state_norm_factors_iterations: int = 100_000  # how many samples to calculate means and stds

        self.percentage_mmse_samples_added_to_exp_buffer: float = 0.0  # [0.0, 1.0] chance for mmse action to be added
        self.only_add_mmse_samples_with_greater_reward: bool = True  # only add samples with reward_mmse > reward_sac

        self.action_bound_mode: str or None = None  # [>None, 'tanh', 'tanh_positive'] bound the actor NN output?
        self.training_args: dict = {
            'future_reward_discount_gamma': 0.0,  # Exponential future reward discount for stability
            'entropy_scale_alpha_initial': 1.0,  # weights the 'soft' entropy penalty against the td error
            'target_entropy': 1.0,  # SAC heuristic impl. = product of action_space.shape
            'entropy_scale_optimizer': tf.keras.optimizers.SGD,
            'entropy_scale_optimizer_args': {
                'learning_rate': 1e-3,  # LR=0.0 -> No adaptive entropy scale -> manually tune initial entropy scale
            },
            'training_minimum_experiences': 1_000,
            'training_batch_size': 1024,
            'training_target_update_momentum_tau': 0,  # How much of the primary network copy to target networks
            'training_l2_norm_scale_value': 0.01,
            'training_l2_norm_scale_policy': 0.01,  # related to weight decay
        }
        self.experience_buffer_args: dict = {
            'buffer_size': 100_000,
            'priority_scale_alpha': 0.0,  # alpha in [0, 1], alpha=0 uniform sampling, 1 is fully prioritized sampling
            'importance_sampling_correction_beta': 1.0  # beta in [0%, 100%], beta=100% is full correction
        }
        self.train_policy_every_k_steps: int = 10  # train policy only every k steps
        self.train_policy_after_j_steps: int = 0  # start training policy after j steps
        self.train_value_every_k_steps: int = 10  # train value only every k steps
        self.train_value_after_j_steps: int = 0  # start training value after j steps
        self.network_args: dict = {
            'value_network_args': {
                'hidden_layer_units': [512, 512, 512, 512, ],
                'activation_hidden': 'leaky_relu',  # >'relu', 'leaky_relu', 'tanh', 'penalized_tanh', 'shaped_tanh'
                'kernel_initializer_hidden': 'glorot_uniform',  # >'glorot_uniform', 'he_uniform'
                'batch_norm_input': False,  # might not be a good idea due to 32bit input
                'batch_norm': True,  # too high learning rate will lead to problems w/ weights changing too fast
            },
            'value_network_optimizer': tf.keras.optimizers.Adam,
            'value_network_optimizer_args': {
                # Matches Table 4's Optuna-tuned Critic LR (8.8e-6) -- was 1e-3
                # (~114x higher), a pre-existing repo default unrelated to the
                # EE reward work, flagged as a likely contributor to the
                # Dinkelbach rolling-window run's NaN divergence.
                'learning_rate': 8.8e-6,
                # 'learning_rate': tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=7e-5,
                #                                                                    first_decay_steps=100),
                'amsgrad': False,
            },
            'policy_network_args': {
                'hidden_layer_units': [512, 512, 512, 512, ],
                'activation_hidden': 'penalized_tanh',  # >'relu', 'leaky_relu', 'tanh', 'penalized_tanh', 'shaped_tanh'
                'kernel_initializer_hidden': 'glorot_uniform',  # >'glorot_uniform', 'he_uniform'
                'batch_norm_input': False,  # might not be a good idea due to 32bit input
                'batch_norm': True,  # too high learning rate will lead to problems w/ weights changing too fast
            },
            'policy_network_optimizer': tf.keras.optimizers.Adam,
            'policy_network_optimizer_args': {
                # Matches Table 4's Optuna-tuned Actor LR (4.2e-5) -- was 1e-4 (~2.4x higher).
                 'learning_rate': 4.2e-5,
                # 'learning_rate': tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=7e-6,
                #                                                                    first_decay_steps=100),
                'amsgrad': True,
            },
        }

        # TRAINING
        self.training_episodes: int = 13_000  # a new episode is a full reset of the simulation environment
        self.training_steps_per_episode: int = 1_000

        # --- PARALLEL PROCESSING -----------------------------------------------
        # Number of independent simulation environments stepped together per training
        # iteration. update_sim/reward calc stay per-environment (cheap, ~0.5ms/call);
        # only the actor forward pass (get_action) is batched across environments,
        # since GPU dispatch overhead -- not compute -- dominates a batch=1 forward
        # pass through a small MLP. Actual transitions collected per episode become
        # (training_steps_per_episode // num_parallel_envs) * num_parallel_envs.
        # Was the single biggest lever for the original ~14h/run training time:
        # profiling showed >90% of wall time was GPU dispatch overhead from calling
        # get_action() one sample at a time, not actual simulation or gradient-step
        # compute. See get_action_batch() in soft_actor_critic.py and the parallel-env
        # loop in EE_sac.py's train_sac_energy_effiency for the implementation.
        self.num_parallel_envs: int = 64

        self._post_init(sat_nr=sat_nr, sat_ant_nr=sat_ant_nr, user_nr=user_nr)

    def _post_init(
            self,
            sat_nr,
            sat_ant_nr,
            user_nr,
    ) -> None:

        self.training_args['training_minimum_experiences'] = max(self.training_args['training_minimum_experiences'],
                                                                 self.training_args['training_batch_size'])

        self.update(sat_nr=sat_nr, sat_ant_nr=sat_ant_nr, user_nr=user_nr)

        # Collected args
        self.algorithm_args = {
            **self.training_args,
            'network_args': self.network_args,
            'experience_buffer_args': self.experience_buffer_args,
            'action_bound_mode': self.action_bound_mode,
        }

    def update(
            self,
            sat_nr: int,
            sat_ant_nr: int,
            user_nr: int,
    ) -> None:
        """Update those config parameters that are calculated from others."""

        if self.get_state == get_state_aods:
            self.network_args['size_state'] = sat_nr * user_nr
        elif self.get_state == get_state_erroneous_channel_state_information:
            if self.get_state_args['csi_format'] in ['rad_phase', 'real_imag']:
                self.network_args['size_state'] = 2 * sat_nr * sat_ant_nr * user_nr
            elif self.get_state_args['csi_format'] == 'rad_phase_reduced':
                self.network_args['size_state'] = sat_nr * sat_ant_nr * user_nr + sat_nr * user_nr

    def policy_training_criterion(
            self,
            simulation_step
    ) -> bool:
        """Train policy networks only every k steps and/or only after j total steps to ensure a good value function"""
        if (
                simulation_step > self.train_policy_after_j_steps
                and
                (simulation_step % self.train_policy_every_k_steps) == 0
        ):
            return True
        return False

    def value_training_criterion(
            self,
            simulation_step
    ) -> bool:
        """Train value networks only every k steps and/or only after j total steps"""
        if (
                simulation_step > self.train_value_after_j_steps
                and
                (simulation_step % self.train_value_every_k_steps) == 0
        ):
            return True
        return False
