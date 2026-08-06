
import numpy as np
import tensorflow as tf

import src
from src.models.precoders.learned_precoder import get_learned_precoder_normalized
from src.models.precoders.learned_precoder import get_learned_precoder_clip_only
from src.models.precoders.learned_precoder import get_learned_precoder_no_norm
from src.models.precoders.learned_precoder import get_learned_precoder_reduced
from src.models.precoders.learned_precoder import get_learned_precoder_decentralized_normalized
from src.models.precoders.learned_precoder import get_learned_rsma_power_factor
from src.models.precoders.learned_precoder import get_learned_rsma_power_and_common_part
from src.models.precoders.adapted_precoder import adapt_robust_slnr_complete_precoder_normed
from src.models.precoders.scaled_precoder import scale_robust_slnr_complete_precoder_normed
from src.data.precoder.mmse_precoder import mmse_precoder_normalized
from src.data.precoder.mmse_precoder import mmse_precoder_user_specific_normalized
from src.data.precoder.mrc_precoder import mrc_precoder_user_specific_normalized
from src.data.precoder.regularized_zero_forcing import regularized_zero_forcing_precoder_user_specific_normalized
from src.data.precoder.regularized_zero_forcing import regularized_zero_forcing_precoder_user_specific_normalized_without_inversion
from src.data.precoder.mmse_precoder_decentral import mmse_precoder_decentral_blind_normed
from src.data.precoder.mmse_precoder_decentral import mmse_precoder_decentral_limited_normalized
from src.data.precoder.mrc_precoder import mrc_precoder_normalized
from src.data.precoder.calc_autocorrelation import calc_autocorrelation
from src.data.precoder.robust_SLNR_precoder import robust_SLNR_precoder_no_norm
from src.data.precoder.rate_splitting import rate_splitting_no_norm


def get_precoding_learned(
        config: 'src.config.config.Config',
        user_manager: 'src.data.user_manager.UserManager',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
        norm_factors: dict,
        precoder_network: tf.keras.models.Model,
) -> np.ndarray:

    state = config.config_learner.get_state(
        config=config,
        user_manager=user_manager,
        satellite_manager=satellite_manager,
        norm_factors=norm_factors,
        **config.config_learner.get_state_args
    )

    w_precoder_normalized = get_learned_precoder_normalized(
        state=state,
        precoder_network=precoder_network,
        **config.learned_precoder_args,
    )

    return w_precoder_normalized


# Prakrit added this for unnormalized (clip-only) power evaluation
def get_precoding_learned_clip_only(
        config: 'src.config.config.Config',
        user_manager: 'src.data.user_manager.UserManager',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
        norm_factors: dict,
        precoder_network: tf.keras.models.Model,
) -> np.ndarray:
    """
    Clip-only counterpart to get_precoding_learned: use this for models
    trained with the 'energy_efficiency_no_normalization_fixed' reward, so
    evaluation only clips down when the raw output exceeds the power budget
    instead of always rescaling to sit exactly on it.
    """

    state = config.config_learner.get_state(
        config=config,
        user_manager=user_manager,
        satellite_manager=satellite_manager,
        norm_factors=norm_factors,
        **config.config_learner.get_state_args
    )

    w_precoder_clipped = get_learned_precoder_clip_only(
        state=state,
        precoder_network=precoder_network,
        **config.learned_precoder_args,
    )

    return w_precoder_clipped


# Added to diagnose whether the clip-only precoder's power is being
# controlled by the reward gradient at all, or is dominated by the clip
# operation itself: this returns the RAW network output before either
# clipping or normalization, so its power can be compared against the
# post-clip power measured by get_precoding_learned_clip_only. If the clip
# triggers on nearly every sample regardless of lambda (see
# tx_power_distribution.py's fixed-lambda summary: only 1-2% of samples
# land below budget even at lambda=3.0/5.0), then whatever the reward
# gradient taught the policy about raw magnitude is invisible in the
# post-clip power the model is actually evaluated on.
def get_precoding_learned_no_norm(
        config: 'src.config.config.Config',
        user_manager: 'src.data.user_manager.UserManager',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
        norm_factors: dict,
        precoder_network: tf.keras.models.Model,
) -> np.ndarray:
    """
    Raw counterpart to get_precoding_learned_clip_only: returns the network's
    output completely unclipped and unnormalized. NOT a valid precoder to
    transmit with (it can exceed the power budget) -- diagnostic use only,
    to measure what the network would output before the clip masks it.
    """

    state = config.config_learner.get_state(
        config=config,
        user_manager=user_manager,
        satellite_manager=satellite_manager,
        norm_factors=norm_factors,
        **config.config_learner.get_state_args
    )

    w_precoder_no_norm = get_learned_precoder_no_norm(
        state=state,
        precoder_network=precoder_network,
        sat_nr=config.learned_precoder_args['sat_nr'],
        sat_ant_nr=config.learned_precoder_args['sat_ant_nr'],
        user_nr=config.learned_precoder_args['user_nr'],
        action_format=config.learned_precoder_args.get('action_format', 'real_imag'),
    )

    return w_precoder_no_norm


def get_precoding_learned_reduced(
        config: 'src.config.config.Config',
        user_manager: 'src.data.user_manager.UserManager',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
        norm_factors: dict,
        precoder_network: tf.keras.models.Model,
) -> np.array:

    state = config.config_learner.get_state(
        config=config,
        user_manager=user_manager,
        satellite_manager=satellite_manager,
        norm_factors=norm_factors,
        **config.config_learner.get_state_args
    )

    regularization_factor, power_factors_private_users = get_learned_precoder_reduced(
        state=state,
        precoder_network=precoder_network,
        user_nr=config.user_nr,
        private_part_precoding_style=config.private_part_precoding_style,
    )

    # Ensuring sum of power of all users is within power_constraint_private_part
    power_factors_private_users_positive = np.clip(power_factors_private_users, 0, config.power_constraint_watt)
    sum_power = np.sum(power_factors_private_users_positive) + 1e-12

    power_scale = min(1, config.power_constraint_watt / sum_power)
    power_factors_private_users_normalized = power_factors_private_users_positive * power_scale

    power_constraint_private_part = np.sum(power_factors_private_users_normalized)

    if regularization_factor is not None:

        if not config.matrix_inversion_approximation:

            private_part_precoding = regularized_zero_forcing_precoder_user_specific_normalized(
                channel_matrix=satellite_manager.erroneous_channel_state_information,
                regularization_factor=regularization_factor,
                power_factors_users=power_factors_private_users_normalized,
            )

        else:
            private_part_precoding = regularized_zero_forcing_precoder_user_specific_normalized_without_inversion(
                channel_matrix=satellite_manager.erroneous_channel_state_information,
                regularization_factor=regularization_factor,
                power_factors_users=power_factors_private_users_normalized,
            )

    else:

        if config.private_part_precoding_style == 'MRT':

            private_part_precoding = mrc_precoder_user_specific_normalized(
                channel_matrix=satellite_manager.erroneous_channel_state_information,
                power_factors_users=power_factors_private_users_normalized,
            )

        elif config.private_part_precoding_style == 'MMSE':

            private_part_precoding = mmse_precoder_user_specific_normalized(
                channel_matrix=satellite_manager.erroneous_channel_state_information,
                noise_power_watt=config.noise_power_watt,
                power_constraint_watt=power_constraint_private_part,
                power_factors_users=power_factors_private_users_normalized,
            )

    w_precoder = private_part_precoding

    power_precoding = np.trace(w_precoder.conj().T @ w_precoder)

    if power_precoding > 1.0001 * config.power_constraint_watt:
        raise ValueError('Warning: The power constraint is not met')

    return w_precoder


def get_precoding_adapted_slnr_complete(
        config: 'src.config.config.Config',
        user_manager: 'src.data.user_manager.UserManager',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
        norm_factors: dict,
        scaling_network: tf.keras.models.Model,
) -> np.ndarray:

    scaler_input_state = config.config_learner.get_state(
        config=config,
        user_manager=user_manager,
        satellite_manager=satellite_manager,
        norm_factors=norm_factors,
        **config.config_learner.get_state_args
    )

    precoding = adapt_robust_slnr_complete_precoder_normed(
        satellite=satellite_manager.satellites[0],
        error_model_config=config.config_error_model,
        error_distribution='uniform',
        channel_matrix=satellite_manager.erroneous_channel_state_information,
        noise_power_watt=config.noise_power_watt,
        power_constraint_watt=config.power_constraint_watt,
        scaling_network=scaling_network,
        scaler_input_state=scaler_input_state,
        sat_nr=config.sat_nr,
        sat_ant_nr=config.sat_ant_nr,
    )

    return precoding


def get_precoding_adapted_slnr_powerscaled(
        config: 'src.config.config.Config',
        user_manager: 'src.data.user_manager.UserManager',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
        norm_factors: dict,
        scaling_network: tf.keras.models.Model,
) -> np.ndarray:

    scaler_input_state = config.config_learner.get_state(
        config=config,
        user_manager=user_manager,
        satellite_manager=satellite_manager,
        norm_factors=norm_factors,
        **config.config_learner.get_state_args
    )

    precoding = scale_robust_slnr_complete_precoder_normed(
        satellite=satellite_manager.satellites[0],
        error_model_config=config.config_error_model,
        error_distribution='uniform',
        channel_matrix=satellite_manager.erroneous_channel_state_information,
        noise_power_watt=config.noise_power_watt,
        power_constraint_watt=config.power_constraint_watt,
        scaling_network=scaling_network,
        scaler_input_state=scaler_input_state,
        sat_nr=config.sat_nr,
        sat_ant_nr=config.sat_ant_nr,
    )

    return precoding


def get_precoding_learned_rsma_complete(
        config: 'src.config.config.Config',
        user_manager: 'src.data.user_manager.UserManager',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
        norm_factors: dict,
        precoder_network: tf.keras.models.Model,
) -> np.ndarray:

    state = config.config_learner.get_state(
        config=config,
        user_manager=user_manager,
        satellite_manager=satellite_manager,
        norm_factors=norm_factors,
        **config.config_learner.get_state_args
    )

    w_precoder_normalized = get_learned_precoder_normalized(
        state=state,
        precoder_network=precoder_network,
        sat_nr=config.sat_nr,
        sat_ant_nr=config.sat_ant_nr,
        user_nr=config.user_nr + 1,
        power_constraint_watt=config.power_constraint_watt,
    )

    return w_precoder_normalized

def get_precoding_learned_rsma_power_scaling(
        config: 'src.config.config.Config',
        user_manager: 'src.data.user_manager.UserManager',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
        norm_factors: dict,
        power_factor_network: tf.keras.models.Model,
) -> np.ndarray:

    state = config.config_learner.get_state(
        config=config,
        user_manager=user_manager,
        satellite_manager=satellite_manager,
        norm_factors=norm_factors,
        **config.config_learner.get_state_args
    )

    rsma_factor = get_learned_rsma_power_factor(
        state=state,
        power_factor_network=power_factor_network,
    )

    w_precoder = rate_splitting_no_norm(
        channel_matrix=satellite_manager.erroneous_channel_state_information,
        noise_power_watt=config.noise_power_watt,
        power_constraint_watt=config.power_constraint_watt,
        rsma_factor=rsma_factor,
        common_part_precoding_style=config.common_part_precoding_style,
    )

    return w_precoder

def get_precoding_learned_rsma_power_and_common_part(
        config: 'src.config.config.Config',
        user_manager: 'src.data.user_manager.UserManager',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
        norm_factors: dict,
        precoder_network: tf.keras.models.Model,
) -> np.array:

    state = config.config_learner.get_state(
        config=config,
        user_manager=user_manager,
        satellite_manager=satellite_manager,
        norm_factors=norm_factors,
        **config.config_learner.get_state_args
    )

    regularization_factor, power_factors_private_users, common_part_precoding_no_norm = get_learned_rsma_power_and_common_part(
        state=state,
        precoder_network=precoder_network,
        user_nr=config.user_nr,
        private_part_precoding_style=config.private_part_precoding_style,
    )

    common_part_norm = np.linalg.norm(common_part_precoding_no_norm, ord=2)
    power_common_part = common_part_norm ** 2
    # common_part_precoding = np.sqrt(power_constraint_common_part) * common_part_precoding_no_norm / common_power

    # Ensuring sum of power of all users is within power_constraint_private_part
    power_factors_private_users_positive = np.clip(power_factors_private_users, 0, config.power_constraint_watt)
    sum_power = np.sum(power_factors_private_users_positive) + power_common_part + 1e-12

    power_scale = min(1, config.power_constraint_watt / sum_power)
    power_factors_private_users_normalized = power_factors_private_users_positive * power_scale

    power_constraint_common_part = power_common_part * power_scale

    if common_part_norm <= 1e-12:
        common_part_precoding = np.zeros_like(common_part_precoding_no_norm)
    else:
        common_part_precoding = (
                np.sqrt(power_constraint_common_part) * common_part_precoding_no_norm / common_part_norm
        )

    power_constraint_private_part = np.sum(power_factors_private_users_normalized)

    if regularization_factor is not None:

        if not config.matrix_inversion_approximation:

            private_part_precoding = regularized_zero_forcing_precoder_user_specific_normalized(
                channel_matrix=satellite_manager.erroneous_channel_state_information,
                regularization_factor=regularization_factor,
                power_factors_users=power_factors_private_users_normalized,
            )

        else:
            private_part_precoding = regularized_zero_forcing_precoder_user_specific_normalized_without_inversion(
                channel_matrix=satellite_manager.erroneous_channel_state_information,
                regularization_factor=regularization_factor,
                power_factors_users=power_factors_private_users_normalized,
            )

    else:

        if config.private_part_precoding_style == 'MRT':

            private_part_precoding = mrc_precoder_user_specific_normalized(
                channel_matrix=satellite_manager.erroneous_channel_state_information,
                power_factors_users=power_factors_private_users_normalized,
            )

        elif config.private_part_precoding_style == 'MMSE':

            private_part_precoding = mmse_precoder_user_specific_normalized(
                channel_matrix=satellite_manager.erroneous_channel_state_information,
                noise_power_watt=config.noise_power_watt,
                power_constraint_watt=power_constraint_private_part,
                power_factors_users=power_factors_private_users_normalized,
            )

    w_precoder = np.hstack([common_part_precoding[np.newaxis].T, private_part_precoding])

    power_precoding = np.trace(w_precoder.conj().T @ w_precoder)

    if power_precoding>1.0001*config.power_constraint_watt:
        raise ValueError('Warning: The power constraint is not met')

    return w_precoder

def get_precoding_learned_decentralized_blind(
        config: 'src.config.config.Config',
        user_manager: 'src.data.user_manager.UserManager',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
        norm_factors: dict,
        precoder_networks: list[tf.keras.models.Model],
) -> np.ndarray:

    states = config.config_learner.get_state(
        config=config,
        user_manager=user_manager,
        satellite_manager=satellite_manager,
        norm_factors=norm_factors,
        **config.config_learner.get_state_args,
        per_sat=True
    )

    w_precoder_normalized = get_learned_precoder_decentralized_normalized(
        states=states,
        precoder_networks=precoder_networks,
        **config.learned_precoder_args,
    )

    return w_precoder_normalized


def get_precoding_learned_decentralized_limited(
        config: 'src.config.config.Config',
        user_manager: 'src.data.user_manager.UserManager',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
        norm_factors: dict,
        precoder_networks: list[tf.keras.models.Model],
) -> np.ndarray:

    states = config.config_learner.get_state(
        config=config,
        user_manager=user_manager,
        satellite_manager=satellite_manager,
        norm_factors=norm_factors,
        **config.config_learner.get_state_args,
    )

    w_precoder_normalized = get_learned_precoder_decentralized_normalized(
        states=states,
        precoder_networks=precoder_networks,
        **config.learned_precoder_args,
    )

    return w_precoder_normalized


def get_precoding_mmse(
        config: 'src.config.config.Config',
        user_manager: 'src.data.user_manager.UserManager',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
) -> np.ndarray:

    w_mmse = mmse_precoder_normalized(
        channel_matrix=satellite_manager.erroneous_channel_state_information,
        **config.mmse_args,
    )

    return w_mmse


def get_precoding_mmse_decentralized_blind(
        config: 'src.config.config.Config',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
) -> np.ndarray:

    w_mmse = mmse_precoder_decentral_blind_normed(
        erroneous_csit_per_sat=satellite_manager.get_erroneous_channel_state_information_per_sat(),
        **config.mmse_args,
    )

    return w_mmse


def get_precoding_mmse_decentralized_limited(
        config: 'src.config.config.Config',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
) -> np.ndarray:

    local_channel_matrices = [
        satellite_manager.get_local_channel_state_information(sat_idx, config.local_csi_own_quality,
                                                              config.local_csi_others_quality)
        for sat_idx in range(config.sat_nr)]

    w_mmse_normalized = mmse_precoder_decentral_limited_normalized(
        local_channel_matrices=local_channel_matrices,
        **config.mmse_args,
    )

    return w_mmse_normalized


def get_precoding_mrc(
        config: 'src.config.config.Config',
        user_manager: 'src.data.user_manager.UserManager',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
) -> np.ndarray:

    w_mrc = mrc_precoder_normalized(
        channel_matrix=satellite_manager.erroneous_channel_state_information,
        **config.mrc_args,
    )

    return w_mrc


# Added -- no ZF wrapper existed before this. Pure zero-forcing is
# regularized_zero_forcing_precoder_user_specific_normalized with
# regularization_factor=0 (a nonzero factor makes it REGULARIZED ZF, which
# with the right factor value is mathematically identical to MMSE -- see
# that function's docstring). Equal power split across users, summing to
# exactly power_constraint_watt, matches the convention get_precoding_mmse/
# get_precoding_mrc already use (both are pinned at the full budget by
# construction too) -- makes the three classical baselines directly
# comparable on the same terms.
def get_precoding_zero_forcing(
        config: 'src.config.config.Config',
        user_manager: 'src.data.user_manager.UserManager',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
) -> np.ndarray:

    power_factors_users = (config.power_constraint_watt / config.user_nr) * np.ones(config.user_nr)

    w_zf = regularized_zero_forcing_precoder_user_specific_normalized(
        channel_matrix=satellite_manager.erroneous_channel_state_information,
        regularization_factor=0.0,
        power_factors_users=power_factors_users,
    )

    return w_zf


def get_precoding_robust_slnr(
        config: 'src.config.config.Config',
        user_manager: 'src.data.user_manager.UserManager',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
) -> np.ndarray:

    autocorrelation = calc_autocorrelation(
        satellite=satellite_manager.satellites[0],
        error_model_config=config.config_error_model,
        error_distribution='uniform',
    )

    w_robust_slnr = robust_SLNR_precoder_no_norm(
        channel_matrix=satellite_manager.erroneous_channel_state_information,
        autocorrelation_matrix=autocorrelation,
        noise_power_watt=config.noise_power_watt,
        power_constraint_watt=config.power_constraint_watt,
    )

    return w_robust_slnr


def get_precoding_rsma(
        config: 'src.config.config.Config',
        user_manager: 'src.data.user_manager.UserManager',
        satellite_manager: 'src.data.satellite_manager.SatelliteManager',
        rsma_factor: float,
        common_part_precoding_style: str,
) -> np.ndarray:

    w_rsma = rate_splitting_no_norm(
        channel_matrix=satellite_manager.erroneous_channel_state_information,
        noise_power_watt=config.noise_power_watt,
        power_constraint_watt=config.power_constraint_watt,
        rsma_factor=rsma_factor,
        common_part_precoding_style=common_part_precoding_style
    )

    return w_rsma
