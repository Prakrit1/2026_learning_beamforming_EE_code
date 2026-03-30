
import numpy as np
import tensorflow as tf

import src
from src.models.precoders.learned_precoder import get_learned_precoder_normalized
from src.models.precoders.learned_precoder import get_learned_precoder_decentralized_normalized
from src.models.precoders.learned_precoder import get_learned_rsma_power_factor
from src.models.precoders.learned_precoder import get_learned_rsma_power_and_common_part
from src.models.precoders.adapted_precoder import adapt_robust_slnr_complete_precoder_normed
from src.models.precoders.scaled_precoder import scale_robust_slnr_complete_precoder_normed
from src.data.precoder.mmse_precoder import mmse_precoder_normalized
from src.data.precoder.mmse_precoder import mmse_precoder_user_specific_normalized
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

    # power_factor_network, power_factors_private_users,common_part_precoding_no_norm = get_learned_rsma_power_and_common_part(
    #     state=state,
    #     precoder_network=precoder_network,
    #     user_nr=config.user_nr,
    # )
    #
    power_factors_private_users,common_part_precoding_no_norm = get_learned_rsma_power_and_common_part(
        state=state,
        precoder_network=precoder_network,
        user_nr=config.user_nr,
    )

    # power_constraint_private_part = config.power_constraint_watt ** power_factor_network
    # power_constraint_common_part = config.power_constraint_watt - power_constraint_private_part
    #
    # common_power = np.linalg.norm(common_part_precoding_no_norm, ord=2) + 1e-12
    #
    # if power_constraint_common_part <= 1e-12:
    #     # no common budget -> common part is zero
    #     common_part_precoding = np.zeros_like(common_part_precoding_no_norm)
    # elif common_power <= 1e-9:
    #     # common_raw is (almost) zero but budget > 0 -> use a stable default direction
    #     common_part_precoding = np.zeros_like(common_part_precoding_no_norm)
    # else:
    #     common_part_precoding = (
    #             np.sqrt(power_constraint_common_part) * common_part_precoding_no_norm / common_power
    #     )
    #
    # # Ensuring sum of power of all users is within power_constraint_private_part
    # power_factors_private_users_positive = np.clip(power_factors_private_users, 0, power_constraint_private_part)
    # sum_power_users = np.sum(power_factors_private_users_positive) + 1e-12
    # private_power_scale = min(1, power_constraint_private_part / sum_power_users)
    # power_factors_private_users_normalized = power_factors_private_users_positive * private_power_scale

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
