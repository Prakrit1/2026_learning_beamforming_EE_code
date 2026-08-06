
import numpy as np
import tensorflow as tf

from src.utils.norm_precoder import norm_precoder, clip_precoder_to_power_budget
from src.utils.real_complex_vector_reshaping import real_vector_to_half_complex_vector, rad_and_phase_to_complex_vector


def get_learned_precoder_no_norm(
        state: np.ndarray,
        precoder_network: tf.keras.Model,
        sat_nr: int,
        sat_ant_nr: int,
        user_nr: int,
        action_format: str = 'real_imag',
) -> np.ndarray:

    w_precoder, _ = precoder_network.call(state.astype('float32')[np.newaxis])
    w_precoder = w_precoder.numpy().flatten()

    if action_format == 'rad_phase':
        w_precoder = rad_and_phase_to_complex_vector(w_precoder)
    else:
        w_precoder = real_vector_to_half_complex_vector(w_precoder)
    w_precoder = w_precoder.reshape((sat_nr * sat_ant_nr, user_nr))

    return w_precoder


def get_learned_precoder_normalized(
        state: np.ndarray,
        precoder_network: tf.keras.Model,
        sat_nr: int,
        sat_ant_nr: int,
        user_nr: int,
        power_constraint_watt: float,
        action_format: str = 'real_imag',
) -> np.ndarray:

    w_precoder_no_norm = get_learned_precoder_no_norm(
        state=state,
        precoder_network=precoder_network,
        sat_nr=sat_nr,
        sat_ant_nr=sat_ant_nr,
        user_nr=user_nr,
        action_format=action_format,
    )

    w_precoder_normalized = norm_precoder(
        precoding_matrix=w_precoder_no_norm,
        power_constraint_watt=power_constraint_watt,
        per_satellite=True,
        sat_nr=sat_nr,
        sat_ant_nr=sat_ant_nr
    )

    return w_precoder_normalized


# Prakrit added this for unnormalized (clip-only) power evaluation
def get_learned_precoder_clip_only(
        state: np.ndarray,
        precoder_network: tf.keras.Model,
        sat_nr: int,
        sat_ant_nr: int,
        user_nr: int,
        power_constraint_watt: float,
        action_format: str = 'real_imag',
) -> np.ndarray:
    """
    Clip-only counterpart to get_learned_precoder_normalized: rescales down
    only if the raw network output exceeds the power budget, otherwise
    passes it through unchanged. Use this for models trained with the
    'energy_efficiency_no_normalization_fixed' reward -- using
    get_learned_precoder_normalized on them instead would forcibly rescale
    every sample to the power budget and hide any power savings they learned.
    """

    w_precoder_no_norm = get_learned_precoder_no_norm(
        state=state,
        precoder_network=precoder_network,
        sat_nr=sat_nr,
        sat_ant_nr=sat_ant_nr,
        user_nr=user_nr,
        action_format=action_format,
    )

    w_precoder_clipped = clip_precoder_to_power_budget(
        precoding_matrix=w_precoder_no_norm,
        power_constraint_watt=power_constraint_watt,
        per_satellite=True,
        sat_nr=sat_nr,
        sat_ant_nr=sat_ant_nr,
    )

    return w_precoder_clipped

def get_learned_precoder_reduced(
        state: np.ndarray,
        precoder_network: tf.keras.Model,
        user_nr: int,
        private_part_precoding_style: str,
) -> tuple[float | None, np.ndarray]:


    network_output, _ = precoder_network.call(state.astype('float32')[np.newaxis])
    network_output = network_output.numpy().flatten()

    if private_part_precoding_style == 'RZF':
        regularization_factor = float(np.clip(network_output[0], 0.0, 10))

        power_factors_private_users = network_output[1:user_nr + 1]

        return regularization_factor, power_factors_private_users

    elif private_part_precoding_style in ('MRT', 'MMSE'):
        power_factors_private_users = network_output[0:user_nr]

        return None, power_factors_private_users

    else:
        raise ValueError(f"Unknown private part precoding mode={private_part_precoding_style}")

def get_learned_rsma_power_factor(
        state: np.ndarray,
        power_factor_network: tf.keras.Model,
) -> np.ndarray:

    power_factor_network = power_factor_network.call(state.astype('float32')[np.newaxis])[0]
    power_factor_network = power_factor_network.numpy().flatten()

    # power_factor_network = 1/2 * (np.tanh(power_factor_network) + 1)
    power_factor_network = np.clip(power_factor_network, 0, 1)

    return power_factor_network

def get_learned_rsma_power_and_common_part(
        state: np.ndarray,
        precoder_network: tf.keras.Model,
        user_nr: int,
        private_part_precoding_style: str,
) -> tuple[float | None, np.ndarray, np.ndarray]:


    network_output, _ = precoder_network.call(state.astype('float32')[np.newaxis])
    network_output = network_output.numpy().flatten()

    if private_part_precoding_style == 'RZF':
        regularization_factor = float(np.clip(network_output[0], 0.0, 10))

        power_factors_private_users = network_output[1:user_nr + 1]

        common_part_precoding_no_norm = real_vector_to_half_complex_vector(
            network_output[user_nr + 1:]
        )
        return regularization_factor, power_factors_private_users, common_part_precoding_no_norm

    elif private_part_precoding_style in ('MRT', 'MMSE'):
        power_factors_private_users = network_output[0:user_nr]

        common_part_precoding_no_norm = real_vector_to_half_complex_vector(
            network_output[user_nr:]
        )
        return None, power_factors_private_users, common_part_precoding_no_norm

    else:
        raise ValueError(f"Unknown private part precoding mode={private_part_precoding_style}")

def get_learned_precoder_decentralized_no_norm(
        states: list[np.ndarray],
        precoder_networks: list[tf.keras.Model],
        sat_nr: int,
        sat_ant_nr: int,
        user_nr: int,
) -> np.ndarray:

    w_precoder = np.zeros((sat_nr * sat_ant_nr, user_nr), dtype='complex128')
    for sat_id, (state, precoder_network) in enumerate(zip(states, precoder_networks)):
        w_precoder_sat, _ = precoder_network.call(state.astype('float32')[np.newaxis])
        w_precoder_sat = w_precoder_sat.numpy().flatten()
        w_precoder_sat = real_vector_to_half_complex_vector(w_precoder_sat)
        w_precoder_sat = w_precoder_sat.reshape(sat_ant_nr, user_nr)
        w_precoder[sat_id * sat_ant_nr:sat_id * sat_nr + sat_ant_nr, :] = w_precoder_sat.copy()  # todo copy necessary?

    return w_precoder


def get_learned_precoder_decentralized_normalized(
        states: list[np.ndarray],
        precoder_networks: list[tf.keras.Model],
        sat_nr: int,
        sat_ant_nr: int,
        user_nr: int,
        power_constraint_watt: float,
        action_format: str = 'real_imag',  # unused here (decentralized path predates this option); accepted so **config.learned_precoder_args splats cleanly
) -> np.ndarray:

    w_precoder_no_norm = get_learned_precoder_decentralized_no_norm(
        states=states,
        precoder_networks=precoder_networks,
        sat_nr=sat_nr,
        sat_ant_nr=sat_ant_nr,
        user_nr=user_nr,
    )

    w_precoder_normalized = norm_precoder(
        precoding_matrix=w_precoder_no_norm,
        power_constraint_watt=power_constraint_watt,
        per_satellite=True,
        sat_nr=sat_nr,
        sat_ant_nr=sat_ant_nr,
    )

    return w_precoder_normalized
