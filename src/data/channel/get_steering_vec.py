
import numpy as np

import src


def get_steering_vec(
        satellite: 'src.data.satellite.Satellite',
        phase_aod_steering: float,
) -> np.ndarray:
    """
    Compared to the center of an antenna array, each individual antenna has a slightly
      longer/shorter distance to target. Steering vector gives the additional phase rotation introduced
      by this extra distance.
    """

    steering_idx = np.arange(0, satellite.antenna_nr, dtype='complex128') - (satellite.antenna_nr - 1) / 2  # todo
    # steering_idx = np.arange(0, satellite.antenna_nr, dtype='complex128')

    constant_factor = -1j * 2 * np.pi / satellite.wavelength * satellite.antenna_distance * phase_aod_steering

    steering_vector_to_user = np.exp(
        constant_factor
        * steering_idx
    )

    return steering_vector_to_user

def get_steering_vecs(
        satellite_manager,
        phase_aod_steering_to_users: np.ndarray,
) -> np.ndarray:

    num_users = phase_aod_steering_to_users.shape[1]
    total_antennas = sum(
        satellite.antenna_nr
        for satellite in satellite_manager.satellites
    )

    steering_vectors = np.zeros(
        (num_users, total_antennas),
        dtype='complex128'
    )

    antenna_start_idx = 0

    for satellite_id, satellite in enumerate(satellite_manager.satellites):

        antenna_end_idx = antenna_start_idx + satellite.antenna_nr

        for user_idx in range(num_users):

            phase_aod_steering = phase_aod_steering_to_users[
                satellite_id,
                user_idx
            ]

            steering_vector_to_user = get_steering_vec(
                satellite=satellite,
                phase_aod_steering=phase_aod_steering,
            )

            steering_vectors[
                user_idx,
                antenna_start_idx:antenna_end_idx
            ] = steering_vector_to_user

        antenna_start_idx = antenna_end_idx

    return steering_vectors

