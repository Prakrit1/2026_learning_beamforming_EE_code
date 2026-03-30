
import numpy as np

from src.utils.norm_precoder import (
    norm_precoder,
)


def mmse_precoder_normalized(
        channel_matrix: np.ndarray,
        noise_power_watt: float,
        power_constraint_watt: float,
        sat_nr: int,
        sat_ant_nr: int,
) -> np.ndarray:
    """TODO: Comment"""

    precoding_matrix = mmse_precoder_no_norm(
        channel_matrix=channel_matrix,
        noise_power_watt=noise_power_watt,
        power_constraint_watt=power_constraint_watt,
    )

    precoding_matrix_normed = norm_precoder(
        precoding_matrix=precoding_matrix,
        power_constraint_watt=power_constraint_watt,
        per_satellite=True,
        sat_nr=sat_nr,
        sat_ant_nr=sat_ant_nr,
    )

    return precoding_matrix_normed

def mmse_precoder_user_specific_normalized(
        channel_matrix: np.ndarray,
        noise_power_watt: float,
        power_constraint_watt: float,
        power_factors_users: np.ndarray,
) -> np.ndarray:
    """Applies individual power factors to each user, the sum should not exceed power_constraint_watt.
        Currently only works for 1 satellite"""

    precoding_matrix = mmse_precoder_no_norm(
        channel_matrix=channel_matrix,
        noise_power_watt=noise_power_watt,
        power_constraint_watt=power_constraint_watt,
    )

    precoding_vectors_norms = np.linalg.norm(precoding_matrix, axis=0) + 1e-12
    precoding_matrix_normalized = precoding_matrix / precoding_vectors_norms

    precoding_matrix_normed = precoding_matrix_normalized * np.sqrt(power_factors_users)[None, :]

    return precoding_matrix_normed


def mmse_precoder_no_norm(
        channel_matrix: np.ndarray,
        noise_power_watt: float,
        power_constraint_watt: float,
) -> np.ndarray:
    """TODO: Comment"""

    inversion_constant_lambda = 1e-22 #np.finfo('float32').tiny
    # inversion_constant_lambda = 0

    user_nr = channel_matrix.shape[0]
    sat_tot_ant_nr = channel_matrix.shape[1]

    precoding_matrix = (
        np.matmul(
            np.linalg.inv(
                np.matmul(channel_matrix.conj().T, channel_matrix)
                + (
                    noise_power_watt
                    * user_nr
                    / power_constraint_watt
                    + inversion_constant_lambda
                ) * np.eye(sat_tot_ant_nr)
            ),
            channel_matrix.conj().T
        )
    )

    return precoding_matrix
