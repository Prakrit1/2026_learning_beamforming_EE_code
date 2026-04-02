
import numpy as np



def regularized_zero_forcing_precoder_user_specific_normalized(
        channel_matrix: np.ndarray,
        regularization_factor: float,
        power_factors_users: np.ndarray,
) -> np.ndarray:
    """Applies individual power factors to each user, the sum should not exceed power_constraint_watt.
        Currently only works for 1 satellite"""

    precoding_matrix = regularized_zero_forcing_no_norm(
        channel_matrix=channel_matrix,
        regularization_factor = regularization_factor,
    )

    precoding_vectors_norms = np.linalg.norm(precoding_matrix, axis=0) + 1e-12
    precoding_matrix_normalized = precoding_matrix / precoding_vectors_norms

    precoding_matrix_normed = precoding_matrix_normalized * np.sqrt(power_factors_users)[None, :]

    return precoding_matrix_normed


def regularized_zero_forcing_no_norm(
        channel_matrix: np.ndarray,
        regularization_factor: float,
) -> np.ndarray:
    """TODO: Comment"""

    sat_tot_ant_nr = channel_matrix.shape[1]

    precoding_matrix = (
        np.matmul(
            np.linalg.inv(
                np.matmul(channel_matrix.conj().T, channel_matrix)
                + ( regularization_factor + 1e-20
                ) * np.eye(sat_tot_ant_nr)
            ),
            channel_matrix.conj().T
        )
    )

    return precoding_matrix

def regularized_zero_forcing_precoder_user_specific_normalized_without_inversion(
        channel_matrix: np.ndarray,
        regularization_factor: float,
        power_factors_users: np.ndarray,
) -> np.ndarray:
    """Applies individual power factors to each user, the sum should not exceed power_constraint_watt.
        Currently only works for 1 satellite"""

    precoding_matrix = regularized_zero_forcing_no_norm_without_inversion(
        channel_matrix=channel_matrix,
        regularization_factor = regularization_factor,
    )

    precoding_vectors_norms = np.linalg.norm(precoding_matrix, axis=0) + 1e-12
    precoding_matrix_normalized = precoding_matrix / precoding_vectors_norms

    precoding_matrix_normed = precoding_matrix_normalized * np.sqrt(power_factors_users)[None, :]

    return precoding_matrix_normed


def regularized_zero_forcing_no_norm_without_inversion(
        channel_matrix: np.ndarray,
        regularization_factor: float,
) -> np.ndarray:
    """TODO: Comment"""

    sat_tot_ant_nr = channel_matrix.shape[1]

    precoding_matrix = (
        np.matmul(
            np.linalg.inv(
                np.matmul(channel_matrix.conj().T, channel_matrix)
                + ( regularization_factor
                ) * np.eye(sat_tot_ant_nr)
            ),
            channel_matrix.conj().T
        )
    )

    return precoding_matrix
