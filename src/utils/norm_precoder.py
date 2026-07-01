
import numpy as np


def norm_precoder(
        precoding_matrix: np.ndarray,
        power_constraint_watt: float or int,
        per_satellite: bool,
        sat_nr: int = 1,
        sat_ant_nr: int = 1,
) -> np.ndarray:
    """
    Normalizes precoding matrix of dimension (sat_nr * ant_nr, user_nr)
    with sat 1 ant 1, sat1 ant 2, sat1 ant 3, sat 2 ant 1...

    tr(A^H * A) is the sum of squared elements
    after applying norm_factor, the trace of norm_factor * (A^H * A) will be == power_constraint_watt
    """

    normalized_precoder = np.empty(shape=precoding_matrix.shape, dtype='complex128')

    if per_satellite:

        # normalize to (power_constraint_watt / sat_nr) for each satellite
        for satellite_id in range(sat_nr):

            satellite_index_start = satellite_id * sat_ant_nr
            w_precoder_slice = precoding_matrix[satellite_index_start:satellite_index_start + sat_ant_nr, :]

            trace = np.trace(np.matmul(w_precoder_slice.conj().T, w_precoder_slice))

            if trace > 0:  # case: no power -> prevent error
                norm_factor_slice = np.sqrt(
                    power_constraint_watt / sat_nr / np.trace(np.matmul(w_precoder_slice.conj().T, w_precoder_slice))
                )
            else:
                norm_factor_slice = 0.0

            w_precoder_slice_normed = norm_factor_slice * w_precoder_slice

            normalized_precoder[satellite_index_start:satellite_index_start + sat_ant_nr, :] = w_precoder_slice_normed.copy()  # todo: is copy required here?

    else:

        # normalize to power constraint
        norm_factor = np.sqrt(power_constraint_watt / np.trace(np.matmul(precoding_matrix.conj().T, precoding_matrix)))
        normalized_precoder = norm_factor * precoding_matrix

    return normalized_precoder


# Prakrit added this for unnormalized (clip-only) power evaluation
def clip_precoder_to_power_budget(
        precoding_matrix: np.ndarray,
        power_constraint_watt: float or int,
        per_satellite: bool,
        sat_nr: int = 1,
        sat_ant_nr: int = 1,
) -> np.ndarray:
    """
    Like norm_precoder, but only rescales DOWN when the raw precoder already
    exceeds the power budget; below budget it is passed through unchanged.
    Mirrors the clip-only logic used to train the
    'energy_efficiency_no_normalization_fixed' reward (src/models/EE_sac.py),
    so evaluation of that model reflects the same action space it was
    trained on instead of norm_precoder's unconditional rescale-to-budget.
    """

    clipped_precoder = np.empty(shape=precoding_matrix.shape, dtype='complex128')

    if per_satellite:

        budget_per_satellite = power_constraint_watt / sat_nr

        for satellite_id in range(sat_nr):

            satellite_index_start = satellite_id * sat_ant_nr
            w_precoder_slice = precoding_matrix[satellite_index_start:satellite_index_start + sat_ant_nr, :]

            trace = np.real(np.trace(np.matmul(w_precoder_slice.conj().T, w_precoder_slice)))

            if trace > budget_per_satellite:
                norm_factor_slice = np.sqrt(budget_per_satellite / trace)
                w_precoder_slice = norm_factor_slice * w_precoder_slice

            clipped_precoder[satellite_index_start:satellite_index_start + sat_ant_nr, :] = w_precoder_slice.copy()

    else:

        trace = np.real(np.trace(np.matmul(precoding_matrix.conj().T, precoding_matrix)))

        if trace > power_constraint_watt:
            norm_factor = np.sqrt(power_constraint_watt / trace)
            clipped_precoder = norm_factor * precoding_matrix
        else:
            clipped_precoder = precoding_matrix.copy()

    return clipped_precoder
