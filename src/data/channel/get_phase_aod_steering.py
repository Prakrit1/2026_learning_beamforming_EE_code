
import numpy as np
import src

def get_phase_aod_steering(
        satellite: 'src.data.satellite.Satellite',
        users: list,
        scale: float = 0,
) -> np.ndarray:
    """
    Calculates the phase value that is handed to the steering vector:
        cos(aod + additive_error_on_aod) + additive_error_on_cosine_of_aod
    """

    phase_aod_steering_to_users = np.zeros(len(users))

    # scale
    errors = {
        'additive_error_on_aod': scale * satellite.estimation_errors['additive_error_on_aod'],
        'additive_error_on_cosine_of_aod': scale * satellite.estimation_errors['additive_error_on_cosine_of_aod'],
    }

    for user in users:

        phase_aod_steering_to_users[user.idx] = (
            np.cos(
                satellite.aods_to_users[user.idx]
                + errors['additive_error_on_aod'][user.idx]
            )
            + errors['additive_error_on_cosine_of_aod'][user.idx]
        )

    return phase_aod_steering_to_users
