from datetime import datetime
from pathlib import Path
import gzip
import pickle

import numpy as np
import matplotlib.pyplot as plt

from src.config.config import Config
from src.data.satellite_manager import SatelliteManager
from src.data.user_manager import UserManager
from src.utils.update_sim import update_sim
from src.utils.format_value import format_value
from src.utils.progress_printer import progress_printer
from src.utils.neumann_matrix_inversion_approximation import (
    neumann_matrix_inversion_approximation,
)


def get_regularization_factor(
    config: Config,
    precoder_type: str,
    inversion_constant_lambda: float = 1e-18,
) -> float:
    """
    Compute the regularization factor based on the selected precoder type.

    ZF   -> 0 * mmse_scale + inversion_constant_lambda
    MMSE -> 1 * mmse_scale + inversion_constant_lambda
    MRT  -> 10 * mmse_scale + inversion_constant_lambda
    """
    mmse_scale = config.noise_power_watt * (
        config.user_nr / config.power_constraint_watt
    )

    precoder_type = precoder_type.upper()

    if precoder_type == 'ZF':
        regularization_factor = 0.0 * mmse_scale
    elif precoder_type == 'MMSE':
        regularization_factor = 1.0 * mmse_scale
    elif precoder_type == 'MRT':
        regularization_factor = 10.0 * mmse_scale
    else:
        raise ValueError(
            f"Unknown precoder_type '{precoder_type}'. "
            f"Choose one of: 'ZF', 'MMSE', 'MRT'."
        )

    regularization_factor += inversion_constant_lambda
    return regularization_factor


def test_neumann_inversion_order_sweep(
    order_sweep_range: np.ndarray,
    monte_carlo_iterations: int,
    precoder_type: str,
    inversion_constant_lambda: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sweep over the Neumann approximation order and evaluate the relative residual

        ||I - A A_approx^{-1}||_F / ||I||_F

    where

        A = H^H H + lambda I

    and lambda depends on the chosen precoder type.

    Parameters
    ----------
    order_sweep_range : np.ndarray
        Array of Neumann approximation orders to test.
    monte_carlo_iterations : int
        Number of Monte-Carlo realizations per order.
    precoder_type : str
        One of 'ZF', 'MMSE', 'MRT'.
    inversion_constant_lambda : float
        Constant regularization added in all cases.

    Returns
    -------
    mean_errors : np.ndarray
        Mean relative residual per order.
    std_errors : np.ndarray
        Standard deviation of the relative residual per order.
    """

    config = Config()
    satellite_manager = SatelliteManager(config)
    user_manager = UserManager(config)

    update_sim(config, satellite_manager, user_manager)

    regularization_factor = get_regularization_factor(
        config=config,
        precoder_type=precoder_type,
        inversion_constant_lambda=inversion_constant_lambda,
    )

    mean_errors = np.zeros(len(order_sweep_range))
    std_errors = np.zeros(len(order_sweep_range))

    start = datetime.now()

    for order_idx, order in enumerate(order_sweep_range):
        order_errors = np.zeros(monte_carlo_iterations)

        for iteration in range(monte_carlo_iterations):
            update_sim(config, satellite_manager, user_manager)

            channel_matrix = satellite_manager.erroneous_channel_state_information
            channel_matrix = channel_matrix
            sat_tot_ant_nr = channel_matrix.shape[1]

            matrix = (
                channel_matrix.conj().T @ channel_matrix
                + regularization_factor * np.eye(
                    sat_tot_ant_nr,
                    dtype=channel_matrix.dtype,
                )
            )

            inv_matrix_perfect = np.linalg.inv(matrix) #/ np.linalg.norm(matrix, 'fro')
            # matrix = matrix/ np.linalg.norm(matrix, 'fro')

            # matrix = np.eye(matrix.shape[0], dtype=matrix.dtype)

            inv_matrix_estimation = neumann_matrix_inversion_approximation(
                matrix=matrix,
                order=int(order),
            )  

            identity = np.eye(matrix.shape[0], dtype=matrix.dtype)
            residual = identity - matrix @ inv_matrix_estimation
            relative_residual = (
                np.linalg.norm(residual, ord='fro')
                / np.linalg.norm(identity, ord='fro')
            )
            # diff = (inv_matrix_estimation - inv_matrix_perfect)
            #
            #
            # rmse = np.sqrt(np.mean(np.abs(diff) ** 2))

            order_errors[iteration] = relative_residual
            # order_errors[iteration] = rmse

        mean_errors[order_idx] = np.mean(order_errors)
        std_errors[order_idx] = np.std(order_errors)

        progress_printer(
            progress=(order_idx + 1) / len(order_sweep_range),
            real_time_start=start,
        )

    return mean_errors, std_errors


if __name__ == '__main__':
    config = Config()

    order_sweep_range = np.arange(1, 21, 1)
    monte_carlo_iterations = 1000

    precoder_type = 'MMSE'   # choose from 'ZF', 'MMSE', 'MRT'
    inversion_constant_lambda = 0#1e-15

    regularization_factor = get_regularization_factor(
        config=config,
        precoder_type=precoder_type,
        inversion_constant_lambda=inversion_constant_lambda,
    )

    mean_errors, std_errors = test_neumann_inversion_order_sweep(
        order_sweep_range=order_sweep_range,
        monte_carlo_iterations=monte_carlo_iterations,
        precoder_type=precoder_type,
        inversion_constant_lambda=inversion_constant_lambda,
    )

    # save
    results_path = Path(config.output_metrics_path, 'inversion_order')
    results_path.mkdir(parents=True, exist_ok=True)

    name = (
        f'{config.sat_nr}sat_'
        f'{config.sat_tot_ant_nr}ant_'
        f'{precoder_type}_'
        f'orders_{format_value(order_sweep_range[0])}-{format_value(order_sweep_range[-1])}_'
        f'mc_{monte_carlo_iterations}_'
        f'lambda_{format_value(regularization_factor)}.gzip'
    )

    with gzip.open(Path(results_path, name), 'wb') as file:
        pickle.dump(
            {
                'order_sweep_range': order_sweep_range,
                'mean_errors': mean_errors,
                'std_errors': std_errors,
                'monte_carlo_iterations': monte_carlo_iterations,
                'precoder_type': precoder_type,
                'inversion_constant_lambda': inversion_constant_lambda,
                'regularization_factor': regularization_factor,
            },
            file=file,
        )

    # plot
    fig, ax = plt.subplots()

    ax.plot(
        order_sweep_range,
        mean_errors,
        marker='o',
        color= 'black',
        label='Mean relative residual',
    )
    ax.fill_between(
        order_sweep_range,
        mean_errors - std_errors,
        mean_errors + std_errors,
        color='black',
        alpha=0.2,
        label='±1 std',
    )

    ax.set_xlabel('Neumann Approximation Order')
    ax.set_ylabel(r'Relative residual $\|I - A\hat{A}^{-1}\|_F / \|I\|_F$')
    # ax.set_title(
    #     f'Neumann inversion error vs. order ({precoder_type}, '
    #     f'lambda={regularization_factor:.3e})'
    # )
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    plt.show()