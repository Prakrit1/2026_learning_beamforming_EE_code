from datetime import datetime
from pathlib import Path
import gzip
import pickle

import numpy as np
from matplotlib.pyplot import show as plt_show

import src
from src.data.satellite_manager import (
    SatelliteManager,
)
from src.data.user_manager import (
    UserManager,
)
from src.utils.plot_sweep import (
    plot_sweep,
)
from src.utils.profiling import (
    start_profiling,
    end_profiling,
)
from src.utils.progress_printer import (
    progress_printer,
)
from src.utils.update_sim import (
    update_sim,
)


def test_precoder_tx_power_distribution(
    config: 'src.config.config.Config',
    distance_sweep_range,
    precoder_name: str,
    monte_carlo_iterations: int,
    mode: str,
    get_precoder_func,
    calc_reward_func,
) -> dict:
    """
    Calculate the sum rates that a given precoder achieves for a given config
    over a given range of inter-user-distances or inter-satellite-distances with no channel error
    mode: ['user', 'satellite']
    """

    def progress_print() -> None:
        progress = (distance_sweep_idx + 1) / (len(distance_sweep_range))
        progress_printer(progress=progress, real_time_start=real_time_start)

    def save_results():
        name = f'testing_{precoder_name}_{mode}sweep_tx_{round(distance_sweep_range[0])}_{round(distance_sweep_range[-1])}.gzip'
        results_path = Path(config.output_metrics_path, config.config_learner.training_name, 'distance_sweep')
        results_path.mkdir(parents=True, exist_ok=True)
        with gzip.open(Path(results_path, name), 'wb') as file:
            pickle.dump([distance_sweep_range, metrics], file=file)

    satellite_manager = SatelliteManager(config=config)
    user_manager = UserManager(config=config)

    real_time_start = datetime.now()

    profiler = None
    if config.profile:
        profiler = start_profiling()

    metrics = {
        'tx_power_per_user': {
            'mean': np.zeros((len(distance_sweep_range), config.user_nr)),
            'std': np.zeros((len(distance_sweep_range), config.user_nr)),
        }
    }

    for distance_sweep_idx, distance_sweep_value in enumerate(distance_sweep_range):

        metrics_per_monte_carlo = np.zeros((monte_carlo_iterations, config.user_nr))

        if mode == 'user':
            config.user_distribution_mode = 'uniform'
            config.user_dist_average = distance_sweep_value
            config.user_dist_bound = 0
        elif mode == 'satellite':
            config.sat_dist_average = distance_sweep_value
            config.sat_dist_bound = 0

        for iter_idx in range(monte_carlo_iterations):

            update_sim(config, satellite_manager, user_manager)

            w_precoder = get_precoder_func(
                config,
                user_manager,
                satellite_manager,
            )

            metrics_per_monte_carlo[iter_idx, :] = calc_reward_func(
                w_precoder=w_precoder,
            )

            if config.verbosity > 0:
                if iter_idx % 50 == 0:
                    progress_print()

        m = metrics['tx_power_per_user']
        m['mean'][distance_sweep_idx, :] = np.mean(metrics_per_monte_carlo, axis=0)
        m['std'][distance_sweep_idx, :] = np.std(metrics_per_monte_carlo, axis=0)

    if profiler is not None:
        end_profiling(profiler)

    if config.verbosity > 0:
        print()
        m = metrics['tx_power_per_user']
        for d, mean_vec, std_vec in zip(distance_sweep_range, m['mean'], m['std']):
            print(f'{d:.2f}: tx_power_per_user sum={mean_vec.sum():.4f} (std sum={std_vec.sum():.4f})')

    save_results()

    if config.show_plots:

        import matplotlib.pyplot as plt

        m = metrics['tx_power_per_user']
        dist_idx = 0  # weil nur ein distance sweep wert

        powers = m['mean'][dist_idx, :]  # shape: (user_nr,)
        x = np.array([0.0])

        bottom = 0.0
        fig, ax = plt.subplots()

        for u in range(config.user_nr):
            ax.bar(
                x,
                powers[u],
                bottom=bottom,
                label=f'User {u}',
            )
            bottom += powers[u]

        ax.set_xticks(x)
        ax.set_xticklabels([f'd={distance_sweep_range[dist_idx]:.2f}'])
        ax.set_ylabel('Transmit power')
        ax.set_title(f'{precoder_name} power distribution (stacked)')
        ax.legend()

        plt.show()

    return metrics
