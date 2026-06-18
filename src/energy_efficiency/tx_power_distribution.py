"""
Histogram of per-user transmit power allocation for the SAC energy-efficiency
precoder, run for both the full_EE and simplified_EE models.
"""

from datetime import datetime
from pathlib import Path
import gzip
import pickle

import numpy as np
import matplotlib.pyplot as plt

import src
from src.data.satellite_manager import SatelliteManager
from src.data.user_manager import UserManager
from src.utils.progress_printer import progress_printer
from src.utils.update_sim import update_sim


def test_precoder_tx_power_histogram(
    config: 'src.config.config.Config',
    distance_value: float,
    precoder_name: str,
    monte_carlo_iterations: int,
    mode: str,
    get_precoder_func,
    calc_reward_func,
) -> np.ndarray:
    """
    Run Monte Carlo iterations at a single user/satellite distance and
    return raw per-user transmit power samples for histogramming.

    Returns
    -------
    samples : np.ndarray, shape (monte_carlo_iterations, config.user_nr)
    """

    def progress_print() -> None:
        progress = (iter_idx + 1) / monte_carlo_iterations
        progress_printer(progress=progress, real_time_start=real_time_start)

    def save_results():
        name = f'tx_power_hist_{precoder_name}_{mode}_{round(distance_value)}.gzip'
        results_path = Path(config.output_metrics_path, config.config_learner.training_name, 'tx_power_histogram')
        results_path.mkdir(parents=True, exist_ok=True)
        with gzip.open(Path(results_path, name), 'wb') as file:
            pickle.dump(samples, file=file)
        print(f'Saved: {Path(results_path, name)}')

    satellite_manager = SatelliteManager(config=config)
    user_manager = UserManager(config=config)

    real_time_start = datetime.now()

    if mode == 'user':
        config.user_distribution_mode = 'uniform'
        config.user_dist_average = distance_value
        config.user_dist_bound = 0
    elif mode == 'satellite':
        config.sat_dist_average = distance_value
        config.sat_dist_bound = 0

    samples = np.zeros((monte_carlo_iterations, config.user_nr))

    for iter_idx in range(monte_carlo_iterations):

        update_sim(config, satellite_manager, user_manager)

        w_precoder = get_precoder_func(
            config,
            user_manager,
            satellite_manager,
        )

        samples[iter_idx, :] = calc_reward_func(w_precoder=w_precoder)

        if config.verbosity > 0 and iter_idx % 50 == 0:
            progress_print()

    # Sanity check: total power across users should respect the power budget (e.g. 100 W)
    total_power = samples.sum(axis=1)
    print(f'[{precoder_name}] Total power: mean={total_power.mean():.2f} W, '
          f'max={total_power.max():.2f} W, min={total_power.min():.2f} W')

    for u in range(config.user_nr):
        print(f'[{precoder_name}] User {u}: mean={samples[:, u].mean():.2f} W, '
              f'max={samples[:, u].max():.2f} W, '
              f'min={samples[:, u].min():.2f} W, '
              f'std={samples[:, u].std():.2f} W')

    save_results()

    # ---- plot histogram, one subplot per user ----
    fig, axes = plt.subplots(1, config.user_nr, figsize=(4 * config.user_nr, 4), sharey=True)
    if config.user_nr == 1:
        axes = [axes]

    for u in range(config.user_nr):
        axes[u].hist(samples[:, u], bins=50, alpha=0.7, color=f'C{u}')
        axes[u].set_title(f'User {u}')
        axes[u].set_xlabel('Transmit power [W]')
        if u == 0:
            axes[u].set_ylabel('Count')

    fig.suptitle(f'{precoder_name} per-user power allocation (d={distance_value:.0f})')
    plt.tight_layout()

    # Save figure as PDF in the same plots folder as previous outputs
    from src.config.config_plotting import PlotConfig
    plot_cfg = PlotConfig()
    pdf_path = Path(plot_cfg.plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)
    pdf_name = f'tx_power_hist_{precoder_name}_{mode}_{round(distance_value)}.pdf'
    fig.savefig(Path(pdf_path, pdf_name), bbox_inches='tight', dpi=800, transparent=True)
    print(f'Saved: {Path(pdf_path, pdf_name)}')

    plt.show()

    return samples


def get_best_model_path(trained_models_path, training_name):
    base_path = Path(trained_models_path, training_name, 'base')
    checkpoints = [p for p in base_path.iterdir() if p.is_dir() and 'full_snap' in p.name]
    best = sorted(checkpoints, key=lambda p: float(p.name.split('_')[-1]))[-1]
    return best


if __name__ == '__main__':
    from src.config.config import Config
    from src.utils.get_precoding import get_precoding_learned
    from src.utils.load_model import load_model
    from src.data.calc_tx_power_distribution import calc_tx_power_distribution

    all_results = {}

    for training_name, precoder_label in [('full_EE', 'sac_full_ee'), ('simplified_EE', 'sac_simplified_ee')]:

        cfg = Config()
        cfg.show_plots = False
        cfg.config_learner.training_name = training_name

        model_path = get_best_model_path(cfg.trained_models_path, training_name)
        precoder_network, norm_factors = load_model(model_path)

        if norm_factors != {}:
            cfg.config_learner.get_state_args['norm_state'] = True

        print(f'\n=== Running {precoder_label} ({training_name}) ===')
        samples = test_precoder_tx_power_histogram(
            config=cfg,
            distance_value=25000,
            precoder_name=precoder_label,
            monte_carlo_iterations=10000,
            mode='user',
            get_precoder_func=lambda c, u, s, net=precoder_network, nf=norm_factors: get_precoding_learned(c, u, s, nf, net),
            calc_reward_func=lambda w_precoder: calc_tx_power_distribution(w_precoder=w_precoder),
        )

        all_results[precoder_label] = samples
