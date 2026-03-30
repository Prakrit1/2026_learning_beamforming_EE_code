
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
from src.utils.calc_channel_correlation import calc_channel_correlation
from src.utils.format_value import format_value
from src.utils.progress_printer import progress_printer
from src.utils.spherical_to_cartesian_coordinates import spherical_to_cartesian_coordinates

def get_pairwise_user_distances_km(user_manager):
    positions = [u.cartesian_coordinates for u in user_manager.users]
    positions = np.stack(positions, axis=0)
    n = positions.shape[0]
    dists = []
    for i in range(n):
        for j in range(i+1, n):
            chord = np.linalg.norm(positions[i] - positions[j])
            dists.append(chord / 1000.0)
    return np.array(dists)

def test_pairwise_user_distance_histogram(monte_carlo_iterations: int = 10000):
    config = Config()
    satellite_manager = SatelliteManager(config)
    user_manager = UserManager(config)

    all_pairwise_dists_km = []

    for it in range(monte_carlo_iterations):
        # neue Konstellation erzeugen (inkl. update_positions)
        update_sim(config, satellite_manager, user_manager)

        dists_km = get_pairwise_user_distances_km(user_manager)
        all_pairwise_dists_km.extend(dists_km)

        if (it + 1) % (monte_carlo_iterations // 10) == 0:
            print(f"{it + 1}/{monte_carlo_iterations} samples ...")

    all_pairwise_dists_km = np.array(all_pairwise_dists_km)

    max_dist = 50  # km
    mask = all_pairwise_dists_km <= max_dist
    all_pairwise_dists_km = all_pairwise_dists_km[mask]

    # Histogramm plotten
    plt.figure(figsize=(6, 4))
    plt.hist(all_pairwise_dists_km, bins=50, alpha=0.7, color='C0')
    plt.xlabel('Pairwise user distance [km]')
    plt.ylabel('Count')
    plt.title('Histogram of pairwise user distances')
    plt.tight_layout()
    plt.show()

    return all_pairwise_dists_km


if __name__ == "__main__":
    dists = test_pairwise_user_distance_histogram(monte_carlo_iterations=200000)