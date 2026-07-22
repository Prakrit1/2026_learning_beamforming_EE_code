"""
Pairwise channel correlation between our K=3 users, under the CURRENT
paper-matching scenario (config.py defaults: N=16 antennas, 100km mean user
distance, +-50km roam), at zero CSIT error (true channel, not the erroneous
estimate -- satellite_manager.channel_state_information, not
erroneous_channel_state_information).

See calc_channel_correlation.py: |correlation|=1 means two users' channels
are spatially indistinguishable (no precoder can separate them without
interference); |correlation|=0 means perfectly separable. This measures
whether high user-channel correlation is a PHYSICAL property of this
scenario, independent of anything about SAC/RL training -- if correlation
is high, that's a real difficulty for ANY precoder (MMSE, ZF, SAC, whatever),
not something a training fix could resolve.

No distance sweep (unlike the existing test_channel_correlation_distance_sweep.py)
-- just a direct Monte Carlo measurement at our actual, current scenario.
"""
from pathlib import Path
import gzip
import pickle

import numpy as np

from src.config.config import Config
from src.data.satellite_manager import SatelliteManager
from src.data.user_manager import UserManager
from src.utils.update_sim import update_sim
from src.utils.calc_channel_correlation import calc_channel_correlation

MONTE_CARLO_ITERATIONS = 10000

if __name__ == '__main__':
    cfg = Config()
    cfg.show_plots = False
    print(f'Scenario: N={cfg.sat_tot_ant_nr} antennas, K={cfg.user_nr} users, '
          f'user_dist_average={cfg.user_dist_average} m, user_dist_bound={cfg.user_dist_bound} '
          f'(roam +-{cfg.user_dist_average * cfg.user_dist_bound:.0f} m)')

    satellite_manager = SatelliteManager(config=cfg)
    user_manager = UserManager(config=cfg)

    user_pairs = [(i, j) for i in range(cfg.user_nr) for j in range(i + 1, cfg.user_nr)]
    correlations = {pair: np.zeros(MONTE_CARLO_ITERATIONS) for pair in user_pairs}

    for iter_idx in range(MONTE_CARLO_ITERATIONS):
        update_sim(cfg, satellite_manager, user_manager)
        # zero CSIT error -- true channel, not erroneous_channel_state_information
        channel = satellite_manager.channel_state_information
        for (i, j) in user_pairs:
            correlations[(i, j)][iter_idx] = abs(calc_channel_correlation(
                channel_1=channel[i, :],
                channel_2=channel[j, :],
            ))
        if iter_idx % 2000 == 0:
            print(f'{iter_idx}/{MONTE_CARLO_ITERATIONS}')

    print('\n' + '=' * 60)
    print('PAIRWISE CHANNEL CORRELATION (zero CSIT error, |correlation|)')
    print('=' * 60)
    all_values = []
    for (i, j), values in correlations.items():
        print(f'User {i+1} <-> User {j+1}: mean={values.mean():.4f}, std={values.std():.4f}, '
              f'min={values.min():.4f}, max={values.max():.4f}')
        all_values.append(values)
    overall = np.concatenate(all_values)
    print(f'\nOverall (all pairs pooled): mean={overall.mean():.4f}, std={overall.std():.4f}')
    print('=' * 60)

    results_path = Path(cfg.output_metrics_path, 'channel_correlation', 'N16K3_zero_error')
    results_path.mkdir(parents=True, exist_ok=True)
    with gzip.open(Path(results_path, 'pairwise_correlation.gzip'), 'wb') as file:
        pickle.dump({'correlations': correlations, 'overall_mean': overall.mean()}, file=file)
    print(f"Saved: {Path(results_path, 'pairwise_correlation.gzip')}")
