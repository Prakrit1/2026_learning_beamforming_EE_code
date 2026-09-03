"""Per-sample power (and rate) rollout for the rate-only SAC baseline.

Companion to power_savings_bars_triplet.py's EE-checkpoint rollout, but for
the SAC_rateonly baseline (EE_REWARD_MODE=sum_rate_only -- see
SAC_rateonly_satg30_p75_nadir.slurm and EE_sac.py's reward_mode branch)
instead. Uses the same get_precoding_learned_clip_only projection at
evaluation time as the EE checkpoint (see deployed_power_histogram_sac.py),
so the two are directly comparable -- per Lemma 1 (sum rate monotonic in
power), a rate-only-trained policy has no incentive to hold back, so under
clip-only projection its own raw output is expected to land at (or very
near) the 75W budget on essentially every realization, unlike the
EE-trained policy's ~35W cluster.

Saves BOTH per-sample total radiated power and per-sample sum rate (not
just their means), so their distributions can be histogrammed directly.

Saves outputs/metrics/EE_lwin5000_3gpp_triplet/power_samples_rateonly.gzip
"""
import gzip
import pickle
from pathlib import Path

import numpy as np

from src.config.config import Config
from src.data.calc_sum_rate import calc_sum_rate
from src.data.calc_tx_power_distribution import calc_tx_power_distribution
from src.data.satellite_manager import SatelliteManager
from src.data.user_manager import UserManager
from src.utils.get_precoding import get_precoding_learned_clip_only
from src.utils.load_model import load_model
from src.utils.update_sim import update_sim
from src.energy_efficiency.plotting_scenario import get_best_model_path

# matches EE_sac.py's reward_mode == 'sum_rate_only' training_name construction
TRAINING_NAME = 'SAC_rateonly_N16K3_satg30_p75_eta0.6_rawpow'
# matches power_savings_bars_triplet.py's eval_error_bound, for direct comparability
CSIT_ERROR_BOUND = 0.0
monte_carlo_iterations = 10000


if __name__ == '__main__':
    cfg = Config()
    print(f'[system] sat_gain_dBi={cfg.sat_gain_dBi}, budget={cfg.power_constraint_watt} W, '
          f'eval error bound={CSIT_ERROR_BOUND}')

    cfg.config_learner.training_name = TRAINING_NAME
    model_path = get_best_model_path(cfg.trained_models_path, TRAINING_NAME)
    print(f'[power_samples_rateonly] checkpoint: {model_path}')

    precoder_network, norm_factors = load_model(model_path)
    if norm_factors != {}:
        cfg.config_learner.get_state_args['norm_state'] = True

    satellite_manager = SatelliteManager(config=cfg)
    user_manager = UserManager(config=cfg)

    error_param = 'additive_error_on_cosine_of_aod'
    cfg.config_error_model.error_rng_parametrizations[error_param]['args']['low'] = -CSIT_ERROR_BOUND
    cfg.config_error_model.error_rng_parametrizations[error_param]['args']['high'] = CSIT_ERROR_BOUND

    power_samples = np.zeros((monte_carlo_iterations, cfg.user_nr))
    rate_samples = np.zeros(monte_carlo_iterations)
    for iter_idx in range(monte_carlo_iterations):
        update_sim(cfg, satellite_manager, user_manager)
        w_precoder = get_precoding_learned_clip_only(cfg, user_manager, satellite_manager, norm_factors, precoder_network)
        power_samples[iter_idx, :] = calc_tx_power_distribution(w_precoder=w_precoder)
        rate_samples[iter_idx] = calc_sum_rate(
            channel_state=satellite_manager.channel_state_information,
            w_precoder=w_precoder,
            noise_power_watt=cfg.noise_power_watt,
        )
        if iter_idx % 1000 == 0:
            print(f'{iter_idx}/{monte_carlo_iterations}')

    total_power = power_samples.sum(axis=1)
    print(f'mean total power: {total_power.mean():.2f} W '
          f'({100 * total_power.mean() / cfg.power_constraint_watt:.1f}% of budget)')
    print(f'mean rate: {rate_samples.mean():.4f} bps/Hz')

    out_path = Path(cfg.output_metrics_path, 'EE_lwin5000_3gpp_triplet')
    out_path.mkdir(parents=True, exist_ok=True)
    gzip_path = Path(out_path, 'power_samples_rateonly.gzip')
    with gzip.open(gzip_path, 'wb') as file:
        pickle.dump({
            'power_samples': power_samples,
            'rate_samples': rate_samples,
            'power_budget': cfg.power_constraint_watt,
            'training_name': TRAINING_NAME,
            'checkpoint': str(model_path),
            'csit_error_bound': CSIT_ERROR_BOUND,
        }, file=file)
    print(f'Saved: {gzip_path}')
