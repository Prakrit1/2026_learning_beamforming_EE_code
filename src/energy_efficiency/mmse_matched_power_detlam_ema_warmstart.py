"""
Matched-power MMSE for the four ema0.3+warmstart detlam retrains
(2026-07-30 session, jobs 155291-94): same technique as
mmse_matched_power_detlam.py, but rescaled per error level to each new
checkpoint's own measured power (from rate_power_detlam_ema_warmstart.gzip,
produced by rate_power_detlam_ema_warmstart.py).

Purpose: this is the direct test of whether ema+warmstart fixed the
2026-07-30 finding that pre-ema detlam checkpoints underperformed
matched-power MMSE by 45-66% at error=0. If the gap shrinks substantially
here, the fix worked; if it doesn't, the undertraining wasn't the (whole)
story.
"""
import gzip
import pickle
from pathlib import Path

import numpy as np

from src.config.config import Config
from src.data.calc_sum_rate import calc_sum_rate
from src.utils.get_precoding import get_precoding_mmse
from src.analysis.helpers.test_precoder_error_sweep import test_precoder_error_sweep

error_sweep_range = np.linspace(0, 0.10, 11)
monte_carlo_iterations = 10000

CHECKPOINTS = [
    'ema_ws_aod0.0',
    'ema_ws_aod0.05',
    'ema_ws_aod0.0_fair1.5',
    'ema_ws_aod0.05_fair1.5',
]

TRAINING_NAMES = {
    'ema_ws_aod0.0': 'EE_dinkelbach_adaptive_detlam512_ema0.3_N16K3_satg30_p75_eta0.6_rawpow_warmstart',
    'ema_ws_aod0.05': 'EE_dinkelbach_adaptive_aod0.05_detlam512_ema0.3_N16K3_satg30_p75_eta0.6_rawpow_warmstart',
    'ema_ws_aod0.0_fair1.5': 'EE_dinkelbach_adaptive_detlam512_ema0.3_N16K3_satg30_p75_eta0.6_rawpow_fair1.5_warmstart',
    'ema_ws_aod0.05_fair1.5': 'EE_dinkelbach_adaptive_aod0.05_detlam512_ema0.3_N16K3_satg30_p75_eta0.6_rawpow_fair1.5_warmstart',
}

cfg = Config()

rate_power_gzip = Path(cfg.output_metrics_path, 'EE_detlam_ema_warmstart_rate_power_sweep', 'rate_power_detlam_ema_warmstart.gzip')
with gzip.open(rate_power_gzip, 'rb') as f:
    rate_power_data = pickle.load(f)
if not np.allclose(rate_power_data['error_sweep_range'], error_sweep_range):
    raise ValueError(f'error grid mismatch between {rate_power_gzip} and this sweep')

matched_results = {}

for key in CHECKPOINTS:
    sac_power_by_error = {
        round(float(err), 6): float(power)
        for err, power in zip(rate_power_data['error_sweep_range'],
                              rate_power_data['results'][key]['mean_power'])
    }

    def make_precoder(power_by_error):
        def get_precoding_mmse_sac_power(cfg_, user_manager, satellite_manager):
            w_mmse = get_precoding_mmse(cfg_, user_manager, satellite_manager)
            current_error = cfg_.config_error_model.error_rng_parametrizations[
                'additive_error_on_cosine_of_aod']['args']['high']
            target_power = power_by_error[round(float(current_error), 6)]
            current_power = np.real(np.trace(np.matmul(w_mmse.conj().T, w_mmse)))
            return w_mmse * np.sqrt(target_power / current_power)
        return get_precoding_mmse_sac_power

    config = Config()
    config.config_learner.training_name = TRAINING_NAMES[key]
    print(f'[{key}] matched-power MMSE using this checkpoint\'s own measured power per error level')
    metrics = test_precoder_error_sweep(
        config=config,
        error_sweep_parameter='additive_error_on_cosine_of_aod',
        error_sweep_range=error_sweep_range,
        precoder_name='mmse_sacpower',
        monte_carlo_iterations=monte_carlo_iterations,
        get_precoder_func=make_precoder(sac_power_by_error),
        calc_reward_funcs=[calc_sum_rate],
    )

# ---- comparison printout -----------------------------------------------
print('\n=== SAC (ema+ws) vs. matched-power MMSE (same power, same error level) ===')
for key in CHECKPOINTS:
    sac_rate = rate_power_data['results'][key]['mean_rate']
    mmse_gzip = Path(cfg.output_metrics_path, TRAINING_NAMES[key], 'error_sweep',
                      'testing_mmse_sacpower_sweep_0.0_0.1.gzip')
    with gzip.open(mmse_gzip, 'rb') as f:
        errs, results = pickle.load(f)
    mmse_rate = results[calc_sum_rate]['mean']
    print(f'\n[{key}]')
    for i, e in enumerate(error_sweep_range):
        gap_pct = 100 * (sac_rate[i] - mmse_rate[i]) / mmse_rate[i]
        print(f'  error={e:.2f}: SAC={sac_rate[i]:.3f}  MMSE(matched power)={mmse_rate[i]:.3f}  '
              f'gap={gap_pct:+.1f}%')
