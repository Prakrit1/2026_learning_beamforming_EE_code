"""
Matched-power MMSE for the target_entropy-annealing checkpoint (job 155683,
2026-07-31): same technique as mmse_matched_power_detlam.py, rescaled per
error level to THIS checkpoint's own measured power (from
rate_power_tanneal.gzip, produced by rate_power_tanneal.py) instead of the
full 75 W budget.

This is the decisive test for whether entropy annealing succeeds where
detlam/EMA+warmstart did not: those checkpoints' matched-power-MMSE gap
stayed in the 43-66% range across two rounds of lambda-mechanism engineering
(2026-07-29/30). If entropy annealing (paired with the documented lwin5000
lambda mode, per docs/EE_formulation.tex Sec 6.3) actually addresses the
root cause, this gap should close substantially at error=0.
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

TANNEAL_TRAINING_NAME = 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_satg30_p75_eta0.6_tanneal1to0_rawpow'

cfg = Config()

rate_power_gzip = Path(cfg.output_metrics_path, 'EE_tanneal_rate_power_sweep', 'rate_power_tanneal.gzip')
with gzip.open(rate_power_gzip, 'rb') as f:
    rate_power_data = pickle.load(f)
if not np.allclose(rate_power_data['error_sweep_range'], error_sweep_range):
    raise ValueError(f'error grid mismatch between {rate_power_gzip} and this sweep')

sac_power_by_error = {
    round(float(err), 6): float(power)
    for err, power in zip(rate_power_data['error_sweep_range'],
                          rate_power_data['results']['tanneal_aod0.05']['mean_power'])
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
config.config_learner.training_name = TANNEAL_TRAINING_NAME
print(f'[tanneal_aod0.05] matched-power MMSE using this checkpoint\'s own measured power per error level')
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
print('\n=== SAC (entropy-anneal) vs. matched-power MMSE (same power, same error level) ===')
sac_rate = rate_power_data['results']['tanneal_aod0.05']['mean_rate']
mmse_gzip = Path(cfg.output_metrics_path, TANNEAL_TRAINING_NAME, 'error_sweep',
                  'testing_mmse_sacpower_sweep_0.0_0.1.gzip')
with gzip.open(mmse_gzip, 'rb') as f:
    errs, results = pickle.load(f)
mmse_rate = results[calc_sum_rate]['mean']
for i, e in enumerate(error_sweep_range):
    gap_pct = 100 * (sac_rate[i] - mmse_rate[i]) / mmse_rate[i]
    print(f'  error={e:.2f}: SAC={sac_rate[i]:.3f}  MMSE(matched power)={mmse_rate[i]:.3f}  '
          f'gap={gap_pct:+.1f}%')

print('\n=== Reference: detlam checkpoints\' matched-power gap at error=0 (from prior sessions) ===')
print('  detlam aod0.0:            -45.3% (pre-ema) / -62.5% (ema+warmstart, WORSE)')
print('  detlam aod0.05:           -66.0% (pre-ema) / -49.3% (ema+warmstart)')
print('  detlam aod0.0, fair1.5:   -46.0% (pre-ema) / -62.9% (ema+warmstart, WORSE)')
print('  detlam aod0.05, fair1.5:  -65.2% (pre-ema) / -43.0% (ema+warmstart)')
print(f'  entropy-anneal aod0.05:   {100 * (sac_rate[0] - mmse_rate[0]) / mmse_rate[0]:+.1f}% (this run, at error=0)')
