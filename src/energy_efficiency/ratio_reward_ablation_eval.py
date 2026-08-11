"""
Evaluate the raw-ratio-reward-vs-Dinkelbach-adaptive-lambda ablation
(2026-08-10 session, commit 0a60e76). Three checkpoints were trained with
EE_REWARD_MODE=energy_efficiency (Scheme I: reward = raw sum_rate/total_power
computed directly every step, no lambda/adaptive pricing at all, precoder
always rescaled to the full power budget -- see EE_sac.py's "Scheme I:
always rescale to the full budget, so power carries no gradient" branch) at
aod=0.0/0.035/0.05, 3GPP Set-1 system params (30 dBi / 75 W), nadir geometry
-- same system as the working-line lwin5000 triplet in rate_power_triplet.py.

This script compares each ratio-reward checkpoint against its matched-error-
bound Dinkelbach-adaptive-lambda counterpart, where one exists:
  - aod=0.0  <-> EE_dinkelbach_adaptive_lwin5000_..._rawpow (job 156358)
  - aod=0.05 <-> EE_dinkelbach_adaptive_aod0.05_lwin5000_..._rawpow (job 153655)
  - aod=0.035 has NO matched lambda checkpoint -- the lambda triplet on disk
    is at aod=0.0/0.025/0.05, not 0.035 (flagged in the 2026-08-10 handoff
    session; don't invent a comparison for it). Evaluated and saved
    standalone, against full-budget MMSE only.

Ratio-reward checkpoints are evaluated with get_precoding_learned (the
always-rescale-to-budget precoder), matching how they were trained -- NOT
get_precoding_learned_clip_only, which is for the lambda checkpoints (see
tx_power_distribution.py's models_to_check table for the same
clip_only=True/False distinction on the older full_EE_aod0.5 checkpoint).

Reuses run_rate_power_sweep / run_matched_power_mmse_sweep / get_best_model_path
from rate_power_triplet.py rather than duplicating them.

Saves: outputs/metrics/EE_ratio_reward_ablation/ratio_reward_ablation.gzip
Plotting: my_plotting_ratio_reward_ablation.py (run after this script).
"""
import os

os.environ.pop('EE_SAT_GAIN_DBI', None)
os.environ.pop('EE_POWER_BUDGET_WATT', None)
os.environ.pop('EE_TARGET_ELEVATION_DEG', None)
os.environ.pop('EE_REWARD_MODE', None)
os.environ.pop('EE_TRAIN_ERROR_BOUND', None)

import gzip
import pickle
from pathlib import Path

from src.config.config import Config
from src.utils.get_precoding import get_precoding_learned, get_precoding_learned_clip_only, get_precoding_mmse
from src.utils.load_model import load_model
from src.energy_efficiency.rate_power_triplet import (
    error_sweep_range,
    get_best_model_path,
    run_rate_power_sweep,
    run_matched_power_mmse_sweep,
)

# aod bound -> training_name. All three trained via
# EE_ratio_reward_aod{bound}_satg30_p75_nadir.slurm (EE_REWARD_MODE=energy_efficiency).
RATIO_REWARD_CHECKPOINTS = {
    'aod0.0': 'full_EE_N16K3_satg30_p75_eta0.6',
    'aod0.035': 'full_EE_aod0.035_N16K3_satg30_p75_eta0.6',
    'aod0.05': 'full_EE_aod0.05_N16K3_satg30_p75_eta0.6',
}

# Matched-error-bound Dinkelbach-adaptive-lambda counterpart, where one
# exists (see rate_power_triplet.py's CHECKPOINTS -- that triplet is at
# aod=0.0/0.025/0.05, so aod0.035 is intentionally absent here).
LAMBDA_CHECKPOINTS = {
    'aod0.0': 'EE_dinkelbach_adaptive_lwin5000_N16K3_satg30_p75_eta0.6_rawpow',
    'aod0.05': 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_satg30_p75_eta0.6_rawpow',
}


if __name__ == '__main__':
    cfg = Config()
    cfg.show_plots = False
    print(f'[system] sat_gain_dBi={cfg.sat_gain_dBi}, budget={cfg.power_constraint_watt} W, '
          f'user_center_aod_earth_deg={cfg.user_center_aod_earth_deg:.2f}')

    out_path = Path(cfg.output_metrics_path, 'EE_ratio_reward_ablation')
    out_path.mkdir(parents=True, exist_ok=True)
    gzip_path = Path(out_path, 'ratio_reward_ablation.gzip')

    results = {}

    # ---- shared full-budget MMSE curve (system-only, same for all corners) ----
    cfg.config_learner.training_name = 'EE_ratio_reward_ablation'
    results['mmse_nadir'] = run_rate_power_sweep(cfg, 'MMSE (3GPP Set-1, nadir, full budget)', get_precoding_mmse)
    results['mmse_nadir']['label'] = 'MMSE (full budget)'

    # ---- ratio-reward (Scheme I) checkpoints: always-rescale evaluation ----
    for aod_key, training_name in RATIO_REWARD_CHECKPOINTS.items():
        cfg.config_learner.training_name = training_name
        model_path = get_best_model_path(cfg.trained_models_path, training_name)
        print(f'[ratio_{aod_key}] checkpoint: {model_path}')

        precoder_network, norm_factors = load_model(model_path)
        if norm_factors != {}:
            cfg.config_learner.get_state_args['norm_state'] = True

        sac_result = run_rate_power_sweep(
            cfg, f'SAC ratio-reward ({aod_key})',
            lambda c, um, sm: get_precoding_learned(c, um, sm, norm_factors, precoder_network),
        )
        sac_result['label'] = f'SAC ratio-reward (err {aod_key.replace("aod", "")})'
        sac_result['training_name'] = training_name
        sac_result['checkpoint'] = str(model_path)

        matched_mmse_result = run_matched_power_mmse_sweep(
            cfg, f'MMSE matched-power (ratio_{aod_key})', sac_result['mean_power'],
        )
        matched_mmse_result['label'] = f'MMSE eq.pow (ratio {aod_key.replace("aod", "")})'

        results[f'sac_ratio_{aod_key}'] = sac_result
        results[f'mmse_matched_ratio_{aod_key}'] = matched_mmse_result

    # ---- matched lambda-triplet checkpoints, for direct comparison ----------
    for aod_key, training_name in LAMBDA_CHECKPOINTS.items():
        cfg.config_learner.training_name = training_name
        model_path = get_best_model_path(cfg.trained_models_path, training_name)
        print(f'[lambda_{aod_key}] checkpoint: {model_path}')

        precoder_network, norm_factors = load_model(model_path)
        if norm_factors != {}:
            cfg.config_learner.get_state_args['norm_state'] = True

        sac_result = run_rate_power_sweep(
            cfg, f'SAC lambda ({aod_key})',
            lambda c, um, sm: get_precoding_learned_clip_only(c, um, sm, norm_factors, precoder_network),
        )
        sac_result['label'] = f'SAC lambda (err {aod_key.replace("aod", "")})'
        sac_result['training_name'] = training_name
        sac_result['checkpoint'] = str(model_path)

        results[f'sac_lambda_{aod_key}'] = sac_result

    with gzip.open(gzip_path, 'wb') as file:
        pickle.dump({'error_sweep_range': error_sweep_range, 'results': results}, file=file)
    print(f'Saved: {gzip_path}')
