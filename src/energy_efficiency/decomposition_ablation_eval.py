"""
Evaluate the input/output decomposition ablation (2026-08-06 session,
commit 06be02a) against the reused baseline. 04_approach.tex claims
"magnitude-phase decomposition works best for the network input and
real-imaginary composition works best for the output, though this is
still under investigation" -- this script tests both halves of that claim
by comparing three checkpoints, all identical except for one env var at
training time (see the two ablation .slurm headers for exact settings):

  - baseline:         EE_CSI_FORMAT=rad_phase (default), EE_ACTION_FORMAT=real_imag (default)
                       -- job 1175, the paper's claimed-best combination, reused unchanged.
  - csi_real_imag:     EE_CSI_FORMAT=real_imag -- tests the input-side half of the claim.
  - action_rad_phase:  EE_ACTION_FORMAT=rad_phase -- tests the output-side half.

Per explicit user policy (handoff 2026-08-06): if neither ablation beats
the baseline, keep the baseline as the working line -- don't adopt either
alternative.

Reuses run_rate_power_sweep / run_matched_power_mmse_sweep / get_best_model_path
from rate_power_triplet.py rather than duplicating them.

Saves: outputs/metrics/EE_decomposition_ablation/decomposition_ablation.gzip
Plotting: my_plotting_decomposition_ablation.py (run after this script).
"""
import os

os.environ.pop('EE_SAT_GAIN_DBI', None)
os.environ.pop('EE_POWER_BUDGET_WATT', None)
os.environ.pop('EE_TARGET_ELEVATION_DEG', None)

import gzip
import pickle
from pathlib import Path

from src.config.config import Config
from src.utils.get_precoding import get_precoding_learned_clip_only, get_precoding_mmse
from src.utils.load_model import load_model
from src.energy_efficiency.rate_power_triplet import (
    error_sweep_range,
    get_best_model_path,
    run_rate_power_sweep,
    run_matched_power_mmse_sweep,
)

# ablation key -> training_name. All three share job 1175's exact settings
# (aod=0.0, lambda-init) except for the one env var noted above.
CHECKPOINTS = {
    'baseline': 'EE_dinkelbach_adaptive_lwin5000_lambdainit12.2-75_N16K3_satg30_p75_eta0.6_rawpow',
    'csi_real_imag': 'EE_dinkelbach_adaptive_lwin5000_lambdainit12.2-75_N16K3_satg30_p75_eta0.6_rawpow_csireal_imag',
    'action_rad_phase': 'EE_dinkelbach_adaptive_lwin5000_lambdainit12.2-75_N16K3_satg30_p75_eta0.6_rawpow_actionrad_phase',
}

# state/action decomposition format each checkpoint was actually TRAINED
# with -- evaluation must match, or the network is fed a state/action
# vector shape or convention it never saw. Omitted key = repo default
# (csi_format='rad_phase', action_format='real_imag').
FORMAT_OVERRIDES = {
    'csi_real_imag': {'csi_format': 'real_imag'},
    'action_rad_phase': {'action_format': 'rad_phase'},
}

LABELS = {
    'baseline': 'SAC (rad_phase in / real_imag out)',
    'csi_real_imag': 'SAC (real_imag in / real_imag out)',
    'action_rad_phase': 'SAC (rad_phase in / rad_phase out)',
}


if __name__ == '__main__':
    cfg = Config()
    cfg.show_plots = False
    print(f'[system] sat_gain_dBi={cfg.sat_gain_dBi}, budget={cfg.power_constraint_watt} W, '
          f'user_center_aod_earth_deg={cfg.user_center_aod_earth_deg:.2f}')

    out_path = Path(cfg.output_metrics_path, 'EE_decomposition_ablation')
    out_path.mkdir(parents=True, exist_ok=True)
    gzip_path = Path(out_path, 'decomposition_ablation.gzip')

    results = {}
    default_csi_format = cfg.config_learner.get_state_args['csi_format']
    default_action_format = cfg.learned_precoder_args['action_format']

    # ---- shared full-budget MMSE curve (system-only, same for all corners) ----
    cfg.config_learner.training_name = 'EE_decomposition_ablation'
    results['mmse_nadir'] = run_rate_power_sweep(cfg, 'MMSE (3GPP Set-1, nadir, full budget)', get_precoding_mmse)
    results['mmse_nadir']['label'] = 'MMSE (full budget)'

    # ---- per-corner SAC + matched-power MMSE -------------------------------
    for key, training_name in CHECKPOINTS.items():
        overrides = FORMAT_OVERRIDES.get(key, {})
        cfg.config_learner.get_state_args['csi_format'] = overrides.get('csi_format', default_csi_format)
        cfg.learned_precoder_args['action_format'] = overrides.get('action_format', default_action_format)

        cfg.config_learner.training_name = training_name
        model_path = get_best_model_path(cfg.trained_models_path, training_name)
        print(f'[{key}] checkpoint: {model_path}')

        precoder_network, norm_factors = load_model(model_path)
        if norm_factors != {}:
            cfg.config_learner.get_state_args['norm_state'] = True

        sac_result = run_rate_power_sweep(
            cfg, f'SAC ({key})',
            lambda c, um, sm: get_precoding_learned_clip_only(c, um, sm, norm_factors, precoder_network),
        )
        sac_result['label'] = LABELS[key]
        sac_result['training_name'] = training_name
        sac_result['checkpoint'] = str(model_path)

        matched_mmse_result = run_matched_power_mmse_sweep(
            cfg, f'MMSE matched-power ({key})', sac_result['mean_power'],
        )
        matched_mmse_result['label'] = f'MMSE eq.pow ({key})'

        results[f'sac_{key}'] = sac_result
        results[f'mmse_matched_{key}'] = matched_mmse_result

    cfg.config_learner.get_state_args['csi_format'] = default_csi_format
    cfg.learned_precoder_args['action_format'] = default_action_format

    with gzip.open(gzip_path, 'wb') as file:
        pickle.dump({'error_sweep_range': error_sweep_range, 'results': results}, file=file)
    print(f'Saved: {gzip_path}')
