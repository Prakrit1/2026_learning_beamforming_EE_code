"""
Adds the missing 'EE model, evaluated at full/total power' curve to the
error-sweep cache used by plotting_scenario.py.

No new training: this loads the already-trained Delta-eps=0.0 EE
(energy_efficiency_dinkelbach_adaptive) checkpoint and re-evaluates it with
get_precoding_learned (always-rescale-to-budget) instead of
get_precoding_learned_clip_only, across the same error_sweep_range already
used by plotting_scenario.py. The result is merged into the existing
rate_power_triplet.gzip cache under the key 'sac_aod0.0_fullpower', so
`python3 src/energy_efficiency/plotting_scenario.py --plot-only` picks it up
without re-running the other (already-cached) curves.
"""

import gzip
import pickle
from pathlib import Path

from src.config.config import Config
from src.utils.get_precoding import get_precoding_learned
from src.utils.load_model import load_model
from src.energy_efficiency.plotting_scenario import (
    run_rate_power_sweep,
    get_best_model_path,
    error_sweep_range,
    CHECKPOINTS,
)

TRAINING_NAME = CHECKPOINTS['aod0.0']
RESULT_KEY = 'sac_aod0.0_fullpower'
CURVE_LABEL = 'SAC (75 W budget)'


def main():
    cfg = Config()
    cfg.config_learner.training_name = TRAINING_NAME

    model_path = get_best_model_path(cfg.trained_models_path, TRAINING_NAME)
    print(f'[add_fullpower_curve] checkpoint: {model_path}')

    precoder_network, norm_factors = load_model(model_path)
    if norm_factors != {}:
        cfg.config_learner.get_state_args['norm_state'] = True

    fullpower_result = run_rate_power_sweep(
        cfg, 'EE (aod0.0, full-power inference)',
        lambda c, um, sm: get_precoding_learned(c, um, sm, norm_factors, precoder_network),
    )
    fullpower_result['label'] = CURVE_LABEL
    fullpower_result['training_name'] = TRAINING_NAME
    fullpower_result['checkpoint'] = str(model_path)

    out_path = Path(cfg.output_metrics_path, 'EE_lwin5000_3gpp_triplet')
    gzip_path = Path(out_path, 'rate_power_triplet.gzip')

    with gzip.open(gzip_path, 'rb') as file:
        data = pickle.load(file)

    assert (data['error_sweep_range'] == error_sweep_range).all(), (
        'error_sweep_range in the cached gzip does not match plotting_scenario.py -- '
        'refusing to merge mismatched sweeps.'
    )

    data['results'][RESULT_KEY] = fullpower_result

    with gzip.open(gzip_path, 'wb') as file:
        pickle.dump(data, file=file)

    print(f'[add_fullpower_curve] merged "{RESULT_KEY}" into {gzip_path}')


if __name__ == '__main__':
    main()
