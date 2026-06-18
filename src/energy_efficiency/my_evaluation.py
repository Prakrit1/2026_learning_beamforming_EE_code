"test script to evaluate learned performance of SAC"
import numpy as np
from src.config.config import Config
from pathlib import Path
from src.analysis.helpers.test_mmse_precoder import test_mmse_precoder_error_sweep
from src.analysis.helpers.test_learned_precoder import test_sac_precoder_error_sweep

error_sweep_range = np.arange(0, 0.11, 0.01)
monte_carlo_iterations = 10000

def get_best_model_path(trained_models_path, training_name):
    base_path = Path(trained_models_path, training_name, 'base')
    checkpoints = [p for p in base_path.iterdir() if p.is_dir() and 'full_snap' in p.name]
    best = sorted(checkpoints, key=lambda p: float(p.name.split('_')[-1]))[-1]
    return best

# ---------------------------------------------------------------------------
# Full EE model -> saves to outputs/metrics/full_EE/error_sweep/
# ---------------------------------------------------------------------------
config = Config()
config.config_learner.training_name = 'full_EE'
model_path_full_EE = get_best_model_path(config.trained_models_path, 'full_EE')
test_sac_precoder_error_sweep(
    config=config,
    model_path=model_path_full_EE,
    error_sweep_parameter='additive_error_on_cosine_of_aod',
    error_sweep_range=error_sweep_range,
    monte_carlo_iterations=monte_carlo_iterations,
    metrics=['sumrate']
)
test_mmse_precoder_error_sweep(
    config=config,
    error_sweep_parameter='additive_error_on_cosine_of_aod',
    error_sweep_range=error_sweep_range,
    monte_carlo_iterations=monte_carlo_iterations,
    metrics=['sumrate']
)

# ---------------------------------------------------------------------------
# Simplified EE model -> saves to outputs/metrics/simplified_EE/error_sweep/
# ---------------------------------------------------------------------------
config2 = Config()
config2.config_learner.training_name = 'simplified_EE'
model_path_simplified_EE = get_best_model_path(config2.trained_models_path, 'simplified_EE')
test_sac_precoder_error_sweep(
    config=config2,
    model_path=model_path_simplified_EE,
    error_sweep_parameter='additive_error_on_cosine_of_aod',
    error_sweep_range=error_sweep_range,
    monte_carlo_iterations=monte_carlo_iterations,
    metrics=['sumrate']
)
test_mmse_precoder_error_sweep(
    config=config2,
    error_sweep_parameter='additive_error_on_cosine_of_aod',
    error_sweep_range=error_sweep_range,
    monte_carlo_iterations=monte_carlo_iterations,
    metrics=['sumrate']
)

# ---------------------------------------------------------------------------
# Full EE WITHOUT normalization -> saves to outputs/metrics/full_EE_without_normalization/error_sweep/
# ---------------------------------------------------------------------------
#config3 = Config()
#config3.config_learner.training_name = 'full_EE_without_normalization'
#model_path_full_EE_without_norm = get_best_model_path(config3.trained_models_path, 'full_EE_without_normalization')
#test_sac_precoder_error_sweep(
#    config=config3,
#    model_path=model_path_full_EE_without_norm,
#    error_sweep_parameter='additive_error_on_cosine_of_aod',
#    error_sweep_range=error_sweep_range,
#    monte_carlo_iterations=monte_carlo_iterations,
#    metrics=['sumrate']
#)
#test_mmse_precoder_error_sweep(
#    config=config3,
#    error_sweep_parameter='additive_error_on_cosine_of_aod',
#    error_sweep_range=error_sweep_range,
#    monte_carlo_iterations=monte_carlo_iterations,
#    metrics=['sumrate']
#)

# ---------------------------------------------------------------------------
# Simplified EE WITHOUT normalization -> saves to outputs/metrics/simplified_ee_without_normalization/error_sweep/
# ---------------------------------------------------------------------------
config4 = Config()
config4.config_learner.training_name = 'simplified_ee_without_normalization'
model_path_simplified_EE_without_norm = get_best_model_path(config4.trained_models_path, 'simplified_ee_without_normalization')
test_sac_precoder_error_sweep(
    config=config4,
    model_path=model_path_simplified_EE_without_norm,
    error_sweep_parameter='additive_error_on_cosine_of_aod',
    error_sweep_range=error_sweep_range,
    monte_carlo_iterations=monte_carlo_iterations,
    metrics=['sumrate']
)
test_mmse_precoder_error_sweep(
    config=config4,
    error_sweep_parameter='additive_error_on_cosine_of_aod',
    error_sweep_range=error_sweep_range,
    monte_carlo_iterations=monte_carlo_iterations,
    metrics=['sumrate']
)
