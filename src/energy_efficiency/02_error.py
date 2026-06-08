"test scirpt to evaluate learned performances"

import numpy as np


from src.config.config import Config
from pathlib import Path
from src.analysis.helpers.test_mmse_precoder import test_mmse_precoder_error_sweep
from src.analysis.helpers.test_learned_precoder import test_sac_precoder_error_sweep
from src.analysis.helpers.test_learned_precoder import test_learned_rsma_complete_error_sweep
from src.analysis.helpers.test_learned_precoder import test_learned_rsma_power_common_error_sweep


error_sweep_range = np.arange(0,0.11, 0.01)
user_dist_sweep_range = np.arange(500,50500, 500)

monte_carlo_iterations = 10000
config = Config()



# model_path= Path(config.trained_models_path, 'RSMA_Journal', '01_mixed_objective_error', 'full_rsma_snap_3.863')
#
# test_learned_rsma_complete_error_sweep(
#     config=config,
#     model_path=model_path,
#     error_sweep_parameter='additive_error_on_cosine_of_aod',
#     error_sweep_range=error_sweep_range,
#     monte_carlo_iterations=monte_carlo_iterations,
#     metrics=['sumrate','fairness']
# )

# model_path= Path(config.trained_models_path, 'RSMA_Journal', '01_mixed_objective', 'full_snap_4.163')
#
# test_sac_precoder_error_sweep(
#     config=config,
#     model_path=model_path,
#     error_sweep_parameter='additive_error_on_cosine_of_aod',
#     error_sweep_range=error_sweep_range,
#     monte_carlo_iterations=monte_carlo_iterations,
#     metrics=['sumrate','fairness']
# )



