
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import gzip
import pickle

from src.config.config import Config
from src.analysis.helpers.test_rsma_precoder import test_rsma_precoder_user_distance_sweep
from src.analysis.helpers.test_rsma_precoder import test_rsma_precoder_error_sweep
from src.analysis.helpers.test_rsma_precoder import test_rsma_precoder_user_number_sweep


sweep_parameter='user_distance_sweep' # available 'error_sweep','user_distance_sweep', 'user_number_sweep'

def test_rsma_best_alpha_user_distance():

    monte_carlo_iterations = 5000

    rsma_factors = np.arange(0, 1+0.01, step=0.01)  # exclusive interval
    distance_sweep_range = np.arange(500,50500, 500) #np.linspace(500, 50_000, 100)

    cfg = Config()
    cfg.show_plots = False

    common_part_precoding_style = cfg.common_part_precoding_style

    results = np.zeros((len(rsma_factors), len(distance_sweep_range)))
    results_fairness = np.zeros((len(rsma_factors), len(distance_sweep_range)))

    for rsma_factor_id, rsma_factor in enumerate(rsma_factors):

        metrics = test_rsma_precoder_user_distance_sweep(
            config=cfg,
            distance_sweep_range=distance_sweep_range,
            rsma_factor=rsma_factor,
            common_part_precoding_style=common_part_precoding_style,
            monte_carlo_iterations=monte_carlo_iterations,
            metrics=['sumrate','fairness']
        )

        results[rsma_factor_id, :] = metrics[list(metrics.keys())[0]]['mean']
        results_fairness[rsma_factor_id, :] = metrics[list(metrics.keys())[1]]['mean']

    fig, ax = plt.subplots()
    indices = ax.twinx()

    best_values = np.max(results, axis=0)
    best_indices = np.argmax(results, axis=0)
    fairness_at_best = results_fairness[best_indices, np.arange(results.shape[1])]
    best_power_factor = rsma_factors[best_indices]

    ax.plot(distance_sweep_range, best_values, label='sum rate')
    ax.plot(distance_sweep_range, fairness_at_best, label='fairness')
    indices.scatter(distance_sweep_range, rsma_factors[best_indices], label='rsma alpha', color='black')

    fig.legend(labelcolor="linecolor")



    plt.show()
    metrics = {
        'calc_sum_rate': {
            'mean': best_values,
            'std': np.zeros_like(best_values),
        },
        'calc_jain_fairness': {
            'mean': fairness_at_best,
            'std': np.zeros_like(fairness_at_best),
        },
        'power_factor': {
            'mean': best_power_factor,
            'std': np.zeros_like(best_power_factor),
        }
    }
    # metrics = {'calc_sum_rate': {'mean': best_values}, 'fairness at best sumrate': fairness_at_best, 'power factor': best_power_factor}

    # print(best_values)
    precoder_name = 'rsma_genie'

    def save_results():
        name = (
            f'testing_{precoder_name}'
            f'_sweep_{distance_sweep_range[0]}_{distance_sweep_range[-1]}'
            f'.gzip'
        )
        results_path = Path(cfg.output_metrics_path, cfg.config_learner.training_name, 'user_distance_sweep_best_alpha')
        results_path.mkdir(parents=True, exist_ok=True)
        with gzip.open(Path(results_path, name), 'wb') as file:
            pickle.dump([distance_sweep_range, metrics], file=file)

    save_results()

def test_rsma_best_alpha_error():

    monte_carlo_iterations = 10000

    rsma_factors = np.arange(0, 1+0.01, step=0.01)  # exclusive interval
    error_sweep_range = np.arange(0,0.11, 0.01)

    cfg = Config()
    cfg.show_plots = False

    common_part_precoding_style = cfg.common_part_precoding_style

    results = np.zeros((len(rsma_factors), len(error_sweep_range)))
    results_fairness = np.zeros((len(rsma_factors), len(error_sweep_range)))

    for rsma_factor_id, rsma_factor in enumerate(rsma_factors):

        metrics = test_rsma_precoder_error_sweep(
            config=cfg,
            error_sweep_parameter='additive_error_on_cosine_of_aod',
            error_sweep_range=error_sweep_range,
            rsma_factor=rsma_factor,
            common_part_precoding_style=common_part_precoding_style,
            monte_carlo_iterations=monte_carlo_iterations,
            metrics=['sumrate','fairness']
        )

        results[rsma_factor_id, :] = metrics[list(metrics.keys())[0]]['mean']
        results_fairness[rsma_factor_id, :] = metrics[list(metrics.keys())[1]]['mean']

    fig, ax = plt.subplots()
    indices = ax.twinx()

    best_values = np.max(results, axis=0)
    best_indices = np.argmax(results, axis=0)
    fairness_at_best = results_fairness[best_indices, np.arange(results.shape[1])]
    best_power_factor = rsma_factors[best_indices]

    ax.plot(error_sweep_range, best_values, label='sum rate')
    ax.plot(error_sweep_range, fairness_at_best, label='fairness')
    indices.scatter(error_sweep_range, rsma_factors[best_indices], label='rsma alpha', color='black')

    fig.legend(labelcolor="linecolor")



    plt.show()
    metrics = {
        'calc_sum_rate': {
            'mean': best_values,
            'std': np.zeros_like(best_values),
        },
        'calc_jain_fairness': {
            'mean': fairness_at_best,
            'std': np.zeros_like(fairness_at_best),
        },
        'power_factor': {
            'mean': best_power_factor,
            'std': np.zeros_like(best_power_factor),
        }
    }
    # metrics = {'calc_sum_rate': {'mean': best_values}, 'fairness at best sumrate': fairness_at_best, 'power factor': best_power_factor}

    # print(best_values)
    precoder_name = 'rsma_genie'

    def save_results():
        name = (
            f'testing_{precoder_name}'
            f'_sweep_{error_sweep_range[0]}_{error_sweep_range[-1]}'
            f'.gzip'
        )
        results_path = Path(cfg.output_metrics_path, cfg.config_learner.training_name, 'error_sweep_best_alpha')
        results_path.mkdir(parents=True, exist_ok=True)
        with gzip.open(Path(results_path, name), 'wb') as file:
            pickle.dump([error_sweep_range, metrics], file=file)

    save_results()

def test_rsma_best_alpha_user_number():

    monte_carlo_iterations = 10000

    rsma_factors = np.arange(0, 1+0.01, step=0.01)  # exclusive interval
    user_number_sweep_range = np.arange(1,16, 1)

    cfg = Config()
    cfg.show_plots = False

    common_part_precoding_style = cfg.common_part_precoding_style

    results = np.zeros((len(rsma_factors), len(user_number_sweep_range)))
    results_fairness = np.zeros((len(rsma_factors), len(user_number_sweep_range)))

    for rsma_factor_id, rsma_factor in enumerate(rsma_factors):

        metrics = test_rsma_precoder_user_number_sweep(
            config=cfg,
            user_number_sweep_range=user_number_sweep_range,
            rsma_factor=rsma_factor,
            common_part_precoding_style=common_part_precoding_style,
            monte_carlo_iterations=monte_carlo_iterations,
            metrics=['sumrate','fairness']
        )

        results[rsma_factor_id, :] = metrics[list(metrics.keys())[0]]['mean']
        results_fairness[rsma_factor_id, :] = metrics[list(metrics.keys())[1]]['mean']

    fig, ax = plt.subplots()
    indices = ax.twinx()

    best_values = np.max(results, axis=0)
    best_indices = np.argmax(results, axis=0)
    fairness_at_best = results_fairness[best_indices, np.arange(results.shape[1])]
    best_power_factor = rsma_factors[best_indices]

    ax.plot(user_number_sweep_range, best_values, label='sum rate')
    ax.plot(user_number_sweep_range, fairness_at_best, label='fairness')
    indices.scatter(user_number_sweep_range, rsma_factors[best_indices], label='rsma alpha', color='black')

    fig.legend(labelcolor="linecolor")



    plt.show()
    metrics = {
        'calc_sum_rate': {
            'mean': best_values,
            'std': np.zeros_like(best_values),
        },
        'calc_jain_fairness': {
            'mean': fairness_at_best,
            'std': np.zeros_like(fairness_at_best),
        },
        'power_factor': {
            'mean': best_power_factor,
            'std': np.zeros_like(best_power_factor),
        }
    }
    # metrics = {'calc_sum_rate': {'mean': best_values}, 'fairness at best sumrate': fairness_at_best, 'power factor': best_power_factor}

    # print(best_values)
    precoder_name = 'rsma_genie'

    def save_results():
        name = (
            f'testing_{precoder_name}'
            f'_sweep_{user_number_sweep_range[0]}_{user_number_sweep_range[-1]}'
            f'.gzip'
        )
        results_path = Path(cfg.output_metrics_path, cfg.config_learner.training_name, 'user_number_sweep_best_alpha')
        results_path.mkdir(parents=True, exist_ok=True)
        with gzip.open(Path(results_path, name), 'wb') as file:
            pickle.dump([user_number_sweep_range, metrics], file=file)

    save_results()

if __name__ == '__main__':
    if sweep_parameter == 'user_distance_sweep':
        test_rsma_best_alpha_user_distance()
    elif sweep_parameter == 'error_sweep':
        test_rsma_best_alpha_error()
    else:
        ValueError("No valid sweep_parameter provided")
