"""
Rate AND power, measured TOGETHER from the same Monte Carlo samples (same
approach as rate_power_error_sweep.py, see that file's docstring for why
measuring only one of the two at a time is insufficient), swept across MEAN
USER DISTANCE instead of CSIT error -- 10/25/50 km, all closer together than
the paper-matching 100 km default this project otherwise always uses.
Smaller user_dist_average is this codebase's existing lever for "users are
less orthogonal": channel_correlation_distance_sweep.py already shows mean
channel correlation rising as user distance shrinks below the 100 km
operating point, so this script logs that same correlation number alongside
rate/power at each swept distance, rather than just asserting the
"closer = less orthogonal" relationship.

Full cross-product, per user request ("do all three" across model-scope,
roam-handling, and CSIT-error-level clarifying questions):
- 6 SAC/EE models: the 3 final raw-power Dinkelbach checkpoints
  (aod=0.0/0.025/0.05) plus the 3 aod=0.05 fairness variants (0.5/1.5/3.0),
  which extend the aod=0.05 checkpoint -- that checkpoint is NOT duplicated,
  it just serves as both "aod=0.05" and "fairness=0" depending on which
  family you're looking at.
- MMSE as a classical reference. Its power is EXACTLY the full budget at
  every distance by mathematical construction (mmse_precoder_normalized
  always rescales to power_constraint_watt -- see
  classical_baselines_zero_error.py's docstring for the algebraic argument),
  so MMSE's power is recorded directly with no Monte Carlo, only its rate
  needs simulating.
- Both roam conventions: fixed distance (user_dist_bound=0, cleanest
  isolation of the distance effect) and the paper's own proportional roam
  (user_dist_bound=0.5, i.e. the 100 km scenario's own +-50% convention).
- Both CSIT error levels this project already standardizes on: 0.0 (isolates
  the distance effect from CSIT-error robustness, a separate already-covered
  question) and 0.05 (this project's other standard evaluation point).
"""
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False

import gzip
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config.config import Config
from src.config.config_plotting import PlotConfig
from src.data.calc_sum_rate import calc_sum_rate
from src.data.calc_tx_power_distribution import calc_tx_power_distribution
from src.data.satellite_manager import SatelliteManager
from src.data.user_manager import UserManager
from src.utils.calc_channel_correlation import calc_channel_correlation
from src.utils.get_precoding import get_precoding_learned_clip_only, get_precoding_mmse
from src.utils.load_model import load_model
from src.utils.update_sim import update_sim

DISTANCE_VALUES = [10_000, 25_000, 50_000]  # meters
ROAM_VARIANTS = [('fixed', 0.0), ('roam', 0.5)]  # (label, user_dist_bound)
ERROR_VALUES = [0.0, 0.05]

# Halved from this project's usual 10000 -- the grid here (3 distances x 2
# roam x 2 error = 12 points per model) is ~12x larger than a single-axis
# sweep elsewhere, so at 5000 MC/point the 6 SAC/EE models still total
# ~360k precoder evaluations, comparable in scale to
# rate_power_error_sweep.py's already-proven ~330k-eval job.
MONTE_CARLO_ITERATIONS = 5000
# Matches test_channel_correlation_user_sweep's existing default -- this is
# a cheap, model-independent context measurement, not the main result.
CHANNEL_CORR_ITERATIONS = 1000

SAC_MODELS = {
    'aod=0.0': 'EE_dinkelbach_adaptive_lwin5000_N16K3_eta0.6_rawpow',
    'aod=0.025': 'EE_dinkelbach_adaptive_aod0.025_lwin5000_N16K3_eta0.6_rawpow_lrc1.82e-06_lra1.17e-06',
    'aod=0.05': 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_eta0.6_rawpow',
    'fair=0.5': 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_eta0.6_rawpow_fair0.5',
    'fair=1.5': 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_eta0.6_rawpow_fair1.5',
    'fair=3.0': 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_eta0.6_rawpow_fair3.0',
}
MODEL_ORDER = list(SAC_MODELS.keys()) + ['MMSE']

# Two ordered families get their own single-hue sequential ramp (light ->
# dark as the underlying parameter increases), per this project's
# color-by-job convention -- NOT six arbitrary categorical hues, since both
# training-error-bound and fairness-weight are ordered parameters, not
# unrelated categories. MMSE is a flat reference, shown in gold (its
# established identity in beampattern_full_comparison.py) with a dashed
# style so it reads as "reference line", not a seventh trend.
MODEL_COLORS = {
    'aod=0.0': '#89b4e1',    # light blue
    'aod=0.025': '#0068b4',  # medium blue
    'aod=0.05': '#00326d',   # dark blue -- also the fairness family's anchor point
    'fair=0.5': '#f4a198',   # light peach
    'fair=1.5': '#d45b65',   # medium peach/red
    'fair=3.0': '#9d2246',   # dark red
    'MMSE': '#caa023',       # gold, reference
}
MODEL_LINESTYLES = {
    'aod=0.0': 'solid', 'aod=0.025': 'solid', 'aod=0.05': 'solid',
    'fair=0.5': 'dashdot', 'fair=1.5': 'dashdot', 'fair=3.0': 'dashdot',
    'MMSE': 'dashed',
}
MODEL_MARKERS = {
    'aod=0.0': 'o', 'aod=0.025': 's', 'aod=0.05': '^',
    'fair=0.5': 'P', 'fair=1.5': 'D', 'fair=3.0': 'v',
    'MMSE': 'x',
}


def get_best_model_path(trained_models_path, training_name):
    """Session-aware checkpoint selection -- see my_evaluation.py's identical function for the full rationale."""
    import os

    base_path = Path(trained_models_path, training_name, 'base')
    checkpoints = [p for p in base_path.iterdir() if p.is_dir() and 'full_snap' in p.name]
    if not checkpoints:
        raise FileNotFoundError(f'No checkpoints found under {base_path}')

    checkpoints_by_time = sorted(checkpoints, key=lambda p: os.path.getmtime(p))

    max_gap_seconds = 90 * 60
    session_start_idx = len(checkpoints_by_time) - 1
    for i in range(len(checkpoints_by_time) - 1, 0, -1):
        gap = os.path.getmtime(checkpoints_by_time[i]) - os.path.getmtime(checkpoints_by_time[i - 1])
        if gap > max_gap_seconds:
            session_start_idx = i
            break
    else:
        session_start_idx = 0

    same_session_checkpoints = checkpoints_by_time[session_start_idx:]
    best = sorted(same_session_checkpoints, key=lambda p: float(p.name.split('_')[-1]))[-1]
    return best


def configure_scenario(cfg, distance, roam_bound, error_value):
    cfg.user_distribution_mode = 'uniform'
    cfg.user_dist_average = distance
    cfg.user_dist_bound = roam_bound
    error_param = 'additive_error_on_cosine_of_aod'
    cfg.config_error_model.error_rng_parametrizations[error_param]['args']['low'] = -error_value
    cfg.config_error_model.error_rng_parametrizations[error_param]['args']['high'] = error_value


def run_sac_point(cfg, precoder_network, norm_factors, label):
    satellite_manager = SatelliteManager(config=cfg)
    user_manager = UserManager(config=cfg)

    rate_samples = np.zeros(MONTE_CARLO_ITERATIONS)
    power_samples = np.zeros(MONTE_CARLO_ITERATIONS)
    for iter_idx in range(MONTE_CARLO_ITERATIONS):
        update_sim(cfg, satellite_manager, user_manager)
        w_precoder = get_precoding_learned_clip_only(cfg, user_manager, satellite_manager, norm_factors, precoder_network)
        rate_samples[iter_idx] = calc_sum_rate(
            channel_state=satellite_manager.channel_state_information,
            w_precoder=w_precoder,
            noise_power_watt=cfg.noise_power_watt,
        )
        power_samples[iter_idx] = calc_tx_power_distribution(w_precoder=w_precoder).sum()

    print(f'  [{label}] rate={rate_samples.mean():.4f} bps/Hz, '
          f'power={power_samples.mean():.2f} W ({100 * power_samples.mean() / cfg.power_constraint_watt:.1f}% of budget)')
    return rate_samples.mean(), rate_samples.std(), power_samples.mean(), power_samples.std()


def run_mmse_point(cfg):
    satellite_manager = SatelliteManager(config=cfg)
    user_manager = UserManager(config=cfg)

    rate_samples = np.zeros(MONTE_CARLO_ITERATIONS)
    for iter_idx in range(MONTE_CARLO_ITERATIONS):
        update_sim(cfg, satellite_manager, user_manager)
        w_precoder = get_precoding_mmse(cfg, user_manager, satellite_manager)
        rate_samples[iter_idx] = calc_sum_rate(
            channel_state=satellite_manager.channel_state_information,
            w_precoder=w_precoder,
            noise_power_watt=cfg.noise_power_watt,
        )
    # power is exactly the full budget by construction -- no MC needed, see module docstring
    power_mean = cfg.power_constraint_watt
    print(f'  [MMSE] rate={rate_samples.mean():.4f} bps/Hz, power={power_mean:.2f} W (100.0% of budget, by construction)')
    return rate_samples.mean(), rate_samples.std(), power_mean, 0.0


def measure_channel_correlation(cfg, distance, roam_bound):
    cfg.user_distribution_mode = 'uniform'
    cfg.user_dist_average = distance
    cfg.user_dist_bound = roam_bound

    satellite_manager = SatelliteManager(config=cfg)
    user_manager = UserManager(config=cfg)

    pair_ids = [(0, 1), (0, 2), (1, 2)]
    pair_corrs = np.zeros((len(pair_ids), CHANNEL_CORR_ITERATIONS))
    for iter_idx in range(CHANNEL_CORR_ITERATIONS):
        update_sim(cfg, satellite_manager, user_manager)
        for pair_idx, (u1, u2) in enumerate(pair_ids):
            pair_corrs[pair_idx, iter_idx] = abs(calc_channel_correlation(
                channel_1=satellite_manager.channel_state_information[u1, :],
                channel_2=satellite_manager.channel_state_information[u2, :],
            ))
    return pair_corrs.mean()


if __name__ == '__main__':
    power_budget = Config().power_constraint_watt

    # ---- load all 6 SAC/EE networks once, reused across every scenario point ----
    loaded_models = {}
    for label, training_name in SAC_MODELS.items():
        cfg_tmp = Config()
        model_path = get_best_model_path(cfg_tmp.trained_models_path, training_name)
        precoder_network, norm_factors = load_model(model_path)
        loaded_models[label] = (precoder_network, norm_factors, training_name)
        print(f'[rate_power_distance_sweep] {label} -> {model_path}')

    # results[roam_label][error_value][distance][model_label] = dict(mean_rate, std_rate, mean_power, std_power)
    results = {roam_label: {ev: {d: {} for d in DISTANCE_VALUES} for ev in ERROR_VALUES} for roam_label, _ in ROAM_VARIANTS}
    # channel_corr[roam_label][distance] = mean |rho|, independent of model/error
    channel_corr = {roam_label: {} for roam_label, _ in ROAM_VARIANTS}

    for roam_label, roam_bound in ROAM_VARIANTS:
        for distance in DISTANCE_VALUES:
            cfg_corr = Config()
            cfg_corr.show_plots = False
            corr = measure_channel_correlation(cfg_corr, distance, roam_bound)
            channel_corr[roam_label][distance] = corr
            print(f'[correlation] roam={roam_label}, distance={distance/1000:.0f}km: mean |rho|={corr:.4f}')

        for error_value in ERROR_VALUES:
            for distance in DISTANCE_VALUES:
                print(f'\n=== roam={roam_label}, error={error_value:.2f}, distance={distance/1000:.0f}km ===')

                for label, (precoder_network, norm_factors, training_name) in loaded_models.items():
                    cfg = Config()
                    cfg.show_plots = False
                    cfg.config_learner.training_name = training_name
                    if norm_factors != {}:
                        cfg.config_learner.get_state_args['norm_state'] = True
                    configure_scenario(cfg, distance, roam_bound, error_value)

                    mean_rate, std_rate, mean_power, std_power = run_sac_point(cfg, precoder_network, norm_factors, label)
                    results[roam_label][error_value][distance][label] = {
                        'mean_rate': mean_rate, 'std_rate': std_rate,
                        'mean_power': mean_power, 'std_power': std_power,
                    }

                cfg_mmse = Config()
                cfg_mmse.show_plots = False
                configure_scenario(cfg_mmse, distance, roam_bound, error_value)
                mean_rate, std_rate, mean_power, std_power = run_mmse_point(cfg_mmse)
                results[roam_label][error_value][distance]['MMSE'] = {
                    'mean_rate': mean_rate, 'std_rate': std_rate,
                    'mean_power': mean_power, 'std_power': std_power,
                }

    # ---- persist raw results so plots can be regenerated without rerunning Monte Carlo ----
    out_path = Path(Config().output_metrics_path, 'rate_power_distance_sweep')
    out_path.mkdir(parents=True, exist_ok=True)
    with gzip.open(Path(out_path, 'rate_power_distance_sweep.gzip'), 'wb') as file:
        pickle.dump({
            'distance_values': DISTANCE_VALUES,
            'roam_variants': ROAM_VARIANTS,
            'error_values': ERROR_VALUES,
            'power_budget': power_budget,
            'results': results,
            'channel_corr': channel_corr,
        }, file=file)
    print(f"\nSaved: {Path(out_path, 'rate_power_distance_sweep.gzip')}")

    # ---- printed summary table ----
    print('\n' + '=' * 100)
    print('SUMMARY: roam | error | distance | model -> rate, % of budget, channel correlation')
    print('=' * 100)
    for roam_label, _ in ROAM_VARIANTS:
        for error_value in ERROR_VALUES:
            for distance in DISTANCE_VALUES:
                corr = channel_corr[roam_label][distance]
                for label in MODEL_ORDER:
                    entry = results[roam_label][error_value][distance][label]
                    pct_budget = 100 * entry['mean_power'] / power_budget
                    print(f'roam={roam_label:5s} error={error_value:.2f} distance={distance/1000:4.0f}km '
                          f'model={label:9s} rate={entry["mean_rate"]:.4f} bps/Hz  '
                          f'power={pct_budget:5.1f}% of budget  channel_corr={corr:.4f}')
    print('=' * 100)

    # ---- plots ----
    plot_cfg = PlotConfig()
    matplotlib.rcParams['text.usetex'] = False  # PlotConfig() resets this on construction

    def plot_metric_grid(metric_key, ylabel, title, filename, is_power):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
        distances_km = np.array(DISTANCE_VALUES) / 1000
        for row, (roam_label, _) in enumerate(ROAM_VARIANTS):
            for col, error_value in enumerate(ERROR_VALUES):
                ax = axes[row, col]
                for label in MODEL_ORDER:
                    values = [results[roam_label][error_value][d][label][metric_key] for d in DISTANCE_VALUES]
                    if is_power:
                        values = [100 * v / power_budget for v in values]
                    ax.plot(
                        distances_km, values,
                        color=MODEL_COLORS[label], linestyle=MODEL_LINESTYLES[label],
                        marker=MODEL_MARKERS[label], markersize=7, linewidth=1.6,
                        markeredgecolor='white', markeredgewidth=0.5, label=label,
                    )
                ax.set_title(f'roam={roam_label}, error={error_value:.2f}', fontsize=10.5)
                ax.grid(True, alpha=0.25, linewidth=0.5)
                ax.set_axisbelow(True)
                if row == 1:
                    ax.set_xlabel('Mean user distance [km]')
                if col == 0:
                    ax.set_ylabel(ylabel)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=7, fontsize=8.5, framealpha=1.0)
        fig.suptitle(title, fontsize=13, y=1.09)
        fig.tight_layout()

        pdf_path = Path(plot_cfg.plots_parent_path, 'pdf')
        pdf_path.mkdir(parents=True, exist_ok=True)
        out = Path(pdf_path, f'{filename}.pdf')
        fig.savefig(out, bbox_inches='tight', dpi=300, transparent=True)
        print(f'Saved: {out}')
        plt.close(fig)

    plot_metric_grid(
        metric_key='mean_power', ylabel='% of power budget used',
        title='Energy savings vs. mean user distance (10/25/50 km, closer than the 100 km paper scenario)',
        filename='power_savings_vs_distance', is_power=True,
    )
    plot_metric_grid(
        metric_key='mean_rate', ylabel='Rate R [bps/Hz]',
        title='Rate vs. mean user distance (10/25/50 km, closer than the 100 km paper scenario)',
        filename='rate_vs_distance', is_power=False,
    )

    # ---- channel correlation context plot: is "closer distance" actually "less orthogonal" here? ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    distances_km = np.array(DISTANCE_VALUES) / 1000
    roam_colors = {'fixed': '#254796', 'roam': '#c9622a'}
    for roam_label, _ in ROAM_VARIANTS:
        values = [channel_corr[roam_label][d] for d in DISTANCE_VALUES]
        ax.plot(distances_km, values, color=roam_colors[roam_label], marker='o', markersize=7,
                linewidth=1.8, markeredgecolor='white', markeredgewidth=0.5, label=f'roam={roam_label}')
    ax.set_xlabel('Mean user distance [km]')
    ax.set_ylabel('Mean channel correlation |ρ|')
    ax.set_title('Channel correlation at the swept distances (context for the energy-savings plots)', fontsize=11)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc='best')
    fig.tight_layout()

    pdf_path = Path(plot_cfg.plots_parent_path, 'pdf')
    out = Path(pdf_path, 'channel_correlation_at_swept_distances.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=300, transparent=True)
    print(f'Saved: {out}')
    plt.close(fig)
