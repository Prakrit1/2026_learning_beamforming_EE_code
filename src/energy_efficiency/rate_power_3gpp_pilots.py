"""
Rate AND power (same Monte Carlo samples, same convention as
rate_power_error_sweep.py) for the two 3GPP-aligned pilot checkpoints
(jobs 153655 nadir / 153656 elev30), plus MMSE evaluated in each geometry
at the SAME 3GPP system parameters (30 dBi / 75 W) as the spatial
reference.

All Monte Carlo in this script runs at the current system defaults
(TR 38.821 Set-1: 30 dBi, 75 W; elev30 additionally sets
EE_TARGET_ELEVATION_DEG=30). Nothing here is evaluated at the old
20 dBi / 100 W parameters: the old aod=0.05 checkpoint's curve is loaded
as a BENCHMARK from the results saved by this script's first run (job
154052, which evaluated it under its own historical system -- see that
run's log) and re-plotted for context, clearly labeled. Rates across old
and new systems are not directly comparable (intentionally different
link budgets); the benchmark is context, not a controlled ablation.

Because the power budgets differ (100 W benchmark vs 75 W current), the
power panel and the rate-vs-power trajectory are plotted in PERCENT OF
EACH SYSTEM'S OWN BUDGET; absolute watts are stored in the gzip.
"""
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False

import gzip
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config.config_plotting import PlotConfig
from src.data.calc_sum_rate import calc_sum_rate
from src.data.calc_tx_power_distribution import calc_tx_power_distribution
from src.data.satellite_manager import SatelliteManager
from src.data.user_manager import UserManager
from src.utils.get_precoding import get_precoding_learned_clip_only, get_precoding_mmse
from src.utils.load_model import load_model
from src.utils.update_sim import update_sim

error_sweep_range = np.linspace(0, 0.10, 11)
monte_carlo_iterations = 10000

BENCHMARK_KEY = 'old_aod0.05'

SYSTEM_ENV_VARS = ['EE_SAT_GAIN_DBI', 'EE_POWER_BUDGET_WATT', 'EE_TARGET_ELEVATION_DEG']

SYSTEMS = [
    {
        # Swapped 2026-08-04 to the aod=0.0 nadir checkpoint (job 156358,
        # lwin5000, natively trained at 30 dBi/75 W) to match the current
        # rate-sweep scenario (my_evaluation.py/my_plotting.py). This
        # OVERWRITES the '3gpp_nadir' key's previous aod=0.05 rate/power
        # numbers in the output gzip -- those are still documented in the
        # 2026-07-28 handoff section if needed again (rate 6.60@0/power
        # ~40W), and re-runnable by pointing training_name back.
        'key': '3gpp_nadir',
        'label': 'SAC (3GPP Set-1, nadir, train err 0.0)',
        'mmse_key': 'mmse_nadir',
        'mmse_label': 'MMSE (3GPP Set-1, nadir)',
        'training_name': 'EE_dinkelbach_adaptive_lwin5000_N16K3_satg30_p75_eta0.6_rawpow',
        'env': {},
    },
    {
        'key': '3gpp_elev30',
        'label': 'SAC (3GPP Set-1, elev 30 deg, train err 0.05)',
        'mmse_key': 'mmse_elev30',
        'mmse_label': 'MMSE (3GPP Set-1, elev 30 deg)',
        'training_name': 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_satg30_p75_elev30_eta0.6_rawpow',
        'env': {'EE_TARGET_ELEVATION_DEG': '30'},
    },
]

COLORS = {
    BENCHMARK_KEY: '#888888',
    '3gpp_nadir': '#307b3b',
    'mmse_nadir': '#caa023',
    '3gpp_elev30': '#1b6faf',
    'mmse_elev30': '#c9622a',
}


def make_config(env: dict):
    """Construct a Config under exactly the given system env overrides (all others cleared)."""
    for var in SYSTEM_ENV_VARS:
        os.environ.pop(var, None)
    os.environ.update(env)
    from src.config.config import Config  # env vars are read in Config.__init__, after the overrides above
    cfg = Config()
    cfg.show_plots = False
    return cfg


def get_best_model_path(trained_models_path, training_name):
    """Session-aware checkpoint selection -- see my_evaluation.py's identical function for the full rationale."""
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


def run_rate_power_sweep(cfg, label, get_precoder_func):
    """get_precoder_func(cfg, user_manager, satellite_manager) -> w_precoder."""
    satellite_manager = SatelliteManager(config=cfg)
    user_manager = UserManager(config=cfg)

    error_param = 'additive_error_on_cosine_of_aod'
    initial_error_config = cfg.config_error_model.error_rng_parametrizations[error_param]['args'].copy()

    mean_rate = np.zeros(len(error_sweep_range))
    mean_power = np.zeros(len(error_sweep_range))
    std_rate = np.zeros(len(error_sweep_range))
    std_power = np.zeros(len(error_sweep_range))

    for error_idx, error_value in enumerate(error_sweep_range):
        cfg.config_error_model.error_rng_parametrizations[error_param]['args']['low'] = -error_value
        cfg.config_error_model.error_rng_parametrizations[error_param]['args']['high'] = error_value

        rate_samples = np.zeros(monte_carlo_iterations)
        power_samples = np.zeros(monte_carlo_iterations)

        for iter_idx in range(monte_carlo_iterations):
            update_sim(cfg, satellite_manager, user_manager)
            w_precoder = get_precoder_func(cfg, user_manager, satellite_manager)
            rate_samples[iter_idx] = calc_sum_rate(
                channel_state=satellite_manager.channel_state_information,
                w_precoder=w_precoder,
                noise_power_watt=cfg.noise_power_watt,
            )
            power_samples[iter_idx] = calc_tx_power_distribution(w_precoder=w_precoder).sum()

        mean_rate[error_idx] = rate_samples.mean()
        std_rate[error_idx] = rate_samples.std()
        mean_power[error_idx] = power_samples.mean()
        std_power[error_idx] = power_samples.std()
        print(f'[{label}] error={error_value:.2f}: rate={mean_rate[error_idx]:.4f} bps/Hz, '
              f'power={mean_power[error_idx]:.2f} W ({100 * mean_power[error_idx] / cfg.power_constraint_watt:.1f}% of budget)')

    cfg.config_error_model.error_rng_parametrizations[error_param]['args'] = initial_error_config

    return {
        'power_budget': cfg.power_constraint_watt,
        'mean_rate': mean_rate, 'std_rate': std_rate,
        'mean_power': mean_power, 'std_power': std_power,
    }


if __name__ == '__main__':
    out_path = Path(make_config({}).output_metrics_path, 'EE_3gpp_pilots_rate_power_sweep')
    out_path.mkdir(parents=True, exist_ok=True)
    gzip_path = Path(out_path, 'rate_power_3gpp_pilots.gzip')

    # benchmark: previous result, loaded not re-simulated (see docstring)
    results = {}
    if gzip_path.exists():
        with gzip.open(gzip_path, 'rb') as file:
            previous = pickle.load(file)
        if BENCHMARK_KEY in previous['results']:
            results[BENCHMARK_KEY] = previous['results'][BENCHMARK_KEY]
            results[BENCHMARK_KEY]['label'] = 'SAC benchmark (old system, 20 dBi, 100 W)'
            print(f'[benchmark] loaded {BENCHMARK_KEY} curve from {gzip_path}')
    if BENCHMARK_KEY not in results:
        print(f'[benchmark] WARNING: no {BENCHMARK_KEY} entry found in {gzip_path} -- plotting without benchmark')

    for system in SYSTEMS:
        cfg = make_config(system['env'])
        cfg.config_learner.training_name = system['training_name']
        model_path = get_best_model_path(cfg.trained_models_path, system['training_name'])
        print(f"[{system['key']}] checkpoint: {model_path}")
        print(f"[{system['key']}] sat_gain_dBi={cfg.sat_gain_dBi}, budget={cfg.power_constraint_watt} W, "
              f"user_center_aod_earth_deg={cfg.user_center_aod_earth_deg:.2f}")

        precoder_network, norm_factors = load_model(model_path)
        if norm_factors != {}:
            cfg.config_learner.get_state_args['norm_state'] = True
        results[system['key']] = run_rate_power_sweep(
            cfg, system['label'],
            lambda c, um, sm: get_precoding_learned_clip_only(c, um, sm, norm_factors, precoder_network),
        )
        results[system['key']]['label'] = system['label']

        results[system['mmse_key']] = run_rate_power_sweep(cfg, system['mmse_label'], get_precoding_mmse)
        results[system['mmse_key']]['label'] = system['mmse_label']

    with gzip.open(gzip_path, 'wb') as file:
        pickle.dump({'error_sweep_range': error_sweep_range, 'results': results}, file=file)
    print(f'Saved: {gzip_path}')

    plot_cfg = PlotConfig()
    matplotlib.rcParams['text.usetex'] = False  # PlotConfig() resets this on construction

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # ---- Panel 1: rate vs error ----
    ax = axes[0]
    for key, data in results.items():
        ax.errorbar(error_sweep_range, data['mean_rate'], yerr=data['std_rate'], marker='o',
                    color=COLORS[key], ecolor=COLORS[key], elinewidth=1, capsize=3, linewidth=1.5,
                    markersize=5, label=data['label'])
    ax.set_xlabel('Error Bound')
    ax.set_ylabel('Rate R [bps/Hz]')
    ax.set_title('Rate vs. CSIT error', fontsize=11)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc='best')

    # ---- Panel 2: power (% of own budget) vs error ----
    ax2 = axes[1]
    for key, data in results.items():
        power_pct = 100 * data['mean_power'] / data['power_budget']
        power_pct_std = 100 * data['std_power'] / data['power_budget']
        ax2.errorbar(error_sweep_range, power_pct, yerr=power_pct_std, marker='o',
                     color=COLORS[key], ecolor=COLORS[key], elinewidth=1, capsize=3, linewidth=1.5,
                     markersize=5, label=data['label'])
    ax2.axhline(100, color='#d03b3b', linestyle='--', linewidth=1.5, label='Power budget')
    ax2.set_xlabel('Error Bound')
    ax2.set_ylabel('Mean transmit power [% of own budget]')
    ax2.set_title('Power vs. CSIT error', fontsize=11)
    ax2.grid(True, alpha=0.25, linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=8, loc='best')

    # ---- Panel 3: rate vs power (% of own budget), error as trajectory parameter ----
    ax3 = axes[2]
    for key, data in results.items():
        power_pct = 100 * data['mean_power'] / data['power_budget']
        ax3.plot(power_pct, data['mean_rate'], color=COLORS[key], linewidth=1, zorder=1)
        ax3.scatter(power_pct, data['mean_rate'], color=COLORS[key], s=60, zorder=5,
                    edgecolor='white', linewidth=0.6, label=data['label'])
        for x, y, e in zip(power_pct, data['mean_rate'], error_sweep_range):
            ax3.annotate(f'{e:.2f}', (x, y), textcoords='offset points', xytext=(5, 4), fontsize=6.5)
    ax3.axvline(100, color='#d03b3b', linestyle='--', linewidth=1.2, label='Power budget')
    ax3.set_xlabel('Mean transmit power [% of own budget]')
    ax3.set_ylabel('Rate R [bps/Hz]')
    ax3.set_title('Rate vs. power (labels = error level)', fontsize=11)
    ax3.grid(True, alpha=0.25, linewidth=0.5)
    ax3.set_axisbelow(True)
    ax3.legend(fontsize=8, loc='best')

    fig.suptitle('3GPP Set-1 pilots (30 dBi, 75 W) vs. MMSE, with old-system SAC benchmark', fontsize=13, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    pdf_path = Path(plot_cfg.plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)
    out = Path(pdf_path, 'rate_power_3gpp_pilots.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=300, transparent=True)
    print(f'Saved: {out}')
    plt.close(fig)
