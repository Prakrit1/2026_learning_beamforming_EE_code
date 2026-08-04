"""
Same verification as parajuli_matched_power_check.py, but evaluated under
the 3GPP Set-1 system parameters (30 dBi / 75 W) instead of the old system
(20 dBi / 100 W) these two checkpoints were actually TRAINED under.

User-requested (2026-08-03): "evaluate these checkpoints with 75 watt and
30 dBi". Note this is deliberately an out-of-distribution eval -- the SAC
policy networks (EE_dinkelbach_adaptive_lwin5000_N16K3_eta0.6_rawpow and
..._aod0.05_lwin5000_N16K3_eta0.6_rawpow) were trained against a 20 dBi
channel and a 100 W clip budget; running them against a 30 dBi channel and
75 W budget tests how that already-fixed policy behaves under a different
link budget/power constraint than it ever saw during training, NOT a
retrained-for-this-system result. (The project's own 3GPP-native
checkpoints, e.g. EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_satg30_p75_
eta0.6_rawpow, are the ones actually trained at 30 dBi/75 W -- see
handoff_prompt_EE_evaluation.txt if a native-trained comparison is wanted
instead.)

Because the antenna gain changes the channel itself (not just a clip
budget), NONE of the existing on-disk gzips (all computed under the old
20 dBi/100 W defaults, before the 2026-07-27 3GPP config change) are reusable
here -- every curve is recomputed fresh: SAC rate+power for both checkpoints,
full-budget MMSE, and matched-power MMSE, all under 30 dBi/75 W.
"""
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False

import os
os.environ['EE_SAT_GAIN_DBI'] = '30'
os.environ['EE_POWER_BUDGET_WATT'] = '75'
os.environ.pop('EE_TARGET_ELEVATION_DEG', None)

import gzip
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.config.config import Config
from src.config.config_plotting import PlotConfig
from src.data.calc_sum_rate import calc_sum_rate
from src.data.calc_tx_power_distribution import calc_tx_power_distribution
from src.data.satellite_manager import SatelliteManager
from src.data.user_manager import UserManager
from src.utils.get_precoding import get_precoding_learned_clip_only, get_precoding_mmse
from src.utils.load_model import load_model
from src.utils.update_sim import update_sim

ERROR_PARAM = 'additive_error_on_cosine_of_aod'
ERROR_SWEEP_RANGE = np.linspace(0, 0.10, 11)
MONTE_CARLO_ITERATIONS = 10000

SAC_AOD0_TRAINING_NAME = 'EE_dinkelbach_adaptive_lwin5000_N16K3_eta0.6_rawpow'
SAC_AOD05_TRAINING_NAME = 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_eta0.6_rawpow'
OUT_GZIP_DIR = 'parajuli_3gpp_matched_power_check'


def get_best_model_path(trained_models_path, training_name):
    base_path = Path(trained_models_path, training_name, 'base')
    checkpoints = [p for p in base_path.iterdir() if p.is_dir() and 'full_snap' in p.name]
    if not checkpoints:
        raise FileNotFoundError(f'No checkpoints found under {base_path}')
    checkpoints_by_time = sorted(checkpoints, key=lambda p: os.path.getmtime(p))
    max_gap_seconds = 90 * 60
    session_start_idx = 0
    for i in range(len(checkpoints_by_time) - 1, 0, -1):
        gap = os.path.getmtime(checkpoints_by_time[i]) - os.path.getmtime(checkpoints_by_time[i - 1])
        if gap > max_gap_seconds:
            session_start_idx = i
            break
    same_session_checkpoints = checkpoints_by_time[session_start_idx:]
    best = sorted(same_session_checkpoints, key=lambda p: float(p.name.split('_')[-1]))[-1]
    return best


def sweep_sac(training_name, label):
    """Rate AND power together across the full error sweep, one MC loop
    (same convention as rate_power_error_sweep.py)."""
    cfg = Config()
    cfg.show_plots = False
    cfg.config_learner.training_name = training_name
    model_path = get_best_model_path(cfg.trained_models_path, training_name)
    precoder_network, norm_factors = load_model(model_path)
    if norm_factors != {}:
        cfg.config_learner.get_state_args['norm_state'] = True

    satellite_manager = SatelliteManager(config=cfg)
    user_manager = UserManager(config=cfg)

    mean_rate = np.zeros(len(ERROR_SWEEP_RANGE))
    std_rate = np.zeros(len(ERROR_SWEEP_RANGE))
    mean_power = np.zeros(len(ERROR_SWEEP_RANGE))
    std_power = np.zeros(len(ERROR_SWEEP_RANGE))

    for idx, error_value in enumerate(ERROR_SWEEP_RANGE):
        cfg.config_error_model.error_rng_parametrizations[ERROR_PARAM]['args']['low'] = -error_value
        cfg.config_error_model.error_rng_parametrizations[ERROR_PARAM]['args']['high'] = error_value

        rate_samples = np.zeros(MONTE_CARLO_ITERATIONS)
        power_samples = np.zeros(MONTE_CARLO_ITERATIONS)
        for i in range(MONTE_CARLO_ITERATIONS):
            update_sim(cfg, satellite_manager, user_manager)
            w = get_precoding_learned_clip_only(cfg, user_manager, satellite_manager, norm_factors, precoder_network)
            rate_samples[i] = calc_sum_rate(
                channel_state=satellite_manager.channel_state_information,
                w_precoder=w,
                noise_power_watt=cfg.noise_power_watt,
            )
            power_samples[i] = calc_tx_power_distribution(w_precoder=w).sum()

        mean_rate[idx], std_rate[idx] = rate_samples.mean(), rate_samples.std()
        mean_power[idx], std_power[idx] = power_samples.mean(), power_samples.std()
        print(f'[{label}] error={error_value:.2f}: rate={mean_rate[idx]:.4f} bps/Hz, '
              f'power={mean_power[idx]:.2f} W ({100*mean_power[idx]/cfg.power_constraint_watt:.1f}% of '
              f'{cfg.power_constraint_watt:.0f} W budget)')

    return mean_rate, std_rate, mean_power, std_power


def sweep_mmse(power_constraint_watt, label):
    cfg = Config()
    cfg.show_plots = False
    cfg.mmse_args['power_constraint_watt'] = float(power_constraint_watt)

    satellite_manager = SatelliteManager(config=cfg)
    user_manager = UserManager(config=cfg)

    mean_rate = np.zeros(len(ERROR_SWEEP_RANGE))
    std_rate = np.zeros(len(ERROR_SWEEP_RANGE))

    for idx, error_value in enumerate(ERROR_SWEEP_RANGE):
        cfg.config_error_model.error_rng_parametrizations[ERROR_PARAM]['args']['low'] = -error_value
        cfg.config_error_model.error_rng_parametrizations[ERROR_PARAM]['args']['high'] = error_value

        rate_samples = np.zeros(MONTE_CARLO_ITERATIONS)
        for i in range(MONTE_CARLO_ITERATIONS):
            update_sim(cfg, satellite_manager, user_manager)
            w = get_precoding_mmse(cfg, user_manager, satellite_manager)
            rate_samples[i] = calc_sum_rate(
                channel_state=satellite_manager.channel_state_information,
                w_precoder=w,
                noise_power_watt=cfg.noise_power_watt,
            )
        mean_rate[idx], std_rate[idx] = rate_samples.mean(), rate_samples.std()
        print(f'[{label}] error={error_value:.2f}: rate={mean_rate[idx]:.4f} bps/Hz')

    return mean_rate, std_rate


if __name__ == '__main__':
    cfg_check = Config()
    print(f'System check: sat_gain_dBi={cfg_check.sat_gain_dBi}, power_constraint_watt={cfg_check.power_constraint_watt}')
    assert cfg_check.sat_gain_dBi == 30.0 and cfg_check.power_constraint_watt == 75.0

    print('=== SAC (error=0.0 checkpoint), full sweep @ 30dBi/75W ===')
    sac0_rate, sac0_rate_std, sac0_power, sac0_power_std = sweep_sac(SAC_AOD0_TRAINING_NAME, 'SAC err=0.0')

    print('=== SAC (error=0.05 checkpoint), full sweep @ 30dBi/75W ===')
    sac05_rate, sac05_rate_std, sac05_power, sac05_power_std = sweep_sac(SAC_AOD05_TRAINING_NAME, 'SAC err=0.05')

    matched_power_watt = float(sac0_power[0])  # SAC(error=0.0)'s own measured power AT error=0
    print(f'\nMatched-power target (SAC err=0.0 @ error=0): {matched_power_watt:.2f} W')

    print('=== MMSE (full 75W budget), full sweep @ 30dBi ===')
    mmse_full_rate, mmse_full_rate_std = sweep_mmse(75.0, 'MMSE (75W)')

    print(f'=== MMSE (matched {matched_power_watt:.2f}W), full sweep @ 30dBi ===')
    mmse_matched_rate, mmse_matched_rate_std = sweep_mmse(matched_power_watt, 'MMSE (matched)')

    # persist raw results for reproducibility / future replotting
    out_path = Path(cfg_check.output_metrics_path, OUT_GZIP_DIR)
    out_path.mkdir(parents=True, exist_ok=True)
    with gzip.open(Path(out_path, 'parajuli_3gpp_sweep.gzip'), 'wb') as f:
        pickle.dump({
            'error_sweep_range': ERROR_SWEEP_RANGE,
            'sat_gain_dBi': 30.0,
            'power_constraint_watt': 75.0,
            'matched_power_watt': matched_power_watt,
            'sac_aod0': {'mean_rate': sac0_rate, 'std_rate': sac0_rate_std, 'mean_power': sac0_power, 'std_power': sac0_power_std},
            'sac_aod05': {'mean_rate': sac05_rate, 'std_rate': sac05_rate_std, 'mean_power': sac05_power, 'std_power': sac05_power_std},
            'mmse_full': {'mean_rate': mmse_full_rate, 'std_rate': mmse_full_rate_std},
            'mmse_matched': {'mean_rate': mmse_matched_rate, 'std_rate': mmse_matched_rate_std},
        }, f)
    print(f'Saved: {Path(out_path, "parajuli_3gpp_sweep.gzip")}')

    # ---- plot: wide figure, small legend (per prior request) ----
    plot_cfg = PlotConfig()
    matplotlib.rcParams['text.usetex'] = False
    plot_width = 2.4 * plot_cfg.textwidth
    plot_height = plot_width * 0.42

    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    curves = [
        (mmse_full_rate, mmse_full_rate_std, 'MMSE (75W)', plot_cfg.cp2['gold'], 'v', ':'),
        (mmse_matched_rate, mmse_matched_rate_std, f'MMSE ({matched_power_watt:.1f}W, matched to SAC err=0.0)', '#8a7000', '^', '-.'),
        (sac0_rate, sac0_rate_std, 'SAC (error=0.0)', '#307b3b', 'o', '-'),
        (sac05_rate, sac05_rate_std, 'SAC (error=0.05)', '#1baf7a', 's', '--'),
    ]
    for mean, std, label, color, marker, linestyle in curves:
        ax.errorbar(ERROR_SWEEP_RANGE, mean, yerr=std, marker=marker, color=color, linestyle=linestyle, label=label)
        ax.plot(ERROR_SWEEP_RANGE, mean, marker=marker, color=color, linestyle=linestyle, fillstyle='none')

    ax.set_xlabel('Error Bound')
    ax.set_ylabel('Rate R [bps/Hz]')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(
        [c[2] for c in curves], loc='upper right', ncols=1, fontsize=7,
        framealpha=1.0, frameon=True, handlelength=1.5, labelspacing=0.3, borderpad=0.4,
    )
    plt.tight_layout(pad=0.2)

    pdf_out = Path(plot_cfg.plots_parent_path, 'pdf', 'parajuli.pdf')
    plt.savefig(pdf_out, bbox_inches='tight', dpi=800, transparent=True)
    print(f'Saved: {pdf_out}')

    print('\n=== Summary (30 dBi / 75 W system) ===')
    print(f'SAC (error=0.0 checkpoint) power @ error=0.0:  {sac0_power[0]:.2f} W ({100*sac0_power[0]/75:.1f}% of 75W budget), rate {sac0_rate[0]:.4f} bps/Hz')
    print(f'SAC (error=0.05 checkpoint) power @ error=0.05: {sac05_power[5]:.2f} W ({100*sac05_power[5]/75:.1f}% of 75W budget), rate {sac05_rate[5]:.4f} bps/Hz')
