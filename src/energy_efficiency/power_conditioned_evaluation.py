"""
Offline EE-optimal-power search for the power-conditioned checkpoint (job
155697, 2026-07-31): EE_power_conditioned_aod0.05_N16K3_satg30_p75_eta0.6_
rawpow_pcond1to75log_uniform.

Background: this checkpoint was trained with the power budget fed to the
actor as an extra state input (sampled log-uniform in [1, 75] W per
transition), the precoder always rescaled to exactly that sampled budget,
and a PURE-RATE reward (no power term at all) -- see EE_sac.py's
'energy_efficiency_power_conditioned' reward mode. The idea (per the
2026-07-31 handoff) is that this learns the rate-maximizing frontier
r*(H, P) for every budget in one network; the EE-optimal P is then found by
a cheap OFFLINE grid search over the converged, non-training network,
instead of a live Dinkelbach-style multiplier. This is the entire point of
the power-conditioned approach -- without this script, the trained network
alone doesn't answer the EE question.

Three stages, all at CSIT error=0.05 (the training-time error bound) unless
noted:
  1. Grid search over P in [1, 75] W (log-spaced): for each P, sample fresh
     channels, call the network with P fed as the state's power feature,
     rescale the raw output to EXACTLY P (matching training's always-
     rescale mechanism), measure mean rate, compute
     EE(P) = mean_rate(P) / (P / pa_efficiency + circuit_power) -- the same
     EE definition ee_power_scaling_diagnostic.py uses. argmax over the
     grid gives P*.
  2. Rate-vs-CSIT-error sweep (0-0.10, 11 pts) with the budget FIXED at P*,
     for direct comparison against every other checkpoint's error curve
     (same format as rate_power_detlam.py/rate_power_tanneal.py).
  3. Matched-power MMSE at P* across the same error sweep -- the same
     "does SAC sit close to what MMSE achieves at the SAME power" sanity
     check used for detlam/tanneal, but here the power is dictated by this
     checkpoint's OWN found optimum instead of an empirically measured one.
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
from src.data.satellite_manager import SatelliteManager
from src.data.user_manager import UserManager
from src.models.precoders.learned_precoder import get_learned_precoder_normalized
from src.utils.get_precoding import get_precoding_mmse
from src.utils.load_model import load_model
from src.utils.update_sim import update_sim

TRAINING_NAME = 'EE_power_conditioned_aod0.05_N16K3_satg30_p75_eta0.6_rawpow_pcond1to75log_uniform'

# must match the training-time defaults exactly (EE_sac.py's
# EE_POWER_COND_MIN_WATT/_MAX_WATT/_SAMPLING, all left at default in job
# 155697's slurm -- see that file for the confirming env-var printout).
PCOND_MIN_WATT = 1.0
PCOND_MAX_WATT = 75.0
PCOND_SAMPLING = 'log_uniform'

CSIT_ERROR_BOUND = 0.05  # training-time error bound
P_GRID = np.geomspace(PCOND_MIN_WATT, PCOND_MAX_WATT, 25)
GRID_MC_ITERATIONS = 3000

error_sweep_range = np.linspace(0, 0.10, 11)
SWEEP_MC_ITERATIONS = 10000

PREFIX_GZIP_REL = ('EE_3gpp_pilots_rate_power_sweep', 'rate_power_3gpp_pilots.gzip')
DETLAM_GZIP_REL = ('EE_detlam_rate_power_sweep', 'rate_power_detlam.gzip')
TANNEAL_GZIP_REL = ('EE_tanneal_rate_power_sweep', 'rate_power_tanneal.gzip')


def make_config():
    for var in ['EE_SAT_GAIN_DBI', 'EE_POWER_BUDGET_WATT', 'EE_TARGET_ELEVATION_DEG']:
        os.environ.pop(var, None)
    from src.config.config import Config
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


def normalize_power_cond_watt(watt: float) -> float:
    """Exact mirror of EE_sac.py's normalize_power_cond_watt -- must match
    training so the power feature the network sees at eval time lands in
    the same normalized range it was trained on."""
    if PCOND_SAMPLING == 'log_uniform':
        log_min, log_max = np.log(PCOND_MIN_WATT), np.log(PCOND_MAX_WATT)
        log_mid, log_half_range = (log_min + log_max) / 2, (log_max - log_min) / 2
        return (np.log(watt) - log_mid) / log_half_range
    mid = (PCOND_MIN_WATT + PCOND_MAX_WATT) / 2
    half_range = (PCOND_MAX_WATT - PCOND_MIN_WATT) / 2
    return (watt - mid) / half_range


def get_precoding_power_conditioned(config, user_manager, satellite_manager, norm_factors, precoder_network, power_watt):
    """Power-conditioned counterpart to get_precoding_learned: appends the
    normalized target-power feature to the state (mirroring EE_sac.py's
    append_power_cond during training), then always-rescales the raw
    network output to EXACTLY power_watt -- get_learned_precoder_normalized
    already implements that always-rescale-to-budget logic (Scheme I's),
    just called here with a variable budget instead of the config
    constant."""
    state = config.config_learner.get_state(
        config=config,
        user_manager=user_manager,
        satellite_manager=satellite_manager,
        norm_factors=norm_factors,
        **config.config_learner.get_state_args
    )
    state = np.concatenate([state, [normalize_power_cond_watt(power_watt)]]).astype('float32')
    return get_learned_precoder_normalized(
        state=state,
        precoder_network=precoder_network,
        sat_nr=config.sat_nr,
        sat_ant_nr=config.sat_ant_nr,
        user_nr=config.user_nr,
        power_constraint_watt=power_watt,
    )


def run_grid_search(cfg, precoder_network, norm_factors):
    satellite_manager = SatelliteManager(config=cfg)
    user_manager = UserManager(config=cfg)

    error_param = 'additive_error_on_cosine_of_aod'
    cfg.config_error_model.error_rng_parametrizations[error_param]['args']['low'] = -CSIT_ERROR_BOUND
    cfg.config_error_model.error_rng_parametrizations[error_param]['args']['high'] = CSIT_ERROR_BOUND

    circuit_power = cfg.sat_nr * cfg.sat_ant_nr * cfg.circuit_power_watt

    mean_rate = np.zeros(len(P_GRID))
    std_rate = np.zeros(len(P_GRID))
    mean_achieved_power = np.zeros(len(P_GRID))
    ee = np.zeros(len(P_GRID))

    print(f'\n=== Grid search: {len(P_GRID)} power points in [{PCOND_MIN_WATT}, {PCOND_MAX_WATT}] W, '
          f'{GRID_MC_ITERATIONS} MC samples each, CSIT error={CSIT_ERROR_BOUND} ===')

    for p_idx, p_watt in enumerate(P_GRID):
        rate_samples = np.zeros(GRID_MC_ITERATIONS)
        power_samples = np.zeros(GRID_MC_ITERATIONS)

        for iter_idx in range(GRID_MC_ITERATIONS):
            update_sim(cfg, satellite_manager, user_manager)
            w_precoder = get_precoding_power_conditioned(
                cfg, user_manager, satellite_manager, norm_factors, precoder_network, p_watt,
            )
            rate_samples[iter_idx] = calc_sum_rate(
                channel_state=satellite_manager.channel_state_information,
                w_precoder=w_precoder,
                noise_power_watt=cfg.noise_power_watt,
            )
            power_samples[iter_idx] = np.real(np.trace(np.matmul(w_precoder.conj().T, w_precoder)))

        mean_rate[p_idx] = rate_samples.mean()
        std_rate[p_idx] = rate_samples.std()
        mean_achieved_power[p_idx] = power_samples.mean()
        total_dc_power = p_watt / cfg.pa_efficiency + circuit_power
        ee[p_idx] = mean_rate[p_idx] / total_dc_power

        print(f'  P={p_watt:6.2f} W: rate={mean_rate[p_idx]:.4f} bps/Hz, '
              f'achieved power={mean_achieved_power[p_idx]:.3f} W (should == P), '
              f'EE={ee[p_idx]:.5f} bps/Hz/W_DC')

    peak_idx = int(np.argmax(ee))
    p_star = float(P_GRID[peak_idx])
    print(f'\nEE peak at P*={p_star:.2f} W (grid point {peak_idx}/{len(P_GRID)}), '
          f'rate*={mean_rate[peak_idx]:.4f} bps/Hz, EE*={ee[peak_idx]:.5f} bps/Hz/W_DC')

    return {
        'p_grid': P_GRID.copy(),
        'mean_rate': mean_rate, 'std_rate': std_rate,
        'mean_achieved_power': mean_achieved_power,
        'ee': ee,
        'p_star': p_star,
        'peak_idx': peak_idx,
    }


def run_error_sweep_at_fixed_power(cfg, precoder_network, norm_factors, p_star):
    satellite_manager = SatelliteManager(config=cfg)
    user_manager = UserManager(config=cfg)

    error_param = 'additive_error_on_cosine_of_aod'
    initial_error_config = cfg.config_error_model.error_rng_parametrizations[error_param]['args'].copy()

    mean_rate = np.zeros(len(error_sweep_range))
    std_rate = np.zeros(len(error_sweep_range))

    print(f'\n=== Rate-vs-error sweep at fixed P*={p_star:.2f} W ===')
    for error_idx, error_value in enumerate(error_sweep_range):
        cfg.config_error_model.error_rng_parametrizations[error_param]['args']['low'] = -error_value
        cfg.config_error_model.error_rng_parametrizations[error_param]['args']['high'] = error_value

        rate_samples = np.zeros(SWEEP_MC_ITERATIONS)
        for iter_idx in range(SWEEP_MC_ITERATIONS):
            update_sim(cfg, satellite_manager, user_manager)
            w_precoder = get_precoding_power_conditioned(
                cfg, user_manager, satellite_manager, norm_factors, precoder_network, p_star,
            )
            rate_samples[iter_idx] = calc_sum_rate(
                channel_state=satellite_manager.channel_state_information,
                w_precoder=w_precoder,
                noise_power_watt=cfg.noise_power_watt,
            )
        mean_rate[error_idx] = rate_samples.mean()
        std_rate[error_idx] = rate_samples.std()
        print(f'  error={error_value:.2f}: rate={mean_rate[error_idx]:.4f} bps/Hz')

    cfg.config_error_model.error_rng_parametrizations[error_param]['args'] = initial_error_config
    return {'mean_rate': mean_rate, 'std_rate': std_rate, 'power': p_star}


def run_matched_power_mmse_sweep(cfg, p_star):
    satellite_manager = SatelliteManager(config=cfg)
    user_manager = UserManager(config=cfg)

    error_param = 'additive_error_on_cosine_of_aod'
    initial_error_config = cfg.config_error_model.error_rng_parametrizations[error_param]['args'].copy()

    mean_rate = np.zeros(len(error_sweep_range))
    std_rate = np.zeros(len(error_sweep_range))

    print(f'\n=== Matched-power MMSE sweep at P*={p_star:.2f} W ===')
    for error_idx, error_value in enumerate(error_sweep_range):
        cfg.config_error_model.error_rng_parametrizations[error_param]['args']['low'] = -error_value
        cfg.config_error_model.error_rng_parametrizations[error_param]['args']['high'] = error_value

        rate_samples = np.zeros(SWEEP_MC_ITERATIONS)
        for iter_idx in range(SWEEP_MC_ITERATIONS):
            update_sim(cfg, satellite_manager, user_manager)
            w_mmse = get_precoding_mmse(cfg, user_manager, satellite_manager)
            current_power = np.real(np.trace(np.matmul(w_mmse.conj().T, w_mmse)))
            w_mmse_matched = w_mmse * np.sqrt(p_star / current_power)
            rate_samples[iter_idx] = calc_sum_rate(
                channel_state=satellite_manager.channel_state_information,
                w_precoder=w_mmse_matched,
                noise_power_watt=cfg.noise_power_watt,
            )
        mean_rate[error_idx] = rate_samples.mean()
        std_rate[error_idx] = rate_samples.std()
        print(f'  error={error_value:.2f}: MMSE(matched to P*)={mean_rate[error_idx]:.4f} bps/Hz')

    cfg.config_error_model.error_rng_parametrizations[error_param]['args'] = initial_error_config
    return {'mean_rate': mean_rate, 'std_rate': std_rate, 'power': p_star}


if __name__ == '__main__':
    cfg = make_config()
    cfg.config_learner.training_name = TRAINING_NAME
    model_path = get_best_model_path(cfg.trained_models_path, TRAINING_NAME)
    print(f'[power_conditioned] checkpoint: {model_path}')

    precoder_network, norm_factors = load_model(model_path)
    if norm_factors != {}:
        cfg.config_learner.get_state_args['norm_state'] = True

    grid_results = run_grid_search(cfg, precoder_network, norm_factors)
    p_star = grid_results['p_star']

    sac_sweep = run_error_sweep_at_fixed_power(cfg, precoder_network, norm_factors, p_star)
    mmse_sweep = run_matched_power_mmse_sweep(cfg, p_star)

    gap_pct = 100 * (sac_sweep['mean_rate'] - mmse_sweep['mean_rate']) / mmse_sweep['mean_rate']
    print('\n=== Power-conditioned SAC (at its own grid-search-optimal P*) vs. matched-power MMSE ===')
    for i, e in enumerate(error_sweep_range):
        print(f'  error={e:.2f}: SAC={sac_sweep["mean_rate"][i]:.3f}  '
              f'MMSE(matched power)={mmse_sweep["mean_rate"][i]:.3f}  gap={gap_pct[i]:+.1f}%')

    # ---- comparison against detlam checkpoints' own operating points -----
    detlam_gzip = Path(cfg.output_metrics_path, *DETLAM_GZIP_REL)
    if detlam_gzip.exists():
        with gzip.open(detlam_gzip, 'rb') as file:
            detlam_data = pickle.load(file)
        err_idx_005 = int(np.argmin(np.abs(detlam_data['error_sweep_range'] - CSIT_ERROR_BOUND)))
        print(f'\n=== Reference: detlam/pre-fix operating points at error={CSIT_ERROR_BOUND:.2f} '
              f'(power-conditioned P*={p_star:.2f} W, rate={grid_results["mean_rate"][grid_results["peak_idx"]]:.3f}) ===')
        for key, data in detlam_data['results'].items():
            print(f"  {data['label']:35s} power={data['mean_power'][err_idx_005]:6.2f} W  "
                  f"rate={data['mean_rate'][err_idx_005]:.3f}")

    out_path = Path(cfg.output_metrics_path, 'EE_power_conditioned_evaluation')
    out_path.mkdir(parents=True, exist_ok=True)
    gzip_path = Path(out_path, 'power_conditioned_evaluation.gzip')
    with gzip.open(gzip_path, 'wb') as file:
        pickle.dump({
            'grid_results': grid_results,
            'error_sweep_range': error_sweep_range,
            'sac_sweep_at_p_star': sac_sweep,
            'mmse_matched_sweep_at_p_star': mmse_sweep,
        }, file=file)
    print(f'\nSaved: {gzip_path}')

    # ---- plots -------------------------------------------------------------
    plot_cfg = PlotConfig()
    matplotlib.rcParams['text.usetex'] = False  # PlotConfig() resets this on construction

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    ax.errorbar(grid_results['p_grid'], grid_results['mean_rate'], yerr=grid_results['std_rate'],
                marker='o', color='#254796', ecolor='#254796', elinewidth=1, capsize=3,
                linewidth=1.5, markersize=5)
    ax.axvline(p_star, color='#d03b3b', linestyle='--', linewidth=1.2, label=f'P*={p_star:.2f} W')
    ax.set_xscale('log')
    ax.set_xlabel('Power budget P [W]')
    ax.set_ylabel('Rate r*(P) [bps/Hz]')
    ax.set_title(f'Learned rate frontier (error={CSIT_ERROR_BOUND})', fontsize=11)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc='best')

    ax2 = axes[1]
    ax2.plot(grid_results['p_grid'], grid_results['ee'], marker='o', markersize=5, color='#307b3b')
    ax2.axvline(p_star, color='#d03b3b', linestyle='--', linewidth=1.2, label=f'P*={p_star:.2f} W')
    ax2.set_xscale('log')
    ax2.set_xlabel('Power budget P [W]')
    ax2.set_ylabel('EE(P) [bps/Hz per W DC]')
    ax2.set_title('Offline EE grid search', fontsize=11)
    ax2.grid(True, alpha=0.25, linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=8, loc='best')

    ax3 = axes[2]
    ax3.errorbar(error_sweep_range, sac_sweep['mean_rate'], yerr=sac_sweep['std_rate'], marker='o',
                 color='#d03b3b', ecolor='#d03b3b', elinewidth=1, capsize=3, linewidth=1.5,
                 markersize=5, label=f'SAC power-conditioned (P*={p_star:.2f} W)')
    ax3.errorbar(error_sweep_range, mmse_sweep['mean_rate'], yerr=mmse_sweep['std_rate'], marker='x',
                 color='#caa023', ecolor='#caa023', elinewidth=1, capsize=3, linewidth=1.5,
                 markersize=5, label='MMSE (matched to P*)')
    ax3.set_xlabel('Error Bound')
    ax3.set_ylabel('Rate R [bps/Hz]')
    ax3.set_title(f'Rate vs. CSIT error at fixed P*={p_star:.2f} W', fontsize=11)
    ax3.grid(True, alpha=0.25, linewidth=0.5)
    ax3.set_axisbelow(True)
    ax3.legend(fontsize=7, loc='best')

    fig.suptitle('Power-conditioned checkpoint (2026-07-31): offline EE-optimal-power search', fontsize=12, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    pdf_path = Path(plot_cfg.plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)
    out = Path(pdf_path, 'power_conditioned_evaluation.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=300, transparent=True)
    print(f'Saved: {out}')
    plt.close(fig)
