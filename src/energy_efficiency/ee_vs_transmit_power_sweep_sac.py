import os
import sys

# Guard against leftover shell env vars from other ablation runs (elevation/
# gain/budget sweeps) silently changing this system's configuration -- the
# sweep must be on the same system as the deployed-policy operating point it
# is plotted against (plotting_scenario.py / power_savings_bars_triplet.py
# apply the same guard).
os.environ.pop('EE_SAT_GAIN_DBI', None)
os.environ.pop('EE_POWER_BUDGET_WATT', None)
os.environ.pop('EE_TARGET_ELEVATION_DEG', None)

import gzip
import pickle
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt

from src.config.config import Config
from src.config.config_plotting import PlotConfig
from src.data.calc_sum_rate import calc_sum_rate
from src.data.satellite_manager import SatelliteManager
from src.data.user_manager import UserManager
from src.utils.get_precoding import get_precoding_learned_no_norm
from src.utils.load_model import load_model
from src.utils.update_sim import update_sim
from src.energy_efficiency.plotting_scenario import CHECKPOINTS, get_best_model_path

"""
Energy-efficiency-vs-transmit-power figure for the deployed SAC policy
(checkpoint 'aod0.0'): single EE axis, two curves.

The constant-power sweep rescales the policy's raw (un-normalized) precoder
to each fixed transmit power P across the budget range, giving rate(P), and

    proposed  EE(P) = rate(P) / P_total(P)   -- pays only the power it uses;
                                                rises, peaks, then falls,

with P_total(P) = P / eta_PA + N_ant * P_circuit. RM always transmits at the
full 75 W budget, so it is a single operating point, not a curve: its EE is
the constant rate(75 W) / P_total(75 W), drawn as a horizontal reference line.
The proposed curve exceeds RM's flat line across the whole back-off region and
meets it exactly at P = 75 W -- i.e. backing off and paying only for the power
actually used is more efficient than always paying the full 75 W budget.

Run fresh (sbatch) to (re)compute the sweep, or with --plot-only to replot
from the cached gzip. Saves reports/figures/{pdf,jpg,png}/
ee_vs_transmit_power_sac_error{X}.*
"""

TRAINING_NAME = CHECKPOINTS['aod0.0']
CSIT_ERROR_BOUND = float(sys.argv[sys.argv.index('--error') + 1]) if '--error' in sys.argv else 0.0
PLOT_ONLY = '--plot-only' in sys.argv

monte_carlo_iterations = 10000
power_sweep_watt = np.linspace(1, 75, 40)  # capped at the actual budget


def total_power_watt(cfg, transmit_power_watt):
    return transmit_power_watt / cfg.pa_efficiency + cfg.sat_nr * cfg.sat_ant_nr * cfg.circuit_power_watt


def run_ee_vs_power_sweep_sac(cfg, norm_factors, precoder_network):
    satellite_manager = SatelliteManager(config=cfg)
    user_manager = UserManager(config=cfg)

    error_param = 'additive_error_on_cosine_of_aod'
    cfg.config_error_model.error_rng_parametrizations[error_param]['args']['low'] = -CSIT_ERROR_BOUND
    cfg.config_error_model.error_rng_parametrizations[error_param]['args']['high'] = CSIT_ERROR_BOUND

    mean_rate = np.zeros(len(power_sweep_watt))
    std_rate = np.zeros(len(power_sweep_watt))

    for power_idx, target_power_watt in enumerate(power_sweep_watt):
        rate_samples = np.zeros(monte_carlo_iterations)

        for iter_idx in range(monte_carlo_iterations):
            update_sim(cfg, satellite_manager, user_manager)
            w_sac_raw = get_precoding_learned_no_norm(cfg, user_manager, satellite_manager, norm_factors, precoder_network)
            current_power = np.real(np.trace(np.matmul(w_sac_raw.conj().T, w_sac_raw)))
            w_precoder = w_sac_raw * np.sqrt(target_power_watt / current_power)
            rate_samples[iter_idx] = calc_sum_rate(
                channel_state=satellite_manager.channel_state_information,
                w_precoder=w_precoder,
                noise_power_watt=cfg.noise_power_watt,
            )

        mean_rate[power_idx] = rate_samples.mean()
        std_rate[power_idx] = rate_samples.std()
        ee = mean_rate[power_idx] / total_power_watt(cfg, target_power_watt)
        print(f'transmit_power={target_power_watt:6.2f} W: '
              f'rate={mean_rate[power_idx]:.4f} bps/Hz, EE={ee:.5f} bps/Hz/W')

    return mean_rate, std_rate


def deployed_operating_point(cfg):
    """The clip-only policy's actual (mean transmit power, rate, EE) at
    Delta-eps = CSIT_ERROR_BOUND, straight from the cached triplet -- no new
    simulation. Returns None if the triplet has not been generated yet."""
    triplet_gzip = Path(cfg.output_metrics_path, 'EE_lwin5000_3gpp_triplet', 'rate_power_triplet.gzip')
    if not triplet_gzip.exists():
        return None
    with gzip.open(triplet_gzip, 'rb') as file:
        triplet_data = pickle.load(file)
    idx = int(np.argmin(np.abs(triplet_data['error_sweep_range'] - CSIT_ERROR_BOUND)))
    p_tx = float(triplet_data['results']['sac_aod0.0']['mean_power'][idx])
    rate = float(triplet_data['results']['sac_aod0.0']['mean_rate'][idx])
    ee = rate / total_power_watt(cfg, p_tx)
    return p_tx, rate, ee


if __name__ == '__main__':
    cfg = Config()
    cfg.show_plots = False
    plot_cfg = PlotConfig()
    # PlotConfig() flips text.usetex back on; the compute nodes have no latex
    # binary, so re-assert it off or savefig crashes silently under sbatch.
    matplotlib.rcParams['text.usetex'] = False

    cfg.config_learner.training_name = TRAINING_NAME
    model_path = get_best_model_path(cfg.trained_models_path, TRAINING_NAME)
    print(f'[ee_vs_transmit_power_sweep_sac] checkpoint: {model_path}, csit_error_bound={CSIT_ERROR_BOUND}')

    out_path = Path(cfg.output_metrics_path, 'EE_vs_transmit_power')
    out_path.mkdir(parents=True, exist_ok=True)
    gzip_path = Path(out_path, f'ee_vs_power_sweep_sac_error{CSIT_ERROR_BOUND:g}.gzip')

    if PLOT_ONLY and gzip_path.exists():
        with gzip.open(gzip_path, 'rb') as file:
            cached = pickle.load(file)
        mean_rate = cached['mean_rate']
        total_power = cached['total_power_watt']
        ee = cached['ee']
        print(f'[plot-only] loaded cached sweep: {gzip_path}')
    else:
        precoder_network, norm_factors = load_model(model_path)
        if norm_factors != {}:
            cfg.config_learner.get_state_args['norm_state'] = True

        mean_rate, std_rate = run_ee_vs_power_sweep_sac(cfg, norm_factors, precoder_network)
        total_power = np.array([total_power_watt(cfg, p) for p in power_sweep_watt])
        ee = mean_rate / total_power

        with gzip.open(gzip_path, 'wb') as file:
            pickle.dump({
                'power_sweep_watt': power_sweep_watt,
                'mean_rate': mean_rate,
                'std_rate': std_rate,
                'total_power_watt': total_power,
                'ee': ee,
                'power_budget': cfg.power_constraint_watt,
                'training_name': TRAINING_NAME,
                'checkpoint': str(model_path),
                'csit_error_bound': CSIT_ERROR_BOUND,
            }, file=file)
        print(f'Saved: {gzip_path}')

    # ---- operating points (both lie on the swept curves) ------------------
    P_full = float(power_sweep_watt[-1])            # 75 W, no back-off baseline
    ee_full = float(ee[-1])
    rate_full = float(mean_rate[-1])

    op = deployed_operating_point(cfg)
    if op is not None:
        P_prop, rate_prop_adaptive, ee_prop = op
        # read the sweep curves at the deployed power so the markers sit on
        # the plotted lines (adaptive vs constant-power differ only slightly)
        rate_prop = float(np.interp(P_prop, power_sweep_watt, mean_rate))
        ee_prop_curve = float(np.interp(P_prop, power_sweep_watt, ee))
    else:
        # fall back to the sweep's EE-maximiser if the triplet is unavailable
        P_prop = float(power_sweep_watt[int(np.argmax(ee))])
        rate_prop = float(np.interp(P_prop, power_sweep_watt, mean_rate))
        ee_prop_curve = float(np.max(ee))
        ee_prop = ee_prop_curve
        print('[warn] triplet not found -- marking the sweep EE-maximiser instead of the deployed point')

    ee_gain_pct = 100.0 * (ee_prop_curve / ee_full - 1.0)
    rate_loss_pct = 100.0 * (1.0 - rate_prop / rate_full)
    print(f'proposed  P={P_prop:.1f} W: EE={ee_prop_curve:.5f} bps/Hz/W, rate={rate_prop:.4f} bps/Hz')
    print(f'full power P={P_full:.1f} W: EE={ee_full:.5f} bps/Hz/W, rate={rate_full:.4f} bps/Hz')
    print(f'-> +{ee_gain_pct:.1f}% EE for -{rate_loss_pct:.1f}% rate by backing off to {P_prop:.0f} W')

    # ---- single EE axis: proposed curve vs RM's fixed operating point -----
    # RM always transmits at the full 75 W budget, so it is a SINGLE operating
    # point, not a curve: its EE is the constant rate(75 W)/P_tot(75 W), drawn
    # as a horizontal reference line. The proposed policy can trade power for
    # efficiency along its curve; it exceeds RM's flat line across the whole
    # back-off region and meets it exactly at P = 75 W.
    rm_ee_const = ee_full  # = rate(75 W) / P_tot(75 W)
    print(f'at operating point P={P_prop:.0f} W: proposed EE={ee_prop_curve:.5f} vs '
          f'RM EE={rm_ee_const:.5f} bps/Hz/W (+{100 * (ee_prop_curve / rm_ee_const - 1):.0f}%)')

    prop_color = plot_cfg.cp2['green']
    rm_color = plot_cfg.cp2['gold']

    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.6  # match the other draft figures' aspect

    fig, ax = plt.subplots(figsize=(plot_width, plot_height))

    line_prop, = ax.plot(power_sweep_watt, ee, color=prop_color, linewidth=2.0,
                         label=r'Proposed:  $R(P)/P_{\mathrm{tot}}(P)$', zorder=3)
    line_rm = ax.axhline(rm_ee_const, color=rm_color, linestyle='--', linewidth=2.0,
                         label=r'RM (fixed 75 W):  $R(75)/P_{\mathrm{tot}}(75)$', zorder=2)

    # rate labels make "stays in the high-rate regime" explicit: the proposed
    # point keeps most of RM's rate while operating at far lower power.
    rate_retained_pct = 100.0 * rate_prop / rate_full

    # deployed operating point on the proposed curve
    ax.axvline(P_prop, color='0.7', linestyle=':', linewidth=1.0, zorder=1)
    ax.scatter([P_prop], [ee_prop_curve], marker='*', s=200, color=plot_cfg.cp2['magenta'],
               edgecolor='black', linewidth=0.6, zorder=5)
    ax.annotate(f'proposed: {P_prop:.0f} W\n{rate_prop:.1f} bits/s/Hz'
                f' ({rate_retained_pct:.0f}% of RM)',
                xy=(P_prop, ee_prop_curve), xytext=(P_prop + 3.5, ee_prop_curve * 0.60),
                fontsize=9.5, ha='left', va='top',
                arrowprops=dict(arrowstyle='-', color='0.4', lw=0.8))

    # RM operating point (full power), where the proposed curve meets the line
    ax.scatter([P_full], [ee_full], marker='o', s=55, facecolor='white',
               edgecolor='black', linewidth=1.0, zorder=5)
    ax.annotate(f'RM: {P_full:.0f} W\n{rate_full:.1f} bits/s/Hz',
                xy=(P_full, ee_full), xytext=(P_full - 3, ee_full + 0.012),
                fontsize=9.5, ha='right', va='bottom',
                arrowprops=dict(arrowstyle='-', color='0.4', lw=0.8))

    ax.set_xlabel(r'Transmit power $P_{\mathrm{tx}}$ [W]', fontsize=13)
    ax.set_ylabel('Energy efficiency [bits/s/Hz/W]', fontsize=13)
    ax.set_xlim(0, 78)
    ax.set_ylim(0, float(np.max(ee)) * 1.18)
    ax.grid(True, axis='y', alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    ax.legend(loc='upper right', fontsize=10, frameon=False)
    fig.tight_layout()

    for subdir, dpi, transparent in [('pdf', 300, True), ('jpg', 200, False), ('png', 200, True)]:
        target = Path(plot_cfg.plots_parent_path, subdir)
        target.mkdir(parents=True, exist_ok=True)
        out = Path(target, f'ee_vs_transmit_power_sac_error{CSIT_ERROR_BOUND:g}.{subdir}')
        fig.savefig(out, bbox_inches='tight', dpi=dpi, transparent=transparent)
        print(f'Saved: {out}')

    plt.close(fig)
