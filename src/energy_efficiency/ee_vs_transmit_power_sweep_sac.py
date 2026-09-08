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
Dual-axis energy-efficiency-vs-transmit-power figure for the deployed SAC
policy (checkpoint 'aod0.0').

The single physical object is the constant-power sweep: the policy's raw
(un-normalized) precoder is rescaled to each fixed transmit power P across
the whole budget range, giving

    rate(P)   -- sum rate at that power (right axis, saturating), and
    EE(P) = rate(P) / P_total(P)   -- energy efficiency (left axis, concave),

where P_total(P) = P / eta_PA + N_ant * P_circuit.

Two operating points are marked ON these curves:
  - proposed (~35 W): where the deployed clip-only policy actually operates
    (mean transmit power read from rate_power_triplet.gzip at Delta-eps=0),
  - full power (75 W): the no-back-off baseline (same policy rescaled to the
    full budget) -- the right end of the sweep.

Reading both curves at those two powers shows the trade-off honestly: going
35 W -> 75 W buys little extra rate (the rate curve has saturated) while
lowering EE, so the ~40 % EE gain is a real move along the rate/power curve,
not an artifact of comparing two differently-labelled bars. It also answers
"why not push to the EE peak (~12 W)?" -- the rate curve there has collapsed.

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

    # ---- dual-axis figure -------------------------------------------------
    ee_color = plot_cfg.cp2['green']
    rate_color = plot_cfg.cp2['blue']
    prop_color = plot_cfg.cp2['magenta']
    full_color = plot_cfg.cp2['gold']

    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.6  # match the other draft figures' aspect

    fig, ax_ee = plt.subplots(figsize=(plot_width, plot_height))
    ax_rate = ax_ee.twinx()

    # faint guides at the two operating powers so both curves are readable there
    for P in (P_prop, P_full):
        ax_ee.axvline(P, color='0.7', linestyle=':', linewidth=1.0, zorder=1)

    # rate on the right axis (dashed, drawn behind), EE on the left (solid)
    line_rate, = ax_rate.plot(power_sweep_watt, mean_rate, color=rate_color,
                              linestyle='--', linewidth=1.8, label='Sum rate', zorder=2)
    line_ee, = ax_ee.plot(power_sweep_watt, ee, color=ee_color,
                          linewidth=2.0, label='Energy efficiency', zorder=3)

    # operating-point markers on the EE curve
    ax_ee.scatter([P_prop], [ee_prop_curve], marker='*', s=200, color=prop_color,
                  edgecolor='black', linewidth=0.6, zorder=5)
    ax_ee.scatter([P_full], [ee_full], marker='o', s=80, color=full_color,
                  edgecolor='black', linewidth=0.6, zorder=5)

    ax_ee.annotate(f'proposed\n{P_prop:.0f} W', xy=(P_prop, ee_prop_curve),
                   xytext=(P_prop + 3.5, ee_prop_curve * 0.72),
                   fontsize=10, color=prop_color, ha='left', va='top',
                   arrowprops=dict(arrowstyle='-', color=prop_color, lw=0.8))
    ax_ee.annotate(f'full power\n{P_full:.0f} W', xy=(P_full, ee_full),
                   xytext=(P_full - 3.5, ee_full * 1.28),
                   fontsize=10, color=full_color, ha='right', va='bottom',
                   arrowprops=dict(arrowstyle='-', color=full_color, lw=0.8))

    # headline gain, placed near the proposed marker
    ax_ee.text(P_prop + 3.5, ee_prop_curve * 1.06,
               f'+{ee_gain_pct:.0f}% EE\n(-{rate_loss_pct:.0f}% rate)',
               fontsize=9.5, color='black', ha='left', va='bottom')

    ax_ee.set_xlabel(r'Transmit power $P_{\mathrm{tx}}$ [W]', fontsize=13)
    ax_ee.set_ylabel('Energy efficiency [bits/s/Hz/W]', fontsize=13, color=ee_color)
    ax_rate.set_ylabel('Sum rate [bits/s/Hz]', fontsize=13, color=rate_color)
    ax_ee.tick_params(axis='y', labelcolor=ee_color)
    ax_rate.tick_params(axis='y', labelcolor=rate_color)

    ax_ee.set_xlim(0, 78)
    ax_ee.set_ylim(0, float(np.max(ee)) * 1.18)
    ax_rate.set_ylim(0, float(np.max(mean_rate)) * 1.12)
    ax_ee.grid(True, axis='y', alpha=0.25, linewidth=0.5)
    ax_ee.set_axisbelow(True)

    fig.legend(handles=[line_ee, line_rate], loc='upper center',
               bbox_to_anchor=(0.5, 1.06), ncol=2, fontsize=11,
               frameon=False, columnspacing=1.6, handletextpad=0.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    for subdir, dpi, transparent in [('pdf', 300, True), ('jpg', 200, False), ('png', 200, True)]:
        target = Path(plot_cfg.plots_parent_path, subdir)
        target.mkdir(parents=True, exist_ok=True)
        out = Path(target, f'ee_vs_transmit_power_sac_error{CSIT_ERROR_BOUND:g}.{subdir}')
        fig.savefig(out, bbox_inches='tight', dpi=dpi, transparent=transparent)
        print(f'Saved: {out}')

    plt.close(fig)
