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
Energy-efficiency-vs-transmit-power figure comparing the deployed EE policy
('aod0.0' Dinkelbach checkpoint) against the genuinely-separate rate-only (RM)
baseline (SAC_rateonly, EE_REWARD_MODE=sum_rate_only): left axis = energy
efficiency (two swept EE curves), right (twin) axis = the EE policy's achieved
sum rate vs transmit power.

For each policy the constant-power sweep rescales its raw (un-normalized)
precoder to each fixed transmit power P across the budget range, giving
rate(P), and

    EE(P) = rate(P) / P_total(P)   -- pays only the power it uses; rises,
                                       peaks, then falls,

with P_total(P) = P / eta_PA + N_ant * P_circuit. Sweeping BOTH policies over
the same power grid isolates beamforming quality from the power operating
point: at a common transmit power the two curves differ only in how well each
policy's learned precoding direction serves the users. The EE policy's real
deployed operating point (~35 W, read from the cached triplet) is marked on
its curve, and the RM policy is marked at that same ~35 W so the two can be
read off at equal power (RM's own native operating point is the 75 W budget,
i.e. its curve's right endpoint).

If the RM checkpoint has not been synced into models/ yet, the script falls
back to the previous behaviour: RM as a single flat reference line at
rate_EE(75 W)/P_tot(75 W) (a placeholder reusing the EE checkpoint's
full-power inference), and prints a warning.

Run fresh (sbatch, GPU) to (re)compute the sweeps, or with --plot-only to
replot from the cached gzip. Saves reports/figures/{pdf,jpg,png}/
ee_vs_transmit_power_sac_error{X}.*
"""

EE_TRAINING_NAME = CHECKPOINTS['aod0.0']
# Genuine rate-only (RM) baseline -- EE_REWARD_MODE=sum_rate_only, same 3GPP
# Set-1 system params (see SAC_rateonly_satg30_p75_nadir.slurm). Add to
# models/ (or CHECKPOINTS) once the checkpoint is synced off the cluster.
RM_TRAINING_NAME = 'SAC_rateonly_N16K3_satg30_p75_eta0.6_rawpow'

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


def load_and_run_sweep(cfg, training_name):
    """Load a checkpoint by training_name and run the constant-power sweep.

    Returns (mean_rate, std_rate, model_path). Raises FileNotFoundError (from
    get_best_model_path) if the checkpoint is not present under models/."""
    cfg.config_learner.training_name = training_name
    model_path = get_best_model_path(cfg.trained_models_path, training_name)
    print(f'[ee_vs_transmit_power_sweep_sac] checkpoint: {model_path}, csit_error_bound={CSIT_ERROR_BOUND}')

    precoder_network, norm_factors = load_model(model_path)
    # per-checkpoint norm state -- an RM checkpoint trained without state
    # normalization must not inherit the EE checkpoint's norm setting
    cfg.config_learner.get_state_args['norm_state'] = (norm_factors != {})

    mean_rate, std_rate = run_ee_vs_power_sweep_sac(cfg, norm_factors, precoder_network)
    return mean_rate, std_rate, model_path


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

    out_path = Path(cfg.output_metrics_path, 'EE_vs_transmit_power')
    out_path.mkdir(parents=True, exist_ok=True)
    gzip_path = Path(out_path, f'ee_vs_power_sweep_sac_error{CSIT_ERROR_BOUND:g}.gzip')

    if PLOT_ONLY and gzip_path.exists():
        with gzip.open(gzip_path, 'rb') as file:
            cached = pickle.load(file)
        mean_rate = cached['mean_rate']
        total_power = cached['total_power_watt']
        ee = cached['ee']
        rm_available = cached.get('rm_available', False)
        rm_mean_rate = cached.get('rm_mean_rate')
        rm_ee = cached.get('rm_ee')
        print(f'[plot-only] loaded cached sweep: {gzip_path} (rm_available={rm_available})')
    else:
        # ---- EE policy sweep ---------------------------------------------
        mean_rate, std_rate, ee_model_path = load_and_run_sweep(cfg, EE_TRAINING_NAME)
        total_power = np.array([total_power_watt(cfg, p) for p in power_sweep_watt])
        ee = mean_rate / total_power

        # ---- RM (rate-only) policy sweep, if the checkpoint is present ----
        try:
            rm_mean_rate, rm_std_rate, rm_model_path = load_and_run_sweep(cfg, RM_TRAINING_NAME)
            rm_ee = rm_mean_rate / total_power
            rm_available = True
        except FileNotFoundError:
            print(f'[warn] RM checkpoint {RM_TRAINING_NAME!r} not found under '
                  f'{cfg.trained_models_path} -- falling back to the flat '
                  f'EE-at-75W placeholder line for RM. Sync the SAC_rateonly '
                  f'checkpoint into models/ and rerun to get the genuine RM curve.')
            rm_mean_rate = None
            rm_std_rate = None
            rm_ee = None
            rm_model_path = None
            rm_available = False

        with gzip.open(gzip_path, 'wb') as file:
            pickle.dump({
                'power_sweep_watt': power_sweep_watt,
                'mean_rate': mean_rate,
                'std_rate': std_rate,
                'total_power_watt': total_power,
                'ee': ee,
                'rm_available': rm_available,
                'rm_mean_rate': rm_mean_rate,
                'rm_std_rate': rm_std_rate,
                'rm_ee': rm_ee,
                'power_budget': cfg.power_constraint_watt,
                'ee_training_name': EE_TRAINING_NAME,
                'rm_training_name': RM_TRAINING_NAME if rm_available else None,
                'checkpoint': str(ee_model_path),
                'rm_checkpoint': str(rm_model_path) if rm_available else None,
                'csit_error_bound': CSIT_ERROR_BOUND,
            }, file=file)
        print(f'Saved: {gzip_path}')

    # ---- operating points -------------------------------------------------
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

    # ---- figure -----------------------------------------------------------
    prop_color = plot_cfg.cp2['green']
    rm_color = plot_cfg.cp2['gold']

    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.6  # match the other draft figures' aspect

    fig, ax = plt.subplots(figsize=(plot_width, plot_height))

    # legend in the trained^ / evaluated-P notation of the other figures
    trained_watt = int(round(cfg.power_constraint_watt))
    ee_eval_watt = int(round(P_prop))
    rm_full_watt = int(round(P_full))

    handles = []

    # EE(P): swept green curve, marked at the deployed ~35 W operating point
    line_prop, = ax.plot(power_sweep_watt, ee, color=prop_color, linewidth=2.0,
                         label=rf'EE$^{{{trained_watt}}}$, $P={ee_eval_watt}$ W', zorder=3)
    handles.append(line_prop)

    ee_curve_max = float(np.max(ee))

    if rm_available:
        # RM(P): genuine rate-only policy, swept the same way. RM's native
        # operating point is the 75 W budget (curve's right endpoint); we also
        # mark it at the EE policy's ~35 W so the two read off at equal power.
        rm_ee_at_prop = float(np.interp(P_prop, power_sweep_watt, rm_ee))
        line_rm, = ax.plot(power_sweep_watt, rm_ee, color=rm_color, linewidth=2.0,
                           linestyle='-',
                           label=rf'RM$^{{{trained_watt}}}$, $P={ee_eval_watt}$ W', zorder=2)
        handles.append(line_rm)

        # RM operating-point marker at the common ~35 W (open square)
        ax.scatter([P_prop], [rm_ee_at_prop], marker='s', s=42, facecolor='white',
                   edgecolor=rm_color, linewidth=1.4, zorder=5)
        # horizontal projection of RM's 35 W value onto the y-axis
        ax.plot([0, P_prop], [rm_ee_at_prop, rm_ee_at_prop], color='0.5', linestyle='--',
                linewidth=1.0, zorder=1)

        y_top = max(ee_curve_max, float(np.max(rm_ee)))
        print(f'at P={P_prop:.0f} W: EE={ee_prop_curve:.5f} vs RM={rm_ee_at_prop:.5f} bps/Hz/W '
              f'(EE +{100 * (ee_prop_curve / rm_ee_at_prop - 1):.0f}% over RM at equal power)')
    else:
        # placeholder: RM as a single flat reference line at its 75 W EE
        rm_ee_const = ee_full  # = rate_EE(75 W) / P_tot(75 W)
        line_rm = ax.axhline(rm_ee_const, color=rm_color, linestyle='--', linewidth=2.0,
                             label=rf'RM$^{{{trained_watt}}}$, $P={rm_full_watt}$ W', zorder=2)
        handles.append(line_rm)
        y_top = ee_curve_max
        print(f'[placeholder RM] at P={P_prop:.0f} W: EE={ee_prop_curve:.5f} vs '
              f'RM(flat)={rm_ee_const:.5f} bps/Hz/W')

    # EE energy-efficient operating point: open circle with dashed projections
    ax.plot([P_prop, P_prop], [0, ee_prop_curve], color='0.5', linestyle='--',
            linewidth=1.0, zorder=1)
    ax.plot([0, P_prop], [ee_prop_curve, ee_prop_curve], color='0.5', linestyle='--',
            linewidth=1.0, zorder=1)
    ax.scatter([P_prop], [ee_prop_curve], marker='o', s=45, facecolor='white',
               edgecolor=prop_color, linewidth=1.4, zorder=5)

    # ---- secondary right axis: achieved sum rate vs transmit power ---------
    # The EE policy's rate(P) rises and saturates while EE(P) peaks (~12 W) then
    # falls; the twin axis makes the rate/efficiency trade-off explicit and
    # shows the ~35 W operating point keeps most of the rate at high efficiency.
    ax_rate = ax.twinx()
    rate_color = plot_cfg.cp2['blue']
    line_rate, = ax_rate.plot(power_sweep_watt, mean_rate, color=rate_color,
                              linewidth=2.0, linestyle=(0, (5, 2)),
                              label='Sum rate', zorder=2)
    ax_rate.scatter([P_prop], [rate_prop], marker='D', s=38, facecolor='white',
                    edgecolor=rate_color, linewidth=1.4, zorder=5)
    ax_rate.set_ylabel('Sum rate [bits/s/Hz]', fontsize=13)
    ax_rate.set_ylim(0, float(np.max(mean_rate)) * 1.12)
    handles.append(line_rate)

    ax.set_xlabel(r'Transmit power $P_{\mathrm{tx}}$ [W]', fontsize=13)
    ax.set_ylabel('Energy efficiency [bits/s/Hz/W]', fontsize=13)
    ax.set_xlim(0, 78)
    ax.set_ylim(0, y_top * 1.18)
    ax.grid(True, axis='y', alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    ax.legend(handles=handles, loc='upper right', fontsize=11, frameon=False)
    fig.tight_layout()

    for subdir, dpi, transparent in [('pdf', 300, True), ('jpg', 200, False), ('png', 200, True)]:
        target = Path(plot_cfg.plots_parent_path, subdir)
        target.mkdir(parents=True, exist_ok=True)
        out = Path(target, f'ee_vs_transmit_power_sac_error{CSIT_ERROR_BOUND:g}.{subdir}')
        fig.savefig(out, bbox_inches='tight', dpi=dpi, transparent=transparent)
        print(f'Saved: {out}')

    plt.close(fig)
