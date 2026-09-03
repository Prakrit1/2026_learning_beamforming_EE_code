"""SAC counterpart to ee_vs_transmit_power_sweep.py: same method (take a
fixed precoding *direction* per channel realization, rescale it to a sweep
of fixed transmit-power levels spanning [1, budget] W, measure resulting
rate and EE = R/P_total), but using the SAC actor network's own raw
(pre-clip) direction instead of MMSE's.

CSIT error bound is a CLI arg (--error X, default 0.0), matching
ee_vs_transmit_power_sweep.py, so the two are directly comparable at the
same error bound. Note this checkpoint was trained at error bound 0.0, so
evaluating it at error>0 (e.g. --error 0.05) tests generalization, same as
every other error-sweep evaluation in this project.

Checkpoint: the canonical aod=0.0 run (job 156358),
EE_dinkelbach_adaptive_lwin5000_N16K3_satg30_p75_eta0.6_rawpow -- same
checkpoint plotting_scenario.py and power_savings_bars_triplet.py use, via
plotting_scenario.py's CHECKPOINTS mapping, so all three figures tell a
consistent story about the same trained network (previously pointed at the
lambda-init variant, job 1175, which the handoff confirms converges to
essentially the same policy -- but using a genuinely different checkpoint
than the other two figures was worth fixing regardless).

Saves outputs/metrics/EE_vs_transmit_power/ee_vs_power_sweep_sac_error{X}.gzip
and reports/figures/pdf/ee_vs_transmit_power_sac_error{X}.pdf.
"""
import sys
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

TRAINING_NAME = CHECKPOINTS['aod0.0']
CSIT_ERROR_BOUND = float(sys.argv[sys.argv.index('--error') + 1]) if '--error' in sys.argv else 0.0
# --hide-maximizer: for the paper figure, omit the per-direction EE-maximizer
# marker/legend entry (still printed to console) so the plot only shows the
# EE-vs-power shape without a number that would sit alongside the deployed
# policy's actual, higher operating power reported elsewhere in the paper.
HIDE_MAXIMIZER = '--hide-maximizer' in sys.argv

monte_carlo_iterations = 10000
power_sweep_watt = np.linspace(1, 75, 40)  # capped at the actual budget, see ee_vs_transmit_power_sweep.py


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


PLOT_ONLY = '--plot-only' in sys.argv

if __name__ == '__main__':
    cfg = Config()
    cfg.show_plots = False
    plot_cfg = PlotConfig()
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
        mean_rate, total_power, ee = cached['mean_rate'], cached['total_power_watt'], cached['ee']
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

    argmax_idx = int(np.argmax(ee))
    print(f'SAC EE maximizer: transmit_power={power_sweep_watt[argmax_idx]:.2f} W '
          f'(budget={cfg.power_constraint_watt:.0f} W), EE={ee[argmax_idx]:.5f} bps/Hz/W, '
          f'rate={mean_rate[argmax_idx]:.4f} bps/Hz')

    # Where the DEPLOYED policy actually sits: it doesn't transmit a fixed
    # power on every channel like the sweep above does -- it picks power
    # adaptively per realization (clip-only projection), and this is the
    # population mean of that adaptive choice. Pulled from the real
    # rate_power_triplet.gzip data (same checkpoint, same error bound), not
    # hardcoded, so this stays correct if the checkpoint is retrained.
    # Plotting both points on the same axes is what makes the "fixed vs.
    # adaptive power" distinction legible instead of looking like two
    # figures disagree with each other.
    deployed_point = None
    triplet_gzip = Path(cfg.output_metrics_path, 'EE_lwin5000_3gpp_triplet', 'rate_power_triplet.gzip')
    if triplet_gzip.exists():
        with gzip.open(triplet_gzip, 'rb') as file:
            triplet_data = pickle.load(file)
        error_idx = int(np.argmin(np.abs(triplet_data['error_sweep_range'] - CSIT_ERROR_BOUND)))
        deployed_power = triplet_data['results']['sac_aod0.0']['mean_power'][error_idx]
        deployed_rate = triplet_data['results']['sac_aod0.0']['mean_rate'][error_idx]
        deployed_total_power = total_power_watt(cfg, deployed_power)
        deployed_ee = deployed_rate / deployed_total_power
        deployed_point = (deployed_power, deployed_ee)
        print(f'Deployed policy (adaptive power, mean): transmit_power={deployed_power:.2f} W, '
              f'EE={deployed_ee:.5f} bps/Hz/W, rate={deployed_rate:.4f} bps/Hz')

    pdf_path = Path(plot_cfg.plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)

    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.62

    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    # green for EE matches its color everywhere else this checkpoint
    # (aod=0.0) appears -- plotting_scenario.py's error-sweep figure and
    # power_savings_bars_triplet.py's bar chart.
    ax.plot(power_sweep_watt, ee, color=plot_cfg.cp2['green'], marker='o', markersize=4, linewidth=1.5,
            label='EE')
    ax.axvline(cfg.power_constraint_watt, color='gray', linestyle=':', linewidth=1.2,
               label=r'$P_{\mathrm{rad}}$')
    if not HIDE_MAXIMIZER:
        # $P^\star$: the single fixed power that maximizes EE on this sweep
        # (a deterministic sweep location, not a mean over anything -- no
        # bar). Value shown inline, matching power_savings_bars_plot.py's
        # "name, $P=XX$ W" convention (there: 'EE, $P=35$ W' etc.) so this
        # figure's legend reads the same way as that one. Full explanation
        # of why this differs from $\bar P_{\mathrm{EE}}$ belongs in the
        # caption, not the legend.
        ax.axvline(power_sweep_watt[argmax_idx], color='gray', linestyle='-.', linewidth=1.3,
                   label=fr'$P^\star \approx {power_sweep_watt[argmax_idx]:.0f}$ W')
    if deployed_point is not None:
        # $\bar P_{\mathrm{EE}}$: mean transmit power of the deployed EE
        # policy (subscript ties it to the same policy the green 'EE' curve
        # sweeps, and to the 'EE' bar in power_savings_bars_plot.py; bar
        # accent marks it as a population mean over adaptive per-channel
        # choices, unlike the single fixed value $P^\star$). Distinct
        # marker/color (gold, matches "RM"/deployed-policy styling
        # elsewhere) -- this is NOT a point on the constant-power curve,
        # it's the deployed policy's own adaptive-power operating point,
        # shown on the same axes on purpose so the "12 W vs 35 W"
        # difference reads as two different quantities instead of a
        # contradiction between figures. Full explanation belongs in the
        # caption, not the legend.
        ax.plot(*deployed_point, color=plot_cfg.cp2['gold'], marker='*', markersize=12,
                linestyle='none', zorder=5,
                label=fr'$\bar P_{{\mathrm{{EE}}}} \approx {deployed_point[0]:.0f}$ W')
    # $P$ matches \precoderpower in the paper's macros (99_custommacros.sty),
    # the same bare-P symbol already used in the P^star / P_bar_EE legend
    # entries above.
    ax.set_xlabel(r'Transmit power $P$, held constant across channels [W]', fontsize=13)
    ax.set_ylabel('EE [bps/Hz/W]', fontsize=13)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    # Legend above the axes (not overlapping the curve/star) -- title text
    # moves to the caption (paper convention), freeing this space.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.08),
               ncol=len(labels), fontsize=11, frameon=False, columnspacing=1.4,
               handletextpad=0.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    name_suffix = '_paper' if HIDE_MAXIMIZER else ''
    out = Path(pdf_path, f'ee_vs_transmit_power_sac_error{CSIT_ERROR_BOUND:g}{name_suffix}.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=300, transparent=True)
    print(f'Saved: {out}')

    jpg_path = Path(plot_cfg.plots_parent_path, 'jpg')
    jpg_path.mkdir(parents=True, exist_ok=True)
    out_jpg = Path(jpg_path, f'ee_vs_transmit_power_sac_error{CSIT_ERROR_BOUND:g}{name_suffix}.jpg')
    fig.savefig(out_jpg, bbox_inches='tight', dpi=200)
    print(f'Saved: {out_jpg}')

    png_path = Path(plot_cfg.plots_parent_path, 'png')
    png_path.mkdir(parents=True, exist_ok=True)
    out_png = Path(png_path, f'ee_vs_transmit_power_sac_error{CSIT_ERROR_BOUND:g}{name_suffix}.png')
    fig.savefig(out_png, bbox_inches='tight', dpi=200, transparent=True)
    print(f'Saved: {out_png}')

    plt.close(fig)
