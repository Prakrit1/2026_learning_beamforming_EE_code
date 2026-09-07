
import os
import sys

# Guard against leftover shell env vars from other ablation runs (elevation/
# gain/budget sweeps) silently changing this system's configuration -- see
# f743f6b's fix to the sibling ee_vs_transmit_power_sweep_sac.py.
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
from src.data.calc_tx_power_distribution import calc_tx_power_distribution
from src.data.satellite_manager import SatelliteManager
from src.data.user_manager import UserManager
from src.utils.get_precoding import get_precoding_learned_clip_only
from src.utils.load_model import load_model
from src.utils.update_sim import update_sim
from src.energy_efficiency.plotting_scenario import CHECKPOINTS, get_best_model_path

"""
Companion to ee_vs_transmit_power_sweep_sac.py, answering a different
question. That script forces every channel realization to the exact same
fixed transmit power P (no adaptation), which is why its EE peaks at a low
~12 W: circuit power dominates once no per-channel adaptation is allowed.

Here we instead keep the trained (Delta-eps=0.0) checkpoint's *adaptive*
clip-only projection (get_precoding_learned_clip_only) -- which only clips
the network's raw output down when it would exceed the available budget,
otherwise passes it through unchanged -- and sweep that budget itself from
1 W up to the training-time 75 W.

Mechanically this is NOT expected to reproduce a "rises to 35 W then falls"
curve: get_precoding_learned_clip_only never pushes power UP toward the
budget, it only clips DOWN when the raw output exceeds it. So once the
budget exceeds wherever this checkpoint's raw output naturally
concentrates, clipping stops engaging and mean power/rate/EE all plateau
-- more budget headroom doesn't make the network spend more, because
nothing forces it to. Expect rise-then-plateau, not rise-then-fall; see
the chat discussion this script came out of for the reasoning. Still a
meaningful result: it shows where the policy's own adaptive operating
point saturates, versus the fixed-power case's rigid 12 W ceiling.

Saves outputs/metrics/EE_vs_transmit_power/ee_vs_power_budget_sweep_sac_error{X}.gzip
Saves reports/figures/{pdf,jpg,png}/ee_vs_power_budget_sweep_sac_error{X}.*
"""

TRAINING_NAME = CHECKPOINTS['aod0.0']
CSIT_ERROR_BOUND = float(sys.argv[sys.argv.index('--error') + 1]) if '--error' in sys.argv else 0.0

monte_carlo_iterations = 10000
budget_sweep_watt = np.linspace(1, 75, 40)  # 75 W = the training-time budget, same checkpoint as elsewhere


def total_power_watt(cfg, transmit_power_watt):
    return transmit_power_watt / cfg.pa_efficiency + cfg.sat_nr * cfg.sat_ant_nr * cfg.circuit_power_watt


def run_ee_vs_power_budget_sweep_sac(cfg, norm_factors, precoder_network):
    satellite_manager = SatelliteManager(config=cfg)
    user_manager = UserManager(config=cfg)

    error_param = 'additive_error_on_cosine_of_aod'
    cfg.config_error_model.error_rng_parametrizations[error_param]['args']['low'] = -CSIT_ERROR_BOUND
    cfg.config_error_model.error_rng_parametrizations[error_param]['args']['high'] = CSIT_ERROR_BOUND

    mean_rate = np.zeros(len(budget_sweep_watt))
    std_rate = np.zeros(len(budget_sweep_watt))
    mean_power = np.zeros(len(budget_sweep_watt))
    std_power = np.zeros(len(budget_sweep_watt))

    for budget_idx, budget_watt in enumerate(budget_sweep_watt):
        # Clip-only projection reads its clipping threshold from both of
        # these -- override so the checkpoint's raw output is clipped to
        # *this* sweep point, not the training-time 75 W budget.
        cfg.power_constraint_watt = budget_watt
        cfg.learned_precoder_args['power_constraint_watt'] = budget_watt

        rate_samples = np.zeros(monte_carlo_iterations)
        power_samples = np.zeros(monte_carlo_iterations)

        for iter_idx in range(monte_carlo_iterations):
            update_sim(cfg, satellite_manager, user_manager)
            w_precoder = get_precoding_learned_clip_only(cfg, user_manager, satellite_manager, norm_factors, precoder_network)
            rate_samples[iter_idx] = calc_sum_rate(
                channel_state=satellite_manager.channel_state_information,
                w_precoder=w_precoder,
                noise_power_watt=cfg.noise_power_watt,
            )
            power_samples[iter_idx] = calc_tx_power_distribution(w_precoder=w_precoder).sum()

        mean_rate[budget_idx] = rate_samples.mean()
        std_rate[budget_idx] = rate_samples.std()
        mean_power[budget_idx] = power_samples.mean()
        std_power[budget_idx] = power_samples.std()
        ee = mean_rate[budget_idx] / total_power_watt(cfg, mean_power[budget_idx])
        print(f'budget={budget_watt:6.2f} W: mean_power={mean_power[budget_idx]:.2f} W '
              f'({100 * mean_power[budget_idx] / budget_watt:.1f}% of that budget), '
              f'rate={mean_rate[budget_idx]:.4f} bps/Hz, EE={ee:.5f} bps/Hz/W')

    return mean_rate, std_rate, mean_power, std_power


PLOT_ONLY = '--plot-only' in sys.argv

if __name__ == '__main__':
    cfg = Config()
    cfg.show_plots = False
    plot_cfg = PlotConfig()
    matplotlib.rcParams['text.usetex'] = False

    cfg.config_learner.training_name = TRAINING_NAME
    model_path = get_best_model_path(cfg.trained_models_path, TRAINING_NAME)
    print(f'[ee_vs_power_budget_sweep_sac] checkpoint: {model_path}, csit_error_bound={CSIT_ERROR_BOUND}')

    out_path = Path(cfg.output_metrics_path, 'EE_vs_transmit_power')
    out_path.mkdir(parents=True, exist_ok=True)
    gzip_path = Path(out_path, f'ee_vs_power_budget_sweep_sac_error{CSIT_ERROR_BOUND:g}.gzip')

    if PLOT_ONLY and gzip_path.exists():
        with gzip.open(gzip_path, 'rb') as file:
            cached = pickle.load(file)
        mean_rate, mean_power, total_power, ee = (
            cached['mean_rate'], cached['mean_power'], cached['total_power_watt'], cached['ee']
        )
    else:
        precoder_network, norm_factors = load_model(model_path)
        if norm_factors != {}:
            cfg.config_learner.get_state_args['norm_state'] = True

        mean_rate, std_rate, mean_power, std_power = run_ee_vs_power_budget_sweep_sac(cfg, norm_factors, precoder_network)
        total_power = np.array([total_power_watt(cfg, p) for p in mean_power])
        ee = mean_rate / total_power

        with gzip.open(gzip_path, 'wb') as file:
            pickle.dump({
                'budget_sweep_watt': budget_sweep_watt,
                'mean_rate': mean_rate,
                'std_rate': std_rate,
                'mean_power': mean_power,
                'std_power': std_power,
                'total_power_watt': total_power,
                'ee': ee,
                'training_name': TRAINING_NAME,
                'checkpoint': str(model_path),
                'csit_error_bound': CSIT_ERROR_BOUND,
            }, file=file)
        print(f'Saved: {gzip_path}')

    argmax_idx = int(np.argmax(ee))
    print(f'SAC adaptive-clip EE maximizer: budget={budget_sweep_watt[argmax_idx]:.2f} W, '
          f'achieved mean_power={mean_power[argmax_idx]:.2f} W, EE={ee[argmax_idx]:.5f} bps/Hz/W, '
          f'rate={mean_rate[argmax_idx]:.4f} bps/Hz')
    print(f'At the full training-time budget (75 W): achieved mean_power={mean_power[-1]:.2f} W, '
          f'EE={ee[-1]:.5f} bps/Hz/W, rate={mean_rate[-1]:.4f} bps/Hz')

    pdf_path = Path(plot_cfg.plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)

    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.62

    fig, ax = plt.subplots(figsize=(plot_width, plot_height))

    ax.plot(budget_sweep_watt, ee, color=plot_cfg.cp2['green'], marker='o', markersize=4, linewidth=1.5,
            label='EE (adaptive, clip-only)')
    ax.axvline(budget_sweep_watt[argmax_idx], color='gray', linestyle='-.', linewidth=1.3,
               label=fr'$B^\star \approx {budget_sweep_watt[argmax_idx]:.0f}$ W')
    ax.plot(budget_sweep_watt[-1], ee[-1], color=plot_cfg.cp2['gold'], marker='*', markersize=12,
            linestyle='none', zorder=5,
            label=fr'Deployed (75 W budget), $\bar P \approx {mean_power[-1]:.0f}$ W')

    ax.set_xlabel(r'Available power budget $B$ [W]', fontsize=13)
    ax.set_ylabel('EE [bps/Hz/W]', fontsize=13)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.08),
               ncol=len(labels), fontsize=11, frameon=False, columnspacing=1.4,
               handletextpad=0.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    out = Path(pdf_path, f'ee_vs_power_budget_sweep_sac_error{CSIT_ERROR_BOUND:g}.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=300, transparent=True)
    print(f'Saved: {out}')

    jpg_path = Path(plot_cfg.plots_parent_path, 'jpg')
    jpg_path.mkdir(parents=True, exist_ok=True)
    out_jpg = Path(jpg_path, f'ee_vs_power_budget_sweep_sac_error{CSIT_ERROR_BOUND:g}.jpg')
    fig.savefig(out_jpg, bbox_inches='tight', dpi=200)
    print(f'Saved: {out_jpg}')

    png_path = Path(plot_cfg.plots_parent_path, 'png')
    png_path.mkdir(parents=True, exist_ok=True)
    out_png = Path(png_path, f'ee_vs_power_budget_sweep_sac_error{CSIT_ERROR_BOUND:g}.png')
    fig.savefig(out_png, bbox_inches='tight', dpi=200, transparent=True)
    print(f'Saved: {out_png}')

    plt.close(fig)
