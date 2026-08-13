"""Empirically demonstrate the non-monotonicity claim from 04_approach.tex
(paragraph after eq:eeobjective): that EE(W) = R(W)/P_total(W) rises while
noise-limited and falls once the marginal rate bought by another watt drops
below P_total's marginal cost, so its maximizer sits strictly below the
power budget rather than on it.

Method: take the MMSE precoding *direction* at a fixed CSIT error bound (a
fixed direction per realization, matching the "scaling a fixed precoding
direction toward the budget" argument in the text), rescale it to a sweep
of fixed transmit-power levels within the power budget, and measure mean
sum rate and resulting EE at each level.

CSIT error bound is a CLI arg (--error X, default 0.0 = perfect CSIT) so
the power/EE tradeoff can be isolated from the CSIT-robustness story
(error=0.0) or measured together with it (e.g. error=0.05).

Saves outputs/metrics/EE_vs_transmit_power/ee_vs_power_sweep_error{X}.gzip
and reports/figures/pdf/ee_vs_transmit_power_error{X}.pdf.
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
from src.utils.get_precoding import get_precoding_mmse
from src.utils.update_sim import update_sim

CSIT_ERROR_BOUND = float(sys.argv[sys.argv.index('--error') + 1]) if '--error' in sys.argv else 0.0

monte_carlo_iterations = 10000
# capped at the actual power budget: no precoder in this system can ever
# transmit above it (MMSE always-rescales exactly to it, SAC's clip-only
# projection can't exceed it either), so sweeping past the budget would
# plot a region no real precoder can reach. The rise-then-fall shape and
# its maximizer both already sit comfortably inside [0, budget].
power_sweep_watt = np.linspace(1, 75, 40)


def total_power_watt(cfg, transmit_power_watt):
    """Same formula as eq:ptotal / EE_sac.py's Dinkelbach branch."""
    return transmit_power_watt / cfg.pa_efficiency + cfg.sat_nr * cfg.sat_ant_nr * cfg.circuit_power_watt


def run_ee_vs_power_sweep(cfg):
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
            w_mmse = get_precoding_mmse(cfg, user_manager, satellite_manager)
            current_power = np.real(np.trace(np.matmul(w_mmse.conj().T, w_mmse)))
            w_precoder = w_mmse * np.sqrt(target_power_watt / current_power)
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


if __name__ == '__main__':
    cfg = Config()
    cfg.show_plots = False
    plot_cfg = PlotConfig()
    matplotlib.rcParams['text.usetex'] = False  # override PlotConfig's reset

    print(f'[system] sat_gain_dBi={cfg.sat_gain_dBi}, budget={cfg.power_constraint_watt} W, '
          f'pa_efficiency={cfg.pa_efficiency}, circuit_power_watt={cfg.circuit_power_watt}, '
          f'csit_error_bound={CSIT_ERROR_BOUND}')

    mean_rate, std_rate = run_ee_vs_power_sweep(cfg)
    total_power = np.array([total_power_watt(cfg, p) for p in power_sweep_watt])
    ee = mean_rate / total_power

    out_path = Path(cfg.output_metrics_path, 'EE_vs_transmit_power')
    out_path.mkdir(parents=True, exist_ok=True)
    gzip_path = Path(out_path, f'ee_vs_power_sweep_error{CSIT_ERROR_BOUND:g}.gzip')
    with gzip.open(gzip_path, 'wb') as file:
        pickle.dump({
            'power_sweep_watt': power_sweep_watt,
            'mean_rate': mean_rate,
            'std_rate': std_rate,
            'total_power_watt': total_power,
            'ee': ee,
            'power_budget': cfg.power_constraint_watt,
            'csit_error_bound': CSIT_ERROR_BOUND,
        }, file=file)
    print(f'Saved: {gzip_path}')

    argmax_idx = int(np.argmax(ee))
    print(f'EE maximizer: transmit_power={power_sweep_watt[argmax_idx]:.2f} W '
          f'(budget={cfg.power_constraint_watt:.0f} W), EE={ee[argmax_idx]:.5f} bps/Hz/W')

    pdf_path = Path(plot_cfg.plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)

    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.62

    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    ax.plot(power_sweep_watt, ee, color=plot_cfg.cp2['blue'], marker='o', markersize=4, linewidth=1.5)
    ax.axvline(cfg.power_constraint_watt, color=plot_cfg.cp2['gold'], linestyle='--', linewidth=1.3,
               label=f'power budget ({cfg.power_constraint_watt:.0f} W)')
    ax.axvline(power_sweep_watt[argmax_idx], color=plot_cfg.cp2['magenta'], linestyle='--', linewidth=1.3,
               label=f'EE maximizer ({power_sweep_watt[argmax_idx]:.1f} W)')
    ax.set_xlabel('Fixed transmit power [W]')
    ax.set_ylabel('EE [bps/Hz/W]')
    ax.set_title(f'MMSE direction, CSIT error bound={CSIT_ERROR_BOUND:g}', fontsize=9)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout(pad=0.4)

    out = Path(pdf_path, f'ee_vs_transmit_power_error{CSIT_ERROR_BOUND:g}.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)
    print(f'Saved: {out}')
