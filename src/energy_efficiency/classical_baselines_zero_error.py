"""
MMSE, MRT ("mrc" in this codebase's file naming, but the actual formula is
Maximum Ratio Transmission), and Zero-Forcing, all at zero CSIT error, under
the paper-matching user distance scenario (config.py's default: 100km mean,
+-50km roam -- NOT overridden, see handoff_prompt_EE_evaluation.txt's
"Distance-scenario bug" section for why that distinction matters).

Power for all three is EXACTLY the full budget by mathematical construction
(all three "_normalized" precoder functions rescale to power_constraint_watt
exactly -- see norm_precoder()/the sqrt(power_factors_users) scaling in
mrc_precoder_normalized and regularized_zero_forcing_precoder_user_specific_normalized).
No Monte Carlo needed to know this; it's algebraically guaranteed, not
measured. Only RATE varies with the channel realization and needs real
Monte Carlo sampling -- that's where these three classical precoders
actually differ from each other and from SAC.

Produces:
1. An extended version of the lay-friendly power bar chart
   (power_savings_bars_classical_vs_sac.pdf) -- MMSE/MRT/ZF all shown
   pinned at 100% of budget (no savings, by design), directly next to the
   two SAC checkpoints' real savings, for a stark visual contrast.
2. A rate comparison (printed + a simple bar chart) across MMSE, MRT, ZF,
   and the two SAC checkpoints (rate numbers for the SAC checkpoints reused
   from the already-computed error-sweep gzips, not re-simulated).
3. A power-matched MMSE baseline: MMSE re-run with
   cfg.mmse_args['power_constraint_watt'] overridden to 56 W -- the actual
   power SAC (error=0.0 checkpoint) uses at zero CSIT error (see
   rate_power_error_sweep.py's results). mmse_precoder_normalized rescales
   exactly to whatever power_constraint_watt it's given (algebraic
   guarantee, same as the full-budget case), so this uses exactly 56 W, not
   100 W -- a fair, power-matched comparison against SAC rather than
   comparing MMSE's full-budget rate to SAC's reduced-power rate.
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
from src.data.satellite_manager import SatelliteManager
from src.data.user_manager import UserManager
from src.utils.get_precoding import get_precoding_mmse, get_precoding_mrc, get_precoding_zero_forcing
from src.utils.update_sim import update_sim
from src.energy_efficiency.tx_power_distribution import plot_power_savings_bars

MONTE_CARLO_ITERATIONS = 10000


def run_classical_baseline(cfg, get_precoder_func, label):
    satellite_manager = SatelliteManager(config=cfg)
    user_manager = UserManager(config=cfg)

    rate_samples = np.zeros(MONTE_CARLO_ITERATIONS)
    for iter_idx in range(MONTE_CARLO_ITERATIONS):
        update_sim(cfg, satellite_manager, user_manager)
        w_precoder = get_precoder_func(cfg, user_manager, satellite_manager)
        rate_samples[iter_idx] = calc_sum_rate(
            channel_state=satellite_manager.channel_state_information,
            w_precoder=w_precoder,
            noise_power_watt=cfg.noise_power_watt,
        )
        if iter_idx % 2000 == 0:
            print(f'[{label}] {iter_idx}/{MONTE_CARLO_ITERATIONS}')
    return rate_samples


def load_sac_zero_error_rate(output_metrics_path, training_name):
    path = Path(output_metrics_path, training_name, 'error_sweep', 'testing_learned_sweep_0.0_0.1.gzip')
    with gzip.open(path, 'rb') as file:
        error_sweep_range, metrics = pickle.load(file)
    key = [k for k in metrics.keys() if 'calc_sum_rate' in str(k)][0]
    return metrics[key]['mean'][0], metrics[key]['std'][0]  # index 0 == error 0.00


if __name__ == '__main__':
    cfg = Config()
    cfg.show_plots = False
    power_budget = cfg.power_constraint_watt

    classical_precoders = [
        (get_precoding_mmse, 'MMSE'),
        (get_precoding_mrc, 'MRT'),
        (get_precoding_zero_forcing, 'ZF'),
    ]

    classical_rate_means = {}
    classical_rate_stds = {}
    for get_precoder_func, label in classical_precoders:
        rate_samples = run_classical_baseline(cfg, get_precoder_func, label)
        classical_rate_means[label] = rate_samples.mean()
        classical_rate_stds[label] = rate_samples.std()
        print(f'{label}: mean rate = {rate_samples.mean():.4f} bps/Hz (std {rate_samples.std():.4f}), '
              f'power = {power_budget:.1f} W (100% of budget, by construction)')

    # power-matched MMSE: same power SAC (error=0.0) actually uses (56 W),
    # not the full 100 W budget. Override the copied-by-value dict entry
    # directly -- cfg.power_constraint_watt itself is a plain float already
    # baked into cfg.mmse_args at Config() construction time, so reassigning
    # cfg.power_constraint_watt alone would NOT propagate here.
    sac_matched_power_watt = 56.0
    cfg.mmse_args['power_constraint_watt'] = sac_matched_power_watt
    mmse_56w_rate_samples = run_classical_baseline(cfg, get_precoding_mmse, 'MMSE (56W)')
    mmse_56w_rate_mean = mmse_56w_rate_samples.mean()
    mmse_56w_rate_std = mmse_56w_rate_samples.std()
    print(f'MMSE (56W, power-matched to SAC): mean rate = {mmse_56w_rate_mean:.4f} bps/Hz '
          f'(std {mmse_56w_rate_std:.4f}), power = {sac_matched_power_watt:.1f} W '
          f'({100*sac_matched_power_watt/power_budget:.0f}% of budget, by construction)')

    sac_rate_mean_0, sac_rate_std_0 = load_sac_zero_error_rate(
        cfg.output_metrics_path, 'EE_dinkelbach_adaptive_lwin5000_N16K3_eta0.6_rawpow'
    )
    sac_rate_mean_005, sac_rate_std_005 = load_sac_zero_error_rate(
        cfg.output_metrics_path, 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_eta0.6_rawpow'
    )

    plot_cfg = PlotConfig()

    # ---- Deliverable 1: extended power bar chart ----
    # All three classical precoders get IDENTICAL synthetic samples pinned
    # exactly at the budget -- not a measurement, a direct expression of the
    # algebraic guarantee described in this file's docstring. Using the same
    # samples_dict shape (monte_carlo_iterations, user_nr) as the real SAC
    # measurements so plot_power_savings_bars' code path is identical either
    # way; user_nr equal split matches how get_precoding_zero_forcing/mrc are
    # actually parametrized above.
    power_samples_dict = {}
    for _, label in classical_precoders:
        equal_share = (power_budget / cfg.user_nr) * np.ones((MONTE_CARLO_ITERATIONS, cfg.user_nr))
        power_samples_dict[label] = equal_share

    # SAC power: reuse the already-validated zero-error Monte Carlo result
    # (56.0 W / 56.5 W, see handoff) rather than re-simulating -- represent
    # as constant-mean synthetic samples matching those known values, sized
    # for cfg.user_nr so plot_power_savings_bars' internal .sum(axis=1)
    # still recovers the right total.
    for label, mean_power in [('SAC (error=0.0)', 56.0), ('SAC (error=0.05)', 56.5)]:
        power_samples_dict[label] = (mean_power / cfg.user_nr) * np.ones((MONTE_CARLO_ITERATIONS, cfg.user_nr))

    # MMSE (56W): also pinned exactly at 56 W, by the same algebraic
    # guarantee as the full-budget classical precoders above -- just at a
    # smaller power_constraint_watt. Shown separately from the full-budget
    # MMSE bar so both the "MMSE always uses the whole budget" point and the
    # "if given SAC's power, what does MMSE get" point are both visible.
    power_samples_dict['MMSE (56W)'] = (sac_matched_power_watt / cfg.user_nr) * np.ones((MONTE_CARLO_ITERATIONS, cfg.user_nr))

    classical_colors = ['#caa023', '#7a4fbf', '#c9622a']  # gold, purple, burnt orange -- distinct from SAC's greens
    sac_colors = ['#307b3b', '#1baf7a']
    mmse_matched_color = ['#8a7000']  # dark gold, ties visually back to MMSE while staying distinct
    plot_power_savings_bars(
        samples_dict=power_samples_dict,
        power_budget=power_budget,
        plots_parent_path=plot_cfg.plots_parent_path,
        name='power_savings_bars_classical_vs_sac',
        model_colors=classical_colors + sac_colors + mmse_matched_color,
        title='Classical precoders (always at budget) vs. SAC (real savings), zero CSIT error',
    )

    # ---- Deliverable 2: rate comparison bar chart ----
    # MMSE (56W) placed directly next to SAC (error=0.0) -- both use exactly
    # 56 W, so the two adjacent bars are the fair, power-matched comparison;
    # the full-budget MMSE/MRT/ZF bars above them show the "unlimited power"
    # reference point instead.
    rate_labels = ['MMSE', 'MRT', 'ZF', 'MMSE (56W)', 'SAC (error=0.0)', 'SAC (error=0.05)']
    rate_means = [classical_rate_means['MMSE'], classical_rate_means['MRT'], classical_rate_means['ZF'],
                  mmse_56w_rate_mean, sac_rate_mean_0, sac_rate_mean_005]
    rate_stds = [classical_rate_stds['MMSE'], classical_rate_stds['MRT'], classical_rate_stds['ZF'],
                 mmse_56w_rate_std, sac_rate_std_0, sac_rate_std_005]
    rate_colors = classical_colors + mmse_matched_color + sac_colors

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_positions = np.arange(len(rate_labels))[::-1]
    ax.barh(y_positions, rate_means, xerr=rate_stds, color=rate_colors, height=0.6,
            error_kw=dict(ecolor='#2b2b2b', elinewidth=1, capsize=3))
    # label position must clear the error bar's whisker (mean + std), not
    # just the bar's mean -- placing it right after the mean let the
    # whisker cap visually cross through the text (looked like a
    # strikethrough on "4.91 bps/Hz" etc in an earlier version of this plot)
    for y, mean_rate, std_rate in zip(y_positions, rate_means, rate_stds):
        ax.text(mean_rate + std_rate + 0.08, y, f'{mean_rate:.2f} bps/Hz',
                 va='center', ha='left', fontsize=10, fontweight='bold')
    ax.set_yticks(y_positions)
    ax.set_yticklabels(rate_labels, fontsize=11)
    ax.set_xlabel('Rate R [bps/Hz]', fontsize=10.5)
    ax.set_xlim(0, max(m + s for m, s in zip(rate_means, rate_stds)) * 1.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='x', alpha=0.2, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_title('Rate at zero CSIT error: classical baselines vs. SAC', fontsize=13, pad=12)
    fig.tight_layout()

    pdf_path = Path(plot_cfg.plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)
    out = Path(pdf_path, 'rate_comparison_classical_vs_sac.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=300, transparent=True)
    print(f'Saved: {out}')
    plt.close(fig)

    print('\n' + '=' * 60)
    print('RATE COMPARISON SUMMARY (zero CSIT error)')
    print('=' * 60)
    for label, mean_rate, std_rate in zip(rate_labels, rate_means, rate_stds):
        print(f'{label}: {mean_rate:.4f} +- {std_rate:.4f} bps/Hz')
    print('=' * 60)
