"""
Does the fixed-lambda (classical Dinkelbach) sweep trace a real (rate,
power) frontier, or do all six runs collapse to the same operating point
regardless of lambda? This is the actual test of the Lagrangian theory --
see handoff_prompt_EE_evaluation.txt's "Jobs submitted Friday" section for
the full reasoning.

Reads:
- rate at zero CSIT error for each of the 6 lambda checkpoints, from the
  error-sweep gzips produced by my_evaluation.py's fixed-lambda blocks
  (outputs/metrics/EE_dinkelbach_adaptive_aod0.5_lambdafixed{value}_N16K3_eta0.6/
  error_sweep/testing_learned_sweep_0.0_0.5.gzip)
- the zero-error MMSE ceiling, reused from full_EE_aod0.5_N16K3_eta0.6's
  MMSE sweep (MMSE doesn't depend on lambda, see my_evaluation.py's comment)
- mean total transmit power per lambda, from the summary gzip written by
  tx_power_distribution.py's fixed-lambda block
  (outputs/metrics/EE_dinkelbach_adaptive_lambda_sweep/tx_power/lambda_sweep_power_summary.gzip)

Run this AFTER both my_evaluation.py and tx_power_distribution.py have
produced their fixed-lambda outputs.
"""
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False

import gzip
import pickle
from pathlib import Path

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

from src.config.config import Config
from src.config.config_plotting import PlotConfig

LAMBDA_VALUES = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]


def _get_sumrate_key(metrics_dict):
    for key in metrics_dict.keys():
        if 'calc_sum_rate' in str(key):
            return key
    raise ValueError('calc_sum_rate key not found in error-sweep metrics dict')


def load_zero_error_rate(output_metrics_path, training_name):
    path = Path(output_metrics_path, training_name, 'error_sweep', 'testing_learned_sweep_0.0_0.5.gzip')
    with gzip.open(path, 'rb') as file:
        error_sweep_range, metrics = pickle.load(file)
    key = _get_sumrate_key(metrics)
    # error_sweep_range[0] == 0.0 -- see my_evaluation.py's error_sweep_range
    return metrics[key]['mean'][0], metrics[key]['std'][0]


def load_mmse_zero_error_rate(output_metrics_path):
    path = Path(output_metrics_path, 'full_EE_aod0.5_N16K3_eta0.6', 'error_sweep', 'testing_mmse_sweep_0.0_0.5.gzip')
    with gzip.open(path, 'rb') as file:
        error_sweep_range, metrics = pickle.load(file)
    key = _get_sumrate_key(metrics)
    return metrics[key]['mean'][0]


def load_power_summary(output_metrics_path):
    path = Path(output_metrics_path, 'EE_dinkelbach_adaptive_lambda_sweep', 'tx_power', 'lambda_sweep_power_summary.gzip')
    with gzip.open(path, 'rb') as file:
        return pickle.load(file)


def plot_lambda_sweep(plots_parent_path, name='lambda_sweep_rate_power'):
    cfg = Config()
    rates, rate_stds = [], []
    for lv in LAMBDA_VALUES:
        training_name = f'EE_dinkelbach_adaptive_aod0.5_lambdafixed{lv}_N16K3_eta0.6'
        mean_rate, std_rate = load_zero_error_rate(cfg.output_metrics_path, training_name)
        rates.append(mean_rate)
        rate_stds.append(std_rate)
    rates = np.array(rates)
    rate_stds = np.array(rate_stds)

    mmse_rate = load_mmse_zero_error_rate(cfg.output_metrics_path)

    power_data = load_power_summary(cfg.output_metrics_path)
    power_budget = power_data['power_budget']
    powers = np.array([power_data['lambda_summary'][lv]['mean_total_power_watt'] for lv in LAMBDA_VALUES])
    power_stds = np.array([power_data['lambda_summary'][lv]['std_total_power_watt'] for lv in LAMBDA_VALUES])

    # PlotConfig() resets text.usetex to True on construction (see
    # plot_lambda_convergence.py's identical override) -- some cluster nodes
    # don't even have latex installed, and the lambda labels below don't
    # need it, so force it back off here rather than relying on the
    # module-level setting from import time.
    matplotlib.rcParams['text.usetex'] = False

    # Same sequential green ramp as tx_power_distribution.py's fixed-lambda
    # panel -- lambda is an ordered parameter, so color goes light->dark with
    # increasing lambda rather than a categorical hue per point.
    ramp = matplotlib.colormaps['Greens']
    point_colors = [
        matplotlib.colors.to_hex(ramp(0.3 + 0.65 * i / (len(LAMBDA_VALUES) - 1)))
        for i in range(len(LAMBDA_VALUES))
    ]
    mmse_color = '#caa023'  # gold, matches plot_cfg.cp2['gold'] used elsewhere for MMSE
    budget_color = '#d03b3b'  # status red, matches tx_power_distribution.py's budget line

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # ---- Panel 1: rate vs lambda ----
    ax = axes[0]
    ax.errorbar(LAMBDA_VALUES, rates, yerr=rate_stds, marker='o', color='#307b3b',
                ecolor='#307b3b', elinewidth=1, capsize=3, linewidth=1.5, markersize=6)
    for x, y, c in zip(LAMBDA_VALUES, rates, point_colors):
        ax.scatter([x], [y], color=c, s=70, zorder=5, edgecolor='white', linewidth=0.6)
    ax.axhline(mmse_rate, color=mmse_color, linestyle='--', linewidth=1.5, label='MMSE (zero error)')
    ax.set_xlabel(r'Fixed $\lambda$')
    ax.set_ylabel('Rate R [bps/Hz] (zero CSIT error)')
    ax.set_title('Rate vs. λ', fontsize=11)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc='best')

    # ---- Panel 2: power vs lambda ----
    ax2 = axes[1]
    ax2.errorbar(LAMBDA_VALUES, powers, yerr=power_stds, marker='o', color='#307b3b',
                 ecolor='#307b3b', elinewidth=1, capsize=3, linewidth=1.5, markersize=6)
    for x, y, c in zip(LAMBDA_VALUES, powers, point_colors):
        ax2.scatter([x], [y], color=c, s=70, zorder=5, edgecolor='white', linewidth=0.6)
    ax2.axhline(power_budget, color=budget_color, linestyle='--', linewidth=1.5, label='Power budget')
    ax2.set_xlabel(r'Fixed $\lambda$')
    ax2.set_ylabel('Mean total transmit power [W]')
    ax2.set_title('Power vs. λ', fontsize=11)
    ax2.grid(True, alpha=0.25, linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=8, loc='best')

    # ---- Panel 3: rate-vs-power scatter, lambda as the varying parameter ----
    # This is the actual frontier plot: if the Lagrangian sweep works, these
    # 6 points should trace a real curve (power falling, rate falling, with
    # some interior lambda giving the best rate/power ratio). If they
    # collapse to one point regardless of lambda, that's the flat-ceiling
    # signature seen everywhere else this week.
    ax3 = axes[2]
    ax3.plot(powers, rates, color='0.6', linewidth=1, zorder=1)
    for x, y, c, lv in zip(powers, rates, point_colors, LAMBDA_VALUES):
        ax3.scatter([x], [y], color=c, s=90, zorder=5, edgecolor='white', linewidth=0.6)
        ax3.annotate(f'λ={lv}', (x, y), textcoords='offset points', xytext=(6, 4), fontsize=8)
    ax3.axvline(power_budget, color=budget_color, linestyle='--', linewidth=1.2, label='Power budget')
    ax3.axhline(mmse_rate, color=mmse_color, linestyle='--', linewidth=1.2, label='MMSE (zero error)')
    ax3.set_xlabel('Mean total transmit power [W]')
    ax3.set_ylabel('Rate R [bps/Hz]')
    ax3.set_title('Rate-power frontier (λ-parametrized)', fontsize=11)
    ax3.grid(True, alpha=0.25, linewidth=0.5)
    ax3.set_axisbelow(True)
    ax3.legend(fontsize=8, loc='best')

    fig.suptitle('Fixed-λ (classical Dinkelbach) sweep: does λ have a causal effect?', fontsize=13, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    pdf_path = Path(plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)
    out = Path(pdf_path, f'{name}.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=300, transparent=True)
    print(f'Saved: {out}')
    plt.close(fig)

    print('\n' + '=' * 60)
    print('FIXED-LAMBDA SWEEP SUMMARY (zero CSIT error)')
    print('=' * 60)
    print(f'MMSE rate ceiling: {mmse_rate:.4f} bps/Hz')
    print(f'Power budget: {power_budget:.2f} W')
    for lv, r, rs, p, ps in zip(LAMBDA_VALUES, rates, rate_stds, powers, power_stds):
        ratio = r / p if p > 0 else float('nan')
        print(f'  lambda={lv:>4}: rate={r:.4f}+-{rs:.4f} bps/Hz, '
              f'power={p:.2f}+-{ps:.2f} W ({100 * p / power_budget:.1f}% of budget), '
              f'rate/power={ratio:.5f}')
    print('=' * 60)


if __name__ == '__main__':
    plot_cfg = PlotConfig()
    plot_lambda_sweep(plots_parent_path=plot_cfg.plots_parent_path)
