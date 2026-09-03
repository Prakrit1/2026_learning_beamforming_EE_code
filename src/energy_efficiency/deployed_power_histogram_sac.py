"""Histogram of the deployed EE policy's own per-channel radiated power.

Unlike ee_vs_transmit_power_sweep_sac.py (which forces the SAME power on
every channel realization, then measures EE -- a curve that structurally
cannot peak near the deployed policy's mean power, see that script's
history/commit log for the derivation), this reads the deployed clip-only
policy's ACTUAL per-realization power choices and shows their distribution.
No forced-fixed-power counterfactual, no re-priced reward -- just what
power the policy already chose on each of 10k independent channel draws.

Reuses already-saved per-sample data from power_savings_bars_triplet.py
(outputs/metrics/EE_lwin5000_3gpp_triplet/power_savings_bars.gzip,
samples_dict['SAC (Δε = 0.0, energy-efficient)'], shape (10000, user_nr)):
that script already ran 10k Monte Carlo iterations of the deployed
clip-only policy and saved the per-user power draw for every iteration, it
just never plotted their distribution. No new simulation needed.

NOTE: this data is perfect-CSIT only (error bound 0.0) -- the
power_savings_bars_triplet.py source data was generated at
eval_error_bound=0.0 only, so there is currently no equivalent per-sample
data at nonzero CSIT error. Extending to e.g. Delta-epsilon=0.05 needs
that script's per-sample saving extended to sweep error too, then a fresh
10k-iteration rollout (submit via sbatch, do not run that part locally).

Saves reports/figures/{pdf,jpg,png}/deployed_power_histogram_sac_error0.*
"""
import os

# Guard against leftover shell env vars from other ablation runs (elevation/
# gain/budget sweeps) silently changing this system's configuration --
# power_savings_bars_triplet.py and plotting_scenario.py already do this;
# added here for consistency even though this script's own numbers come
# from the saved gzip, not fresh Config() values.
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

CSIT_ERROR_BOUND = 0.0  # matches power_savings_bars_triplet.py's source data, see docstring


if __name__ == '__main__':
    cfg = Config()
    plot_cfg = PlotConfig()
    # PlotConfig() resets rcParams['text.usetex'] to True internally, which
    # needs a local `latex` binary the compute nodes don't have -- every
    # other script in this repo re-asserts False here for the same reason
    # (see commit 86244ba, "restore text.usetex=False re-assertion").
    matplotlib.rcParams['text.usetex'] = False

    gzip_path = Path(cfg.output_metrics_path, 'EE_lwin5000_3gpp_triplet', 'power_savings_bars.gzip')
    with gzip.open(gzip_path, 'rb') as file:
        data = pickle.load(file)

    # sum over users -> total radiated power per Monte Carlo iteration
    per_sample_power = data['samples_dict']['SAC (Δε = 0.0, energy-efficient)'].sum(axis=1)
    mean_power = per_sample_power.mean()
    print(f'[deployed_power_histogram_sac] {len(per_sample_power)} samples, '
          f'mean={mean_power:.2f} W, median={np.median(per_sample_power):.2f} W, '
          f'budget={data["power_budget"]:.0f} W')

    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.62

    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    # same green as the EE curve/bar everywhere else this checkpoint (aod=0.0)
    # appears -- ee_vs_transmit_power_sweep_sac.py, power_savings_bars_plot.py.
    ax.hist(per_sample_power, bins=40, range=(0, data['power_budget']),
            color=plot_cfg.cp2['green'], alpha=0.75, edgecolor='white', linewidth=0.4,
            weights=np.full(len(per_sample_power), 1.0 / len(per_sample_power)))
    ax.axvline(mean_power, color=plot_cfg.cp2['gold'], linestyle='-.', linewidth=1.5, zorder=5,
               label=fr'$\bar P_{{\mathrm{{EE}}}} \approx {mean_power:.0f}$ W')
    ax.axvline(data['power_budget'], color='gray', linestyle=':', linewidth=1.2,
               label=r'$P_{\mathrm{rad}}$')

    ax.set_xlabel(r'Radiated power $P$ per channel realization [W]', fontsize=13)
    ax.set_ylabel('Fraction of realizations', fontsize=13)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.08),
               ncol=len(labels), fontsize=11, frameon=False, columnspacing=1.4,
               handletextpad=0.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    pdf_path = Path(plot_cfg.plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)
    out = Path(pdf_path, f'deployed_power_histogram_sac_error{CSIT_ERROR_BOUND:g}.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=300, transparent=True)
    print(f'Saved: {out}')

    jpg_path = Path(plot_cfg.plots_parent_path, 'jpg')
    jpg_path.mkdir(parents=True, exist_ok=True)
    out_jpg = Path(jpg_path, f'deployed_power_histogram_sac_error{CSIT_ERROR_BOUND:g}.jpg')
    fig.savefig(out_jpg, bbox_inches='tight', dpi=200)
    print(f'Saved: {out_jpg}')

    png_path = Path(plot_cfg.plots_parent_path, 'png')
    png_path.mkdir(parents=True, exist_ok=True)
    out_png = Path(png_path, f'deployed_power_histogram_sac_error{CSIT_ERROR_BOUND:g}.png')
    fig.savefig(out_png, bbox_inches='tight', dpi=200, transparent=True)
    print(f'Saved: {out_png}')

    plt.close(fig)
