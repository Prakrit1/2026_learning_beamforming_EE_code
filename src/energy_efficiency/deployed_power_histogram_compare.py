"""Overlay the EE-trained policy's and the rate-only-trained policy's own
per-channel radiated power distributions on the same axes.

Companion/superset of deployed_power_histogram_sac.py: that script only
plots the Dinkelbach-adaptive-EE checkpoint's histogram (source:
power_savings_bars.gzip, already committed). This one adds the
SAC_rateonly baseline's histogram (source: power_samples_rateonly.gzip,
produced by power_samples_rateonly.py -- run that first if missing) on top
of it, both under the same get_precoding_learned_clip_only evaluation
projection and the same perfect-CSIT (error=0) condition, so the two are
directly comparable.

Expected result (per Lemma 1, sum rate monotonic in power): the rate-only
policy has no incentive to hold back, so its histogram should sit as a
tight spike right at the 75W budget, in sharp contrast to the EE policy's
~35W cluster -- i.e. this figure is the direct empirical evidence that the
Dinkelbach reward, not just the clip-only projection mechanism itself, is
what produces non-trivial below-budget power use.

Saves reports/figures/{pdf,jpg,png}/deployed_power_histogram_compare_error0.*
"""
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

CSIT_ERROR_BOUND = 0.0  # matches both source gzips, see docstring


if __name__ == '__main__':
    cfg = Config()
    plot_cfg = PlotConfig()
    # PlotConfig() resets rcParams['text.usetex'] to True internally, which
    # needs a local `latex` binary the compute nodes don't have -- every
    # other script in this repo re-asserts False here for the same reason
    # (see commit 86244ba, "restore text.usetex=False re-assertion").
    matplotlib.rcParams['text.usetex'] = False

    triplet_path = Path(cfg.output_metrics_path, 'EE_lwin5000_3gpp_triplet', 'power_savings_bars.gzip')
    with gzip.open(triplet_path, 'rb') as file:
        ee_data = pickle.load(file)
    ee_power = ee_data['samples_dict']['SAC (Δε = 0.0, energy-efficient)'].sum(axis=1)
    power_budget = ee_data['power_budget']

    rateonly_path = Path(cfg.output_metrics_path, 'EE_lwin5000_3gpp_triplet', 'power_samples_rateonly.gzip')
    if not rateonly_path.exists():
        raise FileNotFoundError(
            f'{rateonly_path} not found -- run power_samples_rateonly.slurm first '
            '(needs the SAC_rateonly checkpoint to already be trained).'
        )
    with gzip.open(rateonly_path, 'rb') as file:
        rateonly_data = pickle.load(file)
    rateonly_power = rateonly_data['power_samples'].sum(axis=1)

    print(f'EE policy: n={len(ee_power)}, mean={ee_power.mean():.2f} W, '
          f'min={ee_power.min():.2f} W, max={ee_power.max():.2f} W')
    print(f'Rate-only policy: n={len(rateonly_power)}, mean={rateonly_power.mean():.2f} W, '
          f'min={rateonly_power.min():.2f} W, max={rateonly_power.max():.2f} W')

    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.62

    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    bins = np.linspace(0, power_budget, 41)
    # green matches the EE checkpoint's color everywhere else it appears;
    # black matches the MMSE/baseline convention used in the other figures.
    ax.hist(ee_power, bins=bins, color=plot_cfg.cp2['green'], alpha=0.6,
            edgecolor='white', linewidth=0.4, label='EE',
            weights=np.full(len(ee_power), 1.0 / len(ee_power)))
    ax.hist(rateonly_power, bins=bins, color=plot_cfg.cp2['black'], alpha=0.6,
            edgecolor='white', linewidth=0.4, label='rate-only',
            weights=np.full(len(rateonly_power), 1.0 / len(rateonly_power)))
    ax.axvline(ee_power.mean(), color=plot_cfg.cp2['gold'], linestyle='-.', linewidth=1.5, zorder=5,
               label=fr'$\bar P_{{\mathrm{{EE}}}} \approx {ee_power.mean():.0f}$ W')
    ax.axvline(power_budget, color='gray', linestyle=':', linewidth=1.2,
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
    out = Path(pdf_path, f'deployed_power_histogram_compare_error{CSIT_ERROR_BOUND:g}.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=300, transparent=True)
    print(f'Saved: {out}')

    jpg_path = Path(plot_cfg.plots_parent_path, 'jpg')
    jpg_path.mkdir(parents=True, exist_ok=True)
    out_jpg = Path(jpg_path, f'deployed_power_histogram_compare_error{CSIT_ERROR_BOUND:g}.jpg')
    fig.savefig(out_jpg, bbox_inches='tight', dpi=200)
    print(f'Saved: {out_jpg}')

    png_path = Path(plot_cfg.plots_parent_path, 'png')
    png_path.mkdir(parents=True, exist_ok=True)
    out_png = Path(png_path, f'deployed_power_histogram_compare_error{CSIT_ERROR_BOUND:g}.png')
    fig.savefig(out_png, bbox_inches='tight', dpi=200, transparent=True)
    print(f'Saved: {out_png}')

    plt.close(fig)
