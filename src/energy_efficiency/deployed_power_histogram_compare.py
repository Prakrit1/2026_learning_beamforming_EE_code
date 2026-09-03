"""Overlay the EE-trained policy's and the rate-only-trained policy's own
per-channel radiated power distributions on the same axes, as continuous
KDE density curves (not binned bars) with an explicit gain annotation.

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
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt

from src.config.config import Config
from src.config.config_plotting import PlotConfig

CSIT_ERROR_BOUND = 0.0  # matches both source gzips, see docstring


def density_curve(samples, x_grid, min_std=1e-3):
    """Gaussian-KDE density over x_grid, or a narrow synthetic spike if the
    samples are (near-)degenerate -- gaussian_kde needs nonzero variance,
    and the rate-only policy is expected to land at essentially the same
    power (the budget) on every realization."""
    std = samples.std()
    if std < min_std:
        spike_std = max(min_std, x_grid[-1] * 0.003)
        return np.exp(-0.5 * ((x_grid - samples.mean()) / spike_std) ** 2) / (spike_std * np.sqrt(2 * np.pi))
    return gaussian_kde(samples)(x_grid)


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
    x_grid = np.linspace(0, power_budget * 1.05, 500)
    ee_density = density_curve(ee_power, x_grid)
    rateonly_density = density_curve(rateonly_power, x_grid)

    # green matches the EE checkpoint's color everywhere else it appears;
    # black matches the MMSE/baseline convention used in the other figures.
    ax.plot(x_grid, ee_density, color=plot_cfg.cp2['green'], linewidth=1.8)
    ax.fill_between(x_grid, ee_density, color=plot_cfg.cp2['green'], alpha=0.35, label='EE')
    ax.plot(x_grid, rateonly_density, color=plot_cfg.cp2['black'], linewidth=1.8)
    ax.fill_between(x_grid, rateonly_density, color=plot_cfg.cp2['black'], alpha=0.35, label='rate-only')

    ax.axvline(power_budget, color='gray', linestyle=':', linewidth=1.2,
               label=r'$P_{\mathrm{rad}}$')

    # Gain annotation: make the Watt/percent gap between the two policies'
    # means an explicit, labeled feature of the figure instead of something
    # the reader has to eyeball off two peak locations.
    ee_mean, rateonly_mean = ee_power.mean(), rateonly_power.mean()
    gain_watt = rateonly_mean - ee_mean
    gain_pct = 100 * gain_watt / rateonly_mean
    arrow_y = 1.08 * max(ee_density.max(), rateonly_density.max())
    ax.annotate('', xy=(rateonly_mean, arrow_y), xytext=(ee_mean, arrow_y),
                arrowprops=dict(arrowstyle='<->', color=plot_cfg.cp2['gold'], linewidth=1.5))
    ax.text((ee_mean + rateonly_mean) / 2, arrow_y * 1.06,
            fr'$\approx${gain_watt:.0f} W saved ({gain_pct:.0f}%)',
            ha='center', va='bottom', fontsize=10.5, color=plot_cfg.cp2['gold'])
    ax.set_ylim(top=arrow_y * 1.35)

    ax.set_xlabel(r'Radiated power $P$ per channel realization [W]', fontsize=13)
    ax.set_ylabel('Probability density', fontsize=13)
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
