"""Two-point EE-vs-power comparison: the deployed EE policy's real
operating point against the rate-only baseline's real operating point.

Unlike ee_vs_transmit_power_sweep_sac.py's fixed-power sweep (which
measures EE along one hypothetical curve that can't pass through the
deployed policy's actual mean power, see that script's history) or
deployed_power_histogram_compare.py's density-over-power view, this plots
exactly two points -- each policy's own real (mean power, achieved EE) --
connected by an arrow annotated with the Watt/rate/EE deltas between them.
The point is to make the actual trade the EE reward makes explicit: less
power for a little less rate, but higher EE.

Sources (both already-saved, no new simulation):
- EE policy: outputs/metrics/EE_lwin5000_3gpp_triplet/rate_power_triplet.gzip,
  results['sac_aod0.0'] at error bound 0.0 (mean_power, mean_rate).
- Rate-only policy: outputs/metrics/EE_lwin5000_3gpp_triplet/
  power_samples_rateonly.gzip (produced by power_samples_rateonly.py --
  run that first if missing).

Saves reports/figures/{pdf,jpg,png}/ee_gain_dumbbell_error0.*
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


def total_power_watt(cfg, transmit_power_watt):
    return transmit_power_watt / cfg.pa_efficiency + cfg.sat_nr * cfg.sat_ant_nr * cfg.circuit_power_watt


if __name__ == '__main__':
    cfg = Config()
    plot_cfg = PlotConfig()
    # PlotConfig() resets rcParams['text.usetex'] to True internally, which
    # needs a local `latex` binary the compute nodes don't have -- every
    # other script in this repo re-asserts False here for the same reason
    # (see commit 86244ba, "restore text.usetex=False re-assertion").
    matplotlib.rcParams['text.usetex'] = False

    triplet_path = Path(cfg.output_metrics_path, 'EE_lwin5000_3gpp_triplet', 'rate_power_triplet.gzip')
    with gzip.open(triplet_path, 'rb') as file:
        triplet_data = pickle.load(file)
    error_idx = int(np.argmin(np.abs(triplet_data['error_sweep_range'] - CSIT_ERROR_BOUND)))
    ee_power = triplet_data['results']['sac_aod0.0']['mean_power'][error_idx]
    ee_rate = triplet_data['results']['sac_aod0.0']['mean_rate'][error_idx]
    ee_ee = ee_rate / total_power_watt(cfg, ee_power)

    rateonly_path = Path(cfg.output_metrics_path, 'EE_lwin5000_3gpp_triplet', 'power_samples_rateonly.gzip')
    if not rateonly_path.exists():
        raise FileNotFoundError(
            f'{rateonly_path} not found -- run power_samples_rateonly.slurm first '
            '(needs the SAC_rateonly checkpoint to already be trained).'
        )
    with gzip.open(rateonly_path, 'rb') as file:
        rateonly_data = pickle.load(file)
    rateonly_power = rateonly_data['power_samples'].sum(axis=1).mean()
    rateonly_rate = rateonly_data['rate_samples'].mean()
    rateonly_ee = rateonly_rate / total_power_watt(cfg, rateonly_power)

    power_gain_watt = rateonly_power - ee_power
    power_gain_pct = 100 * power_gain_watt / rateonly_power
    rate_loss_pct = 100 * (rateonly_rate - ee_rate) / rateonly_rate
    ee_gain_pct = 100 * (ee_ee - rateonly_ee) / rateonly_ee

    print(f'EE policy: power={ee_power:.2f} W, rate={ee_rate:.4f} bps/Hz, EE={ee_ee:.5f} bps/Hz/W')
    print(f'Rate-only policy: power={rateonly_power:.2f} W, rate={rateonly_rate:.4f} bps/Hz, '
          f'EE={rateonly_ee:.5f} bps/Hz/W')
    print(f'Gain: {power_gain_watt:.1f} W saved ({power_gain_pct:.0f}%), '
          f'rate down {rate_loss_pct:.0f}%, EE up {ee_gain_pct:.0f}%')

    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.62

    fig, ax = plt.subplots(figsize=(plot_width, plot_height))

    # connecting arrow first so the markers draw on top of it
    ax.annotate('', xy=(ee_power, ee_ee), xytext=(rateonly_power, rateonly_ee),
                arrowprops=dict(arrowstyle='<-', color=plot_cfg.cp2['gold'], linewidth=1.8,
                                 shrinkA=10, shrinkB=10))

    ax.plot(ee_power, ee_ee, color=plot_cfg.cp2['green'], marker='o', markersize=11,
            linestyle='none', zorder=5, label=fr'EE, $P\approx{ee_power:.0f}$ W')
    ax.annotate(fr'rate $\approx${ee_rate:.1f} bps/Hz', (ee_power, ee_ee),
                textcoords='offset points', xytext=(10, -14), fontsize=10, ha='left')

    ax.plot(rateonly_power, rateonly_ee, color=plot_cfg.cp2['black'], marker='o', markersize=11,
            linestyle='none', zorder=5, label=fr'rate-only, $P\approx{rateonly_power:.0f}$ W')
    ax.annotate(fr'rate $\approx${rateonly_rate:.1f} bps/Hz', (rateonly_power, rateonly_ee),
                textcoords='offset points', xytext=(10, 8), fontsize=10, ha='left')

    mid_x = (ee_power + rateonly_power) / 2
    mid_y = (ee_ee + rateonly_ee) / 2
    ax.annotate(
        fr'$\approx${power_gain_watt:.0f} W saved ({power_gain_pct:.0f}%)' + '\n'
        fr'rate $-${rate_loss_pct:.0f}%, EE $+${ee_gain_pct:.0f}%',
        (mid_x, mid_y), textcoords='offset points', xytext=(-15, 18), fontsize=10.5,
        ha='right', color=plot_cfg.cp2['gold'],
    )

    ax.set_xlabel(r'Radiated power $P$ [W]', fontsize=13)
    ax.set_ylabel('EE [bps/Hz/W]', fontsize=13)
    ax.set_xlim(0, cfg.power_constraint_watt * 1.15)
    ax.set_ylim(0, max(ee_ee, rateonly_ee) * 1.35)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=11, loc='upper right', frameon=False)
    fig.tight_layout(pad=0.4)

    pdf_path = Path(plot_cfg.plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)
    out = Path(pdf_path, f'ee_gain_dumbbell_error{CSIT_ERROR_BOUND:g}.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=300, transparent=True)
    print(f'Saved: {out}')

    jpg_path = Path(plot_cfg.plots_parent_path, 'jpg')
    jpg_path.mkdir(parents=True, exist_ok=True)
    out_jpg = Path(jpg_path, f'ee_gain_dumbbell_error{CSIT_ERROR_BOUND:g}.jpg')
    fig.savefig(out_jpg, bbox_inches='tight', dpi=200)
    print(f'Saved: {out_jpg}')

    png_path = Path(plot_cfg.plots_parent_path, 'png')
    png_path.mkdir(parents=True, exist_ok=True)
    out_png = Path(png_path, f'ee_gain_dumbbell_error{CSIT_ERROR_BOUND:g}.png')
    fig.savefig(out_png, bbox_inches='tight', dpi=200, transparent=True)
    print(f'Saved: {out_png}')

    plt.close(fig)
