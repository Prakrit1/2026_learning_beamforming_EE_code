"""
error_sweep_training_triplet, but with the RM (rate-only) curves for Delta-eps
=0.025 and 0.05 taken from the checkpoints TRAINED AND EVALUATED at the energy-
efficient power (p38 @ 38 W, p40 @ 40 W; produced by rate_only_ee_power_eval.py),
instead of the p75-trained RM scaled down. Delta-eps=0 keeps its p35 RM (already
trained+evaluated at ~35 W). So every RM curve here is "trained and evaluated at
the EE power" -- the clean, no-downscaling baseline.

Reads TWO gzips (non-destructive, does not touch the p75 triplet outputs):
  * rate_power_triplet.gzip          -> EE curves (all 3) + RM Delta-eps=0 (p35)
  * EE_rateonly_ee_power/...gzip     -> RM Delta-eps=0.025 (p38) + 0.05 (p40)
Writes reports/figures/*/error_sweep_training_triplet_eepower.*
"""
import gzip
import pickle
from pathlib import Path

from src.config.config import Config
from src.config.config_plotting import PlotConfig
from src.plotting.plotting import plot_rate_error_sweep

if __name__ == '__main__':
    cfg = Config()

    triplet_path = Path(cfg.output_metrics_path, 'EE_lwin5000_3gpp_triplet', 'rate_power_triplet.gzip')
    with gzip.open(triplet_path, 'rb') as file:
        tri = pickle.load(file)

    eepower_path = Path(cfg.output_metrics_path, 'EE_rateonly_ee_power', 'rate_only_ee_power.gzip')
    with gzip.open(eepower_path, 'rb') as file:
        eep = pickle.load(file)

    # Combined results: EE (all 3) + RM Delta-eps=0 (p35) from the triplet gzip,
    # RM Delta-eps=0.025/0.05 (p38/p40, trained+evaluated at EE power) from eepower gzip.
    results = {
        'sac_aod0.0': tri['results']['sac_aod0.0'],
        'sac_aod0.025': tri['results']['sac_aod0.025'],
        'sac_aod0.05': tri['results']['sac_aod0.05'],
        'rm_aod0.0': tri['results']['rm_matched_aod0.0'],          # p35, native ~35 W
        'rm_aod0.025': eep['results']['rm_eepower_aod0.025'],      # p38, native ~38 W
        'rm_aod0.05': eep['results']['rm_eepower_aod0.05'],        # p40, native ~40 W
    }

    plot_cfg = PlotConfig()
    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.6

    # EE superscript is its training budget (75 W, clip-only lets it operate lower);
    # RM superscript is ITS training budget (== its operating power, since RM here is
    # trained and evaluated at the same EE power). Post-comma / P= is the measured power.
    ee_trained_watt = round(results['sac_aod0.0']['power_budget'])
    ee0_watt = round(results['sac_aod0.0']['mean_power'][0])
    ee0025_watt = round(results['sac_aod0.025']['mean_power'][0])
    ee05_watt = round(results['sac_aod0.05']['mean_power'][0])

    rm0_watt = round(results['rm_aod0.0']['mean_power'][0])
    rm0025_watt = round(results['rm_aod0.025']['mean_power'][0])
    rm05_watt = round(results['rm_aod0.05']['mean_power'][0])

    curves = [
        {'result_key': 'sac_aod0.0', 'label': f'EE$^{{{ee_trained_watt}, \\mathrm{{Δε=0.00}}}}$, $P={ee0_watt}$ W',
         'color': plot_cfg.cp2['green'], 'marker': 'o', 'linestyle': '-', 'markevery': (0, 2)},
        {'result_key': 'rm_aod0.0', 'label': f'RM$^{{{rm0_watt}, \\mathrm{{Δε=0.00}}}}$, $P={rm0_watt}$ W',
         'color': plot_cfg.cp2['green'], 'marker': 's', 'linestyle': '--', 'markevery': (1, 2)},
        {'result_key': 'sac_aod0.025', 'label': f'EE$^{{{ee_trained_watt}, \\mathrm{{Δε=0.025}}}}$, $P={ee0025_watt}$ W',
         'color': plot_cfg.cp2['blue'], 'marker': 'o', 'linestyle': '-', 'markevery': (0, 2)},
        {'result_key': 'rm_aod0.025', 'label': f'RM$^{{{rm0025_watt}, \\mathrm{{Δε=0.025}}}}$, $P={rm0025_watt}$ W',
         'color': plot_cfg.cp2['blue'], 'marker': 's', 'linestyle': '--', 'markevery': (1, 2)},
        {'result_key': 'sac_aod0.05', 'label': f'EE$^{{{ee_trained_watt}, \\mathrm{{Δε=0.05}}}}$, $P={ee05_watt}$ W',
         'color': plot_cfg.cp2['magenta'], 'marker': 'o', 'linestyle': '-', 'markevery': (0, 2)},
        {'result_key': 'rm_aod0.05', 'label': f'RM$^{{{rm05_watt}, \\mathrm{{Δε=0.05}}}}$, $P={rm05_watt}$ W',
         'color': plot_cfg.cp2['magenta'], 'marker': 's', 'linestyle': '--', 'markevery': (1, 2)},
    ]

    plot_rate_error_sweep(
        error_sweep_range=tri['error_sweep_range'],
        results=results,
        curves=curves,
        width=plot_width,
        height=plot_height,
        plots_parent_path=plot_cfg.plots_parent_path,
        name='error_sweep_training_triplet_eepower',
        annotate_power=False,  # power is already in the legend labels
        legend_ncols=3,
        legend_loc='lower center',
        legend_bbox_to_anchor=(0.5, 1.02),
        legend_fontsize=9,
    )
