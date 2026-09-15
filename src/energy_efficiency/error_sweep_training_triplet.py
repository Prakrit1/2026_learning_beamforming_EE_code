"""
Rate-vs-error figure for three training-error SAC checkpoints (Δε =
0.00/0.025/0.05), each paired with the genuine rate-only (RM) policy at
EQUAL transmit power: the EE checkpoint at its own energy-efficient
operating power (solid), and the RM checkpoint re-evaluated at that SAME
per-error power (dashed). The two overlap -- showing that once RM is handed
the EE policy's (reduced) power budget it matches EE's rate -- so markers
are staggered to keep both visible. A separate figure from
plotting_scenario.py's 5-curve MMSE/SAC comparison, answering a different
question: how does training at a larger CSIT error bound trade off rate for
robustness, and does the rate-only policy do any better at the same power.

Plotting-only: reuses the gzip plotting_scenario.py already produces
(outputs/metrics/EE_lwin5000_3gpp_triplet/rate_power_triplet.gzip), no new
Monte Carlo simulation. Run plotting_scenario.py first -- it must have been
run with the aod0.0/aod0.025/aod0.05 EE checkpoints AND the genuine RM
(SAC_rateonly) checkpoint present under models/, so the gzip carries both
the 'sac_{aod_key}' curves and the 'rm_matched_{aod_key}' curves this
figure reads.

Saves reports/figures/{pdf,jpg,png}/error_sweep_training_triplet.*
"""
import gzip
import pickle
from pathlib import Path

from src.config.config import Config
from src.config.config_plotting import PlotConfig
from src.plotting.plotting import plot_rate_error_sweep

if __name__ == '__main__':
    cfg = Config()

    gzip_path = Path(cfg.output_metrics_path, 'EE_lwin5000_3gpp_triplet', 'rate_power_triplet.gzip')
    with gzip.open(gzip_path, 'rb') as file:
        data = pickle.load(file)

    plot_cfg = PlotConfig()
    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.6

    # Notation matches plotting_scenario.py's error-sweep figure: superscript
    # is the fixed training-time power budget (same for every checkpoint
    # here), the post-comma value is each curve's own measured operating
    # power -- watts pulled from the actual measured samples, not hardcoded.
    trained_watt = round(data['results']['sac_aod0.0']['power_budget'])
    aod0_watt = round(data['results']['sac_aod0.0']['mean_power'][0])
    aod0025_watt = round(data['results']['sac_aod0.025']['mean_power'][0])
    aod05_watt = round(data['results']['sac_aod0.05']['mean_power'][0])

    # genuine RM's own measured power once matched to each EE checkpoint's power
    # (~equal to the EE watts above -- that's the point of the equal-power match).
    rm0_watt = round(data['results']['rm_matched_aod0.0']['mean_power'][0])
    rm0025_watt = round(data['results']['rm_matched_aod0.025']['mean_power'][0])
    rm05_watt = round(data['results']['rm_matched_aod0.05']['mean_power'][0])

    # Each Δε checkpoint shown as a pair at EQUAL transmit power: the EE policy
    # at its own energy-efficient operating power (solid), and the genuine RM
    # (rate-only) policy re-evaluated at that SAME power (dashed). The pair
    # overlaps -- so markers are staggered (EE on even x-points, RM on odd) to
    # keep both visible. Interleaved EE/RM per Δε so the 3-column legend fills
    # column-major into matching color pairs.
    curves = [
        {'result_key': 'sac_aod0.0', 'label': f'EE$^{{{trained_watt}, \\mathrm{{Δε=0.00}}}}$, $P={aod0_watt}$ W',
         'color': plot_cfg.cp2['green'], 'marker': 'o', 'linestyle': '-', 'markevery': (0, 2)},
        {'result_key': 'rm_matched_aod0.0', 'label': f'RM$^{{{trained_watt}, \\mathrm{{Δε=0.00}}}}$, $P={rm0_watt}$ W',
         'color': plot_cfg.cp2['green'], 'marker': 's', 'linestyle': '--', 'markevery': (1, 2)},
        {'result_key': 'sac_aod0.025', 'label': f'EE$^{{{trained_watt}, \\mathrm{{Δε=0.025}}}}$, $P={aod0025_watt}$ W',
         'color': plot_cfg.cp2['blue'], 'marker': 'o', 'linestyle': '-', 'markevery': (0, 2)},
        {'result_key': 'rm_matched_aod0.025', 'label': f'RM$^{{{trained_watt}, \\mathrm{{Δε=0.025}}}}$, $P={rm0025_watt}$ W',
         'color': plot_cfg.cp2['blue'], 'marker': 's', 'linestyle': '--', 'markevery': (1, 2)},
        {'result_key': 'sac_aod0.05', 'label': f'EE$^{{{trained_watt}, \\mathrm{{Δε=0.05}}}}$, $P={aod05_watt}$ W',
         'color': plot_cfg.cp2['magenta'], 'marker': 'o', 'linestyle': '-', 'markevery': (0, 2)},
        {'result_key': 'rm_matched_aod0.05', 'label': f'RM$^{{{trained_watt}, \\mathrm{{Δε=0.05}}}}$, $P={rm05_watt}$ W',
         'color': plot_cfg.cp2['magenta'], 'marker': 's', 'linestyle': '--', 'markevery': (1, 2)},
    ]

    plot_rate_error_sweep(
        error_sweep_range=data['error_sweep_range'],
        results=data['results'],
        curves=curves,
        width=plot_width,
        height=plot_height,
        plots_parent_path=plot_cfg.plots_parent_path,
        name='error_sweep_training_triplet',
        annotate_power=False,  # power is already in the legend labels
        legend_ncols=3,
        legend_loc='lower center',
        legend_bbox_to_anchor=(0.5, 1.02),
        legend_fontsize=9,
    )
