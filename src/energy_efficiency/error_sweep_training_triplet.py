"""
Rate-vs-error figure for three training-error SAC checkpoints (Δε =
0.00/0.025/0.05, energy-efficient/clip-only), each shown twice: once
evaluated at its own EE (clip-only) power, once re-evaluated at full
75 W-budget power -- a separate figure from plotting_scenario.py's 5-curve
MMSE/SAC full-vs-matched-power comparison, answering a different question:
how does training at a larger CSIT error bound trade off rate for
robustness, and how much rate is left on the table if that checkpoint were
run at full power instead of its own EE operating point.

Plotting-only: reuses the gzip plotting_scenario.py already produces
(outputs/metrics/EE_lwin5000_3gpp_triplet/rate_power_triplet.gzip), no new
Monte Carlo simulation. Run plotting_scenario.py first if that gzip
doesn't exist yet, or is missing the aod0.025/aod0.05 checkpoints' data
(requires those checkpoints under models/ at the time plotting_scenario.py
was run), then add_fullpower_curve.py aod0.0 / aod0.025 / aod0.05 (or their
.slurm files) to add the 'sac_{aod_key}_fullpower' curves this figure needs.

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

    curves = [
        # interleaved (full-power, own-power) per Δε so the 3-column legend
        # fills column-major into matching color pairs -- top row green/
        # blue/magenta dashed (full power), bottom row the same colors
        # solid (own power).
        {'result_key': 'sac_aod0.0_fullpower', 'label': f'EE$^{{{trained_watt}, \\mathrm{{Δε=0.00}}}}$, $P={trained_watt}$ W',
         'color': plot_cfg.cp2['green'], 'marker': 'D', 'linestyle': '-.'},
        {'result_key': 'sac_aod0.0', 'label': f'EE$^{{{trained_watt}, \\mathrm{{Δε=0.00}}}}$, $P={aod0_watt}$ W',
         'color': plot_cfg.cp2['green'], 'marker': 'o', 'linestyle': '-'},
        {'result_key': 'sac_aod0.025_fullpower', 'label': f'EE$^{{{trained_watt}, \\mathrm{{Δε=0.025}}}}$, $P={trained_watt}$ W',
         'color': plot_cfg.cp2['blue'], 'marker': 'D', 'linestyle': '-.'},
        {'result_key': 'sac_aod0.025', 'label': f'EE$^{{{trained_watt}, \\mathrm{{Δε=0.025}}}}$, $P={aod0025_watt}$ W',
         'color': plot_cfg.cp2['blue'], 'marker': 'o', 'linestyle': '-'},
        {'result_key': 'sac_aod0.05_fullpower', 'label': f'EE$^{{{trained_watt}, \\mathrm{{Δε=0.05}}}}$, $P={trained_watt}$ W',
         'color': plot_cfg.cp2['magenta'], 'marker': 'D', 'linestyle': '-.'},
        {'result_key': 'sac_aod0.05', 'label': f'EE$^{{{trained_watt}, \\mathrm{{Δε=0.05}}}}$, $P={aod05_watt}$ W',
         'color': plot_cfg.cp2['magenta'], 'marker': 'o', 'linestyle': '-'},
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
