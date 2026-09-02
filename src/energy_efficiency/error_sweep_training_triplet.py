"""
Rate-vs-error figure for two training-error SAC checkpoints (Δε =
0.025/0.05, energy-efficient/clip-only) plus the Δε=0.025 checkpoint
re-evaluated at full-power inference -- a separate figure from
plotting_scenario.py's 4-curve MMSE/SAC full-vs-matched-power comparison,
answering a different question: how does training at a larger CSIT error
bound trade off rate for robustness, and how much power does each variant
actually use.

Plotting-only: reuses the gzip plotting_scenario.py already produces
(outputs/metrics/EE_lwin5000_3gpp_triplet/rate_power_triplet.gzip), no new
Monte Carlo simulation. Run plotting_scenario.py first if that gzip
doesn't exist yet, or is missing the aod0.025/aod0.05 checkpoints' data
(requires those checkpoints under models/ at the time plotting_scenario.py
was run), then add_fullpower_curve.py aod0.025 (or its .slurm) to add the
'sac_aod0.025_fullpower' curve this figure also needs.

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

    # Notation from the paper's macros (99_custommacros.sty), matching
    # plotting_scenario.py's error-sweep figure -- watts pulled from the
    # actual measured samples, not hardcoded.
    aod0025_watt = round(data['results']['sac_aod0.025']['mean_power'][0])
    aod05_watt = round(data['results']['sac_aod0.05']['mean_power'][0])
    p_rad = r'$P_{\mathrm{rad}}$'

    curves = [
        {'result_key': 'sac_aod0.025', 'label': f'EE, $P={aod0025_watt}$ W',
         'color': plot_cfg.cp2['blue'], 'marker': 'o', 'linestyle': '-'},
        {'result_key': 'sac_aod0.025_fullpower', 'label': f'EE, $P={aod0025_watt}$ W, {p_rad}',
         'color': plot_cfg.cp2['gold'], 'marker': 'D', 'linestyle': '-.'},
        {'result_key': 'sac_aod0.05', 'label': f'EE, $P={aod05_watt}$ W',
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
        annotate_power=True,
    )
