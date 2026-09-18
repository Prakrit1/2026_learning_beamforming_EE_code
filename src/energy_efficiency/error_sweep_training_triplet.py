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

    # Both EE and RM curves are evaluated at their OWN energy-efficient
    # (clip-only) power -- there is no fixed 75 W operating point here. The
    # superscript is just the training error bound Delta-eps; the post-comma
    # value is each curve's own measured energy-efficient power (three
    # different powers for EE, three for RM), pulled from the measured samples.
    aod0_watt = round(data['results']['sac_aod0.0']['mean_power'][0])
    aod0025_watt = round(data['results']['sac_aod0.025']['mean_power'][0])
    aod05_watt = round(data['results']['sac_aod0.05']['mean_power'][0])

    rm0_watt = round(data['results']['rm_matched_aod0.0']['mean_power'][0])
    rm0025_watt = round(data['results']['rm_matched_aod0.025']['mean_power'][0])
    rm05_watt = round(data['results']['rm_matched_aod0.05']['mean_power'][0])

    curves = [
        {'result_key': 'sac_aod0.0', 'label': f'EE$^{{\\mathrm{{Δε=0.00}}}}$, $P={aod0_watt}$ W',
         'color': plot_cfg.cp2['green'], 'marker': 'o', 'linestyle': '-', 'markevery': (0, 2)},
        {'result_key': 'rm_matched_aod0.0', 'label': f'RM$^{{\\mathrm{{Δε=0.00}}}}$, $P={rm0_watt}$ W',
         'color': plot_cfg.cp2['green'], 'marker': 's', 'linestyle': '--', 'markevery': (1, 2)},
        {'result_key': 'sac_aod0.025', 'label': f'EE$^{{\\mathrm{{Δε=0.025}}}}$, $P={aod0025_watt}$ W',
         'color': plot_cfg.cp2['blue'], 'marker': 'o', 'linestyle': '-', 'markevery': (0, 2)},
        {'result_key': 'rm_matched_aod0.025', 'label': f'RM$^{{\\mathrm{{Δε=0.025}}}}$, $P={rm0025_watt}$ W',
         'color': plot_cfg.cp2['blue'], 'marker': 's', 'linestyle': '--', 'markevery': (1, 2)},
        {'result_key': 'sac_aod0.05', 'label': f'EE$^{{\\mathrm{{Δε=0.05}}}}$, $P={aod05_watt}$ W',
         'color': plot_cfg.cp2['magenta'], 'marker': 'o', 'linestyle': '-', 'markevery': (0, 2)},
        {'result_key': 'rm_matched_aod0.05', 'label': f'RM$^{{\\mathrm{{Δε=0.05}}}}$, $P={rm05_watt}$ W',
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
