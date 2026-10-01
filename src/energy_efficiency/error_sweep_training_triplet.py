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
    plot_height = plot_width * 0.7

    # Notation matches plotting_scenario.py's error-sweep figure: superscript is
    # each curve's measured evaluation power; subscript is the fixed training-time
    # power budget (same for every checkpoint here) and its training error bound
    # Delta-epsilon -- watts pulled from the actual measured samples, not hardcoded.
    trained_watt = round(data['results']['sac_aod0.0']['power_budget'])
    aod0_watt = round(data['results']['sac_aod0.0']['mean_power'][0])
    aod0025_watt = round(data['results']['sac_aod0.025']['mean_power'][0])
    aod05_watt = round(data['results']['sac_aod0.05']['mean_power'][0])

    rm0_watt = round(data['results']['rm_matched_aod0.0']['mean_power'][0])
    rm0025_watt = round(data['results']['rm_matched_aod0.025']['mean_power'][0])
    rm05_watt = round(data['results']['rm_matched_aod0.05']['mean_power'][0])

    # EE vs RM in each colour-pair is told apart by MARKER, not line style (the
    # dashed line was illegible in the legend). Each of the 6 curves gets a
    # distinct marker, and the pair's markevery is staggered (0,2)/(1,2) so the
    # two markers never land on top of each other. EE dashed / RM solid, matching
    # plotting_scenario.py.
    curves = [
        {'result_key': 'sac_aod0.0', 'label': f'EE$^{{{aod0_watt}\\,\\mathrm{{W}}}}_{{{trained_watt}\\,\\mathrm{{W}},\\,\\Delta\\epsilon=0.00}}$',
         'color': plot_cfg.cp2['green'], 'marker': 'o', 'linestyle': '--', 'markevery': (0, 2)},
        {'result_key': 'rm_matched_aod0.0', 'label': f'RM$^{{{rm0_watt}\\,\\mathrm{{W}}}}_{{{trained_watt}\\,\\mathrm{{W}},\\,\\Delta\\epsilon=0.00}}$',
         'color': plot_cfg.cp2['green'], 'marker': 's', 'linestyle': '-', 'markevery': (1, 2)},
        {'result_key': 'sac_aod0.025', 'label': f'EE$^{{{aod0025_watt}\\,\\mathrm{{W}}}}_{{{trained_watt}\\,\\mathrm{{W}},\\,\\Delta\\epsilon=0.025}}$',
         'color': plot_cfg.cp2['blue'], 'marker': '^', 'linestyle': '--', 'markevery': (0, 2)},
        {'result_key': 'rm_matched_aod0.025', 'label': f'RM$^{{{rm0025_watt}\\,\\mathrm{{W}}}}_{{{trained_watt}\\,\\mathrm{{W}},\\,\\Delta\\epsilon=0.025}}$',
         'color': plot_cfg.cp2['blue'], 'marker': 'v', 'linestyle': '-', 'markevery': (1, 2)},
        {'result_key': 'sac_aod0.05', 'label': f'EE$^{{{aod05_watt}\\,\\mathrm{{W}}}}_{{{trained_watt}\\,\\mathrm{{W}},\\,\\Delta\\epsilon=0.05}}$',
         'color': plot_cfg.cp2['magenta'], 'marker': 'D', 'linestyle': '--', 'markevery': (0, 2)},
        {'result_key': 'rm_matched_aod0.05', 'label': f'RM$^{{{rm05_watt}\\,\\mathrm{{W}}}}_{{{trained_watt}\\,\\mathrm{{W}},\\,\\Delta\\epsilon=0.05}}$',
         'color': plot_cfg.cp2['magenta'], 'marker': 'X', 'linestyle': '-', 'markevery': (1, 2)},
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
        legend_fontsize=11,
        legend_handlelength=3.0,
    )
