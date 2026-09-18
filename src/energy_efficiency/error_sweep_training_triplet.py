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

    # Notation matches plotting_scenario.py's error-sweep figure: the
    # superscript is the training-time power budget, the post-comma value is
    # each curve's own measured operating power -- watts pulled from the actual
    # measured samples, not hardcoded. The EE checkpoints all share the 75 W
    # budget; each per-error RM checkpoint carries its own training budget in
    # 'train_power_budget' (p35 at Delta-eps=0, p75 at 0.025/0.05).
    ee_watt = round(data['results']['sac_aod0.0']['power_budget'])
    aod0_watt = round(data['results']['sac_aod0.0']['mean_power'][0])
    aod0025_watt = round(data['results']['sac_aod0.025']['mean_power'][0])
    aod05_watt = round(data['results']['sac_aod0.05']['mean_power'][0])

    rm0 = data['results']['rm_matched_aod0.0']
    rm0025 = data['results']['rm_matched_aod0.025']
    rm05 = data['results']['rm_matched_aod0.05']
    rm0_watt = round(rm0['mean_power'][0])
    rm0025_watt = round(rm0025['mean_power'][0])
    rm05_watt = round(rm05['mean_power'][0])
    # per-curve training budget for the RM superscript (fall back to EE's if an
    # older gzip without 'train_power_budget' is being replotted)
    rm0_train = round(rm0.get('train_power_budget') or ee_watt)
    rm0025_train = round(rm0025.get('train_power_budget') or ee_watt)
    rm05_train = round(rm05.get('train_power_budget') or ee_watt)

    curves = [
        {'result_key': 'sac_aod0.0', 'label': f'EE$^{{{ee_watt}, \\mathrm{{Δε=0.00}}}}$, $P={aod0_watt}$ W',
         'color': plot_cfg.cp2['green'], 'marker': 'o', 'linestyle': '-', 'markevery': (0, 2)},
        {'result_key': 'rm_matched_aod0.0', 'label': f'RM$^{{{rm0_train}, \\mathrm{{Δε=0.00}}}}$, $P={rm0_watt}$ W',
         'color': plot_cfg.cp2['green'], 'marker': 's', 'linestyle': '--', 'markevery': (1, 2)},
        {'result_key': 'sac_aod0.025', 'label': f'EE$^{{{ee_watt}, \\mathrm{{Δε=0.025}}}}$, $P={aod0025_watt}$ W',
         'color': plot_cfg.cp2['blue'], 'marker': 'o', 'linestyle': '-', 'markevery': (0, 2)},
        {'result_key': 'rm_matched_aod0.025', 'label': f'RM$^{{{rm0025_train}, \\mathrm{{Δε=0.025}}}}$, $P={rm0025_watt}$ W',
         'color': plot_cfg.cp2['blue'], 'marker': 's', 'linestyle': '--', 'markevery': (1, 2)},
        {'result_key': 'sac_aod0.05', 'label': f'EE$^{{{ee_watt}, \\mathrm{{Δε=0.05}}}}$, $P={aod05_watt}$ W',
         'color': plot_cfg.cp2['magenta'], 'marker': 'o', 'linestyle': '-', 'markevery': (0, 2)},
        {'result_key': 'rm_matched_aod0.05', 'label': f'RM$^{{{rm05_train}, \\mathrm{{Δε=0.05}}}}$, $P={rm05_watt}$ W',
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
