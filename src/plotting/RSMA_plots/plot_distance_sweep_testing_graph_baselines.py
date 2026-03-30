import gzip
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

from src.config.config import Config
from src.config.config_plotting import (
    PlotConfig,
    save_figures,
    generic_styling,
    change_lightness,
)


def plot_distance_sweep_testing_graph(
        paths,
        name,
        width,
        height,
        plots_parent_path,
        legend: list | None = None,
        colors: list | None = None,
        markerstyle: list | None = None,
        linestyles: list | None = None,
        metric: str = 'sumrate',
        power_factor_from_path_idx: int = 0,
        plot_power_factor: bool = True,
) -> None:

    def get_metric_key(metrics_dict, match_string):
        for key in metrics_dict.keys():
            if match_string in str(key):
                return key
        raise KeyError(f'match string not found: {match_string}. Available: {list(metrics_dict.keys())}')

    def get_power_factor_key(metrics_dict):
        for cand in ['power_factor', 'power factor', 'powerFactor', 'alpha', 'rsma_alpha']:
            if cand in metrics_dict:
                return cand
        raise KeyError(f'No power factor key found. Available: {list(metrics_dict.keys())}')

    if metric == 'sumrate':
        match_string = 'calc_sum_rate'
        y_label = 'Achievable Rate $R$ [bps/Hz]'
    elif metric == 'fairness':
        match_string = 'calc_jain_fairness'
        y_label = 'Fairness $F$'
        plot_power_factor = False
    else:
        raise ValueError(f'unknown metric {metric}')

    # --- load all data
    data = []
    for path in paths:
        with gzip.open(path, 'rb') as f:
            data.append(pickle.load(f))   # [x, metrics_dict]

    # --- figure with two rows, shared x
    if plot_power_factor:
        fig, (ax_alpha, ax) = plt.subplots(
            2, 1, sharex=True,
            figsize=(width, height),
            gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.05},
            constrained_layout=False,
        )
        ax_alpha.tick_params(labelbottom=False)
    else:
        fig, ax = plt.subplots(figsize=(width, height), constrained_layout=False)
        ax_alpha = None

    # --- main metric curves (bottom axis)
    step = 14
    offsets = [0, 0, 0, 0, 7, 0]

    for data_id, (x, metrics_dict) in enumerate(data):
        metric_key = get_metric_key(metrics_dict, match_string)

        marker = markerstyle[data_id] if markerstyle is not None else None
        if marker in ('None', 'none'):
            marker = None

        color = colors[data_id] if colors is not None else None
        linestyle = linestyles[data_id] if linestyles is not None else None

        offset = offsets[data_id] % step if data_id < len(offsets) else (data_id * 3) % step
        markevery = (offset, step)

        hollow = (marker not in (None, '') and marker != 'x')
        mfc = 'none' if hollow else None

        # Genie betonen
        is_genie = (data_id == power_factor_from_path_idx)
        lw = 2.0 if is_genie else 1.4

        ax.plot(
            x,
            metrics_dict[metric_key]['mean'],
            color=color,
            linestyle=linestyle,
            linewidth=lw,
            marker=marker,
            markevery=markevery,
            markerfacecolor=mfc,
            markeredgecolor=color,
            markeredgewidth=1.1,
        )

    ax.set_ylabel(y_label)
    ax.set_xlabel('User Distance $d_\mathcal{K}$ [km]')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{x / 1000:g}'))

    # --- alpha curve (top axis) from selected dataset (genie)
    if plot_power_factor:
        x_pf, metrics_pf = data[power_factor_from_path_idx]
        pf_key = get_power_factor_key(metrics_pf)
        alpha = metrics_pf[pf_key]['mean'] if isinstance(metrics_pf[pf_key], dict) else metrics_pf[pf_key]

        ax_alpha.plot(
            x_pf, alpha,
            linestyle='None',
            marker='o',
            markersize=4,
            color='black',
            markerfacecolor='none',
            markeredgewidth=1.0,
        )
        ax_alpha.set_ylabel(r'$\alpha*$')
        ax_alpha.set_ylim(0.0, 1.2)
        ax_alpha.set_yticks([0.0, 0.5, 1.0])

    # --- styling first (optional)
    generic_styling(ax=ax)
    if plot_power_factor:
        generic_styling(ax=ax_alpha)

    # --- Legend BELOW the plot (figure-level)
    if legend:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(legend,
                  loc='upper center',
                  bbox_to_anchor=(0.48, -0.2),
                  fancybox=True,
                  ncol=3)

        fig.subplots_adjust(bottom=0.25)     # Space for legends

    # --- also give a bit of top margin (prevents cutting)
    fig.subplots_adjust(top=0.98)

    save_figures(plots_parent_path=plots_parent_path, plot_name=name + '_' + metric, padding=0.05)


if __name__ == '__main__':
    cfg = Config()
    plot_cfg = PlotConfig()

    data_paths = [
        Path(cfg.output_metrics_path, '01_user_distance_without_error', '00_baselines',
             'testing_rsma_usersweep_500_50000_alpha_0.gzip'),
        Path(cfg.output_metrics_path, '01_user_distance_without_error', '00_baselines',
             'testing_rsma_usersweep_500_50000_alpha_0.25.gzip'),
        Path(cfg.output_metrics_path, '01_user_distance_without_error', '00_baselines',
             'testing_rsma_usersweep_500_50000_alpha_0.5.gzip'),
        Path(cfg.output_metrics_path, '01_user_distance_without_error', '00_baselines',
             'testing_rsma_usersweep_500_50000_alpha_0.75.gzip'),
        Path(cfg.output_metrics_path, '01_user_distance_without_error', '00_baselines',
             'testing_mmse_usersweep_500_50000.gzip'),
        Path(cfg.output_metrics_path, '01_user_distance_without_error', '00_baselines',
             'testing_rsma_genie_sweep_500_50000.gzip'),
    ]

    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 1

    plot_legend = [
        r'RSMA $\alpha=0$',
        r'RSMA $\alpha=1/4$',
        r'RSMA $\alpha=1/2$',
        r'RSMA $\alpha=3/4$',
        r'MMSE ',
        r'RSMA $\alpha*$',
    ]

    plot_markerstyle = [None, None, None, None, 'x', 'o']

    plot_colors = [
        change_lightness(plot_cfg.cp3['red1'], 1),
        change_lightness(plot_cfg.cp3['red1'], 0.8),
        change_lightness(plot_cfg.cp3['red2'], 1),
        change_lightness(plot_cfg.cp3['red2'], 0.8),
        change_lightness(plot_cfg.cp3['blue2'], 1),
        plot_cfg.cp2['black'],
    ]

    plot_linestyles = ['-', '-.', '--', ':', '-', '-']

    plot_distance_sweep_testing_graph(
        paths=data_paths,
        metric='sumrate',
        name='dist_sweep_test_long',
        width=plot_width,
        height=plot_height,
        legend=plot_legend,
        colors=plot_colors,
        markerstyle=plot_markerstyle,
        linestyles=plot_linestyles,
        plots_parent_path=plot_cfg.plots_parent_path,
        power_factor_from_path_idx=5,
        plot_power_factor=True,
    )

    plt.show()