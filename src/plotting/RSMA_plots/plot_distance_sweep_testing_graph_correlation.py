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
        power_factor_from_path_idx: int = 4,
) -> None:



    y_label = 'Channel Correlation '

    # --- load all data
    data = []
    for path in paths:
        with gzip.open(path, 'rb') as f:
            data.append(pickle.load(f))   # [x, metrics_dict]



    fig, ax = plt.subplots(figsize=(width, height))

    # --- main metric curves (bottom axis)
    step = 14
    offsets = [0, 0, 0, 0, 7, 0]

    for data_id, (x, metrics_dict) in enumerate(data):
        # metric_key = get_metric_key(metrics_dict, match_string)

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
            # metrics_dict[metric_key]['mean'],
            metrics_dict['correlation']['mean'],
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

    if legend:
        ax.legend(legend, ncols=1,loc='upper right')


    # # --- alpha curve (top axis) from selected dataset (genie)

    generic_styling(ax=ax)
    fig.tight_layout(pad=0)


    save_figures(plots_parent_path=plots_parent_path, plot_name=name + '_' , padding=0.05)


if __name__ == '__main__':
    cfg = Config()
    plot_cfg = PlotConfig()

    data_paths = [
        Path(cfg.output_metrics_path, 'channel_correlation', 'user_distance_sweep',
             '1sat_8ant_100k_500-50k_0_1.gzip'),
        Path(cfg.output_metrics_path, 'channel_correlation', 'user_distance_sweep',
             '1sat_8ant_100k_500-50k_0_2.gzip'),
        # Path(cfg.output_metrics_path, 'channel_correlation', 'user_distance_sweep',
        #      '1sat_8ant_100k_500-50k_1_2.gzip'),
        # Path(cfg.output_metrics_path, 'channel_correlation', 'user_distance_sweep',
        #      '1sat_8ant_100k_500-50k_overall.gzip'),

    ]

    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.6

    plot_legend = [
        r'outer user and inner user',
        r'both outer user',
        # r'outer user and inner use',
        # r'overall',

    ]

    plot_markerstyle = [None, None, 'x', 'o', 'x', 'o']

    plot_colors = [
        change_lightness(plot_cfg.cp3['red1'], 1),
        change_lightness(plot_cfg.cp3['red1'], 0.8),
        change_lightness(plot_cfg.cp3['red2'], 1),
        plot_cfg.cp2['black'],
    ]

    plot_linestyles = ['--', '-.', '--', '-', '-', '-']

    plot_distance_sweep_testing_graph(
        paths=data_paths,
        name='dist_sweep_correlation',
        width=plot_width,
        height=plot_height,
        legend=plot_legend,
        colors=plot_colors,
        markerstyle=plot_markerstyle,
        linestyles=plot_linestyles,
        plots_parent_path=plot_cfg.plots_parent_path,
        power_factor_from_path_idx=4,
    )

    plt.show()