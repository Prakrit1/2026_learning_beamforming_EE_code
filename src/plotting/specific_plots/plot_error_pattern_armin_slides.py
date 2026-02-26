
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
from gzip import (
    open as gzip_open,
)
from pickle import (
    load as pickle_load,
)
from pathlib import (
    Path,
)

from src.config.config import (
    Config,
)
from src.config.config_plotting import (
    PlotConfig,
    save_figures,
    generic_styling,
    change_lightness,
)


def export_curves_to_csv(data, metric_key_fn, out_csv: Path):
    x0 = np.asarray(data[0][0], dtype=float)
    ys = []
    for d in data:
        x = np.asarray(d[0], dtype=float)
        if x.shape != x0.shape or np.nanmax(np.abs(x - x0)) > 1e-12:
            raise ValueError("x-Achsen sind nicht identisch.")
        key = metric_key_fn(d)
        ys.append(np.asarray(d[1][key]["mean"], dtype=float))

    M = np.column_stack([x0] + ys)
    header = ["x"] + [f"y{i+1}" for i in range(len(ys))]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_csv, M, delimiter=",", header=",".join(header), comments="")
    return out_csv

def plot_error_sweep_testing_graph(
        paths,
        name,
        width,
        height,
        xlabel,
        ylabel,
        plots_parent_path,
        legend: list or None = None,
        colors: list or None = None,
        markerstyle: list or None = None,
        linestyles: list or None = None,
        metric: str = 'sumrate',
        visible_mask: list[bool] | None = None,   # <--- NEU
        legend_alpha_off: float = 0.15,           # <--- NEU
) -> None:

    def get_metric_key(data_dict):
        for key in data_dict[1].keys():
            if match_string in str(key):
                return key
        raise ValueError('match string not found')

    fig, ax = plt.subplots(figsize=(width, height))

    if metric == 'sumrate':
        match_string = 'calc_sum_rate'
    elif metric == 'fairness':
        match_string = 'calc_jain_fairness'
    else:
        raise ValueError(f'unknown metric {metric}')

    data = []
    for path in paths:
        with gzip_open(path, 'rb') as file:
            data.append(pickle_load(file))

    # Wenn nichts angegeben: alles sichtbar
    if visible_mask is None:
        visible_mask = [True] * len(data)

    if len(visible_mask) != len(data):
        raise ValueError("visible_mask muss die gleiche Länge haben wie paths/data.")

    out_csv = Path(plots_parent_path) / f"{name}_{metric}.csv"
    export_curves_to_csv(
        data=data,
        metric_key_fn=get_metric_key,
        out_csv=out_csv,
    )

    lines = []
    for data_id, data_entry in enumerate(data):
        metric_key = get_metric_key(data_entry)

        marker = markerstyle[data_id] if markerstyle is not None else None
        color = colors[data_id] if colors is not None else None
        linestyle = linestyles[data_id] if linestyles is not None else None

        show = bool(visible_mask[data_id])

        # Kurve ausblenden -> alpha=0 (bleibt aber als Handle für Legende vorhanden)
        line_alpha = 1.0 if show else 0.0

        (line,) = ax.plot(
            data_entry[0],
            data_entry[1][metric_key]['mean'],
            marker=marker,
            color=color,
            linestyle=linestyle,
            label=legend[data_id] if legend is not None else None,
            fillstyle='none',
            markevery=1,
            alpha=line_alpha,
        )
        lines.append(line)

    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)

    if legend:
        leg = ax.legend(ncols=1, fontsize=8)

        # Legende: Einträge für ausgeblendete Reihen transparent machen
        for i, (legline, legtext) in enumerate(zip(leg.get_lines(), leg.get_texts())):
            a = 1.0 if visible_mask[i] else legend_alpha_off
            legline.set_alpha(a)
            legtext.set_alpha(a)

    # Pfeile/Annotationen wie bei dir
    arr = mpatches.FancyArrowPatch(
        (0.095, 1), (0.095, 3.5),
        arrowstyle='-|>',
        mutation_scale=15,
        color='darkgrey',
    )
    ax.add_patch(arr)
    ax.annotate(
        'better',
        (1.0, .7),
        xycoords=arr,
        ha='left',
        va='center',
        rotation=90,
        fontsize=8,
        color=change_lightness('black', 0.7),
    )

    arr2 = mpatches.FancyArrowPatch(
        (0, 0.7), (0.10, 0.7),
        arrowstyle='-|>',
        mutation_scale=15,
        color='darkgrey',
    )
    ax.add_patch(arr2)
    ax.annotate(
        'increasing error on position',
        (0.5, 1.0),
        xycoords=arr2,
        ha='center',
        va='bottom',
        fontsize=8,
        color=change_lightness('black', 0.7),
    )

    generic_styling(ax=ax)
    fig.tight_layout(pad=0)

    save_figures(plots_parent_path=plots_parent_path, plot_name=name, padding=0.05)


if __name__ == '__main__':

    cfg = Config()
    plot_cfg = PlotConfig()

    data_paths = [
        Path(cfg.output_metrics_path,
             'ArminFolien', 'error_sweep',
             'testing_mmse_sweep_0.0_0.1.gzip'),
        # Path(cfg.output_metrics_path,
        #      'ArminFolien', 'error_sweep',
        #      'testing_robust_slnr_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path,
             'ArminFolien', 'error_sweep',
             'testing_learned_sweep_0.0_0.1_error_0.gzip'),
        # Path(cfg.output_metrics_path,
        #      'ArminFolien', 'error_sweep',
        #      'testing_learned_sweep_0.0_0.1_error_0.025.gzip'),
        Path(cfg.output_metrics_path,
             'ArminFolien', 'error_sweep',
             'testing_learned_sweep_0.0_0.1.gzip'),
    ]

    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.64

    plot_legend = [
        'Baseline',
        # 'SLNR',
        'Learned $\Delta\epsilon=0.0$',
        # 'Learned $\Delta\epsilon=0.025$',
        'Learned $\Delta\epsilon=0.05$',
    ]

    plot_markerstyle = [
        'o',
        # 'x',
        's',
        # 'd',
        'D',
    ]
    plot_colors = [
        plot_cfg.cp2['black'],
        # plot_cfg.cp3['green1'],
        plot_cfg.cp3['blue2'],
        # plot_cfg.cp3['red3'],
        plot_cfg.cp3['red2'],
    ]
    plot_linestyles = [
        '-',
        '-',
        '-',
        '-',
        '-',
    ]

    plot_error_sweep_testing_graph(
        paths=data_paths,
        name='error_sweep_1sat_1',
        width=plot_width,
        height=plot_height,
        xlabel='Error Bound $\Delta\\varepsilon$',
        ylabel='Achievable Rate (bits/s/Hz)',
        legend=plot_legend,
        colors=plot_colors,
        markerstyle=plot_markerstyle,
        linestyles=plot_linestyles,
        plots_parent_path=plot_cfg.plots_parent_path,
        visible_mask=[True, False, False],
    )
    plt.show()

