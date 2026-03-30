
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from fractions import Fraction
import matplotlib.patches as mpatches
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


def plot_error_sweep_testing_graph(
        paths,
        name,
        width,
        height,
        plots_parent_path,
        legend: list or None = None,
        colors: list or None = None,
        markerstyle: list or None = None,
        linestyles: list or None = None,
        metric: str = 'sumrate'
) -> None:

    def get_metric_key(data_dict):

        for key_id, key in enumerate(data_dict[1].keys()):
            if match_string in str(key):
                return key
        return ValueError('match string not found')

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

    for data_id, data_entry in enumerate(data):
        metric_key = get_metric_key(data_entry)

        if markerstyle is not None:
            marker = markerstyle[data_id]
        else:
            marker = None

        if colors is not None:
            color = colors[data_id]
        else:
            color = None

        if linestyles is not None:
            linestyle = linestyles[data_id]
        else:
            linestyle = None

        # ax.errorbar(
        #     data_entry[0],
        #     data_entry[1][metric_key]['mean'],
        #     yerr=data_entry[1][metric_key]['std'],
        #     # marker=marker,
        #     color=color,
        #     linestyle=linestyle,
        #     label=legend[data_id],
        #     # solid_capstyle='round',
        #     # ecolor=change_lightness(color=color, amount=0.3),
        #     # elinewidth=2,
        #     # capthick=2,
        #     # markevery=[0, -1],
        #     # markeredgecolor='black',
        #     # fillstyle='none'
        # )

        # filled = (data_id == 3)
        step = 3
        offsets = [0, 0, 1,2,0,0,0,1,2 ]  # pro Kurve ein anderer Start-Offset

        offset = offsets[data_id] % step
        markevery = (offset, step)

        ax.plot(
            data_entry[0],
            data_entry[1][metric_key]['mean'],
            marker=marker,
            markevery=markevery,
            color=color,
            linestyle=linestyle,
            label=legend[data_id],
            fillstyle='none',
        )

    ax.set_xlabel('Error Bound $\Delta \epsilon$')

    if metric == 'sumrate':
        ax.set_ylabel('Achievable Rate $ R $ [bps/Hz]')
        ax.set_ylim(0.5, 3.5)
        # ax.set_yticks([1 / 3, 2 / 3, 1.0])
    elif metric == 'fairness':
        ax.set_ylabel('Fairness $F$')
        ax.set_ylim(0.3, 1.1)
        ax.set_yticks([1 / 3, 2 / 3, 1.0])
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda y, pos: str(Fraction(y).limit_denominator()))
        )

    if legend:
        from matplotlib import container
        handles, labels = ax.get_legend_handles_labels()
        handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]  # remove error bars
        legend = ax.legend(
            handles, legend,
            ncols=2,
            # frameon = False,
            # loc='lower left',
        )
        legend.get_frame().set_linewidth(0.8)

    # # ---- Dreiecke auf/unter der x-Achse hinzufügen (schwarz) ----
    # ymin, ymax = ax.get_ylim()
    # y_tri = ymin - 0.0 * (ymax - ymin)   # 5% unterhalb der Achse
    # x_tris = [0.00, 0.05]                 # <- hier deine x-Positionen eintragen
    #
    # ax.plot(
    #     x_tris,
    #     [y_tri] * len(x_tris),
    #     linestyle='None',
    #     marker='^',        # oder 'v' falls nach unten zeigen soll
    #     color='black',
    #     markersize=4,
    #     clip_on=False,
    #     label='_nolegend_'  # sicher nicht in Legend aufnehmen
    # )
    # ax.set_ylim(ymin, ymax)  # Limits wiederherstellen

    generic_styling(ax=ax)
    fig.tight_layout(pad=0)

    save_figures(plots_parent_path=plots_parent_path, plot_name=name+'_'+metric, padding=0.05)


if __name__ == '__main__':

    cfg = Config()
    plot_cfg = PlotConfig()

    data_paths = [
        Path(cfg.output_metrics_path,
             '01_user_distance_without_error', '02_error_sweep',
             'testing_rsma_genie_sweep_0.0_0.1.gzip'),
        Path(cfg.output_metrics_path,
             '01_user_distance_without_error', '02_error_sweep',
             'testing_learned_sweep_0.0_0.1_without_error.gzip'),
        Path(cfg.output_metrics_path,
             '01_user_distance_without_error', '02_error_sweep',
             'testing_learned_rsma_full_sweep_0.0_0.1_without_error.gzip'),
        Path(cfg.output_metrics_path,
             '01_user_distance_without_error', '02_error_sweep',
             'testing_learned_rsma_full_sweep_0.0_0.1_without_error.gzip'),
        Path(cfg.output_metrics_path,
             '01_user_distance_without_error', '02_error_sweep',
             'testing_learned_sweep_0.0_0.1_error.gzip'),
        Path(cfg.output_metrics_path,
             '01_user_distance_without_error', '02_error_sweep',
             'testing_learned_sweep_0.0_0.1_error.gzip'),
        Path(cfg.output_metrics_path,
             '01_user_distance_without_error', '02_error_sweep',
             'testing_learned_rsma_full_sweep_0.0_0.1_error.gzip'),
        Path(cfg.output_metrics_path,
             '01_user_distance_without_error', '02_error_sweep',
             'testing_learned_rsma_full_sweep_0.0_0.1_error.gzip'),
        # Path(cfg.output_metrics_path,
        #      'RSMA_Journal', 'error_sweep',
        #      'testing_learned_rsma_full_sweep_0.0_0.25.gzip'),

    ]

    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.9

    plot_legend = [r'RSMA $\alpha*$',
                   r'L-SDMA, $\Delta \epsilon = 0$',
                   r'L-RSMA, $\Delta \epsilon = 0$',
                   r'L-RSMA LC, $\Delta \epsilon = 0$',
                   r' ',
                   r'L-RSMA, $\Delta \epsilon = 0.05$',
                   r'L-RSMA, $\Delta \epsilon = 0.05$',
                   r'L-RSMA LC, $\Delta \epsilon = 0.05$']

    plot_markerstyle = ['o','s', 'd', 'x','None', 's','d','x']
    plot_colors = [plot_cfg.cp2['black'], plot_cfg.cp3['blue2'],plot_cfg.cp3['red2'],plot_cfg.cp3['red1'], change_lightness(plot_cfg.cp3['red2'], 0) , plot_cfg.cp3['blue2'], change_lightness(plot_cfg.cp3['red2'], 1),plot_cfg.cp3['red1'] ]
    plot_linestyles = ['-','-', '-', '-', '--', '--','--','--']

    plot_error_sweep_testing_graph(
        paths=data_paths,
        metric='sumrate',
        name='error_sweep_test',
        width=plot_width,
        height=plot_height,
        legend=plot_legend,
        colors=plot_colors,
        markerstyle=plot_markerstyle,
        linestyles=plot_linestyles,
        plots_parent_path=plot_cfg.plots_parent_path,
    )
    plt.show()
