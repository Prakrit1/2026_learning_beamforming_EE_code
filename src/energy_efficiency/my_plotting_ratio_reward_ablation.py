"""
Plot the raw-ratio-reward-vs-Dinkelbach-adaptive-lambda ablation, using the
gzip produced by ratio_reward_ablation_eval.py (run that first).

One figure: error_sweep_ratio_reward_ablation_sumrate.pdf -- rate-vs-error,
full-budget MMSE as context, plus for aod=0.0 and aod=0.05 a matched pair of
curves (ratio-reward vs. the working-line lambda checkpoint, same color,
solid/dashed) and for aod=0.035 the ratio-reward curve alone (no matched
lambda checkpoint exists at that error bound -- see the eval script's
docstring).

Console summary: rate delta (ratio-reward minus lambda) per matched aod
bound, across the whole error sweep -- does raw-ratio-reward come anywhere
close to the working-line Dinkelbach mechanism, or does the pricing term
matter?
"""
import gzip
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt

from src.config.config import Config
from src.config.config_plotting import PlotConfig

MATCHED_AOD_KEYS = ['aod0.0', 'aod0.05']
STANDALONE_AOD_KEYS = ['aod0.035']
COLOR_NAMES = {'aod0.0': 'blue1', 'aod0.05': 'green1', 'aod0.035': 'orange1'}


if __name__ == '__main__':
    cfg = Config()
    plot_cfg = PlotConfig()
    matplotlib.rcParams['text.usetex'] = False  # override PlotConfig's reset
    palette = {**plot_cfg.cp2, **plot_cfg.cp3}  # COLOR_NAMES draws from both

    gzip_path = Path(cfg.output_metrics_path, 'EE_ratio_reward_ablation', 'ratio_reward_ablation.gzip')
    with gzip.open(gzip_path, 'rb') as file:
        data = pickle.load(file)
    error_sweep_range = data['error_sweep_range']
    results = data['results']

    pdf_path = Path(plot_cfg.plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)

    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.6

    fig, ax = plt.subplots(figsize=(plot_width, plot_height))

    mmse = results['mmse_nadir']
    ax.plot(error_sweep_range, mmse['mean_rate'], color=plot_cfg.cp2['gold'],
            marker='v', linestyle='-', linewidth=1.5, markersize=5, label='MMSE (full budget)')

    for aod_key in MATCHED_AOD_KEYS:
        color = palette[COLOR_NAMES[aod_key]]
        ratio = results[f'sac_ratio_{aod_key}']
        lambda_sac = results[f'sac_lambda_{aod_key}']
        ax.plot(error_sweep_range, ratio['mean_rate'], color=color, marker='o',
                linestyle='-', linewidth=1.5, markersize=5, label=ratio['label'])
        ax.plot(error_sweep_range, lambda_sac['mean_rate'], color=color, marker='s',
                linestyle='--', linewidth=1.2, markersize=5, label=lambda_sac['label'])

    for aod_key in STANDALONE_AOD_KEYS:
        color = palette[COLOR_NAMES[aod_key]]
        ratio = results[f'sac_ratio_{aod_key}']
        ax.plot(error_sweep_range, ratio['mean_rate'], color=color, marker='o',
                linestyle='-', linewidth=1.5, markersize=5,
                label=f"{ratio['label']} (no matched lambda ckpt)")

    ax.set_xlabel('Error Bound')
    ax.set_ylabel('Rate R [bps/Hz]')
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(
        loc='upper right',
        ncols=1,
        fontsize=6,
        framealpha=0.9,
        frameon=True,
        handlelength=1.6,
        labelspacing=0.3,
        borderpad=0.3,
        handletextpad=0.5,
    )
    plt.tight_layout(pad=0.2)
    out = Path(pdf_path, 'error_sweep_ratio_reward_ablation_sumrate.pdf')
    plt.savefig(out, bbox_inches='tight', dpi=800, transparent=True)
    plt.close(fig)
    print(f'Saved: {out}')

    # ---- console summary: how far off is raw-ratio-reward, where matched? ---
    for aod_key in MATCHED_AOD_KEYS:
        delta = results[f'sac_ratio_{aod_key}']['mean_rate'] - results[f'sac_lambda_{aod_key}']['mean_rate']
        print(f'[{aod_key}] rate delta ratio-reward minus lambda (per error point, bps/Hz): '
              f'{", ".join(f"{d:+.3f}" for d in delta)} '
              f'(mean {delta.mean():+.3f})')
    print('[aod0.035] no matched lambda checkpoint -- evaluated standalone, see gzip for raw numbers.')
