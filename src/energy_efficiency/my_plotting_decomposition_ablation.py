"""
Plot the input/output decomposition ablation (baseline vs. csi_real_imag
vs. action_rad_phase) against MMSE, using the gzip produced by
decomposition_ablation_eval.py (run that first).

One figure: error_sweep_decomposition_ablation_sumrate.pdf -- rate-vs-error,
full-budget MMSE as context plus, for each of the three corners, its own
SAC curve against MMSE rescaled to that corner's own measured power (same
matched-power convention as the triplet's compact figure).

Per explicit user policy (handoff 2026-08-06): this plot exists to check
whether either ablation beats the baseline -- if neither does, keep the
baseline as the working line.
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

KEYS = ['baseline', 'csi_real_imag', 'action_rad_phase']
COLOR_NAMES = {'baseline': 'purple1', 'csi_real_imag': 'green', 'action_rad_phase': 'magenta'}


if __name__ == '__main__':
    cfg = Config()
    plot_cfg = PlotConfig()
    matplotlib.rcParams['text.usetex'] = False  # override PlotConfig's reset
    palette = {**plot_cfg.cp2, **plot_cfg.cp3}  # COLOR_NAMES draws from both

    gzip_path = Path(cfg.output_metrics_path, 'EE_decomposition_ablation', 'decomposition_ablation.gzip')
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

    for key in KEYS:
        color = palette[COLOR_NAMES[key]]
        sac = results[f'sac_{key}']
        matched = results[f'mmse_matched_{key}']
        ax.plot(error_sweep_range, sac['mean_rate'], color=color, marker='o',
                linestyle='-', linewidth=1.5, markersize=5, label=sac['label'])
        ax.plot(error_sweep_range, matched['mean_rate'], color=color, marker='x',
                linestyle='--', linewidth=1.2, markersize=5, label=f"MMSE eq.pow ({key})")

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
    out = Path(pdf_path, 'error_sweep_decomposition_ablation_sumrate.pdf')
    plt.savefig(out, bbox_inches='tight', dpi=800, transparent=True)
    plt.close(fig)
    print(f'Saved: {out}')

    # ---- console summary: does either ablation beat the baseline? --------
    baseline_rate = results['sac_baseline']['mean_rate']
    for key in ['csi_real_imag', 'action_rad_phase']:
        delta = results[f'sac_{key}']['mean_rate'] - baseline_rate
        print(f'[{key}] rate delta vs. baseline (per error point, bps/Hz): '
              f'{", ".join(f"{d:+.3f}" for d in delta)} '
              f'(mean {delta.mean():+.3f})')
