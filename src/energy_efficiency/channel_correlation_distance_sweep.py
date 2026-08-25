"""
Runs the existing src/analysis/test_channel_correlation_distance_sweep.py
sweep ("for our case"): N16K3 scenario, all 3 user pairs, averaged, plotted
against user distance -- with our actual operating point (user_dist_average
= 100 km, the paper-matching config.py default) marked on the plot.

Reuses test_channel_correlation_user_sweep() unmodified. Only two things
differ from that file's own __main__ block:
  1. Headless-safe: matplotlib.use('Agg') + savefig instead of plt.show()
     (the original script has neither -- would hang forever in a SLURM job,
     same TkAgg gotcha hit earlier this session).
  2. Coarser distance grid: 40 points (step 2500 m) instead of 200 points
     (step 500 m). At ~24ms/update_sim (measured from
     channel_correlation_zero_error.py's job), the original 500m-step,
     1000-iteration, 3-pair sweep is ~600k channel draws (~4 hours); 2500m
     step brings that to ~120k draws (~45 min), still fine resolution for a
     smooth mean-correlation-vs-distance curve. monte_carlo_iterations
     (1000) kept at the original script's own default.

disable_wiggle=True kept (matches the original script's own convention) --
this measures how correlation depends purely on mean distance, holding the
per-draw roam variance out of it; our scenario's actual roam
(user_dist_bound=0.5) is a separate, already-covered question (see
channel_correlation_zero_error.py, run at our real operating point with
wiggle enabled).
"""
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config.config import Config
from src.config.config_plotting import PlotConfig
from src.analysis.test_channel_correlation_distance_sweep import test_channel_correlation_user_sweep

distance_sweep_range = np.arange(500, 100500, 2500)
monte_carlo_iterations = 1000
user_1_id, user_2_id, user_3_id = 0, 1, 2
disable_wiggle = True

if __name__ == '__main__':
    cfg = Config()
    print(f'Scenario: N={cfg.sat_tot_ant_nr} antennas, K={cfg.user_nr} users, '
          f'our operating point user_dist_average={cfg.user_dist_average} m')
    print(f'Sweeping user_dist_average over {distance_sweep_range[0]:.0f}-{distance_sweep_range[-1]:.0f} m '
          f'({len(distance_sweep_range)} points, {monte_carlo_iterations} MC iterations each, disable_wiggle={disable_wiggle})')

    mean_corr_12 = test_channel_correlation_user_sweep(
        distance_sweep_range, user_1_id, user_2_id, disable_wiggle, monte_carlo_iterations)
    mean_corr_13 = test_channel_correlation_user_sweep(
        distance_sweep_range, user_1_id, user_3_id, disable_wiggle, monte_carlo_iterations)
    mean_corr_23 = test_channel_correlation_user_sweep(
        distance_sweep_range, user_2_id, user_3_id, disable_wiggle, monte_carlo_iterations)

    mean_overall = (mean_corr_12 + mean_corr_13 + mean_corr_23) / 3

    our_distance = cfg.user_dist_average
    our_idx = int(np.argmin(np.abs(distance_sweep_range - our_distance)))
    print(f'\nAt our operating distance (~{our_distance:.0f} m): mean channel correlation = {mean_overall[our_idx]:.4f}')

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(distance_sweep_range, mean_overall, color='#307b3b', linewidth=1.8,
            label='Mean channel correlation (3 user pairs)')
    ax.axvline(our_distance, color='#c9622a', linestyle='--', linewidth=1.3,
               label=f'Our scenario (user_dist_average = {our_distance/1000:.0f} km)')
    ax.set_xlabel('User Distance [m]')
    ax.set_ylabel('Mean Channel Correlation |ρ|')
    ax.set_title('Channel correlation vs. user distance (disable_wiggle=True)')
    ax.legend(loc='upper right', fontsize=9, framealpha=1.0)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()

    plot_cfg = PlotConfig()
    pdf_path = Path(plot_cfg.plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)
    out = Path(pdf_path, 'channel_correlation_distance_sweep.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=300, transparent=True)
    print(f'Saved: {out}')

    jpg_path = Path(plot_cfg.plots_parent_path, 'jpg')
    jpg_path.mkdir(parents=True, exist_ok=True)
    out_jpg = Path(jpg_path, 'channel_correlation_distance_sweep.jpg')
    fig.savefig(out_jpg, bbox_inches='tight', dpi=200)
    print(f'Saved: {out_jpg}')

    png_path = Path(plot_cfg.plots_parent_path, 'png')
    png_path.mkdir(parents=True, exist_ok=True)
    out_png = Path(png_path, 'channel_correlation_distance_sweep.png')
    fig.savefig(out_png, bbox_inches='tight', dpi=200, transparent=True)
    print(f'Saved: {out_png}')
