"""
Beampattern generation + beamwidth/coverage quantification for the full
lwin5000/3GPP-native nadir triplet -- aod=0.0 (job 156358), aod=0.025
(first-ever lwin5000/3GPP attempt at this bound), aod=0.05 (job 153655) --
plus full-power MMSE as reference, all nadir geometry, 3GPP Set-1 params
(30 dBi / 75 W). One generate_beampatterns() call so every curve shares the
exact same channel realizations, matching this project's established
beampattern convention (see beampattern_3gpp_nadir_aod0.py, which this
script extends with the aod=0.025 checkpoint).

MMSE here is the standard full-power reference (mmse_precoder_normalized,
i.e. always rescaled to the full power_constraint_watt budget) -- not the
matched-power variant used in the rate/power sweeps; beampattern shape is
what's under study here, so the classical full-budget baseline is the
right comparison.

Evaluated at CSIT error=0.05, matching every other beampattern_*.py script
in this directory. Beamwidth/coverage analysis matches
beampattern_3gpp_nadir_aod0.py's definitions verbatim (half_max_beamwidth,
coverage_ratio = gain(true_angle)/peak_gain), printed and saved to gzip.
"""
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False

import gzip
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config.config import Config
from src.config.config_plotting import PlotConfig
from src.data.satellite_manager import SatelliteManager
from src.data.user_manager import UserManager
from src.analysis.generate_beampatterns import generate_beampatterns
from src.plotting.plot_beam_patterns import plot_beam_patterns, print_realizations

TRAINING_NAME = 'EE_beampattern_3gpp_triplet_N16K3'
ANGLE_SWEEP_RANGE = np.arange(1.2, 1.9, 0.1 * np.pi / 180)
NUM_PATTERNS = 30
CSIT_ERROR_BOUND = 0.05
REALIZATION_TO_PLOT = int(sys.argv[sys.argv.index('--realization') + 1]) if '--realization' in sys.argv else 0

PLOT_ONLY = '--plot-only' in sys.argv

MODEL_KEYS = ['sac_nadir_aod0.0', 'sac_nadir_aod0.025', 'sac_nadir_aod0.05']

CHECKPOINT_TRAINING_NAMES = {
    'sac_nadir_aod0.0': 'EE_dinkelbach_adaptive_lwin5000_N16K3_satg30_p75_eta0.6_rawpow',
    'sac_nadir_aod0.025': 'EE_dinkelbach_adaptive_aod0.025_lwin5000_N16K3_satg30_p75_eta0.6_rawpow',
    'sac_nadir_aod0.05': 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_satg30_p75_eta0.6_rawpow',
}


def get_best_model_path(trained_models_path, training_name):
    """Session-aware checkpoint selection -- see my_evaluation.py's identical function for the full rationale."""
    base_path = Path(trained_models_path, training_name, 'base')
    checkpoints = [p for p in base_path.iterdir() if p.is_dir() and 'full_snap' in p.name]
    if not checkpoints:
        raise FileNotFoundError(f'No checkpoints found under {base_path}')

    checkpoints_by_time = sorted(checkpoints, key=lambda p: os.path.getmtime(p))

    max_gap_seconds = 90 * 60
    session_start_idx = len(checkpoints_by_time) - 1
    for i in range(len(checkpoints_by_time) - 1, 0, -1):
        gap = os.path.getmtime(checkpoints_by_time[i]) - os.path.getmtime(checkpoints_by_time[i - 1])
        if gap > max_gap_seconds:
            session_start_idx = i
            break
    else:
        session_start_idx = 0

    same_session_checkpoints = checkpoints_by_time[session_start_idx:]
    best = sorted(same_session_checkpoints, key=lambda p: float(p.name.split('_')[-1]))[-1]
    return best


def get_config():
    cfg = Config()
    cfg.show_plots = False
    cfg.config_learner.training_name = TRAINING_NAME
    cfg.config_error_model.error_rng_parametrizations['additive_error_on_cosine_of_aod']['args']['low'] = -CSIT_ERROR_BOUND
    cfg.config_error_model.error_rng_parametrizations['additive_error_on_cosine_of_aod']['args']['high'] = CSIT_ERROR_BOUND
    return cfg


def half_max_beamwidth(angle_sweep_range, gain_curve, peak_idx):
    """Angular width (rad) of the contiguous region around peak_idx staying above half the peak gain, edges linearly interpolated."""
    peak_gain = gain_curve[peak_idx]
    half = peak_gain / 2.0

    left_idx = peak_idx
    while left_idx > 0 and gain_curve[left_idx - 1] >= half:
        left_idx -= 1
    if left_idx > 0:
        g0, g1 = gain_curve[left_idx - 1], gain_curve[left_idx]
        a0, a1 = angle_sweep_range[left_idx - 1], angle_sweep_range[left_idx]
        frac = (half - g0) / (g1 - g0) if g1 != g0 else 0.0
        left_edge = a0 + frac * (a1 - a0)
    else:
        left_edge = angle_sweep_range[0]

    right_idx = peak_idx
    while right_idx < len(gain_curve) - 1 and gain_curve[right_idx + 1] >= half:
        right_idx += 1
    if right_idx < len(gain_curve) - 1:
        g0, g1 = gain_curve[right_idx], gain_curve[right_idx + 1]
        a0, a1 = angle_sweep_range[right_idx], angle_sweep_range[right_idx + 1]
        frac = (half - g0) / (g1 - g0) if g1 != g0 else 0.0
        right_edge = a0 + frac * (a1 - a0)
    else:
        right_edge = angle_sweep_range[-1]

    return right_edge - left_edge


def analyze_gzip(data_path, model_keys):
    with gzip.open(data_path, 'rb') as file:
        angle_sweep_range, data = pickle.load(file)

    stats = {m: {'beamwidth_deg': [], 'coverage': [], 'offset_deg': [], 'sum_rate': []} for m in model_keys}
    for entry in data:
        true_angles = entry['user_positions'][0]  # single-satellite
        for model in model_keys:
            if model not in entry:
                continue
            gains_all_users = entry[model]['power_gains']  # (num_users, num_angles, num_sats)
            stats[model]['sum_rate'].append(entry[model]['sum_rate'])
            for user_idx, true_angle in enumerate(true_angles):
                gain_curve = gains_all_users[user_idx, :, 0]
                peak_idx = int(np.argmax(gain_curve))
                peak_gain = gain_curve[peak_idx]
                bw_rad = half_max_beamwidth(angle_sweep_range, gain_curve, peak_idx)
                true_idx = int(np.argmin(np.abs(angle_sweep_range - true_angle)))
                coverage = gain_curve[true_idx] / peak_gain if peak_gain > 0 else 0.0
                offset_deg = np.rad2deg(angle_sweep_range[peak_idx] - true_angle)
                stats[model]['beamwidth_deg'].append(np.rad2deg(bw_rad))
                stats[model]['coverage'].append(coverage)
                stats[model]['offset_deg'].append(offset_deg)

    print(f'\n  {"model":<18} {"mean bw [deg]":>13} {"median bw":>10} {"mean cov":>9} {"median cov":>10} '
          f'{"mean |off| [deg]":>16} {"mean rate":>10}')
    for model in model_keys:
        s = stats[model]
        if not s['beamwidth_deg']:
            continue
        print(f'  {model:<18} {np.mean(s["beamwidth_deg"]):>13.2f} {np.median(s["beamwidth_deg"]):>10.2f} '
              f'{np.mean(s["coverage"]):>9.3f} {np.median(s["coverage"]):>10.3f} '
              f'{np.mean(np.abs(s["offset_deg"])):>16.2f} {np.mean(s["sum_rate"]):>10.3f}')
    return stats


if __name__ == '__main__':
    cfg = get_config()

    data_path = Path(cfg.output_metrics_path, TRAINING_NAME, 'beam_patterns', 'beam_patterns.gzip')

    if not PLOT_ONLY:
        satellite_manager = SatelliteManager(cfg)
        user_manager = UserManager(cfg)

        model_paths = {
            label: get_best_model_path(cfg.trained_models_path, training_name)
            for label, training_name in CHECKPOINT_TRAINING_NAMES.items()
        }
        for label, path in model_paths.items():
            print(f'[beampattern_triplet] {label} -> {path}')

        generate_beampatterns(
            angle_sweep_range=ANGLE_SWEEP_RANGE,
            num_patterns=NUM_PATTERNS,
            config=cfg,
            satellite_manager=satellite_manager,
            user_manager=user_manager,
            learned_model_paths=model_paths,
            generate_mmse=True,
            generate_slnr=False,
            generate_ones=False,
        )

    print_realizations(data_path)
    all_stats = analyze_gzip(data_path, ['mmse'] + MODEL_KEYS)

    stats_out = Path(cfg.output_metrics_path, 'EE_beampattern_3gpp_triplet_stats.gzip')
    with gzip.open(stats_out, 'wb') as file:
        pickle.dump(all_stats, file=file)
    print(f'\nSaved: {stats_out}')

    plot_cfg = PlotConfig()
    matplotlib.rcParams['text.usetex'] = False  # PlotConfig() resets this on construction
    plot_width = 1.46 * plot_cfg.textwidth
    plot_height = plot_width * 0.4

    which_plots = [
        {
            'row': 0,
            'column': 0,
            'realization': REALIZATION_TO_PLOT,
            'precoders': ['mmse'] + MODEL_KEYS,
        },
    ]
    colors = {
        'mmse': plot_cfg.cp2['gold'],
        'sac_nadir_aod0.0': plot_cfg.cp2['green'],
        'sac_nadir_aod0.025': plot_cfg.cp2['blue'],
        'sac_nadir_aod0.05': plot_cfg.cp2['magenta'],
    }
    line_styles = {
        'mmse': 'dotted',
        'sac_nadir_aod0.0': 'solid',
        'sac_nadir_aod0.025': 'dashdot',
        'sac_nadir_aod0.05': 'dashed',
    }
    marker_styles = {
        'mmse': 'x',
        'sac_nadir_aod0.0': 'o',
        'sac_nadir_aod0.025': '^',
        'sac_nadir_aod0.05': 's',
    }
    labels = {
        'mmse': 'MMSE (full power)',
        'sac_nadir_aod0.0': 'SAC (3GPP nadir, train err 0.0)',
        'sac_nadir_aod0.025': 'SAC (3GPP nadir, train err 0.025)',
        'sac_nadir_aod0.05': 'SAC (3GPP nadir, train err 0.05)',
    }

    plot_beam_patterns(
        width=plot_width,
        height=plot_height,
        path=data_path,
        plots=which_plots,
        color_dict=colors,
        line_style_dict=line_styles,
        label_dict=labels,
        marker_style_dict=marker_styles,
        xlim=[1.2, 1.9],
        plots_parent_path=plot_cfg.plots_parent_path,
        name='beampattern_triplet',
    )
    ax = plt.gca()
    ax.set_xlabel('Angle of Departure [rad]')
    ax.set_ylabel('Power Gain')

    if ax.get_legend() is not None:
        ax.get_legend().remove()
    handles, plot_labels = ax.get_legend_handles_labels()
    ax.legend(handles, plot_labels, loc='upper right', fontsize=7, framealpha=1.0, frameon=True)

    pdf_path = Path(plot_cfg.plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)
    out = Path(pdf_path, f'beampattern_triplet_realization{REALIZATION_TO_PLOT}.pdf')
    plt.savefig(out, bbox_inches='tight', dpi=300, transparent=True)
    print(f'Saved: {out}')
