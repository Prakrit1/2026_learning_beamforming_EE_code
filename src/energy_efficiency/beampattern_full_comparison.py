"""
Full beampattern comparison, one realization, ALL relevant precoders on the
same axes for a proper apples-to-apples read: MMSE (spatial reference),
'regular_schemeI' (always-100W SAC, training_name 'full_EE_aod0.5_N16K3_eta0.6'),
the EE aod=0.05 baseline (no fairness term), and all three fairness-weighted
EE aod=0.05 checkpoints (0.5/1.5/3.0).

Unlike beampattern_ee_vs_regular.py / beampattern_fairness_comparison.py
(which each only compare a subset), this generates ALL SIX curves from ONE
generate_beampatterns() call so every precoder is evaluated on the exact
SAME channel realizations -- required for a fair per-realization comparison
(separate scripts/gzips draw independent random realizations).

Also prints a "dropped user" diagnostic per realization: for each of the
five learned/regular precoders (not MMSE), computes each user's own peak
power gain across the angle sweep, then the ratio of the weakest user's
peak to the strongest user's peak. A low ratio means that precoder is
putting almost no power toward one user at all in that realization --
useful for picking a realization that actually shows user-abandonment
behavior rather than eyeballing plots one at a time.

Evaluated at CSIT error=0.05, matching the convention of the other
beampattern_*.py scripts in this directory. angle_sweep_range/NUM_PATTERNS
reused unchanged (already verified against real user AoDs in this scenario).
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

TRAINING_NAME = 'EE_beampattern_full_comparison_N16K3_aod0.05'
ANGLE_SWEEP_RANGE = np.arange(1.2, 1.9, 0.1 * np.pi / 180)
NUM_PATTERNS = 30
CSIT_ERROR_BOUND = 0.05
REALIZATION_TO_PLOT = int(sys.argv[sys.argv.index('--realization') + 1]) if '--realization' in sys.argv else 0

PLOT_ONLY = '--plot-only' in sys.argv

DROPPED_USER_MODELS = ['regular_100W', 'EE_baseline', 'EE_fair0.5', 'EE_fair1.5', 'EE_fair3.0']
DROPPED_USER_RATIO_THRESHOLD = 0.15


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


def print_dropped_user_candidates(data_path):
    with gzip.open(data_path, 'rb') as file:
        data = pickle.load(file)[1]

    print('\nDropped-user candidates (weakest-user peak gain / strongest-user peak gain, per model):')
    candidates = []
    for real_id, entry in enumerate(data):
        row = {'realization': real_id}
        worst_ratio_this_real = 1.0
        worst_model_this_real = None
        for model in DROPPED_USER_MODELS:
            if model not in entry:
                continue
            peak_gains = entry[model]['power_gains'][:, :, 0].max(axis=1)  # per-user peak gain
            ratio = peak_gains.min() / peak_gains.max()
            row[model] = ratio
            if ratio < worst_ratio_this_real:
                worst_ratio_this_real = ratio
                worst_model_this_real = model
        row['worst_ratio'] = worst_ratio_this_real
        row['worst_model'] = worst_model_this_real
        candidates.append(row)

    candidates.sort(key=lambda r: r['worst_ratio'])
    for row in candidates:
        if row['worst_ratio'] < DROPPED_USER_RATIO_THRESHOLD:
            details = ', '.join(f"{m}={row[m]:.3f}" for m in DROPPED_USER_MODELS if m in row)
            print(f"  realization {row['realization']}: worst={row['worst_model']} "
                  f"(ratio={row['worst_ratio']:.3f}) | {details}")


if __name__ == '__main__':
    cfg = get_config()

    data_path = Path(cfg.output_metrics_path, TRAINING_NAME, 'beam_patterns', 'beam_patterns.gzip')

    if not PLOT_ONLY:
        satellite_manager = SatelliteManager(cfg)
        user_manager = UserManager(cfg)

        checkpoint_training_names = {
            'regular_100W': 'full_EE_aod0.5_N16K3_eta0.6',
            'EE_baseline': 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_eta0.6_rawpow',
            'EE_fair0.5': 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_eta0.6_rawpow_fair0.5',
            'EE_fair1.5': 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_eta0.6_rawpow_fair1.5',
            'EE_fair3.0': 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_eta0.6_rawpow_fair3.0',
        }
        model_paths = {
            label: get_best_model_path(cfg.trained_models_path, training_name)
            for label, training_name in checkpoint_training_names.items()
        }
        for label, path in model_paths.items():
            print(f'[beampattern_full_comparison] {label} -> {path}')

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
    print_dropped_user_candidates(data_path)

    plot_cfg = PlotConfig()
    matplotlib.rcParams['text.usetex'] = False  # PlotConfig() resets this on construction
    plot_width = 1.46 * plot_cfg.textwidth
    plot_height = plot_width * 0.4

    which_plots = [
        {
            'row': 0,
            'column': 0,
            'realization': REALIZATION_TO_PLOT,
            'precoders': ['mmse', 'regular_100W', 'EE_baseline', 'EE_fair0.5', 'EE_fair1.5', 'EE_fair3.0'],
        },
    ]
    colors = {
        'mmse': plot_cfg.cp2['gold'],
        'regular_100W': plot_cfg.cp3['blue1'],
        'EE_baseline': plot_cfg.cp2['magenta'],
        'EE_fair0.5': plot_cfg.cp3['peach1'],
        'EE_fair1.5': plot_cfg.cp2['green'],
        'EE_fair3.0': plot_cfg.cp3['purple1'],
    }
    line_styles = {
        'mmse': 'dotted',
        'regular_100W': 'solid',
        'EE_baseline': 'dashed',
        'EE_fair0.5': 'dashdot',
        'EE_fair1.5': (0, (3, 1, 1, 1)),
        'EE_fair3.0': (0, (1, 1)),
    }
    marker_styles = {
        'mmse': 'x',
        'regular_100W': 'o',
        'EE_baseline': 's',
        'EE_fair0.5': '^',
        'EE_fair1.5': 'd',
        'EE_fair3.0': 'v',
    }
    labels = {
        'mmse': 'MMSE',
        'regular_100W': 'SAC (100 W)',
        'EE_baseline': 'SAC (EE, no fairness)',
        'EE_fair0.5': 'SAC (EE, fairness=0.5)',
        'EE_fair1.5': 'SAC (EE, fairness=1.5)',
        'EE_fair3.0': 'SAC (EE, fairness=3.0)',
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
        name='beampattern_full_comparison',
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
    out = Path(pdf_path, f'beampattern_full_comparison_realization{REALIZATION_TO_PLOT}.pdf')
    plt.savefig(out, bbox_inches='tight', dpi=300, transparent=True)
    print(f'Saved: {out}')
