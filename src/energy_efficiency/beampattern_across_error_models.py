"""
Beampattern comparison ACROSS the three error-trained EE (Dinkelbach)
checkpoints: does training at a higher fixed CSIT error (aod0.0 vs
aod0.025 vs aod0.05) change where the precoder points its beams, or just
how it degrades? Unlike beampattern_ee_vs_regular.py (which compares one EE
checkpoint against the "regular"/always-100W checkpoint), this compares the
three EE checkpoints against EACH OTHER.

Checkpoints (the current "final" three-checkpoint set, see
handoff_prompt_EE_evaluation.txt's "Optuna searches" section):
  - 'EE_aod0.0': default-LR, training_name
    'EE_dinkelbach_adaptive_lwin5000_N16K3_eta0.6_rawpow'
  - 'EE_aod0.025': TUNED-LR (newly adopted), training_name
    'EE_dinkelbach_adaptive_aod0.025_lwin5000_N16K3_eta0.6_rawpow_lrc1.82e-06_lra1.17e-06'
  - 'EE_aod0.05': default-LR, training_name
    'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_eta0.6_rawpow'

All three evaluated at the SAME CSIT error (0.05, matching
beampattern_ee_vs_regular.py's "robust case" convention) so any spatial
differences reflect the training error bound, not the evaluation
condition. angle_sweep_range/NUM_PATTERNS reused unchanged from that
script (already verified against real user AoDs in this scenario).

Uses get_best_model_path (same logic as my_evaluation.py/
tx_power_distribution.py) instead of a hardcoded full_snap folder name, so
this keeps working if any of these checkpoints gets retrained again.
"""
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False

import os
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

TRAINING_NAME = 'EE_beampattern_across_error_models_N16K3'
ANGLE_SWEEP_RANGE = np.arange(1.2, 1.9, 0.1 * np.pi / 180)
NUM_PATTERNS = 30
CSIT_ERROR_BOUND = 0.05
REALIZATION_TO_PLOT = int(sys.argv[sys.argv.index('--realization') + 1]) if '--realization' in sys.argv else 0

PLOT_ONLY = '--plot-only' in sys.argv


def get_best_model_path(trained_models_path, training_name):
    base_path = Path(trained_models_path, training_name, 'base')
    checkpoints = [p for p in base_path.iterdir() if p.is_dir() and 'full_snap' in p.name]
    if not checkpoints:
        raise FileNotFoundError(f'No checkpoints found under {base_path}')

    checkpoints_by_time = sorted(checkpoints, key=lambda p: os.path.getmtime(p))

    max_gap_minutes = 90
    max_gap_seconds = max_gap_minutes * 60

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


if __name__ == '__main__':
    cfg = get_config()

    data_path = Path(cfg.output_metrics_path, TRAINING_NAME, 'beam_patterns', 'beam_patterns.gzip')

    if not PLOT_ONLY:
        satellite_manager = SatelliteManager(cfg)
        user_manager = UserManager(cfg)

        checkpoint_training_names = {
            'EE_aod0.0': 'EE_dinkelbach_adaptive_lwin5000_N16K3_eta0.6_rawpow',
            'EE_aod0.025': 'EE_dinkelbach_adaptive_aod0.025_lwin5000_N16K3_eta0.6_rawpow_lrc1.82e-06_lra1.17e-06',
            'EE_aod0.05': 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_eta0.6_rawpow',
        }
        model_paths = {
            label: get_best_model_path(cfg.trained_models_path, training_name)
            for label, training_name in checkpoint_training_names.items()
        }
        for label, path in model_paths.items():
            print(f'[beampattern_across_error_models] {label} -> {path}')

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

    plot_cfg = PlotConfig()
    matplotlib.rcParams['text.usetex'] = False  # PlotConfig() resets this on construction
    plot_width = 1.46 * plot_cfg.textwidth
    plot_height = plot_width * 0.4

    which_plots = [
        {
            'row': 0,
            'column': 0,
            'realization': REALIZATION_TO_PLOT,
            'precoders': ['mmse', 'EE_aod0.0', 'EE_aod0.025', 'EE_aod0.05'],
        },
    ]
    colors = {
        'mmse': plot_cfg.cp2['gold'],
        'EE_aod0.0': plot_cfg.cp2['green'],
        'EE_aod0.025': plot_cfg.cp2['blue'],
        'EE_aod0.05': plot_cfg.cp2['magenta'],
    }
    line_styles = {
        'mmse': 'solid',
        'EE_aod0.0': 'solid',
        'EE_aod0.025': 'dashed',
        'EE_aod0.05': 'dotted',
    }
    marker_styles = {
        'mmse': 'v',
        'EE_aod0.0': 'o',
        'EE_aod0.025': '^',
        'EE_aod0.05': 's',
    }
    labels = {
        'mmse': 'MMSE',
        'EE_aod0.0': 'SAC (EE, trained error=0.0)',
        'EE_aod0.025': 'SAC (EE, trained error=0.025, tuned LR)',
        'EE_aod0.05': 'SAC (EE, trained error=0.05)',
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
        name='beampattern_across_error_models',
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
    out = Path(pdf_path, f'beampattern_across_error_models_realization{REALIZATION_TO_PLOT}.pdf')
    plt.savefig(out, bbox_inches='tight', dpi=300, transparent=True)
    print(f'Saved: {out}')
