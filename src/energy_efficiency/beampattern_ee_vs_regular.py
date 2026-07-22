"""
Beampattern comparison: does the EE (Dinkelbach, real power savings) SAC
checkpoint point its beams in the SAME spatial direction as the "regular"
(Scheme I, always-rescale-to-budget) SAC checkpoint -- just with less
power -- or does it do something spatially different to save power?

Uses the existing, unmodified pipeline:
  - src/analysis/generate_beampatterns.py: sweeps angle-of-departure,
    computes power gain toward each user at each swept angle, for MMSE
    (spatial reference) plus the two SAC checkpoints below.
  - src/plotting/plot_beam_patterns.py: overlays multiple precoders on the
    same axes for direct comparison (this is the "same direction, different
    power" shape -- unlike test_beam_pattern.py, which plots each precoder
    as its own separate figure).

Checkpoints:
  - 'EE_dinkelbach_aod0.05': the error=0.05-trained Dinkelbach checkpoint
    (jobs 147506) -- our most "robust" EE checkpoint (see
    handoff_prompt_EE_evaluation.txt), real power savings (~56% of budget).
  - 'regular_schemeI': training_name 'full_EE_aod0.5_N16K3_eta0.6' -- the
    ONLY "regular"/non-EE (always-rescale-to-budget, Eq. 26-style
    normalization) SAC checkpoint trained at our current N16K3 scenario.
    Trained at the old Delta-epsilon_aod=0.5 regime (predates Finding 3),
    but that only affects ITS OWN rate/robustness, not whether it points
    beams in a sensible direction -- fine for a purely spatial comparison.

Evaluated at CSIT error=0.05 (not zero) -- "the robust case": this is
within both checkpoints' training ranges (exactly the EE checkpoint's own
training error; well within the regular checkpoint's wider 0.0-0.5 range),
so the interesting question is whether a nonzero, non-negligible estimation
error still leaves both checkpoints pointing at essentially the same place.

angle_sweep_range reused from generate_beampatterns.py's own default
(np.arange(1.2, 1.9, 0.1 deg step)) -- verified against 200 fresh samples
under our actual scenario (N16K3, 100km mean, +-50km roam): real user AoDs
fall in ~[1.33, 1.81] rad, comfortably inside this range with margin on
both sides.

Saves beam_patterns.gzip under a new training_name
'EE_beampattern_comparison_N16K3_aod0.05' (doesn't touch any existing
checkpoint's own eval folder) and produces one comparison PDF per
realization -- rerun with `--plot-only --realization N` to pick a
different draw without regenerating the Monte Carlo data (the gzip holds
NUM_PATTERNS=30 realizations, MMSE included even though not plotted below).

Plot shows only the two SAC curves (MMSE dropped) -- directly answers "what
happens to the beam shape as power drops from the 100W regular checkpoint
to the EE checkpoint's actual (lower) power". Checked across several
realizations: sometimes EE's beam closely tracks the regular checkpoint's
directions just more sharply/differently scaled (e.g. realizations 5, 17,
21); other times EE concentrates power on fewer of the three users
entirely, a genuinely different allocation, not just a rescaled version of
the same pattern (e.g. realizations 0, 2, 10) -- see
handoff_prompt_EE_evaluation.txt for the fuller writeup.
"""
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False

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

TRAINING_NAME = 'EE_beampattern_comparison_N16K3_aod0.05'
ANGLE_SWEEP_RANGE = np.arange(1.2, 1.9, 0.1 * np.pi / 180)
NUM_PATTERNS = 30
CSIT_ERROR_BOUND = 0.05
REALIZATION_TO_PLOT = int(sys.argv[sys.argv.index('--realization') + 1]) if '--realization' in sys.argv else 0

PLOT_ONLY = '--plot-only' in sys.argv


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

        model_paths = {
            'EE_dinkelbach_aod0.05': Path(
                cfg.trained_models_path, 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_eta0.6_rawpow',
                'base', 'full_snap_energy_effiency_0.969',
            ),
            'regular_schemeI': Path(
                cfg.trained_models_path, 'full_EE_aod0.5_N16K3_eta0.6',
                'base', 'full_snap_energy_effiency_0.985',
            ),
        }

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
    # Wider/shorter than the default 4:3 (was 0.99*textwidth x 3/4) -- better
    # fits a slide-width presentation figure.
    plot_width = 1.46 * plot_cfg.textwidth
    plot_height = plot_width * 0.4

    # Two-curve comparison only (dropped MMSE) -- directly answers "what
    # happens to the beam shape as power decreases from 100W (regular) to
    # the EE checkpoint's actual power". MMSE data is still generated and
    # saved in the gzip (in case it's wanted again later), just not plotted
    # here.
    which_plots = [
        {
            'row': 0,
            'column': 0,
            'realization': REALIZATION_TO_PLOT,
            'precoders': ['regular_schemeI', 'EE_dinkelbach_aod0.05'],
        },
    ]
    colors = {
        'regular_schemeI': plot_cfg.cp2['blue'],
        'EE_dinkelbach_aod0.05': plot_cfg.cp2['green'],
    }
    line_styles = {
        'regular_schemeI': 'solid',
        'EE_dinkelbach_aod0.05': 'dashed',
    }
    marker_styles = {
        'regular_schemeI': 'o',
        'EE_dinkelbach_aod0.05': 's',
    }
    labels = {
        'regular_schemeI': 'SAC (100 W)',
        'EE_dinkelbach_aod0.05': 'SAC (Energy-Efficient)',
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
        name='beampattern_ee_vs_regular',
    )
    ax = plt.gca()
    ax.set_xlabel('Angle of Departure [rad]')
    ax.set_ylabel('Power Gain')
    # No title -- presentation slide, title/caption handled outside the figure.

    # plot_beam_patterns() builds its own legend at matplotlib's default
    # (large) fontsize -- replace it with a small, compact one, same
    # pattern used elsewhere in this project's plotting scripts.
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    handles, plot_labels = ax.get_legend_handles_labels()
    ax.legend(handles, plot_labels, loc='upper right', fontsize=7, framealpha=1.0, frameon=True)

    pdf_path = Path(plot_cfg.plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)
    out = Path(pdf_path, f'beampattern_ee_vs_regular_realization{REALIZATION_TO_PLOT}.pdf')
    plt.savefig(out, bbox_inches='tight', dpi=300, transparent=True)
    print(f'Saved: {out}')

    # presentation-ready PNG, same convention as the other docs/presentation_assets figures
    presentation_out = Path(
        '/home/parajuli/repos/2025_learning_beamforming_rsma_code/docs/presentation_assets',
        f'beampattern{REALIZATION_TO_PLOT}_final.png',
    )
    plt.savefig(presentation_out, bbox_inches='tight', dpi=350, transparent=True)
    print(f'Saved: {presentation_out}')
