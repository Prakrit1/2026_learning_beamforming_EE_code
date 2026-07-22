"""
Beampattern + pairwise channel correlation, side by side, per realization.

Follow-up to beampattern_ee_vs_regular.py: that script found EE sometimes
abandons one user spatially (realizations 0, 2, 10) and sometimes matches the
regular checkpoint's strategy (realizations 5, 17, 21), and left open WHY --
checking the dropped user's large_scale_fading alone didn't explain it
(realization 0: the abandoned user did NOT have the worst fading).

This script adds the other physical candidate: pairwise channel correlation
between users (see channel_correlation_zero_error.py -- |correlation|=1 means
two users' channels are spatially indistinguishable, a real difficulty for
ANY precoder, not a training artifact). Plotting it next to the beampattern
for the same realization makes it possible to eyeball whether "EE drops the
user who's most correlated with another user" holds up.

Regenerates its own 30-realization dataset (compute_channel_correlations=True
on generate_beampatterns(), a new opt-in kwarg -- see that file) rather than
reusing beampattern_ee_vs_regular.py's existing gzip, which was saved before
channel correlation was captured. Same checkpoints, same CSIT error bound, so
directly comparable in spirit; realization indices will NOT line up with the
earlier script's since both draw fresh random Monte Carlo samples.
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

TRAINING_NAME = 'EE_beampattern_correlation_N16K3_aod0.05'
ANGLE_SWEEP_RANGE = np.arange(1.2, 1.9, 0.1 * np.pi / 180)
NUM_PATTERNS = 30
CSIT_ERROR_BOUND = 0.05

PLOT_ONLY = '--plot-only' in sys.argv
if '--realizations' in sys.argv:
    REALIZATIONS_TO_PLOT = [int(x) for x in sys.argv[sys.argv.index('--realizations') + 1].split(',')]
else:
    REALIZATIONS_TO_PLOT = list(range(NUM_PATTERNS))

PRECODERS = ['regular_schemeI', 'EE_dinkelbach_aod0.05']
COLORS = {'regular_schemeI': None, 'EE_dinkelbach_aod0.05': None}  # filled in from PlotConfig below
LINE_STYLES = {'regular_schemeI': 'solid', 'EE_dinkelbach_aod0.05': 'dashed'}
MARKER_STYLES = {'regular_schemeI': 'o', 'EE_dinkelbach_aod0.05': 's'}
LABELS = {'regular_schemeI': 'SAC (100 W)', 'EE_dinkelbach_aod0.05': 'SAC (Energy-Efficient)'}


def get_config():
    cfg = Config()
    cfg.show_plots = False
    cfg.config_learner.training_name = TRAINING_NAME
    cfg.config_error_model.error_rng_parametrizations['additive_error_on_cosine_of_aod']['args']['low'] = -CSIT_ERROR_BOUND
    cfg.config_error_model.error_rng_parametrizations['additive_error_on_cosine_of_aod']['args']['high'] = CSIT_ERROR_BOUND
    return cfg


def plot_side_by_side(data, angle_sweep_range, realization, plot_cfg, xlim, out_path):
    fig, (ax_beam, ax_corr) = plt.subplots(
        nrows=1, ncols=2,
        figsize=(1.9 * plot_cfg.textwidth, 0.55 * plot_cfg.textwidth),
    )

    # --- left panel: beampattern (same drawing logic as plot_beam_patterns.py) ---
    for user_id, user_position in enumerate(data[realization]['user_positions'][0]):
        label = 'Users' if user_id == 0 else '_UserHidden'
        ax_beam.scatter(user_position, 0, label=label, color='black')
        ax_beam.axvline(user_position, label='_UserHidden', color='black', linestyle='dotted')

    ee_shares = {}
    for precoder in PRECODERS:
        gains = data[realization][precoder]['power_gains']
        num_users = gains.shape[0]
        peaks = gains[:, :, 0].max(axis=1)
        if precoder == 'EE_dinkelbach_aod0.05':
            ee_shares = {u: 100 * peaks[u] / peaks.sum() for u in range(num_users)}
        for user_id in range(num_users):
            label = LABELS[precoder] if user_id == 0 else '_PrecoderHidden'
            curve = gains[user_id, :, 0]
            peak_idx = np.argmax(curve)
            ax_beam.plot(
                angle_sweep_range, curve,
                label=label,
                color=COLORS[precoder],
                linestyle=LINE_STYLES[precoder],
                marker=MARKER_STYLES[precoder],
                markevery=[peak_idx],
            )

    ax_beam.set_xlim(xlim)
    ax_beam.set_xlabel('Angle of Departure [rad]')
    ax_beam.set_ylabel('Power Gain')
    ax_beam.legend(
        fontsize=6, loc='best', framealpha=0.85, markerscale=0.7,
        handlelength=1.5, borderpad=0.3, labelspacing=0.3, handletextpad=0.4,
    )
    ax_beam.set_title(f'Beampattern (realization {realization})', fontsize=10)

    # annotate each user's dotted line with EE's power share, so the
    # "who got dropped" reading doesn't require eyeballing peak heights
    ylim = ax_beam.get_ylim()
    for user_id, user_position in enumerate(data[realization]['user_positions'][0]):
        if user_id in ee_shares:
            ax_beam.text(
                user_position, ylim[1] * 0.97, f'{ee_shares[user_id]:.0f}%',
                ha='center', va='top', fontsize=7, color='#2c7a2c',
            )

    # --- right panel: pairwise channel correlation bar chart ---
    correlations = data[realization]['channel_correlations']
    pair_labels = [f'U{i + 1}-U{j + 1}' for (i, j) in correlations]
    values = [correlations[pair] for pair in correlations]

    bars = ax_corr.bar(pair_labels, values, color=plot_cfg.cp2['blue'])
    for bar, value in zip(bars, values):
        ax_corr.text(
            bar.get_x() + bar.get_width() / 2, value + 0.02, f'{value:.3f}',
            ha='center', va='bottom', fontsize=8,
        )
    ax_corr.set_ylim(0, 1.08)
    ax_corr.set_ylabel('|Channel correlation|')
    ax_corr.set_title('Pairwise channel correlation (true channel)', fontsize=10)
    ax_corr.axhline(1.0, color='grey', linestyle='dashed', linewidth=0.8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)


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
            compute_channel_correlations=True,
        )

    import gzip
    import pickle
    with gzip.open(data_path, 'rb') as file:
        angle_sweep_range, data = pickle.load(file)

    plot_cfg = PlotConfig()
    matplotlib.rcParams['text.usetex'] = False  # PlotConfig() resets this on construction
    COLORS['regular_schemeI'] = plot_cfg.cp2['blue']
    COLORS['EE_dinkelbach_aod0.05'] = plot_cfg.cp2['green']

    pdf_path = Path(plot_cfg.plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)

    for realization in REALIZATIONS_TO_PLOT:
        out = Path(pdf_path, f'beampattern_with_correlation_realization{realization}.pdf')
        plot_side_by_side(
            data=data,
            angle_sweep_range=angle_sweep_range,
            realization=realization,
            plot_cfg=plot_cfg,
            xlim=[1.2, 1.9],
            out_path=out,
        )
        print(f'Saved: {out}')
