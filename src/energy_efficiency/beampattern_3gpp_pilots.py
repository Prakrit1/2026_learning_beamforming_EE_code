"""
Beampattern generation + beamwidth/coverage quantification for the two
3GPP-aligned pilot checkpoints (jobs 153655 nadir / 153656 elev30),
evaluated at CSIT error=0.05 like every other beampattern_*.py here.

Purpose (handoff item 4c): confirm the learned beams at the new link
budget (30 dBi / 75 W) stay in the ~2-4.5 deg beamwidth range consistent
with TR 38.821 Set-1 (4.41 deg 3dB beamwidth), and check aiming/coverage
under CSIT error in both geometries.

Each geometry runs as its own generate_beampatterns() pass under its own
system env (see config.py overrides): nadir = defaults, elev30 =
EE_TARGET_ELEVATION_DEG=30. MMSE is generated alongside as the spatial
reference in each geometry.

The elev30 geometry puts users FAR off array broadside (users appear
~52 deg off nadir at 30 deg elevation), so the fixed 1.2-1.9 rad angle
sweep used by the nadir-scenario scripts would miss them entirely. The
sweep window is therefore centered dynamically: sample user AoDs over a
few update_sim() draws, then sweep [min-margin, max+margin] at the usual
0.1 deg step.

Beamwidth/coverage analysis matches beampattern_robustness_analysis.py's
definitions (half_max_beamwidth copied verbatim; coverage_ratio =
gain(true_angle)/peak_gain), printed per geometry and saved to gzip.
"""
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False

import gzip
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config.config_plotting import PlotConfig
from src.data.satellite_manager import SatelliteManager
from src.data.user_manager import UserManager
from src.analysis.generate_beampatterns import generate_beampatterns
from src.plotting.plot_beam_patterns import plot_beam_patterns
from src.utils.update_sim import update_sim

NUM_PATTERNS = 30
CSIT_ERROR_BOUND = 0.05
ANGLE_STEP_RAD = 0.1 * np.pi / 180
SWEEP_MARGIN_RAD = 0.20
AOD_SAMPLING_DRAWS = 50

SYSTEM_ENV_VARS = ['EE_SAT_GAIN_DBI', 'EE_POWER_BUDGET_WATT', 'EE_TARGET_ELEVATION_DEG']

GEOMETRIES = [
    {
        'key': '3gpp_nadir',
        'label': '3GPP Set-1, nadir',
        'training_name': 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_satg30_p75_eta0.6_rawpow',
        'env': {},
        'output_name': 'EE_beampattern_3gpp_nadir_N16K3_aod0.05',
    },
    {
        'key': '3gpp_elev30',
        'label': '3GPP Set-1, elev 30 deg',
        'training_name': 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_satg30_p75_elev30_eta0.6_rawpow',
        'env': {'EE_TARGET_ELEVATION_DEG': '30'},
        'output_name': 'EE_beampattern_3gpp_elev30_N16K3_aod0.05',
    },
]


def make_config(env: dict):
    """Construct a Config under exactly the given system env overrides (all others cleared)."""
    for var in SYSTEM_ENV_VARS:
        os.environ.pop(var, None)
    os.environ.update(env)
    from src.config.config import Config  # env vars are read in Config.__init__, after the overrides above
    cfg = Config()
    cfg.show_plots = False
    return cfg


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


def sample_angle_sweep_range(cfg):
    """Center the sweep window on where the users actually are in this geometry."""
    satellite_manager = SatelliteManager(cfg)
    user_manager = UserManager(cfg)
    aods = []
    for _ in range(AOD_SAMPLING_DRAWS):
        update_sim(cfg, satellite_manager, user_manager)
        aods.extend(satellite_manager.satellites[0].aods_to_users.copy())
    lo = min(aods) - SWEEP_MARGIN_RAD
    hi = max(aods) + SWEEP_MARGIN_RAD
    print(f'  user AoDs sampled over {AOD_SAMPLING_DRAWS} draws: [{min(aods):.3f}, {max(aods):.3f}] rad '
          f'-> sweep [{lo:.3f}, {hi:.3f}]')
    return np.arange(lo, hi, ANGLE_STEP_RAD)


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

    print(f'\n  {"model":<16} {"mean bw [deg]":>13} {"median bw":>10} {"mean cov":>9} {"median cov":>10} '
          f'{"mean |off| [deg]":>16} {"mean rate":>10}')
    for model in model_keys:
        s = stats[model]
        if not s['beamwidth_deg']:
            continue
        print(f'  {model:<16} {np.mean(s["beamwidth_deg"]):>13.2f} {np.median(s["beamwidth_deg"]):>10.2f} '
              f'{np.mean(s["coverage"]):>9.3f} {np.median(s["coverage"]):>10.3f} '
              f'{np.mean(np.abs(s["offset_deg"])):>16.2f} {np.mean(s["sum_rate"]):>10.3f}')
    return stats


if __name__ == '__main__':
    all_stats = {}
    plot_cfg = PlotConfig()
    matplotlib.rcParams['text.usetex'] = False  # PlotConfig() resets this on construction

    for geom in GEOMETRIES:
        print(f"\n=== {geom['label']} ===")
        cfg = make_config(geom['env'])
        cfg.config_learner.training_name = geom['output_name']
        cfg.config_error_model.error_rng_parametrizations['additive_error_on_cosine_of_aod']['args']['low'] = -CSIT_ERROR_BOUND
        cfg.config_error_model.error_rng_parametrizations['additive_error_on_cosine_of_aod']['args']['high'] = CSIT_ERROR_BOUND
        print(f"  sat_gain_dBi={cfg.sat_gain_dBi}, budget={cfg.power_constraint_watt} W, "
              f"user_center_aod_earth_deg={cfg.user_center_aod_earth_deg:.2f}")

        angle_sweep_range = sample_angle_sweep_range(cfg)

        model_path = get_best_model_path(cfg.trained_models_path, geom['training_name'])
        print(f'  checkpoint: {model_path}')

        satellite_manager = SatelliteManager(cfg)
        user_manager = UserManager(cfg)
        generate_beampatterns(
            angle_sweep_range=angle_sweep_range,
            num_patterns=NUM_PATTERNS,
            config=cfg,
            satellite_manager=satellite_manager,
            user_manager=user_manager,
            learned_model_paths={geom['key']: model_path},
            generate_mmse=True,
            generate_slnr=False,
            generate_ones=False,
        )

        data_path = Path(cfg.output_metrics_path, geom['output_name'], 'beam_patterns', 'beam_patterns.gzip')
        all_stats[geom['key']] = analyze_gzip(data_path, [geom['key'], 'mmse'])

        # one-realization overview plot per geometry
        plot_beam_patterns(
            width=1.46 * plot_cfg.textwidth,
            height=1.46 * plot_cfg.textwidth * 0.4,
            path=data_path,
            plots=[{'row': 0, 'column': 0, 'realization': 0, 'precoders': ['mmse', geom['key']]}],
            color_dict={'mmse': plot_cfg.cp2['gold'], geom['key']: plot_cfg.cp2['green']},
            line_style_dict={'mmse': 'dotted', geom['key']: 'solid'},
            label_dict={'mmse': 'MMSE', geom['key']: f"SAC (EE, {geom['label']})"},
            marker_style_dict={'mmse': 'x', geom['key']: 'o'},
            xlim=[float(angle_sweep_range[0]), float(angle_sweep_range[-1])],
            plots_parent_path=plot_cfg.plots_parent_path,
            name=f"beampattern_{geom['key']}",
        )
        ax = plt.gca()
        ax.set_xlabel('Angle of Departure [rad]')
        ax.set_ylabel('Power Gain')
        pdf_path = Path(plot_cfg.plots_parent_path, 'pdf')
        pdf_path.mkdir(parents=True, exist_ok=True)
        out = Path(pdf_path, f"beampattern_{geom['key']}_realization0.pdf")
        plt.savefig(out, bbox_inches='tight', dpi=300, transparent=True)
        print(f'  Saved: {out}')
        plt.close('all')

    stats_out = Path(make_config({}).output_metrics_path, 'EE_beampattern_3gpp_pilots_stats.gzip')
    with gzip.open(stats_out, 'wb') as file:
        pickle.dump(all_stats, file=file)
    print(f'\nSaved: {stats_out}')
