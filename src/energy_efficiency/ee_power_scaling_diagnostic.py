"""
Post-hoc power-scaling diagnostic: is riding the power budget actually
EE-optimal for the 3GPP pilot checkpoints, or is the policy stuck at the
clip boundary?

Background (2026-07-28): both 3GPP pilots (153655 nadir / 153656 elev30)
converged with raw pre-clip power slightly ABOVE the 75 W budget, i.e.
the transmitted precoder is always the clipped, full-budget one -- SAC is
not exercising the power degree of freedom the Dinkelbach reward exists
to give it. The old 20 dBi / 100 W checkpoints DID settle interior
(56-63% of budget), so the question is whether full power is the genuine
EE optimum at the new link budget (fixed 16 W circuit draw amortizes
better at high power) or a training artifact (early exploration blew raw
power to ~10x budget while lambda was still tiny; above the clip the rate
is flat, so only the weak lambda gradient pulls power back down).

Test: for each checkpoint, draw Monte Carlo channel samples at the
training-time CSIT error (0.05), get the policy's clipped precoder
w_phys, then evaluate the family w_s = s * w_phys for s in (0, 1]:
  rate_s = sum rate with w_s, power_s = s^2 * ||w_phys||^2,
  EE_s = rate_s / (power_s / pa_efficiency + N * Pc)   [bps/Hz per DC watt]
Same channel samples across all s, so the EE(s) curve is paired.

Interpretation: EE peaking at s < 1 proves the policy leaves EE on the
table (down-scaling its own output would beat it); EE peaking at s = 1
means the budget boundary is the true constrained optimum in the radial
direction. The old aod=0.05 checkpoint runs as a control -- it chose an
interior power, so its EE should peak at/near s=1 if well-converged.
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
from src.data.calc_sum_rate import calc_sum_rate
from src.data.satellite_manager import SatelliteManager
from src.data.user_manager import UserManager
from src.utils.get_precoding import get_precoding_learned_clip_only
from src.utils.load_model import load_model
from src.utils.update_sim import update_sim

CSIT_ERROR_BOUND = 0.05
MONTE_CARLO_SAMPLES = 2000
SCALE_GRID = np.linspace(0.1, 1.0, 19)

SYSTEM_ENV_VARS = ['EE_SAT_GAIN_DBI', 'EE_POWER_BUDGET_WATT', 'EE_TARGET_ELEVATION_DEG']

SYSTEMS = [
    {
        'key': 'old_aod0.05',
        'label': 'old system (20 dBi, 100 W, nadir) [control]',
        'training_name': 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_eta0.6_rawpow',
        'env': {'EE_SAT_GAIN_DBI': '20', 'EE_POWER_BUDGET_WATT': '100'},
    },
    {
        'key': '3gpp_nadir',
        'label': '3GPP Set-1 (30 dBi, 75 W, nadir)',
        'training_name': 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_satg30_p75_eta0.6_rawpow',
        'env': {},
    },
    {
        'key': '3gpp_elev30',
        'label': '3GPP Set-1 (30 dBi, 75 W, elev 30 deg)',
        'training_name': 'EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_satg30_p75_elev30_eta0.6_rawpow',
        'env': {'EE_TARGET_ELEVATION_DEG': '30'},
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


def run_scaling_diagnostic(cfg, model_path, label):
    precoder_network, norm_factors = load_model(model_path)
    if norm_factors != {}:
        cfg.config_learner.get_state_args['norm_state'] = True

    satellite_manager = SatelliteManager(config=cfg)
    user_manager = UserManager(config=cfg)

    error_param = 'additive_error_on_cosine_of_aod'
    cfg.config_error_model.error_rng_parametrizations[error_param]['args']['low'] = -CSIT_ERROR_BOUND
    cfg.config_error_model.error_rng_parametrizations[error_param]['args']['high'] = CSIT_ERROR_BOUND

    circuit_power = cfg.sat_nr * cfg.sat_ant_nr * cfg.circuit_power_watt

    rate = np.zeros((MONTE_CARLO_SAMPLES, len(SCALE_GRID)))
    ee = np.zeros((MONTE_CARLO_SAMPLES, len(SCALE_GRID)))
    phys_power = np.zeros(MONTE_CARLO_SAMPLES)

    for iter_idx in range(MONTE_CARLO_SAMPLES):
        update_sim(cfg, satellite_manager, user_manager)
        w_phys = get_precoding_learned_clip_only(cfg, user_manager, satellite_manager, norm_factors, precoder_network)
        phys_power[iter_idx] = np.real(np.trace(np.matmul(w_phys.conj().T, w_phys)))
        for s_idx, s in enumerate(SCALE_GRID):
            rate[iter_idx, s_idx] = calc_sum_rate(
                channel_state=satellite_manager.channel_state_information,
                w_precoder=s * w_phys,
                noise_power_watt=cfg.noise_power_watt,
            )
            power_s = s ** 2 * phys_power[iter_idx]
            ee[iter_idx, s_idx] = rate[iter_idx, s_idx] / (power_s / cfg.pa_efficiency + circuit_power)

    mean_rate = rate.mean(axis=0)
    mean_ee = ee.mean(axis=0)
    peak_idx = int(np.argmax(mean_ee))
    print(f'\n[{label}]')
    print(f'  policy transmit power (clipped): mean {phys_power.mean():.2f} W '
          f'({100 * phys_power.mean() / cfg.power_constraint_watt:.1f}% of {cfg.power_constraint_watt:.0f} W budget)')
    print(f'  {"s":>5} {"power [W]":>10} {"rate":>8} {"EE [bps/Hz/W_DC]":>17}')
    for s_idx, s in enumerate(SCALE_GRID):
        marker = '  <-- EE peak' if s_idx == peak_idx else ''
        print(f'  {s:>5.2f} {s ** 2 * phys_power.mean():>10.2f} {mean_rate[s_idx]:>8.3f} {mean_ee[s_idx]:>17.5f}{marker}')
    print(f'  EE at s=1.00: {mean_ee[-1]:.5f}; EE at peak s={SCALE_GRID[peak_idx]:.2f}: {mean_ee[peak_idx]:.5f} '
          f'({100 * (mean_ee[peak_idx] / mean_ee[-1] - 1):+.1f}%)')

    return {
        'label': label,
        'power_budget': cfg.power_constraint_watt,
        'scale_grid': SCALE_GRID.copy(),
        'mean_phys_power': phys_power.mean(),
        'mean_rate': mean_rate,
        'mean_ee': mean_ee,
        'std_ee': ee.std(axis=0),
        'peak_scale': float(SCALE_GRID[peak_idx]),
    }


if __name__ == '__main__':
    results = {}
    for system in SYSTEMS:
        cfg = make_config(system['env'])
        cfg.config_learner.training_name = system['training_name']
        model_path = get_best_model_path(cfg.trained_models_path, system['training_name'])
        print(f"[{system['key']}] checkpoint: {model_path}")
        results[system['key']] = run_scaling_diagnostic(cfg, model_path, system['label'])

    out_path = Path(make_config({}).output_metrics_path, 'EE_power_scaling_diagnostic')
    out_path.mkdir(parents=True, exist_ok=True)
    with gzip.open(Path(out_path, 'power_scaling_diagnostic.gzip'), 'wb') as file:
        pickle.dump(results, file=file)
    print(f"\nSaved: {Path(out_path, 'power_scaling_diagnostic.gzip')}")

    plot_cfg = PlotConfig()
    matplotlib.rcParams['text.usetex'] = False  # PlotConfig() resets this on construction

    fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 4.2))
    for ax, (key, data) in zip(np.atleast_1d(axes), results.items()):
        ax.plot(data['scale_grid'], data['mean_ee'], marker='o', markersize=4, color='#307b3b')
        peak_i = int(np.argmax(data['mean_ee']))
        ax.axvline(data['scale_grid'][peak_i], color='#d03b3b', linestyle='--', linewidth=1,
                   label=f"EE peak at s={data['scale_grid'][peak_i]:.2f}")
        ax.set_xlabel('precoder scale s (s=1: policy output, clipped)')
        ax.set_ylabel('EE [bps/Hz per W DC]')
        ax.set_title(data['label'], fontsize=10)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, loc='best')

    fig.suptitle('Post-hoc power scaling: does down-scaling the trained precoder improve EE?', fontsize=12, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    pdf_path = Path(plot_cfg.plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)
    out = Path(pdf_path, 'ee_power_scaling_diagnostic.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=300, transparent=True)
    print(f'Saved: {out}')
    plt.close(fig)
