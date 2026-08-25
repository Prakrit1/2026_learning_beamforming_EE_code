"""Recover and plot the Dinkelbach lambda trajectory for the lambda-init run
(job 1175, training_name below), to check the claim made in 04_approach.tex's
"Reward Formulation Inspired by the Dinkelbach Transform": that seeding
lambda_0 from a closed-form MMSE rate/power prior (instead of 0) shortens the
reward-scale mis-pricing window (rate up to ~12 bps/Hz vs. power up to 75 W)
and that lambda then stabilizes.

lambda is NOT saved anywhere in the training pipeline's own metrics gzip --
EE_sac.py's top-level `metrics` dict only tracks mean_reward_per_episode
(see save_results()). The only per-episode record of lambda is the
"Dinkelbach lambda updated to X" line EE_sac.py logs once per episode, which
SLURM redirects to the raw .out file at the repo root (this project's
convention -- job logs land at the repo root, not next to the .slurm
scripts). This script parses that text log directly.

Saves outputs/metrics/<TRAINING_NAME>/base/lambda_convergence.gzip and
reports/figures/pdf/lambda_convergence_lambdainit.pdf.
"""
import gzip
import pickle
import re
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt

from src.config.config import Config
from src.config.config_plotting import PlotConfig

TRAINING_NAME = 'EE_dinkelbach_adaptive_lwin5000_lambdainit12.2-75_N16K3_satg30_p75_eta0.6_rawpow'
LOG_PATH = Path(__file__).resolve().parents[2] / 'EE_3gpp_nadir_aod0_lambdainit_1175.out'  # job 1175

# seed used by the .slurm launcher (EE_DINKELBACH_LAMBDA_INIT_RATE/_POWER),
# read off the MMSE baseline's own mean rate/power at the same 75 W budget
LAMBDA_INIT_RATE = 12.2
LAMBDA_INIT_POWER = 75.0
LAMBDA_0 = LAMBDA_INIT_RATE / LAMBDA_INIT_POWER

# how many trailing episodes define "converged" for the reference line/stats
CONVERGED_WINDOW = 2000

LAMBDA_LINE_RE = re.compile(
    r'Dinkelbach lambda updated to ([\d.]+) '
    r'\(episode mean rate ([\d.]+), mean power ([\d.]+) x budget '
    r'= ([\d.]+) W DC draw\)'
)


def parse_lambda_trajectory(log_path):
    """One 'Dinkelbach lambda updated to ...' line is logged per training
    episode, in order -- so the Nth match is episode N (see EE_sac.py's
    per-episode logging block, which fires unconditionally every episode
    for the per-step-EMA lambda mode used by this run)."""
    lambdas, rates, power_fracs, power_watts = [], [], [], []
    with open(log_path) as file:
        for line in file:
            match = LAMBDA_LINE_RE.search(line)
            if match:
                lambdas.append(float(match.group(1)))
                rates.append(float(match.group(2)))
                power_fracs.append(float(match.group(3)))
                power_watts.append(float(match.group(4)))

    if not lambdas:
        raise ValueError(f'No "Dinkelbach lambda updated to" lines found in {log_path}')

    return {
        'episode': np.arange(len(lambdas)),
        'lambda_ee': np.array(lambdas),
        'episode_mean_rate': np.array(rates),
        'episode_mean_power_frac': np.array(power_fracs),
        'episode_mean_power_watt': np.array(power_watts),
    }


if __name__ == '__main__':
    cfg = Config()
    plot_cfg = PlotConfig()
    matplotlib.rcParams['text.usetex'] = False  # override PlotConfig's reset

    data = parse_lambda_trajectory(LOG_PATH)
    print(f'Parsed {len(data["episode"])} episodes from {LOG_PATH.name}')

    converged_value = data['lambda_ee'][-CONVERGED_WINDOW:].mean()
    converged_std = data['lambda_ee'][-CONVERGED_WINDOW:].std()
    print(f'lambda_0 (seed) = {LAMBDA_0:.4f}')
    print(f'lambda at episode 0 (first logged) = {data["lambda_ee"][0]:.4f}')
    print(f'lambda over last {CONVERGED_WINDOW} episodes: mean={converged_value:.4f}, std={converged_std:.4f}')

    results_path = Path(cfg.output_metrics_path, TRAINING_NAME, 'base')
    results_path.mkdir(parents=True, exist_ok=True)
    gzip_path = Path(results_path, 'lambda_convergence.gzip')
    with gzip.open(gzip_path, 'wb') as file:
        pickle.dump({
            **data,
            'lambda_0': LAMBDA_0,
            'lambda_init_rate': LAMBDA_INIT_RATE,
            'lambda_init_power': LAMBDA_INIT_POWER,
            'converged_value': converged_value,
            'converged_std': converged_std,
            'converged_window': CONVERGED_WINDOW,
        }, file=file)
    print(f'Saved: {gzip_path}')

    pdf_path = Path(plot_cfg.plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)

    plot_width = 0.99 * plot_cfg.textwidth
    plot_height = plot_width * 0.55

    fig, axes = plt.subplots(1, 2, figsize=(2 * plot_width, plot_height))

    # ---- Panel 1: full trajectory ----
    ax0 = axes[0]
    ax0.plot(data['episode'], data['lambda_ee'], color=plot_cfg.cp2['blue'], linewidth=1.2)
    ax0.axhline(LAMBDA_0, color=plot_cfg.cp2['gold'], linestyle='--', linewidth=1.3,
                label=fr'seed $\lambda_0={LAMBDA_0:.3f}$ (MMSE $\overline{{R}}/\overline{{P}}$)')
    ax0.axhline(converged_value, color=plot_cfg.cp2['magenta'], linestyle='--', linewidth=1.3,
                label=fr'converged $\approx{converged_value:.3f}$ (last {CONVERGED_WINDOW} ep.)')
    ax0.set_xlabel('Training episode')
    ax0.set_ylabel(r'Dinkelbach $\lambda$')
    ax0.set_title('Full training run', fontsize=11)
    ax0.grid(True, alpha=0.25, linewidth=0.5)
    ax0.set_axisbelow(True)
    ax0.legend(fontsize=8, loc='lower right')

    # ---- Panel 2: early transient, zoomed ----
    zoom_episodes = 2000
    ax1 = axes[1]
    ax1.plot(data['episode'][:zoom_episodes], data['lambda_ee'][:zoom_episodes],
              color=plot_cfg.cp2['blue'], linewidth=1.2)
    ax1.axhline(LAMBDA_0, color=plot_cfg.cp2['gold'], linestyle='--', linewidth=1.3)
    ax1.axhline(converged_value, color=plot_cfg.cp2['magenta'], linestyle='--', linewidth=1.3)
    ax1.set_xlabel('Training episode')
    ax1.set_ylabel(r'Dinkelbach $\lambda$')
    ax1.set_title(f'First {zoom_episodes} episodes (mis-pricing window)', fontsize=11)
    ax1.grid(True, alpha=0.25, linewidth=0.5)
    ax1.set_axisbelow(True)

    fig.suptitle('Dinkelbach $\\lambda$ convergence, lambda-init run (job 1175)', fontsize=13, y=1.03)
    fig.tight_layout()

    out = Path(pdf_path, 'lambda_convergence_lambdainit.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=300, transparent=True)
    print(f'Saved: {out}')

    jpg_path = Path(plot_cfg.plots_parent_path, 'jpg')
    jpg_path.mkdir(parents=True, exist_ok=True)
    out_jpg = Path(jpg_path, 'lambda_convergence_lambdainit.jpg')
    fig.savefig(out_jpg, bbox_inches='tight', dpi=200)
    print(f'Saved: {out_jpg}')

    plt.close(fig)
