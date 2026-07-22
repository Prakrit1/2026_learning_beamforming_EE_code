"""
Plot how lambda evolves over training in the two VARIABLE-lambda Dinkelbach
modes (EMA and block-update) -- as opposed to the fixed-lambda sweep
(jobs 145428-145433), where lambda is a constant by construction and has no
convergence trajectory to plot.

There is no structured/persisted record of lambda's per-episode value
anywhere in this codebase -- EE_sac.py only ever logs it as a text INFO line
("Dinkelbach lambda updated to X ..." / "... (block boundary, N ep) to X ...",
see src/models/EE_sac.py's training loop) and never writes it to a
gzip/array. This script parses that line directly out of the raw SLURM
stdout logs, which is the ONLY place this data exists. If the underlying log
message format ever changes, the regex below needs to change with it.

Hardcoded to the two specific training runs identified as producing the
CURRENT checkpoints under models/EE_dinkelbach_adaptive_aod0.5_lwin5000_N16K3_eta0.6
and models/EE_dinkelbach_adaptive_aod0.5_block1_N16K3_eta0.6 (matched by
comparing checkpoint mtimes against .out file completion times -- there were
multiple lwin5000_N16K3_eta0.6 runs on disk (jobs 141885/141955/142617);
142617 is the one whose checkpoints (Jul 7 12:12-15:19) actually match the
current model folder's most recent training session).
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
import numpy as np

from src.config.config_plotting import PlotConfig

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

LOG_PATTERN = re.compile(r'Dinkelbach lambda updated(?: \(block boundary[^)]*\))? to ([\d.]+)')


def parse_lambda_trajectory(log_path: Path) -> np.ndarray:
    values = []
    with open(log_path, errors='replace') as file:
        for line in file:
            match = LOG_PATTERN.search(line)
            if match:
                values.append(float(match.group(1)))
    if not values:
        raise ValueError(f'No "Dinkelbach lambda updated" lines found in {log_path}')
    return np.array(values)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    # 'valid' shortens the array by window-1; pad the front with the
    # unsmoothed values so the smoothed curve still starts at episode 0
    smoothed = np.convolve(values, kernel, mode='valid')
    pad = values[:window - 1]
    return np.concatenate([pad, smoothed])


def plot_lambda_convergence(
    ema_log_path: Path,
    block_log_path: Path,
    plots_parent_path: Path,
    name: str = 'lambda_convergence',
    smoothing_window: int = 200,
):
    ema_lambda = parse_lambda_trajectory(ema_log_path)
    block_lambda = parse_lambda_trajectory(block_log_path)

    ema_smooth = rolling_mean(ema_lambda, smoothing_window)
    block_smooth = rolling_mean(block_lambda, smoothing_window)

    # PlotConfig() resets text.usetex to True on construction (see
    # my_plotting.py/tx_power_distribution.py's identical override) -- LaTeX
    # isn't set up for the \lambda label used below, so force it back off
    # here rather than relying on the module-level setting from import time.
    matplotlib.rcParams['text.usetex'] = False

    # same colors/linestyles as my_plotting.py's error-sweep comparison, so
    # 'EMA lambda' and 'block lambda' read as the same two series across
    # both figures
    ema_color = '#307b3b'  # green, plot_cfg.cp2['green']
    block_color = '#d01b88'  # magenta, plot_cfg.cp2['magenta']

    fig, ax = plt.subplots(figsize=(8, 5))

    ema_episodes = np.arange(len(ema_lambda))
    block_episodes = np.arange(len(block_lambda))

    # raw per-episode trace, faint -- shows the actual noise
    ax.plot(ema_episodes, ema_lambda, color=ema_color, alpha=0.15, linewidth=0.6)
    ax.plot(block_episodes, block_lambda, color=block_color, alpha=0.15, linewidth=0.6)

    # rolling-mean trace, bold -- shows the convergence trend
    ax.plot(ema_episodes, ema_smooth, color=ema_color, linewidth=2,
             linestyle='-', label=f'EMA lambda (smoothed, w={smoothing_window})')
    ax.plot(block_episodes, block_smooth, color=block_color, linewidth=2,
             linestyle='--', label=f'Block lambda (smoothed, w={smoothing_window})')

    ax.set_xlabel('Training episode')
    ax.set_ylabel(r'$\lambda$')
    ax.set_title('Dinkelbach λ convergence over training (variable-λ modes)', fontsize=12)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc='lower right')

    fig.tight_layout()

    pdf_path = Path(plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)
    out = Path(pdf_path, f'{name}.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=300, transparent=True)
    print(f'Saved: {out}')
    plt.close(fig)

    print('\n' + '=' * 60)
    print('LAMBDA CONVERGENCE SUMMARY')
    print('=' * 60)
    for label, values, smooth in [('EMA', ema_lambda, ema_smooth), ('Block', block_lambda, block_smooth)]:
        print(f'{label}: episode 0 lambda={values[0]:.4f}, '
              f'final smoothed lambda={smooth[-1]:.4f}, '
              f'last {smoothing_window} episodes std={values[-smoothing_window:].std():.4f}')
    print('=' * 60)


if __name__ == '__main__':
    plot_cfg = PlotConfig()
    plot_lambda_convergence(
        ema_log_path=Path(REPO_ROOT, 'EE_dinkelbach_aod0.5_lwin5000_N16K3_142617.out'),
        block_log_path=Path(REPO_ROOT, 'EE_dinkelbach_aod0.5_N16K3_block1_144104.out'),
        plots_parent_path=plot_cfg.plots_parent_path,
    )
