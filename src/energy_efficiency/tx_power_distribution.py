import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = False

"""
Clear comparison plot: does the no-normalization EE model actually save power
compared to the paper-normalization (Scheme I) model?

Produces a single figure with:
1. Overlaid total-power histograms (both models on same axes) — the main answer
2. A boxplot/violin showing the power distributions side by side — easy at-a-glance read
3. Printed summary stats (mean, % of budget used, % below budget)

Run this AFTER you've generated the .gzip files via test_power_histogram_ee_vs_no_norm.py
(or it re-runs Monte Carlo itself if you prefer — see __main__ below).
"""

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False

from pathlib import Path
import gzip
import pickle

import numpy as np
import matplotlib.pyplot as plt


def plot_power_savings_comparison(
    samples_dict: dict,
    power_budget: float,
    plots_parent_path: Path,
    name: str = 'power_savings_comparison',
):
    """
    samples_dict: {label: samples_array} where samples_array has shape
                  (monte_carlo_iterations, user_nr). Total power is computed
                  internally as samples.sum(axis=1).
    """
    import matplotlib
    matplotlib.rcParams['text.usetex'] = False

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    colors = ['#254796', '#307b3b', '#caa023', '#d01b88']
    total_powers = {}

    # ---- Panel 1: overlaid histograms ----
    # Shared bin edges across all datasets: if left to auto-compute per
    # dataset, a zero-variance array (e.g. a model that is always pinned
    # exactly to the power budget) collapses to a near-zero-width bin,
    # producing a huge density spike that dwarfs any spread-out
    # distribution plotted alongside it and makes it look empty.
    ax = axes[0]
    for label, samples in samples_dict.items():
        total_powers[label] = samples.sum(axis=1)
    all_total_power = np.concatenate(list(total_powers.values()))
    bin_edges = np.linspace(all_total_power.min(), all_total_power.max(), 51)
    for (label, total_power), color in zip(total_powers.items(), colors):
        ax.hist(total_power, bins=bin_edges, alpha=0.5, label=label, color=color, density=True)

    # log scale: a model clipped exactly to budget puts a large point-mass
    # in a single bin, which would otherwise dwarf a spread-out tail on a
    # linear density axis and hide it
    ax.set_yscale('log')

    ax.axvline(power_budget, color='red', linestyle='--', linewidth=1.5, label='Power budget')
    ax.set_xlabel('Total transmit power [W]')
    ax.set_ylabel('Density')
    ax.set_title('Total power distribution (overlaid)')
    ax.legend(fontsize=8)

    # ---- Panel 2: boxplot, the clearest at-a-glance comparison ----
    ax2 = axes[1]
    labels = list(total_powers.keys())
    data = [total_powers[l] for l in labels]
    bp = ax2.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax2.axhline(power_budget, color='red', linestyle='--', linewidth=1.5, label='Power budget')
    ax2.set_ylabel('Total transmit power [W]')
    ax2.set_title('Total power: distribution summary')
    ax2.tick_params(axis='x', rotation=15)
    ax2.legend(fontsize=8)

    fig.suptitle('Is power being saved? Total transmit power vs budget', fontsize=12)
    plt.tight_layout()

    pdf_path = Path(plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)
    out = Path(pdf_path, f'{name}.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=800, transparent=True)
    print(f'Saved: {out}')
    plt.close(fig)

    # ---- printed summary, the actual numeric answer ----
    print('\n' + '=' * 60)
    print('POWER SAVINGS SUMMARY')
    print('=' * 60)
    for label, total_power in total_powers.items():
        pct_of_budget = 100 * total_power.mean() / power_budget
        pct_below_budget = 100 * (total_power < power_budget - 0.5).mean()
        print(f'{label}:')
        print(f'  Mean total power: {total_power.mean():.2f} W ({pct_of_budget:.1f}% of budget)')
        print(f'  Samples below budget (>0.5W margin): {pct_below_budget:.1f}%')
        print(f'  Min: {total_power.min():.2f} W, Max: {total_power.max():.2f} W, Std: {total_power.std():.2f} W')
    print('=' * 60)


if __name__ == '__main__':
    from src.config.config import Config
    from src.config.config_plotting import PlotConfig
    from src.utils.get_precoding import get_precoding_learned, get_precoding_learned_clip_only
    from src.utils.load_model import load_model
    from src.data.calc_tx_power_distribution import calc_tx_power_distribution
    from src.data.satellite_manager import SatelliteManager
    from src.data.user_manager import UserManager
    from src.utils.update_sim import update_sim

    def get_best_model_path(trained_models_path, training_name):
        base_path = Path(trained_models_path, training_name, 'base')
        checkpoints = [p for p in base_path.iterdir() if p.is_dir() and 'full_snap' in p.name]
        best = sorted(checkpoints, key=lambda p: float(p.name.split('_')[-1]))[-1]
        return best

    def run_monte_carlo(cfg, model_path, label, clip_only=False, monte_carlo_iterations=10000, distance_value=25000):
        precoder_network, norm_factors = load_model(model_path)
        if norm_factors != {}:
            cfg.config_learner.get_state_args['norm_state'] = True

        cfg.user_distribution_mode = 'uniform'
        cfg.user_dist_average = distance_value
        cfg.user_dist_bound = 0

        satellite_manager = SatelliteManager(config=cfg)
        user_manager = UserManager(config=cfg)

        # Prakrit added this for unnormalized (clip-only) power evaluation
        get_precoder_func = get_precoding_learned_clip_only if clip_only else get_precoding_learned

        samples = np.zeros((monte_carlo_iterations, cfg.user_nr))
        for iter_idx in range(monte_carlo_iterations):
            update_sim(cfg, satellite_manager, user_manager)
            w_precoder = get_precoder_func(cfg, user_manager, satellite_manager, norm_factors, precoder_network)
            samples[iter_idx, :] = calc_tx_power_distribution(w_precoder=w_precoder)
            if iter_idx % 1000 == 0:
                print(f'[{label}] {iter_idx}/{monte_carlo_iterations}')
        return samples

    # Edit these to match your actual trained model folder names
    # clip_only=True uses get_precoding_learned_clip_only (rescale down only
    # if over budget) instead of the always-rescale-to-budget normalization --
    # this matches how 'Energy_efficiency_without_normalization_fix' was
    # actually trained (see src/models/EE_sac.py); using the always-rescale
    # path here would hide any power savings it learned.
    models_to_check = [
        ('full_EE', 'SAC (EE, paper normalization)', False),
        ('Energy_efficiency_without_normalization_fix', 'SAC (EE, no normalization fix)', True),
    ]

    samples_dict = {}
    power_budget = None

    for training_name, label, clip_only in models_to_check:
        cfg = Config()
        cfg.show_plots = False
        cfg.config_learner.training_name = training_name
        power_budget = cfg.power_constraint_watt

        model_path = get_best_model_path(cfg.trained_models_path, training_name)
        samples_dict[label] = run_monte_carlo(cfg, model_path, label, clip_only=clip_only)

    plot_cfg = PlotConfig()
    plot_power_savings_comparison(
        samples_dict=samples_dict,
        power_budget=power_budget,
        plots_parent_path=plot_cfg.plots_parent_path,
        name='power_savings_comparison',
    )
