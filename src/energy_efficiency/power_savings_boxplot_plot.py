import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_power_savings_comparison(
    samples_dict: dict,
    power_budget: float,
    plots_parent_path: Path,
    name: str = 'power_savings_comparison',
    model_colors: list = None,
):
    """
    3-panel figure: overlaid total-power histograms, a total-power boxplot,
    and a per-user grouped boxplot -- the detailed distribution view (as
    opposed to power_savings_bars_plot.py's single-glance loading bars).

    samples_dict: {label: samples_array} where samples_array has shape
                  (monte_carlo_iterations, user_nr). Total power is computed
                  internally as samples.sum(axis=1).
    model_colors: optional explicit color list, one per entry in
                  samples_dict (same order). If omitted, falls back to the
                  original fixed 2-color categorical pair -- pass an explicit
                  list for anything other than a plain 2-model comparison.
    """
    # Re-assert here, not just at import time: PlotConfig() (constructed by
    # the calling script right before this) resets text.usetex to True,
    # which makes matplotlib invoke real LaTeX for every text/label and
    # crash if it isn't installed.
    matplotlib.rcParams['text.usetex'] = False
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    if model_colors is None:
        model_colors = ['#2a78d6', '#1baf7a']  # blue, aqua
    user_colors = ['#eda100', '#4a3aa7']  # yellow, violet
    budget_color = '#d03b3b'  # status red

    # Wrap long labels onto two lines so they fit horizontally on tick axes.
    def wrap_label(label, width=18):
        words = label.replace('(', '').replace(')', '').split(' ')
        lines, current = [], ''
        for word in words:
            candidate = f'{current} {word}'.strip()
            if len(candidate) > width and current:
                lines.append(current)
                current = word
            else:
                current = candidate
        lines.append(current)
        return '\n'.join(lines)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    total_powers = {}

    # ---- Panel 1: overlaid histograms ----
    ax = axes[0]
    for label, samples in samples_dict.items():
        total_powers[label] = samples.sum(axis=1)
    all_total_power = np.concatenate(list(total_powers.values()))
    hist_min, hist_max = all_total_power.min(), all_total_power.max()
    # A model pinned exactly to budget isn't bit-identical across samples
    # (floating-point rescale noise, ~1e-5 W) -- that's not degenerate to
    # np.linspace but is physically meaningless, so both this histogram and
    # panel 2's boxplot would otherwise autoscale to that noise. Threshold
    # on a physically meaningful fraction of the budget instead.
    degenerate_range = (hist_max - hist_min) < 0.02 * power_budget
    pad = max(0.5, 0.01 * power_budget)
    if degenerate_range:
        hist_min, hist_max = power_budget - pad, power_budget + pad
    bin_edges = np.linspace(hist_min, hist_max, 51)
    for (label, total_power), color in zip(total_powers.items(), model_colors):
        ax.hist(
            total_power, bins=bin_edges, alpha=0.55, label=label, color=color,
            density=True, edgecolor='white', linewidth=0.3
        )

    # log scale: a model clipped exactly to budget puts a large point-mass
    # in a single bin that would otherwise dwarf a spread-out tail
    ax.set_yscale('log')
    ax.grid(True, axis='y', alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    ax.axvline(power_budget, color=budget_color, linestyle='--', linewidth=1.5, label='Power budget')
    ax.set_xlabel('Total transmit power [W]', fontsize=13)
    ax.set_ylabel('Density (log scale)', fontsize=13)
    ax.set_title('Total power distribution (overlaid)', fontsize=11)
    ax.legend(fontsize=11, loc='upper left')

    # Callout for any series pinned exactly at the budget: its histogram bar
    # is a near-zero-width spike, easy to miss next to a spread-out
    # distribution. Grouped by each series' OWN mean (not a hardcoded
    # power_budget x-position) since with >2 curves it's common to have
    # series pinned at different fixed values, not just the budget.
    #
    # Plain ax.text() on a blended transform, NOT ax.annotate(...,
    # arrowprops=...) with data-coordinate xy/xytext on a log-scale y-axis:
    # that combination corrupts the saved PDF's bounding box (matplotlib's
    # FancyArrowPatch tight-bbox calc doesn't handle it) -- anchoring y to
    # the axes fraction sidesteps the bug.
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    pinned_groups: dict[float, list[str]] = {}
    for label, total_power in total_powers.items():
        if total_power.std() < 0.01 * power_budget:
            pinned_groups.setdefault(round(total_power.mean(), 1), []).append(label)
    for pinned_value, labels_at_value in pinned_groups.items():
        at_budget = abs(pinned_value - power_budget) < 0.01 * power_budget
        suffix = '(pinned at budget)' if at_budget else f'(pinned at {pinned_value:.1f} W)'
        combined_label = ', '.join(wrap_label(l, width=22) for l in labels_at_value)
        # zorder=10: legend artists default to zorder=5 regardless of
        # add-order, so a pinned value under the legend's footprint would
        # otherwise render hidden behind it.
        ax.text(
            pinned_value, 0.72, f'{combined_label}\n{suffix}',
            transform=trans, fontsize=7.5, ha='center', va='top', zorder=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='0.5', alpha=0.85),
        )

    # ---- Panel 2: boxplot ----
    # showfliers=False + whis=[0, 100]: with 10k MC samples, default
    # 1.5*IQR whiskers flag thousands as "outliers"; using the 0th/100th
    # percentile as the whisker ends shows the true min/max cleanly instead.
    ax2 = axes[1]
    labels = list(total_powers.keys())
    wrapped_labels = [wrap_label(l) for l in labels]
    data = [total_powers[l] for l in labels]
    bp = ax2.boxplot(
        data, labels=wrapped_labels, patch_artist=True, showmeans=True,
        showfliers=False, whis=[0, 100], widths=0.5
    )
    for patch, color in zip(bp['boxes'], model_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    ax2.axhline(power_budget, color=budget_color, linestyle='--', linewidth=1.5, label='Power budget')
    ax2.grid(True, axis='y', alpha=0.25, linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.set_ylabel('Total transmit power [W]', fontsize=13)
    ax2.set_title('Total power: distribution summary', fontsize=11)
    ax2.tick_params(axis='x', labelsize=9)
    ax2.legend(fontsize=11, loc='lower left')
    if degenerate_range:
        ax2.set_ylim(power_budget - pad, power_budget + pad)

    # ---- Panel 3: per-user power breakdown ----
    ax3 = axes[2]
    labels = list(samples_dict.keys())
    user_nr = next(iter(samples_dict.values())).shape[1]
    group_width = 0.7
    box_width = group_width / user_nr

    positions = []
    per_user_data = []
    box_colors = []
    group_centers = np.arange(len(labels))
    for model_idx, label in enumerate(labels):
        samples = samples_dict[label]
        for user_idx in range(user_nr):
            positions.append(group_centers[model_idx] - group_width / 2 + box_width * (user_idx + 0.5))
            per_user_data.append(samples[:, user_idx])
            box_colors.append(user_colors[user_idx % len(user_colors)])

    bp3 = ax3.boxplot(
        per_user_data, positions=positions, widths=box_width * 0.85,
        patch_artist=True, showmeans=True, showfliers=False, whis=[0, 100]
    )
    for patch, color in zip(bp3['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    ax3.set_xticks(group_centers)
    ax3.set_xticklabels([wrap_label(l) for l in labels])
    ax3.axhline(power_budget, color=budget_color, linestyle='--', linewidth=1.2)
    ax3.grid(True, axis='y', alpha=0.25, linewidth=0.5)
    ax3.set_axisbelow(True)
    ax3.set_ylabel('Per-user transmit power [W]', fontsize=13)
    ax3.set_title('Per-user power allocation', fontsize=11)
    ax3.tick_params(axis='x', labelsize=9)
    legend_handles = [
        Patch(facecolor=user_colors[u % len(user_colors)], alpha=0.65, label=f'User {u + 1}')
        for u in range(user_nr)
    ] + [Line2D([0], [0], color=budget_color, linestyle='--', label='Power budget')]
    ax3.legend(handles=legend_handles, fontsize=11, loc='upper right')

    fig.suptitle('Is power being saved? Total transmit power vs budget', fontsize=13, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # rect leaves headroom for the suptitle

    pdf_path = Path(plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(pdf_path, f'{name}.pdf'), bbox_inches='tight', dpi=300, transparent=True)
    print(f'Saved: {Path(pdf_path, f"{name}.pdf")}')

    jpg_path = Path(plots_parent_path, 'jpg')
    jpg_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(jpg_path, f'{name}.jpg'), bbox_inches='tight', dpi=200)
    print(f'Saved: {Path(jpg_path, f"{name}.jpg")}')

    png_path = Path(plots_parent_path, 'png')
    png_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(png_path, f'{name}.png'), bbox_inches='tight', dpi=200, transparent=True)
    print(f'Saved: {Path(png_path, f"{name}.png")}')

    plt.close(fig)

    # ---- printed summary ----
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
        samples = samples_dict[label]
        for user_idx in range(samples.shape[1]):
            user_power = samples[:, user_idx]
            print(
                f'    User {user_idx + 1}: mean {user_power.mean():.2f} W '
                f'({100 * user_power.mean() / total_power.mean():.1f}% of this model\'s total), '
                f'std {user_power.std():.2f} W'
            )
    print('=' * 60)
