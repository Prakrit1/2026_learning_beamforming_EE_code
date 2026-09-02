import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_power_savings_bars(
    samples_dict: dict,
    power_budget: float,
    plots_parent_path: Path,
    name: str = 'power_savings_bars',
    model_colors: list = None,
    title: str = 'How much power does each model actually use?',
):
    """
    One horizontal loading-bar per model: a light gray track spans the full
    power budget, a colored fill shows the actual mean power used. The
    unfilled portion is the savings, readable at a glance without a legend.

    samples_dict: {label: samples_array}, samples_array shape
                  (monte_carlo_iterations, user_nr). Total power per draw is
                  samples.sum(axis=1); only the mean is shown as the bar's
                  fill (a thin whisker marks +-1 std).
    """
    track_color = '#e2e2e2'
    budget_line_color = '#d03b3b'
    text_color = '#2b2b2b'

    if model_colors is None:
        model_colors = ['#254796', '#307b3b', '#d01b88', '#caa023', '#7a4fbf', '#1baf7a', '#c9622a', '#5a5a5a']

    labels = list(samples_dict.keys())
    means = []
    stds = []
    for label in labels:
        total_power = samples_dict[label].sum(axis=1)
        means.append(total_power.mean())
        stds.append(total_power.std())

    n = len(labels)
    fig_height = max(2.2, 0.9 * n + 0.8)
    fig, ax = plt.subplots(figsize=(9, fig_height))

    y_positions = np.arange(n)[::-1]  # first entry in samples_dict drawn at the top
    bar_height = 0.5

    for y, label, mean_power, std_power, color in zip(y_positions, labels, means, stds, model_colors):
        ax.barh(y, power_budget, height=bar_height, color=track_color, zorder=1, edgecolor='none')
        ax.barh(y, mean_power, height=bar_height, color=color, zorder=2, edgecolor='none')
        ax.plot([max(0, mean_power - std_power), min(power_budget, mean_power + std_power)],
                [y, y], color=text_color, alpha=0.5, linewidth=1, zorder=3, solid_capstyle='round')

        pct_used = 100 * mean_power / power_budget
        value_label = f'{mean_power:.0f} W ({pct_used:.0f}%)'

        # No reliable way to know rendered text width before drawing, so
        # approximate it (generously, to avoid overflow) to decide whether
        # the value label fits past the bar or needs to go inside it.
        approx_char_width = 0.013 * power_budget
        label_width_est = approx_char_width * len(value_label)
        margin = 0.02 * power_budget

        space_after_bar = power_budget - mean_power
        if space_after_bar >= label_width_est + margin:
            ax.text(mean_power + margin, y, value_label,
                    va='center', ha='left', fontsize=9.5, color=text_color, zorder=4, fontweight='bold')
        else:
            ax.text(mean_power - margin, y, value_label,
                    va='center', ha='right', fontsize=9.5, color='white', zorder=4, fontweight='bold')

    ax.axvline(power_budget, color=budget_line_color, linestyle='--', linewidth=1.5, zorder=1)
    # $P_{\mathrm{rad}}$ notation (99_custommacros.sty's \transmitpower),
    # matching the error-sweep figure's legend.
    ax.text(power_budget, n - 1 + bar_height / 2 + 0.15, r'$P_{\mathrm{rad}}$',
            ha='center', va='bottom', fontsize=11, color=budget_line_color)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=11)  # this chart's equivalent of a legend
    ax.set_xlabel('Transmit power [W]', fontsize=13)
    ax.set_xlim(0, power_budget * 1.6)
    ax.set_ylim(-0.7, n - 1 + bar_height / 2 + 0.55)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.grid(True, axis='x', alpha=0.2, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=13, pad=14)

    fig.tight_layout()

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
