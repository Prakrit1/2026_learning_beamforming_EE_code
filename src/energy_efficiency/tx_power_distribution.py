import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = False

"""
Clear comparison plot: does the no-normalization EE model actually save power
compared to the paper-normalization (Scheme I) model?

Produces a single figure with:
1. Overlaid total-power histograms (both models on same axes) — the main answer
2. A boxplot/violin showing the power distributions side by side — easy at-a-glance read
3. Per-user grouped boxplots — which user the power actually goes to, not just the total
4. Printed summary stats (mean, % of budget used, % below budget, per-user breakdown)

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
    model_colors: list = None,
):
    """
    samples_dict: {label: samples_array} where samples_array has shape
                  (monte_carlo_iterations, user_nr). Total power is computed
                  internally as samples.sum(axis=1).
    model_colors: optional explicit color list, one per entry in
                  samples_dict (same order). If omitted, falls back to the
                  original fixed 2-color categorical pair -- pass an explicit
                  list for anything other than a plain 2-model comparison
                  (e.g. a sequential ramp for an ordered parameter sweep).
    """
    import matplotlib
    matplotlib.rcParams['text.usetex'] = False
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    # Fixed-order categorical palette (validated set, see dataviz skill
    # references/palette.md) -- model colors and user colors are drawn from
    # non-adjacent slots in the same validated 8-hue sequence so both stay
    # colorblind-safe without needing their own separate validation pass.
    # Budget threshold uses this palette's reserved status-red, distinct from
    # every categorical slot, so it never reads as "a third series".
    if model_colors is None:
        model_colors = ['#2a78d6', '#1baf7a']  # blue, aqua
    user_colors = ['#eda100', '#4a3aa7']  # yellow, violet
    budget_color = '#d03b3b'  # status red

    # Wrap long labels onto two lines -- a single-line "SAC (EE, Dinkelbach,
    # no normalization)" tick label either overflows the axis or forces a
    # rotation that's awkward to read; two short lines fit horizontally.
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
    # Shared bin edges across all datasets: if left to auto-compute per
    # dataset, a zero-variance array (e.g. a model that is always pinned
    # exactly to the power budget) collapses to a near-zero-width bin,
    # producing a huge density spike that dwarfs any spread-out
    # distribution plotted alongside it and makes it look empty.
    ax = axes[0]
    for label, samples in samples_dict.items():
        total_powers[label] = samples.sum(axis=1)
    all_total_power = np.concatenate(list(total_powers.values()))
    hist_min, hist_max = all_total_power.min(), all_total_power.max()
    # A model that always rescales exactly to budget does NOT produce
    # bit-identical samples -- floating-point noise from the rescale
    # computation gives it a "std ~ 0.00" (rounds to zero at 2dp) but a
    # real, nonzero range on the order of 1e-5 W. Checking hist_min ==
    # hist_max misses this: bin_edges ends up spanning that ~1e-5 W range,
    # which is not degenerate to np.linspace/ax.hist but IS physically
    # meaningless -- both this histogram and panel 2's boxplot then
    # autoscale to that noise and label the axis with a confusing
    # "1e-5+1e2" offset instead of showing power against the budget.
    # Threshold on a physically meaningful fraction of the budget instead.
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
    # in a single bin, which would otherwise dwarf a spread-out tail on a
    # linear density axis and hide it
    ax.set_yscale('log')
    ax.grid(True, axis='y', alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    ax.axvline(power_budget, color=budget_color, linestyle='--', linewidth=1.5, label='Power budget')
    # Explicit fontsizes here (not PlotConfig's rc default, ~18.3pt for axis
    # labels) match plot_rate_error_sweep's legend/axis-label balance --
    # see that function's own comment for why the rc default looks oversized.
    ax.set_xlabel('Total transmit power [W]', fontsize=13)
    ax.set_ylabel('Density (log scale)', fontsize=13)
    ax.set_title('Total power distribution (overlaid)', fontsize=11)
    ax.legend(fontsize=11, loc='upper left')

    # Callout for any series pinned exactly at the budget: its histogram bar
    # is a single near-zero-width spike sitting right on top of the budget
    # line, easy to miss next to a spread-out distribution's much shorter
    # bars -- an explicit label makes clear it's there and what it means.
    #
    # Plain ax.text() on a blended (x=data, y=axes-fraction) transform, NOT
    # ax.annotate(..., arrowprops=...) with xy/xytext in DATA coordinates on
    # a log-scale y-axis: that combination was confirmed (see tx power
    # distribution work, N8K2 single-model case where std==0 exactly) to
    # corrupt the saved PDF's bounding box to ~150 million points wide under
    # bbox_inches='tight' -- matplotlib's FancyArrowPatch tight-bbox
    # calculation doesn't handle the log-scale data-coordinate transform
    # correctly here. Anchoring y to the axes fraction instead of
    # ylim-derived data coordinates sidesteps the bug entirely and doesn't
    # depend on the (log-scale-sensitive) ylim padding matplotlib happens to
    # pick.
    # Group pinned (near-zero-std) series by their OWN mean, not a hardcoded
    # power_budget x-position: with >2 curves it's common to have series
    # pinned at different fixed values (e.g. RM/MMSE pinned at the full
    # budget, a matched-power MMSE pinned at some other fixed watt value)
    # -- placing every callout at x=power_budget regardless of which value
    # a series is actually pinned at stacked them on top of each other and
    # mislabeled the ones not actually at budget.
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
        # y=0.72, not higher: the legend (loc='upper left') occupies roughly
        # the top ~20% of the panel, and this callout is centered at the
        # series' own pinned value -- for a single pinned series the x-axis
        # autoscales tightly around the spike, putting that x-position
        # directly under the legend if placed too high. zorder=10: Legend
        # artists default to zorder=5 regardless of add-order, so a pinned
        # value that lands under the legend's footprint (more likely now
        # that multiple curves can be pinned at different values) would
        # otherwise render hidden behind it despite being added after.
        ax.text(
            pinned_value, 0.72, f'{combined_label}\n{suffix}',
            transform=trans, fontsize=7.5, ha='center', va='top', zorder=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='0.5', alpha=0.85),
        )

    # ---- Panel 2: boxplot, the clearest at-a-glance comparison ----
    # showfliers=False + whis=[0, 100]: with 10k Monte Carlo samples, default
    # 1.5*IQR whiskers flag thousands of points as "outliers" and render them
    # as a solid smear of overlapping dots that buries the box itself. Using
    # the 0th/100th percentile AS the whisker ends shows the true min/max as
    # clean lines instead, losing no information while fixing readability.
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
        # same reasoning as panel 1's bin-range override: boxplot
        # autoscaling would otherwise zoom into ~1e-5 W floating-point
        # noise and label the axis with a confusing offset notation
        ax2.set_ylim(power_budget - pad, power_budget + pad)

    # ---- Panel 3: per-user power breakdown ----
    # samples_dict values are (monte_carlo_iterations, user_nr) -- per-user
    # power was already being computed (calc_tx_power_distribution returns
    # one entry per user) but silently discarded by the sum(axis=1) above.
    # This grouped boxplot -- one box per user, grouped by model -- answers
    # "which user is the power actually going to", not just the total.
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
    # rect leaves headroom for the suptitle -- plain tight_layout() doesn't
    # know about it and lets it collide with the panel titles.
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    pdf_path = Path(plots_parent_path, 'pdf')
    pdf_path.mkdir(parents=True, exist_ok=True)
    out = Path(pdf_path, f'{name}.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=300, transparent=True)
    print(f'Saved: {out}')

    jpg_path = Path(plots_parent_path, 'jpg')
    jpg_path.mkdir(parents=True, exist_ok=True)
    out_jpg = Path(jpg_path, f'{name}.jpg')
    fig.savefig(out_jpg, bbox_inches='tight', dpi=200)
    print(f'Saved: {out_jpg}')

    png_path = Path(plots_parent_path, 'png')
    png_path.mkdir(parents=True, exist_ok=True)
    out_png = Path(png_path, f'{name}.png')
    fig.savefig(out_png, bbox_inches='tight', dpi=200, transparent=True)
    print(f'Saved: {out_png}')

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
        samples = samples_dict[label]
        for user_idx in range(samples.shape[1]):
            user_power = samples[:, user_idx]
            print(
                f'    User {user_idx + 1}: mean {user_power.mean():.2f} W '
                f'({100 * user_power.mean() / total_power.mean():.1f}% of this model\'s total), '
                f'std {user_power.std():.2f} W'
            )
    print('=' * 60)


def plot_power_savings_bars(
    samples_dict: dict,
    power_budget: float,
    plots_parent_path: Path,
    name: str = 'power_savings_bars',
    model_colors: list = None,
    title: str = 'How much power does each model actually use?',
):
    """
    A "how much are we saving" chart anyone can read at a glance, no
    statistics background needed -- the boxplot/histogram version
    (plot_power_savings_comparison above) answers a different, more
    technical question (what does the full distribution look like) that
    requires knowing what a quartile or a log-density axis means. This one
    answers a single question directly: how does each model's power use
    compare to the budget, and how much is being saved?

    Each model gets ONE horizontal bar: a light gray "track" spans the full
    power budget (0 to power_budget), and a colored "fill" bar on top shows
    the actual mean power used -- the same visual language as a battery or
    loading-bar indicator. The unfilled gray portion IS the savings; no
    legend or distribution shape needs to be decoded to see it. Exact
    numbers are written directly on the chart (direct labeling, no legend
    needed for a single-series-per-row layout).

    samples_dict: {label: samples_array} where samples_array has shape
                  (monte_carlo_iterations, user_nr). Total power is computed
                  internally as samples.sum(axis=1); only the MEAN is shown
                  as the bar's fill (a thin whisker marks +-1 std for anyone
                  who does want the spread, but it's secondary, not the
                  headline).
    """
    import matplotlib
    matplotlib.rcParams['text.usetex'] = False

    track_color = '#e2e2e2'  # neutral light gray -- reads as "the whole budget", not a series
    budget_line_color = '#d03b3b'  # status red, same as the detailed plot above, for visual consistency
    text_color = '#2b2b2b'

    if model_colors is None:
        # validated categorical palette (dataviz skill), fixed order
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
        # track: the full budget, always the same length -- this is what
        # makes "how much is unfilled" directly readable as "how much saved"
        ax.barh(y, power_budget, height=bar_height, color=track_color, zorder=1, edgecolor='none')
        # fill: actual mean power used, drawn on top of the track
        ax.barh(y, mean_power, height=bar_height, color=color, zorder=2, edgecolor='none')
        # a thin whisker for +-1 std, secondary to the headline mean -- capped so it
        # never visually reads as a second bar
        ax.plot([max(0, mean_power - std_power), min(power_budget, mean_power + std_power)],
                [y, y], color=text_color, alpha=0.5, linewidth=1, zorder=3, solid_capstyle='round')

        pct_used = 100 * mean_power / power_budget
        watts_saved = power_budget - mean_power
        pct_saved = 100 * watts_saved / power_budget
        value_label = f'{mean_power:.0f} W ({pct_used:.0f}%)'

        # Fixed-width text on a data-coordinate x-axis: there's no reliable
        # way to know rendered text width before drawing without a two-pass
        # render, so approximate it instead of guessing a single threshold
        # that only happens to work for one particular set of numbers (a
        # flat "pct_used >= 25%" cutoff overflowed a real 34%-full bar in an
        # earlier version of this function -- the text just didn't fit).
        # ~1.3 data-units per character is a reasonable estimate for this
        # figure's width/fontsize combination; err on the generous side
        # (overestimate) since underestimating is what causes overflow.
        approx_char_width = 0.013 * power_budget
        label_width_est = approx_char_width * len(value_label)
        margin = 0.02 * power_budget

        space_after_bar = power_budget - mean_power  # remaining gray track
        if space_after_bar >= label_width_est + margin:
            # enough gray track room -- place just past the fill, dark text
            ax.text(mean_power + margin, y, value_label,
                    va='center', ha='left', fontsize=9.5, color=text_color, zorder=4, fontweight='bold')
        else:
            # not enough gray room -- place inside the filled bar instead,
            # right-aligned, white (readable against any of the categorical
            # fill colors); this is only reachable when the bar itself is at
            # least as wide as the label, since power_budget = mean_power +
            # space_after_bar and space_after_bar failed the check above
            ax.text(mean_power - margin, y, value_label,
                    va='center', ha='right', fontsize=9.5, color='white', zorder=4, fontweight='bold')

        # "how much are we saving" headline -- ALWAYS anchored past the
        # budget line, independent of the bar's own length, so it can never
        # collide with the value label above
        if watts_saved > 0.5:
            savings_text = f'saves {watts_saved:.0f} W ({pct_saved:.0f}%)'
            savings_color = '#1a7a3c'  # a confident green, read as "good news"
        else:
            savings_text = 'no savings (at budget)'
            savings_color = budget_line_color
        ax.text(power_budget * 1.06, y, savings_text, va='center', ha='left',
                fontsize=9.5, color=savings_color, zorder=4, style='italic')

    ax.axvline(power_budget, color=budget_line_color, linestyle='--', linewidth=1.5, zorder=1)
    # annotation sits ABOVE the highest bar's row (n - 1 is the topmost
    # y-position, bar_height/2 clears its top edge), not squeezed against it
    ax.text(power_budget, n - 1 + bar_height / 2 + 0.15, f'Power budget ({power_budget:.0f} W)',
            ha='center', va='bottom', fontsize=9, color=budget_line_color)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_xlabel('Transmit power [W]', fontsize=10.5)
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
    out = Path(pdf_path, f'{name}.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=300, transparent=True)
    print(f'Saved: {out}')

    # JPG alongside the PDF for quick viewing/sharing -- PDF stays the one
    # referenced from the LaTeX paper, this is not a replacement for it.
    # No transparent=True: JPG has no alpha channel, matplotlib just fills
    # with white.
    jpg_path = Path(plots_parent_path, 'jpg')
    jpg_path.mkdir(parents=True, exist_ok=True)
    out_jpg = Path(jpg_path, f'{name}.jpg')
    fig.savefig(out_jpg, bbox_inches='tight', dpi=200)
    print(f'Saved: {out_jpg}')

    png_path = Path(plots_parent_path, 'png')
    png_path.mkdir(parents=True, exist_ok=True)
    out_png = Path(png_path, f'{name}.png')
    fig.savefig(out_png, bbox_inches='tight', dpi=200, transparent=True)
    print(f'Saved: {out_png}')

    plt.close(fig)


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
        """
        Session-aware checkpoint selection -- picks the highest-reward
        checkpoint restricted to the most recent training session (gap >
        max_gap_minutes between saves marks a new session). Sorting by reward
        number across ALL checkpoints ever saved to this folder would silently
        pick weeks-old checkpoints from an earlier, unrelated session/reward
        formula whenever they happened to have a higher reward number -- see
        the identical function in my_evaluation.py for the case this was
        verified against.
        """
        import os

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

    def run_monte_carlo(cfg, model_path, label, clip_only=False, monte_carlo_iterations=10000, distance_value=25000):
        """
        distance_value=None: do NOT override user_distribution_mode/
        user_dist_average/user_dist_bound at all -- inherit config.py's
        defaults (100km mean, +-50km roam, matching the reference paper's
        Fig. 3 scenario, and the SAME scenario every training run and every
        rate-sweep evaluation this project has used). The historical
        default here (a hardcoded FIXED 25km distance, zero variance) is a
        different, non-paper-matching scenario that predates this session's
        investigation -- confirmed by the user mid-session that this was
        silently giving different (and wrong) power percentages (34%/64%
        of budget) versus the paper-matching scenario used elsewhere
        (56%/56-59%, see rate_power_error_sweep.py) for the SAME two
        checkpoints. Left as the default for EXISTING call sites in this
        file (the original 2-model comparison, lambda sweep, N8K2 block) to
        avoid silently changing already-established historical results
        without being asked to -- but any NEW call site should pass
        distance_value=None unless there's a specific reason not to.
        """
        precoder_network, norm_factors = load_model(model_path)
        if norm_factors != {}:
            cfg.config_learner.get_state_args['norm_state'] = True

        if distance_value is not None:
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
    # this matches how 'EE_dinkelbach_adaptive_aod0.5_lwin5000_N16K3_eta0.6'
    # was actually trained (see src/models/EE_sac.py,
    # EE_REWARD_MODE='energy_efficiency_dinkelbach_adaptive' ->
    # training_name='EE_dinkelbach_adaptive' + error/lwin/system/eta
    # suffixes); using the always-rescale path here would hide any power
    # savings it learned.
    # NOTE: these previously pointed at 'full_EE_aod0.5' /
    # 'EE_dinkelbach_adaptive_aod0.5' (N=8/K=2, eta=0.35) -- swapped to the
    # N=16/K=3, eta=0.60 models (matching the paper's Fig. 3 scenario and Ha
    # et al.'s satellite HPA efficiency). Evaluating the old checkpoints
    # against the CURRENT config.py (N=16/K=3) would crash with a state-size
    # mismatch, since Config() always reflects the current system size.
    models_to_check = [
        ('full_EE_aod0.5_N16K3_eta0.6', 'SAC (EE, paper normalization)', False),
        ('EE_dinkelbach_adaptive_aod0.5_lwin5000_N16K3_eta0.6', 'SAC (EE, Dinkelbach, no normalization)', True),
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

    # -----------------------------------------------------------------------
    # Fixed-lambda (classical Dinkelbach) sweep -- jobs 145428-145433. Six
    # independently trained checkpoints at N16K3/eta0.6/aod0.5, differing
    # only in EE_DINKELBACH_LAMBDA_FIXED. lambda is an ORDERED parameter, not
    # a set of unrelated categories, so per the dataviz skill's color-by-job
    # rule this uses a single-hue SEQUENTIAL ramp (light->dark as lambda
    # increases) instead of extending the categorical palette above to 6
    # arbitrary hues -- a rainbow of unrelated colors would visually imply
    # these are unrelated models rather than points along one continuum.
    #
    # Raw per-model power samples are also persisted to gzip (not just
    # plotted) so the rate-vs-lambda/power-vs-lambda plot (a separate script)
    # can reuse this Monte Carlo run instead of re-simulating it.
    # -----------------------------------------------------------------------
    import gzip
    import pickle

    lambda_values = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
    lambda_models_to_check = [
        (f'EE_dinkelbach_adaptive_aod0.5_lambdafixed{lv}_N16K3_eta0.6', f'SAC (Dinkelbach, λ={lv})', lv)
        for lv in lambda_values
    ]
    # sample a sequential green ramp (matches the 'green' categorical slot
    # used for Dinkelbach-family curves elsewhere in this project, e.g.
    # my_plotting.py's plot_cfg.cp2['green']), light->dark with increasing
    # lambda; skip the very lightest/darkest 15% so no sample is
    # near-invisible against a white page or near-black/illegible
    ramp = matplotlib.colormaps['Greens']
    lambda_colors = [
        matplotlib.colors.to_hex(ramp(0.3 + 0.65 * i / (len(lambda_values) - 1)))
        for i in range(len(lambda_values))
    ]

    lambda_samples_dict = {}
    lambda_power_budget = None
    lambda_summary = {}

    for training_name, label, lv in lambda_models_to_check:
        cfg = Config()
        cfg.show_plots = False
        cfg.config_learner.training_name = training_name
        lambda_power_budget = cfg.power_constraint_watt

        model_path = get_best_model_path(cfg.trained_models_path, training_name)
        samples = run_monte_carlo(cfg, model_path, label, clip_only=True)
        lambda_samples_dict[label] = samples
        total_power = samples.sum(axis=1)
        lambda_summary[lv] = {
            'mean_total_power_watt': float(total_power.mean()),
            'std_total_power_watt': float(total_power.std()),
        }

    plot_power_savings_comparison(
        samples_dict=lambda_samples_dict,
        power_budget=lambda_power_budget,
        plots_parent_path=plot_cfg.plots_parent_path,
        name='power_savings_comparison_lambda_sweep',
        model_colors=lambda_colors,
    )

    summary_path = Path(
        Config().output_metrics_path, 'EE_dinkelbach_adaptive_lambda_sweep', 'tx_power'
    )
    summary_path.mkdir(parents=True, exist_ok=True)
    with gzip.open(Path(summary_path, 'lambda_sweep_power_summary.gzip'), 'wb') as file:
        pickle.dump({'power_budget': lambda_power_budget, 'lambda_summary': lambda_summary}, file=file)
    print(f"Saved: {Path(summary_path, 'lambda_sweep_power_summary.gzip')}")

    # -----------------------------------------------------------------------
    # Raw-power-fixed Dinkelbach reward (see EE_sac.py's dead-gradient fix),
    # trained at Delta-epsilon_aod=0.0, 0.025, and 0.05 -- jobs
    # 147505/152304/147506. Tests "Finding 3": zero-error and
    # paper-matching-error training, as opposed to this week's original
    # (10x-too-large) 0.5 error bound. error=0.025 uses the TUNED-LR
    # checkpoint (job 152304), NOT the original default-LR one -- see
    # handoff_prompt_EE_evaluation.txt's "Optuna searches" section: this is
    # the only one of the three error bounds where the Optuna-tuned LR was
    # actually adopted (aod=0.0/0.05 tuned-LR variants did not improve on
    # default-LR and were not adopted). Zero CSIT-error power distribution
    # here, at the PAPER-MATCHING user distance scenario (config.py's
    # default: 100km mean, +-50km roam -- distance_value=None below, NOT
    # this script's other blocks' hardcoded fixed-25km-distance default; see
    # run_monte_carlo's docstring for why that distinction matters here
    # specifically). For the error-swept (rate, power) trajectory instead,
    # see rate_power_error_sweep.py, a separate script that already used the
    # correct, unoverridden scenario.
    # -----------------------------------------------------------------------
    error_bound_models_to_check = [
        ('EE_dinkelbach_adaptive_lwin5000_N16K3_eta0.6_rawpow', 'SAC (Dinkelbach, raw-power fix, error=0.0)'),
        ('EE_dinkelbach_adaptive_aod0.025_lwin5000_N16K3_eta0.6_rawpow_lrc1.82e-06_lra1.17e-06', 'SAC (Dinkelbach, raw-power fix, error=0.025, tuned LR)'),
        ('EE_dinkelbach_adaptive_aod0.05_lwin5000_N16K3_eta0.6_rawpow', 'SAC (Dinkelbach, raw-power fix, error=0.05)'),
    ]
    error_bound_colors = ['#307b3b', '#2f6fbf', '#1baf7a']  # green, blue, aqua -- matches Dinkelbach-family convention elsewhere

    error_bound_samples_dict = {}
    error_bound_power_budget = None

    for training_name, label in error_bound_models_to_check:
        cfg = Config()
        cfg.show_plots = False
        cfg.config_learner.training_name = training_name
        error_bound_power_budget = cfg.power_constraint_watt

        model_path = get_best_model_path(cfg.trained_models_path, training_name)
        error_bound_samples_dict[label] = run_monte_carlo(cfg, model_path, label, clip_only=True, distance_value=None)

    plot_power_savings_comparison(
        samples_dict=error_bound_samples_dict,
        power_budget=error_bound_power_budget,
        plots_parent_path=plot_cfg.plots_parent_path,
        name='power_savings_comparison_error_bound_0_0.025_0.05',
        model_colors=error_bound_colors,
    )
    # lay-friendly version of the same result, same samples (no extra Monte
    # Carlo needed) -- see plot_power_savings_bars' docstring for why this
    # exists alongside the boxplot/histogram version above.
    plot_power_savings_bars(
        samples_dict=error_bound_samples_dict,
        power_budget=error_bound_power_budget,
        plots_parent_path=plot_cfg.plots_parent_path,
        name='power_savings_bars_error_bound_0_0.025_0.05',
        model_colors=error_bound_colors,
        title='Fixed-training-error models: power used vs. budget (paper scenario, zero eval error)',
    )

    # -----------------------------------------------------------------------
    # N8K2 paper-reproduction sanity check -- job 145427. Scheme I (pure
    # sum-rate reward, always-rescale-to-budget precoder, same as the first
    # entry in models_to_check above) but at the OLDER N=8/K=2 system size,
    # via the EE_USER_NR/EE_SAT_TOT_ANT_NR env var overrides (see
    # my_evaluation.py's matching block for the same override). clip_only is
    # deliberately False here, NOT True like the lambda-sweep block above --
    # this model was trained with EE_REWARD_MODE=energy_efficiency
    # (always-rescale), not the Dinkelbach clip-only precoder.
    #
    # Because this precoder always rescales to the full budget, total power
    # is pinned (std ~ 0) by construction -- the informative part of this
    # plot is Panel 3, the per-user allocation, not the total-power
    # histogram/boxplot (Panels 1-2 will show a single spike). Still run
    # through the same plot_power_savings_comparison for consistency and to
    # numerically confirm the precoder is behaving as expected at this
    # system size (Mean total power should print at ~100% of budget).
    #
    # IMPORTANT: this block must run LAST in this file, same reasoning as
    # my_evaluation.py's N8K2 block -- the env var override below is not
    # restored afterward.
    # -----------------------------------------------------------------------
    import os
    os.environ['EE_USER_NR'] = '2'
    os.environ['EE_SAT_TOT_ANT_NR'] = '8'

    cfg_n8k2 = Config()
    cfg_n8k2.show_plots = False
    cfg_n8k2.config_learner.training_name = 'full_EE_aod0.5_N8K2_eta0.6'
    n8k2_power_budget = cfg_n8k2.power_constraint_watt

    model_path_n8k2 = get_best_model_path(cfg_n8k2.trained_models_path, 'full_EE_aod0.5_N8K2_eta0.6')
    n8k2_samples = run_monte_carlo(cfg_n8k2, model_path_n8k2, 'SAC (EE, paper normalization, N8K2 repro)', clip_only=False)

    plot_power_savings_comparison(
        samples_dict={'SAC (EE, paper normalization, N8K2 repro)': n8k2_samples},
        power_budget=n8k2_power_budget,
        plots_parent_path=plot_cfg.plots_parent_path,
        name='power_savings_N8K2_repro',
        model_colors=['#254796'],  # blue, matches plot_cfg.cp2['blue'] used for Scheme I elsewhere
    )
