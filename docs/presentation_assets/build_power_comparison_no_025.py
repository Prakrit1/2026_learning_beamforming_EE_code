import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

labels = ['MMSE', 'SAC (100 W, regular)', 'SAC (Energy-Efficient, error=0.0)',
          'SAC (Energy-Efficient, error=0.05)']
powers = [100.0, 100.0, 56.0, 56.3]
colors = ['#c9a227', '#3b6fa8', '#2c7a2c', '#4fc27a']
budget = 100.0

fig, ax = plt.subplots(figsize=(8.0, 5.2))
y_pos = range(len(labels))
ax.barh(y_pos, powers, color=colors, height=0.6, zorder=3)
ax.barh(y_pos, [budget - p for p in powers], left=powers, color='#e0e0e0', height=0.6, zorder=2)
ax.axvline(budget, color='crimson', linestyle='--', linewidth=1.5, zorder=4)
ax.text(budget, -0.75, 'Power budget (100 W)', color='crimson', ha='center', va='bottom', fontsize=11)

for i, p in enumerate(powers):
    if p >= budget - 0.5:
        ax.text(p - 3, i, f'{p:.0f} W ({p:.0f}%)', va='center', ha='right', fontsize=11, fontweight='bold', color='white')
        ax.text(budget + 5, i, 'no savings (at budget)', va='center', fontsize=10, color='crimson', style='italic')
    else:
        ax.text(p + 2, i, f'{p:.1f} W ({p:.0f}%)', va='center', fontsize=11, fontweight='bold')
        ax.text(budget + 12, i, f'saves {budget-p:.1f} W ({budget-p:.0f}%)', va='center', fontsize=10,
                 color='#2c7a2c', style='italic')

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=12)
ax.invert_yaxis()
ax.set_xlim(0, 185)
ax.set_ylim(len(labels) - 0.3, -1.1)
ax.set_xlabel('Transmit power [W]', fontsize=12)
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()

fig.savefig('/home/parajuli/repos/2025_learning_beamforming_rsma_code/reports/figures/pdf/power_comparison_mmse_regular_ee.pdf',
            bbox_inches='tight')
fig.savefig('/home/parajuli/repos/2025_learning_beamforming_rsma_code/docs/presentation_assets/power_comparison_new.png',
            dpi=350, transparent=True, bbox_inches='tight')
print('saved')
