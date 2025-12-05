import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
adam_df = pd.read_csv('out/ViT Adam/lop_results.csv')
muon_df = pd.read_csv('out/Vit Muon /lop_results.csv')

# Get layer columns for dup_frac, eff_rank, and frozen_frac
dup_frac_cols = [col for col in adam_df.columns if col.startswith('dup_frac_L')]
eff_rank_cols = [col for col in adam_df.columns if col.startswith('eff_rank_L')]
frozen_frac_cols = [col for col in adam_df.columns if col.startswith('frozen_frac_L')]

# Compute average across layers
adam_df['avg_dup_frac'] = adam_df[dup_frac_cols].mean(axis=1)
adam_df['avg_eff_rank'] = adam_df[eff_rank_cols].mean(axis=1)
adam_df['avg_frozen_frac'] = adam_df[frozen_frac_cols].mean(axis=1)
muon_df['avg_dup_frac'] = muon_df[dup_frac_cols].mean(axis=1)
muon_df['avg_eff_rank'] = muon_df[eff_rank_cols].mean(axis=1)
muon_df['avg_frozen_frac'] = muon_df[frozen_frac_cols].mean(axis=1)

# Create figure with 8 subplots (4x2)
fig, axes = plt.subplots(4, 2, figsize=(14, 18))

# Plot 1: Validation Loss
ax1 = axes[0, 0]
ax1.plot(adam_df['task'], adam_df['val_CE'], label='Adam', color='blue', linewidth=2)
ax1.plot(muon_df['task'], muon_df['val_CE'], label='Muon', color='red', linewidth=2)
ax1.set_xlabel('Task')
ax1.set_ylabel('Validation Loss (CE)')
ax1.set_title('Validation Loss: Adam vs Muon')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Validation Accuracy
ax2 = axes[0, 1]
ax2.plot(adam_df['task'], adam_df['val_Acc'], label='Adam', color='blue', linewidth=2)
ax2.plot(muon_df['task'], muon_df['val_Acc'], label='Muon', color='red', linewidth=2)
ax2.set_xlabel('Task')
ax2.set_ylabel('Validation Accuracy')
ax2.set_title('Validation Accuracy: Adam vs Muon')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Average Duplicated Fraction
ax3 = axes[1, 0]
ax3.plot(adam_df['task'], adam_df['avg_dup_frac'], label='Adam', color='blue', linewidth=2)
ax3.plot(muon_df['task'], muon_df['avg_dup_frac'], label='Muon', color='red', linewidth=2)
ax3.set_xlabel('Task')
ax3.set_ylabel('Average Duplicated Fraction')
ax3.set_title('Average Duplicated Fraction: Adam vs Muon')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Average Effective Rank
ax4 = axes[1, 1]
ax4.plot(adam_df['task'], adam_df['avg_eff_rank'], label='Adam', color='blue', linewidth=2)
ax4.plot(muon_df['task'], muon_df['avg_eff_rank'], label='Muon', color='red', linewidth=2)
ax4.set_xlabel('Task')
ax4.set_ylabel('Average Effective Rank')
ax4.set_title('Average Effective Rank: Adam vs Muon')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Gradient Norm
ax5 = axes[2, 0]
ax5.plot(adam_df['task'], adam_df['grad_norm'], label='Adam', color='blue', linewidth=2)
ax5.plot(muon_df['task'], muon_df['grad_norm'], label='Muon', color='red', linewidth=2)
ax5.set_xlabel('Task')
ax5.set_ylabel('Gradient Norm')
ax5.set_title('Gradient Norm: Adam vs Muon')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Average Frozen Fraction
ax6 = axes[2, 1]
ax6.plot(adam_df['task'], adam_df['avg_frozen_frac'], label='Adam', color='blue', linewidth=2)
ax6.plot(muon_df['task'], muon_df['avg_frozen_frac'], label='Muon', color='red', linewidth=2)
ax6.set_xlabel('Task')
ax6.set_ylabel('Average Frozen Fraction')
ax6.set_title('Average Frozen Fraction: Adam vs Muon')
ax6.legend()
ax6.grid(True, alpha=0.3)

# Plot 7: Training Final CE Loss
ax7 = axes[3, 0]
ax7.plot(adam_df['task'], adam_df['final_CE'], label='Adam', color='blue', linewidth=2)
ax7.plot(muon_df['task'], muon_df['final_CE'], label='Muon', color='red', linewidth=2)
ax7.set_xlabel('Task')
ax7.set_ylabel('Final Training Loss (CE)')
ax7.set_title('Training Final CE Loss: Adam vs Muon')
ax7.legend()
ax7.grid(True, alpha=0.3)

# Plot 8: Training Min CE Loss
ax8 = axes[3, 1]
ax8.plot(adam_df['task'], adam_df['min_CE'], label='Adam', color='blue', linewidth=2)
ax8.plot(muon_df['task'], muon_df['min_CE'], label='Muon', color='red', linewidth=2)
ax8.set_xlabel('Task')
ax8.set_ylabel('Min Training Loss (CE)')
ax8.set_title('Training Min CE Loss: Adam vs Muon')
ax8.legend()
ax8.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('out/adam_vs_muon_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("Plot saved to out/adam_vs_muon_comparison.png")
