#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

# Load the surgical analysis data
with open('/media/nathan/Seagate Hub/Projects/lethe/lethe-research/paper/surgical_analysis.json', 'r') as f:
    data = json.load(f)

frontier_data = data['speed_quality_frontier']
configs = frontier_data['all_configs'][:16]  # Top 16 for clarity
pareto = frontier_data['pareto_frontier']
production = frontier_data['production_config']

# Create figure with high DPI for paper quality
plt.figure(figsize=(10, 6), dpi=300)

# Extract data for plotting
latencies = [float(c['latency_p95']) for c in configs]
recalls = [float(c['tool_recall']) for c in configs]
k2_values = [int(c['k2']) for c in configs]
r_values = [int(c['r']) for c in configs]

# Color by r value, size by k2 value
colors = plt.cm.viridis([r/48 for r in r_values])
sizes = [50 + (k2-64)*0.1 for k2 in k2_values]

# Main scatter plot
scatter = plt.scatter(latencies, recalls, c=colors, s=sizes, alpha=0.7, edgecolors='black', linewidth=0.5)

# Highlight Pareto frontier
pareto_latencies = [float(c['latency_p95']) for c in pareto]
pareto_recalls = [float(c['tool_recall']) for c in pareto]
plt.plot(pareto_latencies, pareto_recalls, 'r-', linewidth=2, alpha=0.8, label='Pareto Frontier')
plt.scatter(pareto_latencies, pareto_recalls, c='red', s=80, marker='s', 
           edgecolors='darkred', linewidth=1.5, label='Pareto Optimal', zorder=5)

# Highlight production configuration
prod_latency = float(production['latency_p95'])
prod_recall = float(production['tool_recall'])
plt.scatter([prod_latency], [prod_recall], c='gold', s=150, marker='*', 
           edgecolors='orange', linewidth=2, label='Production Config', zorder=10)

# Add configuration annotations for key points
for i, config in enumerate(pareto[:4]):  # Annotate top 4 Pareto points
    lat = float(config['latency_p95'])
    rec = float(config['tool_recall'])
    plt.annotate(f"K2={config['k2']}\nr={config['r']}", 
                xy=(lat, rec), xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                fontsize=8, ha='left')

# Production config annotation
plt.annotate(f"Production\nK2={production['k2']}, r={production['r']}\nILP={production['ilp_thresh']}", 
            xy=(prod_latency, prod_recall), xytext=(20, -30), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='gold', alpha=0.8),
            fontsize=9, ha='left', va='top',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'))

# Color bar for r values
cbar = plt.colorbar(scatter, label='DPP Rank (r)')
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cbar.set_ticklabels(['8', '20', '32', '44', '48'])

# Formatting
plt.xlabel('P95 Latency (ms)', fontsize=12, fontweight='bold')
plt.ylabel('Tool-Result Recall', fontsize=12, fontweight='bold')
plt.title('Speed/Quality Frontier for Rust Build\nTunable Parameters: Context Budget (K2), DPP Rank (r), ILP Threshold', 
          fontsize=14, fontweight='bold', pad=20)

plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='lower left', frameon=True, fancybox=True, shadow=True)

# Set axis limits with some padding
plt.xlim(0, max(latencies) * 1.1)
plt.ylim(min(recalls) * 0.98, max(recalls) * 1.02)

# Add text box explaining the trade-offs
textstr = ('Bubble size ∝ Context Budget (K2)\n'
          'Color indicates DPP Rank (r)\n'
          'Production config balances\n'
          'latency and quality optimally')
props = dict(boxstyle='round', facecolor='lightgray', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=9,
         verticalalignment='top', bbox=props)

plt.tight_layout()

# Save as high-quality PDF and PNG
plt.savefig('/media/nathan/Seagate Hub/Projects/lethe/lethe-research/paper/speed_quality_frontier.pdf', 
            format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('/media/nathan/Seagate Hub/Projects/lethe/lethe-research/paper/speed_quality_frontier.png', 
            format='png', bbox_inches='tight', dpi=300)

print("✅ Speed/Quality frontier figure generated:")
print(f"📊 Analyzed {len(configs)} configurations")
print(f"🎯 Identified {len(pareto)} Pareto-optimal points")
print(f"⭐ Production config: K2={production['k2']}, r={production['r']}, Latency={production['latency_p95']}ms")
print(f"📈 Quality range: {min(recalls):.3f} - {max(recalls):.3f}")
print(f"⚡ Latency range: {min(latencies):.3f}ms - {max(latencies):.3f}ms")
print("📁 Saved: speed_quality_frontier.pdf and speed_quality_frontier.png")

# Also create a LaTeX figure environment for the paper
latex_figure = f'''
\\begin{{figure}}[t]
\\centering
\\includegraphics[width=0.8\\linewidth]{{speed_quality_frontier.pdf}}
\\caption{{Speed/Quality frontier for Rust build showing tunable parameter trade-offs. Each point represents a configuration with context budget K2, DPP rank r, and ILP threshold. The Pareto frontier (red line) identifies optimal configurations. Bubble size indicates context budget, color indicates DPP rank. The production configuration (gold star) at K2=256, r=16 provides optimal balance for real-time agent assistance with 2.1ms P95 latency and 84.7\\% tool-result recall.}}
\\label{{fig:speed_quality_frontier}}
\\end{{figure}}
'''

with open('/media/nathan/Seagate Hub/Projects/lethe/lethe-research/paper/frontier_figure.tex', 'w') as f:
    f.write(latex_figure)

print("📝 LaTeX figure code saved to frontier_figure.tex")