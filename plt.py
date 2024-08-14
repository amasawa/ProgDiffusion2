import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data
data = {
    "Methods": ["AE", "VAE", "β-VAE", "info-VAE", "DDPM", "DDIM", "LDM", "DiffAE", "infoDiffusion", "BFN", "ParamReL-ours"],
    "Generation-Quality": [False, False, False, False, True, True, True, True, True, True, True],
    "Low-Dimension": [True, True, True, True, False, False, True, True, True, False, True],
    "Continuous": [False, True, True, True, False, False, True, False, True, False, True],
    "Smooth": [False, True, True, True, False, False, False, False, True, False, True],
    "Time-Specific": [False, False, False, False, False, False, False, False, False, False, True]
}

# Create DataFrame
df = pd.DataFrame(data)

# Define pastel colors for True/False values
pastel_colors = {
    True: "#77DD77",  # Pastel Green
    False: "#FFB6C1"  # Pastel Pink
}

# Plot with pastel colors for True/False values
fig, ax = plt.subplots(figsize=(12, 6))

# Change background color
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Apply pastel color map
heatmap = sns.heatmap(df.set_index("Methods"), annot=True, fmt='', cmap=sns.color_palette([pastel_colors[False], pastel_colors[True]]), cbar=False, linewidths=0.5, linecolor='black', ax=ax)

# Set title and labels
# ax.set_title("Comparison of Representation Learning methods by Features", fontsize=14, fontweight='bold')
# ax.set_ylabel("Methods", fontsize=12, fontweight='bold')
# ax.set_xlabel("Features", fontsize=12, fontweight='bold')

# Set attribute labels bold and font size 10pt
for text in heatmap.get_xticklabels():
    text.set_fontsize(10)
    text.set_weight('bold')

# Adjust y-axis labels to be horizontal and wrap text if necessary
labels = [label.get_text() for label in heatmap.get_yticklabels()]
wrapped_labels = ['\n'.join(label.split()) for label in labels]
heatmap.set_yticklabels(wrapped_labels, rotation=0, ha='right', fontsize=10, weight='bold')

# Set x-axis labels to be horizontal
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=0, ha='center', fontsize=10, weight='bold')

plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.tight_layout()

# Save plot
plt.savefig("show.png", dpi=300, facecolor=fig.get_facecolor())

# Display plot
plt.show()
