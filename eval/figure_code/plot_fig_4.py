#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import os

# Set publication quality styling - Nature journal standards
plt.style.use('default')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11

# ============================================================================
# Typography and styling configuration
# ============================================================================
# Font sizes (in points) following Nature Medicine guidelines
FONTSIZE_PANEL_LABEL = 10      # Panel labels (a), b), c)) - bold
FONTSIZE_AXIS_LABEL = 9        # Axis labels (e.g., "Reasoning Complexity") - regular
FONTSIZE_TICK_LABEL = 7.5      # Tick labels - regular (7.5-8 pt range)
FONTSIZE_DATASET_LABEL = 8     # Dataset labels - regular (7-8 pt range)

# Font weights
FONTWEIGHT_PANEL_LABEL = 'bold'
FONTWEIGHT_REGULAR = 'normal'
FONTWEIGHT_DATASET_LABEL = 'bold'

# Figure dimensions
FIG_WIDTH = 7                  # inches
FIG_HEIGHT = 5.0               # inches (adjusted for stacked layout)

# Gridspec configuration
# Panel A: wider, Panels B and C: narrower (stacked)
# For 2x2 grid: width_ratios = [A_width, B/C_width]
# Making Panel A wider and right side narrower
PANEL_WIDTH_RATIOS = [0.75, 0.25]  # Panel A left (wider), B/C right (narrower)
PANEL_HEIGHT_RATIOS = [1, 1]          # Panel B top, Panel C bottom (equal heights)
GRIDSPEC_HSPACE = 0.50                # Vertical spacing between B and C (increased for better separation)
GRIDSPEC_WSPACE = 0.35                # Horizontal spacing between A and B/C
GRIDSPEC_LEFT = 0.12
GRIDSPEC_RIGHT = 0.96
GRIDSPEC_TOP = 0.88
GRIDSPEC_BOTTOM = 0.20

# Panel label positioning (consistent with fig2)
PANEL_LABEL_X_OFFSET = -0.2   # Offset from left edge
PANEL_LABEL_Y_OFFSET = 0.98    # Offset above panel top

# Line widths (adjusted for 7" figure)
LINEWIDTH_AXES = 0.8           # Axes/spines line width
LINEWIDTH_TICKS = 0.8          # Tick mark line width
LINEWIDTH_ELLIPSE_EDGE = 2.0   # Ellipse edge line width (matching dataset_scoring.ipynb)
LINEWIDTH_BAR_EDGE = 0.8       # Bar edge line width

# Marker sizes (in points^2 for scatter plots)
MARKERSIZE_SCATTER = 14       # Scatter plot markers (Panel A)
MARKERSIZE_LOLLIPOP = 9      # Lollipop chart markers (Panels B and C)

# Ellipse settings (matching dataset_scoring.ipynb exactly)
ELLIPSE_ALPHA = 0.25           # Ellipse fill transparency

# Tick sizes
TICKSIZE_MAJOR = 4             # Major tick size

# Apply line width and tick size settings
plt.rcParams['axes.linewidth'] = LINEWIDTH_AXES
plt.rcParams['xtick.major.width'] = LINEWIDTH_TICKS
plt.rcParams['ytick.major.width'] = LINEWIDTH_TICKS
plt.rcParams['xtick.major.size'] = TICKSIZE_MAJOR
plt.rcParams['ytick.major.size'] = TICKSIZE_MAJOR

# ============================================================================
# Data and helper functions (matching dataset_scoring.ipynb exactly)
# ============================================================================
# Offset map for label positioning (pos_off, x_off, y_off)
offset_map = {
    "PMC-VQA":    (1, 0.00, 0.01),
    "Path-VQA":   (-1,  0.00, 0.02),
    "NEJM":       (1,  0.00, 0.02),
    "SLAKE":      (-1, -0.04, 0.00),
    "JAMA":       (1,  0.00, 0.02),
    "MIMIC-CXR":  (1,  0.10, -0.05),
    "MMMU-C-M":   (-1, -0.00, 0.02),
    "VQA-RAD":    (1,  -0.07, -0.05),
    "OmniMedVQA": (-1, 0.00, 0.02),
}

# Color map (using provided color palette)
color_map = {
    "PMC-VQA":    "#4C72B0",  # navy blue
    "Path-VQA":   "#55A868",  # soft green
    "NEJM":       "#8172B2",  # muted purple
    "SLAKE":      "#C17C48",  # burnt orange
    "JAMA":       "#64A6A8",  # teal
    "MIMIC-CXR":  "#B04F6E",  # rose/maroon
    "MMMU-C-M":   "#CCB974",  # olove
    "VQA-RAD":    "#4878A8",  # slate blue-gray
    "OmniMedVQA": "#6A8A82",  # lighter muted purple (similar to #8172B2)
}

def get_offset(ds, default=(0.0, 0.0)):
    """return (pos_off, x_off, y_off) based on dataset name"""
    return offset_map.get(ds, default)

def get_color(ds, default="#1f77b4"):
    """return color based on dataset name"""
    return color_map.get(ds, default)

# Marker map - different marker types for each dataset
marker_map = {
    "PMC-VQA":    "o",  # circle
    "Path-VQA":   "s",  # square
    "NEJM":       "^",  # triangle up
    "SLAKE":      "D",  # diamond
    "JAMA":       "v",  # triangle down
    "MIMIC-CXR":  "p",  # pentagon
    "MMMU-C-M":   "h",  # hexagon
    "VQA-RAD":    "*",  # star
    "OmniMedVQA": "X",  # x
}

def get_marker(ds, default="o"):
    """return marker type based on dataset name"""
    return marker_map.get(ds, default)

def lighten_color(color, factor=0.5):
    """
    Lighten a color by blending with white.
    Args:
        color: Color in hex format (e.g., '#4C72B0') or matplotlib color name
        factor: Lightening factor (0.0 = original color, 1.0 = white)
    Returns:
        Lightened color in hex format
    """
    import matplotlib.colors as mcolors
    import numpy as np
    
    # Convert color to RGB
    rgb = mcolors.to_rgb(color)
    
    # Blend with white
    lightened_rgb = tuple(np.array(rgb) + (1 - np.array(rgb)) * factor)
    
    # Convert back to hex
    return mcolors.to_hex(lightened_rgb)

# Data dictionary (matching structure from previous version)
data = {    
    'JAMA': {
        'reasoning': 0.8916666666666670,
        'reasoning_var': 0.0385593220338983,
        'visual': 0.5,
        'visual_var': 0.083427495291902,
    },
    'MIMIC-CXR': {
        'reasoning': 0.0895833333333333,
        'reasoning_var': 0.039989406779661,
        'visual': 0.7000000000000000,
        'visual_var': 0.0722693032015066,
    },
    'MMMU-C-M': {
        'reasoning': 0.1979166666666670,
        'reasoning_var': 0.073534604519774,
        'visual': 0.5666666666666670,
        'visual_var': 0.0875706214689265,
    },
    'NEJM': {
        'reasoning': 0.5541666666666670,
        'reasoning_var': 0.0833686440677966,
        'visual': 0.5888888888888890,
        'visual_var': 0.1161487758945390,
    },
    'OmniMedVQA': {
        'reasoning': 0.05,
        'reasoning_var': 0.0268714689265536,
        'visual': 0.4916666666666670,
        'visual_var': 0.0671139359698681,
    },
    'PMC-VQA': {
        'reasoning': 0.1,
        'reasoning_var': 0.0429731638418079,
        'visual': 0.7083333333333330,
        'visual_var': 0.0865112994350282,
    },
    'Path-VQA': {
        'reasoning': 0.09375,
        'reasoning_var': 0.037058615819209,
        'visual': 0.6138888888888890,
        'visual_var': 0.0536487758945385,
    },
    'SLAKE': {
        'reasoning': 0.01875,
        'reasoning_var': 0.0088100282485875,
        'visual': 0.5916666666666670,
        'visual_var': 0.0723870056497175,
    },
    'VQA-RAD': {
        'reasoning': 0.0604166666666666,
        'reasoning_var': 0.0225459039548022,
        'visual': 0.6944444444444440,
        'visual_var': 0.0686911487758945,
    },  
}

# Convert data dict to DataFrame format (matching dataset_scoring.ipynb approach)
datasets = list(data.keys())
stats_data = []
for dataset in datasets:
    stats_data.append({
        'Dataset': dataset,
        'Reasoning_mean': data[dataset]['reasoning'],
        'Reasoning_var': data[dataset]['reasoning_var'],
        'Visual_mean': data[dataset]['visual'],
        'Visual_var': data[dataset]['visual_var'],
    })
stats_df = pd.DataFrame(stats_data)

# Prepare ranking data for panels B and C
# Sort by reasoning complexity (descending) for Panel B
reasoning_ranked = stats_df.sort_values('Reasoning_mean', ascending=False).reset_index(drop=True)
# Sort by visual complexity (descending) for Panel C
visual_ranked = stats_df.sort_values('Visual_mean', ascending=False).reset_index(drop=True)

# ============================================================================
# Create figure with 2x2 layout: A on left (spans both rows), B and C stacked on right
# ============================================================================
fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
gs = fig.add_gridspec(2, 2, width_ratios=PANEL_WIDTH_RATIOS, 
                      height_ratios=PANEL_HEIGHT_RATIOS,
                      hspace=GRIDSPEC_HSPACE, wspace=GRIDSPEC_WSPACE,
                      left=GRIDSPEC_LEFT, right=GRIDSPEC_RIGHT, 
                      top=GRIDSPEC_TOP, bottom=GRIDSPEC_BOTTOM)

# Panel A spans both rows in column 0
ax_a = fig.add_subplot(gs[:, 0])
# Panel B is top right (row 0, column 1)
ax_b = fig.add_subplot(gs[0, 1])
# Panel C is bottom right (row 1, column 1)
ax_c = fig.add_subplot(gs[1, 1])

axs = [ax_a, ax_b, ax_c]

# ============================================================================
# Panel A: Scatter plot (matching dataset_scoring.ipynb exactly)
# ============================================================================
ax = axs[0]
ax.set_facecolor('white')
ax.set_axisbelow(True)

# No grid for Panel A
ax.grid(False)

# Style parameters (matching fig_4v2.ipynb)
USE_GLOW = True
GLOW_LAYERS = 3
GLOW_ALPHA_BASE = 0.08
MARKER_EDGE_WIDTH = 2.0
MARKER_EDGE_COLOR = 'white'
TEXT_COLOR = '#000000'  # Black text

# Label offset map (matching fig_4v2.ipynb)
label_offset_map = {
    "PMC-VQA":    (0.12, 0.72),
    "Path-VQA":   (0.11, 0.60),
    "SLAKE":      (0.04, 0.57),
    "NEJM":       (0.48, 0.57),
    "JAMA":       (0.8, 0.5),
    "MIMIC-CXR":  (0.12, 0.7),
    "MMMU-C-M":   (0.21, 0.55),
    "VQA-RAD":    (0.04, 0.67),
    "OmniMedVQA": (0.07, 0.4716666),
}

# Plot data (matching fig_4v2.ipynb style)
for idx, row in stats_df.iterrows():
    dataset = row["Dataset"]
    colour = get_color(dataset)
    
    # Mean positions
    mean_x = row["Reasoning_mean"]
    mean_y = row["Visual_mean"]
    
    # Base ellipse dimensions
    base_width = row["Reasoning_var"] * 2 
    base_height = row["Visual_var"] * 2 
    
    # --- Subtle glow effect (scaled by variance) ---
    if USE_GLOW:
        for layer in range(GLOW_LAYERS, 0, -1):
            scale = 0.8 + (layer * 0.3)
            alpha = GLOW_ALPHA_BASE / layer
            
            glow = patches.Ellipse(
                (mean_x, mean_y),
                width=base_width * scale,
                height=base_height * scale,
                facecolor=colour,
                edgecolor='none',
                alpha=alpha,
                zorder=0
            )
            ax.add_patch(glow)
    
    # --- Uncertainty ellipses (1 SD) ---
    uncertainty = patches.Ellipse(
        (mean_x, mean_y),
        width=base_width,
        height=base_height,
        facecolor='none',
        edgecolor=colour,
        linewidth=1.5,
        linestyle='--',
        alpha=0.4,
        zorder=1
    )
    ax.add_patch(uncertainty)
    
    # --- Plot main points (no edges) ---
    ax.scatter(
        mean_x, mean_y,
        s=MARKERSIZE_SCATTER,
        c=[colour],
        marker=get_marker(dataset),
        alpha=0.9,
        zorder=3
    )
    
    # --- Labels near each point (black text, no background) ---
    if dataset in label_offset_map:
        offset_x, offset_y = label_offset_map[dataset]
        
        ax.text(offset_x, offset_y, dataset,
                fontsize=FONTSIZE_DATASET_LABEL, 
                fontweight='500',
                color=TEXT_COLOR,
                ha='left', va='center', 
                zorder=4)


# --- Axis styling (matching fig_4v2.ipynb) ---
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(0.35, 0.85)
ax.set_xlabel('Reasoning complexity', 
              fontsize=FONTSIZE_AXIS_LABEL,
              fontweight='normal',
              color=TEXT_COLOR)
ax.set_ylabel('Visual complexity', 
              fontsize=FONTSIZE_AXIS_LABEL,
              fontweight='normal',
              color=TEXT_COLOR)

# NO GRID for Nature style
ax.grid(False)
ax.set_axisbelow(True)

# Spines
for spine in ax.spines.values():
    spine.set_color('#222222')
    spine.set_linewidth(LINEWIDTH_AXES)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.tick_params(
    axis='both', 
    which='major',
    width=1.0, 
    length=4,
    colors=TEXT_COLOR,
    labelsize=FONTSIZE_TICK_LABEL
)

# ============================================================================
# Panel B: Ranking by Reasoning Complexity (Lollipop chart)
# ============================================================================
ax = axs[1]
ax.set_facecolor('white')

# Create lollipop chart (ranked by reasoning complexity)
y_pos = np.arange(len(reasoning_ranked))
colors_b = [get_color(row["Dataset"]) for _, row in reasoning_ranked.iterrows()]
markers_b = [get_marker(row["Dataset"]) for _, row in reasoning_ranked.iterrows()]
values_b = reasoning_ranked['Reasoning_mean'].values
var_b = reasoning_ranked['Reasoning_var'].values  # var as half-range

# Draw horizontal lines (lollipop stems) - use lighter colors
for i, (y, val, color) in enumerate(zip(y_pos, values_b, colors_b)):
    light_color = lighten_color(color, factor=0.6)  # Lighten by 60%
    ax.hlines(y, 0, val, colors=light_color, linewidth=LINEWIDTH_BAR_EDGE, alpha=0.8)

# Draw variance as range [mean - var, mean + var]
for i, (y, val, color, var) in enumerate(zip(y_pos, values_b, colors_b, var_b)):
    lo = max(0, val - var)
    hi = min(1, val + var)
    ax.hlines(y, lo, hi, colors=color, linewidth=1.5, alpha=0.7, zorder=2)
    ax.plot([lo, lo], [y - 0.15, y + 0.15], color=color, linewidth=0.8, alpha=0.7, zorder=2)
    ax.plot([hi, hi], [y - 0.15, y + 0.15], color=color, linewidth=0.8, alpha=0.7, zorder=2)

# Draw markers at the end (lollipop heads) - each with different marker type
for i, (val, y, color, marker) in enumerate(zip(values_b, y_pos, colors_b, markers_b)):
    ax.scatter(val, y, s=MARKERSIZE_LOLLIPOP, c=color, alpha=0.8, 
               edgecolors=color, linewidths=LINEWIDTH_BAR_EDGE, 
               marker=marker, zorder=3)

# Vertical gray dashed line at mean, with legend
mean_b = float(values_b.mean())
ax.axvline(mean_b, color='gray', linestyle='--', linewidth=1, alpha=0.8, zorder=1)
mean_label_b = f'Mean = {mean_b:.3f}'
leg_handle_b = Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5, label=mean_label_b)
ax.legend(handles=[leg_handle_b], loc='upper right', bbox_to_anchor=(1.0, 1.15), fontsize=FONTSIZE_TICK_LABEL, frameon=False)

# Y-axis: no spine, no tick marks; keep dataset labels
ax.set_yticks(y_pos)
ax.set_yticklabels(reasoning_ranked['Dataset'], fontsize=FONTSIZE_TICK_LABEL)
ax.spines['left'].set_visible(False)

# Style axis
ax.set_xlabel("Reasoning Complexity", fontsize=FONTSIZE_AXIS_LABEL, fontweight=FONTWEIGHT_REGULAR)
ax.set_xlim(0, 1)
ax.set_xticks([0, 0.5, 1])
ax.invert_yaxis()  # Top to bottom ranking (highest at top)

# Spine styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#222222')
ax.spines['bottom'].set_linewidth(LINEWIDTH_AXES)

# Tick styling (x only for length; y has no tick marks)
ax.tick_params(axis='both', which='major', 
              width=LINEWIDTH_TICKS, 
              length=TICKSIZE_MAJOR,
              labelsize=FONTSIZE_TICK_LABEL,
              colors='#555555')
ax.tick_params(axis='y', length=0)  # no y tick marks, keep labels

# No grid (no background vertical lines)
ax.grid(False)
ax.set_axisbelow(True)

# ============================================================================
# Panel C: Ranking by Visual Complexity (Lollipop chart)
# ============================================================================
ax = axs[2]
ax.set_facecolor('white')

# Create lollipop chart (ranked by visual complexity)
y_pos = np.arange(len(visual_ranked))
colors_c = [get_color(row["Dataset"]) for _, row in visual_ranked.iterrows()]
markers_c = [get_marker(row["Dataset"]) for _, row in visual_ranked.iterrows()]
values_c = visual_ranked['Visual_mean'].values
var_c = visual_ranked['Visual_var'].values  # var as half-range

# Draw horizontal lines (lollipop stems) - use lighter colors
for i, (y, val, color) in enumerate(zip(y_pos, values_c, colors_c)):
    light_color = lighten_color(color, factor=0.6)  # Lighten by 60%
    ax.hlines(y, 0, val, colors=light_color, linewidth=LINEWIDTH_BAR_EDGE, alpha=0.8)

# Draw variance as range [mean - var, mean + var]
for i, (y, val, color, var) in enumerate(zip(y_pos, values_c, colors_c, var_c)):
    lo = max(0, val - var)
    hi = min(1, val + var)
    ax.hlines(y, lo, hi, colors=color, linewidth=1.5, alpha=0.7, zorder=2)
    ax.plot([lo, lo], [y - 0.15, y + 0.15], color=color, linewidth=0.8, alpha=0.7, zorder=2)
    ax.plot([hi, hi], [y - 0.15, y + 0.15], color=color, linewidth=0.8, alpha=0.7, zorder=2)

# Draw markers at the end (lollipop heads) - each with different marker type
for i, (val, y, color, marker) in enumerate(zip(values_c, y_pos, colors_c, markers_c)):
    ax.scatter(val, y, s=MARKERSIZE_LOLLIPOP, c=color, alpha=0.8, 
               edgecolors=color, linewidths=LINEWIDTH_BAR_EDGE, 
               marker=marker, zorder=3)

# Vertical gray dashed line at mean, with legend
mean_c = float(values_c.mean())
ax.axvline(mean_c, color='gray', linestyle='--', linewidth=1, alpha=0.8, zorder=1)
mean_label_c = f'Mean = {mean_c:.3f}'
leg_handle_c = Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5, label=mean_label_c)
ax.legend(handles=[leg_handle_c], loc='upper right', bbox_to_anchor=(1.0, 1.15), fontsize=FONTSIZE_TICK_LABEL, frameon=False)

# Y-axis: no spine, no tick marks; keep dataset labels
ax.set_yticks(y_pos)
ax.set_yticklabels(visual_ranked['Dataset'], fontsize=FONTSIZE_TICK_LABEL)
ax.spines['left'].set_visible(False)

# Style axis
ax.set_xlabel("Visual Complexity", fontsize=FONTSIZE_AXIS_LABEL, fontweight=FONTWEIGHT_REGULAR)
ax.set_xlim(0, 1)
ax.set_xticks([0, 0.5, 1])
ax.invert_yaxis()  # Top to bottom ranking (highest at top)

# Spine styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#222222')
ax.spines['bottom'].set_linewidth(LINEWIDTH_AXES)

# Tick styling (x only for length; y has no tick marks)
ax.tick_params(axis='both', which='major', 
              width=LINEWIDTH_TICKS, 
              length=TICKSIZE_MAJOR,
              labelsize=FONTSIZE_TICK_LABEL,
              colors='#555555')
ax.tick_params(axis='y', length=0)  # no y tick marks, keep labels

# No grid (no background vertical lines)
ax.grid(False)
ax.set_axisbelow(True)

# ============================================================================
# Add panel labels (consistent with fig2)
# ============================================================================
panel_labels = ['a)', 'b)', 'c)']

# Get positions first to calculate common y position for panels A and B
pos_a = axs[0].get_position()
pos_b = axs[1].get_position()

# Use Panel B's top y position as reference for both A and B labels
common_label_y = pos_b.y1 + 0.01  # Slightly above panel B top

for i, (ax, label) in enumerate(zip(axs, panel_labels)):
    if i == 0:
        # Panel A: top left, same height as b), aligned left with b)/c)
        pos = ax.get_position()
        label_x = pos.x0 - 0.09  # Same offset as b) and c), so a) is further left
        label_y = common_label_y  # Same height as b)
        ax.text(label_x, label_y, label,
                transform=ax.figure.transFigure,
                fontsize=FONTSIZE_PANEL_LABEL, 
                fontweight=FONTWEIGHT_PANEL_LABEL,
                va='bottom', ha='right')
    else:
        # Panels B and C: position to the left, aligned with y-axis
        # Use figure coordinates to position relative to y-axis
        pos = ax.get_position()
        # Position label to the left of the y-axis (at x0 - small offset)
        label_x = pos.x0 - 0.09  # Small offset to the left of y-axis
        if i == 1:
            # Panel B: use common y position
            label_y = common_label_y
        else:
            # Panel C: use its own top position
            label_y = pos.y1 + 0.01  # Slightly above panel top
        ax.text(label_x, label_y, label,
                transform=ax.figure.transFigure,
                fontsize=FONTSIZE_PANEL_LABEL, 
                fontweight=FONTWEIGHT_PANEL_LABEL,
                va='bottom', ha='right')

# ============================================================================
# Adjust panel spacing
# ============================================================================
plt.tight_layout(rect=[GRIDSPEC_LEFT, GRIDSPEC_BOTTOM, GRIDSPEC_RIGHT, GRIDSPEC_TOP])

# Get positions
pos_a = axs[0].get_position()
pos_b = axs[1].get_position()
pos_c = axs[2].get_position()

# Ensure Panel A height matches combined height of B and C
# Panel A should span from B's top to C's bottom
combined_bc_height = pos_b.y1 - pos_c.y0
combined_bc_y0 = pos_c.y0

# Adjust Panel A to match B+C combined height and bottom position
axs[0].set_position([pos_a.x0, combined_bc_y0, pos_a.width, combined_bc_height])

# Add extra spacing between A and B/C if needed
extra_spacing_horizontal = 0.02
pos_a = axs[0].get_position()
pos_b = axs[1].get_position()
pos_c = axs[2].get_position()

# Ensure panels B and C are vertically aligned (same x-position and width)
# Use Panel B's position as reference for alignment
new_b_x0 = pos_b.x0 + extra_spacing_horizontal
new_b_width = pos_b.width

# Ensure proper vertical gap between B and C
# Calculate desired gap (in figure coordinates)
desired_gap = 0.12  # Gap in figure coordinates (approximately 0.12 = ~8.4mm for 7" figure, increased for better separation)
current_gap = pos_c.y1 - pos_b.y0  # Current gap (should be negative if overlapping)

# Align Panel C with Panel B (same x-position and width)
new_c_x0 = new_b_x0  # Same x-position as Panel B (ensures y-axes align)
new_c_width = new_b_width  # Same width as Panel B

# If panels are too close or overlapping, adjust positions
if current_gap < desired_gap:
    # Move panel C down to create proper gap
    new_c_y0 = pos_b.y0 - desired_gap - pos_c.height
    axs[2].set_position([new_c_x0, new_c_y0, new_c_width, pos_c.height])
    
    # Re-adjust Panel A to match new B+C combined height
    pos_b = axs[1].get_position()
    pos_c = axs[2].get_position()
    combined_bc_height = pos_b.y1 - pos_c.y0
    combined_bc_y0 = pos_c.y0
    axs[0].set_position([pos_a.x0, combined_bc_y0, pos_a.width, combined_bc_height])
else:
    # Apply horizontal spacing and ensure alignment
    axs[1].set_position([new_b_x0, pos_b.y0, new_b_width, pos_b.height])
    axs[2].set_position([new_c_x0, pos_c.y0, new_c_width, pos_c.height])

# Final alignment check: ensure y-axes are on the same vertical line
# Get final positions after all adjustments
pos_b_final = axs[1].get_position()
pos_c_final = axs[2].get_position()

# Use the leftmost x-position to ensure y-axes align
aligned_x0 = min(pos_b_final.x0, pos_c_final.x0)

# Adjust both panels to use the same left edge (y-axis position)
axs[1].set_position([aligned_x0, pos_b_final.y0, pos_b_final.width, pos_b_final.height])
axs[2].set_position([aligned_x0, pos_c_final.y0, pos_c_final.width, pos_c_final.height])

# ============================================================================
# Save figure
# ============================================================================
# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

plt.savefig('output/figure4_complexity.pdf', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')

plt.show()

# %%
