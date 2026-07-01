#%%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

# Set publication quality styling - Nature journal standards
plt.style.use('default')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11

# ============================================================================
# Typography and styling configuration
# ============================================================================
# Font sizes (in points) following Nature Medicine guidelines
FONTSIZE_PANEL_LABEL = 10      # Panel labels (a), b), c)) - bold
FONTSIZE_AXIS_LABEL = 8.5        # Axis labels (e.g., "Accuracy (%)") - regular
FONTSIZE_TICK_LABEL = 7      # Tick labels - regular (7.5-8 pt range, 7 pt OK if crowded)
FONTSIZE_BAR_ANNOTATION = 7    # Bar value annotations - regular (only if necessary)
FONTSIZE_LEGEND = 7            # Legend text - regular (7-8 pt range, match tick-label size)
FONTSIZE_CAPTION = 7           # Panel captions - regular (never bold)
FONTSIZE_ROW_LABEL = 10        # Row labels (Image+Text, Text only) - regular

# Font weights
FONTWEIGHT_PANEL_LABEL = 'bold'
FONTWEIGHT_REGULAR = 'normal'

# Figure dimensions
FIG_WIDTH = 8                  # inches
FIG_HEIGHT = 7               # inches (adjusted for 8" width)

# Gridspec configuration
GRIDSPEC_HSPACE = 0.3         # Vertical spacing between rows (increased)
GRIDSPEC_WSPACE = 0.3         # Horizontal spacing between columns
GRIDSPEC_LEFT = 0.10           # Left margin
GRIDSPEC_RIGHT = 0.97          # Right margin
GRIDSPEC_TOP = 0.92            # Top margin
GRIDSPEC_BOTTOM = 0.22         # Bottom margin (space for legends and captions)

# Row label positioning
ROW_LABEL_MARGIN = 0.04       # 5mm margin from left edge
ROW_LABEL_Y_OFFSET = 0.04     # Offset above row top

# Column header positioning
COL_HEADER_OFFSET = 0.1      # Offset below second row (moved lower)

# Legend positioning
LEGEND_ANCHOR_Y = -0.25        # Negative value drops legend below axes
LEGEND_FIRST_OFFSET = 0.04     # Offset below captions for first legend row
LEGEND_SPACING = 0.035          # Spacing between legend rows

# Panel caption positioning
CAPTION_OFFSET = -0.50         # Offset below axes for captions

# Panel label positioning
PANEL_LABEL_X_OFFSET = 0.03    # Offset to the left of y-axis
PANEL_LABEL_Y_OFFSET = 0.01   # Offset above panel top

# Line widths (adjusted for 8" figure)
LINEWIDTH_AXES = 0.8           # Axes/spines line width
LINEWIDTH_TICKS = 0.8          # Tick mark line width
LINEWIDTH_PLOT = 1.0           # Plot lines (e.g., panel C/G)
LINEWIDTH_ERRORBAR = 0.8       # Error bar line width
LINEWIDTH_CHANCE = 1.0         # Chance level reference line
LINEWIDTH_LEGEND = 1.0         # Legend line width
LINEWIDTH_GRID = 0.3           # Grid line width
LINEWIDTH_HATCH = 0.8          # Hatch pattern line width (separate from edge)
LINEWIDTH_BAR_EDGE = 0.6       # Bar edge line width

# Marker and cap sizes (adjusted for 7" figure)
MARKERSIZE_PLOT = 4            # Plot markers (circles, squares)
MARKERSIZE_LEGEND = 5          # Legend markers
CAPSIZE_ERRORBAR = 1.2         # Error bar cap size

# Hatch pattern styling
HATCH_PATTERN = '///'            # Hatch pattern for perturbed bars
BAR_EDGE_COLOR_HATCHED = '#000000'  # Edge color for hatched bars (black)
HATCH_FACECOLOR = 'white'       # Face color for hatched bars
HATCH_DARKEN_FACTOR = 0.75      # Factor to darken model color for edge (0.75 = 75% darker)

# Tick sizes (adjusted for 7" figure)
TICKSIZE_MAJOR = 3             # Major tick size

# Apply line width and tick size settings
plt.rcParams['axes.linewidth'] = LINEWIDTH_AXES
plt.rcParams['xtick.major.width'] = LINEWIDTH_TICKS
plt.rcParams['ytick.major.width'] = LINEWIDTH_TICKS
plt.rcParams['xtick.major.size'] = TICKSIZE_MAJOR
plt.rcParams['ytick.major.size'] = TICKSIZE_MAJOR
plt.rcParams['hatch.linewidth'] = LINEWIDTH_HATCH  # Hatch pattern line width

# ============================================================================
# Helper function to darken colors
# ============================================================================
def darken_color(color, factor):
    """
    Darken a color by a given factor.
    Args:
        color: Color in hex format (e.g., '#4A90A4') or matplotlib color name
        factor: Darkening factor (0.0 = black, 1.0 = original color)
    Returns:
        Darkened color in hex format
    """
    import matplotlib.colors as mcolors
    # Convert to RGB
    rgb = mcolors.to_rgb(color)
    # Darken by multiplying by factor
    darkened_rgb = tuple(c * factor for c in rgb)
    # Convert back to hex
    return mcolors.rgb2hex(darkened_rgb)

# ============================================================================

# Models and colors (matching figure 2)
models = ['GPT-5', 'Gemini-2.5 Pro', 'OpenAI-o3', 'OpenAI-o4-mini','Claude-sonnet-3.5','GPT-4o']
model_colors = ['#4A90A4', '#C77D4A', '#7A9E8F', '#D4A574', '#8B7A9E', '#6A4A7A']  # Muted teal, coral, green-teal, tan, purple-gray

# Data (from tables)
# T3: Format Perturbation
text_orig = [41.32, 40.10, 38.58, 37.97, 33.40, 16.35]
text_reord = [39.70, 39.80, 38.88, 36.85, 33.40, 19.39]

imgtxt_orig = [70.05, 67.21, 65.18, 62.64, 36.75, 49.24]
imgtxt_reord = [70.46, 68.12, 65.18, 62.74, 39.09, 50.96]

# T4: Distractor Perturbations
text_base = [41.32, 40.10, 38.58, 37.97, 33.40, 16.35]
text_unk  = [45.69, 44.67, 40.71, 45.08, 28.63, 25.08]
text_1r   = [35.23, 41.22, 32.39, 38.98, 38.07, 18.27]
text_2r   = [33.20, 39.19, 30.56, 35.63, 28.43, 12.08]
text_3r   = [24.26, 27.31, 24.06, 28.53, 26.90, 9.24]
text_4r   = [21.02, 20.81, 17.87, 19.19, 19.59, 13.50]

imgtxt_base = [70.05, 67.21, 65.18, 62.64, 36.75, 49.24]
imgtxt_unk  = [71.88, 70.05, 67.31, 66.70, 40.81, 48.12]
imgtxt_1r   = [71.57, 70.46, 67.82, 63.96, 43.96, 54.92]
imgtxt_2r   = [74.62, 71.78, 72.28, 69.04, 52.39, 56.75]
imgtxt_3r   = [84.06, 79.49, 80.20, 78.17, 54.11, 67.92]
imgtxt_4r   = [90.15, 90.25, 90.96, 85.69, 77.46, 82.13]

# T5: Visual Substitution
imgtxt_orig_t5  = [84.00, 76.00, 83.50, 64.50, 59.00, 26.50]
imgtxt_subst_t5 = [53.00, 52.50, 50.50, 41.50, 27.50, 36.00]

# 95% CI data (from tables)
# T3: Format Perturbation CIs [lower, upper] for each model
t3_text_orig_ci    = [[39.74, 42.90], [38.76, 41.44], [37.10, 40.06], [34.56, 41.38], [31.71, 35.09], [14.71, 17.98]]
t3_text_reord_ci   = [[37.73, 41.66], [37.87, 41.72], [37.07, 40.70], [32.43, 41.28], [32.71, 34.09], [18.36, 20.43]]
t3_imgtxt_orig_ci  = [[68.64, 71.46], [65.39, 69.02], [64.22, 66.13], [59.29, 65.99], [34.29, 39.22], [47.10, 51.38]]
t3_imgtxt_reord_ci = [[68.21, 72.70], [67.09, 69.16], [64.22, 66.13], [61.10, 64.38], [37.14, 41.03], [49.74, 52.19]]

# T4: Distractor Perturbations CIs

t4_text_base_ci = [[39.74, 42.90], [38.76, 41.44], [37.10, 40.06], [34.56, 41.38], [31.71, 35.09], [14.71, 17.98]]
t4_text_unk_ci  = [[45.05, 46.32], [43.41, 45.93], [36.45, 44.97], [41.34, 48.82], [27.18, 30.08], [24.12, 26.03]]
t4_text_1r_ci   = [[33.41, 37.04], [39.02, 43.42], [30.05, 34.72], [36.88, 41.09], [36.89, 39.25], [17.50, 19.05]]
t4_text_2r_ci   = [[29.15, 37.25], [37.33, 41.05], [27.83, 33.28], [32.77, 38.50], [27.43, 29.42], [10.39, 13.77]]
t4_text_3r_ci   = [[21.43, 27.10], [24.93, 29.68], [20.89, 27.23], [25.70, 31.36], [25.36, 28.45], [7.61, 10.87]]
t4_text_4r_ci   = [[18.07, 23.96], [19.82, 21.81], [16.01, 19.73], [16.29, 22.09], [18.21, 20.97], [11.98, 15.02]]


t4_imgtxt_base_ci = [[68.64, 71.46], [65.39, 69.02], [64.22, 66.13], [59.29, 65.99], [34.29, 39.22], [47.10, 51.38]]
t4_imgtxt_unk_ci  = [[69.91, 73.85], [68.87, 71.23], [66.75, 67.87], [64.45, 68.96], [39.36, 42.26], [46.76, 49.49]]
t4_imgtxt_1r_ci   = [[70.16, 72.98], [68.30, 72.61], [65.56, 70.07], [61.48, 66.44], [42.90, 45.01], [52.72, 57.13]]
t4_imgtxt_2r_ci   = [[73.62, 75.62], [70.55, 73.01], [70.47, 74.10], [66.90, 71.17], [50.88, 53.89], [55.54, 57.96]]
t4_imgtxt_3r_ci   = [[82.92, 85.21], [78.44, 80.55], [78.94, 81.46], [76.23, 80.12], [52.80, 55.42], [66.98, 68.85]]
t4_imgtxt_4r_ci   = [[89.31, 91.00], [89.56, 90.94], [89.60, 92.33], [84.86, 86.51], [76.41, 78.52], [81.10, 83.17]]

# T5: Visual Substitution CIs
# Model order: ['GPT-5', 'Gemini-2.5 Pro', 'OpenAI-o3', 'OpenAI-o4-mini', 'Claude-sonnet-3.5', 'GPT-4o']

t5_imgtxt_orig_ci  = [[80.46, 87.54], [74.30, 77.70], [78.79, 88.21], [60.45, 68.55], [55.46, 62.54], [23.72, 29.28]]
t5_imgtxt_subst_ci = [[50.40, 55.60], [48.70, 56.30], [45.40, 55.60], [32.29, 50.71], [20.22, 34.78], [29.56, 42.44]]

# ── Per-run data for T3 panels a) and e) (5 runs × 6 models) ──────────────
# Model order: GPT-5, Gemini-2.5 Pro, OpenAI-o3, OpenAI-o4-mini, Claude-sonnet-3.5, GPT-4o
t3_imgtxt_orig_runs  = [
    np.array([68.53, 71.57, 70.56, 69.54, 70.05]),  # GPT-5
    np.array([65.99, 65.48, 68.02, 67.51, 69.04]),  # Gemini-2.5 Pro
    np.array([64.97, 64.47, 64.47, 65.99, 65.99]),  # OpenAI-o3
    np.array([59.39, 66.50, 63.45, 62.94, 60.91]),  # OpenAI-o4-mini
    np.array([39.59, 34.01, 37.06, 36.55, 36.55]),  # Claude-sonnet-3.5
    np.array([50.76, 51.27, 48.22, 47.21, 48.73]),  # GPT-4o
]
t3_text_orig_runs    = [
    np.array([43.15, 41.62, 39.59, 41.12, 41.12]),  # GPT-5
    np.array([41.62, 40.10, 40.61, 39.09, 39.09]),  # Gemini-2.5 Pro
    np.array([38.58, 39.09, 39.09, 39.59, 36.55]),  # OpenAI-o3
    np.array([36.55, 42.13, 37.06, 39.09, 35.03]),  # OpenAI-o4-mini
    np.array([33.50, 35.53, 32.49, 33.50, 31.98]),  # Claude-sonnet-3.5
    np.array([15.74, 16.24, 14.72, 18.27, 16.75]),  # GPT-4o
]
t3_imgtxt_reord_runs = [
    np.array([70.05, 70.05, 73.60, 69.54, 69.04]),  # GPT-5
    np.array([67.51, 68.02, 69.54, 67.51, 68.02]),  # Gemini-2.5 Pro
    np.array([66.50, 64.97, 64.97, 64.47, 64.97]),  # OpenAI-o3
    np.array([63.45, 62.44, 62.44, 60.91, 64.47]),  # OpenAI-o4-mini
    np.array([41.62, 39.09, 38.07, 39.09, 37.56]),  # Claude-sonnet-3.5
    np.array([49.75, 50.25, 51.27, 51.27, 52.28]),  # GPT-4o
]
t3_text_reord_runs = [
    np.array([39.09, 41.12, 41.62, 38.07, 38.58]),  # GPT-5
    np.array([40.61, 40.61, 37.06, 40.61, 40.10]),  # Gemini-2.5 Pro
    np.array([40.10, 40.10, 36.55, 38.58, 39.09]),  # OpenAI-o3
    np.array([36.55, 36.04, 40.10, 31.47, 40.10]),  # OpenAI-o4-mini
    np.array([32.49, 33.50, 34.01, 33.50, 33.50]),  # Claude-sonnet-3.5
    np.array([18.27, 19.80, 18.78, 19.80, 20.30]),  # GPT-4o
]

# ── Per-run data for T4a panels b) and f) ─────────────────────────────────
# Base runs reuse T3 original runs (same condition)
t4_imgtxt_base_runs = t3_imgtxt_orig_runs
t4_text_base_runs   = t3_text_orig_runs
# Unk runs (5 runs × 6 models), vqa → image+text, text → text only
t4_imgtxt_unk_runs = [
    np.array([73.60, 71.07, 70.56, 73.60, 70.56]),  # GPT-5
    np.array([68.53, 70.56, 70.05, 70.05, 71.07]),  # Gemini-2.5 Pro
    np.array([68.02, 67.01, 67.01, 67.51, 67.01]),  # OpenAI-o3
    np.array([65.99, 64.47, 65.99, 68.02, 69.04]),  # OpenAI-o4-mini
    np.array([40.61, 40.61, 39.09, 41.62, 42.13]),  # Claude-sonnet-3.5
    np.array([49.75, 48.22, 47.72, 48.22, 46.70]),  # GPT-4o
]
t4_text_unk_runs = [
    np.array([45.18, 46.19, 45.18, 45.69, 46.19]),  # GPT-5
    np.array([44.16, 45.18, 43.15, 45.69, 45.18]),  # Gemini-2.5 Pro
    np.array([35.53, 41.12, 45.18, 41.12, 40.61]),  # OpenAI-o3
    np.array([49.24, 45.69, 46.19, 42.13, 42.13]),  # OpenAI-o4-mini
    np.array([28.43, 28.43, 29.44, 26.90, 29.95]),  # Claude-sonnet-3.5
    np.array([25.89, 25.38, 23.86, 24.87, 25.38]),  # GPT-4o
]

# ── Per-run data for T5 panel d) (5 runs × 6 models, vqa/image+text only) ────────
# Model order: GPT-5, Gemini-2.5 Pro, OpenAI-o3, OpenAI-o4-mini, Claude-sonnet-3.5, GPT-4o
t5_imgtxt_orig_runs = [
    np.array([82.50, 80.00, 85.00, 85.00, 87.50]),  # GPT-5
    np.array([75.00, 75.00, 77.50, 77.50, 75.00]),  # Gemini-2.5 Pro
    np.array([82.50, 85.00, 87.50, 77.50, 85.00]),  # OpenAI-o3
    np.array([60.00, 65.00, 62.50, 67.50, 67.50]),  # OpenAI-o4-mini
    np.array([57.50, 60.00, 62.50, 55.00, 60.00]),  # Claude-sonnet-3.5
    np.array([25.00, 27.50, 25.00, 30.00, 25.00]),  # GPT-4o
]
t5_imgtxt_subst_runs = [
    np.array([52.50, 55.00, 50.00, 55.00, 52.50]),  # GPT-5
    np.array([52.50, 52.50, 55.00, 55.00, 50.00]),  # Gemini-2.5 Pro
    np.array([50.00, 57.50, 50.00, 47.50, 47.50]),  # OpenAI-o3
    np.array([50.00, 35.00, 45.00, 45.00, 32.50]),  # OpenAI-o4-mini
    np.array([25.00, 22.50, 32.50, 35.00, 22.50]),  # Claude-sonnet-3.5
    np.array([30.00, 32.50, 40.00, 42.50, 35.00]),  # GPT-4o
]


def get_ci_upper(value, ci_data, model_idx):
    """Get upper 95% CI from actual data"""
    return ci_data[model_idx][1]

def get_ci_lower(value, ci_data, model_idx):
    """Get lower 95% CI from actual data"""
    return ci_data[model_idx][0]

# Create figure with 2 rows × 4 columns
fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1], hspace=GRIDSPEC_HSPACE, wspace=GRIDSPEC_WSPACE, 
                      left=GRIDSPEC_LEFT, right=GRIDSPEC_RIGHT, 
                      top=GRIDSPEC_TOP, bottom=GRIDSPEC_BOTTOM)

# Create axes - each column is now a separate stress test
axs = []
for row in range(2):
    row_axes = []
    for col in range(4):
        ax = fig.add_subplot(gs[row, col])
        row_axes.append(ax)
    axs.append(row_axes)

# Panel letters (will be added at end using fig.text)
panel_letters = [['a)', 'b)', 'c)', 'd)'], ['e)', 'f)', 'g)']]

# Column headers (below second row, under x-axis tick labels)
col_headers = [
    'T3: Option order reordered',
    "T4a: 1 distractor replaced with ‘Unknown'",
    'T4b: 1–4 unrelated distractors',
    'T5: Visual substitution'
]

# Add column headers below second row
for col, header in enumerate(col_headers):
    pos = gs[1, col].get_position(fig)  # Second row (row index 1)
    fig.text(0.5 * (pos.x0 + pos.x1),
             pos.y0 - COL_HEADER_OFFSET,  # Below x-axis tick labels
             header, ha='center', va='top', fontsize=FONTSIZE_CAPTION, fontweight=FONTWEIGHT_REGULAR)

# Row labels (top of each row, left aligned)
row_labels = ['Image+Text', 'Text only']
for row, label in enumerate(row_labels):
    # Position at top of row, with margin from left edge
    pos = gs[row, 0].get_position(fig)
    label_x = ROW_LABEL_MARGIN  # Margin from left edge
    label_y = pos.y1 + ROW_LABEL_Y_OFFSET  # Top of row with offset
    fig.text(label_x, label_y, label, ha='left', va='bottom',
             fontsize=FONTSIZE_ROW_LABEL, fontweight=FONTWEIGHT_REGULAR, rotation=0, color='black')

# Panel A & E: T3 - Grouped bars (Original vs Reordered)
DOT_COLOR = '#555555'   # gray dot color
DOT_SIZE  = 0.5          # dot marker size
DOT_OFFSET  = -0.08     # dots: left of bar center
ERR_OFFSET  =  0.08     # error bar: right of bar center

for row_idx, (orig_data, reord_data, orig_ci, reord_ci, orig_runs, reord_runs, ax) in enumerate([
    (imgtxt_orig, imgtxt_reord, t3_imgtxt_orig_ci, t3_imgtxt_reord_ci,
     t3_imgtxt_orig_runs, t3_imgtxt_reord_runs, axs[0][0]),  # Image+Text (first row)
    (text_orig, text_reord, t3_text_orig_ci, t3_text_reord_ci,
     t3_text_orig_runs, t3_text_reord_runs, axs[1][0])  # Text-only (second row)
]):
    x = np.arange(len(models))
    width = 0.35
    x1 = x - width/2
    x2 = x + width/2
    box_w = width * 0.88

    # Original (solid color) boxplots
    bplot_orig = ax.boxplot(
        [orig_runs[i] for i in range(len(models))],
        positions=x1, widths=box_w, patch_artist=True,
        medianprops=dict(color='black', linewidth=0.5, zorder=5),
        whiskerprops=dict(color='#444444', linewidth=0.35),
        capprops=dict(color='#444444', linewidth=0.35),
        flierprops=dict(visible=False),
        boxprops=dict(linewidth=0.8),
        whis=(0, 100), zorder=2
    )
    for patch, color in zip(bplot_orig['boxes'], model_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor('#444444')
    # Overlay individual run dots
    for i in range(len(models)):
        ax.scatter(np.full(len(orig_runs[i]), x1[i]), orig_runs[i],
                   s=DOT_SIZE, color=DOT_COLOR, zorder=10)

    # Reordered (hatched) boxplots
    bplot_reord = ax.boxplot(
        [reord_runs[i] for i in range(len(models))],
        positions=x2, widths=box_w, patch_artist=True,
        medianprops=dict(color='black', linewidth=0.5, zorder=5),
        whiskerprops=dict(color='#444444', linewidth=0.35),
        capprops=dict(color='#444444', linewidth=0.35),
        flierprops=dict(visible=False),
        boxprops=dict(linewidth=LINEWIDTH_HATCH),
        whis=(0, 100), zorder=2
    )
    for patch, color in zip(bplot_reord['boxes'], model_colors):
        patch.set_facecolor(HATCH_FACECOLOR)
        patch.set_hatch(HATCH_PATTERN)
        patch.set_edgecolor(darken_color(color, HATCH_DARKEN_FACTOR))
    # Overlay individual run dots
    for i in range(len(models)):
        ax.scatter(np.full(len(reord_runs[i]), x2[i]), reord_runs[i],
                   s=DOT_SIZE, color=DOT_COLOR, zorder=10)
    
    ax.set_xticks(x)
    # Only second row (panel e) has x-axis tick labels
    if row_idx == 1:
        ax.set_xticklabels(models, fontsize=FONTSIZE_TICK_LABEL, rotation=30, ha='right')
    else:
        ax.set_xticklabels([])  # Hide tick labels for first row
    # Panels a and e (T3) have y-axis labels
    ax.set_ylabel('Accuracy (%)', fontsize=FONTSIZE_AXIS_LABEL, fontweight=FONTWEIGHT_REGULAR)
    # Set y-axis tick label font size
    ax.tick_params(axis='y', labelsize=FONTSIZE_TICK_LABEL)
    if row_idx == 0:
        ax.set_ylim(30, 80)
        ax.set_yticks([30, 40, 50, 60, 70, 80])
    else:
        ax.set_ylim(10, 50)
        ax.set_yticks([10, 20, 30, 40, 50])
    ax.grid(True, alpha=0.25, linewidth=LINEWIDTH_GRID, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.axhline(y=20, color='#666666', linestyle='--', alpha=0.9, linewidth=LINEWIDTH_CHANCE, zorder=0)
    

# Panel B & F: T4a - Grouped boxplots (Unperturbed vs None of the above)
for row_idx, (base_data, unk_data, base_ci, unk_ci, base_runs, unk_runs, ax) in enumerate([
    (imgtxt_base, imgtxt_unk, t4_imgtxt_base_ci, t4_imgtxt_unk_ci,
     t4_imgtxt_base_runs, t4_imgtxt_unk_runs, axs[0][1]),  # Image+Text (first row)
    (text_base, text_unk, t4_text_base_ci, t4_text_unk_ci,
     t4_text_base_runs, t4_text_unk_runs, axs[1][1])  # Text-only (second row)
]):
    x = np.arange(len(models))
    width = 0.35
    x1 = x - width/2
    x2 = x + width/2
    box_w = width * 0.88

    # Base/unperturbed (solid color) boxplots
    bplot_base = ax.boxplot(
        [base_runs[i] for i in range(len(models))],
        positions=x1, widths=box_w, patch_artist=True,
        medianprops=dict(color='black', linewidth=0.5, zorder=5),
        whiskerprops=dict(color='#444444', linewidth=0.35),
        capprops=dict(color='#444444', linewidth=0.35),
        flierprops=dict(visible=False),
        boxprops=dict(linewidth=0.8),
        whis=(0, 100), zorder=2
    )
    for patch, color in zip(bplot_base['boxes'], model_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor('#444444')
    # Overlay individual run dots
    for i in range(len(models)):
        ax.scatter(np.full(len(base_runs[i]), x1[i]), base_runs[i],
                   s=DOT_SIZE, color=DOT_COLOR, zorder=10)

    # Unknown/perturbed (hatched) boxplots
    bplot_unk = ax.boxplot(
        [unk_runs[i] for i in range(len(models))],
        positions=x2, widths=box_w, patch_artist=True,
        medianprops=dict(color='black', linewidth=0.5, zorder=5),
        whiskerprops=dict(color='#444444', linewidth=0.35),
        capprops=dict(color='#444444', linewidth=0.35),
        flierprops=dict(visible=False),
        boxprops=dict(linewidth=LINEWIDTH_HATCH),
        whis=(0, 100), zorder=2
    )
    for patch, color in zip(bplot_unk['boxes'], model_colors):
        patch.set_facecolor(HATCH_FACECOLOR)
        patch.set_hatch(HATCH_PATTERN)
        patch.set_edgecolor(darken_color(color, HATCH_DARKEN_FACTOR))
    # Overlay individual run dots
    for i in range(len(models)):
        ax.scatter(np.full(len(unk_runs[i]), x2[i]), unk_runs[i],
                   s=DOT_SIZE, color=DOT_COLOR, zorder=10)
    
    ax.set_xticks(x)
    # Only second row (panel f) has x-axis tick labels
    if row_idx == 1:
        ax.set_xticklabels(models, fontsize=FONTSIZE_TICK_LABEL, rotation=30, ha='right')
    else:
        ax.set_xticklabels([])  # Hide tick labels for first row
    # Set y-axis tick label font size
    ax.tick_params(axis='y', labelsize=FONTSIZE_TICK_LABEL)
    if row_idx == 0:
        ax.set_ylim(30, 80)
        ax.set_yticks([30, 40, 50, 60, 70, 80])
    else:
        ax.set_ylim(10, 50)
        ax.set_yticks([10, 20, 30, 40, 50])
    ax.grid(True, alpha=0.25, linewidth=LINEWIDTH_GRID, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.axhline(y=20, color='#666666', linestyle='--', alpha=0.9, linewidth=LINEWIDTH_CHANCE, zorder=0)
    

# Panel C & G: T4b - Lines with markers (0=Base, 1R-4R)
for row_idx, (base_data, r1_data, r2_data, r3_data, r4_data, 
              base_ci, r1_ci, r2_ci, r3_ci, r4_ci, ax) in enumerate([
    (imgtxt_base, imgtxt_1r, imgtxt_2r, imgtxt_3r, imgtxt_4r,
     t4_imgtxt_base_ci, t4_imgtxt_1r_ci, t4_imgtxt_2r_ci, t4_imgtxt_3r_ci, t4_imgtxt_4r_ci, axs[0][2]),  # Image+Text (first row)
    (text_base, text_1r, text_2r, text_3r, text_4r,
     t4_text_base_ci, t4_text_1r_ci, t4_text_2r_ci, t4_text_3r_ci, t4_text_4r_ci, axs[1][2])  # Text-only (second row)
]):
    # X positions: 0=Base, 1R=1, 2R=2, 3R=3, 4R=4
    x_positions = np.array([0, 1, 2, 3, 4])
    x_labels = ['Original', '1 replaced', '2 replaced', '3 replaced', '4 replaced']
    # make
    # Plot lines for each model
    for i, model in enumerate(models):
        values = [base_data[i], r1_data[i], r2_data[i], r3_data[i], r4_data[i]]
        
        # Calculate CI for error bars using actual CI data
        ci_data_list = [base_ci, r1_ci, r2_ci, r3_ci, r4_ci]
        ci_upper = [get_ci_upper(values[j], ci_data_list[j], i) for j in range(5)]
        ci_lower = [get_ci_lower(values[j], ci_data_list[j], i) for j in range(5)]
        yerr_upper = [u - v for u, v in zip(ci_upper, values)]
        yerr_lower = [v - l for l, v in zip(ci_lower, values)]
        
        # All lines solid, Original point filled circle, 1-4 open circles
        # Plot solid line connecting all points
        ax.plot(x_positions, values, '-', color=model_colors[i],
               linewidth=LINEWIDTH_PLOT, alpha=1.0, zorder=2)
        
        # Plot Original point (x=0) as filled circle
        ax.plot(x_positions[0], values[0], 'o', color=model_colors[i],
               markersize=MARKERSIZE_PLOT, alpha=1.0, zorder=3, fillstyle='full')
        
        # Plot 1-4 points (x=1,2,3,4) as filled squares
        ax.plot(x_positions[1:], values[1:], 's', color=model_colors[i],
               markersize=MARKERSIZE_PLOT, alpha=1.0, zorder=3, fillstyle='full')
        
        # # Error bars
        # ax.errorbar(x_positions, values, yerr=[yerr_lower, yerr_upper],
        #            fmt='none', color=model_colors[i], alpha=0.6, capsize=CAPSIZE_ERRORBAR,
        #            capthick=LINEWIDTH_ERRORBAR, linewidth=LINEWIDTH_ERRORBAR, zorder=1)
    
    ax.set_xticks(x_positions)
    # Only second row (panel g) has x-axis tick labels
    if row_idx == 1:
        ax.set_xticklabels(['0', '1', '2', '3', '4'], fontsize=FONTSIZE_TICK_LABEL, rotation=0, ha='center')
        ax.set_xlabel('# distractor(s) replaced', fontsize=FONTSIZE_AXIS_LABEL, fontweight=FONTWEIGHT_REGULAR)
    else:
        ax.set_xticklabels([])  # Hide tick labels for first row
    # Set y-axis tick label font size
    ax.tick_params(axis='y', labelsize=FONTSIZE_TICK_LABEL)
    ax.set_xlim(-0.3, 4.3)  # Padding to accommodate 0-4 range
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(True, alpha=0.25, linewidth=LINEWIDTH_GRID, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=20, color='#666666', linestyle='--', alpha=0.9, linewidth=LINEWIDTH_CHANCE, zorder=0)

# Panel D: T5 - Grouped boxplots (Original vs Substituted) - Image+Text only
ax = axs[0][3]  # Image+Text (first row only)

x = np.arange(len(models))
width = 0.35
x1 = x - width/2
x2 = x + width/2
box_w = width * 0.88

# Original (solid color) boxplots
bplot_t5_orig = ax.boxplot(
    [t5_imgtxt_orig_runs[i] for i in range(len(models))],
    positions=x1, widths=box_w, patch_artist=True,
    medianprops=dict(color='black', linewidth=0.5, zorder=5),
    whiskerprops=dict(color='#444444', linewidth=0.35),
    capprops=dict(color='#444444', linewidth=0.35),
    flierprops=dict(visible=False),
    boxprops=dict(linewidth=0.8),
    whis=(0, 100), zorder=2
)
for patch, color in zip(bplot_t5_orig['boxes'], model_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
    patch.set_edgecolor('#444444')
for i in range(len(models)):
    ax.scatter(np.full(len(t5_imgtxt_orig_runs[i]), x1[i]), t5_imgtxt_orig_runs[i],
               s=DOT_SIZE, color=DOT_COLOR, zorder=10)

# Substituted (hatched) boxplots
bplot_t5_subst = ax.boxplot(
    [t5_imgtxt_subst_runs[i] for i in range(len(models))],
    positions=x2, widths=box_w, patch_artist=True,
    medianprops=dict(color='black', linewidth=0.5, zorder=5),
    whiskerprops=dict(color='#444444', linewidth=0.35),
    capprops=dict(color='#444444', linewidth=0.35),
    flierprops=dict(visible=False),
    boxprops=dict(linewidth=LINEWIDTH_HATCH),
    whis=(0, 100), zorder=2
)
for patch, color in zip(bplot_t5_subst['boxes'], model_colors):
    patch.set_facecolor(HATCH_FACECOLOR)
    patch.set_hatch(HATCH_PATTERN)
    patch.set_edgecolor(darken_color(color, HATCH_DARKEN_FACTOR))
for i in range(len(models)):
    ax.scatter(np.full(len(t5_imgtxt_subst_runs[i]), x2[i]), t5_imgtxt_subst_runs[i],
               s=DOT_SIZE, color=DOT_COLOR, zorder=10)

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=FONTSIZE_TICK_LABEL, rotation=30, ha='right')
# Set y-axis tick label font size
ax.tick_params(axis='y', labelsize=FONTSIZE_TICK_LABEL)
ax.set_ylim(20, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.grid(True, alpha=0.25, linewidth=0.4, axis='y')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.axhline(y=20, color='#666666', linestyle='--', alpha=0.9, linewidth=1.5, zorder=0)

# ── Significance brackets above each orig/subst pair in panel D ──────────
# Stars for each model pair (update as needed based on statistical test)
t5_sig_labels = ['*', '*', '*', '*', '*', '*']
BRACKET_PAD  = 3.5   # gap above data top to bracket line (data units)
BRACKET_TICK = 1.5   # length of the downward tick at each end (data units)
BRACKET_LW   = 0.7   # bracket line width
BRACKET_FS   = 5.5   # font size for stars

for i in range(len(models)):
    orig_top  = max(t5_imgtxt_orig_runs[i])
    subst_top = max(t5_imgtxt_subst_runs[i])
    y_bracket = max(orig_top, subst_top) + BRACKET_PAD
    # clip so bracket stays within y-axis
    y_bracket = min(y_bracket, ax.get_ylim()[1] - BRACKET_TICK - 1)
    lx, rx = x1[i], x2[i]
    # horizontal bar
    ax.plot([lx, rx], [y_bracket, y_bracket],
            color='black', linewidth=BRACKET_LW, clip_on=False, zorder=8)
    # left tick (down)
    ax.plot([lx, lx], [y_bracket - BRACKET_TICK, y_bracket],
            color='black', linewidth=BRACKET_LW, clip_on=False, zorder=8)
    # right tick (down)
    ax.plot([rx, rx], [y_bracket - BRACKET_TICK, y_bracket],
            color='black', linewidth=BRACKET_LW, clip_on=False, zorder=8)
    # significance text
    # ax.text((lx + rx) / 2, y_bracket + 0.3, t5_sig_labels[i],
    #         ha='center', va='bottom', fontsize=BRACKET_FS, color='black', zorder=9)

# Hide all axes for panel h (empty T5 Text-only panel)
ax_h = axs[1][3]
ax_h.set_xticks([])
ax_h.set_yticks([])
ax_h.spines['top'].set_visible(False)
ax_h.spines['right'].set_visible(False)
ax_h.spines['bottom'].set_visible(False)
ax_h.spines['left'].set_visible(False)
ax_h.axis('off')

# Add panel labels at top-left of each panel (like figure 2)
# Get positions for all panels
pos_a = axs[0][0].get_position()  # T3 Image+Text (first row)
pos_b = axs[0][1].get_position()  # T4a Image+Text (first row)
pos_c = axs[0][2].get_position()  # T4b Image+Text (first row)
pos_d = axs[0][3].get_position()  # T5 Image+Text (first row)
pos_e = axs[1][0].get_position()  # T3 Text-only (second row)
pos_f = axs[1][1].get_position()  # T4a Text-only (second row)
pos_g = axs[1][2].get_position()  # T4b Text-only (second row)
pos_h = axs[1][3].get_position()  # Empty panel below panel D

# Create unified legend below panel D: bars and lines
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
chance_color = '#666666'  # Darker neutral gray
legend_elements = [
    Patch(facecolor='gray', alpha=0.8, label='Original'),  # Solid bar
    Patch(facecolor=HATCH_FACECOLOR, edgecolor=darken_color('gray', HATCH_DARKEN_FACTOR), 
          linewidth=LINEWIDTH_HATCH, hatch=HATCH_PATTERN, label='Perturbed'),  # Hatched bar
    Line2D([0], [0], color='gray', linestyle='-', linewidth=LINEWIDTH_LEGEND, marker='o', 
           markersize=MARKERSIZE_LEGEND, fillstyle='full', label='Original'),  # Solid line with filled circle
    Line2D([0], [0], color='gray', linestyle='-', linewidth=LINEWIDTH_LEGEND, marker='s',
           markersize=MARKERSIZE_LEGEND, fillstyle='full', label='Perturbed'),  # Solid line with filled square
    Line2D([0], [0], color=chance_color, linestyle='--', linewidth=LINEWIDTH_CHANCE, alpha=0.9, label='Chance level performance (20%)')  # Chance level line
]

# Position legends below panel captions in two rows
# Get the position of the captions (below second row)
caption_y = gs[1, 0].get_position(fig).y0 - COL_HEADER_OFFSET  # Position of captions

# First row: unified legend (arranged horizontally)
unified_legend_y = caption_y - LEGEND_FIRST_OFFSET  # Below captions
fig.legend(legend_elements, ['Original', 'Perturbed', 'Original', 'Perturbed', 'Chance level performance (20%)'],
           bbox_to_anchor=(0.5, unified_legend_y), loc='center',
           ncol=5, frameon=False, fontsize=FONTSIZE_LEGEND)

# Create model legend elements for T4b (solid line with model colors)
model_legend_elements = [
    Line2D([0], [0], color=model_colors[i], linestyle='-', linewidth=LINEWIDTH_LEGEND, 
           label=model)
    for i, model in enumerate(models)
]

# Second row: model legend (arranged horizontally)
model_legend_y = caption_y - LEGEND_FIRST_OFFSET - LEGEND_SPACING  # Below first legend row
fig.legend(model_legend_elements, models,
           bbox_to_anchor=(0.5, model_legend_y), loc='center',
           ncol=6, frameon=False, fontsize=FONTSIZE_LEGEND, handlelength=1.5)

# Add panel labels to the left of '100' y-axis tick
# Get positions for all panels
pos_a = axs[0][0].get_position()
pos_b = axs[0][1].get_position()
pos_c = axs[0][2].get_position()
pos_d = axs[0][3].get_position()
pos_e = axs[1][0].get_position()
pos_f = axs[1][1].get_position()
pos_g = axs[1][2].get_position()

# Position panel labels at top-left of each panel
for ax, label, pos in [
    (axs[0][0], 'a)', pos_a),
    (axs[0][1], 'b)', pos_b),
    (axs[0][2], 'c)', pos_c),
    (axs[0][3], 'd)', pos_d),
    (axs[1][0], 'e)', pos_e),
    (axs[1][1], 'f)', pos_f),
    (axs[1][2], 'g)', pos_g)
]:
    fig.text(pos.x0 - PANEL_LABEL_X_OFFSET, pos.y1 + PANEL_LABEL_Y_OFFSET, label,
             fontsize=FONTSIZE_PANEL_LABEL, fontweight=FONTWEIGHT_PANEL_LABEL,
             va='bottom', ha='right')

plt.savefig('output/figure3_perturbation_v1.pdf', dpi=600, bbox_inches='tight')
plt.show()


# %%
