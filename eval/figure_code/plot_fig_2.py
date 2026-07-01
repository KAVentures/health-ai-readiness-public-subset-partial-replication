#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from scipy.stats import pearsonr

# Data for Stress Test 1 (modality sensitivity) and delta
models = ['GPT-5', 'Gemini-2.5 Pro', 'OpenAI-o3', 'OpenAI-o4-mini', 'Claude-sonnet-3.5','GPT-4o']

# Palette tuned for Nature Medicine styling (model hues fixed globally)
palette = {
    'nejm_text': '#5E6C7B',
    'jama_text': '#B7A48C',
    'image_alpha': 0.15,
    'chance_line': '#5C5C5C',
    'shortcut_zone': '#D6CBA5',
    'stack': {
        'refused': '#7FBF9E',
        'incorrect': '#D8D8D8',
        'correct': '#E8DF93'
    },
    'model_bars': ['#4A7D90', '#C77D4A', '#7A9E8F', '#D4A574', '#8B7A9E', '#6A4A7A']
}

nejm_full = np.array([81.33, 81.12, 80.42, 76.62, 59.77, 67.16])
nejm_text = np.array([67.41, 67.43, 66.65, 66.48, 61.41, 54.23])
nejm_delta = nejm_full - nejm_text

jama_full = np.array([86.93, 84.19, 86.13, 80.53, 76.04, 71.51])
jama_text = np.array([82.51, 80.91, 82.79, 77.47, 76.79, 67.46])
jama_delta = jama_full - jama_text

# Data for Stress Test 2 (text-only on image-required NEJM questions)
stress2_text_only = np.array([41.32, 40.10, 37.26, 37.97, 33.40, 16.35])  # All 6 models
chance_level = 20.0

# Get abstention rates for circle sizing (same order as models)
# todo: These should match the actual abstention rate data
abstention_rates = np.array([0, 0, 0, 0, 0.91, 50.36])  # GPT-5, Gemini-2.5 Pro, OpenAI-o3, OpenAI-o4-mini, GPT-4o
guess_correct_rates = np.array([41.32, 40.10, 37.26, 37.97, 33.40, 16.35])

# Set publication-quality style
plt.style.use('default')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 13
plt.rcParams['axes.linewidth'] = 0.6
plt.rcParams['xtick.major.width'] = 0.6
plt.rcParams['ytick.major.width'] = 0.6
plt.rcParams['xtick.major.size'] = 3
plt.rcParams['ytick.major.size'] = 3

# ============================================================================
# Typography and styling configuration
# ============================================================================
# Font sizes (in points) following Nature Medicine guidelines
FONTSIZE_PANEL_LABEL = 10      # Panel labels (a), b), c)) - bold
FONTSIZE_AXIS_LABEL = 9        # Axis labels (e.g., "Accuracy (%)") - regular
FONTSIZE_TICK_LABEL = 7.5        # Tick labels - regular (7.5-8 pt range)
FONTSIZE_BAR_ANNOTATION = 6.2    # Bar value annotations - regular
FONTSIZE_LEGEND = 7          # Legend text - regular (7-8 pt range)
FONTSIZE_CAPTION = 7           # Panel captions - regular (never bold)

# Font weights
FONTWEIGHT_PANEL_LABEL = 'bold'
FONTWEIGHT_REGULAR = 'normal'

# Figure dimensions
FIG_WIDTH = 8                  # inches
FIG_HEIGHT = 3.5               # inches

# Gridspec configuration
PANEL_WIDTH_RATIOS = [2, 1, 1]  # Panel A, B, C relative widths
GRIDSPEC_HSPACE = 0.3
GRIDSPEC_WSPACE = 0.40
GRIDSPEC_LEFT = 0.12
GRIDSPEC_RIGHT = 0.96
GRIDSPEC_TOP = 0.88
GRIDSPEC_BOTTOM = 0.28

# Legend positioning
LEGEND_ANCHOR_Y = -0.25        # Negative value drops legend below axes

# Panel caption positioning
CAPTION_OFFSET = -0.50         # Offset below axes for captions

# Panel label positioning
PANEL_LABEL_X_OFFSET = -0.2   # Offset from left edge
PANEL_LABEL_Y_OFFSET = 0.98    # Offset above panel top

# ============================================================================
# Plotting with improved visualizations - 1x3 layout for 3 panels (A, B, C)
# ============================================================================
# Nature Medicine figure: optimized panel sizes for better proportions
# 7 inch width with appropriate height for panels, legends, and captions
fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
gs = fig.add_gridspec(1, 3, width_ratios=PANEL_WIDTH_RATIOS, 
                      hspace=GRIDSPEC_HSPACE, wspace=GRIDSPEC_WSPACE,
                      left=GRIDSPEC_LEFT, right=GRIDSPEC_RIGHT, 
                      top=GRIDSPEC_TOP, bottom=GRIDSPEC_BOTTOM)
axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), 
       fig.add_subplot(gs[0, 2])]

# Column headers will be placed below legends (defined later)
col_headers = [
    'T1: Modality sensitivity',
    'T2a: Accuracy on NEJM-VS',
    'T2b: Response types on NEJM-VS'
]

# Panel A: Waterfall chart showing modality contributions
x = np.arange(len(models))

# Nature-style color scheme: professional, muted, colorblind-friendly
# Distinct from Panel B/D colors to avoid visual confusion
# Using more muted, neutral tones that differ from model colors
nejm_color = palette['nejm_text']  # Blue-gray distinct from GPT-5
jama_color = palette['jama_text']  # Dusty tan distinct from other fills


# Create waterfall chart for NEJM with clear visual hierarchy
# Bottom: Text-only performance (starts from 0)
nejm_text_adjusted = nejm_text  # Full text-only values
bar_width = 0.375 # width of the bars in the waterfall chart
nejm_text_bars = axs[0].bar(x - bar_width/2, nejm_text_adjusted, bottom=0, color=nejm_color, alpha=0.9, width=bar_width, label='Text only (NEJM)', zorder=2, edgecolor=nejm_color, linewidth=0.8)
# Top: Performance drop when images removed - Panel A
nejm_visual_bars = axs[0].bar(x - bar_width/2 - 0.01, nejm_delta, bottom=nejm_text, color=nejm_color, alpha=palette['image_alpha'], width=bar_width-0.02, label='Image contribution (NEJM)', zorder=3, edgecolor=nejm_color, linewidth=0.8)
# Separate edge and fill alpha - edge fully opaque, fill with alpha 0.2
for bar in nejm_visual_bars:
    bar.set_alpha(None)  # Reset alpha inheritance
    # Convert nejm_color to RGBA with alpha set for subtle overlays
    fill_rgba = mcolors.to_rgba(nejm_color, alpha=palette['image_alpha'])
    bar.set_facecolor(fill_rgba)  # RGBA for fill (alpha 0.2)
    # Convert nejm_color to RGBA with alpha 1.0 for edge (same as below bar)
    edge_rgba = mcolors.to_rgba(nejm_color, alpha=1.0)
    bar.set_edgecolor(edge_rgba)  # Opaque edge, same color as below bar
    bar.set_linewidth(0.8)
    bar.set_zorder(3)

# Create waterfall chart for JAMA with clear visual hierarchy
# Bottom: Text-only performance (starts from 0)
jama_text_adjusted = jama_text  # Full text-only values
jama_text_bars = axs[0].bar(x + bar_width/2, jama_text_adjusted, bottom=0, color=jama_color, alpha=0.9, width=bar_width, label='Text only (JAMA)', zorder=2, edgecolor=jama_color, linewidth=0.8)
# Top: Performance drop when images removed - Panel A
jama_visual_bars = axs[0].bar(x + bar_width/2, jama_delta, bottom=jama_text, color=jama_color, alpha=palette['image_alpha'], width=bar_width, label='Image contribution (JAMA)', zorder=3, edgecolor=jama_color, linewidth=0.8)
# Separate edge and fill alpha - edge fully opaque, fill with alpha 0.2
for bar in jama_visual_bars:
    bar.set_alpha(None)  # Reset alpha inheritance
    # Convert jama_color to RGBA with alpha set for subtle overlays
    fill_rgba = mcolors.to_rgba(jama_color, alpha=palette['image_alpha'])
    bar.set_facecolor(fill_rgba)  # RGBA for fill (alpha 0.2)
    # Convert jama_color to RGBA with alpha 1.0 for edge (same as below bar)
    edge_rgba = mcolors.to_rgba(jama_color, alpha=1.0)
    bar.set_edgecolor(edge_rgba)  # Opaque edge, same color as below bar
    bar.set_linewidth(0.8)
    bar.set_zorder(3)

# Add value labels on bars with clear hierarchy and drop indicators
for i, (nejm_t, nejm_d, nejm_f, jama_t, jama_d, jama_f) in enumerate(zip(nejm_text, nejm_delta, nejm_full, jama_text, jama_delta, jama_full)):
    # NEJM labels
    # Text-only performance (bottom) - positioned at the center of the text bar
    if nejm_t >= 3:
        axs[0].text(i - bar_width/2, nejm_text_adjusted[i]/2, f'{nejm_t:.1f}', ha='center', va='center', fontsize=FONTSIZE_BAR_ANNOTATION, fontweight=FONTWEIGHT_REGULAR, color='white')
    # Visual contribution (top) - show as drop when images removed
    if nejm_d >= 3:
        axs[0].text(i - bar_width/2, nejm_t + nejm_d/2, f'{int(nejm_d)}pp', ha='center', va='center', fontsize=FONTSIZE_BAR_ANNOTATION, fontweight=FONTWEIGHT_REGULAR, color='black')
    # Total performance (top of bar)
    if nejm_f >= 3:
        axs[0].text(i - bar_width/2, nejm_f + 1, f'{nejm_f:.1f}', ha='center', va='bottom', fontsize=FONTSIZE_BAR_ANNOTATION, fontweight=FONTWEIGHT_REGULAR, color=nejm_color)
    
    # JAMA labels
    # Text-only performance (bottom) - positioned at the center of the text bar
    if jama_t >= 3:
        axs[0].text(i + bar_width/2, jama_text_adjusted[i]/2, f'{jama_t:.1f}', ha='center', va='center', fontsize=FONTSIZE_BAR_ANNOTATION, fontweight=FONTWEIGHT_REGULAR, color='white')
    # Visual contribution (top) - show as drop when images removed
    if jama_d >= 3:
        axs[0].text(i + bar_width/2, jama_t + jama_d/2, f'{int(jama_d)}pp', ha='center', va='center', fontsize=FONTSIZE_BAR_ANNOTATION, fontweight=FONTWEIGHT_REGULAR, color='black')
    # Total performance (top of bar)
    if jama_f >= 3:
        axs[0].text(i + bar_width/2, jama_f + 1, f'{jama_f:.1f}', ha='center', va='bottom', fontsize=FONTSIZE_BAR_ANNOTATION, fontweight=FONTWEIGHT_REGULAR, color=jama_color)

# Set x-axis with Figure 1 styling
axs[0].set_xticks(x)
axs[0].set_xticklabels(models, rotation=30, ha='right', fontsize=FONTSIZE_TICK_LABEL, fontweight=FONTWEIGHT_REGULAR)
# Set x-axis limits with equal padding to ensure first and last bars are fully visible
# Bars extend from x-bar_width to x+bar_width, with width 0.375, so we need padding beyond -0.375 and 5.375
axs[0].set_xlim(-0.5, 5.5)
axs[0].set_ylabel('Accuracy (%)', fontsize=FONTSIZE_AXIS_LABEL, fontweight=FONTWEIGHT_REGULAR)
# Set y-axis tick label font size
axs[0].tick_params(axis='y', labelsize=FONTSIZE_TICK_LABEL)
# Legend will be positioned below x tick labels (done later)
handles_a, labels_a = axs[0].get_legend_handles_labels()

# axs[0].legend(fontsize=11, frameon=False, loc='upper right')
axs[0].grid(axis='y', linestyle='--', alpha=0.25, linewidth=0.4)
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)

# Panel label (a) will be positioned after layout adjustments

# Highlight GPT-5 (first model) to align with Figure 1
# Removed bold formatting

# Panel C: Horizontal stacked bars showing response categories on visual required subset
x2 = np.arange(len(models)) * 0.98  # Reduced spacing between bars
models_stress2 = models  # Use all 5 models
models_stress2 = models_stress2[::-1]  # reverse order to make GPT-5 first (on top)

# Data for stacked bars: refused to answer, answered incorrectly, answered correctly
# Order: GPT-5, Gemini-2.5 Pro, OpenAI-o3, OpenAI-o4-mini, GPT-4o
refused = abstention_rates  # Refused to answer
correct_wo_image = guess_correct_rates  # Answered correctly w/o image
incorrect_wo_image = 100.0 - abstention_rates - guess_correct_rates  # Answered incorrectly w/o image

# Reverse to match Panel C order (GPT-4o first, GPT-5 last)
refused_reversed = refused[::-1]
correct_reversed = correct_wo_image[::-1]
incorrect_reversed = incorrect_wo_image[::-1]

# Colors for stacked bars - grayscale with clear patterns
stacked_colors = palette['stack']  # Encodes desired/neutral/undesired behavior

# Create horizontal stacked bars with hatching and separators
bar_height = 0.7  # Increased thickness
bottom = np.zeros(len(models))

# First segment: refused to answer - grayscale with black outline
bars_refused = axs[2].barh(x2, refused_reversed, left=bottom, 
                           color=stacked_colors['refused'], alpha=0.9, height=bar_height,
                           edgecolor='black', linewidth=0.8,
                           label='Refused (desired)')
bottom += refused_reversed

# Second segment: answered incorrectly (with diagonal hatching) - grayscale with black outline
bars_incorrect = axs[2].barh(x2, incorrect_reversed, left=bottom,
                              color=stacked_colors['incorrect'], alpha=0.9, height=bar_height,
                              hatch='\\', edgecolor='black', linewidth=0.8,
                              label='Incorrect (neutral)')
bottom += incorrect_reversed

# Third segment: answered correctly (with dotted hatching) - grayscale with black outline
bars_correct = axs[2].barh(x2, correct_reversed, left=bottom,
                            color=stacked_colors['correct'], alpha=0.9, height=bar_height,
                            hatch='//', edgecolor='black', linewidth=0.8,
                            label='Correct (undesired)')

# Add white separator lines between segments (vertical lines at boundaries)
for i in range(len(models)):
    y_pos = x2[i]
    y_top = y_pos + bar_height/2
    y_bottom = y_pos - bar_height/2
    
    # Separator between refused and incorrect
    if refused_reversed[i] > 0:
        sep_x = refused_reversed[i]
        axs[2].plot([sep_x, sep_x], [y_bottom, y_top],
                    color='white', linewidth=2, zorder=10, clip_on=False)
    
    # Separator between incorrect and correct
    if incorrect_reversed[i] > 0:
        sep_x = refused_reversed[i] + incorrect_reversed[i]
        axs[2].plot([sep_x, sep_x], [y_bottom, y_top],
                    color='white', linewidth=2, zorder=10, clip_on=False)

# Set y-axis labels (model names)
axs[2].set_yticks(x2)
# Right-align the y-axis labels so they don't overlap with bars
axs[2].set_yticklabels(models_stress2, fontsize=FONTSIZE_TICK_LABEL, fontweight=FONTWEIGHT_REGULAR)
# Move y-axis tick labels further to the left (increase padding) and ensure right alignment
axs[2].tick_params(axis='y', labelsize=FONTSIZE_TICK_LABEL, pad=8)
axs[2].set_xlabel('Percentage of responses (%)', fontsize=FONTSIZE_AXIS_LABEL, fontweight=FONTWEIGHT_REGULAR)
# Set x-axis tick label font size
axs[2].tick_params(axis='x', labelsize=FONTSIZE_TICK_LABEL)

    
# Legend will be positioned below x tick labels (done later)
legend_handles_c = [
    patches.Patch(facecolor=stacked_colors['refused'], edgecolor='black', linewidth=0.8,
                  label='Refused (desired)'),
    patches.Patch(facecolor=stacked_colors['incorrect'], edgecolor='black', linewidth=0.8,
                  hatch='\\', label='Incorrect (neutral)'),
    patches.Patch(facecolor=stacked_colors['correct'], edgecolor='black', linewidth=0.8,
                  hatch='//', label='Correct (undesired)')
]
axs[2].grid(axis='x', linestyle='--', alpha=0.25, linewidth=0.4)
axs[2].spines['top'].set_visible(False)
axs[2].spines['right'].set_visible(False)
axs[2].set_xlim(0, 100)

# Panel label (c) will be positioned after layout adjustments

# Set consistent y-axis limits for Panel A
axs[0].set_ylim(0, 100)


# Panel B: Vertical bar chart with CI bars showing accuracy without image
x3 = np.arange(len(models))

# Confidence intervals for stress test 2 (text-only accuracy)
# Order: GPT-5, Gemini-2.5 Pro, OpenAI-o3, OpenAI-o4-mini, Claude-sonnet-3.5, GPT-4o
ci_lower = np.array([39.74, 38.76, 35.05, 34.56, 31.71, 14.71])
ci_upper = np.array([42.90, 41.44, 39.47, 41.38, 35.09, 17.98])

cdot_gpt5 = np.array([43.15, 41.62, 39.59, 41.12, 41.12]) 
cdot_gemini = np.array([41.62, 40.10, 40.61, 39.09, 39.09]) 
cdot_openai_o3 = np.array([38.58, 39.09, 39.09, 39.59, 36.55]) 
cdot_openai_o4_mini = np.array([36.55, 42.13, 37.06, 39.09, 35.03]) 
cdot_claude_sonnet_3_5 = np.array([33.50, 35.53, 32.49, 33.50, 31.98]) 
cdot_openai_4o = np.array([15.74, 16.24, 14.72, 18.27, 16.75]) 
# Collect all per-run dot arrays in model order
cdot_all = [cdot_gpt5, cdot_gemini, cdot_openai_o3, cdot_openai_o4_mini, cdot_claude_sonnet_3_5, cdot_openai_4o]

 # Text-only accuracy for stress test 2 (NEJM-VS)

# Model colors matching figure 3
model_colors = palette['model_bars']  # Keep model hues consistent across figures

# Box-and-whisker plot: one box per model, data = 5 per-run values (cdot_all)
bplot = axs[1].boxplot(
    cdot_all,
    positions=x3,
    widths=0.55,
    patch_artist=True,
    medianprops=dict(color='black', linewidth=0.5, zorder=5),
    whiskerprops=dict(color='#444444', linewidth=0.5),
    capprops=dict(color='#444444', linewidth=0.5),
    flierprops=dict(marker='o', markersize=2, markerfacecolor='#555555',
                    markeredgecolor='none', linestyle='none'),
    boxprops=dict(linewidth=0.8),
    whis=(0, 100),   # whiskers span full data range (min–max) given n=5
    zorder=2
)
# Color each box with its model color
for patch, color in zip(bplot['boxes'], model_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)
    patch.set_edgecolor('#444444')

# Overlay individual run dots (jittered slightly for visibility)
for i, runs in enumerate(cdot_all):
    axs[1].scatter(np.full(len(runs), x3[i]), runs,
                   s=2, color='#333333', zorder=10, linewidths=0)

# Add chance level horizontal dashed line and shaded region
chance_color = palette['chance_line']  # Dark neutral reference line
axs[1].axhline(chance_level, linestyle='--', color=chance_color, linewidth=1.5, alpha=0.9, label='chance-level performance (20%)')
# Add shaded region from 20 to 60 for potential shortcut reliance zone
axs[1].axhspan(20, 60, alpha=0.25, color=palette['shortcut_zone'], label='Potential shortcut zone')

# Set x-axis labels (model names)
axs[1].set_xticks(x3)
axs[1].set_xticklabels(models, fontsize=FONTSIZE_TICK_LABEL, fontweight=FONTWEIGHT_REGULAR, rotation=30, ha='right')
axs[1].set_ylabel('Accuracy (%)', fontsize=FONTSIZE_AXIS_LABEL, fontweight=FONTWEIGHT_REGULAR)
# Set y-axis tick label font size
axs[1].tick_params(axis='y', labelsize=FONTSIZE_TICK_LABEL)

# Create custom legend with gray outline for zone
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=chance_color, linestyle='--', linewidth=1.5, alpha=0.9, label='chance-level performance (20%)'),
    Patch(facecolor=palette['shortcut_zone'], alpha=0.25, edgecolor='#999999', linewidth=1, label='Potential shortcut reliance zone')
]
# Legend will be positioned below x tick labels (done later)
axs[1].grid(axis='y', linestyle='--', alpha=0.25, linewidth=0.4)
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)

# Set y-axis limits and tick interval
axs[1].set_ylim(10, 50)
axs[1].set_yticks(np.arange(10, 51, 10))

# Panel label (b) will be positioned after layout adjustments



# Panel D: Scatter plot showing lack of correlation between full input accuracy and robustness
# COMMENTED OUT - Panel D removed, only showing A, B, C in a row
# # Calculate average full input accuracy (NEJM and JAMA)
# avg_full_accuracy = (nejm_full + jama_full) / 2
# 
# # Calculate robustness as 20 - accuracy where accuracy is text-only from stress test 2
# # Higher values indicate better robustness (lower accuracy above chance = less memorization)
# robustness = 20.0 - stress2_text_only
# 
# 
# # Calculate circle sizes proportional to abstention rates
# # Scale so sizes are visually proportional (area proportional to abstention rate)
# # Increased sizes for better visibility in Panel D
# min_size = 150
# max_size = 500
# min_abstention = abstention_rates.min()
# max_abstention = abstention_rates.max()
# if max_abstention != min_abstention:
#     circle_sizes = min_size + (abstention_rates - min_abstention) / (max_abstention - min_abstention) * (max_size - min_size)
# else:
#     circle_sizes = np.full_like(abstention_rates, (min_size + max_size) / 2)
# 
# # Create scatter plot - Nature-style color palette (colorblind-friendly)
# colors_scatter = ['#4A90A4', '#C77D4A', '#7A9E8F', '#D4A574', '#8B7A9E']  # Muted teal, coral, green-teal, tan, purple-gray
# 
# # Define custom label offsets to avoid overlaps for close models (GPT-5, Gemini-2.5 Pro, OpenAI-o3)
# label_offsets = [
#     (-0, 10),    # GPT-5: right and up
#     (-90, -4),   # Gemini-2.5 Pro: left and up
#     (-25, -16),   # OpenAI-o3: right and down
#     (-50, 10),     # OpenAI-o4-mini: default right-up
#     (-15, 15),     # GPT-4o: default right-up
# ]
# 
# for i, (full_acc, rob, model, size) in enumerate(zip(avg_full_accuracy, robustness, models, circle_sizes)):
#     offset = label_offsets[i]
#     # Draw labels first (lower zorder) so circles appear on top
#     if i == 0:  # GPT-5
#         axs[3].annotate(model, (full_acc, rob), xytext=offset, 
#                        textcoords='offset points', fontsize=12, 
#                        fontweight='normal', color='black',
#                        zorder=3)
#         axs[3].scatter(full_acc, rob, s=size, color=colors_scatter[i], 
#                        alpha=0.9, zorder=10, edgecolor='black', linewidth=2)
#     else:
#         axs[3].annotate(model, (full_acc, rob), xytext=offset, 
#                        textcoords='offset points', fontsize=12,
#                        fontweight='normal', color='black',
#                        zorder=3)
#         axs[3].scatter(full_acc, rob, s=size, color=colors_scatter[i], 
#                        alpha=0.7, zorder=10, edgecolor='black', linewidth=2)
# 
# axs[3].set_xlabel('Accuracy with full input (%)', fontsize=14, fontweight='medium')
# axs[3].set_ylabel('Visual robustness gap\n(chance-level - text-only accuracy on NEJM-VS)', fontsize=14, fontweight='medium')
# # Adjust axis limits to use more space and reduce empty area
# x_padding = (avg_full_accuracy.max() - avg_full_accuracy.min()) * 0.15
# y_padding = (robustness.max() - robustness.min()) * 0.15
# axs[3].set_xlim(avg_full_accuracy.min() - x_padding, avg_full_accuracy.max() + x_padding)
# axs[3].set_ylim(robustness.min() - y_padding, robustness.max() + y_padding)
# # Add horizontal dashed line at zero (chance-level robustness) - Nature-style
# axs[3].axhline(y=0, color='#757575', linestyle='--', linewidth=1.5, alpha=0.7, label='chance-level robustness')
# axs[3].grid(True, linestyle='--', alpha=0.25, linewidth=0.4)
# axs[3].spines['top'].set_visible(False)
# axs[3].spines['right'].set_visible(False)
# # Add legend for the chance-level line - positioned at top
# axs[3].legend(fontsize=11, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.1))
# 
# # Panel label (d) will be positioned after layout adjustments

# Layout is fully controlled by gridspec left/right/top/bottom parameters above

# Increase spacing between panels A and B, and B and C
pos_a = axs[0].get_position()
pos_b = axs[1].get_position()
pos_c = axs[2].get_position()

# Reduce spacing between A and B by moving B to the left (closer to A)
extra_spacing_ab = 0.005  # Reduced from 0.035 to make A and B closer
new_b_x0 = pos_b.x0 + extra_spacing_ab
new_b_width = pos_b.width
axs[1].set_position([new_b_x0, pos_b.y0, new_b_width, pos_b.height])

# Increase spacing between B and C by moving C further to the right
# Calculate C's position relative to B's NEW position to maintain spacing
extra_spacing_bc = 0.055  # Increased from 0.035 to make B and C further apart
# Original gap between B and C
original_gap = pos_c.x0 - pos_b.x1
# New position: B's new right edge + original gap + extra spacing
new_c_x0 = new_b_x0 + new_b_width + original_gap + extra_spacing_bc
new_c_width = pos_c.width
axs[2].set_position([new_c_x0, pos_c.y0, new_c_width, pos_c.height])

# Position legends below x tick labels using axis coordinates so they adapt to figsize
legend_kwargs = dict(fontsize=FONTSIZE_LEGEND, frameon=False, loc='upper center')

axs[0].legend(handles_a, labels_a, ncol=2,
              bbox_to_anchor=(0.5, LEGEND_ANCHOR_Y),
              bbox_transform=axs[0].transAxes, **legend_kwargs)
axs[1].legend(handles=legend_elements, ncol=1,
              bbox_to_anchor=(0.5, LEGEND_ANCHOR_Y),
              bbox_transform=axs[1].transAxes, **legend_kwargs)
axs[2].legend(handles=legend_handles_c, ncol=1,
              bbox_to_anchor=(0.5, LEGEND_ANCHOR_Y),
              bbox_transform=axs[2].transAxes, **legend_kwargs)

# Add column headers directly under each axis so spacing follows subplot width
for ax, header in zip(axs, col_headers):
    ax.text(0.5, CAPTION_OFFSET, header, ha='center', va='top',
            fontsize=FONTSIZE_CAPTION, fontweight=FONTWEIGHT_REGULAR, transform=ax.transAxes,
            clip_on=False)

# Add panel labels relative to axes for better responsiveness to figsize tweaks
panel_labels = ['a)', 'b)', 'c)']
for ax, label in zip(axs, panel_labels):
    ax.text(PANEL_LABEL_X_OFFSET, PANEL_LABEL_Y_OFFSET, label, 
            fontsize=FONTSIZE_PANEL_LABEL, fontweight=FONTWEIGHT_PANEL_LABEL,
            va='bottom', ha='right', transform=ax.transAxes, clip_on=False)

plt.savefig('./output/figure2_modality_robustness_v1.pdf', dpi=600, bbox_inches='tight')
plt.show()

# %%
