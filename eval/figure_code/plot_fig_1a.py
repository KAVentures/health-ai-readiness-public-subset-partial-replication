#%%
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import matplotlib.patches as patches
import os

# font_path = '/usr/share/fonts/truetype/msttcorefonts/arial_regular.ttf'

# if os.path.exists(font_path):
#     plt.rcParams['font.family'] = 'Arial'
#     print("Arial added")
# else:
#     print(f"Error: Font file not found.")


# Set publication-quality style with careful attention to balance
plt.style.use('default')
# plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['font.style'] = 'normal'
plt.rcParams['axes.linewidth'] = 0.6
plt.rcParams['xtick.major.width'] = 0.6
plt.rcParams['ytick.major.width'] = 0.6
plt.rcParams['xtick.major.size'] = 3
plt.rcParams['ytick.major.size'] = 3

# Data
models = ["GPT-4o", "Claude-sonnet-3.5", "OpenAI-o4-mini", "OpenAI-o3", "Gemini-2.5 Pro", "GPT-5"]
benchmarks = ["VQA-RAD", "PMC-VQA", "OmniMedVQA", "JAMA", "NEJM"]
scores = np.array([
    [57.56, 52.08, 78.70, 71.51, 67.16],  # GPT-4o
    [57.92, 51.84, 71.30, 76.04, 59.77],  # Claude-sonnet-3.5
    [60.62, 59.12, 84.20, 80.53, 76.62],  # OpenAI-o4-mini
    [65.01, 60.24, 86.34, 86.13, 80.42],  # OpenAI-o3
    [66.16, 58.88, 85.99, 84.19, 81.12],  # Gemini-2.5 Pro
    [64.75, 59.16, 87.18, 86.93, 81.33],  # GPT-5
])

# Calculate mean scores for trend visualization
means = scores.mean(axis=1)

# Add fragility scores from Panel B (higher = worse fragility)
# These are the actual calculated values from the stress test functions
# Note: DeepSeek-VL2 is not included in Panel B, so we'll use a placeholder

# Calculate average fragility for models that appear in Panel B
# This will be the mean across T1-T5 stress tests for each model
# The actual values should come from running plot_fig_1b.py and extracting the frag matrix

# Add the actual fragility calculation functions and data from plot_fig_1b.py
# Item counts for weighting
n_jama = 1314
n_nejm = 947

# JAMA
jama_img = {"GPT-5": 86.93, "Gemini-2.5 Pro": 84.19, "OpenAI-o3": 86.13, "OpenAI-o4-mini": 80.53, "GPT-4o": 71.51, "Claude-sonnet-3.5": 76.04}
jama_txt = {"GPT-5": 82.51, "Gemini-2.5 Pro": 80.91, "OpenAI-o3": 82.79, "OpenAI-o4-mini": 77.47, "GPT-4o": 67.46, "Claude-sonnet-3.5": 76.79}
# NEJM
nejm_img = {"GPT-5": 81.33, "Gemini-2.5 Pro": 81.12, "OpenAI-o3": 80.42, "OpenAI-o4-mini": 76.62, "GPT-4o": 67.16, "Claude-sonnet-3.5": 59.77}
nejm_txt = {"GPT-5": 67.41, "Gemini-2.5 Pro": 67.43, "OpenAI-o3": 66.65, "OpenAI-o4-mini": 66.48, "GPT-4o": 54.23, "Claude-sonnet-3.5": 61.41}


# Stress Test 2: Text-only accuracy on visually-required items (NEJM subset)
t2_text_visreq = {"GPT-5": 41.32, "Gemini-2.5 Pro": 40.10, "OpenAI-o3": 37.26, "OpenAI-o4-mini": 37.97, "GPT-4o": 16.35, "Claude-sonnet-3.5": 33.40}

# Stress Test 3: Text (orig) vs Text (reord) (NEJM visually-required subset)
t3_text_orig = {"GPT-5": 41.32, "Gemini-2.5 Pro": 40.10, "OpenAI-o3": 37.26, "OpenAI-o4-mini": 37.97, "GPT-4o": 16.35, "Claude-sonnet-3.5": 33.40}
t3_text_reord = {"GPT-5":39.70, "Gemini-2.5 Pro": 39.80, "OpenAI-o3": 39.80, "OpenAI-o4-mini": 36.85, "GPT-4o": 19.39, "Claude-sonnet-3.5": 33.40}

# Stress Test 4: Distractor Replacement
# Text base and 4R; Img+Txt base and 4R; UNK (Text)
t4_text_base = {"GPT-5": 41.32, "Gemini-2.5 Pro": 40.10, "OpenAI-o3": 37.26, "OpenAI-o4-mini": 37.97, "GPT-4o": 16.35, "Claude-sonnet-3.5": 33.40}
t4_text_4r = {"GPT-5": 21.02, "Gemini-2.5 Pro": 20.81, "OpenAI-o3": 17.87, "OpenAI-o4-mini": 19.19, "GPT-4o": 13.50, "Claude-sonnet-3.5": 19.59}

t4_img_base = {"GPT-5": 70.05, "Gemini-2.5 Pro": 67.21, "OpenAI-o3": 65.18, "OpenAI-o4-mini": 62.64, "GPT-4o": 51.87, "Claude-sonnet-3.5": 36.75}
t4_img_4r = {"GPT-5": 90.15, "Gemini-2.5 Pro": 90.25, "OpenAI-o3": 90.96, "OpenAI-o4-mini": 85.69, "GPT-4o": 82.13, "Claude-sonnet-3.5": 77.46}

t4_text_unk_base = {"GPT-5": 41.32, "Gemini-2.5 Pro": 40.10, "OpenAI-o3": 37.26, "OpenAI-o4-mini": 37.97, "GPT-4o": 16.35, "Claude-sonnet-3.5": 33.40}
t4_text_unk = {"GPT-5": 45.69, "Gemini-2.5 Pro": 44.67, "OpenAI-o3": 40.71, "OpenAI-o4-mini": 45.08, "GPT-4o": 25.08, "Claude-sonnet-3.5": 28.63}

# Stress Test 5: Visual Substitution (Original vs Substituted)
t5_orig = {"GPT-5": 84.00, "Gemini-2.5 Pro": 76.00, "OpenAI-o3": 83.50, "OpenAI-o4-mini": 64.50, "GPT-4o": 26.50, "Claude-sonnet-3.5": 59.00}
t5_subs = {"GPT-5": 53.00, "Gemini-2.5 Pro": 52.50, "OpenAI-o3": 50.50, "OpenAI-o4-mini": 41.50, "GPT-4o": 36.00, "Claude-sonnet-3.5": 27.50}

# Fragility calculation functions
def weighted_drop(model):
    dj = max(0.0, jama_img[model] - jama_txt[model])
    dn = max(0.0, nejm_img[model] - nejm_txt[model])
    return (dj*n_jama + dn*n_nejm) / (n_jama + n_nejm)

def t2_frag(model):
    above = max(0.0, t2_text_visreq[model] - 20.0)
    return above / 80.0

def t3_frag(model):
    drop = max(0.0, t3_text_orig[model] - t3_text_reord[model])/t3_text_orig[model]
    return drop

def t4_frag(model):
    d_T = max(0.0, t4_text_base[model] - t4_text_4r[model])
    g_V = max(0.0, t4_img_4r[model] - t4_img_base[model])
    u_T = max(0.0, t4_text_unk[model] - t4_text_unk_base[model])
    return 0.5*(d_T/100.0) + 0.3*(g_V/100.0) + 0.2*(u_T/100.0)

def t5_frag(model):
    drop = max(0.0, t5_orig[model] - t5_subs[model])
    return drop / 100.0

# Calculate actual fragility scores for each model
# All models now have stress test data from Panel B
models_with_data = ["GPT-4o", "Claude-sonnet-3.5", "OpenAI-o4-mini", "OpenAI-o3", "Gemini-2.5 Pro", "GPT-5"]

fragility_scores = []
fragility_scores_geometric = []
for model in models_with_data:
    t1 = weighted_drop(model)/100.0
    t2 = t2_frag(model)
    t3 = t3_frag(model)
    t4 = t4_frag(model)
    t5 = t5_frag(model)
    # Arithmetic mean
    avg_fragility = (t1 + t2 + t3 + t4 + t5) / 5.0
    fragility_scores.append(avg_fragility)
    # Geometric mean: (t1 * t2 * t3 * t4 * t5)^(1/5)
    # Add small epsilon to avoid issues with zero values
    epsilon = 1e-10
    geom_fragility = np.power((t1 + epsilon) * (t2 + epsilon) * (t3 + epsilon) * (t4 + epsilon) * (t5 + epsilon), 1.0/5.0)
    fragility_scores_geometric.append(geom_fragility)

# Convert fragility to robustness: robustness = 1 - fragility
# This ensures: fragility 0.0 → robustness 1.0 (perfect)
#               fragility 0.4 → robustness 0.6 (poor)
robustness_scores = [1.0 - frag for frag in fragility_scores]
robustness_scores_geometric = [1.0 - frag for frag in fragility_scores_geometric]

# Print the calculated scores for verification
print("Fragility scores (arithmetic mean across T1-T5):")
for i, model in enumerate(models_with_data):
    print(f"{model}: {fragility_scores[i]:.3f}")

print("\nRobustness scores (1 - fragility):")
for i, model in enumerate(models_with_data):
    print(f"{model}: {robustness_scores[i]:.3f}")

# Create figure with balanced proportions for composite figure
fig, ax = plt.subplots(figsize=(9, 3.5))

# Create secondary y-axis for fragility scores
ax2 = ax.twinx()

# Sophisticated color palette: cool blues with teal accents
# Each color has similar saturation and brightness for harmony
colors = ["#8ECAE6", "#57A49B", "#219EBC", "#126782", "#1B4965", "#0B3D61"]

# Position setup with balanced spacing
x = np.arange(len(models))
bar_width = 0.14  # Slightly wider bars for better proportions
spacing = 0.02   # Better spacing between groups

# Plot grouped bars with refined styling and enhanced visual appeal
for i, benchmark in enumerate(benchmarks):
    positions = x + i * (bar_width + spacing)
    bars = ax.bar(positions, scores[:, i], width=bar_width, 
                  label=benchmark, color=colors[i % len(colors)],
                  alpha=0.9, edgecolor='white', linewidth=1.0)
    
    # Add subtle shadow effect for depth
    for bar in bars:
        bar.set_alpha(0.95)  # Slightly more opaque for better color saturation

# Add elegant trend line connecting mean scores - much more refined
trend_x = x + (len(benchmarks) - 1) * (bar_width + spacing) / 2
ax.plot(trend_x, means, '-', color='#2A9D8F', linewidth=2.5, 
        alpha=0.9, zorder=5, solid_capstyle='round')

# Add subtle trend markers at each point with enhanced styling
ax.scatter(trend_x, means, color='#2A9D8F', s=30, zorder=6, 
          edgecolor='white', linewidth=1.0, alpha=0.95)

# All 7 models now match between Panel A and robustness scores
# Add robustness trend line on secondary y-axis (positive relationship)
if True:
    ax2.plot(trend_x, robustness_scores, '-', color='#E63946', linewidth=2.5, 
            alpha=0.8, zorder=7, label='Robustness Trend')

    # Add robustness markers
    ax2.scatter(trend_x, robustness_scores, color='#E63946', s=25, zorder=8, 
            edgecolor='white', linewidth=1.0, alpha=0.9, marker='s')
 
else:
    # Using geometric mean version
    ax2.plot(trend_x, robustness_scores_geometric, '-', color='#E63946', linewidth=2.5, 
            alpha=0.8, zorder=7, label='Robustness Trend')

    # Add robustness markers
    ax2.scatter(trend_x, robustness_scores_geometric, color='#E63946', s=25, zorder=8, 
            edgecolor='white', linewidth=1.0, alpha=0.9, marker='s')

# Remove the ugly arrow and replace with elegant annotation
# ax.text(trend_x[-1] + 0.08, means[-1] + 2, 'Steady\nImprovement', 
#         fontsize=8, color='#2A9D8F', ha='left', va='bottom', fontweight='medium',
#         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95, 
#                   edgecolor='#2A9D8F', linewidth=0.6))

# Axis formatting with balanced typography
ax.set_xticks(x + (len(benchmarks) - 1) * (bar_width + spacing) / 2)
ax.set_xticklabels(models, rotation=0, ha='center', fontsize=11, fontweight='normal')
ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight='medium')
# ax.set_xlabel("Models (Chronological Order)", fontsize=8, fontweight='normal')
ax.set_ylim(20, 100)

# Configure secondary y-axis for robustness scores
ax2.set_ylabel("Mean Robustness Score", fontsize=11, fontweight='medium', color='black')
ax2.set_ylim(0.8, 1.0)
ax2.tick_params(axis='y', colors='black', labelsize=10)
ax2.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
ax2.set_yticklabels(['0.80', '0.85', '0.90', '0.95', '1.00'])

# Highlight GPT-5 text label (last model - newest)
ax.get_xticklabels()[-1].set_fontweight('bold')
ax.get_xticklabels()[-1].set_color('#2A9D8F')  # Using our teal color

# Add subtle background highlight behind GPT-5 bars (now last model)
# Calculate the full width needed to cover all GPT-5 bars
gpt5_start = x[-1] - 0.15
gpt5_end = x[-1] + (len(benchmarks) - 1) * (bar_width + spacing) + 0.15
gpt5_width = gpt5_end - gpt5_start

gpt5_rect = patches.Rectangle((gpt5_start, 20), gpt5_width, 80, 
                              linewidth=0, facecolor='#2A9D8F', alpha=0.1)
ax.add_patch(gpt5_rect)

# Refined ticks and grid - match Panel B styling
ax.tick_params(axis='both', which='major', labelsize=11)
ax.tick_params(axis='y', which='major', length=8, width=1.2, labelsize=11)
ax.tick_params(axis='x', which='major', length=4, width=0.8, labelsize=11)
ax.set_yticks([20, 40, 60, 80, 100])
ax.grid(axis='y', linestyle='--', alpha=0.15, linewidth=0.3, color='gray')
# Remove top and right spines for cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)

# Also remove top border from secondary y-axis for consistency
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(True)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)

# Legend with balanced layout and refined styling - match Panel B
# First, get handles and labels from both axes
handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

# Combine handles and labels
all_handles = handles1 + handles2
all_labels = labels1 + labels2

legend = ax.legend(all_handles, all_labels, bbox_to_anchor=(0.5, 1.03), 
                  loc='lower center', ncol=6, fontsize=9,
                  frameon=False, fancybox=False, shadow=False,
                  borderpad=0.5)

# Add performance improvement annotation with balanced styling
# improvement = ((means[-1] - means[0]) / means[0]) * 100
# ax.text(0.02, 0.96, f'Overall Improvement: +{improvement:.1f}%', 
#         transform=ax.transAxes, fontsize=8, color='#2A9D8F', fontweight='medium',
#         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.95, 
#                   edgecolor='#2A9D8F', linewidth=0.5))

# Add key insight annotation linking to Panel B
# ax.text(0.02, 0.96, 'Key Insight: High leaderboard scores ≠ High robustness', 
#         transform=ax.transAxes, fontsize=9, color='#E63946', fontweight='medium',
#         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95, 
#                   edgecolor='#E63946', linewidth=0.8))

plt.tight_layout()
plt.savefig("./output/fig1_a.pdf", dpi=600, bbox_inches='tight')
plt.savefig("./output/fig1_a.png", dpi=600, bbox_inches='tight')
plt.show()
plt.close()

# %%
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os

# 1. 直接指定物理路径（请确保路径正确）
font_path = '/usr/share/fonts/truetype/msttcorefonts/arial_regular.ttf'

if os.path.exists(font_path):
    # 2. 手动将该字体注册到当前进程的 fontManager
    font_manager.fontManager.addfont(font_path)
    
    # 3. 设置全局字体名称
    plt.rcParams['font.family'] = 'Arial'
    print("Arial 成功手动加载，已避开缓存系统。")
else:
    print(f"错误：未在 {font_path} 找到字体文件。")

# 绘图测试
plt.plot([1, 2], [3, 4])
plt.title("Fixed: Arial without Cache Lock Error")
plt.show()
# %%
