#%%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches


# Item counts for weighting
n_jama = 1314
n_nejm = 947

# Remove DeepSeek-VL2 since we don't have heatmap data for it
models_heatmap = ["GPT-4o", "Claude-sonnet-3.5","OpenAI-o4-mini", "OpenAI-o3", "Gemini-2.5 Pro", "GPT-5"]

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


# T1: modality sensitivity (weighted drop, normalized to 0..1)
def weighted_drop(model):
    dj = max(0.0, jama_img[model] - jama_txt[model])  # pp
    dn = max(0.0, nejm_img[model] - nejm_txt[model])  # pp
    return (dj*n_jama + dn*n_nejm) / (n_jama + n_nejm)

# T2: necessity (above-chance text-only accuracy on visually-required items)
def t2_frag(model):
    above = max(0.0, t2_text_visreq[model] - 20.0)  # 0..80
    return above / 80.0  # normalize to 0..1

# T3: format perturbation (text drop, pp)
def t3_frag(model):
    # drop = max(0.0, t3_text_orig[model] - t3_text_reord[model])/100.0
    drop = max(0.0, t3_text_orig[model] - t3_text_reord[model])/t3_text_orig[model]
    return drop

# T4: distractor replacement (composite)
def t4_frag(model):
    d_T = max(0.0, t4_text_base[model] - t4_text_4r[model])   # pp drop
    g_V = max(0.0, t4_img_4r[model] - t4_img_base[model])     # pp gain
    u_T = max(0.0, t4_text_unk[model] - t4_text_unk_base[model])  # pp gain
    return 0.5*(d_T/100.0) + 0.3*(g_V/100.0) + 0.2*(u_T/100.0)

# T5: visual substitution (drop, pp)
def t5_frag(model):
    drop = max(0.0, t5_orig[model] - t5_subs[model])
    return drop / 100.0

# Define row labels for stress tests
rows = ["T1: Modality\nSensitivity",
        "T2: Modality\nNecessity", 
        "T3: Format\nPerturbation",
        "T4: Distractor\nReplacement",
        "T5: Visual\nSubstitution"]

# Compute fragility matrix
frag = np.zeros((len(rows), len(models_heatmap)))
for j, model in enumerate(models_heatmap):
    frag[0, j] = weighted_drop(model)/100.0  # convert pp to 0..1
    frag[1, j] = t2_frag(model)
    frag[2, j] = t3_frag(model)
    frag[3, j] = t4_frag(model)
    frag[4, j] = t5_frag(model)

# Compute robustness matrix (R = 1 - F)
robust = 1.0 - frag

# Create Panel B heatmap
plt.style.use('default')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 0.6
plt.rcParams['xtick.major.width'] = 0.6
plt.rcParams['ytick.major.width'] = 0.6
plt.rcParams['xtick.major.size'] = 3
plt.rcParams['ytick.major.size'] = 3

fig, ax = plt.subplots(figsize=(8, 2.5))

# Create heatmap with custom colormap (red = worse robustness)
# Use reversed Reds colormap where darker red = lower robustness
im = ax.imshow(robust, aspect='auto', cmap='Reds_r', vmin=0.6, vmax=1.0)

# Add colorbar with Panel A styling
cbar = plt.colorbar(im, ax=ax, shrink=1.0, aspect=30, pad=0.02)
cbar.set_label("Robustness Score", fontsize=11, fontweight='medium', labelpad=5)
cbar.ax.tick_params(labelsize=10)

# Reduce number of ticks for cleaner appearance
cbar.set_ticks([0.6, 0.8, 1.0])
cbar.set_ticklabels(['0.6', '0.8', '1.0'])

# Move tick labels back to the right side for standard positioning
cbar.ax.yaxis.set_tick_params(pad=0)

# Remove colorbar border for cleaner appearance
cbar.outline.set_linewidth(0)

# Set ticks and labels with Panel A styling
ax.set_xticks(np.arange(len(models_heatmap)))
# Remove model name labels, keep only tick marks
ax.set_xticklabels([''] * len(models_heatmap))
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

# Set y-axis labels with clear descriptions
y_labels = [
    "T1: Modality\nSensitivity",
    "T2: Modality\nNecessity", 
    "T3: Format\nPerturbation",
    "T4: Distractor\nPerturbation",
    "T5: Visual\nSubstitution"
]

ax.set_yticks(range(len(y_labels)))
ax.set_yticklabels(y_labels, fontsize=9, fontweight='medium')

# Set x and y axis labels
# ax.set_xlabel("Models", fontsize=12, fontweight='medium')
ax.set_ylabel("Stress Tests", fontsize=11, fontweight='medium')

# Enhanced y-axis tick styling for main contributions prominence
ax.tick_params(axis='y', which='major', length=8, width=1.2, labelsize=11)
ax.tick_params(axis='x', which='major', length=4, width=0.8, labelsize=11)

# Annotate cells with robustness values using Panel A styling
for i in range(robust.shape[0]):
    for j in range(robust.shape[1]):
        value = robust[i, j]  # Use 0-1 scale
        # Use white text for dark cells (lower robustness), black for light cells (higher robustness)
        color = 'white' if value < 0.8 else 'black'
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", 
                fontsize=9, fontweight='medium', color=color)

# Highlight GPT-5 column (latest model) with Panel A teal color
# ax.axvline(x=len(models_heatmap)-1, color='#2A9D8F', linewidth=2.2, alpha=0.8)

# Highlight GPT-5 text label (last model - newest) in red to match fragility theme
ax.get_xticklabels()[-1].set_fontweight('bold')
ax.get_xticklabels()[-1].set_color('#E63946')  # Red color to match fragility theme

# Remove background highlighting - it interferes with heatmap colors
# gpt5_rect = patches.Rectangle((len(models_heatmap)-1 - 0.5, -0.5), 1.0, len(rows) + 0.5, 
#                               linewidth=0, facecolor='#2A9D8F', alpha=0.1)
# ax.add_patch(gpt5_rect)

# Remove title as per Panel A style
# ax.set_title("Panel B — Stress Test Fragility", fontsize=12, fontweight='medium', pad=15)

# Remove spines for cleaner look (matching Panel A)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Add subtle grid for better readability (matching Panel A style)
ax.grid(axis='both', linestyle='--', alpha=0.15, linewidth=0.3, color='gray')

plt.tight_layout()
plt.savefig("./output/fig1_b.pdf", dpi=600, bbox_inches='tight')
plt.savefig("./output/fig1_b.png", dpi=600, bbox_inches='tight')
plt.show()

print("Panel B robustness heatmap created successfully!")

# %%
