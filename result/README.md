# Result File Manifest

This folder contains saved model responses generated during the public-subset update.

Primary manuscript results are the `public_original_subset_v1` files for:

- `VQA-RAD-no-cot`
- `PMC-VQA`
- `OmniVQA`

The primary model panel is:

- `gpt55`
- `grok43`
- `claude48opus`
- `gemini35flash`

Files containing `public_no_image_sensitivity_v1` are the sampled text-only sensitivity runs reported in the manuscript and supplement.

Files under `*-sample/` are low-volume smoke tests used only to verify the model routes. They are not performance estimates.

Files under `gemini31pro/` are archived exploratory outputs from the quota-limited Gemini 3.1 Pro attempt. They are retained for transparency but are excluded from the primary analysis and manuscript model panel.
