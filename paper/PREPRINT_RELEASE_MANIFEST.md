# Preprint Release Manifest

This checklist defines the minimum public release package that should accompany a medRxiv or similar preprint submission.

## Include

- `src/`, `scripts/`, and the top-level analysis/build scripts needed to rerun extraction, scoring, figures, manuscript generation, and supplementary tables.
- Public benchmark configuration files and sample-index metadata for VQA-RAD no-CoT, PMC-VQA sampled, and OmniMedVQA sampled.
- Saved model-response JSON files for `public_original_subset_v1`.
- Saved model-response JSON files for `public_no_image_sensitivity_v1`, clearly labelled as sampled sensitivity runs.
- Analysis outputs in `analysis/updated_public/`, including summary tables, paired comparisons, scoring sensitivity, no-image sensitivity, failure examples, and figures.
- Final manuscript and supplementary material in Markdown, PDF, and DOCX.
- A README explaining exact run dates, model identifiers, reasoning settings, and known incomplete cells.

## Exclude

- API keys, `.env` files, local notes containing credentials, and provider billing information.
- Restricted JAMA, NEJM, proprietary unseen-set, T5/T6, or visual-substitution images that were not redistributed by the upstream repository.
- Large public image folders if the destination repository requires users to download them from the original data source instead.

## Before Posting

- Replace local-only paths in repository documentation with public repository URLs.
- Confirm dataset licenses and terms for redistributed derived files.
- Add a permanent archive DOI if possible.
