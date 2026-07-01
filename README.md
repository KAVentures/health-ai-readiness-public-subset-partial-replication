# Public-Subset Replication of Health-AI Readiness Benchmarks

This repository contains the code, saved outputs, analysis tables, figures, manuscript, and supplementary material for:

**Public-Subset Replication of Health-AI Readiness Benchmarks With Updated Frontier Multimodal Models**

Author: Koyar Afrasyab, M.D.  
Affiliation: Kinvectum AB

## Scope

This is a public-subset replication and model update based on the original `aiden-ygu/health-ai-readiness-eval` repository. It evaluates newer hosted frontier multimodal models on locally available public benchmark components:

- VQA-RAD no-CoT subset
- PMC-VQA sampled file
- OmniMedVQA sampled file

Restricted JAMA, NEJM, unseen-image, and visual-substitution assets are not included.

## Included

- `paper/`: final manuscript, supplementary material, PDF, DOCX, release manifest, and medRxiv checklist.
- `analysis/updated_public/`: summary tables, per-sample outputs, paired comparisons, scoring sensitivity, no-image sensitivity, failure examples, and generated figures.
- `result/`: saved model-output JSON files, with primary, sensitivity, smoke-test, and exploratory outputs identified in `result/README.md`.
- `src/`, `scripts/`, and top-level build/analysis scripts used to run and analyze the public-subset update.
- `data/`: public metadata/sample files needed to identify evaluated cases, excluding large/raw image folders.

## Excluded

This repository intentionally excludes:

- `.env` files, API keys, credential examples, and API-key setup documentation.
- Provider billing details.
- Restricted JAMA/NEJM/proprietary images and unavailable original-study stress-test images.
- Large/raw public image folders that should be obtained from the original dataset sources.
- Local logs, caches, temporary images, and rendering intermediates.

## Reproduce Analysis From Saved Outputs

The saved result files are included. To regenerate analysis tables, figures, and manuscript artifacts from those saved outputs:

```bash
python analyze_updated_public_results.py
python build_paper_from_results.py
python build_pdf_from_paper.py
python build_docx_from_paper.py
```

## Manuscript

Primary manuscript files:

- `paper/manuscript_results_filled.pdf`
- `paper/manuscript_results_filled.docx`
- `paper/manuscript_results_filled.md`
- `paper/supplementary_results_filled.md`

## Funding and Competing Interests

This project was funded by Kinvectum AB. Koyar Afrasyab, M.D. is the founder of Kinvectum AB.

## Upstream Project

Original evaluation repository:

https://github.com/aiden-ygu/health-ai-readiness-eval
