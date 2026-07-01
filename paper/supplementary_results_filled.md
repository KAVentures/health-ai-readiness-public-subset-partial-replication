# Supplementary Material

## Supplementary Methods

### Repository and Code Changes

The original `aiden-ygu/health-ai-readiness-eval` repository was used as the evaluation base. The update added direct provider support for GPT-5.5, Grok 4.3, Claude Opus 4.8, and Gemini 3.5 Flash; added high-reasoning configuration controls; added public benchmark runner scripts; and added a reproducible analysis script for per-sample outputs, confidence intervals, paired comparisons, and figures.

### Data Validation

The primary public run used VQA-RAD no-CoT (`n=100`), PMC-VQA sampled (`n=500`), and OmniMedVQA sampled (`n=524`). JAMA, NEJM, ST_v0-ST_v8, ST_v10, and T5/T6 were excluded because required source images were absent locally.

### Scoring Audit

Multiple-choice answers with single-letter ground truths were scored by option letter. Relaxed substring matching was not used for those one-character labels, preventing accidental matches such as a reference of `B` matching unrelated words containing the letter `b`. VQA-RAD retained deterministic normalized scoring and is flagged as the main candidate for semantic adjudication sensitivity analysis.

### Provider Anomalies

An exploratory Gemini 3.1 Pro run was quota-limited and was replaced by Gemini 3.5 Flash for the primary model panel. If a primary cell remains incomplete, missing responses are retained as missing rather than inferred from the ground truth.

Gemini 3.5 Flash format audit: VQA-RAD-no-cot: 1/100 responses lacked complete answer tags, 0/100 included text outside tags, and 0/100 showed search- or reasoning-like leakage; PMC-VQA: 56/500 responses lacked complete answer tags, 3/500 included text outside tags, and 48/500 showed search- or reasoning-like leakage; OmniVQA: 6/524 responses lacked complete answer tags, 2/524 included text outside tags, and 7/524 showed search- or reasoning-like leakage.

## Supplementary Table S1. Model Configuration

| Model | API route | Configured model identifier | Reasoning setting | Run window |
|---|---|---|---|---|
| GPT-5.5 | OpenAI Responses API | `gpt-5.5` | `OPENAI_REASONING_EFFORT=high` | June 28-30, 2026 |
| Grok 4.3 | xAI Responses-compatible API | `grok-4.3` | `XAI_REASONING_EFFORT=high` | June 28-30, 2026 |
| Claude Opus 4.8 | Anthropic Messages API | `claude-opus-4-8` | `ANTHROPIC_REASONING_EFFORT=high` | June 28-30, 2026 |
| Gemini 3.5 Flash | Google Gemini API | `gemini-3.5-flash` | `GOOGLE_REASONING_EFFORT=HIGH` | June 28-30, 2026 |

## Supplementary Table S2. Completeness and Token Audit

| dataset_label | model | n | expected_n | cell_status | result_file_status | total_tokens |
| --- | --- | --- | --- | --- | --- | --- |
| VQA-RAD sampled | GPT-5.5 | 100 | 100 | complete | complete | 141,976 |
| VQA-RAD sampled | Grok 4.3 | 100 | 100 | complete | complete | 251,544 |
| VQA-RAD sampled | Claude Opus 4.8 | 100 | 100 | complete | complete | 106,866 |
| VQA-RAD sampled | Gemini 3.5 Flash | 100 | 100 | complete | complete | 173,361 |
| PMC-VQA sampled | GPT-5.5 | 500 | 500 | complete | complete | 790,727 |
| PMC-VQA sampled | Grok 4.3 | 500 | 500 | complete | complete | 1,194,560 |
| PMC-VQA sampled | Claude Opus 4.8 | 499 | 500 | incomplete (499/500) | complete | 206,024 |
| PMC-VQA sampled | Gemini 3.5 Flash | 500 | 500 | complete | complete | 1,137,862 |
| OmniMedVQA sampled | GPT-5.5 | 524 | 524 | complete | complete | 1,179,426 |
| OmniMedVQA sampled | Grok 4.3 | 524 | 524 | complete | complete | 945,509 |
| OmniMedVQA sampled | Claude Opus 4.8 | 524 | 524 | complete | complete | 765,580 |
| OmniMedVQA sampled | Gemini 3.5 Flash | 524 | 524 | complete | complete | 821,184 |

## Supplementary Table S3. Discordant Pair Counts

| dataset_label | model_a | model_b | n_paired | a_only_correct | b_only_correct | mcnemar_exact_p |
| --- | --- | --- | --- | --- | --- | --- |
| OmniMedVQA sampled | GPT-5.5 | Grok 4.3 | 524 | 43 | 33 | 0.3019 |
| OmniMedVQA sampled | GPT-5.5 | Claude Opus 4.8 | 524 | 16 | 44 | 0.0004 |
| OmniMedVQA sampled | GPT-5.5 | Gemini 3.5 Flash | 524 | 18 | 37 | 0.0145 |
| OmniMedVQA sampled | Grok 4.3 | Claude Opus 4.8 | 524 | 14 | 52 | <0.0001 |
| OmniMedVQA sampled | Grok 4.3 | Gemini 3.5 Flash | 524 | 21 | 50 | 0.0008 |
| OmniMedVQA sampled | Claude Opus 4.8 | Gemini 3.5 Flash | 524 | 29 | 20 | 0.2529 |
| PMC-VQA sampled | GPT-5.5 | Grok 4.3 | 500 | 53 | 53 | 1.0000 |
| PMC-VQA sampled | GPT-5.5 | Claude Opus 4.8 | 499 | 62 | 53 | 0.4558 |
| PMC-VQA sampled | GPT-5.5 | Gemini 3.5 Flash | 500 | 53 | 44 | 0.4168 |
| PMC-VQA sampled | Grok 4.3 | Claude Opus 4.8 | 499 | 59 | 50 | 0.4437 |
| PMC-VQA sampled | Grok 4.3 | Gemini 3.5 Flash | 500 | 71 | 62 | 0.4880 |
| PMC-VQA sampled | Claude Opus 4.8 | Gemini 3.5 Flash | 499 | 55 | 56 | 1.0000 |
| VQA-RAD sampled | GPT-5.5 | Grok 4.3 | 100 | 18 | 7 | 0.0433 |
| VQA-RAD sampled | GPT-5.5 | Claude Opus 4.8 | 100 | 19 | 8 | 0.0522 |
| VQA-RAD sampled | GPT-5.5 | Gemini 3.5 Flash | 100 | 11 | 9 | 0.8238 |
| VQA-RAD sampled | Grok 4.3 | Claude Opus 4.8 | 100 | 15 | 15 | 1.0000 |
| VQA-RAD sampled | Grok 4.3 | Gemini 3.5 Flash | 100 | 12 | 21 | 0.1628 |
| VQA-RAD sampled | Claude Opus 4.8 | Gemini 3.5 Flash | 100 | 9 | 18 | 0.1221 |

## Supplementary Table S4. Strict-versus-Relaxed Scoring Sensitivity

| dataset_label | model | n | accuracy_strict | accuracy_relaxed | relaxed_minus_strict |
| --- | --- | --- | --- | --- | --- |
| VQA-RAD sampled | Claude Opus 4.8 | 100 | 54.0% | 62.0% | 8.0 pp |
| VQA-RAD sampled | GPT-5.5 | 100 | 65.0% | 73.0% | 8.0 pp |
| VQA-RAD sampled | Gemini 3.5 Flash | 100 | 64.0% | 71.0% | 7.0 pp |
| VQA-RAD sampled | Grok 4.3 | 100 | 55.0% | 62.0% | 7.0 pp |
| PMC-VQA sampled | Claude Opus 4.8 | 499 | 60.5% | 60.5% | 0.0 pp |
| PMC-VQA sampled | GPT-5.5 | 500 | 62.4% | 62.4% | 0.0 pp |
| PMC-VQA sampled | Gemini 3.5 Flash | 500 | 60.6% | 60.6% | 0.0 pp |
| PMC-VQA sampled | Grok 4.3 | 500 | 62.4% | 62.4% | 0.0 pp |
| OmniMedVQA sampled | Claude Opus 4.8 | 524 | 66.6% | 90.5% | 23.9 pp |
| OmniMedVQA sampled | GPT-5.5 | 524 | 61.6% | 85.1% | 23.5 pp |
| OmniMedVQA sampled | Gemini 3.5 Flash | 524 | 66.2% | 88.7% | 22.5 pp |
| OmniMedVQA sampled | Grok 4.3 | 524 | 59.9% | 83.2% | 23.3 pp |

## Supplementary Table S5. Paired Text-Only No-Image Sensitivity

| dataset_label | model | n_paired | image_accuracy_relaxed | text_only_accuracy_relaxed | image_minus_text_accuracy | text_only_correct_image_wrong | image_correct_text_only_wrong |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VQA-RAD sampled | Grok 4.3 | 25 | 76.0% | 52.0% | 24.0% | 0 | 6 |
| VQA-RAD sampled | Claude Opus 4.8 | 25 | 68.0% | 48.0% | 20.0% | 1 | 6 |
| VQA-RAD sampled | Gemini 3.5 Flash | 25 | 76.0% | 24.0% | 52.0% | 0 | 13 |
| PMC-VQA sampled | Grok 4.3 | 25 | 56.0% | 48.0% | 8.0% | 2 | 4 |
| PMC-VQA sampled | Claude Opus 4.8 | 25 | 64.0% | 20.0% | 44.0% | 0 | 11 |
| PMC-VQA sampled | Gemini 3.5 Flash | 25 | 52.0% | 44.0% | 8.0% | 3 | 5 |
| OmniMedVQA sampled | Grok 4.3 | 25 | 88.0% | 72.0% | 16.0% | 1 | 5 |
| OmniMedVQA sampled | Claude Opus 4.8 | 25 | 96.0% | 52.0% | 44.0% | 0 | 11 |
| OmniMedVQA sampled | Gemini 3.5 Flash | 25 | 92.0% | 68.0% | 24.0% | 1 | 7 |

## Supplementary Table S6. Representative Discordant Failure Examples

| dataset_label | sample_index | ground_truth | correct_models | incorrect_model_predictions |
| --- | --- | --- | --- | --- |
| OmniMedVQA sampled | 76 | Melanoma | Claude Opus 4.8 | GPT-5.5: D. Actinic keratosis; Gemini 3.5 Flash: D. Actinic keratosis; Grok 4.3: D. Actinic keratosis |
| OmniMedVQA sampled | 90 | No | Claude Opus 4.8 | GPT-5.5: B. Yes; Gemini 3.5 Flash: B. Yes; Grok 4.3: B. Yes |
| OmniMedVQA sampled | 136 | Age-related Macular degeneration (AMD). | Claude Opus 4.8 | GPT-5.5: D. Age-related Macular degeneration (AMD); Gemini 3.5 Flash: D. Age-related Macular degeneration (AMD); Grok 4.3: C. Stargardt disease |
| OmniMedVQA sampled | 195 | Phase-contrast microscopy | GPT-5.5 | Claude Opus 4.8: A. Confocal microscopy; Gemini 3.5 Flash: A. Confocal microscopy; Grok 4.3: A. Confocal microscopy |
| PMC-VQA sampled | 11 | A | GPT-5.5 | Claude Opus 4.8: D: Benign; Gemini 3.5 Flash: D: Benign; Grok 4.3: D: Benign |
| PMC-VQA sampled | 14 | C | Gemini 3.5 Flash | Claude Opus 4.8: D: Directly into the ICA; GPT-5.5: A: Anterior to the ICA; Grok 4.3: D: Directly into the ICA |
| PMC-VQA sampled | 15 | A | Gemini 3.5 Flash | Claude Opus 4.8: C: Front; GPT-5.5: C: Front; Grok 4.3: B: Right |
| PMC-VQA sampled | 22 | B | GPT-5.5 | Claude Opus 4.8: D: None of the above.; Gemini 3.5 Flash: D: None of the above.; Grok 4.3: D: None of the above. |
| VQA-RAD sampled | 1 | 2.5cm x 1.7cm x 1.6cm | Grok 4.3 | Claude Opus 4.8: The mass measures approximately 1 cm in diameter.; GPT-5.5: Approximately 1.5 cm; Gemini 3.5 Flash: 1.5 cm |
| VQA-RAD sampled | 5 | chest | Grok 4.3 | Claude Opus 4.8: Musculoskeletal system; GPT-5.5: Respiratory system; Gemini 3.5 Flash: Gastrointestinal |
| VQA-RAD sampled | 17 | yes | GPT-5.5 | Claude Opus 4.8: No; Gemini 3.5 Flash: No; Grok 4.3: No |
| VQA-RAD sampled | 27 | It is less than half the width of the thorax | Gemini 3.5 Flash | Claude Opus 4.8: Less than half the size; GPT-5.5: Less than half the thoracic width.; Grok 4.3: The heart is <50% the width of the thorax (normal CTR <0.5). |

## Reproducibility Commands

```bash
python data/verify_data_setup.py
./scripts/run_updated_frontier_smoke.sh
EXP_ID=public_original_subset_v1 REASON_LEVEL=high ./scripts/run_updated_public_benchmarks.sh <model>
python src/test_runner.py VQA-RAD-no-cot gpt-5.5 public_no_image_sensitivity_v1 text high --sample 25 --seed 20260630
python analyze_updated_public_results.py
python build_paper_from_results.py
```
