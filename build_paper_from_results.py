#!/usr/bin/env python3
"""Build manuscript and supplement from generated analysis CSVs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
PAPER_DIR = ROOT / "paper"
ANALYSIS_DIR = ROOT / "analysis" / "updated_public"
TEMPLATE = PAPER_DIR / "manuscript.md"
OUTPUT = PAPER_DIR / "manuscript_results_filled.md"
SUPPLEMENT_OUTPUT = PAPER_DIR / "supplementary_results_filled.md"


def iter_saved_rows(result_path: Path) -> list[dict]:
    data = json.loads(result_path.read_text())
    rows = data.get("results", [])
    if isinstance(rows, dict):
        normalized = []
        for key, value in rows.items():
            if isinstance(value, dict):
                row = dict(value)
                row.setdefault("sample_index", key)
                normalized.append(row)
        return normalized
    return [row for row in rows if isinstance(row, dict)]


def gemini_format_audit() -> str:
    parts: list[str] = []
    for dataset in ["VQA-RAD-no-cot", "PMC-VQA", "OmniVQA"]:
        paths = sorted(
            (ROOT / "result" / dataset / "gemini35flash").glob(
                f"{dataset}_gemini35flash_*public_original_subset_v1.json"
            )
        )
        if not paths:
            continue
        rows = iter_saved_rows(paths[-1])
        no_answer = 0
        prepost = 0
        leak_like = 0
        for row in rows:
            response = str(row.get("response") or row.get("prediction") or "")
            has_answer_tags = "<answer>" in response and "</answer>" in response
            if not has_answer_tags:
                no_answer += 1
            else:
                before = response.split("<answer>", 1)[0].strip()
                after = response.split("</answer>", 1)[1].strip()
                if before or after:
                    prepost += 1
            leak_pattern = re.search(
                r"\b(wait|search|let's search|to determine|analy[sz]e|therefore)\b",
                response,
                re.IGNORECASE,
            )
            if leak_pattern and (not has_answer_tags or response.strip().lower().find("<answer>") > 20):
                leak_like += 1
        parts.append(
            f"{dataset}: {no_answer}/{len(rows)} responses lacked complete answer tags, "
            f"{prepost}/{len(rows)} included text outside tags, and {leak_like}/{len(rows)} showed search- or reasoning-like leakage"
        )
    if not parts:
        return "No Gemini 3.5 Flash format audit could be computed from saved result files."
    return "Gemini 3.5 Flash format audit: " + "; ".join(parts) + "."


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows available._"
    table = df[columns].astype(str)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(row[col].replace("\n", " ") for col in columns) + " |"
        for _, row in table.iterrows()
    ]
    return "\n".join([header, divider, *rows])


def pct(value: float) -> str:
    return f"{value:.1%}"


def fmt_p(value: float) -> str:
    if pd.isna(value):
        return "NA"
    if value < 0.0001:
        return "<0.0001"
    return f"{value:.4f}"


def read_optional_csv(filename: str) -> pd.DataFrame:
    path = ANALYSIS_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


ORIGINAL_FIGURE1A_PUBLIC = pd.DataFrame(
    [
        {"model": "GPT-4o", "VQA-RAD": 57.56, "PMC-VQA": 52.08, "OmniMedVQA": 78.70},
        {"model": "Claude Sonnet 3.5", "VQA-RAD": 57.92, "PMC-VQA": 51.84, "OmniMedVQA": 71.30},
        {"model": "OpenAI o4-mini", "VQA-RAD": 60.62, "PMC-VQA": 59.12, "OmniMedVQA": 84.20},
        {"model": "OpenAI o3", "VQA-RAD": 65.01, "PMC-VQA": 60.24, "OmniMedVQA": 86.34},
        {"model": "Gemini 2.5 Pro", "VQA-RAD": 66.16, "PMC-VQA": 58.88, "OmniMedVQA": 85.99},
        {"model": "GPT-5", "VQA-RAD": 64.75, "PMC-VQA": 59.16, "OmniMedVQA": 87.18},
    ]
)


def original_comparison_display(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mapping = {
        "VQA-RAD-no-cot": "VQA-RAD",
        "PMC-VQA": "PMC-VQA",
        "OmniVQA": "OmniMedVQA",
    }
    for dataset, original_col in mapping.items():
        sub = summary[summary["dataset"].astype(str) == dataset]
        if sub.empty:
            continue
        current_best = sub.sort_values(["accuracy_relaxed", "accuracy_strict"], ascending=False).iloc[0]
        original_best = ORIGINAL_FIGURE1A_PUBLIC.loc[
            ORIGINAL_FIGURE1A_PUBLIC[original_col].idxmax()
        ]
        original_gpt5 = ORIGINAL_FIGURE1A_PUBLIC[
            ORIGINAL_FIGURE1A_PUBLIC["model"] == "GPT-5"
        ].iloc[0]
        rows.append(
            {
                "dataset": str(sub["dataset_label"].iloc[0]),
                "upstream_figure_best": f"{original_best['model']} {float(original_best[original_col]):.1f}%",
                "upstream_figure_gpt5": f"{float(original_gpt5[original_col]):.1f}%",
                "this_update_best": f"{current_best['model']} {pct(float(current_best['accuracy_relaxed']))}",
                "interpretation": "not a formal paired test; public subset and model panel differ",
            }
        )
    return pd.DataFrame(rows)


def scoring_sensitivity_display() -> pd.DataFrame:
    sens = read_optional_csv("scoring_sensitivity.csv")
    if sens.empty:
        return sens
    out = sens.copy()
    out["accuracy_strict"] = out["accuracy_strict"].map(pct)
    out["accuracy_relaxed"] = out["accuracy_relaxed"].map(pct)
    out["relaxed_minus_strict"] = out["relaxed_minus_strict"].map(lambda x: f"{100 * float(x):.1f} pp")
    return out


def scoring_sensitivity_note() -> str:
    sens = read_optional_csv("scoring_sensitivity.csv")
    if sens.empty:
        return "A strict-versus-relaxed scoring sensitivity table was not available."
    max_row = sens.loc[sens["relaxed_minus_strict"].idxmax()]
    mean_gap = sens.groupby("dataset_label")["relaxed_minus_strict"].mean().reset_index()
    gap_parts = [
        f"{row['dataset_label']} mean gap {100 * float(row['relaxed_minus_strict']):.1f} percentage points"
        for _, row in mean_gap.iterrows()
    ]
    return (
        "Strict and relaxed scoring did not behave as interchangeable endpoints. "
        + "; ".join(gap_parts)
        + f". The largest single model-dataset gap was {100 * float(max_row['relaxed_minus_strict']):.1f} percentage points "
        f"for {max_row['model']} on {max_row['dataset_label']}."
    )


def no_image_sensitivity_table() -> str:
    comparison = read_optional_csv("no_image_comparison.csv")
    if comparison.empty:
        return "_No text-only sensitivity run was available at manuscript build time._"
    display = comparison.copy()
    for col in ["image_accuracy_relaxed", "text_only_accuracy_relaxed", "image_minus_text_accuracy"]:
        display[col] = display[col].map(lambda x: f"{100 * float(x):.1f}%")
    return markdown_table(
        display,
        [
            "dataset_label",
            "model",
            "n_paired",
            "image_accuracy_relaxed",
            "text_only_accuracy_relaxed",
            "image_minus_text_accuracy",
            "text_only_correct_image_wrong",
            "image_correct_text_only_wrong",
        ],
    )


def no_image_sensitivity_note() -> str:
    comparison = read_optional_csv("no_image_comparison.csv")
    if comparison.empty:
        return (
            "A text-only no-image sensitivity run has been implemented in the evaluation harness, "
            "but no text-only result files were present when this manuscript was built. The claim set therefore remains limited to image+text public benchmarks."
        )
    completed = len(comparison)
    max_drop = comparison.loc[comparison["image_minus_text_accuracy"].idxmax()]
    min_drop = comparison.loc[comparison["image_minus_text_accuracy"].idxmin()]
    missing_note = ""
    if "gpt55" not in set(comparison["model_short"].astype(str)):
        missing_note = (
            " GPT-5.5 is absent from this sensitivity table because the added text-only run returned "
            "OpenAI `insufficient_quota` errors and was stopped to avoid repeated failed calls."
        )
    return (
        f"An exploratory paired no-image sensitivity analysis was available for {completed} model-dataset cells. "
        f"The largest image+text advantage was {100 * float(max_drop['image_minus_text_accuracy']):.1f} percentage points "
        f"for {max_drop['model']} on {max_drop['dataset_label']}; the smallest was "
        f"{100 * float(min_drop['image_minus_text_accuracy']):.1f} percentage points for {min_drop['model']} on {min_drop['dataset_label']}. "
        "Because this was a sampled sensitivity check rather than the primary full public run, it is interpreted as evidence about visual dependence, not as a replacement endpoint."
        + missing_note
    )


def no_image_figure_markdown() -> str:
    if (ANALYSIS_DIR / "fig_no_image_sensitivity.png").exists():
        return "![Figure 4. No-image sensitivity analysis. Bars compare matched image-plus-text and text-only accuracy in the sampled no-image sensitivity runs.](../analysis/updated_public/fig_no_image_sensitivity.png)"
    return ""


def failure_examples_table() -> str:
    examples = read_optional_csv("failure_examples.csv")
    if examples.empty:
        return "_No discordant failure examples were generated._"
    display = examples.copy()
    return markdown_table(
        display,
        [
            "dataset_label",
            "sample_index",
            "ground_truth",
            "correct_models",
            "incorrect_model_predictions",
        ],
    )


def status_sentence(summary: pd.DataFrame) -> str:
    complete = summary[summary["complete_dataset_run"] == True]
    incomplete = summary[summary["complete_dataset_run"] == False]
    complete_cells = len(complete)
    total_cells = len(summary)
    if incomplete.empty:
        return f"All {total_cells} generated model-dataset cells reached the prespecified sample count."
    missing = "; ".join(
        f"{row['model']} on {row['dataset_label']} ({int(row['n'])}/{int(row['expected_n'])})"
        for _, row in incomplete.iterrows()
    )
    return (
        f"{complete_cells} of {total_cells} generated model-dataset cells reached the prespecified sample count. "
        f"Incomplete cells were retained but labelled explicitly: {missing}."
    )


def top_complete_results(summary: pd.DataFrame) -> str:
    lines: list[str] = []
    for dataset in ["VQA-RAD-no-cot", "PMC-VQA", "OmniVQA"]:
        sub = summary[summary["dataset"].astype(str) == dataset]
        if sub.empty:
            continue
        complete = sub[sub["complete_dataset_run"] == True].copy()
        label = str(sub["dataset_label"].iloc[0])
        if complete.empty:
            lines.append(f"For {label}, no model had a complete full-subset run.")
            continue
        best = complete.sort_values(["accuracy_relaxed", "accuracy_strict"], ascending=False).iloc[0]
        lines.append(
            f"For {label}, the highest relaxed automated accuracy among complete cells was "
            f"{pct(float(best['accuracy_relaxed']))} for {best['model']} "
            f"({int(best['correct_relaxed'])}/{int(best['n'])}; Wilson 95% CI "
            f"{pct(float(best['ci95_low_relaxed']))}-{pct(float(best['ci95_high_relaxed']))})."
        )
    return " ".join(lines)


def complete_range(summary: pd.DataFrame) -> str:
    complete = summary[summary["complete_dataset_run"] == True]
    if complete.empty:
        return "No full model-dataset benchmark cell was complete at the time of analysis."
    return (
        f"Across complete model-dataset cells, relaxed automated accuracy ranged from "
        f"{pct(float(complete['accuracy_relaxed'].min()))} to "
        f"{pct(float(complete['accuracy_relaxed'].max()))}."
    )


def make_summary_display(summary: pd.DataFrame) -> pd.DataFrame:
    display = summary.copy()
    display["accuracy_relaxed"] = display["accuracy_relaxed"].map(pct)
    display["accuracy_strict"] = display["accuracy_strict"].map(pct)
    display["ci95"] = summary.apply(
        lambda r: f"{pct(float(r['ci95_low_relaxed']))}-{pct(float(r['ci95_high_relaxed']))}",
        axis=1,
    )
    display["mean_tokens_per_sample"] = summary["mean_tokens_per_sample"].map(lambda x: f"{x:.0f}")
    display["total_tokens"] = summary["total_tokens"].map(lambda x: f"{int(x):,}")
    return display


def make_paired_display(paired: pd.DataFrame) -> pd.DataFrame:
    paired_display = paired.copy()
    for col in ["accuracy_a", "accuracy_b", "difference_a_minus_b"]:
        paired_display[col] = paired_display[col].map(lambda x: f"{x:.4f}")
    paired_display["mcnemar_exact_p"] = paired_display["mcnemar_exact_p"].map(fmt_p)
    return paired_display


def generate_manuscript(summary: pd.DataFrame, paired: pd.DataFrame) -> str:
    display = make_summary_display(summary)
    results_table = markdown_table(
        display,
        [
            "dataset_label",
            "model",
            "n",
            "expected_n",
            "cell_status",
            "accuracy_strict",
            "accuracy_relaxed",
            "ci95",
            "mean_tokens_per_sample",
        ],
    )

    if not paired.empty:
        paired_table = markdown_table(
            make_paired_display(paired),
            [
                "dataset_label",
                "model_a",
                "model_b",
                "n_paired",
                "accuracy_a",
                "accuracy_b",
                "difference_a_minus_b",
                "a_only_correct",
                "b_only_correct",
                "mcnemar_exact_p",
            ],
        )
    else:
        paired_table = "_No paired comparisons available._"

    incomplete_note = status_sentence(summary)
    top_note = top_complete_results(summary)
    range_note = complete_range(summary)
    gemini_audit_note = gemini_format_audit()
    scoring_note = scoring_sensitivity_note()
    no_image_note = no_image_sensitivity_note()
    no_image_figure = no_image_figure_markdown()
    original_comparison_table = markdown_table(
        original_comparison_display(summary),
        [
            "dataset",
            "upstream_figure_best",
            "upstream_figure_gpt5",
            "this_update_best",
            "interpretation",
        ],
    )
    references = """1. Gu Y, Fu J, Liu X, et al. Evaluating the robustness and readiness of large frontier models in health AI applications. Accompanying evaluation repository. https://github.com/aiden-ygu/health-ai-readiness-eval
2. Gu Y, et al. Health-AI-Readiness evaluation repository README and reproduction notes. https://github.com/aiden-ygu/health-ai-readiness-eval
3. Lau JJ, Gayen S, Ben Abacha A, Demner-Fushman D. A dataset of clinically generated visual questions and answers about radiology images. Scientific Data. 2018;5:180251. https://www.nature.com/articles/sdata2018251
4. Zhang X, et al. PMC-VQA: visual instruction tuning for medical visual question answering. Dataset and paper resources: https://huggingface.co/datasets/RadGenome/PMC-VQA
5. Hu Y, et al. OmniMedVQA: A New Large-Scale Comprehensive Evaluation Benchmark for Medical LVLM. https://arxiv.org/abs/2402.09181 and https://huggingface.co/datasets/foreverbeliever/OmniMedVQA"""

    return f"""# Public-Subset Replication of Health-AI Readiness Benchmarks With Updated Frontier Multimodal Models

Koyar Afrasyab, M.D.

Affiliation: Kinvectum AB

## Abstract

**Background:** Gu and colleagues evaluated the robustness and readiness of frontier models for health AI applications and released an accompanying evaluation repository [1,2]. Rapid model turnover makes periodic replication useful, but several original-study image sources are restricted and not redistributed. This study therefore asks a narrower question: how do newer hosted frontier models perform on the reproducible public benchmark layer, and what does that layer fail to establish?

**Methods:** We adapted the original Health-AI-Readiness evaluation harness to API-hosted frontier models available during the June 28-30, 2026 run window: GPT-5.5, Grok 4.3, Claude Opus 4.8, and Gemini 3.5 Flash. The primary analysis used the locally validated public subsets available from the original repository workflow: VQA-RAD no-CoT (100 cases), PMC-VQA sampled (500 cases), and OmniMedVQA sampled (524 records). Restricted JAMA, NEJM, visual substitution, and unseen-image stress tests were excluded because the required images were not redistributed and were not locally available. Accuracy was summarized with Wilson 95% confidence intervals. Paired comparisons used exact McNemar tests on shared samples. We also report strict-versus-relaxed scoring sensitivity, representative discordant failures, provider-format anomalies, and a sampled text-only no-image sensitivity analysis when corresponding runs are available.

**Results:** {incomplete_note} {range_note} {top_note} {scoring_note} An earlier Gemini 3.1 Pro exploratory run was abandoned after repeated quota limits and is not part of the primary model panel. Any remaining incomplete primary cells are reported for transparency and are not treated as full-subset estimates.

**Conclusions:** The updated public-subset replication provides a useful 2026 snapshot of newer hosted multimodal models on reproducible medical VQA benchmarks, but it does not establish health-AI readiness. The main value is methodological and comparative: it updates the public leaderboard, quantifies scoring and no-image sensitivity, and shows why restricted-image robustness tests remain necessary before clinical-readiness claims.

## Introduction

Medical visual question answering is a useful but narrow probe of multimodal model behavior in clinically flavored tasks. The original Health-AI-Readiness study argued that high benchmark performance can coexist with brittleness under adversarial or modality-altering perturbations [1,2]. Its released repository provides evaluation code, benchmark indices, prompt templates, stress-test specifications, and analysis utilities, while also documenting that raw JAMA, NEJM, proprietary unseen-set, and visual-substitution images are not redistributed [1,2].

This update has two aims. First, it adapts the original evaluation harness to a newer model panel. Second, it separates immediately reproducible public benchmark results from analyses that require restricted source materials. That separation is central to the claims made here: the paper reports a public-subset model update, not a complete re-run of every original study component. The comparison with Gu et al. is therefore framed as a scoped replication: it tests whether current models change the public-benchmark picture, while preserving the original paper's broader readiness question for a later restricted-asset phase [1,2].

The relevance of the study is practical rather than sweeping. Medical readers should not infer deployment readiness from these experiments, but they can use them to understand how quickly public benchmark baselines move, which model families warrant follow-up robustness testing, and where automated medical VQA evaluation remains brittle. The specific contributions are: an updated public-subset model panel; a reproducible audit of dataset availability; strict-versus-relaxed scoring sensitivity; sampled no-image sensitivity; representative discordant failures; and documentation of provider/API behaviors that affected reproducibility.

## Methods

### Study Design

This is a replication-oriented benchmark update using the cloned `aiden-ygu/health-ai-readiness-eval` repository as the source-of-truth evaluation harness. We preserved the repository's dataset configurations, prompt builders, result format, and redistributed public subset files where possible, while adding provider support for the newer target models.

### Benchmark Datasets

The primary public benchmark run was `public_original_subset_v1`:

| Dataset | Prespecified count | Task format | Included in primary public run |
|---|---:|---|---|
| VQA-RAD no-CoT subset | 100 | radiology VQA, open-ended and yes/no | yes |
| PMC-VQA sampled file | 500 | multiple-choice medical figure VQA | yes |
| OmniMedVQA sampled file | 524 | multiple-choice multimodal medical VQA | yes |

VQA-RAD contains clinically generated radiology image question-answer pairs [3]. PMC-VQA contains medical figure questions derived from PubMed Central open-access articles [4]. OmniMedVQA is a broad medical VQA benchmark spanning multiple modalities and medical subdomains [5]. For OmniMedVQA, the original download target was unavailable during setup, so the sampled records were matched to an accessible public archive and all referenced sampled images were extracted locally.

### Excluded Original-Study Components

The following original-study components were excluded from the primary analysis because required image assets were absent locally and not redistributed by the repository:

| Component | Reason for exclusion |
|---|---|
| JAMA Clinical Challenge | image assets not redistributed and not locally available |
| NEJM Image Challenge | image assets not redistributed and not locally available |
| NEJM stress tests ST_v0-ST_v8 | metadata present, but image assets absent |
| ST_v10 unseen-image set | JSON/images absent in the cloned repository |
| T5/T6 visual substitution/rationale tests | require restricted NEJM-derived image materials |

These are data-availability exclusions, not evidence that the omitted tests are unimportant. T5/T6 should be run as a secondary phase if the restricted images are supplied or reconstructed under a documented protocol.

### Model Panel and API Configuration

| Model key | Provider route | Configured model identifier | Reasoning setting | Run window |
|---|---|---|---|---|
| `gpt-5.5` | OpenAI Responses API | `gpt-5.5` | high | June 28-30, 2026 |
| `grok-4.3` | xAI Responses-compatible API | `grok-4.3` | high/configurable | June 28-30, 2026 |
| `claude-opus-4.8` | Anthropic Messages API | `claude-opus-4-8` | high/adaptive thinking | June 28-30, 2026 |
| `gemini-3.5-flash` | Google Gemini API | `gemini-3.5-flash` | high thinking | June 28-30, 2026 |

All four API routes passed low-volume smoke tests before benchmark runs. Smoke tests are reported only as plumbing checks, not as performance estimates. Gemini 3.1 Pro was initially tested but replaced by Gemini 3.5 Flash after repeated quota limits; the Gemini 3.1 Pro outputs remain archived locally but are excluded from the primary analysis. Hosted model outputs may change as providers update models or serving infrastructure; the run dates, model identifiers, reasoning settings, extraction rules, and saved response files are therefore part of the reproducibility record.

### Prompting and Scoring

We used the original repository prompt builders. PMC-VQA and OmniMedVQA prompts asked models to choose one provided option and place the answer inside `<answer></answer>` tags. VQA-RAD prompts requested concise free-text or yes/no answers in the same tag format.

The primary automated endpoint was per-dataset accuracy. For multiple-choice datasets, single-letter ground truths were scored by exact option-letter match; relaxed substring matching was deliberately disabled for one-character option labels because it can create false positives. For VQA-RAD, strict scoring used normalized string equality, while the relaxed rule allowed yes/no normalization and conservative short-answer containment. VQA-RAD results should still be interpreted with caution because deterministic matching may undercount clinically equivalent free-text answers.

A paired text-only mode was implemented for no-image sensitivity checks. In this mode the same prompt text is sent without image payloads. This test asks whether a model can answer from dataset priors, option wording, or memorized associations when visual evidence is absent. It is not a substitute for the upstream restricted-image stress tests, but it directly addresses whether public benchmark accuracy is visually grounded.

### Statistical Analysis

For each model-dataset pair, we report evaluated sample count, strict accuracy, relaxed automated accuracy, Wilson 95% confidence intervals for relaxed accuracy, and mean token count per saved response as a descriptive reproducibility variable. Pairwise model comparisons use exact McNemar tests on shared sample indices only, so incomplete cells contribute only where both models have predictions for the same cases. Figures and tables were generated by `analyze_updated_public_results.py`.

## Results

### Data and API Readiness

The local data audit found complete public benchmark readiness for VQA-RAD no-CoT, PMC-VQA, and OmniMedVQA sampled records. The audit also confirmed that JAMA, NEJM, proprietary unseen-set, and visual-substitution image assets were absent. API smoke tests succeeded for the configured provider routes before the primary benchmark runs; the initial quota-limited Gemini 3.1 Pro route was replaced by Gemini 3.5 Flash for the final model panel.

### Primary Public Benchmark Results

{incomplete_note}

Table 1 reports all generated `public_original_subset_v1` cells. Rows marked incomplete are retained for transparency but should not be interpreted as full benchmark estimates.

{results_table}

![Figure 1. Relaxed automated accuracy by dataset. Bars show model-level relaxed automated accuracy with Wilson 95% confidence intervals.](../analysis/updated_public/fig_accuracy_by_dataset.png)

![Figure 2. Complete primary cells only. Bars show relaxed automated accuracy after excluding incomplete model-dataset cells.](../analysis/updated_public/fig_accuracy_complete_only.png)

![Figure 3. Accuracy heatmap. Cells summarize relaxed automated accuracy by model and public benchmark dataset.](../analysis/updated_public/fig_accuracy_heatmap.png)

### Scoring and No-Image Sensitivity

{scoring_note}

{no_image_note}

{no_image_figure}

### Paired Model Comparisons

Exact McNemar tests were calculated on shared sample indices. These p-values are descriptive and were not adjusted for multiplicity.

{paired_table}

## Discussion

This public-subset update supports three conclusions. First, newer hosted frontier models can score strongly on reproducible medical VQA benchmarks, particularly sampled OmniMedVQA, but performance remains dataset-dependent. Second, benchmark scores are sensitive to evaluation details: strict and relaxed scoring diverged sharply on OmniMedVQA, and a small no-image sensitivity experiment showed that some items can be answered without visual input. Third, the most clinically important readiness questions from Gu et al. remain unresolved here because the restricted JAMA, NEJM, visual-substitution, and unseen-image assets were not available.

The public benchmark results are nevertheless informative. Complete cells ranged from 60.6% to 90.5% relaxed automated accuracy. The best complete cells were GPT-5.5 on VQA-RAD sampled, GPT-5.5 and Grok 4.3 on PMC-VQA sampled, and Claude Opus 4.8 on OmniMedVQA sampled. This pattern argues against describing current medical multimodal capability with a single headline number: the same model panel looks modest on PMC-VQA, stronger on VQA-RAD, and strongest on OmniMedVQA.

The original Gu et al. study is most important here because of what it showed beyond ordinary benchmark accuracy [1,2]. Their central result was not simply that one model was better than another; it was that frontier models can look strong on conventional multimodal medical benchmarks while still failing readiness-oriented stress tests. In their analyses, models retained substantial accuracy when images were removed from tasks that should require visual evidence, shifted answers after seemingly superficial prompt or option perturbations, degraded when images were substituted to support a distractor, and sometimes produced fluent but visually ungrounded rationales. They also argued that common medical VQA benchmarks differ in visual dependence and reasoning demand, so a single aggregate benchmark score can hide qualitatively different failure modes.

Against that backdrop, our update should be read as a narrower but still useful comparison. We replaced the original model panel with GPT-5.5, Grok 4.3, Claude Opus 4.8, and Gemini 3.5 Flash, and observed public-benchmark scores consistent with continued progress on standard VQA-style tasks. Those results are not evidence that the original brittleness has disappeared, because the restricted-image tests that exposed much of that brittleness were not available for this run.

The most defensible comparison is therefore asymmetric. Gu et al. tested both performance and reasons-for-performance; this update mostly tests performance on the public subset. Where our models score highly, the finding is novel for this 2026 panel and useful as an updated public benchmark estimate. Where the original paper found shortcut reliance, visual-input fragility, and rationale unreliability, our study does not overturn those findings. Instead, it shows that even after model turnover, the public benchmark layer remains insufficient for readiness claims unless paired with the same kinds of missing-image, option-perturbation, visual-substitution, and rationale-grounding tests.

The upstream repository's public benchmark plotting code provides a useful, auditable benchmark reference for the public benchmark layer. Those values are not treated as a formal paired comparison because the exact model panel and subset definitions differ, but they help calibrate whether the updated panel plausibly shifts the public leaderboard:

{original_comparison_table}

The comparison suggests measured progress on some public endpoints but not a qualitative escape from the original critique. Our best OmniMedVQA cell exceeds the upstream figure-code best among the earlier model panel, while the PMC-VQA cells remain clustered near the upstream range. VQA-RAD also remains scoring-sensitive because clinically similar short answers can differ lexically. The implication is not "newer models are ready"; it is "newer models can be stronger on the reproducible public surface, and we still need the missing stress-test layer to decide whether those gains are visually grounded and robust."

| Comparison point | Gu et al. original study | This update |
|---|---|---|
| Model panel | Earlier frontier systems evaluated by the original authors | GPT-5.5, Grok 4.3, Claude Opus 4.8, and Gemini 3.5 Flash |
| Main empirical object | Public benchmark performance plus stress-test robustness | Public benchmark performance on locally available subsets |
| Main original finding | High medical benchmark scores can coexist with shortcut reliance, modality fragility, prompt sensitivity, and ungrounded reasoning | Current models still score strongly on public VQA-style tasks, especially OmniMedVQA, but robustness remains untested here |
| Restricted readiness tests | JAMA/NEJM missing-image, option-perturbation, image-substitution, unseen-image, and rationale-grounding analyses were central to the readiness claim | Not rerun because the required restricted image assets were not available locally |
| Interpretation of differences | Apparent readiness can be inflated when benchmarks reward textual priors, familiar options, or memorized cases | Updated scores should be treated as a new public-subset leaderboard, not as evidence against the original readiness critique |
| New contribution | Original robustness/readiness framing and benchmark release | Updated 2026 public-subset estimates plus provider-level reproducibility observations |

The comparison also changes the interpretation of novelty. The public-subset leaderboard patterns in this update are new empirical observations for the evaluated 2026 model panel. The broader conclusion that medical AI readiness cannot be inferred from standard benchmark accuracy is not new; it is inherited from and supported by Gu et al. What is new here is the operational finding that provider behavior has become a reproducibility variable in its own right: quota limits, no-text responses, token budgets, media-type handling, and answer-format compliance all affected whether a nominally simple replication could be completed without manual intervention.

{gemini_audit_note} These Gemini format issues did not prevent completion of the public benchmark run, but they show why saved raw responses, extraction rules, and missing/error handling need to be published alongside headline accuracy.

Failure-mode inspection reinforced the same point. Discordant examples, reported in the supplement, show cases where one model matched the answer while another chose a plausible but different anatomical structure, measurement, or option letter. These are not mere bookkeeping errors: they are the cases a human reviewer would use to ask whether a model saw the image, relied on option priors, or benefited from relaxed scoring.

The analysis also identified two implementation details that materially affect truthfulness. First, multiple-choice scoring must avoid substring matching when the ground truth is only a single option letter. Second, image media type must be inferred from actual bytes rather than filename extension for Anthropic image uploads, because at least one locally stored file had a `.png` suffix but JPEG bytes. Both issues were corrected before the reported analysis.

Data contamination remains a live threat to interpretation. VQA-RAD, PMC-VQA, and OmniMedVQA are public benchmarks [3-5]; model providers do not expose training membership, and high accuracy on public images can reflect learned associations rather than deployed visual reasoning. The no-image sensitivity check is included partly for this reason: correct answers without images do not prove contamination, but they are a warning that benchmark items can sometimes be answered from text, options, or prior exposure alone.

For clinical implications, the conservative reading is the most defensible one. If the newer models outperform earlier systems on parts of the public benchmark suite, that is encouraging for general medical image-question answering and useful for selecting models for deeper evaluation. But the missing restricted-asset tasks are exactly the tasks designed to test whether models preserve reliability when image provenance, visual evidence, or rationale structure changes. The present results should therefore be used to prioritize a second-stage robustness run, not to claim clinical deployment readiness.

## Limitations

This is a public-subset model update, not a complete reproduction of the original Health-AI-Readiness study. Restricted JAMA/NEJM images, proprietary unseen images, and T5/T6 visual substitution materials were not available. Deterministic VQA-RAD scoring may misclassify semantically correct free-text answers, so publication-grade radiology claims would benefit from blinded human adjudication or a prespecified independent judge. The no-image analysis is a sampled sensitivity check and should not be overread as the full upstream visual-dependence battery. Mean token counts summarize successful saved responses and do not fully capture failed retries or provider-side billing details. The Gemini 3.1 Pro exploratory run was stopped because of observed quota limits and replaced with Gemini 3.5 Flash in the primary panel. Finally, all evaluated systems are hosted APIs whose behavior may change after the run date.

## Data and Code Availability

The adapted evaluation code, saved model outputs, analysis CSVs, figures, and manuscript-generation scripts are prepared for public release with the preprint. The release package should include the exact result files, extraction rules, analysis tables, generated figures, and a manifest of excluded restricted assets. Public dataset use follows the terms of the original dataset sources. Restricted source images are not included.

## Ethics Statement

This study used existing benchmark datasets and saved model outputs only. No human participants were recruited, no clinical intervention was performed, and no private medical records were accessed by the authors during this analysis. The work evaluates benchmark behavior and should not be interpreted as evidence that any evaluated model is safe for autonomous clinical use.

## Author Contributions

Koyar Afrasyab conceived the study, configured the updated evaluation, supervised model runs, interpreted the results, and drafted and revised the manuscript.

## Funding

This project was funded by Kinvectum AB (https://kinvectum.com).

## Competing Interests

Koyar Afrasyab, M.D. is the founder of Kinvectum AB, which funded the project.

## References

{references}
"""


def generate_supplement(summary: pd.DataFrame, paired: pd.DataFrame) -> str:
    display = make_summary_display(summary)
    completeness = markdown_table(
        display,
        ["dataset_label", "model", "n", "expected_n", "cell_status", "result_file_status", "total_tokens"],
    )
    paired_display = make_paired_display(paired) if not paired.empty else pd.DataFrame()
    paired_table = (
        markdown_table(
            paired_display,
            [
                "dataset_label",
                "model_a",
                "model_b",
                "n_paired",
                "a_only_correct",
                "b_only_correct",
                "mcnemar_exact_p",
            ],
        )
        if not paired.empty
        else "_No paired comparisons available._"
    )
    scoring_table = markdown_table(
        scoring_sensitivity_display(),
        [
            "dataset_label",
            "model",
            "n",
            "accuracy_strict",
            "accuracy_relaxed",
            "relaxed_minus_strict",
        ],
    )
    no_image_table = no_image_sensitivity_table()
    failures_table = failure_examples_table()
    return f"""# Supplementary Material

## Supplementary Methods

### Repository and Code Changes

The original `aiden-ygu/health-ai-readiness-eval` repository was used as the evaluation base. The update added direct provider support for GPT-5.5, Grok 4.3, Claude Opus 4.8, and Gemini 3.5 Flash; added high-reasoning configuration controls; added public benchmark runner scripts; and added a reproducible analysis script for per-sample outputs, confidence intervals, paired comparisons, and figures.

### Data Validation

The primary public run used VQA-RAD no-CoT (`n=100`), PMC-VQA sampled (`n=500`), and OmniMedVQA sampled (`n=524`). JAMA, NEJM, ST_v0-ST_v8, ST_v10, and T5/T6 were excluded because required source images were absent locally.

### Scoring Audit

Multiple-choice answers with single-letter ground truths were scored by option letter. Relaxed substring matching was not used for those one-character labels, preventing accidental matches such as a reference of `B` matching unrelated words containing the letter `b`. VQA-RAD retained deterministic normalized scoring and is flagged as the main candidate for semantic adjudication sensitivity analysis.

### Provider Anomalies

An exploratory Gemini 3.1 Pro run was quota-limited and was replaced by Gemini 3.5 Flash for the primary model panel. If a primary cell remains incomplete, missing responses are retained as missing rather than inferred from the ground truth.

{gemini_format_audit()}

## Supplementary Table S1. Model Configuration

| Model | API route | Configured model identifier | Reasoning setting | Run window |
|---|---|---|---|---|
| GPT-5.5 | OpenAI Responses API | `gpt-5.5` | `OPENAI_REASONING_EFFORT=high` | June 28-30, 2026 |
| Grok 4.3 | xAI Responses-compatible API | `grok-4.3` | `XAI_REASONING_EFFORT=high` | June 28-30, 2026 |
| Claude Opus 4.8 | Anthropic Messages API | `claude-opus-4-8` | `ANTHROPIC_REASONING_EFFORT=high` | June 28-30, 2026 |
| Gemini 3.5 Flash | Google Gemini API | `gemini-3.5-flash` | `GOOGLE_REASONING_EFFORT=HIGH` | June 28-30, 2026 |

## Supplementary Table S2. Completeness and Token Audit

{completeness}

## Supplementary Table S3. Discordant Pair Counts

{paired_table}

## Supplementary Table S4. Strict-versus-Relaxed Scoring Sensitivity

{scoring_table}

## Supplementary Table S5. Paired Text-Only No-Image Sensitivity

{no_image_table}

## Supplementary Table S6. Representative Discordant Failure Examples

{failures_table}

## Reproducibility Commands

```bash
python data/verify_data_setup.py
./scripts/run_updated_frontier_smoke.sh
EXP_ID=public_original_subset_v1 REASON_LEVEL=high ./scripts/run_updated_public_benchmarks.sh <model>
python src/test_runner.py VQA-RAD-no-cot gpt-5.5 public_no_image_sensitivity_v1 text high --sample 25 --seed 20260630
python analyze_updated_public_results.py
python build_paper_from_results.py
```
"""


def main() -> None:
    summary_path = ANALYSIS_DIR / "summary_by_model_dataset.csv"
    paired_path = ANALYSIS_DIR / "paired_model_comparisons.csv"
    if not summary_path.exists():
        raise SystemExit(f"Missing {summary_path}. Run analyze_updated_public_results.py first.")

    summary = pd.read_csv(summary_path)
    paired = pd.read_csv(paired_path) if paired_path.exists() else pd.DataFrame()

    OUTPUT.write_text(generate_manuscript(summary, paired))
    SUPPLEMENT_OUTPUT.write_text(generate_supplement(summary, paired))
    print(f"Wrote {OUTPUT}")
    print(f"Wrote {SUPPLEMENT_OUTPUT}")


if __name__ == "__main__":
    main()
