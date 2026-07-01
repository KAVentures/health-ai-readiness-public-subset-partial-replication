#!/usr/bin/env python3
"""
Analyze updated public benchmark results for the Health-AI-Readiness replication.

Outputs:
- analysis/updated_public/per_sample_results.csv
- analysis/updated_public/summary_by_model_dataset.csv
- analysis/updated_public/paired_model_comparisons.csv
- analysis/updated_public/scoring_sensitivity.csv
- analysis/updated_public/failure_examples.csv
- analysis/updated_public/no_image_summary_by_model_dataset.csv, when text-only runs exist
- analysis/updated_public/no_image_comparison.csv, when text-only runs exist
- analysis/updated_public/fig_accuracy_by_dataset.png
- analysis/updated_public/fig_accuracy_complete_only.png
- analysis/updated_public/fig_accuracy_heatmap.png
- analysis/updated_public/fig_no_image_sensitivity.png, when text-only runs exist
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint


ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "result"
OUT_DIR = ROOT / "analysis" / "updated_public"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

import matplotlib.pyplot as plt


MODEL_ORDER = ["gpt55", "grok43", "claude48opus", "gemini35flash"]
MODEL_LABELS = {
    "gpt55": "GPT-5.5",
    "grok43": "Grok 4.3",
    "claude48opus": "Claude Opus 4.8",
    "gemini35flash": "Gemini 3.5 Flash",
    "gemini31pro": "Gemini 3.1 Pro (archived exploratory)",
}
DATASET_ORDER = ["VQA-RAD-no-cot", "PMC-VQA", "OmniVQA"]
DATASET_LABELS = {
    "VQA-RAD-no-cot": "VQA-RAD sampled",
    "PMC-VQA": "PMC-VQA sampled",
    "OmniVQA": "OmniMedVQA sampled",
}
EXPECTED_N = {
    "VQA-RAD-no-cot": 100,
    "PMC-VQA": 500,
    "OmniVQA": 524,
}
EXPERIMENT_ID = "public_original_subset_v1"
NO_IMAGE_EXPERIMENT_ID = "public_no_image_sensitivity_v1"
PRIMARY_MODEL_SHORTS = set(MODEL_ORDER)


@dataclass
class ParsedResult:
    dataset: str
    model_short: str
    model_label: str
    expected_n: int
    result_file_status: str
    complete_dataset_run: bool
    sample_index: int
    prediction_raw: str
    prediction: str
    ground_truth: str
    correct_strict: bool
    correct_relaxed: bool
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    total_tokens: int


def normalize_text(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"<\/?answer>", " ", text, flags=re.I)
    text = text.replace("&nbsp;", " ")
    text = re.sub(r"^[a-d]\s*[:.)-]\s*", "", text, flags=re.I)
    text = re.sub(r"\b(option|answer)\b", " ", text)
    text = re.sub(r"[^a-z0-9.+/% -]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_answer(response: str) -> str:
    text = str(response or "")
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.I | re.S)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_letter(value: str) -> Optional[str]:
    text = str(value or "").strip()
    match = re.match(r"(?:<answer>\s*)?([A-D])\s*[:.)-]?", text, flags=re.I)
    if match:
        return match.group(1).upper()
    return None


def yes_no_value(value: str) -> Optional[bool]:
    norm = normalize_text(value)
    if norm in {"yes", "y", "true", "present"}:
        return True
    if norm in {"no", "n", "false", "absent"}:
        return False
    if norm.startswith("yes "):
        return True
    if norm.startswith("no "):
        return False
    return None


def is_correct(dataset: str, prediction_raw: str, ground_truth: str) -> Tuple[bool, bool, str]:
    pred = extract_answer(prediction_raw)
    pred_norm = normalize_text(pred)
    gt_norm = normalize_text(ground_truth)

    pred_letter = extract_letter(pred)
    gt_letter = extract_letter(str(ground_truth))
    if gt_letter:
        strict = pred_letter == gt_letter
        # If the reference is only an option letter, substring matching is invalid:
        # a ground truth of "B" would otherwise match almost any answer containing
        # the character "b". Only allow text containment when the normalized
        # reference is longer than a single option letter.
        relaxed = strict or (len(gt_norm) > 1 and gt_norm in pred_norm)
        return strict, relaxed, pred

    strict = pred_norm == gt_norm

    pred_yes_no = yes_no_value(pred)
    gt_yes_no = yes_no_value(ground_truth)
    if gt_yes_no is not None:
        relaxed = pred_yes_no == gt_yes_no
        return strict, relaxed, pred

    relaxed = strict
    if gt_norm and len(gt_norm) >= 4:
        relaxed = relaxed or gt_norm in pred_norm or pred_norm in gt_norm

    return strict, relaxed, pred


def iter_result_files(experiment_id: str = EXPERIMENT_ID) -> Iterable[Path]:
    for dataset in DATASET_ORDER:
        dataset_dir = RESULT_ROOT / dataset
        if not dataset_dir.exists():
            continue
        for model_dir in dataset_dir.iterdir():
            if not model_dir.is_dir():
                continue
            if model_dir.name not in PRIMARY_MODEL_SHORTS:
                continue
            files = sorted(model_dir.glob(f"{dataset}_{model_dir.name}_*_{experiment_id}.json"))
            if files:
                yield files[-1]


def load_results(experiment_id: str = EXPERIMENT_ID) -> pd.DataFrame:
    rows: List[ParsedResult] = []
    for path in iter_result_files(experiment_id):
        with path.open() as f:
            payload = json.load(f)
        metadata = payload.get("metadata", {})
        dataset = metadata.get("dataset") or path.parents[1].name
        model_short = path.parent.name
        model_label = MODEL_LABELS.get(model_short, model_short)
        results = payload.get("results", {})
        expected_n = EXPECTED_N.get(dataset, len(results))
        complete_dataset_run = len(results) >= expected_n
        result_file_status = str(metadata.get("status", "unknown"))
        for key, item in results.items():
            prediction_raw = item.get("response", "")
            if str(prediction_raw).startswith("Error:"):
                continue
            ground_truth = item.get("ground_truth", "")
            strict, relaxed, prediction = is_correct(dataset, prediction_raw, ground_truth)
            usage = item.get("token_usage") or {}
            rows.append(
                ParsedResult(
                    dataset=dataset,
                    model_short=model_short,
                    model_label=model_label,
                    expected_n=expected_n,
                    result_file_status=result_file_status,
                    complete_dataset_run=complete_dataset_run,
                    sample_index=int(item.get("index", key)),
                    prediction_raw=prediction_raw,
                    prediction=prediction,
                    ground_truth=str(ground_truth),
                    correct_strict=bool(strict),
                    correct_relaxed=bool(relaxed),
                    prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                    completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                    reasoning_tokens=int(usage.get("reasoning_tokens", 0) or 0),
                    total_tokens=int(usage.get("total_tokens", 0) or 0),
                )
            )
    return pd.DataFrame([r.__dict__ for r in rows])


def summarize(df: pd.DataFrame, metric: str = "correct_relaxed") -> pd.DataFrame:
    records = []
    for (dataset, model_short, model_label), group in df.groupby(["dataset", "model_short", "model_label"]):
        n = len(group)
        k = int(group[metric].sum())
        ci_low, ci_high = proportion_confint(k, n, alpha=0.05, method="wilson") if n else (math.nan, math.nan)
        expected_n = int(group["expected_n"].iloc[0])
        complete_dataset_run = bool(n >= expected_n)
        records.append(
            {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS.get(dataset, dataset),
                "model_short": model_short,
                "model": model_label,
                "n": n,
                "expected_n": expected_n,
                "complete_dataset_run": complete_dataset_run,
                "cell_status": "complete" if complete_dataset_run else f"incomplete ({n}/{expected_n})",
                "result_file_status": str(group["result_file_status"].iloc[0]),
                "correct_strict": int(group["correct_strict"].sum()),
                "accuracy_strict": group["correct_strict"].mean(),
                "correct_relaxed": int(group["correct_relaxed"].sum()),
                "accuracy_relaxed": group["correct_relaxed"].mean(),
                "ci95_low_relaxed": ci_low,
                "ci95_high_relaxed": ci_high,
                "total_tokens": int(group["total_tokens"].sum()),
                "prompt_tokens": int(group["prompt_tokens"].sum()),
                "completion_tokens": int(group["completion_tokens"].sum()),
                "reasoning_tokens": int(group["reasoning_tokens"].sum()),
                "mean_tokens_per_sample": group["total_tokens"].mean(),
            }
        )
    out = pd.DataFrame(records)
    if out.empty:
        return out
    out["dataset"] = pd.Categorical(out["dataset"], DATASET_ORDER, ordered=True)
    out["model_short"] = pd.Categorical(out["model_short"], MODEL_ORDER, ordered=True)
    return out.sort_values(["dataset", "model_short"]).reset_index(drop=True)


def paired_comparisons(df: pd.DataFrame, metric: str = "correct_relaxed") -> pd.DataFrame:
    records = []
    for dataset, dset in df.groupby("dataset"):
        pivot = dset.pivot_table(index="sample_index", columns="model_short", values=metric, aggfunc="first")
        for i, model_a in enumerate(MODEL_ORDER):
            for model_b in MODEL_ORDER[i + 1 :]:
                if model_a not in pivot.columns or model_b not in pivot.columns:
                    continue
                pair = pivot[[model_a, model_b]].dropna()
                if pair.empty:
                    continue
                a = pair[model_a].astype(bool)
                b = pair[model_b].astype(bool)
                both_correct = int((a & b).sum())
                a_only = int((a & ~b).sum())
                b_only = int((~a & b).sum())
                both_wrong = int((~a & ~b).sum())
                table = [[both_correct, a_only], [b_only, both_wrong]]
                try:
                    p_value = float(mcnemar(table, exact=True).pvalue)
                except Exception:
                    p_value = math.nan
                records.append(
                    {
                        "dataset": dataset,
                        "dataset_label": DATASET_LABELS.get(dataset, dataset),
                        "model_a": MODEL_LABELS.get(model_a, model_a),
                        "model_b": MODEL_LABELS.get(model_b, model_b),
                        "n_paired": int(len(pair)),
                        "accuracy_a": float(a.mean()),
                        "accuracy_b": float(b.mean()),
                        "difference_a_minus_b": float(a.mean() - b.mean()),
                        "a_only_correct": a_only,
                        "b_only_correct": b_only,
                        "mcnemar_exact_p": p_value,
                    }
                )
    return pd.DataFrame(records)


def scoring_sensitivity(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    out = summary[
        [
            "dataset",
            "dataset_label",
            "model",
            "n",
            "accuracy_strict",
            "accuracy_relaxed",
        ]
    ].copy()
    out["relaxed_minus_strict"] = out["accuracy_relaxed"] - out["accuracy_strict"]
    return out.sort_values(["dataset", "model"]).reset_index(drop=True)


def failure_examples(df: pd.DataFrame, max_per_dataset: int = 4) -> pd.DataFrame:
    records = []
    if df.empty:
        return pd.DataFrame()
    for dataset, dset in df.groupby("dataset"):
        pivot_correct = dset.pivot_table(
            index="sample_index",
            columns="model_label",
            values="correct_relaxed",
            aggfunc="first",
        )
        pivot_pred = dset.pivot_table(
            index="sample_index",
            columns="model_label",
            values="prediction",
            aggfunc="first",
        )
        gt = dset.groupby("sample_index")["ground_truth"].first()
        candidates = []
        for sample_index, row in pivot_correct.iterrows():
            values = row.dropna().astype(bool)
            if values.empty or values.all() or (~values).all():
                continue
            candidates.append((sample_index, int((~values).sum()), int(values.sum())))
        candidates.sort(key=lambda item: (-item[1], item[0]))
        for sample_index, _, _ in candidates[:max_per_dataset]:
            correct_models = []
            incorrect_parts = []
            for model in pivot_correct.columns:
                value = pivot_correct.loc[sample_index, model]
                if pd.isna(value):
                    continue
                pred = str(pivot_pred.loc[sample_index, model])
                pred = re.sub(r"\s+", " ", pred).strip()
                if len(pred) > 90:
                    pred = pred[:87] + "..."
                if bool(value):
                    correct_models.append(model)
                else:
                    incorrect_parts.append(f"{model}: {pred}")
            records.append(
                {
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS.get(dataset, dataset),
                    "sample_index": int(sample_index),
                    "ground_truth": str(gt.loc[sample_index]),
                    "correct_models": "; ".join(correct_models),
                    "incorrect_model_predictions": "; ".join(incorrect_parts),
                }
            )
    return pd.DataFrame(records)


def no_image_comparison(primary_df: pd.DataFrame, no_image_df: pd.DataFrame) -> pd.DataFrame:
    if primary_df.empty or no_image_df.empty:
        return pd.DataFrame()
    merged = primary_df.merge(
        no_image_df,
        on=["dataset", "model_short", "sample_index"],
        suffixes=("_image", "_text"),
    )
    if merged.empty:
        return pd.DataFrame()
    records = []
    for (dataset, model_short), group in merged.groupby(["dataset", "model_short"]):
        image_correct = group["correct_relaxed_image"].astype(bool)
        text_correct = group["correct_relaxed_text"].astype(bool)
        records.append(
            {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS.get(dataset, dataset),
                "model_short": model_short,
                "model": MODEL_LABELS.get(model_short, model_short),
                "n_paired": int(len(group)),
                "image_accuracy_relaxed": float(image_correct.mean()),
                "text_only_accuracy_relaxed": float(text_correct.mean()),
                "image_minus_text_accuracy": float(image_correct.mean() - text_correct.mean()),
                "text_only_correct_image_wrong": int((text_correct & ~image_correct).sum()),
                "image_correct_text_only_wrong": int((image_correct & ~text_correct).sum()),
            }
        )
    out = pd.DataFrame(records)
    if out.empty:
        return out
    out["dataset"] = pd.Categorical(out["dataset"], DATASET_ORDER, ordered=True)
    out["model_short"] = pd.Categorical(out["model_short"], MODEL_ORDER, ordered=True)
    return out.sort_values(["dataset", "model_short"]).reset_index(drop=True)


def plot_accuracy(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(13, 4), sharey=True)
    if len(DATASET_ORDER) == 1:
        axes = [axes]
    palette = ["#2f6f73", "#b4513e", "#6b6f8f", "#c8a23a"]
    for ax, dataset in zip(axes, DATASET_ORDER):
        sub = summary[summary["dataset"].astype(str) == dataset]
        x = np.arange(len(sub))
        y = sub["accuracy_relaxed"].to_numpy(dtype=float)
        yerr = np.vstack([
            y - sub["ci95_low_relaxed"].to_numpy(dtype=float),
            sub["ci95_high_relaxed"].to_numpy(dtype=float) - y,
        ])
        bars = ax.bar(x, y, yerr=yerr, capsize=4, color=palette[: len(sub)], edgecolor="#1f2933", linewidth=0.7)
        for bar, (_, row) in zip(bars, sub.iterrows()):
            label = f"n={int(row['n'])}"
            if not bool(row["complete_dataset_run"]):
                bar.set_hatch("//")
                label = f"{int(row['n'])}/{int(row['expected_n'])}"
            text_y = max(0.04, min(float(row["accuracy_relaxed"]) - 0.035, 0.92))
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                text_y,
                label,
                ha="center",
                va="top",
                fontsize=7,
                color="white" if text_y > 0.20 else "#1f2933",
                rotation=90 if not bool(row["complete_dataset_run"]) and float(row["accuracy_relaxed"]) < 0.22 else 0,
            )
        ax.set_title(DATASET_LABELS.get(dataset, dataset))
        ax.set_xticks(x)
        ax.set_xticklabels(sub["model"].tolist(), rotation=35, ha="right")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Accuracy (relaxed rule, 95% Wilson CI)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_accuracy_by_dataset.png", dpi=220)
    plt.close(fig)


def plot_complete_accuracy(summary: pd.DataFrame) -> None:
    complete = summary[summary["complete_dataset_run"] == True].copy()
    if complete.empty:
        return
    plot_summary = complete[complete["dataset"].astype(str).isin(DATASET_ORDER)]
    fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(13, 4), sharey=True)
    if len(DATASET_ORDER) == 1:
        axes = [axes]
    palette = ["#2f6f73", "#b4513e", "#6b6f8f", "#c8a23a"]
    for ax, dataset in zip(axes, DATASET_ORDER):
        sub = plot_summary[plot_summary["dataset"].astype(str) == dataset]
        x = np.arange(len(sub))
        y = sub["accuracy_relaxed"].to_numpy(dtype=float)
        if len(sub):
            yerr = np.vstack([
                y - sub["ci95_low_relaxed"].to_numpy(dtype=float),
                sub["ci95_high_relaxed"].to_numpy(dtype=float) - y,
            ])
            ax.bar(x, y, yerr=yerr, capsize=4, color=palette[: len(sub)], edgecolor="#1f2933", linewidth=0.7)
        ax.set_title(DATASET_LABELS.get(dataset, dataset))
        ax.set_xticks(x)
        ax.set_xticklabels(sub["model"].tolist(), rotation=35, ha="right")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Accuracy, complete cells only (95% Wilson CI)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_accuracy_complete_only.png", dpi=220)
    plt.close(fig)


def plot_heatmap(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    pivot = summary.pivot(index="model", columns="dataset", values="accuracy_relaxed")
    pivot = pivot.reindex([MODEL_LABELS[m] for m in MODEL_ORDER if MODEL_LABELS[m] in pivot.index])
    pivot = pivot[[d for d in DATASET_ORDER if d in pivot.columns]]
    pivot = pivot.rename(columns=DATASET_LABELS)
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(pivot.to_numpy(dtype=float), vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            model_label = pivot.index[i]
            dataset_label = pivot.columns[j]
            row = summary[
                (summary["model"] == model_label)
                & (summary["dataset_label"] == dataset_label)
            ]
            if pd.isna(val):
                ax.text(j, i, "NA", ha="center", va="center", color="#1f2933")
                continue
            suffix = ""
            if not row.empty and not bool(row.iloc[0]["complete_dataset_run"]):
                suffix = "*"
            ax.text(j, i, f"{val:.1%}{suffix}", ha="center", va="center", color="white" if val < 0.65 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Accuracy")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_accuracy_heatmap.png", dpi=220)
    plt.close(fig)


def plot_no_image_sensitivity(comparison: pd.DataFrame) -> None:
    if comparison.empty:
        return
    fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(13, 4), sharey=True)
    if len(DATASET_ORDER) == 1:
        axes = [axes]
    palette = ["#2f6f73", "#b4513e", "#6b6f8f", "#c8a23a"]
    for ax, dataset in zip(axes, DATASET_ORDER):
        sub = comparison[comparison["dataset"].astype(str) == dataset]
        x = np.arange(len(sub))
        width = 0.36
        ax.bar(
            x - width / 2,
            sub["image_accuracy_relaxed"],
            width=width,
            color=palette[: len(sub)],
            alpha=0.95,
            edgecolor="#1f2933",
            linewidth=0.6,
            label="image+text",
        )
        ax.bar(
            x + width / 2,
            sub["text_only_accuracy_relaxed"],
            width=width,
            color=palette[: len(sub)],
            alpha=0.35,
            edgecolor="#1f2933",
            linewidth=0.6,
            hatch="//",
            label="text only",
        )
        for i, (_, row) in enumerate(sub.iterrows()):
            ax.text(i, 0.03, f"n={int(row['n_paired'])}", ha="center", va="bottom", fontsize=7)
        ax.set_title(DATASET_LABELS.get(dataset, dataset))
        ax.set_xticks(x)
        ax.set_xticklabels(sub["model"].tolist(), rotation=35, ha="right")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Relaxed accuracy on paired sampled records")
    axes[-1].legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_no_image_sensitivity.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_results(EXPERIMENT_ID)
    if df.empty:
        raise SystemExit("No result rows found.")
    df.to_csv(OUT_DIR / "per_sample_results.csv", index=False)
    summary = summarize(df)
    summary.to_csv(OUT_DIR / "summary_by_model_dataset.csv", index=False)
    paired = paired_comparisons(df)
    paired.to_csv(OUT_DIR / "paired_model_comparisons.csv", index=False)
    scoring_sensitivity(summary).to_csv(OUT_DIR / "scoring_sensitivity.csv", index=False)
    failure_examples(df).to_csv(OUT_DIR / "failure_examples.csv", index=False)
    plot_accuracy(summary)
    plot_complete_accuracy(summary)
    plot_heatmap(summary)
    no_image_df = load_results(NO_IMAGE_EXPERIMENT_ID)
    if not no_image_df.empty:
        no_image_df.to_csv(OUT_DIR / "no_image_per_sample_results.csv", index=False)
        no_image_summary = summarize(no_image_df)
        no_image_summary.to_csv(OUT_DIR / "no_image_summary_by_model_dataset.csv", index=False)
        comparison = no_image_comparison(df, no_image_df)
        comparison.to_csv(OUT_DIR / "no_image_comparison.csv", index=False)
        plot_no_image_sensitivity(comparison)
    print(f"Loaded {len(df)} result rows")
    print(summary.to_string(index=False))
    if not no_image_df.empty:
        print(f"Loaded {len(no_image_df)} text-only sensitivity rows")


if __name__ == "__main__":
    main()
