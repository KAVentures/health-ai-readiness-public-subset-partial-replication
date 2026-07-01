#!/usr/bin/env python3
"""
Data Setup Verification Tool

Checks that all required data files and images are correctly set up.

Usage:
    python data/verify_data_setup.py
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# ── Colour helpers ──────────────────────────────────────────────────────────

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):    print(f"  {GREEN}✓{RESET} {msg}")
def warn(msg):  print(f"  {YELLOW}⚠{RESET} {msg}")
def fail(msg):  print(f"  {RED}✗{RESET} {msg}")
def header(msg): print(f"\n{BOLD}{msg}{RESET}")

# ── Helpers ─────────────────────────────────────────────────────────────────

def count_images(dir_path: Path) -> int:
    if not dir_path.exists():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    return sum(1 for f in dir_path.rglob("*") if f.suffix.lower() in exts)


def check_json(path: Path, min_records: int = 0) -> tuple[bool, int]:
    if not path.exists():
        return False, 0
    try:
        with open(path) as f:
            data = json.load(f)
        n = len(data) if isinstance(data, list) else 0
        return n >= min_records, n
    except Exception:
        return False, 0


# ── Checks ──────────────────────────────────────────────────────────────────

CORE_DATASETS = {
    "JAMA":       {"json": "JAMA/jama_processed.json",             "image_dir": "JAMA/image_proc",              "min": 1300},
    "NEJM":       {"json": "NEJM_image_challenge/test.json",       "image_dir": "NEJM_image_challenge/images",  "min": 900},
    "VQA-RAD":    {"json": "VQA_RAD/test.json",                    "image_dir": "VQA_RAD/images",               "min": 400},
    "OmniMedVQA": {"json": "OmniMedVQA/sampled_vqa_500.json",     "image_dir": "OmniMedVQA/Images",            "min": 500},
    "PMC-VQA":    {"json": "PMC-VQA/test_2_500_sample.json",      "image_dir": "PMC-VQA/figures",              "min": 400},
}

STRESS_TEST_JSONS = [
    "NEJM_image_challenge/test_no_context_v0.json",
    "NEJM_image_challenge/test_reformat_0_replace_ans_reorder_v1.json",
    "NEJM_image_challenge/test_reformat_1_replace_ans_reorder_v2.json",
    "NEJM_image_challenge/test_reformat_2_replace_ans_reorder_v3.json",
    "NEJM_image_challenge/test_reformat_3_replace_ans_reorder_v4.json",
    "NEJM_image_challenge/test_reformat_4_replace_ans_reorder_v5.json",
    "NEJM_image_challenge/test_reformat_replace_unknown_v6.json",
    "NEJM_image_challenge/test_Image_aug_v7.json",
    "NEJM_image_challenge/test_Image_aug_base_v8.json",
    "NEJM_image_challenge/test_sample_120.json",
]

PROPRIETARY = {
    "NEJM_testing": {"dir": "NEJM_image_challenge/NEJM_testing", "expected": 40},
    "test_unseen":  {"json": "test_unseen/refined_mcq_selected_final_v10.json", "image_dir": "test_unseen/image", "expected": 80},
}


def main():
    print(f"{BOLD}{'='*55}")
    print("Health-AI-Readiness — Data Setup Verification")
    print(f"{'='*55}{RESET}")

    # ── Core Datasets ───────────────────────────────────────────────────
    header("Core Benchmark Datasets")
    for name, s in CORE_DATASETS.items():
        valid, n = check_json(DATA_DIR / s["json"], s["min"])
        n_img = count_images(DATA_DIR / s["image_dir"])
        json_status = f"{n} records" if valid else ("NOT FOUND" if n == 0 else f"{n} records (< {s['min']})")
        img_status = f"{n_img} images" if n_img > 0 else "empty"
        (ok if valid else fail)(f"{name:12s}  json: {json_status}")
        (ok if n_img > 0 else warn)(f"{'':12s}  images: {img_status}  ({s['image_dir']}/)")

    # ── Stress Tests ────────────────────────────────────────────────────
    header("Stress Test Variants")
    for rel in STRESS_TEST_JSONS:
        (ok if (DATA_DIR / rel).exists() else fail)(rel)

    # ── Proprietary ─────────────────────────────────────────────────────
    header("Proprietary Test Data")
    sp = PROPRIETARY["NEJM_testing"]
    n = count_images(DATA_DIR / sp["dir"])
    (ok if n >= sp["expected"] else warn)(f"NEJM_testing   {n}/{sp['expected']} images")

    sp = PROPRIETARY["test_unseen"]
    valid, n_rec = check_json(DATA_DIR / sp["json"])
    n_img = count_images(DATA_DIR / sp["image_dir"])
    (ok if valid else warn)(f"test_unseen    json: {n_rec} records" if valid else f"test_unseen    json: not found")
    (ok if n_img >= sp["expected"] else warn)(f"{'':14s} images: {n_img}/{sp['expected']}")

    print()


if __name__ == "__main__":
    main()
