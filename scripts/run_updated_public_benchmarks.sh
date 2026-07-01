#!/bin/bash
set -u

###############################################################################
# Updated public benchmark runner
#
# Runs the public, repo-bundled benchmark subsets that do not require restricted
# JAMA/NEJM image assets. The VQA-RAD target uses the original repo's 100-case
# no-CoT evaluation subset rather than the full 451-case VQA-RAD test file.
#
# Usage:
#   ./scripts/run_updated_public_benchmarks.sh
#   ./scripts/run_updated_public_benchmarks.sh gpt-5.5
#
# Optional environment:
#   EXP_ID=public_original_subset_v1 REASON_LEVEL=high ./scripts/run_updated_public_benchmarks.sh
###############################################################################

DATASETS=(
    "VQA-RAD-no-cot"
    "PMC-VQA"
    "OmniVQA"
)

ALL_MODELS=(
    "gpt-5.5"
    "grok-4.3"
    "claude-opus-4.8"
    "gemini-3.5-flash"
)

if [ $# -eq 0 ]; then
    MODELS_TO_TEST=("${ALL_MODELS[@]}")
else
    MODELS_TO_TEST=("$1")
fi

EXP_ID="${EXP_ID:-public_original_subset_v1}"
MODE="${MODE:-vqa}"
REASON_LEVEL="${REASON_LEVEL:-high}"
PYTHON_BIN="${PYTHON:-python3}"

TOTAL=0
FAILED=0

echo "Updated public benchmark run"
echo "Models: ${MODELS_TO_TEST[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Reasoning: ${REASON_LEVEL}"
echo

for MODEL in "${MODELS_TO_TEST[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        TOTAL=$((TOTAL + 1))
        echo "Running ${MODEL} on ${DATASET}"
        "$PYTHON_BIN" src/test_runner.py \
            "$DATASET" \
            "$MODEL" \
            "$EXP_ID" \
            "$MODE" \
            "$REASON_LEVEL"
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 130 ]; then
            echo "Interrupted; aborting benchmark run."
            exit 130
        fi
        if [ $EXIT_CODE -ne 0 ]; then
            FAILED=$((FAILED + 1))
            echo "FAILED: ${MODEL} on ${DATASET}"
        else
            echo "PASSED: ${MODEL} on ${DATASET}"
        fi
        echo
    done
done

echo "Finished ${TOTAL} benchmark runs; failures: ${FAILED}"
if [ $FAILED -ne 0 ]; then
    exit 1
fi
