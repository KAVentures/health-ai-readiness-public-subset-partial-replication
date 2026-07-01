#!/bin/bash

###############################################################################
# Sample Dataset Test Runner
# Tests 3 sample datasets (VQA-RAD, OmniVQA, PMC-VQA)
# Tests all available models
# 
# Usage:
#     ./run_test_sample_datasets.sh [model]
# 
# Examples:
#     ./run_test_sample_datasets.sh                    # Test all models
#     ./run_test_sample_datasets.sh gpt-4o             # Test specific model only
###############################################################################

echo "=========================================="
echo "Sample Datasets Test Suite - All Models"
echo "=========================================="

# Define sample datasets to test
DATASETS=(
    "VQA-RAD-sample"
    "OmniVQA-sample"
    "PMC-VQA-sample"
)

# Define all available models
ALL_MODELS=(
    "gpt-4o"
    "o3"
    "o4-mini"
    "gpt-5"
    "deepseek"
    "gemini-2.5-pro"
    "claude-sonnet-3.5"
    "qwen3-vl-thinking"
)

# Use provided model or test all models
if [ $# -eq 0 ]; then
    MODELS_TO_TEST=("${ALL_MODELS[@]}")
    echo "Testing ALL models: ${ALL_MODELS[*]}"
else
    MODELS_TO_TEST=("$1")
    echo "Testing single model: $1"
fi

# Test configuration
EXP_ID="test"
MODE="vqa"
REASON_LEVEL="zero"
MAX_CONCURRENT=2  # Lower concurrency for sample testing

# Sync models that need test_runner.py instead of test_runner_async.py
SYNC_MODELS=("gemini-2.5-pro" "claude-sonnet-3.5")

# Function to check if model requires sync runner
is_sync_model() {
    local model=$1
    for sm in "${SYNC_MODELS[@]}"; do
        if [ "$model" = "$sm" ]; then
            return 0
        fi
    done
    return 1
}

echo ""
echo "Experiment ID: $EXP_ID"
echo "Mode: $MODE"
echo "Reasoning level: $REASON_LEVEL"
echo "=========================================="
echo ""

# Test results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Main testing loop - iterate through models
for MODEL in "${MODELS_TO_TEST[@]}"; do
    echo ""
    echo "=========================================="
    echo "TESTING MODEL: $MODEL"
    echo "=========================================="
    
    # Choose appropriate runner
    if is_sync_model "$MODEL"; then
        RUNNER="src/test_runner.py"
        echo "Using SYNC runner for model: $MODEL"
    else
        RUNNER="src/test_runner_async.py"
        echo "Using ASYNC runner for model: $MODEL"
    fi
    
    # Test each dataset with this model
    for dataset in "${DATASETS[@]}"; do
        echo ""
        echo "----------------------------------------"
        echo "Model: $MODEL | Dataset: $dataset"
        echo "----------------------------------------"
        
        ((TOTAL_TESTS++))
        
        # Run test
        echo "Running: python $RUNNER $dataset $MODEL $EXP_ID $MODE $REASON_LEVEL $MAX_CONCURRENT"
        
        python "$RUNNER" \
            "$dataset" \
            "$MODEL" \
            "$EXP_ID" \
            "$MODE" \
            "$REASON_LEVEL" \
            "$MAX_CONCURRENT"
        
        exit_code=$?
        
        if [ $exit_code -ne 0 ]; then
            echo "FAILED: Test FAILED for $MODEL on $dataset (exit code: $exit_code)"
            ((FAILED_TESTS++))
        else
            echo "PASSED: Test PASSED for $MODEL on $dataset"
            ((PASSED_TESTS++))
        fi
        
        # Small delay between tests
        sleep 1
    done
    
    echo "=========================================="
    echo "Completed testing model: $MODEL"
    echo "=========================================="
done

# Summary
echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Total tests run: $TOTAL_TESTS"
echo "Passed: $PASSED_TESTS"
echo "Failed: $FAILED_TESTS"
echo ""
echo "Models tested: ${#MODELS_TO_TEST[@]}"
echo "Datasets per model: ${#DATASETS[@]}"
echo ""

if [ $FAILED_TESTS -gt 0 ]; then
    echo "Some tests failed!"
    echo ""
    echo "Check logs in: log/"
    echo "Check results in: result/"
    exit 1
else
    echo "All tests passed!"
    echo ""
    echo "Results saved in: result/"
    echo "Logs saved in: log/"
    exit 0
fi
