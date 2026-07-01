#!/bin/bash


# Async test script for VQA-RAD dataset
# Tests multiple models with both VQA and text settings
# Supports high-performance async processing with configurable concurrency

echo "=========================================="
echo "VQA-RAD-no-cot Async Test Suite"
echo "=========================================="

# Define dataset
DATASET="VQA-RAD-no-cot"

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

# Define reasoning models and sync models
SYNC_MODELS=("gemini-2.5-pro" "claude-sonnet-3.5")
REASONING_MODELS=("o3" "o4-mini" "gpt-5")

# Define text flags
# TEXT_FLAGS=("vqa" "text")
TEXT_FLAGS=("vqa")

# Default reasoning level
REASON_LEVEL="zero"

# Max concurrent requests
MAX_CONCURRENT=4

# Number of runs per experiment
NUM_RUNS=5

# Function to check if model is a reasoning model
is_reasoning_model() {
    local model=$1
    for rm in "${REASONING_MODELS[@]}"; do
        if [ "$model" = "$rm" ]; then
            return 0
        fi
    done
    return 1
}

is_sync_model() {
    local model=$1
    for sm in "${SYNC_MODELS[@]}"; do
        if [ "$model" = "$sm" ]; then
            return 0
        fi
    done
    return 1
}

# Main testing loop
for model in "${MODELS_TO_TEST[@]}"; do
    echo ""
    echo "=========================================="
    echo "Testing model: $model"
    echo "=========================================="
    
    # Determine reason level
    if is_reasoning_model "$model"; then
        REASON_LEVEL="zero"
        echo "Using reasoning level: $REASON_LEVEL (standard model)"
    fi

    # Choose runner script
    if is_sync_model "$model"; then
        RUNNER="src/test_runner.py"
        echo "🧠 Using SYNC runner: $RUNNER"
    else
        RUNNER="src/test_runner_async.py"
        echo "⚡ Using ASYNC runner: $RUNNER"
    fi
    
    for text_flag in "${TEXT_FLAGS[@]}"; do
        echo ""
        echo "Testing with mode: $text_flag"
        echo "----------------------------------------"
        
        for run in $(seq 0 $((NUM_RUNS - 1))); do
            exp_id="t${run}"
            echo ""
            echo "Run ${run}/${NUM_RUNS}: $model - $text_flag - $exp_id"

            python "$RUNNER" \
                "$DATASET" \
                "$model" \
                "$exp_id" \
                "$text_flag" \
                "$REASON_LEVEL" \
                "$MAX_CONCURRENT"
            
            # Run the test with test_runner_async.py
            # python test_runner_async.py "$DATASET" "$model" "$exp_id" "$text_flag" "$REASON_LEVEL" "$MAX_CONCURRENT"
            # python test_runner.py "$DATASET" "$model" "$exp_id" "$text_flag" "$REASON_LEVEL" "$MAX_CONCURRENT"
            
            exit_code=$?
            if [ $exit_code -ne 0 ]; then
                echo "⚠️  Warning: Test failed with exit code $exit_code"
                echo "Continuing with next experiment..."
            else
                echo "✅ Test completed successfully"
            fi
            
            # Small delay between runs
            sleep 2
        done
    done
    
    echo ""
    echo "✅ Completed all tests for $model"
    echo "=========================================="
done

echo ""
echo "=========================================="
echo "All tests completed!"
echo "=========================================="
echo ""
echo "Results saved in: result/$DATASET/"
echo "Logs saved in: log/$DATASET/"
