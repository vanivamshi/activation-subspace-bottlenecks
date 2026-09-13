#!/bin/bash

# Script to run all steering tests with the same parameters
# Usage: ./run_all_steering_tests.sh

cd /home/vamshi && source new-env/bin/activate && cd /home/HDD/ATAF/vamshi/LLM_paper/post_hoc_extention_1

echo "=========================================="
echo "Running Steering Tests for All Variants"
echo "=========================================="
echo ""

# Create logs directory if it doesn't exist
mkdir -p experiment_logs

# Function to run a test
run_test() {
    local script_name=$1
    local log_file="experiment_logs/${script_name}_run.log"
    
    echo "=========================================="
    echo "Running: $script_name"
    echo "Log file: $log_file"
    echo "=========================================="
    echo ""
    
    python "$script_name" \
        --trained_model ./models/mamba_trained_on_pile \
        --use_custom_prompts \
        --use_dataset_prompts \
        --dataset_file experiment_logs/classified_dataset_questions.json \
        --max_per_category 100 \
        2>&1 | tee "$log_file"
    
    local exit_code=${PIPESTATUS[0]}
    
    echo ""
    echo "=========================================="
    if [ $exit_code -eq 0 ]; then
        echo "✅ $script_name completed successfully"
    else
        echo "❌ $script_name failed with exit code $exit_code"
    fi
    echo "=========================================="
    echo ""
    
    return $exit_code
}

# Run all tests sequentially
run_test "steering_s6_types_densemamba.py"
run_test "steering_s6_types_hyena.py"
run_test "steering_s6_types_mamba2.py"
run_test "steering_s6_types_miniplm.py"

echo "=========================================="
echo "All tests completed!"
echo "Check experiment_logs/ for individual log files"
echo "=========================================="


