"""
Unified Test Runner for Medical VQA Datasets

A clean, modular test script that:
- Loads dataset configuration from configs/dataset_config.py
- Loads model configuration from configs/model_config.py
- Loads prompts from prompts.py
- Uses shared utilities from core_utils.py
- Reads experiment settings from command line

Usage:
    python test_runner.py <dataset> <model> <exp_id> <mode> <reasoning>
    
Arguments:
    dataset: JAMA | NEJM
    model: gpt-4o | o3 | o4-mini | gpt-5 | deepseek | gemini-2.0-flash | 
           gemini-2.5-pro | gemini-2.5-flash | claude-sonnet-3.5 | 
           qwen-vl-max | qwen-vl-plus
    exp_id: Experiment identifier (e.g., t0, t1, t2)
    mode: vqa (with images) | text (text only)
    reasoning: zero | low | medium | high

Examples:
    python test_runner.py JAMA gpt-4o t0 vqa zero
    python test_runner.py NEJM o3 t1 vqa high
    python test_runner.py JAMA gemini-2.0-flash t0 text zero
"""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from configs.dataset_config import get_dataset_config
from configs.model_config import get_model_config
from prompts import SYSTEM_PROMPTS, QUERY_BUILDERS

from logging_utils import (
    setup_per_run_logging,
    install_exception_hook,
    get_logger,
)

from core_utils import (
    load_dataset,
    load_existing_results,
    save_results,
    call_model_api,
    get_image_paths,
    initialize_token_usage,
    update_token_usage,
)

os.environ["MKL_THREADING_LAYER"] = "GNU"


# ==================================================
# Configuration
# ==================================================

# Get project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent

EXPERIMENT_DATE = "Jan10"


# ==================================================
# Argument Parsing
# ==================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run a configured medical VQA benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset", nargs="?", default="JAMA")
    parser.add_argument("model", nargs="?", default="gpt-4o")
    parser.add_argument("exp_id", nargs="?", default="t0")
    parser.add_argument("mode", nargs="?", default="vqa", choices=["vqa", "text"])
    parser.add_argument("reasoning", nargs="?", default="zero")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional number of dataset records to evaluate after deterministic sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1618,
        help="Seed used when --sample is set.",
    )
    args = parser.parse_args()

    dataset_name = args.dataset
    model_key = args.model
    exp_id = args.exp_id
    mode = args.mode
    reasoning = args.reasoning
    
    # Validate dataset and model
    try:
        dataset_config = get_dataset_config(dataset_name)
        model_config = get_model_config(model_key)
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    return dataset_name, model_key, exp_id, mode, reasoning, dataset_config, model_config, args.sample, args.seed


# ==================================================
# Main Processing
# ==================================================

def main():
    """Main processing function."""
    # Parse arguments
    dataset_name, model_key, exp_id, mode, reasoning, dataset_config, model_config, sample_n, seed = parse_arguments()
    
    # Setup paths (absolute paths to project root)
    log_dir = PROJECT_ROOT / "log" / dataset_name / model_config['short_name']
    result_dir = PROJECT_ROOT / "result" / dataset_name / model_config['short_name']

    run_name = (
        f"{dataset_name}_{model_config['short_name']}_"
        f"{EXPERIMENT_DATE}_{mode}_{reasoning}_{exp_id}"
    )

    # ---------- logger ----------
    log_path = setup_per_run_logging(
        log_dir=log_dir,
        run_name=run_name,
    )
    install_exception_hook()

    logger = get_logger(__name__)

    logger.info("Starting experiment")
    logger.info(
        f"Dataset={dataset_name} | "
        f"Model={model_config['full_name']} | "
        f"Mode={mode} | Reasoning={reasoning} | "
        f"ExpID={exp_id} | Sample={sample_n or 'all'} | Seed={seed}"
    )

    # ---------- result file ----------
    result_filename = result_dir / f"{run_name}.json"

    # ---------- load dataset ----------
    test_set = load_dataset(dataset_config["data_path"])
    selected_indices = list(range(len(test_set)))
    if sample_n is not None:
        if sample_n <= 0:
            raise ValueError("--sample must be positive when provided")
        rng = random.Random(seed)
        selected_indices = sorted(rng.sample(selected_indices, min(sample_n, len(selected_indices))))
        logger.info(f"Selected {len(selected_indices)} deterministic sample indices with seed={seed}")

    # ---------- restore results ----------
    result_dict, token_usage_restore = load_existing_results(result_filename)
    total_token_usage = initialize_token_usage()
    if token_usage_restore:
        total_token_usage.update(token_usage_restore)
    
    # Get prompt templates
    system_prompt = SYSTEM_PROMPTS[dataset_config["system_prompt"]]
    query_builder = QUERY_BUILDERS[dataset_config["query_builder"]]
    
    # Processing loop
    save_interval = dataset_config["save_interval"]   
    # =================================
    # Main loop
    # =================================
    for idx in selected_indices:
        sample = test_set[idx]
        print(f"\n################ Sample {idx} ################\n")
        
        # Skip if already processed
        if str(idx) in result_dict:
            logger.info(f"Sample {idx} already processed, skipping...")
            continue
        
        try:
            # Build query
            query = query_builder(sample)
            
            # Get image paths
            image_paths = get_image_paths(
                sample,
                dataset_config["image_base_path"],
                dataset_config["image_key"],
                dataset_config["supports_multiple_images"],
            )
            
            # Log request
            # logger.info(f"[{idx}] Q: {query[:100]}... | Images: {len(image_paths)}")
            logger.info(f"[{idx}] Q: {query}... | Images: {len(image_paths)}")
            
            # Call model
            response, token_usage = call_model_api(
                query=query,
                sys_prompt=system_prompt,
                image_paths=image_paths,
                model_name=model_config["full_name"],
                api_function=model_config["api_function"],
                api_type=model_config.get("api_type"),
                mode=mode,
                reasoning=reasoning,
            )
            
            # Get ground truth
            answer_key = dataset_config["answer_key"]
            ground_truth = sample.get(answer_key, "N/A")
            
            # Store result
            result_dict[str(idx)] = {
                "index": idx,
                "query": query,
                "response": response,
                "ground_truth": ground_truth,
                "token_usage": token_usage if token_usage else {}
            }
            
            # Update token usage
            update_token_usage(total_token_usage, token_usage)
            
            # Log result
            tokens = f"(tokens: {token_usage.get('total_tokens', 0)})" if token_usage else ""
            # logger.info(f"[{idx}] Pred: {response[:100]}... | GT: {ground_truth} {tokens}")
            logger.info(f"[{idx}] Pred: {response}... | GT: {ground_truth} {tokens}")
            
            # Console output
            # print(f"Prediction: {response[:100]}...")
            print(f"Prediction: {response}...")
            print(f"Ground Truth: {ground_truth}")
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            result_dict[str(idx)] = {
                "index": idx,
                "query": query if "query" in locals() else "",
                "response": f"Error: {str(e)}",
                "ground_truth": sample.get(dataset_config["answer_key"], "N/A"),
                "token_usage": {}
            }
        
        
        # Periodic save
        if (idx + 1) % save_interval == 0:
            metadata = {
                "dataset": dataset_name,
                "model": model_config["full_name"],
                "mode": mode,
                "reasoning": reasoning,
                "experiment": exp_id,
                "total_samples": len(result_dict),
                "dataset_size": len(test_set),
                "selected_sample_count": len(selected_indices),
                "sample_seed": seed if sample_n is not None else None,
                "status": "intermediate"
            }
            save_results(result_filename, result_dict, total_token_usage, metadata)
            logger.info(f"Checkpoint: {idx+1} samples | Tokens: {total_token_usage['total_tokens']:,}")
    
    # Final save
    metadata = {
        "dataset": dataset_name,
        "model": model_config["full_name"],
        "mode": mode,
        "reasoning": reasoning,
        "experiment": exp_id,
        "total_samples": len(result_dict),
        "dataset_size": len(test_set),
        "selected_sample_count": len(selected_indices),
        "sample_seed": seed if sample_n is not None else None,
        "status": "complete"
    }
    save_results(result_filename, result_dict, total_token_usage, metadata)
    
    # Summary
    logger.info(f"✓ Complete: {len(result_dict)} samples | "
                 f"Tokens: P={total_token_usage['prompt_tokens']:,} C={total_token_usage['completion_tokens']:,} "
                 f"T={total_token_usage['total_tokens']:,}")


# ==================================================
# Entry Point
# ==================================================

if __name__ == "__main__":
    
    logger = get_logger(__name__)
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
