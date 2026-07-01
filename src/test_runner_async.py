"""
Async Test Runner for Medical VQA Datasets

High-performance async version with concurrent processing.
Uses the same modular structure as test_runner.py but with asyncio.

Usage:
    python test_runner_async.py <dataset> <model> <exp_id> <mode> <reasoning> [max_concurrent]
    
Arguments:
    dataset: JAMA | NEJM
    model: gpt-4o | o3 | o4-mini | gpt-5 | deepseek | gemini-2.0-flash | etc.
    exp_id: Experiment identifier (e.g., t0, t1, t2)
    mode: vqa (with images) | text (text only)
    reasoning: zero | low | medium | high
    max_concurrent: Optional, max concurrent requests (default: 8)

Examples:
    python test_runner_async.py JAMA gpt-4o t0 vqa zero
    python test_runner_async.py JAMA gpt-5 t0 vqa zero
    python test_runner_async.py NEJM o3 t1 vqa high 16
    
    python test_runner_async.py JAMA  t0 vqa zero
"""

import os
import sys
import random
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple
from tqdm import tqdm
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
)

os.environ["MKL_THREADING_LAYER"] = "GNU"

# ==================================================
# Configuration
# ==================================================
# Get project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENT_DATE = "Jan11"

result_lock = asyncio.Lock()

# Global state
TOTAL_TOKEN_USAGE: Dict = {}
RESULT_DICT: Dict = {}


# ==================================================
# Argument Parsing
# ==================================================

def parse_arguments():
    """Parse and validate command line arguments."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "JAMA"
    model_key = sys.argv[2] if len(sys.argv) > 2 else "gpt-4o"
    exp_id = sys.argv[3] if len(sys.argv) > 3 else "t0"
    mode = sys.argv[4] if len(sys.argv) > 4 else "vqa"
    reasoning = sys.argv[5] if len(sys.argv) > 5 else "zero"
    max_concurrent = int(sys.argv[6]) if len(sys.argv) > 6 else 4
    
    # Validate dataset and model
    try:
        dataset_config = get_dataset_config(dataset_name)
        model_config = get_model_config(model_key)
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    return dataset_name, model_key, exp_id, mode, reasoning, max_concurrent, dataset_config, model_config


# ==================================================
# Async Helpers
# ==================================================

async def call_model_with_retry(
    query: str,
    system_prompt: str,
    image_paths: List[str],
    model_config: Dict,
    mode: str,
    reasoning: str,
    executor: ThreadPoolExecutor,
    max_retries: int = 5,
) -> Tuple[str, Dict]:

    logger = get_logger(__name__)

    for attempt in range(max_retries):
        try:
            loop = asyncio.get_event_loop()
            response, token_usage = await loop.run_in_executor(
                executor,
                call_model_api,
                query,
                system_prompt,
                image_paths,
                model_config["full_name"],
                model_config["api_function"],
                model_config.get("api_type"),
                mode,
                reasoning,
            )
            return response, token_usage
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt + random.random()
            await asyncio.sleep(wait_time)

def log_result(logger, result: Dict) -> None:
    """
    Log per-sample result.
    """
    logger.info(
        f"[{result['index']}] Q: {result['query'][:200]}... "
        f"| images={len(result.get('image_paths', []))}"
    )

    if result.get("success", True):
        logger.info(f"[{result['index']}] Pred: {result['response'][:200]}...")
        logger.info(f"[{result['index']}] GT: {result['ground_truth']}")

        usage = result.get("token_usage", {})
        if usage:
            logger.info(
                f"[{result['index']}] Tokens: "
                f"P={usage.get('prompt_tokens', 0)} "
                f"C={usage.get('completion_tokens', 0)} "
                f"R={usage.get('reasoning_tokens', 0)} "
                f"T={usage.get('total_tokens', 0)}"
            )
    else:
        logger.error(f"[{result['index']}] ERROR: {result.get('error', 'Unknown error')}")


# async def log_result(result: Dict) -> None:
#     """Log result with async-safe locking."""
#     async with log_lock:
#         logger.info(f"Q: {result["query"][:200]}..., Images: {result.get("image_paths", [])}")
        
#         if result["success"]:
#             logger.info(f"Prediction: {result["response"][:200]}...")
#             logger.info(f"Ground Truth: {result["ground_truth"]}")
#             if result["token_usage"]:
#                 usage = result["token_usage"]
#                 logger.info(f"Token usage - Prompt: {usage.get("prompt_tokens", 0)}, "
#                            f"Completion: {usage.get("completion_tokens", 0)}, "
#                            f"Reasoning: {usage.get("reasoning_tokens", 0)}, "
#                            f"Total: {usage.get("total_tokens", 0)}")
#         else:
#             logger.error(f"[{idx}] Error: {result.get("error", "Unknown error")}")

async def update_token_usage(token_usage: Dict) -> None:
    """Update global token usage with thread-safe locking."""
    if token_usage:
        async with result_lock:
            TOTAL_TOKEN_USAGE["prompt_tokens"] += token_usage.get("prompt_tokens", 0)
            TOTAL_TOKEN_USAGE["completion_tokens"] += token_usage.get("completion_tokens", 0)
            TOTAL_TOKEN_USAGE["reasoning_tokens"] += token_usage.get("reasoning_tokens", 0)
            TOTAL_TOKEN_USAGE["total_tokens"] += token_usage.get("total_tokens", 0)


async def save_intermediate_results(result_filename: str, metadata: Dict, completed_count: int) -> None:
    """Save intermediate results with async-safe locking."""
    async with result_lock:
        save_results(result_filename, RESULT_DICT, TOTAL_TOKEN_USAGE, metadata)


# ==================================================
# Async Processing
# ==================================================
async def process_sample_async(
    idx: int,
    sample: Dict,
    system_prompt: str,
    query_builder,
    dataset_config: Dict,
    model_config: Dict,
    mode: str,
    reasoning: str,
    executor: ThreadPoolExecutor,
    semaphore: asyncio.Semaphore,
) -> Dict:

    logger = get_logger(__name__)
    async with semaphore:
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
            
            # Call model with retry
            response, token_usage = await call_model_with_retry(
                query,
                system_prompt,
                image_paths,
                model_config,
                mode,
                reasoning,
                executor,
            )
            
            # Get ground truth
            ground_truth = sample.get(dataset_config["answer_key"], "N/A")
            
            return {
                "index": idx,
                "query": query,
                "response": response,
                "ground_truth": ground_truth,
                "token_usage": token_usage if token_usage else {},
                "image_paths": image_paths,
                "success": True,
            }
            
        except Exception as e:
            logger.exception(f"[{idx}] Sample failed")

            return {
                "index": idx,
                "query": query if "query" in locals() else "",
                "response": f"Error: {str(e)}",
                "ground_truth": sample.get(dataset_config["answer_key"], "N/A"),
                "token_usage": {},
                "image_paths": image_paths if "image_paths" in locals() else [],
                "success": False,
                "error": str(e),
            }


# ==================================================
# Main Async
# ==================================================

async def main_async():
    global TOTAL_TOKEN_USAGE, RESULT_DICT
    
    # Parse arguments
    dataset_name, model_key, exp_id, mode, reasoning, max_concurrent, dataset_config, model_config = parse_arguments()
    # ---------- run name ----------
    run_name = (
        f"{dataset_name}_{model_config['short_name']}_"
        f"{EXPERIMENT_DATE}_{mode}_{reasoning}_{exp_id}"
    )

    # ---------- logger ----------
    log_dir = PROJECT_ROOT / "log" / dataset_name / model_config['short_name']
    setup_per_run_logging(log_dir=log_dir, run_name=run_name)
    install_exception_hook()

    logger = get_logger(__name__)

    logger.info(
        f"Start | Dataset={dataset_name} | "
        f"Model={model_config['full_name']} | "
        f"Mode={mode} | Reasoning={reasoning} | "
        f"ExpID={exp_id} | MaxConcurrent={max_concurrent}"
    )

    # ---------- paths ----------
    result_dir = PROJECT_ROOT / "result" / dataset_name / model_config['short_name']
    result_filename = result_dir / f"{run_name}.json"

    # ---------- load data ----------
    test_set = load_dataset(dataset_config["data_path"])

    RESULT_DICT, token_restore = load_existing_results(result_filename)
    TOTAL_TOKEN_USAGE = initialize_token_usage()
    if token_restore:
        TOTAL_TOKEN_USAGE.update(token_restore)
    
    # Get prompt templates
    system_prompt = SYSTEM_PROMPTS[dataset_config["system_prompt"]]
    query_builder = QUERY_BUILDERS[dataset_config["query_builder"]]
    
    # Prepare tasks
    save_interval = dataset_config["save_interval"]
    semaphore = asyncio.Semaphore(max_concurrent)
    samples_to_process = [
        (idx, sample) for idx, sample in enumerate(test_set)
        if str(idx) not in RESULT_DICT
    ]
    
    logger.info(f"Processing {len(samples_to_process)} samples (skipping {len(RESULT_DICT)} completed)")
    
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        # Create all async tasks
        tasks = [
            process_sample_async(
                idx,
                sample,
                system_prompt,
                query_builder,
                dataset_config,
                model_config,
                mode,
                reasoning,
                executor,
                semaphore,
            )
            for idx, sample in samples_to_process
        ]
        
        # Process with real-time progress updates
        completed_count = 0
        for coro in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=f"{model_config['short_name']}",
        ):
            try:
                result = await coro
                
                if isinstance(result, Exception):
                    logger.error(f"Unhandled exception: {result}")
                    continue
                
                idx = result["index"]
                
                # Log, store, and update
                log_result(logger, result)
                
                async with result_lock:
                    RESULT_DICT[str(idx)] = {
                        "index": idx,
                        "query": result["query"],
                        "response": result["response"],
                        "ground_truth": result["ground_truth"],
                        "token_usage": result["token_usage"],
                    }
                
                await update_token_usage(result["token_usage"])
                
                # Console output
                print(f"\n############## Sample {idx} ################")
                print(f"Question: {result['query'][:100]}...")
                print(f"Prediction: {result['response'][:100]}...")
                print(f"Ground Truth: {result['ground_truth']}")
                
                completed_count += 1
                
                # Periodic save
                if completed_count % save_interval == 0:
                    metadata = {
                        "dataset": dataset_name,
                        "model": model_config["full_name"],
                        "mode": mode,
                        "reasoning": reasoning,
                        "experiment": exp_id,
                        "total_samples": len(RESULT_DICT),
                        "status": "intermediate"
                    }
                    await save_intermediate_results(result_filename, metadata, completed_count)
                    
            except Exception as e:
                logger.error(f"Error processing task: {e}")
                continue
    
    # Final save
    metadata = {
        "dataset": dataset_name,
        "model": model_config["full_name"],
        "mode": mode,
        "reasoning": reasoning,
        "experiment": exp_id,
        "total_samples": len(RESULT_DICT),
        "status": "complete",
    }
    save_results(result_filename, RESULT_DICT, TOTAL_TOKEN_USAGE, metadata)
    
    # Summary
    logger.info(
        f"✓ Done | samples={len(RESULT_DICT)} | "
        f"P={TOTAL_TOKEN_USAGE['prompt_tokens']:,} "
        f"C={TOTAL_TOKEN_USAGE['completion_tokens']:,} "
        f"T={TOTAL_TOKEN_USAGE['total_tokens']:,}"
    )


# ==================================================
# Entry Point
# ==================================================

if __name__ == "__main__":
    
    logger = get_logger(__name__)
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception:
        logger.exception("Fatal error")
        raise
