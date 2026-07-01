"""
Core Utilities for Medical VQA Testing

Shared utilities for loading data, saving results, logger, and API calls.
"""

import os
import json
# import logger
from typing import Dict, List, Tuple, Optional, Any

from configs.model_config import REASONING_EFFORT_MAP
from tools.llm_openai import ChatGPT_VQA
from tools.llm_extended import (
    Claude_vqa_sys,
    Qwen_vqa_sys,
    Gemini_vqa_sys,
    Deepseek_vqa_sys,
    XAI_vqa_sys,
)

from logging_utils import get_logger


logger = get_logger(__name__)
# ==================================================
# Data Loading
# ==================================================

def load_dataset(data_path: str) -> List[Dict]:
    """
    Load dataset from JSON file.
    
    Args:
        data_path: Path to dataset JSON file
    
    Returns:
        List of dataset samples
    
    Raises:
        FileNotFoundError: If file doesn't exist
        JSONDecodeError: If file is not valid JSON
    """
    try:
        with open(data_path, "r") as f:
            dataset = json.load(f)
        logger.info(f"✓ Loaded {len(dataset)} samples from {data_path}")
        return dataset

    except FileNotFoundError:
        logger.error(f"✗ File not found: {data_path}")
        raise

    except json.JSONDecodeError as e:
        logger.error(f"✗ Invalid JSON in {data_path}: {e}")
        raise


def load_existing_results(filepath: str) -> Tuple[Dict, Dict]:
    """
    Load existing results if available.
    
    Args:
        filepath: Path to results JSON file
    
    Returns:
        Tuple of (results_dict, token_usage_dict)
    """
    if not os.path.exists(filepath):
        return {}, {}

    try:
        if os.path.getsize(filepath) == 0:
            logger.warning(f"File {filepath} is empty, starting fresh")
            return {}, {}

        with open(filepath, "r") as file:
            data = json.load(file)

        if "results" in data:
            results = data["results"]
            logger.info(f"✓ Loaded {len(results)} existing results")

            if "token_usage_summary" in data:
                token_usage = data["token_usage_summary"]
            else:
                token_usage = initialize_token_usage()
                for item in results.values():
                    update_token_usage(token_usage, item.get("token_usage"))

            return results, token_usage

        logger.info(f"✓ Loaded {len(data)} existing results (legacy format)")
        return data, {}

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON: {e}. Starting fresh.")
        return {}, {}

    except Exception:
        logger.exception("Unexpected error while loading existing results")
        return {}, {}


# ==================================================
# Result Saving
# ==================================================

def save_results(
    filepath: str,
    results: Dict,
    token_usage: Dict,
    metadata: Dict,
    indent: int = 4,
) -> None:
    """
    Save results to JSON file with sorted keys.
    """
    sorted_results = dict(sorted(results.items(), key=lambda x: int(x[0])))

    output_data = {
        "results": sorted_results,
        "token_usage_summary": token_usage,
        "metadata": metadata,
    }

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(output_data, f, indent=indent)

    logger.info(f"✓ Results saved to {filepath}")


# ==================================================
# Model API Interface
# ==================================================

def call_model_api(
    query: str,
    sys_prompt: str,
    image_paths: Optional[List[str]],
    model_name: str,
    api_function: str,
    api_type: Optional[str] = None,
    mode: str = "vqa",
    reasoning: str = "zero",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    max_retries: int = 5,
) -> Tuple[str, Optional[Dict]]:
    """
    Unified function to call any model API.
    """
    if mode == "text":
        image_paths = None
    # Single image for models that don't support multiple
    # elif api_function in ['qwen', 'deepseek', 'claude'] and image_paths:
    elif api_function in ['qwen', 'deepseek'] and image_paths:
        image_paths = image_paths[:1]

    try:
        if api_function == "gpt":
            reasoning_effort = REASONING_EFFORT_MAP.get(reasoning, "medium")
            return ChatGPT_VQA(
                query=query,
                sys_prompt=sys_prompt,
                api_type=api_type,
                model=model_name,
                image_paths=image_paths,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                max_retries=max_retries,
            )

        if api_function == "qwen":
            return Qwen_vqa_sys(
                query=query,
                sys_prompt=sys_prompt,
                api_type=api_type,
                image_paths=image_paths,
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                max_retries=max_retries,
            )

        if api_function == "deepseek":
            return Deepseek_vqa_sys(
                query=query,
                sys_prompt=sys_prompt,
                api_type=api_type,
                image_paths=image_paths,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
            )

        if api_function == "xai":
            return XAI_vqa_sys(
                query=query,
                sys_prompt=sys_prompt,
                api_type=api_type,
                image_paths=image_paths,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
            )

        if api_function == "gemini":
            return Gemini_vqa_sys(
                query=query,
                sys_prompt=sys_prompt,
                api_type=api_type,
                image_paths=image_paths,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
            )

        if api_function == "claude":
            return Claude_vqa_sys(
                query=query,
                sys_prompt=sys_prompt,
                api_type=api_type,
                image_paths=image_paths,
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                max_retries=max_retries,
            )

        raise ValueError(f"Unknown API function: {api_function}")

    except Exception:
        logger.exception(
            f"Model API call failed "
            f"(api_function={api_function}, model={model_name})"
        )
        raise


# ==================================================
# Image Path Utilities
# ==================================================

def get_image_paths(
    sample: Dict,
    image_base_path: str,
    image_key: str,
    supports_multiple: bool,
) -> List[str]:
    """
    Extract and format image paths from sample.
    """
    # Special handling for VQA-RAD dataset (image_folder + image)
    if image_key == 'image_path_full' and 'image_folder' in sample and 'image' in sample:
        return [f"{image_base_path}{sample['image_folder']}/{sample['image']}"]

    if supports_multiple and isinstance(sample[image_key], list):
        return [f"{image_base_path}{p}" for p in sample[image_key]]

    return [f"{image_base_path}{sample[image_key]}"]


# ==================================================
# Token Usage Tracking
# ==================================================

def initialize_token_usage() -> Dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
    }


def update_token_usage(total_usage: Dict, new_usage: Optional[Dict]) -> None:
    if not new_usage:
        return

    total_usage["prompt_tokens"] += new_usage.get("prompt_tokens", 0)
    total_usage["completion_tokens"] += new_usage.get("completion_tokens", 0)
    total_usage["reasoning_tokens"] += new_usage.get("reasoning_tokens", 0)
    total_usage["total_tokens"] += new_usage.get("total_tokens", 0)
