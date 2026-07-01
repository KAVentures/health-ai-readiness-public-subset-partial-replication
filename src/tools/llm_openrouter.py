"""OpenRouter LLM API utilities for unified access to multiple model families.

Runtime credential setup is intentionally omitted from this public results
package. The saved outputs and analysis can be inspected without credentials.
"""

import os
import base64
import logging
import time
import json
import requests
from typing import List, Optional, Dict, Tuple
from PIL import Image
from dotenv import load_dotenv

# Import common utilities
try:
    from .llm_utils import convert_to_supported_format
except ImportError:
    from llm_utils import convert_to_supported_format

# Load runtime environment variables when present.
load_dotenv()

from logging_utils import get_logger

logger = get_logger(__name__)

# ================================
# Runtime Provider Configuration
# ================================
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ================================
# Model Name Mappings
# ================================
OPENROUTER_MODEL_MAP = {
    # Claude models
    "claude-sonnet-3.5": "anthropic/claude-3.5-sonnet",
    "claude-3-opus": "anthropic/claude-3-opus",
    "claude-3-haiku": "anthropic/claude-3-haiku",
    
    # Qwen models
    "qwen3-vl-instruct": "qwen/qwen3-vl-235b-a22b-instruct",
    "qwen3-vl-thinking": "qwen/qwen3-vl-235b-a22b-thinking",
    "qwen3-vl-30B-thinking": "qwen/qwen3-vl-30b-a3b-thinking",
    "qwen3-vl-30B-instruct": "qwen/qwen3-vl-30b-a3b-instruct",
    
    # Gemini models
    "gemini-2.0-flash": "google/gemini-2.0-flash-exp:free",
    "gemini-2.5-pro": "google/gemini-2.5-pro-preview-06-05",
    
    # DeepSeek models
    "deepseek-vl2": "deepseek/deepseek-chat",
    "deepseek-chat": "deepseek/deepseek-chat",
    
    # OpenAI models (for reference)
    "gpt-4o": "openai/gpt-4o",
    "gpt-4-turbo": "openai/gpt-4-turbo",
    "gpt-4-vision": "openai/gpt-4-vision-preview",
}


def get_openrouter_model_name(model: str) -> str:
    """
    Convert local model name to OpenRouter model identifier.
    
    Args:
        model: Local model name or OpenRouter model name
        
    Returns:
        OpenRouter model identifier
    """
    # If already in OpenRouter format (contains '/'), return as is
    if '/' in model:
        return model
    
    # Try to map from our local names
    if model in OPENROUTER_MODEL_MAP:
        return OPENROUTER_MODEL_MAP[model]
    
    # If not found, return as is (might be a valid OpenRouter name)
    return model


def encode_image_base64(image_path: str, max_size_mb: int = 5) -> Tuple[Optional[str], Optional[str]]:
    """
    Encode image to base64, with automatic compression if too large.
    
    Args:
        image_path: Path to image file
        max_size_mb: Maximum size in MB before compression (default: 5)
        
    Returns:
        Tuple of (base64_string, format) or (None, None) on error
    """
    try:
        # Convert to supported format
        image_path, img_format = convert_to_supported_format(image_path)
        if not image_path:
            return None, None
        
        # Check if compression needed
        # if os.path.getsize(image_path) >= max_size_mb * 1024 * 1024:
        #     temp_dir = os.path.dirname(image_path) or "./temp_image"
        #     compressed_path = os.path.join(temp_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_compressed.jpg")
            
        #     with Image.open(image_path) as img:
        #         # Convert to RGB if needed
        #         if img.mode in ['RGBA', 'P']:
        #             img = img.convert('RGB')
                
        #         # Save with reasonable compression (quality=85 is a good balance)
        #         img.save(compressed_path, "JPEG", quality=85, optimize=True)
            
        #     image_path = compressed_path
        #     img_format = 'jpeg'
        #     logging.info(f"Compressed image to {os.path.getsize(image_path) / (1024 * 1024):.2f} MB")
        
        # Encode to base64
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('ascii'), img_format
        
    except Exception as e:
        logging.error(f"Error encoding image {image_path}: {e}")
        return None, None


def openrouter_vqa_sys(
    query: str,
    sys_prompt: str,
    image_paths: Optional[List[str]] = None,
    model: str = "anthropic/claude-3.5-sonnet",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    max_retries: int = 5
) -> tuple:
    """
    Query any OpenRouter-supported model with optional images.
    
    Args:
        query: The text query/question
        sys_prompt: System prompt for the model
        image_paths: Optional list of image file paths
        model: Model identifier (will be converted to OpenRouter format)
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        max_retries: Maximum retry attempts on failure
        
    Returns:
        Tuple of (response_text, token_usage_dict) where token_usage_dict contains:
            - prompt_tokens: Number of tokens in the prompt
            - completion_tokens: Number of tokens in the completion
            - total_tokens: Total tokens used
            - reasoning_tokens: Number of reasoning tokens (usually 0)
        
    Raises:
        ValueError: If provider credentials are unavailable
        Exception: On API errors after retries
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OpenRouter runtime credentials are unavailable")
    
    # Convert model name to OpenRouter format
    openrouter_model = get_openrouter_model_name(model)
    
    # Build messages
    messages = []
    
    # Add system message if provided
    if sys_prompt:
        messages.append({
            "role": "system",
            "content": sys_prompt
        })
    
    # Build user message content
    user_content = []
    
    # Add images if provided
    if image_paths:
        for img_path in image_paths:
            if not os.path.exists(img_path):
                logging.warning(f"Image not found: {img_path}")
                continue
            
            # Encode image to base64
            base64_image, img_format = encode_image_base64(img_path)
            if base64_image:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{img_format};base64,{base64_image}"
                    }
                })
    
    # Add text query
    user_content.append({
        "type": "text",
        "text": query
    })
    
    messages.append({
        "role": "user",
        "content": user_content
    })
    
    # Prepare request
    url = f"{OPENROUTER_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/your-repo",  # Optional: for ranking
        "X-Title": "Medical VQA System"  # Optional: for ranking
    }
    
    payload = {
        "model": openrouter_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    # Retry loop
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            
            # Check for errors in response
            if "error" in result:
                error_msg = result["error"].get("message", str(result["error"]))
                logging.warning(f"OpenRouter API error (attempt {attempt + 1}/{max_retries}): {error_msg}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt * 5
                    logging.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"OpenRouter API error: {error_msg}")
            
            # Extract response text and token usage
            if "choices" in result and len(result["choices"]) > 0:
                response_text = result["choices"][0]["message"]["content"]
                
                # Extract token usage
                usage = result.get("usage", {})
                token_usage = {
                    'prompt_tokens': usage.get('prompt_tokens', 0),
                    'completion_tokens': usage.get('completion_tokens', 0),
                    'total_tokens': usage.get('total_tokens', 0),
                    'reasoning_tokens': 0
                }
                
                return response_text, token_usage
            else:
                raise Exception(f"Unexpected response format: {result}")
                
        except requests.exceptions.Timeout:
            logging.warning(f"Request timeout (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt * 5)
            else:
                raise
                
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt * 10)
            else:
                raise
                
        except Exception as e:
            logging.error(f"Error calling OpenRouter API (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt * 5)
            else:
                raise


# ================================
# Convenience wrapper functions
# ================================

def Claude_vqa_sys(
    query: str,
    sys_prompt: str,
    image_paths: Optional[List[str]] = None,
    model: str = "claude-3-5-sonnet-20241022",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    max_retries: int = 5
) -> tuple:
    """
    Query Claude models via OpenRouter (compatible with original interface).
    Returns: (response_text, token_usage_dict)
    """
    return openrouter_vqa_sys(query, sys_prompt, image_paths, model, max_tokens, temperature, max_retries)


def Qwen_vqa_sys(
    query: str,
    sys_prompt: str,
    image_paths: Optional[List[str]] = None,
    model: str = "qwen-vl-max",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    max_retries: int = 5
) -> tuple:
    """
    Query Qwen models via OpenRouter (compatible with original interface).
    Returns: (response_text, token_usage_dict)
    """
    return openrouter_vqa_sys(query, sys_prompt, image_paths, model, max_tokens, temperature, max_retries)


def Gemini_vqa_sys(
    query: str,
    sys_prompt: str,
    image_paths: Optional[List[str]] = None,
    model: str = "gemini-2.0-flash",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    max_retries: int = 5
) -> tuple:
    """
    Query Gemini models via OpenRouter (compatible with original interface).
    Returns: (response_text, token_usage_dict)
    """
    if "think step by step" in query.lower():
        max_tokens = 4096  # Increase token limit for reasoning
    else:
        max_tokens = 2024

    return openrouter_vqa_sys(query, sys_prompt, image_paths, model, max_tokens, temperature, max_retries)


def Deepseek_vqa_sys(
    query: str,
    sys_prompt: str,
    image_paths: Optional[List[str]] = None,
    model: str = "deepseek-vl2",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    max_retries: int = 5
) -> tuple:
    """
    Query DeepSeek models via OpenRouter (compatible with original interface).
    Returns: (response_text, token_usage_dict)
    """
    return openrouter_vqa_sys(query, sys_prompt, image_paths, model, max_tokens, temperature, max_retries)


# ================================
# Utility Functions
# ================================

def list_available_models() -> List[str]:
    """
    List all models available through OpenRouter.
    
    Returns:
        List of model identifiers
    """
    if not OPENROUTER_API_KEY:
        logging.warning("OPENROUTER_API_KEY not set")
        return list(OPENROUTER_MODEL_MAP.values())
    
    try:
        url = f"{OPENROUTER_BASE_URL}/models"
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        models_data = response.json()
        return [model["id"] for model in models_data.get("data", [])]
        
    except Exception as e:
        logging.error(f"Error fetching models list: {e}")
        return list(OPENROUTER_MODEL_MAP.values())


def get_model_info(model: str) -> Optional[Dict]:
    """
    Get information about a specific model.
    
    Args:
        model: Model identifier
        
    Returns:
        Model information dictionary or None
    """
    if not OPENROUTER_API_KEY:
        return None
    
    try:
        # First, try to get the model from the list of all models
        openrouter_model = get_openrouter_model_name(model)
        url = f"{OPENROUTER_BASE_URL}/models"
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        models_data = response.json()
        # Find the specific model in the list
        for model_info in models_data.get("data", []):
            if model_info.get("id") == openrouter_model:
                return model_info
        
        # If not found in list, return None
        logging.warning(f"Model {openrouter_model} not found in available models")
        return None
        
    except Exception as e:
        logging.error(f"Error fetching model info: {e}")
        return None


if __name__ == "__main__":
    print("Testing OpenRouter unified API...")
    print(f"Provider credentials available: {bool(OPENROUTER_API_KEY)}")
    
    # Test query
    test_query = "What is shown in this medical image?"
    test_sys_prompt = "You are a helpful medical assistant."
    test_image = "/path/to/test/image.jpg"
    
    # Example 1: List available models
    print("\nFetching available models...")
    models = list_available_models()
    print(f"Total models available: {len(models)}")
    print("Sample models:", models[:5])
    
    # Example 2: Test different models (uncomment to run)
    # response = Claude_vqa_sys(test_query, test_sys_prompt, [test_image])
    # print("\nClaude response:", response)
    
    # response = Qwen_vqa_sys(test_query, test_sys_prompt, [test_image])
    # print("\nQwen response:", response)
    
    # response = Gemini_vqa_sys(test_query, test_sys_prompt, [test_image])
    # print("\nGemini response:", response)
    
    # response = Deepseek_vqa_sys(test_query, test_sys_prompt, [test_image])
    # print("\nDeepSeek response:", response)
    
    print("\nOpenRouter API module loaded successfully!")
