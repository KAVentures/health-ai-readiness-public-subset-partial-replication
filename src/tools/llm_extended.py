"""Additional LLM API utilities used by the public-subset benchmark runner.

Runtime credential setup is intentionally omitted from this public results
package. The saved outputs and analysis can be inspected without credentials.
"""

import os
import base64
import time
import json
import requests
from typing import List, Optional
from PIL import Image
from dotenv import load_dotenv

# Third-party API imports
from google import genai
from google.genai import types as genai_types
import anthropic
from openai import OpenAI

# Import common utilities
try:
    from .llm_utils import convert_to_supported_format
except ImportError:
    from llm_utils import convert_to_supported_format

# Optional imports
try:
    import dashscope
    from dashscope import MultiModalConversation
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False

# Load runtime environment variables when present.
load_dotenv()

from logging_utils import get_logger

logger = get_logger(__name__)



# ================================
# Runtime Provider Configuration
# ================================
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
SILICONFLOW_DEEPSEEK_API_KEY = os.environ.get("SILICONFLOW_DEEPSEEK_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
XAI_API_KEY = os.environ.get("XAI_API_KEY", "")

# Try to import OpenRouter module
try:
    from .llm_openrouter import openrouter_vqa_sys
    OPENROUTER_AVAILABLE = True
except ImportError:
    try:
        from llm_openrouter import openrouter_vqa_sys
        OPENROUTER_AVAILABLE = True
    except ImportError:
        OPENROUTER_AVAILABLE = False
        logger.info("OpenRouter module not found. Using direct provider clients.")


# ================================
# Unified Interface Functions
# ================================
# These functions use OpenRouter when configured, otherwise they fall back to
# direct provider implementations.



def Claude_vqa_sys(
    query: str, 
    sys_prompt: str, 
    image_paths: Optional[List[str]] = None, 
    model: str = "claude-3-5-sonnet",
    api_type: str = "openrounter_key",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    max_retries: int = 5
) -> tuple:
    """
    Query Claude models with optional images.
    
    Automatically uses OpenRouter if OPENROUTER_API_KEY is set,
    otherwise falls back to direct Anthropic API.
    
    Args:
        query: The text query/question
        sys_prompt: System prompt for the model
        image_paths: Optional list of image file paths
        model: Claude model name
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature (0.0-1.0)
        
    Returns:
        Tuple of (response_text, token_usage_dict)
    """
    # Use OpenRouter if available
    if api_type == "openrouter_key" and OPENROUTER_API_KEY and OPENROUTER_AVAILABLE:
        logger.info("Using OpenRouter for Claude")
        return openrouter_vqa_sys(query, sys_prompt, image_paths, model, max_tokens, temperature, max_retries)
        # Fall back to direct Anthropic API
    logger.info("Using direct Anthropic API for Claude")
    return _claude_direct_api(query, sys_prompt, image_paths, model, max_tokens, temperature, max_retries)


def _claude_direct_api(
    query: str, 
    sys_prompt: str, 
    image_paths: Optional[List[str]] = None, 
    model: str = "claude-3-5-sonnet",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    max_retries: int = 5
) -> tuple:
    """
    Direct Claude API implementation (original version).
    Returns: (response_text, token_usage_dict)
    """
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
    
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    messages_content = []
    
    # Add images if provided
    if image_paths:
        for img_path in image_paths:
            if not os.path.exists(img_path):
                logger.warning(f"Image not found: {img_path}")
                continue

            with Image.open(img_path) as img:
                detected_format = (img.format or "").lower()
            if detected_format in {"jpeg", "jpg", "png", "gif", "webp"}:
                image_path_for_upload = img_path
                img_format = "jpeg" if detected_format == "jpg" else detected_format
            else:
                image_path_for_upload, img_format = convert_to_supported_format(img_path)
                if not image_path_for_upload:
                    logger.warning(f"Unsupported image format: {img_path}")
                    continue
            media_type_map = {
                'jpg': 'image/jpeg',
                'jpeg': 'image/jpeg',
                'png': 'image/png',
                'gif': 'image/gif',
                'webp': 'image/webp',
            }
            media_type = media_type_map.get(str(img_format).lower(), 'image/jpeg')

            with open(image_path_for_upload, 'rb') as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            messages_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_data
                }
            })
    
    # Add text query
    messages_content.append({
        "type": "text",
        "text": query
    })
    
    # Retry logic with exponential backoff
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "system": sys_prompt if sys_prompt else "You are a helpful medical assistant.",
                "messages": [{"role": "user", "content": messages_content}],
            }
            if os.getenv("ANTHROPIC_REASONING_EFFORT", "high"):
                kwargs["thinking"] = {"type": "adaptive", "display": "omitted"}
            else:
                kwargs["temperature"] = temperature
            message = client.messages.create(**kwargs)

            text_blocks = [
                block.text
                for block in message.content
                if getattr(block, "type", None) == "text" and hasattr(block, "text")
            ]
            response = "\n".join(text_blocks).strip()
            if not response and "thinking" in kwargs:
                # Some high-thinking Anthropic responses can exhaust the response
                # budget before emitting a text block. Retry once without thinking
                # so the sample receives an answer instead of an API-format error.
                logger.warning("Claude returned no text block with thinking enabled; retrying once without thinking")
                fallback_kwargs = dict(kwargs)
                fallback_kwargs.pop("thinking", None)
                message = client.messages.create(**fallback_kwargs)
                text_blocks = [
                    block.text
                    for block in message.content
                    if getattr(block, "type", None) == "text" and hasattr(block, "text")
                ]
                response = "\n".join(text_blocks).strip()

            usage = message.usage
            token_usage = {
                'prompt_tokens': usage.input_tokens,
                'completion_tokens': usage.output_tokens,
                'total_tokens': usage.input_tokens + usage.output_tokens,
                'reasoning_tokens': 0
            }
            if not response:
                raise ValueError("Claude response did not contain a text block")
            return response, token_usage
            
        except (anthropic.RateLimitError, anthropic.APITimeoutError) as e:
            wait_time = min(2 ** attempt * 10, 300)
            logger.warning(f"Claude error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(wait_time)
            else:
                raise
                
        except Exception as e:
            error_str = str(e)
            is_rate_limit = '429' in error_str or 'Too Many Requests' in error_str
            wait_time = 2 ** attempt * (10 if is_rate_limit else 5)
            
            logger.error(f"Claude API error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(min(wait_time, 300))
            else:
                raise


def Qwen_vqa_sys(
    query: str,
    sys_prompt: str,
    image_paths: Optional[List[str]] = None,
    model: str = "qwen-vl-max",
    api_type: str = "openrouter_key",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    max_retries: int = 5
) -> tuple:
    """
    Query Qwen VL models with optional images.
    
    Automatically uses OpenRouter if OPENROUTER_API_KEY is set,
    otherwise falls back to direct DashScope API.
    
    Args:
        query: The text query/question
        sys_prompt: System prompt for the model
        image_paths: Optional list of image file paths
        model: Qwen model name (qwen-vl-max, qwen-vl-plus)
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature (0.0-1.0)
        
    Returns:
        Tuple of (response_text, token_usage_dict)
    """
    # Use OpenRouter if available
    if api_type == "openrouter_key" and OPENROUTER_API_KEY and OPENROUTER_AVAILABLE:
        logger.info("Using OpenRouter for Qwen")
        return openrouter_vqa_sys(query, sys_prompt, image_paths, model, max_tokens, temperature, max_retries)
    
    # Fall back to direct DashScope API
    logger.info("Using direct DashScope API for Qwen")
    return _qwen_direct_api(query, sys_prompt, image_paths, model, max_tokens, temperature, max_retries)


def _qwen_direct_api(
    query: str,
    sys_prompt: str,
    image_paths: Optional[List[str]] = None,
    model: str = "qwen-vl-max",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    max_retries: int = 5
) -> tuple:
    """
    Direct Qwen API implementation (original version).
    Returns: (response_text, token_usage_dict)
    """
    if not DASHSCOPE_API_KEY:
        raise ValueError("DASHSCOPE_API_KEY environment variable is not set")
    
    # Initialize Qwen client using OpenAI-compatible interface
    client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    messages_content = []
    
    if image_paths:
        for img_path in image_paths:
            if os.path.exists(img_path):
                messages_content.append({
                        "type": "image_url", 
                        "image_url": {"url": f"file://{img_path}"}
                    })
            else:
                logger.warning(f"Image not found: {img_path}")
    
    messages_content.append({"type": "text", "text": query})
    
    # Retry logic with exponential backoff
    for attempt in range(max_retries):
        try:
            # Create completion
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": sys_prompt if sys_prompt else "You are a helpful medical assistant."
                    },
                    {
                        "role": "user",
                        "content": messages_content
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=60.0
            )
            
            # Extract token usage
            usage = completion.usage
            token_usage = {
                'prompt_tokens': usage.prompt_tokens,
                'completion_tokens': usage.completion_tokens,
                'total_tokens': usage.total_tokens,
                'reasoning_tokens': 0
            }
            
            response = completion.choices[0].message.content
            return response, token_usage
            
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = '429' in error_str or 'too many requests' in error_str or 'rate limit' in error_str
            is_timeout = 'timeout' in error_str
            wait_time = min(2 ** attempt * (10 if is_rate_limit else 1 if is_timeout else 5), 300)
            
            logger.warning(f"Qwen API error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(wait_time)
            else:
                raise


# Alternative Qwen implementation using native DashScope SDK
def Qwen_vqa_sys_native(
    query: str,
    sys_prompt: str,
    image_paths: Optional[List[str]] = None,
    model: str = "qwen-vl-max"
) -> str:
    """
    Query Qwen VL models using native DashScope SDK.
    
    Requires: pip install dashscope
    """
    if not DASHSCOPE_AVAILABLE:
        raise ImportError("Please install dashscope: pip install dashscope")
    
    if not DASHSCOPE_API_KEY:
        raise ValueError("DASHSCOPE_API_KEY environment variable is not set")
    
    try:
        # Configure the SDK from the runtime environment.
        dashscope.api_key = DASHSCOPE_API_KEY
        
        # Prepare messages
        content = []
        
        # Add images
        if image_paths:
            for img_path in image_paths:
                if not os.path.exists(img_path):
                    logger.warning(f"Image not found: {img_path}")
                    continue
                content.append({"image": f"file://{img_path}"})
        
        # Add text
        full_query = f"{sys_prompt}\n\n{query}" if sys_prompt else query
        content.append({"text": full_query})
        
        # Call API
        messages = [{"role": "user", "content": content}]
        
        response = MultiModalConversation.call(
            model=model,
            messages=messages
        )
        
        if response.status_code == 200:
            return response.output.choices[0].message.content[0]["text"]
        else:
            raise Exception(f"Qwen API error: {response.code} - {response.message}")
            
    except Exception as e:
        logger.error(f'Qwen native API error: {e}')
        raise


def Gemini_vqa_sys(
    query: str,
    sys_prompt: str,
    image_paths: list,
    model: str = "gemini-2.0-flash",
    api_type: str = "openrouter_key",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    max_retries: int = 5
) -> tuple:
    """
    Query Google Gemini models with optional images.
    
    Automatically uses OpenRouter if OPENROUTER_API_KEY is set,
    otherwise falls back to direct Google API.
    
    Args:
        query: The text query/question
        sys_prompt: System prompt for the model
        image_paths: List of image file paths
        model: Gemini model name (default: gemini-2.0-flash)
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature 0.0-1.0 (default: 0.0)
    
    Returns:
        Tuple of (response_text, token_usage_dict)
    
    Requires:
        pip install google-generativeai pillow
    """
    
    if "think step by step" in query.lower():
        max_tokens = 4096  # Increase token limit for reasoning
    else:
        max_tokens = 2024
    # Use OpenRouter if available
    if api_type == "openrouter_key" and OPENROUTER_API_KEY and OPENROUTER_AVAILABLE:
        logger.info("Using OpenRouter for Gemini")
        return openrouter_vqa_sys(query, sys_prompt, image_paths, model, max_tokens, temperature, max_retries)
    
    # Fall back to direct Google API
    logger.info("Using direct Google API for Gemini")
    return _gemini_direct_api(query, sys_prompt, image_paths, model, max_tokens, temperature, max_retries)


def _gemini_direct_api(
    query: str, 
    sys_prompt: str, 
    image_paths: list, 
    model: str = "gemini-2.0-flash", 
    max_tokens: int = 1024,
    temperature: float = 0.0,
    max_retries: int = 5
) -> tuple:
    """
    Direct Gemini API implementation using google-genai package.
    Returns: (response_text, token_usage_dict)
    """
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    
    # Create client using new API
    client = genai.Client(api_key=GOOGLE_API_KEY)
    
    # Prepare config
    thinking_level = os.getenv("GOOGLE_REASONING_EFFORT", "HIGH").upper()
    config = genai_types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        thinking_config=genai_types.ThinkingConfig(thinking_level=thinking_level),
    )
    if sys_prompt:
        config.system_instruction = sys_prompt
    
    # Prepare prompt parts
    prompt_parts = [query]
    if image_paths:
        for img_path in image_paths:
            try:
                new_image_path, _ = convert_to_supported_format(img_path)
                prompt_parts.append(Image.open(new_image_path))
            except Exception as e:
                logger.warning(f"Error loading image {img_path}: {e}")
    
    # Retry logic with exponential backoff
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt_parts,
                config=config
            )
            
            empty_token_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'reasoning_tokens': 0}
            
            if response.parts:
                usage = response.usage_metadata
                token_usage = {
                    'prompt_tokens': usage.prompt_token_count if usage.prompt_token_count else 0,
                    'completion_tokens': usage.candidates_token_count if usage.candidates_token_count else 0,
                    'total_tokens': usage.total_token_count if usage.total_token_count else 0,
                    'reasoning_tokens': 0
                }
                
                return response.text, token_usage
            else:
                return f"Error: Empty response. Feedback: {response.prompt_feedback}", empty_token_usage
                
        except Exception as e:
            error_str = str(e).lower()                
            is_rate_limit = '429' in error_str or 'too many requests' in error_str or 'resource_exhausted' in error_str
            is_timeout = 'timeout' in error_str
            wait_time = min(2 ** attempt * (10 if is_rate_limit else 1 if is_timeout else 5), 300)
            
            logger.warning(f"Gemini API error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(wait_time)
            else:
                raise


def Deepseek_vqa_sys(
    query: str,
    sys_prompt: str,
    image_paths: list,
    model: str = "deepseek-ai/deepseek-vl2",
    api_type: str = "siliconflow_key",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    max_retries: int = 5
) -> tuple:
    """
    Query DeepSeek VL models with optional images.
    
    Automatically uses OpenRouter if OPENROUTER_API_KEY is set,
    otherwise falls back to direct SiliconFlow API.
    
    Args:
        query: The text query/question
        sys_prompt: System prompt for the model
        image_paths: List of image file paths
        model: DeepSeek model name (default: deepseek-ai/deepseek-vl2)
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature 0.0-1.0 (default: 0.0)
    
    Returns:
        Tuple of (response_text, token_usage_dict)
    """
    # Check for runtime provider configuration.
    if api_type == "siliconflow_key" and SILICONFLOW_DEEPSEEK_API_KEY:
        logger.info("Using SiliconFlow DeepSeek API")
        return _deepseek_direct_api(
            query, sys_prompt, image_paths, model,
            max_tokens=max_tokens, temperature=temperature, max_retries=max_retries
        )
    else:
        raise ValueError(
            "SILICONFLOW_DEEPSEEK_API_KEY environment variable is not set"
        )


def _deepseek_direct_api(
    query: str,
    sys_prompt: str,
    image_paths: list,
    model: str = "deepseek-ai/deepseek-vl2",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    max_retries: int = 5
) -> tuple:
    """
    Direct DeepSeek API implementation via SiliconFlow (original version).
    Returns: (response_text, token_usage_dict)
    """
    
    url = "https://api.siliconflow.cn/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    max_size_mb = 5
    MAX_SIZE_BYTES = max_size_mb * 1024 * 1024
    image_blocks = []
    
    if image_paths:
        for img_path in image_paths:
            try:
                with Image.open(img_path) as img:
                    img.verify()
                
                new_image_path, img_format = convert_to_supported_format(img_path)
                
                # Check the size
                resized_new_image_path = new_image_path
                original_size = os.path.getsize(new_image_path)
                
                if original_size >= MAX_SIZE_BYTES:
                    try:
                        img = Image.open(new_image_path)
                        if img.mode == 'RGBA':
                            img = img.convert('RGB')
                        elif img.mode == 'P':
                            img = img.convert('RGB')
                        
                        base_name, _ = os.path.splitext(os.path.basename(new_image_path))
                        resized_new_image_path = os.path.join(os.path.dirname(new_image_path), f"{base_name}_compressed.jpg")
                        
                        quality = 90
                        while quality >= 10:
                            img.save(resized_new_image_path, "JPEG", quality=quality, optimize=True)
                            current_size = os.path.getsize(resized_new_image_path)
                            
                            print(f" try quality = {quality}，current size: {current_size / (1024 * 1024):.2f} MB")
                            
                            if current_size <= MAX_SIZE_BYTES:
                                print(f"compresed to {max_size_mb} MB, save to '{resized_new_image_path}'")
                                break
                            quality -= 5
                    except Exception as e:
                        print(f"resize error: {e}")
                        return None
                
                with open(resized_new_image_path, 'rb') as f:
                    img_bytes = f.read()
                base64_image = base64.b64encode(img_bytes).decode('ascii')
                
                image_blocks.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{img_format};base64,{base64_image}"
                    }
                })
            except Exception as e:
                print(f'{img_path} error!')
                logger.info(f'{img_path} error!')
    
    messages = []
    if sys_prompt:
        messages.append({
            "role": "system",
            "content": sys_prompt
        })
    
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": query
            }
        ] + image_blocks
    })
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            
            # Handle rate limit
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                logger.warning(f"Rate limit (429). Retry after {retry_after}s (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(retry_after)
                    continue
                else:
                    raise Exception(f"Rate limit exceeded after {max_retries} attempts")
            
            response.raise_for_status()
            result = response.json()
            
            usage = result.get("usage", {})
            token_usage = {
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0),
                'reasoning_tokens': 0
            }
            
            response_text = result["choices"][0]["message"]["content"]
            return response_text, token_usage
            
        except (requests.exceptions.HTTPError, requests.exceptions.Timeout, KeyError) as e:
            logger.warning(f"DeepSeek API error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(min(2 ** attempt * 5, 300))
            else:
                raise


def XAI_vqa_sys(
    query: str,
    sys_prompt: str,
    image_paths: Optional[List[str]] = None,
    model: str = "grok-4.3",
    api_type: str = "xai_key",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    max_retries: int = 5,
) -> tuple:
    """Query xAI Grok through its OpenAI-compatible Responses API."""
    if not XAI_API_KEY:
        raise ValueError("XAI_API_KEY environment variable is not set")

    client = OpenAI(
        api_key=XAI_API_KEY,
        base_url="https://api.x.ai/v1",
        timeout=float(os.getenv("API_TIMEOUT_SECONDS", "90")),
    )
    content = [{"type": "input_text", "text": query}]
    if image_paths:
        for img_path in image_paths:
            try:
                new_image_path, img_format = convert_to_supported_format(img_path)
                with open(new_image_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                content.append({
                    "type": "input_image",
                    "image_url": f"data:image/{img_format};base64,{b64}",
                })
            except Exception as e:
                logger.warning(f"Error loading image {img_path}: {e}")

    input_messages = []
    if sys_prompt:
        input_messages.append({
            "role": "system",
            "content": [{"type": "input_text", "text": sys_prompt}],
        })
    input_messages.append({"role": "user", "content": content})

    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=model,
                input=input_messages,
                reasoning={"effort": os.getenv("XAI_REASONING_EFFORT", "high")},
                text={"verbosity": "low"},
            )
            usage = getattr(resp, "usage", None)
            token_usage = {
                "prompt_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
                "completion_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
                "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
                "reasoning_tokens": 0,
            }
            return resp.output_text, token_usage
        except Exception as e:
            logger.warning(f"xAI API error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(min(2 ** attempt * 5, 300))
            else:
                raise


if __name__ == "__main__":
    # Example usage
    print("Testing Claude, Qwen, Gemini, and DeepSeek API functions...")
    
    # Test query
    test_query = "What is shown in this medical image?"
    test_sys_prompt = "You are a helpful medical assistant."
    test_image = "/path/to/test/image.jpg"
    
    # Uncomment to test:
    # response = Claude_vqa_sys(test_query, test_sys_prompt, [test_image])
    # print("Claude response:", response)
    
    # response = Qwen_vqa_sys(test_query, test_sys_prompt, [test_image])
    # print("Qwen response:", response)
