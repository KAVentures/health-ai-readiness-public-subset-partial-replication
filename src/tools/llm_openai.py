import os
import time
import json
import requests
import base64
import re
import sys
from PIL import Image

from openai import AzureOpenAI, OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.ai.inference import ChatCompletionsClient
from azure.identity import DefaultAzureCredential, ChainedTokenCredential, AzureCliCredential
from dotenv import load_dotenv

# Import common utilities
from .llm_utils import convert_to_supported_format

# Load runtime environment variables when present.
load_dotenv()


from logging_utils import get_logger

logger = get_logger(__name__)

# Runtime credential setup is intentionally omitted from this public results
# package. The saved outputs and analysis can be inspected without credentials.


def build_multimodal_message(text: str, image_paths: list[str]):
    content = [{"type": "text", "text": text}]
    if image_paths:
        for img_path in image_paths:
            try:
                with Image.open(img_path) as img:
                    img.verify()
                
                new_image_path, img_format = convert_to_supported_format(img_path)
                with open(new_image_path, 'rb') as f:
                    img_bytes = f.read()
                base64_image = base64.b64encode(img_bytes).decode('ascii')
                
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{img_format};base64,{base64_image}"
                    }
                })
            except Exception as e:
                print(f"⚠️  Warning: Failed to load image {img_path}: {e}")
                logger.info(f"⚠️  Warning: Failed to load image {img_path}: {e}")
                
    return {"role": "user", "content": content}

def ChatGPT_VQA(
    query: str,
    sys_prompt: str = "You are an AI assistant that helps people find information.",
    api_type: str = "azure",
    model: str = "gpt-4o",
    image_paths: list = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    reasoning_effort: str = "medium",
    max_retries: int = 5
) -> tuple:
    """
    Unified ChatGPT function supporting both Azure OpenAI (token-based) and OpenAI (key-based) APIs.
    Supports vision models (GPT-4o) and reasoning models (o3, o4-mini, gpt-5) with images.
    
    Returns:
        tuple: (response_text, token_usage_dict) where token_usage_dict contains:
            - prompt_tokens: Number of tokens in the prompt
            - completion_tokens: Number of tokens in the completion
            - reasoning_tokens: Number of reasoning tokens (for reasoning models)
            - total_tokens: Total tokens used
        
    Examples:
        # Using GPT-4o with images (Azure)
        response, usage = ChatGPT_VQA("What's in this image?", api_type="azure_token", 
                          model="gpt-4o", image_paths=["image.jpg"])
        
        # Using o3 reasoning model (Azure)
        response, usage = ChatGPT_VQA("Solve this problem", api_type="azure_key", 
                          model="o3", reasoning_effort="high")
        
        # Using standard OpenAI-compatible access
        response, usage = ChatGPT_VQA("What is AI?", api_type="openai", model="gpt-4o")
    """
    
    api_type = api_type.lower()
    
    if api_type not in ["azure_token", "azure_key", "openai", "openai_key"]:
        raise ValueError(
            f"Invalid api_type '{api_type}'. Must be 'azure_token', "
            "'azure_key', 'openai', or 'openai_key'"
        )
    
    # Determine if this is a reasoning model
    reasoning_models = ["o3", "o4-mini", "gpt-5", "gpt-5.5"]
    is_reasoning_model = model in reasoning_models
    
    if api_type == "azure_token":
        # Use Azure OpenAI with token-based authentication
        return _chat_azure_token(query, sys_prompt, model, image_paths, 
                                temperature, top_p, max_tokens, reasoning_effort, 
                                is_reasoning_model, max_retries)
    elif api_type == "azure_key":
        return _chat_azure_key(query, sys_prompt, model, image_paths,
                               temperature, top_p, max_tokens, reasoning_effort,
                               is_reasoning_model, max_retries)
    else:
        # Use standard OpenAI with key-based authentication
        return _chat_openai_key(query, sys_prompt, model, image_paths,
                               temperature, top_p, max_tokens, reasoning_effort,
                               is_reasoning_model, max_retries)


def _chat_azure_token(
    query: str,
    sys_prompt: str,
    model: str,
    image_paths: list,
    temperature: float,
    top_p: float,
    max_tokens: int,
    reasoning_effort: str,
    is_reasoning_model: bool,
    max_retries: int
) -> tuple:
    """
    Internal function for Azure OpenAI with token-based authentication.
    Supports both vision models (with images) and reasoning models (o3, o4, gpt-5).
    Returns: (response_text, token_usage_dict)
    """
    credential = ChainedTokenCredential(
        AzureCliCredential(),
        DefaultAzureCredential(
            exclude_cli_credential=True,
            exclude_environment_credential=True,
            exclude_shared_token_cache_credential=True,
            exclude_developer_cli_credential=True,
            exclude_powershell_credential=True,
            exclude_interactive_browser_credential=True,
            exclude_visual_studio_code_credentials=True,
            managed_identity_client_id=os.environ.get("DEFAULT_IDENTITY_CLIENT_ID"),
        )
    )
    
    api_version = '2025-01-01-preview'
    
    # Model version mapping
    model_config = {
        "gpt-4": ("gpt-4", "turbo-2024-04-09"),
        "gpt-4o": ("gpt-4o", "2024-08-06"),
        "gpt-4.1": ("gpt-4.1", "2025-04-14"),
        "o3": ("o3", "2025-04-16"),
        "o4-mini": ("o4-mini", "2025-04-16"),
        "gpt-5": ("gpt-5", "2025-08-07"),
    }
    
    if model not in model_config:
        raise ValueError(f"Unsupported model '{model}' for Azure token authentication")
    
    model_name, model_version = model_config[model]
    scopes = ["api://trapi/.default"]
    deployment_name = re.sub(r'[^a-zA-Z0-9-_.]', '', f'{model_name}_{model_version}')
    
    instance = 'gcr/shared'
    endpoint = f'https://trapi.research.microsoft.com/{instance}'
    # For reasoning models, use AzureOpenAI client
    token_provider = get_bearer_token_provider(credential, "api://trapi/.default")
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version=api_version,
    )
    
    # Build message content
    user_msg = build_multimodal_message(query, image_paths)
    messages = [
        {"role": "system", "content": sys_prompt},
        user_msg,
    ]
    
    for attempt in range(max_retries):
        try:
            if is_reasoning_model:
                # Use reasoning_effort instead of temperature
                response = client.chat.completions.create(
                    model=deployment_name,
                    messages=messages,
                    reasoning_effort=reasoning_effort,
                    stream=False,
                )
            else:
                # Standard chat completion
                response = client.chat.completions.create(
                    model=deployment_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stream=False,
                )
            
            time.sleep(3)
            content = response.choices[0].message.content
            # Extract token usage
            usage = response.usage
            token_usage = {
                'prompt_tokens': usage.prompt_tokens,
                'completion_tokens': usage.completion_tokens,
                'total_tokens': usage.total_tokens,
                'reasoning_tokens': 0
            }
            
            # For 'o' series models (o1, o3, o4), extract reasoning tokens
            if hasattr(usage, 'completion_tokens_details') and hasattr(usage.completion_tokens_details, 'reasoning_tokens'):
                token_usage['reasoning_tokens'] = usage.completion_tokens_details.reasoning_tokens or 0
            
               
            return response.choices[0].message.content, token_usage            
        except Exception as e:
            error_str = str(e)
            if 'content_filter' in error_str:
                return 'content_filter', None           
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"⚠️  Azure API error: {e} — Retrying in {wait_time} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"❌ Failed after {max_retries} attempts. Last error: {e}")
                logger.info(f"❌ Failed after {max_retries} attempts. Last error: {e}")
                raise

def _chat_azure_key(
    query: str,
    sys_prompt: str,
    model: str,
    image_paths: list,
    temperature: float,
    top_p: float,
    max_tokens: int,
    reasoning_effort: str,
    is_reasoning_model: bool,
    max_retries: int
) -> tuple:
    """
    Internal function for Azure OpenAI with key-based authentication.
    Supports both vision models (with images) and reasoning models (o3, o4, gpt-5).
    Returns: (response_text, token_usage_dict)
    """
    # Model deployment mapping (deployment names in Azure)
    model_deployment_map = {
        "gpt-4o": "GPT4O",
        "o3": "O3",
        "o4-mini": "O4MINI",
        "gpt-5": "GPT5",
    }
    
    deployment_name = model_deployment_map.get(model, model)
    AZURE_OPENAI_KEY = os.environ.get(f"AZURE_OPENAI_{deployment_name}_KEY", "")
    AZURE_OPENAI_ENDPOINT = os.environ.get(f"AZURE_OPENAI_{deployment_name}_ENDPOINT", "" )
    
    if not AZURE_OPENAI_KEY:
        raise ValueError("AZURE_OPENAI_GPT4O_KEY environment variable is not set")
    if not AZURE_OPENAI_ENDPOINT:
        raise ValueError("AZURE_OPENAI_GPT4O_ENDPOINT environment variable is not set")
    
    try:
        client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_KEY,
            api_version="2025-01-01-preview",
        )
        
        # Build message content
        user_msg = build_multimodal_message(query, image_paths)
        messages = [
            {"role": "system", "content": sys_prompt},
            user_msg,
        ]
        
        for attempt in range(max_retries):
            try:
                if is_reasoning_model:
                    # Use reasoning_effort instead of temperature
                    response = client.chat.completions.create(
                        model=deployment_name,
                        messages=messages,
                        reasoning_effort=reasoning_effort,
                        stream=False,
                    )
                else:
                    # Standard chat completion
                    response = client.chat.completions.create(
                        model=deployment_name,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        stream=False,
                    )
                
                # Extract token usage
                usage = response.usage
                token_usage = {
                    'prompt_tokens': usage.prompt_tokens,
                    'completion_tokens': usage.completion_tokens,
                    'total_tokens': usage.total_tokens,
                    'reasoning_tokens': 0
                }
                
                # For 'o' series models (o1, o3, o4), extract reasoning tokens
                if hasattr(usage, 'completion_tokens_details') and hasattr(usage.completion_tokens_details, 'reasoning_tokens'):
                    token_usage['reasoning_tokens'] = usage.completion_tokens_details.reasoning_tokens or 0
                
            
                return response.choices[0].message.content, token_usage
            
            except Exception as e:
                error_str = str(e)
                if 'content_filter' in error_str:
                    return 'content_filter',None
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"⚠️  Azure Key API error: {e} — Retrying in {wait_time} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Failed after {max_retries} attempts. Last error: {e}")
                    logger.info(f"❌ Failed after {max_retries} attempts. Last error: {e}")
                    raise
                    
    except Exception as e:
        print(f"❌ Failed to initialize Azure OpenAI client with key: {e}")
        logger.info(f"❌ Failed to initialize Azure OpenAI client with key: {e}")
        raise

def _chat_openai_key(
    query: str,
    sys_prompt: str,
    model: str,
    image_paths: list,
    temperature: float,
    top_p: float,
    max_tokens: int,
    reasoning_effort: str,
    is_reasoning_model: bool,
    max_retries: int
) -> tuple:
    """
    Internal function for standard OpenAI with key-based authentication.
    Supports both vision models (with images) and reasoning models.
    Returns: (response_text, token_usage_dict)
    """
    
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY, timeout=float(os.getenv("API_TIMEOUT_SECONDS", "90")))
        
        # Build message content
        user_msg = build_multimodal_message(query, image_paths)
        messages = [
            {"role": "system", "content": sys_prompt},
            user_msg,
        ]
        
        for attempt in range(max_retries):
            try:
                if is_reasoning_model:
                    content = [{"type": "input_text", "text": query}]
                    if image_paths:
                        for img_path in image_paths:
                            try:
                                new_image_path, img_format = convert_to_supported_format(img_path)
                                with open(new_image_path, "rb") as f:
                                    b64 = base64.b64encode(f.read()).decode("ascii")
                                content.append({
                                    "type": "input_image",
                                    "image_url": f"data:image/{img_format};base64,{b64}"
                                })
                            except Exception as e:
                                print(f"⚠️  Warning: Failed to load image {img_path}: {e}")
                    input_messages = []
                    if sys_prompt:
                        input_messages.append({
                            "role": "system",
                            "content": [{"type": "input_text", "text": sys_prompt}],
                        })
                    input_messages.append({"role": "user", "content": content})
                    response = client.responses.create(
                        model=model,
                        input=input_messages,
                        reasoning={"effort": reasoning_effort},
                        text={"verbosity": os.getenv("OPENAI_TEXT_VERBOSITY", "low")},
                    )
                    usage = getattr(response, "usage", None)
                    token_usage = {
                        "prompt_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
                        "completion_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
                        "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
                        "reasoning_tokens": 0,
                    }
                    details = getattr(usage, "output_tokens_details", None) if usage else None
                    if details is not None:
                        token_usage["reasoning_tokens"] = getattr(details, "reasoning_tokens", 0) or 0
                    return response.output_text, token_usage

                # Build completion parameters
                completion_params = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens
                }
                
                # Add appropriate parameters based on model type
                if is_reasoning_model:
                    # For reasoning models (o1, o3, etc.), use reasoning_effort
                    # Note: Standard OpenAI reasoning models might use different parameters
                    completion_params["reasoning_effort"] = reasoning_effort
                else:
                    # For standard models, use temperature and top_p
                    completion_params["temperature"] = temperature
                    completion_params["top_p"] = top_p
                    completion_params["max_tokens"] = max_tokens
                
                response = client.chat.completions.create(**completion_params)
                
                # Extract token usage
                usage = response.usage
                token_usage = {
                    'prompt_tokens': usage.prompt_tokens,
                    'completion_tokens': usage.completion_tokens,
                    'total_tokens': usage.total_tokens,
                    'reasoning_tokens': 0
                }
                
                # For reasoning models, extract reasoning tokens if available
                if hasattr(usage, 'completion_tokens_details') and hasattr(usage.completion_tokens_details, 'reasoning_tokens'):
                    token_usage['reasoning_tokens'] = usage.completion_tokens_details.reasoning_tokens or 0

                return response.choices[0].message.content, token_usage

            except Exception as e:
                error_str = str(e)
                if 'content_filter' in error_str or 'content_policy' in error_str:
                    return 'content_filter', None
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"⚠️  OpenAI API error: {e} — Retrying in {wait_time} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Failed after {max_retries} attempts. Last error: {e}")
                    logger.info(f"❌ Failed after {max_retries} attempts. Last error: {e}")
                    raise
                    
    except Exception as e:
        print(f"❌ Failed to initialize OpenAI client: {e}")
        logger.info(f"❌ Failed to initialize OpenAI client: {e}")
        raise

if __name__ == "__main__":
    """
    Test script for ChatGPT_VQA function with different authentication modes and models.
    
    Usage:
        1. Set up environment variables in .env file
        2. Run: python llm_openai.py
        3. Optionally provide image path as command argument: python llm_openai.py path/to/image.jpg
    
    Example: 
        # {"qid": 676, "image_name": "synpic31955.jpg", "image_organ": "CHEST", "answer": "Right lung hilum", 
        # "answer_type": "OPEN", "question_type": "POS", "question": "Where is/are the lesion located?", 
        # "phrase_type": "frame", "id": "vqa_rad_676", "image_folder": "/mnt/hanoverdev/scratch/yanboxu/BioMed_CLIP/VQA_meter/raw_data/data_RAD/images", 
        # "image": "synpic31955.jpg", "task": "VQA_RAD", "conversations": [{"from": "human", "value": "<image>\nWhere is/are the lesion located? Answer in a short word or phrase."}, 
        # {"from": "gpt", "value": "Right lung hilum"}]}
    """
    
    print("=" * 80)
    print("ChatGPT VQA Function Test Suite")
    print("=" * 80)
    
    # Check if image path provided via command line
    test_image = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Common test prompts for comparison across models/APIs
    test_image = "temp_image/synpic676.jpg"  # Update with your test image path
    sys_prompt = "You are a helpful AI assistant."
    text_prompt = "<image>\nWhere is/are the lesion located? Answer in a short word or phrase."
    
    # Test configurations - using same prompts for fair comparison
    if api_type == "azure_token":
        model_list = ["gpt-4o","o3","o4-mini","gpt-5"]
        for model_name in model_list:
            print(f"\n{'-' * 80}")
            print(f"Testing Model: {model_name} (Azure Token Authentication)")
            print(f"{'-' * 80}")
            try:
                response, token_usage = ChatGPT_VQA(
                    query=text_prompt,
                    sys_prompt=sys_prompt,
                    api_type="azure_token",
                    model=model_name,
                    image_paths=[test_image] if test_image else None,
                    temperature=0.0,
                    max_tokens=1024,
                    reasoning_effort="medium"
                )
                print(f"Response:\n{response}\n")
                if token_usage:
                    print(f"Token Usage: {token_usage}\n")
            except Exception as e:
                print(f"Error during test: {e}\n")
        
    elif api_type == "azure_key":
        model_list = ["gpt-4o","gpt-5"]       
        for model_name in model_list:
            print(f"\n{'-' * 80}")
            print(f"Testing Model: {model_name} (Azure Key Authentication)")
            print(f"{'-' * 80}")       
            try:
                response = ChatGPT_VQA(
                    query=text_prompt,
                    sys_prompt=sys_prompt,
                    api_type="azure_key",
                    model=model_name,
                    image_paths=[test_image] if test_image else None,
                    temperature=0.0,
                    max_tokens=1024,
                    reasoning_effort="medium"
                )
                print(f"Response:\n{response}\n")
            except Exception as e:
                print(f"Error during test: {e}\n")
     
    elif api_type == "openai":
        model_list = ["gpt-4o"] 
        for model_name in model_list:
            print(f"\n{'-' * 80}")
            print(f"Testing Model: {model_name} (OpenAI Key Authentication)")
            print(f"{'-' * 80}")       
            try:
                response = ChatGPT_VQA(
                    query=text_prompt,
                    sys_prompt=sys_prompt,
                    api_type="openai",
                    model=model_name,
                    image_paths=[test_image] if test_image else None,
                    temperature=0.0,
                    max_tokens=1024,
                    reasoning_effort="medium"
                )
                print(f"Response:\n{response}\n")
            except Exception as e:
                print(f"Error during test: {e}\n")
    else:
        print("Please set api_type to 'azure_token', 'azure_key', or 'openai' in the script.")
