"""
Model Configuration

Contains model-specific configurations for all supported LLMs.
"""

MODEL_CONFIGS = {
    'gpt-4o': {
        'full_name': 'gpt-4o',
        'short_name': 'gpt4o',
        'api_function': 'gpt',
        'api_type': 'azure_token' # azure_key, openai_key
    },
    'o3': {
        'full_name': 'o3',
        'short_name': 'o3',
        'api_function': 'gpt',
        'api_type': 'azure_token' # azure_key, openai_key
    },
    'o4-mini': {
        'full_name': 'o4-mini',
        'short_name': 'o4mini',
        'api_function': 'gpt',
        'api_type': 'azure_token' # azure_key, openai_key
    },
    'gpt-5': {
        'full_name': 'gpt-5',
        'short_name': 'gpt5',
        'api_function': 'gpt',
        'api_type': 'azure_token'  # azure_key, openai_key
    },
    'gpt-5.5': {
        'full_name': 'gpt-5.5',
        'short_name': 'gpt55',
        'api_function': 'gpt',
        'api_type': 'openai_key'
    },
    'grok-4.3': {
        'full_name': 'grok-4.3',
        'short_name': 'grok43',
        'api_function': 'xai',
        'api_type': 'xai_key'
    },
    'deepseek': {
        'full_name': 'deepseek-ai/deepseek-vl2',
        'short_name': 'deepseek',
        'api_function': 'deepseek',
        'api_type': 'siliconflow_key'
    },
    'qwen3-vl-235b-thinking': {
        'full_name': 'Qwen/Qwen3-VL-235B-A22B-Thinking',
        'short_name': 'qwen3vl235b',
        'api_function': 'deepseek',
        'api_type': 'siliconflow_key'
    },
    'gemini-2.0-flash': {
        'full_name': 'gemini-2.0-flash',
        'short_name': 'gemini2.0',
        'api_function': 'gemini',
        'api_type': 'openrouter_key'
    },
    'gemini-2.5-pro': {
        'full_name': 'gemini-2.5-pro',
        'short_name': 'gemini2.5pro',
        'api_function': 'gemini',
        'api_type': 'openrouter_key'
    },
    'gemini-2.5-flash': {
        'full_name': 'gemini-2.5-flash',
        'short_name': 'gemini2.5flash',
        'api_function': 'gemini',
        'api_type': 'openrouter_key'
    },
    'gemini-3.1-pro': {
        'full_name': 'gemini-3.1-pro-preview',
        'short_name': 'gemini31pro',
        'api_function': 'gemini',
        'api_type': 'google_key'
    },
    'gemini-3.5-flash': {
        'full_name': 'gemini-3.5-flash',
        'short_name': 'gemini35flash',
        'api_function': 'gemini',
        'api_type': 'google_key'
    },
    'claude-sonnet-3.5': {
        'full_name': 'claude-sonnet-3.5',
        'short_name': 'claude3.5',
        'api_function': 'claude',
        'api_type': 'openrouter_key'
    },
    'claude-opus-4.8': {
        'full_name': 'claude-opus-4-8',
        'short_name': 'claude48opus',
        'api_function': 'claude',
        'api_type': 'anthropic_key'
    },
    'qwen3-vl-instruct': {
        'full_name': 'qwen3-vl-instruct',
        'short_name': 'qwen3instr',
        'api_function': 'qwen',
        'api_type': 'openrouter_key'
    },
    'qwen3-vl-thinking': {
        'full_name': 'qwen3-vl-thinking',
        'short_name': 'qwen3think',
        'api_function': 'qwen',
        'api_type': 'openrouter_key'
    },
}

# Reasoning effort mapping for models that support it
REASONING_EFFORT_MAP = {
    'zero': 'medium',
    'low': 'low',
    'medium': 'medium',
    'high': 'high'
}


def get_model_config(model_key):
    """
    Get configuration for a specific model.
    
    Args:
        model_key: Model identifier
    
    Returns:
        dict: Model configuration
    
    Raises:
        ValueError: If model not found
    """
    if model_key not in MODEL_CONFIGS:
        available = ', '.join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model '{model_key}'. Available: {available}")
    
    return MODEL_CONFIGS[model_key]


def list_models():
    """List all available models."""
    return list(MODEL_CONFIGS.keys())
