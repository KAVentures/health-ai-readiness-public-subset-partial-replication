"""Tools package for LLM API calls."""

# Import OpenRouter functions for unified access (recommended)
try:
    from .llm_openrouter import (
        openrouter_vqa_sys,
        list_available_models,
        get_model_info
    )
    OPENROUTER_AVAILABLE = True
except ImportError:
    OPENROUTER_AVAILABLE = False

# Import extended LLM utilities (with OpenRouter fallback)
try:
    from .llm_extended import (
        Claude_vqa_sys,
        Qwen_vqa_sys,
        Gemini_vqa_sys,
        Deepseek_vqa_sys
    )
except ImportError:
    pass

# Import OpenAI utilities
try:
    from .llm_openai import ChatGPT_VQA
except ImportError:
    pass

__all__ = [
    'openrouter_vqa_sys',
    'list_available_models',
    'get_model_info',
    'Claude_vqa_sys',
    'Qwen_vqa_sys',
    'Gemini_vqa_sys',
    'Deepseek_vqa_sys',
    'ChatGPT_VQA',
    'OPENROUTER_AVAILABLE'
]
