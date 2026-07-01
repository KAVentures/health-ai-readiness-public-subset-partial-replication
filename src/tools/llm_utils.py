"""
Common LLM utilities shared across all LLM API modules.

This module provides shared helper functions used by:
- llm_openai.py (Azure OpenAI and OpenAI APIs)
- llm_utils_extended.py (Claude, Qwen, Gemini, DeepSeek)
- llm_openrouter.py (OpenRouter unified interface)
"""

import os
# import logger
from typing import Tuple, Optional
from PIL import Image
from logging_utils import get_logger

logger = get_logger(__name__)


def convert_to_supported_format(image_path: str, output_format: str = "JPEG") -> Tuple[Optional[str], Optional[str]]:
    """
    Convert image to supported format if needed.
    
    Supports: JPEG, PNG (kept as-is), BMP/TIFF (converted to JPEG)
    
    Args:
        image_path: Path to input image
        output_format: Desired output format (default: JPEG)
        
    Returns:
        Tuple of (converted_path, format_string) or (None, None) on error
        
    Examples:
        >>> path, fmt = convert_to_supported_format("image.jpg")
        >>> # Returns ("image.jpg", "jpeg")
        
        >>> path, fmt = convert_to_supported_format("image.bmp")
        >>> # Returns ("./temp_image/image.jpg", "jpeg")
    """
    temp_image_folder = "./temp_image"
    try:
        os.makedirs(temp_image_folder, exist_ok=True)
        
        file_name = os.path.basename(image_path)
        file_purename, file_ext = os.path.splitext(file_name)
        file_ext = file_ext.lower()
        
        # Already in supported format
        if file_ext in ['.jpeg', '.jpg', '.png']:
            return image_path, 'jpeg' if file_ext in ['.jpeg', '.jpg'] else 'png'
        
        # Convert BMP/TIFF to JPEG
        elif file_ext in ['.bmp', '.tif', '.tiff']:
            img = Image.open(image_path)
            temp_file = os.path.join(temp_image_folder, file_purename + '.jpg')
            img.convert('RGB').save(temp_file, 'JPEG')
            img.close()
            return temp_file, 'jpeg'
        
        # Unsupported format
        else:
            logger.warning(f"Unsupported image format: {file_ext} for file: {image_path}")
            return None, None
            
    except FileNotFoundError:
        logger.error(f"File not found: {image_path}")
        return None, None
    except Exception as e:
        logger.error(f"Error processing image: {image_path} - {e}")
        return None, None


def validate_image_path(image_path: str) -> bool:
    """
    Validate that an image file exists and is readable.
    
    Args:
        image_path: Path to image file
        
    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(image_path):
        logger.warning(f"Image not found: {image_path}")
        return False
    
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception as e:
        logger.warning(f"Invalid image file: {image_path} - {e}")
        return False


if __name__ == "__main__":
    # Test the utilities
    print("Testing LLM utilities...")
    
    # Test convert_to_supported_format
    test_path = "/tmp/test.jpg"
    if os.path.exists(test_path):
        result = convert_to_supported_format(test_path)
        print(f"convert_to_supported_format: {result}")
