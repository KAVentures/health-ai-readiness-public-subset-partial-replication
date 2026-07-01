"""
Dataset Configuration File

Contains dataset-specific settings for different medical VQA datasets.
Each dataset config includes:
- Data paths (relative to BASE_DIR)
- Image paths (relative to BASE_DIR)
- Dataset-specific parameters
- Answer format specifications
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory for data files
# By default, points to 'data/' relative to project root
# Can be overridden by setting DATA_BASE_DIR environment variable

_DEFAULT_BASE_DIR = Path(__file__).parent.parent.parent / 'data'
BASE_DIR = os.environ.get('DATA_BASE_DIR', str(_DEFAULT_BASE_DIR))

DATASET_CONFIGS = {
    'JAMA': {
        'name': 'JAMA',
        'data_path': 'JAMA/jama_processed.json',
        'image_base_path': 'JAMA/',
        'answer_key': 'answer_idx',
        'image_key': 'image_paths',  # Can be list of paths
        'supports_multiple_images': True,
        'query_builder': 'JAMA',
        'system_prompt': 'default',
        'save_interval': 20,
        'description': 'JAMA Clinical Challenge dataset with multiple images per sample'
    },
    
    'NEJM': {
        'name': 'NEJM',
        'data_path': 'NEJM_image_challenge/test.json',
        'image_base_path': 'NEJM_image_challenge/',
        'answer_key': 'correct_answer',
        'image_key': 'image_rel_path',  # Single image path
        'supports_multiple_images': False,
        'query_builder': 'NEJM',
        'system_prompt': 'default',
        'save_interval': 20,
        'description': 'NEJM Image Challenge dataset with single image per sample'
    },
    
    'OmniVQA': {
        'name': 'OmniVQA',
        'data_path': 'OmniMedVQA/sampled_vqa_500.json',
        'image_base_path': 'OmniMedVQA/',
        'answer_key': 'gt_answer',
        'image_key': 'image_path',  # Single image path
        'supports_multiple_images': False,
        'query_builder': 'OmniVQA',
        'system_prompt': 'default',
        'save_interval': 20,
        'description': 'OmniMedVQA dataset with multiple choice questions across medical modalities'
    },
    
    'PMC-VQA': {
        'name': 'PMC-VQA',
        'data_path': 'PMC-VQA/test_2_500_sample.json',
        'image_base_path': 'PMC-VQA/figures/',
        'answer_key': 'Answer',
        'image_key': 'Figure_path',  # Single image path
        'supports_multiple_images': False,
        'query_builder': 'PMC-VQA',
        'system_prompt': 'default',
        'save_interval': 20,
        'description': 'PMC-VQA dataset with medical literature figure questions'
    },
    
    'VQA-RAD': {
        'name': 'VQA-RAD',
        'data_path': 'VQA_RAD/test.json',
        'image_base_path': 'VQA_RAD/',  
        'answer_key': 'answer',
        'image_key': 'image_path_full',  # Full path (image_folder + image)
        'supports_multiple_images': False,
        'query_builder': 'VQA-RAD',
        'system_prompt': 'vqa_rad',
        'save_interval': 20,
        'description': 'VQA-RAD radiology dataset with open-ended and closed questions'
    },
    
    # Stress testing 
    'ST_v0': {
        'name': 'ST_v0',
        'data_path': 'NEJM_image_challenge/test_no_context_v0.json',
        'image_base_path': 'NEJM_image_challenge/',
        'answer_key': 'correct_answer',
        'image_key': 'image_rel_path',  # Single image path
        'supports_multiple_images': False,
        'query_builder': 'NEJM',
        'system_prompt': 'default',
        'save_interval': 20,
        'description': 'NEJM stress test (197) with single image per sample'
    },
    
    'ST_reorder_v1': {
        'name': 'ST_reorder_v1',
        'data_path': 'NEJM_image_challenge/test_reformat_0_replace_ans_reorder_v1.json',
        'image_base_path': 'NEJM_image_challenge/',
        'answer_key': 'correct_answer',
        'image_key': 'image_rel_path',  # Single image path
        'supports_multiple_images': False,
        'query_builder': 'NEJM',
        'system_prompt': 'default',
        'save_interval': 20,
        'description': 'NEJM stress test (197) with single image per sample，reordered options'
    },
    
    'ST_replace1_v2': {
        'name': 'ST_replace1_v2',
        'data_path': 'NEJM_image_challenge/test_reformat_1_replace_ans_reorder_v2.json',
        'image_base_path': 'NEJM_image_challenge/',
        'answer_key': 'correct_answer',
        'image_key': 'image_rel_path',  # Single image path
        'supports_multiple_images': False,
        'query_builder': 'NEJM',
        'system_prompt': 'default',
        'save_interval': 20,
        'description': 'NEJM stress test (197) with single image per sample, replace 1 option'
    },
    
    'ST_replace2_v3': {
        'name': 'ST_replace2_v3',
        'data_path': 'NEJM_image_challenge/test_reformat_2_replace_ans_reorder_v3.json',
        'image_base_path': 'NEJM_image_challenge/',
        'answer_key': 'correct_answer',
        'image_key': 'image_rel_path',  # Single image path
        'supports_multiple_images': False,
        'query_builder': 'NEJM',
        'system_prompt': 'default',
        'save_interval': 20,
        'description': 'NEJM stress test (197) with single image per sample, replace 2 options'
    },
    
    'ST_replace3_v4': {
        'name': 'ST_replace3_v4',
        'data_path': 'NEJM_image_challenge/test_reformat_3_replace_ans_reorder_v4.json',
        'image_base_path': 'NEJM_image_challenge/',
        'answer_key': 'correct_answer',
        'image_key': 'image_rel_path',  # Single image path
        'supports_multiple_images': False,
        'query_builder': 'NEJM',
        'system_prompt': 'default',
        'save_interval': 20,
        'description': 'NEJM stress test (197) with single image per sample, replace 3 options'
    },
    
    'ST_replace4_v5': {
        'name': 'ST_replace4_v5',
        'data_path': 'NEJM_image_challenge/test_reformat_4_replace_ans_reorder_v5.json',
        'image_base_path': 'NEJM_image_challenge/',
        'answer_key': 'correct_answer',
        'image_key': 'image_rel_path',  # Single image path
        'supports_multiple_images': False,
        'query_builder': 'NEJM',
        'system_prompt': 'default',
        'save_interval': 20,
        'description': 'NEJM stress test (197) with single image per sample, replace 4 options'
    },
    
    'ST_unknown_v6': {
        'name': 'ST_unknown_v6',
        'data_path': 'NEJM_image_challenge/test_reformat_replace_unknown_v6.json',
        'image_base_path': 'NEJM_image_challenge/',
        'answer_key': 'correct_answer',
        'image_key': 'image_rel_path',  # Single image path
        'supports_multiple_images': False,
        'query_builder': 'NEJM',
        'system_prompt': 'default',
        'save_interval': 20,
        'description': 'NEJM stress test (197) with single image per sample, replace with unknown option'
    },   
    
    'ST_substitute_v7': {
        'name': 'ST_substitute_v7',
        'data_path': 'NEJM_image_challenge/test_Image_aug_v7.json',
        'image_base_path': 'NEJM_image_challenge/',
        'answer_key': 'answer',
        'image_key': 'image_rel_path',  # Single image path
        'supports_multiple_images': False,
        'query_builder': 'NEJM',
        'system_prompt': 'default',
        'save_interval': 20,
        'description': 'NEJM stress test (40) with single image per sample, substitute image with similar one'
    },
    
    'ST_no_substitute_v8': {
        'name': 'ST_no_substitute_v8',
        'data_path': 'NEJM_image_challenge/test_Image_aug_base_v8.json',
        'image_base_path': 'NEJM_image_challenge/',
        'answer_key': 'correct_answer',
        'image_key': 'image_rel_path',  # Single image path
        'supports_multiple_images': False,
        'query_builder': 'NEJM',
        'system_prompt': 'default',
        'save_interval': 20,
        'description': 'NEJM stress test (40) with single image per sample, no substitution'
    },
    
    'ST_unseen_v10': {
        'name': 'ST_unseen_v10',
        'data_path': 'test_unseen/refined_mcq_selected_final_v10.json',
        'image_base_path': 'test_unseen/',
        'answer_key': 'answer',
        'image_key': 'image_rel_path',  # Single image path
        'supports_multiple_images': False,
        'query_builder': 'NEJM',
        'system_prompt': 'default',
        'save_interval': 20,
        'description': 'NEJM stress test unseen (60) with single image per sample, unseen images'
    },
    
    'NEJM-cot': {
        'name': 'NEJM-cot',
        'data_path': 'NEJM_image_challenge/test_sample_120.json',
        'image_base_path': 'NEJM_image_challenge/',
        'answer_key': 'correct_answer',
        'image_key': 'image_rel_path',  # Single image path
        'supports_multiple_images': False,
        'query_builder': 'NEJM-cot',
        'system_prompt': 'default',
        'save_interval': 20,
        'description': 'NEJM cot test (120) with single image per sample, includes chain-of-thought in query'
    }, 
    
    'VQA-RAD-cot': {
        'name': 'VQA-RAD-cot',
        'data_path': 'VQA_RAD/test_sample_100.json',
        'image_base_path': 'VQA_RAD/',  
        'answer_key': 'answer',
        'image_key': 'image_path_full',  # Full path (image_folder + image)
        'supports_multiple_images': False,
        'query_builder': 'VQA-RAD-cot',
        'system_prompt': 'vqa_rad',
        'save_interval': 20,
        'description': 'VQA-RAD cot test (100) with single image per sample, includes chain-of-thought in query'
    },
    
    'NEJM-no-cot': {
        'name': 'NEJM-no-cot',
        'data_path': 'NEJM_image_challenge/test_sample_120.json',
        'image_base_path': 'NEJM_image_challenge/',
        'answer_key': 'correct_answer',
        'image_key': 'image_rel_path',  # Single image path
        'supports_multiple_images': False,
        'query_builder': 'NEJM', # vqa without cot
        'system_prompt': 'default',
        'save_interval': 20,
        'description': 'NEJM no-cot test (120) with single image per sample, no chain-of-thought in query'
    }, 
    
    'VQA-RAD-no-cot': {
        'name': 'VQA-RAD-no-cot',
        'data_path': 'VQA_RAD/test_sample_100.json',
        'image_base_path': 'VQA_RAD/',  
        'answer_key': 'answer',
        'image_key': 'image_path_full',  # Full path (image_folder + image)
        'supports_multiple_images': False,
        'query_builder': 'VQA-RAD', #v qa without cot
        'system_prompt': 'vqa_rad',
        'save_interval': 20,
        'description': 'VQA-RAD no-cot test (100) with single image per sample, no chain-of-thought in query'
    },

    # Sample datasets for testing
    'VQA-RAD-sample': {
        'name': 'VQA-RAD-sample',
        'data_path': 'sample_data/vqa_rad_sample.json',
        'image_base_path': 'sample_data/',
        'answer_key': 'answer',
        'image_key': 'image_path',
        'supports_multiple_images': False,
        'query_builder': 'VQA-RAD',
        'system_prompt': 'vqa_rad',
        'save_interval': 5,
        'description': 'VQA-RAD sample dataset (3 samples) for quick testing'
    },
    
    'OmniVQA-sample': {
        'name': 'OmniVQA-sample',
        'data_path': 'sample_data/omnivqa_sample.json',
        'image_base_path': 'sample_data/',
        'answer_key': 'gt_answer',
        'image_key': 'image_path',
        'supports_multiple_images': False,
        'query_builder': 'OmniVQA',
        'system_prompt': 'default',
        'save_interval': 5,
        'description': 'OmniMedVQA sample dataset (3 samples) for quick testing'
    },
    
    'PMC-VQA-sample': {
        'name': 'PMC-VQA-sample',
        'data_path': 'sample_data/pmc_vqa_sample.json',
        'image_base_path': 'sample_data/',
        'answer_key': 'Answer',
        'image_key': 'image_path',
        'supports_multiple_images': False,
        'query_builder': 'PMC-VQA',
        'system_prompt': 'default',
        'save_interval': 5,
        'description': 'PMC-VQA sample dataset (3 samples) for quick testing'
    },
    
}


def get_dataset_config(dataset_name):
    """
    Get configuration for a specific dataset.
    Automatically resolves relative paths to absolute paths using BASE_DIR.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'JAMA', 'NEJM-cot', 'VQA-RAD-cot')
    
    Returns:
        dict: Dataset configuration with absolute paths
    
    Raises:
        ValueError: If dataset not found
    """
    if dataset_name not in DATASET_CONFIGS:
        available = ', '.join(DATASET_CONFIGS.keys())
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")
    
    # Get the base configuration
    config = DATASET_CONFIGS[dataset_name].copy()
    
    # Resolve relative paths to absolute paths
    if 'data_path' in config and config['data_path']:
        if not os.path.isabs(config['data_path']):
            config['data_path'] = os.path.join(BASE_DIR, config['data_path'])
    
    if 'image_base_path' in config and config['image_base_path']:
        if not os.path.isabs(config['image_base_path']):
            config['image_base_path'] = os.path.join(BASE_DIR, config['image_base_path'])
    
    return config


def list_datasets():
    """List all available datasets."""
    return list(DATASET_CONFIGS.keys())


def list_sample_datasets():
    """List available sample datasets for quick testing."""
    return [name for name in DATASET_CONFIGS.keys() if '-sample' in name]


def get_sample_datasets_info():
    """Get information about all sample datasets."""
    sample_datasets = list_sample_datasets()
    return {
        name: {
            'description': DATASET_CONFIGS[name]['description'],
            'data_path': DATASET_CONFIGS[name]['data_path']
        }
        for name in sample_datasets
    }
