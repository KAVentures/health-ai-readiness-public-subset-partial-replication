"""
Prompt Templates for Medical VQA Tasks

Contains system prompts and query builders for different datasets.
"""

# ==================================================
# System Prompts
# ==================================================

SYSTEM_PROMPTS = {
    'default': "You are a helpful medical assistant that answers multiple choice questions about the provided image.",
    'empty': "",
    'vqa_rad': "You are a radiology expert. Please analyze the following medical image and answer the question based on the image and your medical knowledge."
}

# ==================================================
# Query Builders
# ==================================================

def build_mcq_query_jama(sample):
    """
    Build multiple choice query for JAMA dataset.
    
    Args:
        sample: Dict with keys 'question', 'opa', 'opb', 'opc', 'opd'
    
    Returns:
        str: Formatted query string
    """
    query = "The following is a multiple choice question with answer options."
    query += f"\n{sample['question']}"
    query += f"\nOptions: A. {sample['opa']}   B. {sample['opb']}    C. {sample['opc']}    D. {sample['opd']}"
    query += "\nPlease respond only with the index and the content of the option you believe is correct. Enclose your answer within <answer></answer> tags."
    return query


def build_mcq_query_nejm(sample):
    """
    Build multiple choice query for NEJM dataset.
    
    Args:
        sample: Dict with 'conversations' key containing question
    
    Returns:
        str: Formatted query string
    """
    query = "The following is a multiple choice question with answer options."
    query += f"\n{sample['conversations'][0]['value']}"
    query += "\nPlease respond only with the index and the content of the option you believe is correct. Enclose your answer within <answer></answer> tags."
    return query


def build_mcq_query_omnivqa(sample):
    """
    Build multiple choice query for OmniVQA dataset.
    
    Args:
        sample: Dict with keys 'question', 'option_A', 'option_B', 'option_C', 'option_D'
    
    Returns:
        str: Formatted query string
    """
    query = "The following is a multiple choice question with answer options."
    query += '\nQuestion: ' + sample['question'] + '\n' + "Options: "
    
    options = []
    options.append(f"A. {sample['option_A']}")
    options.append(f"B. {sample['option_B']}")
    if 'option_C' in sample:
        options.append(f"C. {sample['option_C']}")
    if 'option_D' in sample:
        options.append(f"D. {sample['option_D']}")
    
    query += " ".join(options)
    query += "\nPlease respond only with the index and the content of the option you believe is correct. Enclose your answer within <answer></answer> tags."
    return query


def build_mcq_query_pmcvqa(sample):
    """
    Build multiple choice query for PMC-VQA dataset.
    
    Args:
        sample: Dict with keys 'Question', 'Choice A', 'Choice B', 'Choice C', 'Choice D'
    
    Returns:
        str: Formatted query string
    """
    query = "The following is a multiple choice question with answer options."
    query += '\nQuestion: ' + sample['Question'] + '\n' + "Options: "
    
    options = []
    options.append(sample['Choice A'])
    options.append(sample['Choice B'])
    options.append(sample['Choice C'])
    options.append(sample['Choice D'])
    
    query += " ".join(options)
    query += "\nPlease respond only with the index and the content of the option you believe is correct. Enclose your answer within <answer></answer> tags."
    return query


def build_query_vqarad(sample):
    """
    Build query for VQA-RAD dataset (open-ended or closed).
    
    Args:
        sample: Dict with keys 'question', 'answer_type'
    
    Returns:
        str: Formatted query string
    """
    query = 'Question: ' + sample['question']
    
    QA_type = sample.get('answer_type', 'OPEN')
    if QA_type == 'CLOSED':
        query += "\n Answer the question using a single word or phrase. Enclose your answer within <answer></answer> tags."
    else:  # 'OPEN'
        query += "\n Answer the question concisely. Enclose your answer within <answer></answer> tags."
    
    return query


def build_generic_mcq_query(question, options):
    """
    Build generic multiple choice query.
    
    Args:
        question: Question text
        options: Dict or list of options
    
    Returns:
        str: Formatted query string
    """
    query = "The following is a multiple choice question with answer options.\n"
    query += f"{question}\n"
    
    if isinstance(options, dict):
        for key, value in options.items():
            query += f"{key}. {value}   "
    elif isinstance(options, list):
        for i, option in enumerate(options):
            query += f"{chr(65+i)}. {option}   "
    
    query += "\nPlease respond only with the index and the content of the option you believe is correct. Enclose your answer within <answer></answer> tags."
    return query


def build_mcq_query_nejm_cot(sample):
    """
    Build multiple choice query for NEJM dataset with chain-of-thought prompting.
    
    Args:
        sample: Dict with 'conversations' key containing question
    
    Returns:
        str: Formatted query string
    """
    query = "The following is a multiple choice question with answer options."
    query += f"\n{sample['conversations'][0]['value']}"
    query += "\nPlease think step by step, enclosing the thought process within <thinking></thinking> tags. Provide answer with the index and content of the option, and place it within <answer></answer> tags."
    return query


def build_query_vqarad_cot(sample):
    """
    Build query for VQA-RAD dataset with chain-of-thought prompting.
    
    Args:
        sample: Dict with keys 'question', 'answer_type'
    
    Returns:
        str: Formatted query string
    """
    query = 'Question: ' + sample['question']
    
    QA_type = sample.get('answer_type', 'OPEN')
    if QA_type == 'CLOSED':
        query += "\nPlease think step by step, enclosing the thought process within <thinking></thinking> tags. Answer the question using a single word or phrase, and place the answer within <answer></answer> tags."
    else:  # 'OPEN'
        query += "\nPlease think step by step, enclosing the thought process within <thinking></thinking> tags. Answer the question concisely, and place the answer within <answer></answer> tags."
    
    return query

# ==================================================
# Query Builder Mapping
# ==================================================

QUERY_BUILDERS = {
    'JAMA': build_mcq_query_jama,
    'NEJM': build_mcq_query_nejm,
    'OmniVQA': build_mcq_query_omnivqa,
    'PMC-VQA': build_mcq_query_pmcvqa,
    'VQA-RAD': build_query_vqarad,
    'generic': build_generic_mcq_query,
    'NEJM-cot': build_mcq_query_nejm_cot,
    'VQA-RAD-cot': build_query_vqarad_cot,
}
