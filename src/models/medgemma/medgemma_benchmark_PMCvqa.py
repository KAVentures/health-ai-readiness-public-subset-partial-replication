#!/usr/bin/env python3
"""
MedGemma Benchmark Evaluation - PMC-VQA

Evaluates MedGemma on the PMC-VQA dataset.
"""

import argparse
import json
import os
import logging
from datetime import datetime
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_inference(model, processor, is_thinking, image_path, question, 
                  system_instruction="You are an expert medical AI assistant."):
    """
    Run inference on a single image-question pair.
    
    Args:
        model: MedGemma model
        processor: Model processor
        is_thinking: Whether thinking mode is enabled
        image_path: Path to the image file
        question: Question text
        system_instruction: System prompt for the model
    
    Returns:
        Tuple of (answer_text, inference_time)
    """
    import time
    
    system_instruction = """You are a medical imaging expert."""
    
    try:
        start_time = time.time()
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Prepare messages
        if is_thinking:
            system_instruction = f"SYSTEM INSTRUCTION: think silently if needed. {system_instruction}"
            max_new_tokens = 1300
        else:
            max_new_tokens = 800
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_instruction}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image", "image": image}
                ]
            }
        ]
        
        # Process inputs
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)
        
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate response
        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            generation = generation[0][input_len:]
        
        response = processor.decode(generation, skip_special_tokens=True)
        
        # Extract final response if thinking mode is enabled
        if is_thinking and "<unused95>" in response:
            response = response.split("<unused95>")[-1].strip()
        
        duration = time.time() - start_time
        return response, duration
    
    except Exception as e:
        logging.error(f"Error processing {image_path}: {e}")
        return "", 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-variant", type=str, default="medgemma-27b-it",
                        choices=["medgemma-1.5-4b-it", "medgemma-4b-it", "medgemma-27b-it"])
    parser.add_argument("--use-quantization", action="store_true", default=True)
    parser.add_argument("--thinking-mode", action="store_false", default=False)
    parser.add_argument("--output-dir", type=str, default="benchmark_results")
    parser.add_argument("--data-dir", type=str, default="../../../data/PMC-VQA/", help="Path to PMC-VQA data directory")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "PMC-VQA"), exist_ok=True)
    
    # Setup logging filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(args.output_dir, f"PMC-VQA/pmcvqa_{args.model_variant}_{timestamp}.json")
    live_log_filename = os.path.join(args.output_dir, f"PMC-VQA/pmcvqa_{args.model_variant}_{timestamp}_live.json")
    
    # Initialize live log file
    with open(live_log_filename, "w") as f:
        f.write("[\n")
    
    logging.info(f"Live results will be logged to: {live_log_filename}")
    
    # Load model
    model_id = f"google/{args.model_variant}"
    
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # if args.use_quantization:
    #     model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    
    logging.info(f"Loading {args.model_variant}...")
    model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_id)
    logging.info("Model loaded successfully!")
    
    # Load dataset
    path = os.path.join(args.data_dir, "test_2_500_sample_filter.json")
    image_folder = os.path.join(args.data_dir, "figures/")
    
    try:
        with open(path, 'r') as f:
            test_set = json.load(f)
            total_questions = len(test_set)
        logging.info(f'Loaded test set with {total_questions} samples.')
    except Exception as e:
        logging.error(f'Failed to load test set: {e}')
        raise

    results = {
        "model": args.model_variant,
        "timestamp": datetime.now().isoformat(),
        "total_questions": total_questions,
        "results": [],
    }

    questions_processed = 0
    correct_answers = 0
    skipped_questions = 0

    # Process each sample
    for sample in tqdm(test_set, desc="Processing PMC-VQA"):
        question_id = questions_processed
        questions_processed += 1

        # Extract question info - PMC-VQA uses capitalized keys
        question_txt = sample['Question']
        
        # Build options
        options = []
        options.append(sample['Choice A'])
        options.append(sample['Choice B'])
        options.append(sample['Choice C'])
        options.append(sample['Choice D'])
        question_options = " ".join(options)
        
        # Build prompt
        prompt = f"""Given the following medical case:
Please answer this multiple choice question:
{question_txt}
Options: {question_options}
Base your answer only on the provided images and case information. Only respond with your SINGLE LETTER answer: """

        # Get image path
        figure_path = sample['Figure_path']
        if isinstance(figure_path, str):
            image_path = os.path.join(image_folder, figure_path)
        elif isinstance(figure_path, list):
            image_path = os.path.join(image_folder, figure_path[0])
        else:
            image_path = str(figure_path)
        
        # Check if image exists
        if not os.path.exists(image_path):
            skipped_questions += 1
            logging.warning(f"Question {question_id}: Skipped (Image not found: {image_path})")
            
            skipped_entry = {
                "question_id": question_id,
                "ori_index": sample.get('index', question_id),
                "status": "skipped",
                "reason": "Image not found",
            }
            with open(live_log_filename, "a") as live_log_file:
                json.dump(skipped_entry, live_log_file, indent=2)
                live_log_file.write(",\n")
            continue

        # Run inference
        answer, duration = run_inference(
            model, processor, args.thinking_mode,
            image_path, prompt
        )
        
        # Get ground truth
        gt_answer = sample.get('Answer', '')
        
        # Get modality type if available
        modality_type = sample.get('modality_type', '')
        
        # Check correctness
        is_correct = (answer.strip().upper() == gt_answer.strip().upper())
        if is_correct:
            correct_answers += 1
        
        # Prepare log entry
        log_entry = {
            "question_id": question_id,
            "ori_index": sample.get('index', question_id),
            "question": question_txt,
            "options": question_options,
            "input_prompt": prompt,
            "image": figure_path,
            "modality_type": modality_type,
            "prediction": answer,
            "ground_truth": gt_answer,
            "correct": is_correct,
            "duration": duration,
        }
        
        results["results"].append(log_entry)
        
        # Write to live log
        with open(live_log_filename, "a") as live_log_file:
            json.dump(log_entry, live_log_file, indent=2)
            live_log_file.write(",\n")
        
        # Print progress
        if (question_id + 1) % 10 == 0:
            current_acc = (correct_answers / questions_processed) * 100
            logging.info(f"Processed {questions_processed}/{total_questions} | Accuracy: {current_acc:.2f}%")

    # Close live log file
    with open(live_log_filename, "a") as f:
        f.write("]\n")

    # Calculate final metrics
    accuracy = (correct_answers / questions_processed) * 100 if questions_processed > 0 else 0
    
    results["summary"] = {
        "total_processed": questions_processed,
        "correct": correct_answers,
        "skipped": skipped_questions,
        "accuracy": accuracy,
    }

    # Save final results
    with open(log_filename, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    logging.info("\n" + "="*60)
    logging.info("PMC-VQA Benchmark Results")
    logging.info("="*60)
    logging.info(f"Model: {args.model_variant}")
    logging.info(f"Total questions: {total_questions}")
    logging.info(f"Processed: {questions_processed}")
    logging.info(f"Correct: {correct_answers}")
    logging.info(f"Skipped: {skipped_questions}")
    logging.info(f"Accuracy: {accuracy:.2f}%")
    logging.info(f"\nResults saved to: {log_filename}")


if __name__ == "__main__":
    main()
