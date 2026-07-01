#!/usr/bin/env python3
"""
MedGemma Benchmark Evaluation - VQA-RAD

Evaluates MedGemma on the VQA-RAD dataset.
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
    parser.add_argument("--data-dir", type=str, default="../../../data/VQA_RAD/", help="Path to VQA-RAD data directory")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "VQA-RAD"), exist_ok=True)
    
    # Setup logging filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(args.output_dir, f"VQA-RAD/vqarad_{args.model_variant}_{timestamp}.json")
    live_log_filename = os.path.join(args.output_dir, f"VQA-RAD/vqarad_{args.model_variant}_{timestamp}_live.json")
    
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
    path = os.path.join(args.data_dir, "test.json")
    
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
    for sample in tqdm(test_set, desc="Processing VQA-RAD"):
        question_id = questions_processed
        questions_processed += 1

        # Extract question info
        question_txt = sample['question']
        
        # Build prompt - VQA-RAD has different format
        prompt = f"""You are a radiology expert. Please analyze the following medical image and answer the question based on the image and your medical knowledge.
Question: {question_txt}
Please answer concisely:
"""

        # Get image path - VQA-RAD stores image folder in sample
        image_rel_folder = sample.get('image_folder')
        image_folder = os.path.join(args.data_dir, image_rel_folder)
        
        image_name = sample['image']
        
        if isinstance(image_name, str):
            image_path = f"{image_folder}/{image_name}"
        elif isinstance(image_name, list):
            image_path = f"{image_folder}/{image_name[0]}"
        else:
            image_path = []
        
        # Check if image exists
        if not image_path or not os.path.exists(image_path):
            skipped_questions += 1
            logging.warning(f"Question {question_id}: Skipped (Image not found: {image_path})")
            
            skipped_entry = {
                "question_id": question_id,
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
        gt_answer = sample['answer']
        
        # Check correctness (case-insensitive match)
        is_correct = (answer.strip().lower() == gt_answer.strip().lower())
        if is_correct:
            correct_answers += 1
        
        # Prepare log entry
        log_entry = {
            "question_id": question_id,
            "question": question_txt,
            "input_prompt": prompt,
            "image": image_name,
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
    logging.info("VQA-RAD Benchmark Results")
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
