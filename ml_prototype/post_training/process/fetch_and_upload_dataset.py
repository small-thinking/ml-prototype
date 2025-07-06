#!/usr/bin/env python3
"""
Script to fetch the first X records from the big-reasoning-traces dataset
and upload them to a specified Hugging Face profile in preference format.

The script transforms the original <think>/<answer> format into preference pairs in OpenAI conversation format:
- chosen: [{"role": "user", "content": "<prompt>"}, {"role": "assistant", "content": "<full_response_with_thinking>"}]
- rejected: [{"role": "user", "content": "<prompt>"}, {"role": "assistant", "content": "<answer_only>"}]

Usage:
    python fetch_and_upload_dataset.py --num_records 10000 --hf_username tech-tao --dataset_name dpo-reasoning-traces-10k --config_name DeepSeek
"""

import argparse
import logging
import re
from typing import Optional, Dict, Any
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi, login
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_thinking_and_answer(content: str) -> tuple[str, str]:
    """
    Extract thinking and answer sections from content that contains <thinking> and <answer> tags.
    
    Args:
        content: String containing <thinking> and <answer> tags
        
    Returns:
        tuple: (thinking_content, answer_content) - extracted content without tags
    """
    # Extract thinking section
    thinking_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
    thinking_content = thinking_match.group(1).strip() if thinking_match else ""
    
    # Extract answer section
    answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    answer_content = answer_match.group(1).strip() if answer_match else ""
    
    return thinking_content, answer_content


def transform_to_preference_format(
    records: list[dict],
    response_field: str = "response",
    prompt_field: str = "prompt"
) -> list[dict]:
    """
    Transform records from <think>/<answer> tag format to preference dataset format.
    Creates chosen and rejected responses in OpenAI conversation format.
    
    Args:
        records: List of dataset records (dicts)
        response_field: Field name containing the response string
        prompt_field: Field name containing the prompt string
        
    Returns:
        List of modified records with preference format (chosen, rejected fields)
    """
    preference_records = []
    
    for record in records:
        if response_field in record and prompt_field in record:
            response = record[response_field]
            prompt = record[prompt_field]
            
            # Extract thinking and answer from the response string
            thinking_content, answer_content = extract_thinking_and_answer(response)
            if thinking_content and answer_content:
                # Create OpenAI conversation format
                # Chosen: full original response (thinking + answer)
                # Rejected: only the answer part
                chosen_conversation = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
                
                rejected_conversation = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer_content}
                ]
                
                # Create new record with preference format
                preference_record = {
                    "chosen": chosen_conversation,
                    "rejected": rejected_conversation
                }
                
                # Copy other fields from original record (excluding prompt and response)
                for key, value in record.items():
                    if key not in [response_field, prompt_field]:
                        preference_record[key] = value
                
                preference_records.append(preference_record)
    
    return preference_records


def fetch_dataset_sample(
    dataset_name: str = "allenai/big-reasoning-traces",
    config_name: str = "DeepSeek",
    num_records: int = 1000,
    split: str = "train",
    streaming: bool = True
) -> Dataset:
    """
    Fetch the first X records from the specified dataset using streaming.
    
    Args:
        dataset_name: Name of the dataset on Hugging Face
        config_name: Dataset configuration name (e.g., 'DeepSeek', 'DeepSeek_debug')
        num_records: Number of records to fetch
        split: Dataset split to use
        streaming: Whether to use streaming mode
    
    Returns:
        Dataset: A Hugging Face Dataset object with the fetched records
    """
    logger.info(f"Loading dataset {dataset_name} with config {config_name} and streaming={streaming}")
    
    try:
        # Load dataset with streaming and config
        dataset = load_dataset(
            dataset_name,
            config_name,
            split=split,
            streaming=streaming
        )
        
        logger.info(f"Dataset loaded successfully. Fetching first {num_records} records...")
        
        # Collect the first num_records
        records = []
        for i, record in enumerate(dataset):
            if i >= num_records:
                break
            records.append(record)
            if (i + 1) % 100 == 0:
                logger.info(f"Fetched {i + 1} records...")
        
        logger.info(f"Successfully fetched {len(records)} records")
        
        # Transform records to preference format
        records = transform_to_preference_format(records, prompt_field="prompt")
        
        # Convert to regular Dataset (non-streaming)
        return Dataset.from_list(records)
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def upload_to_hf(
    dataset: Dataset,
    username: str,
    dataset_name: str,
    private: bool = False,
    token: Optional[str] = None
) -> str:
    """
    Upload the dataset to Hugging Face Hub.
    
    Args:
        dataset: The dataset to upload
        username: Your Hugging Face username
        dataset_name: Name for the new dataset
        private: Whether the dataset should be private
        token: Hugging Face token (if not provided, will use login)
    
    Returns:
        str: The URL of the uploaded dataset
    """
    if token:
        login(token=token)
    else:
        # Check if already logged in
        try:
            api = HfApi()
            api.whoami()
            logger.info("Already logged in to Hugging Face")
        except Exception:
            logger.info("Please log in to Hugging Face")
            login()
    
    # Create the full dataset name
    full_dataset_name = f"{username}/{dataset_name}"
    
    logger.info(f"Uploading dataset to {full_dataset_name}")
    
    try:
        # Push to hub
        dataset.push_to_hub(
            full_dataset_name,
            private=private,
            commit_message=f"Upload {len(dataset)} records from big-reasoning-traces in preference format (chosen/rejected)"
        )
        
        dataset_url = f"https://huggingface.co/datasets/{full_dataset_name}"
        logger.info(f"Dataset uploaded successfully: {dataset_url}")
        return dataset_url
        
    except Exception as e:
        logger.error(f"Error uploading dataset: {e}")
        raise


def main():
    """Main function to orchestrate the dataset fetching and uploading process."""
    parser = argparse.ArgumentParser(
        description="Fetch records from big-reasoning-traces and upload to Hugging Face"
    )
    parser.add_argument(
        "--num_records",
        type=int,
        default=1000,
        help="Number of records to fetch (default: 1000)"
    )
    parser.add_argument(
        "--hf_username",
        type=str,
        required=True,
        help="Your Hugging Face username"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name for the new dataset on your profile"
    )
    parser.add_argument(
        "--source_dataset",
        type=str,
        default="allenai/big-reasoning-traces",
        help="Source dataset name (default: allenai/big-reasoning-traces)"
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="DeepSeek",
        help="Dataset configuration name (default: DeepSeek, options: DeepSeek, DeepSeek_debug)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the uploaded dataset private"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        help="Hugging Face token (optional, will prompt for login if not provided)"
    )
    parser.add_argument(
        "--no_streaming",
        action="store_true",
        help="Disable streaming mode (uses more memory)"
    )
    
    args = parser.parse_args()
    
    logger.info("Starting dataset fetch and upload process")
    logger.info(f"Source: {args.source_dataset}")
    logger.info(f"Config: {args.config_name}")
    logger.info(f"Records to fetch: {args.num_records}")
    logger.info(f"Target: {args.hf_username}/{args.dataset_name}")
    
    try:
        # Fetch the dataset sample
        dataset = fetch_dataset_sample(
            dataset_name=args.source_dataset,
            config_name=args.config_name,
            num_records=args.num_records,
            split=args.split,
            streaming=not args.no_streaming
        )
        
        # Upload to Hugging Face
        dataset_url = upload_to_hf(
            dataset=dataset,
            username=args.hf_username,
            dataset_name=args.dataset_name,
            private=args.private,
            token=args.hf_token
        )
        
        logger.info("Process completed successfully!")
        logger.info(f"Dataset available at: {dataset_url}")
        
    except Exception as e:
        logger.error(f"Process failed: {e}")
        raise


if __name__ == "__main__":
    main() 