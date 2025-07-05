#!/usr/bin/env python3
"""
Script to fetch the first X records from the big-reasoning-traces dataset
and upload them to a specified Hugging Face profile.

Usage:
    python fetch_and_upload_dataset.py --num_records 10000 --hf_username tech-tao --dataset_name my-reasoning-traces-10k --config_name DeepSeek
"""

import argparse
import logging
from typing import Optional, Dict, Any
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi, login
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a thoughtful assistant that can conduct reasoning tasks.
You will be given a question and you would need to reason step by step to answer the question.
Place your reasoning between the <thinking> and </thinking> tags.
Then provide your final answer between the <answer> and </answer> tags.
"""


def inject_system_prompt(
    records: list[dict],
    prompt: str = SYSTEM_PROMPT,
    message_field: str = "messages"
) -> list[dict]:
    """
    Inject a system prompt as a message with role 'system' at the beginning of each record's message field.
    If the message field is a string, convert it to a list of messages.
    Args:
        records: List of dataset records (dicts)
        prompt: System prompt to inject
        message_field: Field name to inject prompt into
    Returns:
        List of modified records
    """
    for record in records:
        if message_field in record:
            # If message is a string, convert to list
            if isinstance(record[message_field], str):
                record[message_field] = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": record[message_field]}
                ]
            elif isinstance(record[message_field], list):
                # Insert system prompt at the beginning if not already present
                if not (record[message_field] and record[message_field][0].get("role") == "system"):
                    record[message_field].insert(0, {"role": "system", "content": prompt})
    return records


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
        
        # Inject system prompt into each record's message field
        records = inject_system_prompt(records)
        
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
            commit_message=f"Upload {len(dataset)} records from big-reasoning-traces"
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