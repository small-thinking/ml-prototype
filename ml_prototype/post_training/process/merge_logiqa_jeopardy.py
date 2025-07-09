#!/usr/bin/env python3
"""
Script to merge jiacheng-ye/logiqa-zh and jeggers/jeopardy datasets into one unified dataset.

For jeopardy: keeps question and answer, renames them as prompt and completion
For logiqa-zh: concatenates context, query, and options into one string as prompt, 
and uses the correct option as completion

Usage:
    python merge_logiqa_jeopardy.py --hf_username tech-tao --dataset_name mini-reasoning-dataset --jeopardy_sample_ratio 0.2
"""

import argparse
import logging
from typing import Optional, List, Dict, Any
from datasets import load_dataset, Dataset, concatenate_datasets
from huggingface_hub import HfApi, login

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_logiqa_dataset(
    dataset_name: str = "jiacheng-ye/logiqa-zh",
    split: str = "train",
    max_samples: Optional[int] = None
) -> Dataset:
    """
    Process the logiqa-zh dataset by concatenating context, query, and options into a prompt,
    and using the correct option as completion.
    
    Args:
        dataset_name: Name of the logiqa dataset
        split: Dataset split to use
        max_samples: Maximum number of samples to process (None for all)
        
    Returns:
        Processed dataset with prompt/completion format
    """
    logger.info(f"Loading logiqa dataset: {dataset_name}")
    
    try:
        dataset = load_dataset(dataset_name, split=split)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
            logger.info(f"Selected {len(dataset)} samples from logiqa dataset")
        
        def format_logiqa_example(example: Dict[str, Any]) -> Dict[str, str]:
            """Format a single logiqa example into prompt/completion format."""
            context = example["context"]
            query = example["query"]
            options = example["options"]
            correct_option_idx = example["correct_option"]
            
            # Concatenate context, query, and options into prompt
            options_text = "\n".join([f"{i+1}. {option}" for i, option in enumerate(options)])
            prompt = f"{context}\n\n问题: {query}\n\n选项:\n{options_text}\n\n请选择正确的选项。"
            
            # Get the correct option as completion
            completion = options[correct_option_idx]
            
            return {
                "prompt": prompt,
                "completion": completion,
                "source": "logiqa-zh",
                "correct_option_idx": correct_option_idx
            }
        
        # Apply formatting to all examples
        processed_dataset = dataset.map(format_logiqa_example)
        logger.info(f"Processed {len(processed_dataset)} logiqa examples")
        
        return processed_dataset
        
    except Exception as e:
        logger.error(f"Error processing logiqa dataset: {e}")
        raise


def process_jeopardy_dataset(
    dataset_name: str = "jeggers/jeopardy",
    split: str = "train",
    sample_ratio: Optional[float] = 0.1,
    max_samples: Optional[int] = None
) -> Dataset:
    """
    Process the jeopardy dataset by keeping question and answer, renaming them as prompt and completion.
    
    Args:
        dataset_name: Name of the jeopardy dataset
        split: Dataset split to use
        sample_ratio: Ratio of samples to keep (0.0 to 1.0, default: 0.1)
        max_samples: Maximum number of samples to process (None for all)
        
    Returns:
        Processed dataset with prompt/completion format
    """
    logger.info(f"Loading jeopardy dataset: {dataset_name}")
    
    try:
        dataset = load_dataset(dataset_name, split=split)
        original_size = len(dataset)
        logger.info(f"Original jeopardy dataset size: {original_size}")
        
        # Apply sample ratio if specified
        if sample_ratio is not None:
            if not 0.0 < sample_ratio <= 1.0:
                raise ValueError("sample_ratio must be between 0.0 and 1.0")
            
            # Calculate number of samples to keep
            samples_to_keep = int(original_size * sample_ratio)
            # Randomly sample from the dataset
            import random
            random.seed(42)  # For reproducibility
            indices = random.sample(range(original_size), samples_to_keep)
            dataset = dataset.select(indices)
            logger.info(f"Applied sample ratio {sample_ratio}: selected {len(dataset)} samples from {original_size}")
        
        # Apply max_samples limit if specified (after ratio sampling)
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
            logger.info(f"Applied max_samples limit: selected {len(dataset)} samples from jeopardy dataset")
        
        def format_jeopardy_example(example: Dict[str, Any]) -> Dict[str, str]:
            """Format a single jeopardy example into prompt/completion format."""
            question = example["question"]
            answer = example["answer"]
            
            # Create prompt with category context
            prompt = f"Question/Puzzle: {question}"
            
            return {
                "prompt": prompt,
                "completion": answer,
                "source": "jeopardy",
                "value": example.get("value", None)
            }
        
        # Apply formatting to all examples
        processed_dataset = dataset.map(format_jeopardy_example)
        logger.info(f"Processed {len(processed_dataset)} jeopardy examples")
        
        return processed_dataset
        
    except Exception as e:
        logger.error(f"Error processing jeopardy dataset: {e}")
        raise


def merge_datasets(
    logiqa_dataset: Dataset,
    jeopardy_dataset: Dataset
) -> Dataset:
    """
    Merge the two processed datasets into one unified dataset.
    Only keeps prompt and completion fields.
    
    Args:
        logiqa_dataset: Processed logiqa dataset
        jeopardy_dataset: Processed jeopardy dataset
        
    Returns:
        Merged dataset with only prompt and completion fields
    """
    logger.info("Merging datasets...")
    
    # Concatenate datasets
    merged_dataset = concatenate_datasets([logiqa_dataset, jeopardy_dataset])
    
    logger.info(f"Merged dataset contains {len(merged_dataset)} total examples")
    logger.info(f"  - Logiqa examples: {len(logiqa_dataset)}")
    logger.info(f"  - Jeopardy examples: {len(jeopardy_dataset)}")
    
    # Remove all columns except prompt and completion
    columns_to_remove = [col for col in merged_dataset.column_names if col not in ["prompt", "completion"]]
    if columns_to_remove:
        merged_dataset = merged_dataset.remove_columns(columns_to_remove)
        logger.info(f"Removed columns: {columns_to_remove}")
    logger.info("Filtered dataset to keep only prompt and completion fields")
    
    return merged_dataset


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
            commit_message=f"Upload merged logiqa-zh and jeopardy dataset with {len(dataset)} examples"
        )
        
        dataset_url = f"https://huggingface.co/datasets/{full_dataset_name}"
        logger.info(f"Dataset uploaded successfully: {dataset_url}")
        return dataset_url
        
    except Exception as e:
        logger.error(f"Error uploading dataset: {e}")
        raise


def main():
    """Main function to orchestrate the dataset merging and uploading process."""
    parser = argparse.ArgumentParser(
        description="Merge logiqa-zh and jeopardy datasets and upload to Hugging Face"
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
        help="Name for the new merged dataset on your profile"
    )
    parser.add_argument(
        "--logiqa_max_samples",
        type=int,
        help="Maximum number of samples to use from logiqa dataset (default: all)"
    )
    parser.add_argument(
        "--jeopardy_max_samples",
        type=int,
        help="Maximum number of samples to use from jeopardy dataset (default: all)"
    )
    parser.add_argument(
        "--jeopardy_sample_ratio",
        type=float,
        help="Ratio of samples to keep from jeopardy dataset (0.0 to 1.0, e.g., 0.1 for 10%%, default: all)"
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
        "--no_upload",
        action="store_true",
        help="Skip uploading to Hugging Face (just create the merged dataset locally)"
    )
    parser.add_argument(
        "--no_shuffle",
        action="store_true",
        help="Do not shuffle the merged dataset (default: shuffle enabled)"
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)"
    )
    
    args = parser.parse_args()
    
    logger.info("Starting dataset merge process")
    logger.info(f"Logiqa max samples: {args.logiqa_max_samples or 'all'}")
    logger.info(f"Jeopardy max samples: {args.jeopardy_max_samples or 'all'}")
    logger.info(f"Jeopardy sample ratio: {args.jeopardy_sample_ratio or 'all'}")
    
    try:
        # Process logiqa dataset
        logiqa_dataset = process_logiqa_dataset(
            max_samples=args.logiqa_max_samples
        )
        
        # Process jeopardy dataset
        jeopardy_dataset = process_jeopardy_dataset(
            max_samples=args.jeopardy_max_samples,
            sample_ratio=args.jeopardy_sample_ratio
        )
        
        # Merge datasets
        merged_dataset = merge_datasets(logiqa_dataset, jeopardy_dataset)
        
        # Shuffle if not disabled
        if not args.no_shuffle:
            merged_dataset = merged_dataset.shuffle(seed=args.shuffle_seed)
            logger.info(f"Shuffled merged dataset with seed {args.shuffle_seed}")
        
        # Show some examples
        logger.info("\nExample from merged dataset:")
        logger.info(f"Prompt: {merged_dataset[0]['prompt'][:200]}...")
        logger.info(f"Completion: {merged_dataset[0]['completion']}")
        
        if not args.no_upload:
            # Upload to Hugging Face
            dataset_url = upload_to_hf(
                dataset=merged_dataset,
                username=args.hf_username,
                dataset_name=args.dataset_name,
                private=args.private,
                token=args.hf_token
            )
            
            logger.info("Process completed successfully!")
            logger.info(f"Dataset available at: {dataset_url}")
        else:
            logger.info("Process completed successfully! Dataset not uploaded (--no_upload flag used)")
            logger.info(f"Local dataset contains {len(merged_dataset)} examples")
        
    except Exception as e:
        logger.error(f"Process failed: {e}")
        raise


if __name__ == "__main__":
    main() 