"""
GRPO Fine-tuning for Llama-3.2-3B-Instruct

This script contains a complete, well-organized implementation of GRPO fine-tuning
for the Llama-3.2-3B-Instruct model on reasoning traces dataset.

Key Features:
- Modular design within a single file
- Comprehensive configuration management
- Robust data processing and reward functions
- Memory-efficient LoRA fine-tuning
- Proper error handling and logging
"""
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """Model configuration parameters."""
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    max_seq_length: int = 2048
    load_in_4bit: bool = False
    fast_inference: bool = False  # True if using VLLM
    max_lora_rank: int = 64
    gpu_memory_utilization: float = 0.6


@dataclass
class LoRAConfig:
    """LoRA configuration parameters."""
    r: int = 64
    lora_alpha: int = 64
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 3407
    target_modules: List[str] = None

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]


@dataclass
class DatasetConfig:
    """Dataset configuration parameters."""
    dataset_name: str = "tech-tao/my-reasoning-traces-10k"
    dataset_split: str = None  # None means no config, use default
    dataset_subset: str = "train"
    max_response_length: int = 1024  # Maximum response length in tokens


@dataclass
class PromptConfig:
    """Prompt formatting configuration."""
    reasoning_start: str = "<think>"
    reasoning_end: str = "</think>"
    answer_start: str = "<answer>"
    answer_end: str = "</answer>"
    
    @property
    def system_prompt(self) -> str:
        return f"""
        You are a thoughtful assistant.
        You will first think systematically before answering.
        Place your thought process between {self.reasoning_start} and {self.reasoning_end}.
        Then, provide your answer between {self.answer_start} and {self.answer_end}
        """


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    learning_rate: float = 5e-6
    weight_decay: float = 0.1
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_8bit"
    logging_steps: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_generations: int = 4
    max_prompt_length: int = 1024
    max_steps: int = 500
    save_steps: int = 250
    max_grad_norm: float = 1.0
    report_to: str = "none"
    output_dir: str = "outputs"


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = None
    lora: LoRAConfig = None
    dataset: DatasetConfig = None
    prompt: PromptConfig = None
    training: TrainingConfig = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.lora is None:
            self.lora = LoRAConfig()
        if self.dataset is None:
            self.dataset = DatasetConfig()
        if self.prompt is None:
            self.prompt = PromptConfig()
        if self.training is None:
            self.training = TrainingConfig()


# =============================================================================
# DATA PROCESSING
# =============================================================================

class DataProcessor:
    """Handles dataset loading and formatting for GRPO training."""
    
    def __init__(self, config: Config):
        self.config = config
        self._setup_regex_patterns()
    
    def _setup_regex_patterns(self) -> None:
        """Setup regex patterns for answer extraction and format matching."""
        prompt_config = self.config.prompt
        
        # Pattern to match the expected response format
        self.match_format = re.compile(
            rf"^[\s]{{0,}}"\
            rf"{re.escape(prompt_config.reasoning_start)}.+?{re.escape(prompt_config.reasoning_end)}.*?"\
            rf"{re.escape(prompt_config.answer_start)}(.+?){re.escape(prompt_config.answer_end)}"\
            rf"[\s]{{0,}}$",
            flags=re.MULTILINE | re.DOTALL
        )
        
        # Pattern to extract numbers from answer tags
        self.match_numbers = re.compile(
            re.escape(prompt_config.answer_start) + r".*?([\d\.\,]{1,})",
            flags=re.MULTILINE | re.DOTALL
        )
    
    def extract_hash_answer(self, text: str) -> Optional[str]:
        """Extract the answer from text containing #### separator.
        
        Args:
            text: Input text that may contain #### separator
            
        Returns:
            Extracted answer or None if not found
        """
        if "####" not in text:
            return None
        return text.split("####")[1].strip()
    
    def load_dataset(self):
        """Load the reasoning traces dataset.
        
        Returns:
            Loaded dataset
        """
        if self.config.dataset.dataset_split is None:
            # Load without config
            dataset = load_dataset(
                self.config.dataset.dataset_name,
                split=self.config.dataset.dataset_subset
            )
        else:
            # Try with config first
            try:
                dataset = load_dataset(
                    self.config.dataset.dataset_name,
                    self.config.dataset.dataset_split,
                    split=self.config.dataset.dataset_subset
                )
            except ValueError as e:
                if "not found" in str(e):
                    # If config not found, try without config
                    print(f"Config '{self.config.dataset.dataset_split}' not found, trying without config...")
                    dataset = load_dataset(
                        self.config.dataset.dataset_name,
                        split=self.config.dataset.dataset_subset
                    )
                else:
                    raise e
        
        return dataset
    
    def format_dataset(self, dataset):
        """Format dataset for GRPO training with proper prompt structure.
        
        Args:
            dataset: Raw dataset
            
        Returns:
            Formatted dataset ready for training
        """
        def format_example(x: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "prompt": [
                    {"role": "system", "content": self.config.prompt.system_prompt},
                    {"role": "user", "content": x["prompt"]},
                ],
                "completion": self.extract_hash_answer(x["response"]),
            }
        
        return dataset.map(format_example)
    
    def filter_by_length(self, dataset, tokenizer):
        """Filter dataset to remove examples with responses that are too long.
        
        Args:
            dataset: Dataset to filter
            tokenizer: Tokenizer for length calculation
            
        Returns:
            Filtered dataset
        """
        print(f"Filtering dataset: removing responses longer than {self.config.dataset.max_response_length} tokens")
        
        original_size = len(dataset)
        
        def filter_example(example):
            response_text = example["response"]
            response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
            return len(response_tokens) <= self.config.dataset.max_response_length
        
        filtered_dataset = dataset.filter(filter_example)
        filtered_size = len(filtered_dataset)
        
        print(f"Dataset size: {original_size} -> {filtered_size} (removed {original_size - filtered_size} examples)")
        
        return filtered_dataset
    
    def peek_dataset(self, dataset, num_examples: int = 1) -> None:
        """Display sample examples from the dataset.
        
        Args:
            dataset: Dataset to peek at
            num_examples: Number of examples to display
        """
        print(f"Dataset size: {len(dataset)}")
        print(f"Sample examples:")
        
        for i in range(min(num_examples, len(dataset))):
            print(f"\n--- Example {i+1} ---")
            if 'messages' in dataset[i]:
                print(f"Messages: {dataset[i]['messages']}")
            elif 'prompt' in dataset[i]:
                print(f"Prompt: {dataset[i]['prompt']}")
            print(f"Response: {dataset[i]['response']}")
            if 'answer' in dataset[i]:
                print(f"Extracted answer: {dataset[i]['answer']}")
    
    def test_format_matching(self) -> None:
        """Test the format matching regex patterns."""
        test_response = (
            f"{self.config.prompt.reasoning_start}Let me think!{self.config.prompt.reasoning_end}"
            f"{self.config.prompt.answer_start}2{self.config.prompt.answer_end}"
        )
        
        match = self.match_format.search(test_response)
        print(f"Test response: {test_response}")
        print(f"Format match: {match is not None}")
        if match:
            print(f"Extracted answer: {match.group(1)}")
    
    def test_number_extraction(self) -> None:
        """Test number extraction from answer tags."""
        test_cases = [
            f"{self.config.prompt.answer_start}  0.34  {self.config.prompt.answer_end}",
            f"{self.config.prompt.answer_start}  123,456  {self.config.prompt.answer_end}",
        ]
        
        for test_case in test_cases:
            numbers = self.match_numbers.findall(test_case)
            print(f"Test case: {test_case}")
            print(f"Extracted numbers: {numbers}")


# =============================================================================
# MODEL SETUP
# =============================================================================

class ModelSetup:
    """Handles model loading and LoRA configuration."""
    
    def __init__(self, config: Config):
        self.config = config
        self._setup_environment()
    
    def _setup_environment(self) -> None:
        """Setup environment variables for GPU usage."""
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    def load_model_and_tokenizer(self) -> Tuple[FastLanguageModel, any]:
        """Load the base model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        print(f"Loading model: {self.config.model.model_name}")
        print(f"Max sequence length: {self.config.model.max_seq_length}")
        print(f"LoRA rank: {self.config.model.max_lora_rank}")
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model.model_name,
            max_seq_length=self.config.model.max_seq_length,
            load_in_4bit=self.config.model.load_in_4bit,
            fast_inference=self.config.model.fast_inference,
            max_lora_rank=self.config.model.max_lora_rank,
            gpu_memory_utilization=self.config.model.gpu_memory_utilization,
        )
        
        return model, tokenizer
    
    def setup_lora(self, model: FastLanguageModel) -> FastLanguageModel:
        """Setup LoRA configuration for the model.
        
        Args:
            model: Base model to configure
            
        Returns:
            Model with LoRA configuration applied
        """
        print(f"Setting up LoRA with rank: {self.config.lora.r}")
        print(f"Target modules: {self.config.lora.target_modules}")
        
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.config.lora.r,
            target_modules=self.config.lora.target_modules,
            lora_alpha=self.config.lora.lora_alpha,
            use_gradient_checkpointing=self.config.lora.use_gradient_checkpointing,
            random_state=self.config.lora.random_state,
        )
        
        return model
    
    def get_model_info(self, model: FastLanguageModel) -> None:
        """Print model information and memory usage.
        
        Args:
            model: Model to analyze
        """
        print("\n=== Model Information ===")
        print(f"Model type: {type(model)}")
        print(f"Device: {next(model.parameters()).device}")
        
        # Calculate model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
        
        # Memory usage
        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    def setup_complete_model(self) -> Tuple[FastLanguageModel, any]:
        """Complete model setup including LoRA configuration.
        
        Returns:
            Tuple of (configured model, tokenizer)
        """
        # Load base model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer()
        
        # Setup LoRA
        model = self.setup_lora(model)
        
        # Print model information
        self.get_model_info(model)
        
        return model, tokenizer


# =============================================================================
# REWARD FUNCTIONS
# =============================================================================

class RewardFunctions:
    """Collection of reward functions for GRPO training."""
    
    def __init__(self, config: Config, data_processor):
        self.config = config
        self.data_processor = data_processor
    
    def match_format_exactly(self, completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
        """Reward if the format is seen exactly.
        
        Args:
            completions: List of completion responses
            
        Returns:
            List of reward scores
        """
        # Debug: Print the structure of completions
        print(f"DEBUG: completions type: {type(completions)}")
        print(f"DEBUG: completions length: {len(completions)}")
        if completions:
            print(f"DEBUG: completions[0] type: {type(completions[0])}")
            print(f"DEBUG: completions[0] value: {completions[0]}")
            if isinstance(completions[0], list) and completions[0]:
                print(f"DEBUG: completions[0][0] type: {type(completions[0][0])}")
                print(f"DEBUG: completions[0][0] value: {completions[0][0]}")
        
        scores = []
        for completion in completions:
            score = 0.0
            
            # Handle different possible data structures
            if isinstance(completion, str):
                response = completion
            elif isinstance(completion, list) and completion:
                if isinstance(completion[0], dict):
                    response = completion[0]["content"]
                elif isinstance(completion[0], str):
                    response = completion[0]
                else:
                    print(f"DEBUG: Unexpected completion[0] type: {type(completion[0])}")
                    response = str(completion[0])
            elif isinstance(completion, dict):
                response = completion.get("content", str(completion))
            else:
                print(f"DEBUG: Unexpected completion type: {type(completion)}")
                response = str(completion)
            
            # Match if format is seen exactly!
            if self.data_processor.match_format.search(response) is not None:
                score += 3.0
            scores.append(score)
        
        return scores
    
    def match_format_approximately(self, completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
        """Reward if the format is seen approximately.
        
        Args:
            completions: List of completion responses
            
        Returns:
            List of reward scores
        """
        scores = []
        prompt_config = self.config.prompt
        
        for completion in completions:
            score = 0.0
            
            # Handle different possible data structures
            if isinstance(completion, str):
                response = completion
            elif isinstance(completion, list) and completion:
                if isinstance(completion[0], dict):
                    response = completion[0]["content"]
                elif isinstance(completion[0], str):
                    response = completion[0]
                else:
                    response = str(completion[0])
            elif isinstance(completion, dict):
                response = completion.get("content", str(completion))
            else:
                response = str(completion)
            
            # Count how many keywords are seen - we penalize if too many!
            # If we see 1, then plus some points!
            score += 0.5 if response.count(prompt_config.reasoning_start) == 1 else -1.0
            score += 0.5 if response.count(prompt_config.reasoning_end) == 1 else -1.0
            score += 0.5 if response.count(prompt_config.answer_start) == 1 else -1.0
            score += 0.5 if response.count(prompt_config.answer_end) == 1 else -1.0
            
            scores.append(score)
        
        return scores
    
    def check_answer(self, prompts: List[List[Dict[str, str]]], 
                    completions: List[List[Dict[str, str]]], 
                    answer: List[str], **kwargs) -> List[float]:
        """Check if the extracted answer matches the ground truth.
        
        Args:
            prompts: List of prompt sequences
            completions: List of completion responses
            answer: List of ground truth answers
            
        Returns:
            List of reward scores
        """
        # Extract responses with flexible handling
        responses = []
        for completion in completions:
            if isinstance(completion, str):
                responses.append(completion)
            elif isinstance(completion, list) and completion:
                if isinstance(completion[0], dict):
                    responses.append(completion[0]["content"])
                elif isinstance(completion[0], str):
                    responses.append(completion[0])
                else:
                    responses.append(str(completion[0]))
            elif isinstance(completion, dict):
                responses.append(completion.get("content", str(completion)))
            else:
                responses.append(str(completion))
        
        extracted_responses = [
            guess.group(1) if (guess := self.data_processor.match_format.search(r)) is not None else None
            for r in responses
        ]
        
        scores = []
        for guess, true_answer in zip(extracted_responses, answer):
            score = 0.0
            if guess is None:
                scores.append(0.0)
                continue
            
            # Correct answer gets 3 points!
            if guess == true_answer:
                score += 3.0
            # Match if spaces are seen, but less reward
            elif guess.strip() == true_answer.strip():
                score += 1.5
            else:
                # We also reward it if the answer is close via ratios!
                # Ie if the answer is within some range, reward it!
                try:
                    ratio = float(guess) / float(true_answer)
                    if 0.9 <= ratio <= 1.1:
                        score += 1.0
                    elif 0.8 <= ratio <= 1.2:
                        score += 0.5
                    else:
                        score -= 1.5  # Penalize wrong answers
                except (ValueError, ZeroDivisionError):
                    score -= 1.5  # Penalize non-numeric answers
            
            scores.append(score)
        
        return scores
    
    def check_numbers(self, prompts: List[List[Dict[str, str]]], 
                     completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
        """Check if numbers are properly formatted in answers.
        
        Args:
            prompts: List of prompt sequences
            completions: List of completion responses
            
        Returns:
            List of reward scores
        """
        scores = []
        
        for completion in completions:
            score = 0.0
            
            # Handle different possible data structures
            if isinstance(completion, str):
                response = completion
            elif isinstance(completion, list) and completion:
                if isinstance(completion[0], dict):
                    response = completion[0]["content"]
                elif isinstance(completion[0], str):
                    response = completion[0]
                else:
                    response = str(completion[0])
            elif isinstance(completion, dict):
                response = completion.get("content", str(completion))
            else:
                response = str(completion)
            
            # Extract numbers from answer tags
            numbers = self.data_processor.match_numbers.findall(response)
            
            # Reward if numbers are found in answer tags
            if numbers:
                score += 0.5
            
            scores.append(score)
        
        return scores
    
    def get_all_reward_functions(self) -> List:
        """Get all reward functions for GRPO training.
        
        Returns:
            List of reward function references
        """
        return [
            self.match_format_exactly,
            # self.match_format_approximately,
            # self.check_answer,
            # self.check_numbers,
        ]


# =============================================================================
# TRAINING
# =============================================================================

class GRPOTrainerSetup:
    """Handles GRPO training configuration and execution."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def create_training_config(self) -> GRPOConfig:
        """Create GRPO training configuration.
        
        Returns:
            GRPOConfig object with training parameters
        """
        training_config = self.config.training
        
        args = GRPOConfig(
            learning_rate=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            warmup_ratio=training_config.warmup_ratio,
            lr_scheduler_type=training_config.lr_scheduler_type,
            optim=training_config.optim,
            logging_steps=training_config.logging_steps,
            per_device_train_batch_size=training_config.per_device_train_batch_size,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            num_generations=training_config.num_generations,
            max_prompt_length=training_config.max_prompt_length,
            max_completion_length=self.config.model.max_seq_length - training_config.max_prompt_length,
            max_steps=training_config.max_steps,
            save_steps=training_config.save_steps,
            max_grad_norm=training_config.max_grad_norm,
            report_to=training_config.report_to,
            output_dir=training_config.output_dir,
        )
        
        return args
    
    def create_trainer(self, model: Any, tokenizer: Any, 
                      reward_functions: list, dataset: Any) -> GRPOTrainer:
        """Create GRPO trainer with all components.
        
        Args:
            model: Fine-tuned model
            tokenizer: Tokenizer
            reward_functions: List of reward functions
            dataset: Training dataset
            
        Returns:
            Configured GRPOTrainer
        """
        args = self.create_training_config()
        
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=reward_functions,
            args=args,
            train_dataset=dataset,
        )
        
        return trainer
    
    def print_training_config(self) -> None:
        """Print training configuration summary."""
        training_config = self.config.training
        
        print("\n=== Training Configuration ===")
        print(f"Learning rate: {training_config.learning_rate}")
        print(f"Batch size: {training_config.per_device_train_batch_size}")
        print(f"Gradient accumulation steps: {training_config.gradient_accumulation_steps}")
        print(f"Effective batch size: {training_config.per_device_train_batch_size * training_config.gradient_accumulation_steps}")
        print(f"Max steps: {training_config.max_steps}")
        print(f"Save steps: {training_config.save_steps}")
        print(f"Max prompt length: {training_config.max_prompt_length}")
        print(f"Max completion length: {self.config.model.max_seq_length - training_config.max_prompt_length}")
        print(f"Number of generations: {training_config.num_generations}")
        print(f"Output directory: {training_config.output_dir}")
    
    def train(self, model: Any, tokenizer: Any, 
              reward_functions: list, dataset: Any) -> None:
        """Execute the complete training process.
        
        Args:
            model: Fine-tuned model
            tokenizer: Tokenizer
            reward_functions: List of reward functions
            dataset: Training dataset
        """
        print("\n=== Starting GRPO Training ===")
        
        # Print configuration
        self.print_training_config()
        
        # Create trainer
        trainer = self.create_trainer(model, tokenizer, reward_functions, dataset)
        
        # Start training
        print("\nStarting training...")
        trainer.train()
        
        print("\nTraining completed!")
        print(f"Model saved to: {self.config.training.output_dir}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main training pipeline."""
    print("=== GRPO Fine-tuning Pipeline ===")
    print("Model: Llama-3.2-3B-Instruct")
    print("Dataset: Reasoning Traces")
    print("=" * 40)
    
    # Initialize configuration
    config = Config()
    
    try:
        # Step 1: Setup model and tokenizer
        print("\n1. Setting up model and tokenizer...")
        model_setup = ModelSetup(config)
        model, tokenizer = model_setup.setup_complete_model()
        
        # Step 2: Load and process dataset
        print("\n2. Loading and processing dataset...")
        data_processor = DataProcessor(config)
        
        # Load raw dataset
        raw_dataset = data_processor.load_dataset()
        print(f"Loaded dataset with {len(raw_dataset)} examples")
        
        # Filter dataset by length (before formatting)
        filtered_dataset = data_processor.filter_by_length(raw_dataset, tokenizer)
        
        # Format dataset for training
        formatted_dataset = data_processor.format_dataset(filtered_dataset)
        print(f"Formatted dataset ready for training")
        
        # Optional: Peek at dataset
        data_processor.peek_dataset(formatted_dataset, num_examples=2)
        
        # Optional: Test format matching
        print("\n3. Testing format matching...")
        data_processor.test_format_matching()
        data_processor.test_number_extraction()
        
        # Step 3: Setup reward functions
        print("\n4. Setting up reward functions...")
        reward_functions = RewardFunctions(config, data_processor)
        reward_funcs = reward_functions.get_all_reward_functions()
        print(f"Configured {len(reward_funcs)} reward functions")
        
        # Step 4: Setup and start training
        print("\n5. Setting up training...")
        trainer_setup = GRPOTrainerSetup(config)
        
        # Execute training
        trainer_setup.train(model, tokenizer, reward_funcs, formatted_dataset)
        
        print("\n=== Training Pipeline Completed Successfully ===")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        raise


if __name__ == "__main__":
    main() 