from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import re
import random
import os
from datetime import datetime
from peft import LoraConfig


model_size = "3B"
use_lora = False  # Set to False for full fine-tuning

if model_size == "8B":
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
elif model_size == "3B":
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
elif model_size == "0.5B":
    model_name = "Qwen/Qwen2-0.5B-Instruct"
elif model_size == "1.5B":
    model_name = "Qwen/Qwen2-1.5B-Instruct"
else:
    raise ValueError(f"Invalid model size: {model_size}")

# tech-tao/mini-reasoning-dataset dataset
dataset = load_dataset("tech-tao/mini-reasoning-dataset", split="train")
dataset = dataset.map(
    lambda x: {
        "prompt": """
        The following question requires reasoning.
        In addition to provide your answer, you should also provide your DETAILED thought process about how you arrive at your answer.
        Put your thought process between <think></think> tags and then put your answer between <answer></answer> tags.

        The question is:
        """ + x["prompt"],
        "ground_truth": x["completion"]
    }
)

# Build an index keyed by ground truth and value is the prompt
index = {}
for i, row in enumerate(dataset):
    index[row["ground_truth"]] = row["prompt"]

reasoning_start = "<think>"
reasoning_end = "</think>"
answer_start = "<answer>"
answer_end = "</answer>"

# Setup logging file (moved outside function to avoid recreation on each call)
log_dir = "debug_logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"grpo_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# Global step counter
step_counter = 0


def match_format_func(completions, **kwargs):
    """Format penalty function: perfect format gets 0, violations get penalties"""
    scores = []
    # regex for matching the format like this: <think>content</think><answer>content</answer>
    match_format = re.compile(
        rf"^[\s]{{0,}}"\
        rf"{reasoning_start}.*?{reasoning_end}"\
        rf"{answer_start}.*?{answer_end}"\
        rf"[\s]{{0,}}$",
        flags = re.MULTILINE | re.DOTALL
    )
    for completion in completions:
        penalty = 0
        # ===== FORMAT COMPLIANCE CHECKING =====
        if match_format.search(completion) is not None:
            # Format is perfect - no penalty, skip all other checks
            scores.append(penalty)
            continue
        else:
            # 1. Missing or incorrect tags
            penalty -= 1.0 if completion.count(reasoning_start) != 1 else 0
            penalty -= 1.0 if completion.count(reasoning_end) != 1 else 0
            penalty -= 1.0 if completion.count(answer_start) != 1 else 0
            penalty -= 1.0 if completion.count(answer_end) != 1 else 0
        
        # ===== CONTENT STRUCTURE PENALTIES =====
        # 2. Unwrapped content (content not in <think> or <answer> tags)
        content_without_tags = re.sub(r'<think>.*?</think>', '', completion, flags=re.DOTALL)
        content_without_tags = re.sub(r'<answer>.*?</answer>', '', content_without_tags, flags=re.DOTALL)
        content_without_tags = content_without_tags.strip()
        
        if content_without_tags:
            penalty -= 5.0  # Penalty for unwrapped content
        
        # 3. Wrong order (answer before thinking)
        think_pos = completion.find(reasoning_start)
        answer_pos = completion.find(answer_start)
        
        if think_pos != -1 and answer_pos != -1:  # Both sections exist
            if answer_pos < think_pos:  # Answer comes before thinking
                penalty -= 1.0  # Penalty for wrong order
        
        # 4. Multiple sections (should be exactly one of each)
        think_count = completion.count(reasoning_start)
        answer_count = completion.count(answer_start)
        
        if think_count > 1:
            penalty -= 2.0  # Penalty for multiple think sections
        if answer_count > 1:
            penalty -= 2.0  # Penalty for multiple answer sections
                
        scores.append(penalty)
    return scores


def penalize_short_think_func(completions, **kwargs):
    """Penalize thinking sections that are too short with gradual penalty under 200 characters"""
    scores = []
    for completion in completions:
        score = 0
        # Extract thinking content
        think_match = re.search(rf"{reasoning_start}(.+?){reasoning_end}", completion, flags=re.DOTALL)
        if think_match:
            think_content = think_match.group(1).strip()
        else:
            think_content = completion
        content_length = len(think_content)
        # Gradual penalty for short thinking (under 200 characters)
        if content_length < 200:
            penalty_ratio = (200 - content_length) / 200
            score -= 10.0 * penalty_ratio  # Gradual penalty from 0 to -10.0
        scores.append(score)
    return scores


def check_answer_func(completions, ground_truth, **kwargs):
    """Reward if the answer is correct with partial matching for knowledge-based scoring"""
    global step_counter
    step_counter += 1
    
    scores = []
    
    # Always print when there's a full score, occasionally print other cases
    should_print = False
    print_reason = ""
    first_completion = completions[0]
    answer_match = re.search(rf"{answer_start}\s*(.+?)\s*{answer_end}", first_completion, flags=re.DOTALL)
    if answer_match:
        extracted_answer = answer_match.group(1).strip()
        if extracted_answer.lower() == ground_truth[0].lower():
            should_print = True
            print_reason = "🎯 FULL SCORE (8.0) - Exact match!"
        elif random.random() < 0.1:  # 10% chance for other cases
            should_print = True
            if ground_truth[0].lower() in extracted_answer.lower():
                print_reason = "✅ PARTIAL SCORE (3.0) - Contains ground truth"
            else:
                print_reason = "❌ WRONG ANSWER (-1.0) - No match"
    elif random.random() < 0.1:  # 10% chance for no tags case
        should_print = True
        print_reason = "❌ No answer tags found (-1.0 penalty)"
    
    if should_print:
        # Calculate individual function scores for debugging
        format_reward = match_format_func([first_completion])[0]
        think_reward = penalize_short_think_func([first_completion])[0]
        
        # Calculate answer score manually to avoid recursion
        answer_reward = 0
        answer_match_debug = re.search(rf"{answer_start}\s*(.+?)\s*{answer_end}", first_completion, flags=re.DOTALL)
        if answer_match_debug is None:
            answer_reward = -1.0
        else:
            answer = answer_match_debug.group(1).strip()
            if answer.lower() == ground_truth[0].lower():
                answer_reward = 8.0
            elif ground_truth[0].lower() in answer.lower():
                answer_reward = 3.0
            else:
                answer_reward = -3.0
        
        total_reward = format_reward + think_reward + answer_reward
        
        debug_output = []
        debug_output.append("\n" + "="*60)
        debug_output.append(f"SPOT CHECK: PROMPT AND COMPLETIONS (Step: {step_counter})")
        debug_output.append("="*60)
        debug_output.append(f"==Prompt:==\n {index[ground_truth[0]]}\n")
        debug_output.append(f"==Completion:==\n {first_completion}\n")
        debug_output.append(f"==Ground Truth:==\n {ground_truth[0] if ground_truth else 'N/A'}")
        if answer_match:
            debug_output.append(f"==Extracted Answer: '{extracted_answer}'\n")
        debug_output.append(f"{print_reason}")
        debug_output.append(f"==SCORE BREAKDOWN:==")
        debug_output.append(f"  Format reward: {format_reward}")
        debug_output.append(f"  Think reward: {think_reward}")
        debug_output.append(f"  Answer reward: {answer_reward}")
        debug_output.append(f"  TOTAL REWARD: {total_reward}")
        debug_output.append("="*60 + "\n")
        
        # Print to console
        for line in debug_output:
            print(line)
        
        # Write to file
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write('\n'.join(debug_output))
            f.write('\n')
    
    for completion, ground_truth in zip(completions, ground_truth):
        score = 0
        # extract the answer from the completion with error handling
        answer_match = re.search(rf"{answer_start}\s*(.+?)\s*{answer_end}", completion, flags=re.DOTALL)
        if answer_match is None:
            # No answer tags found - treat as wrong answer
            score -= 1.0
        else:
            answer = answer_match.group(1).strip()
            
            # Exact match gets full score (including case-insensitive)
            if answer.lower() == ground_truth.lower():
                score += 8.0
            # Partial match if answer contains ground truth
            elif ground_truth.lower() in answer.lower():
                score += 3.0  # Small portion for partial match
            else:
                score -= 1.0  # Penalty for wrong answers
                    
        scores.append(score)
    return scores


# LoRA configuration
lora_config = None
if use_lora:
    lora_config = LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=32,  # LoRA alpha parameter
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Target attention modules
        lora_dropout=0.01,
        bias="none",
        task_type="CAUSAL_LM"
    )

training_args = GRPOConfig(
    output_dir=f"{model_name}-{'LoRA' if use_lora else 'Full'}-GRPO",
    learning_rate=1e-6,
    temperature=1.0,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_generations=8,
    max_prompt_length=768,
    max_steps=500,
    report_to="wandb",
    run_name=f"{model_name}-{'LoRA' if use_lora else 'Full'}-GRPO",
    fp16=True,
    fp16_full_eval=False,
    fp16_opt_level="O1",
    save_strategy="no",  # Don't save intermediate checkpoints
    save_total_limit=1,  # Keep only the last checkpoint
)

trainer = GRPOTrainer(
    model=model_name,
    reward_funcs=[match_format_func, penalize_short_think_func, check_answer_func],
    args=training_args,
    train_dataset=dataset,
    peft_config=lora_config,  # Pass LoRA config if using LoRA
)

trainer.train()
