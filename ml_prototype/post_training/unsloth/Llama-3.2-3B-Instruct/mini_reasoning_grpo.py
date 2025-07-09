from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import re

# # dataset = load_dataset("trl-lib/tldr", split="train")
# dataset = load_dataset("tech-tao/my-reasoning-traces-10k", "default", split = "train")
# # process to rename presponse as completions
# dataset = dataset.rename_column("response", "completion").remove_columns(["text", "messages", "source", "num_tokens"])

# # Change the content of the prompt by adding CoT required format
# dataset = dataset.map(lambda x: {"prompt": "Answer the following question step by step. Put your thought in <think> tags and your answer in <answer> tags. " + x["prompt"]})

# jeggers/jeopardy dataset
dataset = load_dataset("jeggers/jeopardy", split="train")
# dataset = dataset.rename_column("question", "prompt").rename_column("answer", "ground_truth")
dataset = dataset.map(
    lambda x: {
        "prompt": """
        The following question is about Jeopardy!
        In addition to provide your answer, you should also provide your DETAILED thought process about how you arrive at your answer.
        Put your thought in <think>your thought process</think> tags and then put your answer in <answer>your answer</answer> tags.

        The question is:
        """ + x["question"],
        "ground_truth": x["answer"]
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

match_format = re.compile(
    rf"^[\s]{{0,}}"\
    rf"{reasoning_start}.+?{reasoning_end}.*?"\
    rf"{answer_start}(.+?){answer_end}"\
    rf"[\s]{{0,}}$",
    flags = re.MULTILINE | re.DOTALL
)

def match_format_func(completions, **kwargs):
    """Format penalty function: perfect format gets 0, violations get penalties"""
    scores = []
    for completion in completions:
        penalty = 0
        
        # ===== FORMAT COMPLIANCE CHECKING =====
        # Perfect format gets no penalty
        if match_format.search(completion) is not None:
            # Format is perfect - no penalty
            pass
        else:
            # ===== FORMAT VIOLATION PENALTIES =====
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
            content_length = len(think_content)
            
            # Gradual penalty for short thinking (under 200 characters)
            if content_length < 200:
                # Linear penalty: 0 at 200 chars, -5.0 at 0 chars
                penalty_ratio = (200 - content_length) / 200
                score -= 5.0 * penalty_ratio  # Gradual penalty from 0 to -5.0
        else:
            # No thinking section at all - already penalized by format function
            pass
            
        scores.append(score)
    return scores


def check_answer_func(completions, ground_truth, **kwargs):
    """Reward if the answer is correct with partial matching for knowledge-based scoring"""
    scores = []
    
    # Always print when there's a full score, occasionally print other cases
    import random
    should_print = False
    print_reason = ""
    
    # Check first completion for full score
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
        print_reason = "==No answer tags found (-1.0 penalty)=="
    
    if should_print:
        print("\n" + "="*60)
        print("DEBUG: PROMPT AND COMPLETIONS")
        print("="*60)
        print(f"==Prompt:==\n {index[ground_truth[0]]}\n")
        print(f"==Completion:==\n {first_completion}\n")
        print(f"==Ground Truth:==\n {ground_truth[0] if ground_truth else 'N/A'}")
        if answer_match:
            print(f"==Extracted Answer: '{extracted_answer}'\n")
        print(f"{print_reason}")
        print("="*60 + "\n")
    
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


training_args = GRPOConfig(
    output_dir="Qwen2-1.5B-GRPO",
    learning_rate=1e-5,
    warmup_ratio=0.1,lr_scheduler_type="cosine",
    logging_steps=1,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=8,
    num_generations=32,
    max_prompt_length=1024,
    max_steps=500,
    report_to="wandb",
    run_name="qwen-1.5b-grpo"
)
trainer = GRPOTrainer(
    model="Qwen/Qwen2-1.5B-Instruct",
    reward_funcs=[match_format_func, penalize_short_think_func, check_answer_func],
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
