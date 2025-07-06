from unsloth import FastLanguageModel
import torch
import os
from datasets import load_dataset
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def extract_hash_answer(text):
    # Extract the answer from the text
    if "####" not in text: return None
    return text.split("####")[1].strip()

max_seq_length = 4096 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

# Load the model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/Llama-3.2-3B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = False, # False for LoRA 16bit
    fast_inference = False, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.6, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

# Load the dataset
dataset = load_dataset("tech-tao/my-reasoning-traces-10k", "default", split = "train")

# Peek at the data
print(dataset[0]["prompt"])
print(dataset[0]["response"])

extract_hash_answer(dataset[0]["response"])


reasoning_start = "<think>"
reasoning_end   = "</think>"
answer_start = "<answer>"
answer_end = "</answer>"

system_prompt = \
f"""You are a thoughtful assistant.
You will first think systematically before answering.
Place your thought process between {reasoning_start} and {reasoning_end}.
Then, provide your answer between {answer_start} and {answer_end}"""
system_prompt

# dataset = dataset.map(lambda x: {
#     "prompt" : [
#         {"role": "system", "content": system_prompt},
#         {"role": "user",   "content": x["prompt"]},
#     ],
#     "completion": extract_hash_answer(x["response"]),
# })

dataset = dataset.map(lambda x: {
    "prompt": x["prompt"],
    "completion": extract_hash_answer(x["response"]),
})

match_format = re.compile(
    rf"^[\s]{{0,}}"\
    rf"{reasoning_start}.+?{reasoning_end}.*?"\
    rf"{answer_start}(.+?){answer_end}"\
    rf"[\s]{{0,}}$",
    flags = re.MULTILINE | re.DOTALL
)

match_format.search(
    "<think>Let me think!</think>"\
    "<answer>2</answer>",
)


def match_format_exactly(completions, **kwargs):
    """Reward if the format is seen exactly"""
    
    scores = []
    for completion in completions:
        score = 0
        # Handle both string and dict formats
        if isinstance(completion, str):
            response = completion
        elif isinstance(completion, list) and len(completion) > 0:
            response = completion[0].get("content", "")
        elif isinstance(completion, dict):
            response = completion.get("content", "")
        
        # Match if format is seen exactly!
        if match_format.search(response) is not None: score += 3.0
        scores.append(score)
    return scores

def match_format_approximately(completions, **kwargs):
    """Reward if the format is seen approximately"""
    scores = []
    for completion in completions:
        score = 0
        # Handle both string and dict formats
        if isinstance(completion, str):
            response = completion
        elif isinstance(completion, list) and len(completion) > 0:
            response = completion[0].get("content", "")
        elif isinstance(completion, dict):
            response = completion.get("content", "")
        
        # Count how many keywords are seen - we penalize if too many!
        # If we see 1, then plus some points!
        score += 0.5 if response.count(reasoning_start) == 1 else -1.0
        score += 0.5 if response.count(reasoning_end)   == 1 else -1.0
        score += 0.5 if response.count(answer_start)  == 1 else -1.0
        score += 0.5 if response.count(answer_end)    == 1 else -1.0
        scores.append(score)
    return scores

def check_answer(prompts, completions, answer, **kwargs):
    """Check the answer"""
    question = prompts[0][-1]["content"]
    
    # Handle both string and dict formats for completions
    responses = []
    for completion in completions:
        if isinstance(completion, str):
            responses.append(completion)
        elif isinstance(completion, list) and len(completion) > 0:
            responses.append(completion[0].get("content", ""))
        elif isinstance(completion, dict):
            responses.append(completion.get("content", ""))

    extracted_responses = [
        guess.group(1)
        if (guess := match_format.search(r)) is not None else None \
        for r in responses
    ]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        if guess is None:
            scores.append(0)
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
                if   ratio >= 0.9 and ratio <= 1.1: score += 1.0
                elif ratio >= 0.8 and ratio <= 1.2: score += 0.5
                else: score -= 1.5 # Penalize wrong answers
            except:
                score -= 1.5 # Penalize
        scores.append(score)
    return scores


match_numbers = re.compile(
    answer_start + r".*?([\d\.\,]{1,})",
    flags = re.MULTILINE | re.DOTALL
)
print(match_numbers.findall("<answer>  0.34  </answer>"))
print(match_numbers.findall("<answer>  123,456  </answer>"))

# Fine tune the model with GRPO
max_prompt_length = 512  # Reduced to allow space for completions

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    learning_rate = 1e-5,
    weight_decay = 0.01,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4, # Increase to 4 for smoother training
    num_generations = 16,
    max_prompt_length = max_prompt_length,
    max_completion_length = max_seq_length - max_prompt_length,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 500,
    save_steps = 250,
    max_grad_norm = 1.0,
    report_to = "wandb", # Enable Weights & Biases logging
    output_dir = "outputs",
    wandb_project = "llama3-2-3b-grpo",
    wandb_run_name = "reasoning-format-training",
)

# Reward System Overview:
# The total reward is the sum of all individual reward functions:
# 1. match_format_exactly: +3.0 if response matches <think>...</think><answer>...</answer> format exactly
# 2. match_format_approximately: +0.5 for each correct tag (<think>, </think>, <answer>, </answer>), -1.0 for missing/duplicate tags
# 3. check_answer: +3.0 for exact answer match, +1.5 for whitespace-only difference, +1.0/+0.5 for close numerical ratios, -1.5 for wrong answers
# Total possible reward per completion: 3.0 + 2.0 + 3.0 = 8.0 points
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        match_format_exactly,          # Exact format matching
        match_format_approximately,    # Approximate format matching  
        # check_answer,                  # Answer correctness
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()