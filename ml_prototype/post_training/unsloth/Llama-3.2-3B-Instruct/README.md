# GRPO Fine-tuning for Llama-3.2-3B-Instruct

This directory contains a refactored implementation of GRPO (Group Relative Policy Optimization) fine-tuning for the Llama-3.2-3B-Instruct model on reasoning traces dataset.

## Project Structure

```
Llama-3.2-3B-Instruct/
├── grpo_finetune_refactored.py  # Single file with all components
├── test_single_file.py          # Test script for single file version
├── README.md                    # This file
└── grpo_finetune_llama3_2_3b.py # Original notebook code
```

## 🎯 **Single File Architecture**

The refactored code is now organized into a single comprehensive file (`grpo_finetune_refactored.py`) with clear sections:

### **1. Configuration Section**
- **ModelConfig**: Model parameters and settings
- **LoRAConfig**: LoRA fine-tuning configuration
- **DatasetConfig**: Dataset loading parameters
- **PromptConfig**: Prompt formatting and system messages
- **TrainingConfig**: Training hyperparameters
- **Config**: Main configuration class that combines all configs

### **2. Data Processing Section**
- **DataProcessor**: Handles dataset loading, formatting, and validation
- **Regex Patterns**: Centralized pattern matching for answer extraction
- **Testing Utilities**: Built-in functions to test data processing

### **3. Model Setup Section**
- **ModelSetup**: Model loading and LoRA configuration
- **Environment Management**: GPU setup and memory configuration
- **Model Information**: Detailed statistics and memory usage

### **4. Reward Functions Section**
- **RewardFunctions**: Collection of all GRPO reward functions
- **Format Matching**: Rewards for proper response formatting
- **Answer Validation**: Rewards for correct numerical answers

### **5. Training Section**
- **GRPOTrainerSetup**: Training orchestration and GRPO configuration
- **Training Monitoring**: Progress tracking and configuration display
- **Error Handling**: Robust training execution

### **6. Main Execution Section**
- **main()**: Complete training pipeline orchestration
- **Error Handling**: Comprehensive error management

## 🚀 **Quick Start**

### **Run Training**
```bash
python grpo_finetune_refactored.py
```

### **Test Components**
```bash
python test_single_file.py
```

## 🔧 **Key Features**

### **1. Modular Design in Single File**
- **Clear Sections**: Each component is in its own section with clear headers
- **Self-Contained**: Everything needed is in one file
- **Easy Navigation**: Well-organized with comments and docstrings

### **2. Configuration Management**
- **Centralized Settings**: All hyperparameters in one place
- **Type Safety**: Uses dataclasses with type annotations
- **Easy Customization**: Modify settings without touching core logic

### **3. Robust Data Processing**
- **Flexible Dataset Loading**: Handles datasets with or without configs
- **Format Validation**: Built-in testing for regex patterns
- **Error Handling**: Graceful handling of dataset loading issues

### **4. Memory Efficient Training**
- **LoRA Integration**: Parameter-efficient fine-tuning
- **Memory Monitoring**: Real-time GPU memory tracking
- **Configurable Settings**: Easy to adjust for different hardware

### **5. Comprehensive Reward System**
- **Multi-Objective**: Combines format matching and answer accuracy
- **Flexible Scoring**: Configurable reward weights and thresholds
- **Extensible**: Easy to add new reward functions

## 📋 **Configuration Options**

### **Model Configuration**
```python
config = Config()
config.model.max_seq_length = 4096  # Increase sequence length
config.model.load_in_4bit = True    # Enable 4-bit quantization
config.model.gpu_memory_utilization = 0.8  # Adjust memory usage
```

### **LoRA Configuration**
```python
config.lora.r = 128                 # Higher rank for better performance
config.lora.target_modules = ["q_proj", "v_proj"]  # Custom target modules
config.lora.use_gradient_checkpointing = "unsloth"  # Memory optimization
```

### **Training Configuration**
```python
config.training.learning_rate = 1e-5        # Adjust learning rate
config.training.max_steps = 1000            # More training steps
config.training.per_device_train_batch_size = 2  # Larger batch size
config.training.num_generations = 8         # More generations for GRPO
```

### **Dataset Configuration**
```python
config.dataset.dataset_name = "your-dataset"  # Custom dataset
config.dataset.dataset_split = None          # No config (default)
config.dataset.dataset_subset = "train"      # Training split
```

## 🛠 **Customization Examples**

### **Custom Reward Function**
```python
# Add to RewardFunctions class
def custom_reward(self, completions, **kwargs):
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        # Your custom logic here
        score = 1.0 if "custom_pattern" in response else 0.0
        scores.append(score)
    return scores

# Add to get_all_reward_functions method
return [
    self.match_format_exactly,
    self.match_format_approximately,
    self.check_answer,
    self.check_numbers,
    self.custom_reward,  # Add your custom function
]
```

### **Custom Data Processing**
```python
# Modify DataProcessor class
def custom_format_dataset(self, dataset):
    def format_example(x):
        return {
            "prompt": [
                {"role": "system", "content": "Your custom system prompt"},
                {"role": "user", "content": x["custom_field"]},
            ],
            "answer": self.custom_answer_extraction(x["response"]),
        }
    return dataset.map(format_example)
```

## 🔍 **Troubleshooting**

### **Memory Issues**
- Reduce `max_seq_length` or `per_device_train_batch_size`
- Lower LoRA rank (`config.lora.r`)
- Enable gradient checkpointing
- Use 4-bit quantization (`config.model.load_in_4bit = True`)

### **Dataset Issues**
- Check dataset name and format
- Verify dataset has required fields (`prompt`, `response`)
- Test data processing with `test_single_file.py`

### **Training Issues**
- Monitor training logs for errors
- Check reward functions are working correctly
- Verify model and tokenizer compatibility

## 📊 **Performance Tips**

1. **GPU Memory**: Monitor with `nvidia-smi` and adjust batch sizes
2. **LoRA Rank**: Higher rank = better performance but more memory
3. **Sequence Length**: Longer sequences need more memory
4. **Gradient Accumulation**: Use for larger effective batch sizes
5. **Mixed Precision**: Enable for faster training (handled by Unsloth)

## 🎯 **Expected Output**

The training will show:
- Model loading and LoRA setup
- Dataset loading and formatting
- Format matching tests
- Training configuration summary
- Training progress with reward scores
- Final model saved to `outputs/` directory

## 📝 **Dependencies**

```bash
pip install torch transformers diffusers datasets trl unsloth
```

The single file approach makes it much easier to:
- **Deploy**: Just copy one file
- **Modify**: All code in one place
- **Debug**: Clear section organization
- **Share**: Single file to distribute
- **Version Control**: Track changes in one file 