
# https://huggingface.co/learn/llm-course/chapter11/4?fw=pt

# Import necessary libraries
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, PeftModel # Import LoRA config

# pip install torch transformers datasets trl accelerate bitsandbytes peft

# --- Configuration ---
# Specify the pre-trained model name from Hugging Face Hub
base_model_name = "HuggingFaceH4/zephyr-7b-beta" # Base model for LoRA
# Specify the dataset name from Hugging Face Hub
dataset_name = "HuggingFaceTB/smoltalk"
# Specify the subset/split of the dataset (if applicable)
dataset_subset = "all"
# Specify the output directory to save the LoRA adapter weights and logs
output_dir = "./lora_sft_output"
# Specify the device to use for training ("cuda" for GPU, "cpu" for CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load Dataset ---
print(f"Loading dataset '{dataset_name}' ({dataset_subset})...")
# Load the dataset. The structure might vary based on the dataset.
# For datasets with 'messages' field, SFTTrainer automatically applies chat templates.
# For other structures, you might need a formatting_func (see below).
dataset = load_dataset(dataset_name, dataset_subset)
print("Dataset loaded.")
# Assuming the dataset has 'train' and 'test' splits, like 'smoltalk'
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# --- Load Base Model and Tokenizer ---
print(f"Loading base model '{base_model_name}'...")
# --- Quantization Config (QLoRA) ---
# Load the base model with 4-bit quantization to reduce memory usage
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # Use NF4 (Normal Float 4) quantization type
    bnb_4bit_compute_dtype=torch.bfloat16 # Compute type for matrix multiplications
)

# Load the pre-trained base model for causal language modeling with quantization
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quantization_config,
    device_map="auto" # Automatically distribute model layers across devices (GPU/CPU)
)
print("Base model loaded.")
print(f"Loading tokenizer for '{base_model_name}'...")
# Load the tokenizer associated with the pre-trained model
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
# Set pad token if it's not already set (common requirement)
if tokenizer.pad_token is None:
    # Important: Setting pad_token to eos_token for open-end generation
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id # Ensure model config is updated
print("Tokenizer loaded.")


# --- Configure LoRA (PEFT) ---
print("Configuring LoRA...")
lora_config = LoraConfig(
    r=8,                    # Rank dimension - How many parameters to train in the low-rank matrices. Typically 4-32.
    lora_alpha=16,          # LoRA scaling factor (often 2*r). Controls the magnitude of the adaptation.
    lora_dropout=0.05,      # Dropout probability for LoRA layers to prevent overfitting.
    bias="none",            # Specifies if bias parameters should be trained ("none", "all", or "lora_only"). "none" is common.
    target_modules="all-linear", # Apply LoRA to all linear layers. Can specify layers like ["q_proj", "v_proj"].
    task_type="CAUSAL_LM",  # Task type for the model architecture.
)
print("LoRA configured.")

# --- Configure Training (SFTConfig) ---
# SFTConfig allows configuring the training process.
# Refer to the documentation for detailed explanations of each parameter.
training_args = SFTConfig(
    output_dir=output_dir, # Directory to save adapter checkpoints and logs

    # --- Training Duration ---
    # num_train_epochs=3,        # Total number of training epochs (alternative to max_steps)
    max_steps=1000,             # Max number of training steps (overrides epochs if set)

    # --- Batch Size & Gradient Accumulation ---
    per_device_train_batch_size=4, # Batch size per GPU/CPU during training
    per_device_eval_batch_size=4,  # Batch size per GPU/CPU during evaluation
    gradient_accumulation_steps=2, # Accumulate gradients over N steps for larger effective batch size (4*2=8)

    # --- Learning Rate ---
    learning_rate=5e-5,         # Controls the step size for weight updates (can be lower for LoRA, e.g., 2e-4)
    # warmup_ratio=0.1,         # Portion of training for learning rate warmup (optional)
    # lr_scheduler_type="cosine", # Learning rate scheduler type (optional)

    # --- Monitoring & Saving ---
    logging_steps=10,           # Log training metrics every N steps
    eval_strategy="steps",      # Evaluate model during training ("steps" or "epoch")
    eval_steps=50,              # Evaluate every N steps (if eval_strategy="steps")
    save_strategy="steps",      # Save checkpoints strategy ("steps" or "epoch")
    save_steps=100,             # Save an adapter checkpoint every N steps

    # --- Dataset Packing (Optional) ---
    # packing=True,             # Pack multiple short examples into one sequence for efficiency (requires formatting_func if dataset structure differs)
    # eval_packing=False,       # Usually disable packing for evaluation dataset

    # --- Other Parameters ---
    # max_seq_length=512,       # Max sequence length for tokenization (optional, SFTTrainer handles this)
    # Add other SFTConfig parameters as needed
)

# --- Optional: Custom Formatting Function for Packing ---
# (Same as in llm_sft.py, include if using packing=True with specific dataset formats)
# def formatting_func(examples):
#     # ... (implementation depends on dataset structure)
#     return examples["text"] # Or formatted strings


# --- Initialize Trainer ---
print("Initializing SFTTrainer with LoRA...")
trainer = SFTTrainer(
    model=model,                # The base model (will be adapted with LoRA)
    processing_class=tokenizer, # The tokenizer for processing data
    args=training_args,         # The training configuration (SFTConfig)
    peft_config=lora_config,    # Pass the LoRA configuration here!
    train_dataset=train_dataset,# The training dataset
    eval_dataset=eval_dataset,  # The evaluation dataset
    # --- Optional arguments ---
    # formatting_func=formatting_func, # Provide if packing=True and custom format needed
    # max_seq_length=512,       # Max sequence length for tokenization (optional)
    # data_collator=None,       # Custom data collator (optional)
)
print("Trainer initialized.")

# --- Start Training ---
print("Starting LoRA fine-tuning...")
trainer.train()
print("Training finished.")

# --- Save the LoRA adapter ---
# This saves only the adapter weights and config, not the full model.
print(f"Saving LoRA adapter model to {output_dir}...")
trainer.save_model(output_dir)
# Alternatively, you can save the PeftModel directly:
# model.save_pretrained(output_dir) # Saves adapter config & weights
print("LoRA adapter saved.")




# --- (Optional) Merge LoRA adapter with base model and save ---
# This section demonstrates how to merge the trained LoRA weights back into the
# base model and save the complete, merged model. Requires sufficient RAM/GPU memory.
# --- (Optional) Merge LoRA adapter with base model and save ---
# This section demonstrates how to merge the trained LoRA weights back into the
# base model and save the complete, merged model. Requires sufficient RAM/GPU memory.
#
# Benefits of merging:
# 1. Deployment Simplicity: Creates a single, self-contained model instead of 
#    needing to load both base model and adapter separately.
# 2. Inference Performance: May offer slight latency improvements as adapter 
#    calculations are baked directly into model weights.
# 3. Compatibility: Some deployment environments expect standard model formats
#    rather than base+adapter configurations.
#
# Trade-offs:
# 1. Memory Requirements: Merging process requires enough memory to hold both
#    the full base model and adapter.
# 2. Storage Size: Merged model is large (similar to original base model size),
#    losing the storage benefits of separate adapters.
# 3. Flexibility: You lose the ability to easily swap different LoRA adapters
#    on the same base model.

# print("--- Merging LoRA adapter with base model ---")
# # Ensure the base model is loaded (if not continuing directly from training)
# print(f"Loading base model '{base_model_name}' for merging...")
# base_model_reload = AutoModelForCausalLM.from_pretrained(
#     base_model_name,
#     quantization_config=quantization_config, # Keep quantization if used during training
#     # torch_dtype=torch.float16, # Use if not using quantization or want specific dtype
#     device_map="auto",
# )
# print("Base model reloaded.")

# print(f"Loading PEFT model (base + adapter) from {output_dir}...")
# # Load the PeftModel, which combines the base model and the adapter
# peft_model = PeftModel.from_pretrained(
#     base_model_reload,
#     output_dir # Path to the saved LoRA adapter directory
# )
# print("PEFT model loaded.")

# print("Merging adapter weights into the base model...")
# merged_model = peft_model.merge_and_unload()
# print("Merging complete.")

# merged_model_dir = f"{output_dir}_merged"
# print(f"Saving merged model and tokenizer to {merged_model_dir}...")
# merged_model.save_pretrained(merged_model_dir)
# tokenizer.save_pretrained(merged_model_dir) # Important: save the tokenizer too!
# print("Merged model and tokenizer saved.")

print("Script finished successfully.") 