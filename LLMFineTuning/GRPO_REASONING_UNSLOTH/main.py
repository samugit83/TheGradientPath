#!/usr/bin/env python

import re
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from difflib import SequenceMatcher
from vllm import SamplingParams



max_seq_length = 1024               # Total tokens (prompt + completion)
lora_rank = 32                      # LoRA low-rank adaptation dimension
max_prompt_length = 256             # Max tokens for the prompt in GRPO

SYSTEM_PROMPT = """
Respond **only** in the exact format below, with no extra text, no deviations, and preserving these tags (including newlines):

<reasoning>
.....
</reasoning>
<answer>
.....
</answer>

Note:
- The answer must be a number, and the units must be included if the question asks for them.

"""

# ==========================================
# 1. Load and Prepare Model for LoRA Fine-Tuning
# ==========================================
# Load the pre-trained Gemma 3 1B instruct model optimized by Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-3-1b-it",  # Choose the base model
    max_seq_length=max_seq_length,       # Support long reasoning traces
    load_in_4bit=True,                   # Quantize model to 4-bit to save GPU memory
    fast_inference=True,                 # Enable vLLM acceleration for generation
    max_lora_rank=lora_rank,             # Set maximum LoRA rank
    gpu_memory_utilization=0.6,          # Cap GPU memory usage to avoid OOM
)

# Apply LoRA (Low-Rank Adaptation) on selected projection layers
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[                   # Layers to fine-tune via LoRA
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",  # Save memory for long contexts
    random_state=3407,                      # Ensure reproducibility
)

# ==========================================
# 2. Data Preparation: GSM8K with XML Chain-of-Thought
# ==========================================
def extract_xml_answer(text: str) -> str:
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL)
    return m.group(1).strip() if m else ""

def extract_hash_answer(text: str) -> str | None:
    return text.split('####')[1].strip() if '####' in text else None


def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]
    return data.map(lambda x: {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": x["question"]},
        ],
        "answer": extract_hash_answer(x["answer"]),
    })

# Prepare the training dataset
dataset = get_gsm8k_questions()

# ==========================================
# 3. Reward Functions with Debug Prints
# These guide the GRPO trainer and log detailed per-step info
# ==========================================
def correctness_reward_func(prompts, completions, answer, **kwargs):
    """
    Reward = 2.0 * (0.8 * content_score + tag_bonus), where:
      - content_score ∈ [0,1]: relative-error for numeric or string-similarity for text
      - tag_bonus ∈ {0, 0.2}: +0.2 if exactly one of each <reasoning>…</reasoning> and <answer>…</answer>
    """
    def string_similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()
    
    def extract_between(text: str, start: str, end: str) -> str:
        m = re.search(fr"{re.escape(start)}\s*(.*?)\s*{re.escape(end)}", text, flags=re.DOTALL)
        return m.group(1).strip() if m else ""
    
    rewards = []
    for gen_list, gold in zip(completions, answer):
        full      = gen_list[0]["content"]
        reasoning = extract_between(full, "<reasoning>", "</reasoning>")
        pred      = extract_between(full, "<answer>",    "</answer>")
        
        try:
            num_pattern = r"[-+]?\d*\.?\d+"
            m_pred = re.search(num_pattern, pred)
            m_gold = re.search(num_pattern, gold)

            if m_pred and m_gold:
                p_f = float(m_pred.group())
                g_f = float(m_gold.group())
                rel_err = abs(p_f - g_f) / (abs(g_f) + 1e-8)
                numeric_score = max(0.0, 1 - rel_err)

                unit_pred = pred[m_pred.end():].strip()
                unit_gold = gold[m_gold.end():].strip()

                if unit_pred or unit_gold:
                    unit_score = string_similarity(unit_pred, unit_gold)
                    content_score = 0.5 * numeric_score + 0.5 * unit_score
                else:
                    content_score = numeric_score
            else:
                content_score = string_similarity(pred, gold)
        except Exception:
            content_score = string_similarity(pred, gold)
        
        reward = 2.0 * content_score
        
        question = prompts[0][-1]["content"]
        print(f"=== GRPO Step ===\n"
              f"Question: {question}\n\n"
              f"--- Reasoning ---\n{reasoning}\n\n"
              f"--- Predicted Answer ---\n{pred}\n\n"
              f"--- Gold Answer ---\n{gold}\n\n"
              f"Content_score={content_score:.3f}, "
              f"Reward={reward:.3f}\n")
        
        rewards.append(reward)
    return rewards


def int_reward_func(completions, **kwargs):
    """
    Reward function that checks if the answer is a pure integer.
    """
    # Reward numeric-only answers with 0.5
    responses = [c[0]["content"] for c in completions]
    extracted = [extract_xml_answer(r) for r in responses]
    rewards = [0.5 if r.isdigit() else 0.0 for r in extracted]
    print(f"[int_reward] Ans: {extracted} | R: {rewards}\n")
    return rewards


def strict_format_reward_func(completions, **kwargs):
    # Reward strict XML formatting
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [c[0]["content"] for c in completions]
    matches = [bool(re.match(pattern, r)) for r in responses]
    rewards = [0.5 if m else 0.0 for m in matches]
    print(f"[strict_format] matches={matches} | r={rewards}\n")
    return rewards


def soft_format_reward_func(completions, **kwargs):
    # Reward looser XML formatting
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [c[0]["content"] for c in completions]
    matches = [bool(re.match(pattern, r)) for r in responses]
    rewards = [0.5 if m else 0.0 for m in matches]
    print(f"[soft_format] matches={matches} | r={rewards}\n")
    return rewards


def count_xml(text: str) -> float:
    # Penalize extra content, reward correct tag counts
    score = 0.0
    if text.count("<reasoning>\n") == 1: score += 0.125
    if text.count("\n</reasoning>\n") == 1: score += 0.125
    if text.count("\n<answer>\n") == 1:
        score += 0.125
        score -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        score += 0.125
        score -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return score


def xmlcount_reward_func(completions, **kwargs):
    # Reward based on XML tag usage
    responses = [c[0]["content"] for c in completions]
    rewards = [count_xml(r) for r in responses]
    print(f"[xmlcount] R: {rewards}\n")
    return rewards

# ==========================================
# 4. Configure and Initialize GRPO Trainer
# ==========================================
training_args = GRPOConfig(
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,                  # Log every step
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_generations=6,                # Generations per prompt
    max_prompt_length=max_prompt_length,
    max_completion_length=max_seq_length - max_prompt_length,
    max_steps=250,                    # Total training steps
    save_steps=250,                   # Save checkpoint at end
    max_grad_norm=0.1,
    report_to="none",
    output_dir="outputs",            # Directory for outputs
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=dataset,
)

# ==========================================
# 5. Start Training
# Detailed prints show each generation, reasoning, and reward
# ==========================================
trainer.train()

# ==========================================
# 6. Save LoRA Weights for Later Use
# ==========================================
model.save_lora("grpo_saved_lora")  # Save only the adapted weights

# ==========================================
# 7. Test the Fine-Tuned Model
# ==========================================
chat_input = tokenizer.apply_chat_template(
    [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": "Calculate pi."},
    ],
    tokenize=False,
    add_generation_prompt=True,
)
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=1024,
)
output = (
    model.fast_generate(
        chat_input,
        sampling_params=sampling_params,
        lora_request=model.load_lora("grpo_saved_lora"),
    )[0]
    .outputs[0]
    .text
)
print("\n=== Model Output ===\n", output)

# ==========================================
# 8. Save the model
# ==========================================
model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")
