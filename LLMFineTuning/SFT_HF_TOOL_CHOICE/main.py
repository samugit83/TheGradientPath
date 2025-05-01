
import random
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from trl import SFTConfig, SFTTrainer
from gen_dataset import generate_raw_examples

raw_examples = generate_raw_examples(10000)
special = "<my_tool_selection>"


dataset = Dataset.from_list([
    {
        "prompt":     f"User: {q} {special}\nAssistant:",
        "completion": f" {tool}"
    }
    for q, tool in raw_examples
])


splits = dataset.train_test_split(test_size=0.2, seed=42)
train_ds, eval_ds = splits["train"], splits["test"]

model_name = "HuggingFaceTB/SmolLM2-135M"


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded.")


tokenizer.add_special_tokens({"additional_special_tokens": [special]})
model.resize_token_embeddings(len(tokenizer))
print(f"Added special token {special}")


sft_args = SFTConfig(
    output_dir                  = "./tool_choice_sft",
    num_train_epochs            = 2,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 1,
    learning_rate               = 5e-5,
    warmup_ratio                = 0.1,
    logging_steps               = 5,
    eval_strategy               = "steps",
    eval_steps                  = 20,
    save_steps                  = 50,
    report_to                   = ["none"]
)


trainer = SFTTrainer( 
    model=model,
    args=sft_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer
) 


if __name__ == "__main__":
    print("Starting SFT training with 4-bit quantized model…")
    trainer.train()
    print("Training complete.")

    model.save_pretrained("./tool_choice_final")
    tokenizer.save_pretrained("./tool_choice_final")

    model = trainer.model

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=5,    
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,                  # ← greedy decoding
    )

    prompts = [
        f"User: Can you show me tomorrow's weather in Tokyo? {special}\nAssistant:",
        f"User: Please calculate the square root of 256. {special}\nAssistant:",
        f"User: I need to set a reminder to pay rent on the 5th. {special}\nAssistant:",
    ]

    for p in prompts:
        out = gen(p)[0]["generated_text"]
        print("→ Predicted tool:", out.strip())
