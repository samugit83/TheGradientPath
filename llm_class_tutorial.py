import math
import random
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
)
from dataset_tutorial import generate_raw_examples  # ensure correct module name

# 1. Generate raw examples and build label mapping
raw_examples = generate_raw_examples(10000)
tools = sorted({tool for _, tool in raw_examples})
tool2id = {tool: idx for idx, tool in enumerate(tools)}
id2tool = {idx: tool for tool, idx in tool2id.items()}

# 2. Prepare dataset with text and numerical labels
dataset = Dataset.from_list([
    {"text": q, "label": tool2id[tool]}
    for q, tool in raw_examples
])
splits = dataset.train_test_split(test_size=0.2, seed=42)
train_ds, eval_ds = splits['train'], splits['test']

# 3. Load tokenizer and model for classification
model_name = "HuggingFaceTB/SmolLM2-135M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Ensure pad token exists
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(tools),
    id2label=id2tool,
    label2id=tool2id,
    device_map="auto"
)
# Resize embeddings if new pad token was added
model.resize_token_embeddings(len(tokenizer))

# 4. Tokenization function
def preprocess(batch):
    enc = tokenizer(
        batch['text'],
        padding='max_length',
        truncation=True,
        max_length=128,
    )
    enc['labels'] = batch['label']
    return enc

train_enc = train_ds.map(preprocess, batched=True, remove_columns=['text', 'label'])
eval_enc = eval_ds.map(preprocess, batched=True, remove_columns=['text', 'label'])

# 5. Setup training arguments and Trainer
training_args = TrainingArguments(
    output_dir='./tool_choice_classification',
    overwrite_output_dir=True,
    evaluation_strategy='steps',
    eval_steps=50,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=5e-5,
    logging_steps=20,
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    greater_is_better=True,
    report_to=[]  # disable logging integrations
)

# Simple accuracy metric
def compute_metrics(p):
    import numpy as np
    preds = np.argmax(p.predictions, axis=1)
    return {'accuracy': (preds == p.label_ids).astype(float).mean()}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_enc,
    eval_dataset=eval_enc,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

# 6. Training and saving
if __name__ == '__main__':
    print("Starting classification training…")
    trainer.train()
    print("Training complete.")
    trainer.save_model("./tool_choice_classification")
    tokenizer.save_pretrained("./tool_choice_classification")

    # 7. Inference pipeline
    cls_pipe = pipeline(
        "text-classification",
        model=trainer.model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        return_all_scores=False,
    )

    prompts = [
        "Can you show me tomorrow's weather in Tokyo?",
        "Please calculate the square root of 256.",
        "I need to set a reminder to pay rent on the 5th.",
    ]

    for q in prompts:
        result = cls_pipe(q)[0]
        label_id = int(result['label'])
        score = result['score']
        print(f"→ Predicted tool: {id2tool[label_id]} (score: {score:.4f})")
