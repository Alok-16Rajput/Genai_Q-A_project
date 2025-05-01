import json
from transformers import (
    AutoTokenizer, AutoModelForQuestionAnswering,
    Trainer, TrainingArguments
)
from datasets import Dataset
import torch

# Load and encode dataset
with open("data/qa_data.json", "r") as f:
    data = json.load(f)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Encoding function
def encode(example):
    inputs = tokenizer(
        example["context"],
        example["question"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_offsets_mapping=True
    )

    answer = example["answer"]
    start_char = example["context"].find(answer)
    end_char = start_char + len(answer)

    offset_mapping = inputs["offset_mapping"]
    start_position = end_position = 0

    for idx, (start, end) in enumerate(offset_mapping):
        if start <= start_char < end:
            start_position = idx
        if start < end_char <= end:
            end_position = idx
            break

    inputs.pop("offset_mapping")  # Remove to avoid Trainer error
    inputs["start_positions"] = start_position
    inputs["end_positions"] = end_position
    return inputs

# Convert to Dataset and tokenize
dataset = Dataset.from_list(data).map(encode)

# Load model
model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")

# Training arguments
training_args = TrainingArguments(
    output_dir="model/fine_tuned",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    logging_dir="./logs",
    save_steps=10,
    logging_steps=5,
    no_cuda=not torch.cuda.is_available()
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Train
trainer.train()

# Save model and tokenizer
model.save_pretrained("model/fine_tuned")
tokenizer.save_pretrained("model/fine_tuned")
