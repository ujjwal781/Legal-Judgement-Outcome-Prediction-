
# IMPORTS

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np


# LOAD DATASET

print("Loading dataset...")

dataset = load_dataset(
    "csv",
    data_files={
        "train": "train_multi.csv",
        "validation": "dev.csv",
        "test": "test.csv"
    }
)

print(dataset)
print("\nSample row:\n", dataset["train"][0])


# DEFINE COLUMNS

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"


# CLEAN TEXT

def clean_text(example):
    example["text"] = (
        str(example["text"]) + " " + str(example["Reasoning"])
    ).lower()
    return example


# TOKENIZATION

print("\nTokenizing...")

tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

def tokenize(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

tokenized_dataset = dataset.map(tokenize, batched=True)


# FORMAT DATA

tokenized_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)


# LOAD MODEL

print("\nLoading model...")

model = AutoModelForSequenceClassification.from_pretrained(
    "nlpaueb/legal-bert-base-uncased",
    num_labels=2
)


# TRAINING SETUP

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",   
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    save_strategy="epoch"
)


# METRICS

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}


# TRAINER

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics
)


#TRAIN MODEL

print("\nTraining started...")
trainer.train()


# EVALUATE

print("\nEvaluating...")
results = trainer.evaluate()
print("\nResults:", results)


# PREDICT FUNCTION

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    
    logits = outputs.logits
    pred = logits.argmax().item()

    if pred == 0:
        return "Outcome: Class 0"
    else:
        return "Outcome: Class 1"


# TEST PREDICTION

print("\nTest Prediction:")
print(predict("illegal transaction during moratorium"))