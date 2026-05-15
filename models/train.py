"""
Fine-tune BERT-base on CLINC150 for intent classification.
Target: 91%+ weighted F1 on test set.

Usage:
    python models/train.py
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
import torch
from torch.utils.data import Dataset

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "bert_intent")
PRETRAINED = "bert-base-uncased"
MAX_LEN = 64
BATCH_SIZE = 32
EPOCHS = 5
LR = 2e-5
SEED = 42

# ── Dataset ───────────────────────────────────────────────────────────────────
class IntentDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts = df["text"].tolist()
        self.labels = df["intent"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = (preds == labels).mean()
    return {"accuracy": acc, "f1_weighted": f1}


def train():
    # Load data
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df   = pd.read_csv(os.path.join(DATA_DIR, "validation.csv"))
    test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    num_labels = train_df["intent"].nunique()
    print(f"Labels: {num_labels} | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Save id2label / label2id
    label_map_df = pd.read_csv(os.path.join(DATA_DIR, "label_map.csv"))
    id2label  = {row["id"]: row["intent"] for _, row in label_map_df.iterrows()}
    label2id  = {v: k for k, v in id2label.items()}
    with open(os.path.join(os.path.dirname(__file__), "id2label.json"), "w") as f:
        json.dump(id2label, f)

    # Tokenizer + model
    tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED)
    model = BertForSequenceClassification.from_pretrained(
        PRETRAINED,
        num_labels=num_labels,
        id2label={int(k): v for k, v in id2label.items()},
        label2id=label2id,
        ignore_mismatched_sizes=True,   # silences the cls head mismatch warnings
    )

    train_dataset = IntentDataset(train_df, tokenizer, MAX_LEN)
    val_dataset   = IntentDataset(val_df,   tokenizer, MAX_LEN)
    test_dataset  = IntentDataset(test_df,  tokenizer, MAX_LEN)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # TrainingArguments — use eval_strategy (new name) with fallback for older versions
    common = dict(
        output_dir=MODEL_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=64,
        learning_rate=LR,
        warmup_ratio=0.1,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        seed=SEED,
        logging_dir=os.path.join(MODEL_DIR, "logs"),
        logging_steps=50,
        report_to="none",
    )
    try:
        # Transformers >= 4.41 renamed evaluation_strategy -> eval_strategy
        args = TrainingArguments(eval_strategy="epoch", **common)
    except TypeError:
        args = TrainingArguments(evaluation_strategy="epoch", **common)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Starting training...")
    trainer.train()

    # Save model + tokenizer
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"\nModel saved to {MODEL_DIR}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    preds_output = trainer.predict(test_dataset)
    preds  = np.argmax(preds_output.predictions, axis=-1)
    labels = test_df["intent"].tolist()
    intent_names = [id2label[str(i)] for i in range(num_labels)]

    report = classification_report(labels, preds, target_names=intent_names, digits=4)
    print(report)

    metrics = {
        "accuracy":    float((np.array(preds) == np.array(labels)).mean()),
        "f1_weighted": float(f1_score(labels, preds, average="weighted")),
    }
    with open(os.path.join(MODEL_DIR, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nTest Accuracy : {metrics['accuracy']:.4f}")
    print(f"Weighted F1   : {metrics['f1_weighted']:.4f}")

    cm = confusion_matrix(labels, preds)
    np.save(os.path.join(MODEL_DIR, "confusion_matrix.npy"), cm)
    print("Confusion matrix saved.")


if __name__ == "__main__":
    train()
