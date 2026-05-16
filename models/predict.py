"""
Inference wrapper for the fine-tuned BERT intent classifier.
Loads from local disk if available, otherwise downloads from HuggingFace Hub (Render deployment).
"""

import os
import json
import torch
import numpy as np
from transformers import BertTokenizerFast, BertForSequenceClassification

LOCAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), "bert_intent")
HF_MODEL_ID     = "prakharanand85/contact-center-bert-clinc150"
MODEL_SOURCE    = LOCAL_MODEL_DIR if os.path.isdir(LOCAL_MODEL_DIR) else HF_MODEL_ID

MAX_LEN = 64
CONFIDENCE_THRESHOLD = 0.60


class IntentClassifier:
    def __init__(self):
        print(f"Loading BERT from: {MODEL_SOURCE}")
        self.tokenizer = BertTokenizerFast.from_pretrained(MODEL_SOURCE)
        self.model = BertForSequenceClassification.from_pretrained(MODEL_SOURCE)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        id2label_path = os.path.join(os.path.dirname(__file__), "id2label.json")
        if os.path.exists(id2label_path):
            with open(id2label_path) as f:
                self.id2label = json.load(f)
        else:
            self.id2label = {str(k): v for k, v in self.model.config.id2label.items()}

        print(f"Classifier ready. Labels: {len(self.id2label)} | Device: {self.device}")

    def predict(self, text: str) -> dict:
        enc = self.tokenizer(
            text,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        top3_idx = np.argsort(probs)[::-1][:3]
        top3 = [(self.id2label[str(i)], float(probs[i])) for i in top3_idx]
        intent, confidence = top3[0]

        return {
            "intent": intent,
            "confidence": confidence,
            "low_confidence": confidence < CONFIDENCE_THRESHOLD,
            "top3": top3,
        }


_classifier = None

def get_classifier() -> IntentClassifier:
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier()
    return _classifier
