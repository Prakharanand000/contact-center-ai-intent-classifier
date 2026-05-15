"""
Inference wrapper for the fine-tuned BERT intent classifier.
Loads model once, exposes predict() for use by the API and retrieval layer.
"""

import os
import json
import torch
import numpy as np
from transformers import BertTokenizerFast, BertForSequenceClassification

MODEL_DIR = os.path.join(os.path.dirname(__file__), "bert_intent")
MAX_LEN = 64
CONFIDENCE_THRESHOLD = 0.60  # below this, hand off to retrieval fallback


class IntentClassifier:
    def __init__(self):
        print(f"Loading BERT intent classifier from {MODEL_DIR}...")
        self.tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
        self.model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        id2label_path = os.path.join(os.path.dirname(__file__), "id2label.json")
        with open(id2label_path) as f:
            self.id2label = json.load(f)
        print(f"Classifier ready. Labels: {len(self.id2label)} | Device: {self.device}")

    def predict(self, text: str) -> dict:
        """
        Returns:
            intent (str): predicted intent label
            confidence (float): softmax probability of top prediction
            low_confidence (bool): True if below CONFIDENCE_THRESHOLD
            top3 (list): top 3 (intent, score) pairs
        """
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


# Singleton — load once at import time for API use
_classifier = None

def get_classifier() -> IntentClassifier:
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier()
    return _classifier
