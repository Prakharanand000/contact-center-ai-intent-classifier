"""
Hybrid Information Retrieval fallback.
Used when BERT confidence < threshold.

Two-stage retrieval:
  1. BM25 (lexical) — rank_bm25
  2. Dense (semantic) — sentence-transformers all-MiniLM-L6-v2

Final score = 0.4 * bm25_norm + 0.6 * dense_cosine

Reusable algorithm: build_index() takes any list of (text, label) pairs,
making this module portable across any intent dataset.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
INDEX_DIR = os.path.dirname(__file__)
DENSE_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
BM25_WEIGHT = 0.4
DENSE_WEIGHT = 0.6


class HybridRetriever:
    """
    Reusable algorithm: initialize with any corpus of (text, label) pairs.
    Exposes retrieve(query, top_k) returning ranked candidates with scores.
    """

    def __init__(self, corpus_texts: list, corpus_labels: list):
        self.corpus_texts = corpus_texts
        self.corpus_labels = corpus_labels

        # BM25 index
        tokenized = [t.lower().split() for t in corpus_texts]
        self.bm25 = BM25Okapi(tokenized)

        # Dense index
        print("Building dense index (sentence-transformers)...")
        self.encoder = SentenceTransformer(DENSE_MODEL)
        self.dense_embeddings = self.encoder.encode(
            corpus_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
        )
        print(f"Index ready. Corpus size: {len(corpus_texts)}")

    def retrieve(self, query: str, top_k: int = TOP_K) -> list:
        """
        Returns top_k results, each a dict:
            text, label, bm25_score, dense_score, hybrid_score
        """
        # BM25
        bm25_scores = np.array(self.bm25.get_scores(query.lower().split()))
        bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-9)

        # Dense
        q_emb = self.encoder.encode([query], normalize_embeddings=True)
        dense_scores = cosine_similarity(q_emb, self.dense_embeddings)[0]

        # Hybrid
        hybrid = BM25_WEIGHT * bm25_norm + DENSE_WEIGHT * dense_scores
        top_idx = np.argsort(hybrid)[::-1][:top_k]

        return [
            {
                "text": self.corpus_texts[i],
                "label": self.corpus_labels[i],
                "bm25_score": float(bm25_norm[i]),
                "dense_score": float(dense_scores[i]),
                "hybrid_score": float(hybrid[i]),
            }
            for i in top_idx
        ]

    def predict_from_retrieval(self, query: str, top_k: int = TOP_K) -> dict:
        """
        Majority vote over top_k retrieved labels.
        Returns predicted label + confidence (vote share).
        """
        results = self.retrieve(query, top_k)
        from collections import Counter
        vote = Counter(r["label"] for r in results)
        top_label, top_count = vote.most_common(1)[0]
        return {
            "intent": top_label,
            "confidence": top_count / top_k,
            "retrieved": results,
        }

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"texts": self.corpus_texts, "labels": self.corpus_labels}, f)
        np.save(path.replace(".pkl", "_embeddings.npy"), self.dense_embeddings)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls.__new__(cls)
        obj.corpus_texts = data["texts"]
        obj.corpus_labels = data["labels"]
        tokenized = [t.lower().split() for t in obj.corpus_texts]
        obj.bm25 = BM25Okapi(tokenized)
        obj.encoder = SentenceTransformer(DENSE_MODEL)
        emb_path = path.replace(".pkl", "_embeddings.npy")
        obj.dense_embeddings = np.load(emb_path)
        print(f"Loaded retrieval index: {len(obj.corpus_texts)} docs")
        return obj


def build_and_save_index():
    """Build retrieval index from training data and save to disk."""
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    label_map_df = pd.read_csv(os.path.join(DATA_DIR, "label_map.csv"))
    id2label = {row["id"]: row["intent"] for _, row in label_map_df.iterrows()}

    texts = train_df["text"].tolist()
    labels = [id2label[i] for i in train_df["intent"].tolist()]

    retriever = HybridRetriever(texts, labels)
    save_path = os.path.join(INDEX_DIR, "retrieval_index.pkl")
    retriever.save(save_path)
    print(f"Index saved to {save_path}")
    return retriever


# Singleton
_retriever = None

def get_retriever() -> HybridRetriever:
    global _retriever
    index_path = os.path.join(INDEX_DIR, "retrieval_index.pkl")
    if _retriever is None:
        if os.path.exists(index_path):
            _retriever = HybridRetriever.load(index_path)
        else:
            print("No saved index found. Building from training data...")
            _retriever = build_and_save_index()
    return _retriever
