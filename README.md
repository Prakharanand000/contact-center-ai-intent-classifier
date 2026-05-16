# Contact Center AI Intent Classifier

Live: https://contact-center-ai-intent-classifier.onrender.com

Production-grade NLP intent classification pipeline for Contact Center AI use cases.  
Built by [Prakhar Anand](https://github.com/Prakharanand000) | [Portfolio](https://prakharanand000.github.io/)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688)](https://fastapi.tiangolo.com/)
[![BERT](https://img.shields.io/badge/BERT-base--uncased-blue)](https://huggingface.co/bert-base-uncased)
[![CLINC150](https://img.shields.io/badge/Dataset-CLINC150-orange)](https://huggingface.co/datasets/clinc_oos)

---

## What it does

Takes a raw customer support utterance and returns:
- **Predicted intent** (150 classes + out-of-scope)
- **Confidence score** with low-confidence flagging
- **Hybrid retrieval fallback** when BERT is uncertain
- **Dialog state** across a 3-turn session window
- **Slot extraction** for key entities per intent domain

**Live demo:** [https://contact-center-ai.onrender.com/docs](https://contact-center-ai.onrender.com/docs)

---

## Architecture

```
Customer Utterance
       │
       ▼
┌─────────────────────┐
│  BERT Fine-tuned    │  ← bert-base-uncased, fine-tuned on CLINC150
│  Intent Classifier  │    150 intents, 91.4% weighted F1
└──────────┬──────────┘
           │ confidence ≥ 0.60
           ├──────────────────────► Return BERT prediction
           │
           │ confidence < 0.60
           ▼
┌──────────────────────────────────┐
│  Hybrid Information Retrieval    │
│  BM25 (lexical)   40% weight     │  ← rank_bm25
│  Dense (semantic) 60% weight     │  ← sentence-transformers all-MiniLM-L6-v2
│  Top-K majority vote             │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────┐
│  Dialog State        │  ← 3-turn sliding window
│  Tracker             │    slot extraction, escalation detection
└──────────────────────┘
           │
           ▼
     FastAPI Response
```

---

## Dataset

**CLINC150** (plus split) — [huggingface.co/datasets/clinc_oos](https://huggingface.co/datasets/clinc_oos/viewer/plus)

| Split      | Samples | Notes                        |
|------------|---------|------------------------------|
| Train      | 15,100  | 100 per intent class         |
| Validation | 3,100   | ~20 per class                |
| Test       | 5,500   | ~36 per class + OOS          |
| **Total**  | 23,700  | 150 intents + out-of-scope   |

Intent domains: banking, travel, utilities, kitchen & dining, home, auto & commute, small talk, meta, work, and more.

---

## Algorithm Card

### Intent Classifier (BERT)

| Property            | Value                        |
|---------------------|------------------------------|
| Base model          | bert-base-uncased            |
| Fine-tuning dataset | CLINC150 (plus split)        |
| Max sequence length | 64 tokens                    |
| Batch size          | 32                           |
| Learning rate       | 2e-5 with warmup (10%)       |
| Epochs              | 5 (early stopping patience 2)|
| Optimizer           | AdamW, weight decay 0.01     |
| **Test Accuracy**   | **91.4%**                    |
| **Weighted F1**     | **91.4%**                    |

### Retrieval Fallback (Hybrid BM25 + Dense)

| Property            | Value                              |
|---------------------|------------------------------------|
| Lexical retrieval   | BM25Okapi (rank_bm25)              |
| Dense retrieval     | all-MiniLM-L6-v2 (384-dim)        |
| Hybrid weight       | 0.40 BM25 + 0.60 dense             |
| Top-K               | 5 candidates, majority vote        |
| Confidence threshold| 0.60 (BERT confidence gate)        |
| Index size          | 15,100 training utterances         |

### Dialog State Tracker

| Property            | Value                              |
|---------------------|------------------------------------|
| Context window      | 3 turns (sliding)                  |
| States              | greeting / in_progress / resolved / escalate |
| Resolution trigger  | Same intent, confidence ≥ 0.80, 2 consecutive turns |
| Escalation trigger  | All 3 turns low confidence OR 3 different intents |
| Slot domains        | 10 intent domains, 2–3 slots each  |

### Handle Time Reduction

Simulated on CLINC150 test set (5,500 utterances):

| Scenario             | Avg Turns to Resolution |
|----------------------|------------------------|
| Baseline (no model)  | 3.00                   |
| With BERT + Retrieval| ~2.16                  |
| **Reduction**        | **~28%**               |

---

## Project Structure

```
contact-center-ai-intent-classifier/
├── data/
│   └── download_data.py        # Download CLINC150 from HuggingFace
├── models/
│   ├── train.py                # BERT fine-tuning script
│   ├── predict.py              # Inference wrapper (singleton)
│   └── bert_intent/            # Saved model (after training, gitignored)
├── retrieval/
│   └── hybrid_retriever.py     # BM25 + dense retrieval, reusable algorithm
├── dialog/
│   └── state_tracker.py        # 3-turn dialog state + slot extraction
├── api/
│   └── main.py                 # FastAPI endpoints
├── notebooks/
│   └── evaluate.py             # Evaluation: metrics, confusion matrix, handle time
├── requirements.txt
├── Procfile                    # Render deployment
└── README.md
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset
python data/download_data.py

# 3. Fine-tune BERT (GPU recommended, ~20 min on T4)
python models/train.py

# 4. Build retrieval index (~5 min)
python -c "from retrieval.hybrid_retriever import build_and_save_index; build_and_save_index()"

# 5. Run evaluation
python notebooks/evaluate.py

# 6. Start API
uvicorn api.main:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

---

## API Reference

### `POST /predict`

```json
{
  "text": "I need to check on my recent order",
  "session_id": "optional-session-uuid"
}
```

Response:
```json
{
  "text": "I need to check on my recent order",
  "prediction": {
    "intent": "order_status",
    "confidence": 0.9742,
    "source": "bert",
    "low_confidence": false,
    "top3": [
      ["order_status", 0.9742],
      ["track_package", 0.0183],
      ["cancel_order", 0.0041]
    ]
  },
  "session_id": "abc123",
  "dialog_state": "in_progress",
  "turn_count": 1,
  "resolved": false,
  "escalate": false,
  "avg_handle_turns": 1.0
}
```

### `GET /intents`
Returns all 151 intent labels (150 + out-of-scope).

### `GET /health`
Health check.

---

## Reusability

Both core algorithms are designed as standalone, portable modules:

**`HybridRetriever`** — accepts any `(texts, labels)` corpus. Plug in a different dataset and call `build_and_save_index()` to get a production-ready retrieval layer for any classification problem.

**`DialogStateTracker`** — stateless `SessionState` dataclass is JSON-serializable. Drop it into any conversational system; swap `SLOT_PATTERNS` for your domain.

---

## Skills demonstrated

`BERT` · `HuggingFace Transformers` · `NLP` · `Intent Classification` · `Information Retrieval` · `BM25` · `Sentence Transformers` · `Dialog Management` · `FastAPI` · `Agentic AI` · `Python` · `scikit-learn` · `Reusable Algorithms`
