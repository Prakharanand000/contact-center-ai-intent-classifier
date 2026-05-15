"""
FastAPI serving layer.

Endpoints:
  GET  /                 — frontend UI
  POST /predict          — single utterance, returns intent + dialog state
  GET  /health           — health check
  GET  /intents          — list all known intents

Run locally:
    uvicorn api.main:app --reload --port 8000
    Then open: http://localhost:8000

Deploy to Render:
    Start command: uvicorn api.main:app --host 0.0.0.0 --port $PORT
"""

import uuid
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

from models.predict import get_classifier
from retrieval.hybrid_retriever import get_retriever
from dialog.state_tracker import DialogStateTracker

app = FastAPI(
    title="Contact Center AI Intent Classifier",
    description=(
        "Production-grade NLP intent classification pipeline using BERT fine-tuning "
        "and hybrid Information Retrieval (BM25 + dense vector search). "
        "Built for Contact Center AI use cases."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend
_api_dir = os.path.dirname(__file__)

# In-memory session store (use Redis in production)
_sessions: dict = {}
_tracker = DialogStateTracker()


# ── Request/Response schemas ──────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str
    session_id: Optional[str] = None

class TurnResult(BaseModel):
    intent: str
    confidence: float
    source: str          # "bert" | "retrieval_fallback"
    low_confidence: bool
    top3: list
    retrieved_examples: list = []

class PredictResponse(BaseModel):
    text: str
    prediction: TurnResult
    session_id: str
    dialog_state: str
    turn_count: int
    resolved: bool
    escalate: bool
    avg_handle_turns: float


# ── Core prediction logic ─────────────────────────────────────────────────────

def run_prediction(text: str) -> dict:
    """
    Two-stage pipeline:
      1. BERT classifier — fast, high accuracy
      2. Hybrid retrieval fallback if confidence < threshold
    """
    classifier = get_classifier()
    result = classifier.predict(text)

    source = "bert"
    retrieved = []

    if result["low_confidence"]:
        retriever = get_retriever()
        retrieval_result = retriever.predict_from_retrieval(text)
        retrieved = retrieval_result["retrieved"]

        if retrieval_result["confidence"] > result["confidence"]:
            result["intent"] = retrieval_result["intent"]
            result["confidence"] = retrieval_result["confidence"]
            source = "retrieval_fallback"

    return {**result, "source": source, "retrieved_examples": retrieved}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(os.path.join(_api_dir, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "model": "bert-base-uncased-clinc150"}


@app.get("/intents")
def list_intents():
    classifier = get_classifier()
    return {"intents": list(classifier.id2label.values()), "count": len(classifier.id2label)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")

    pred = run_prediction(req.text)

    session_id = req.session_id or str(uuid.uuid4())
    if session_id not in _sessions:
        _sessions[session_id] = _tracker.new_session(session_id)

    state = _sessions[session_id]
    state = _tracker.update(
        state,
        utterance=req.text,
        intent=pred["intent"],
        confidence=pred["confidence"],
        source=pred["source"],
    )
    _sessions[session_id] = state

    return PredictResponse(
        text=req.text,
        prediction=TurnResult(
            intent=pred["intent"],
            confidence=pred["confidence"],
            source=pred["source"],
            low_confidence=pred["low_confidence"],
            top3=pred["top3"],
            retrieved_examples=pred.get("retrieved_examples", [])[:3],
        ),
        session_id=session_id,
        dialog_state=state.dialog_state,
        turn_count=state.turn_count,
        resolved=state.resolved,
        escalate=state.escalate,
        avg_handle_turns=_tracker.get_avg_handle_turns(state),
    )
