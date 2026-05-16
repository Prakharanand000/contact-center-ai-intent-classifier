"""
FastAPI serving layer.

Endpoints:
  GET  /                 - frontend UI
  POST /predict          - single utterance, returns intent + dialog state
  GET  /health           - health check
  GET  /intents          - list all known intents

Run locally:
    uvicorn api.main:app --reload --port 8000

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
from pydantic import BaseModel
from typing import Optional

from models.predict import get_classifier
from dialog.state_tracker import DialogStateTracker

# Only import retriever if index or training data exists
_RETRIEVAL_AVAILABLE = os.path.exists(
    os.path.join(os.path.dirname(__file__), "..", "retrieval", "retrieval_index.pkl")
) or os.path.exists(
    os.path.join(os.path.dirname(__file__), "..", "data", "train.csv")
)

app = FastAPI(
    title="Contact Center AI Intent Classifier",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_api_dir = os.path.dirname(__file__)
_sessions: dict = {}
_tracker = DialogStateTracker()


class PredictRequest(BaseModel):
    text: str
    session_id: Optional[str] = None

class TurnResult(BaseModel):
    intent: str
    confidence: float
    source: str
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


def run_prediction(text: str) -> dict:
    classifier = get_classifier()
    result = classifier.predict(text)

    source = "bert"
    retrieved = []

    # Only attempt retrieval fallback if data/index is available
    if result["low_confidence"] and _RETRIEVAL_AVAILABLE:
        try:
            from retrieval.hybrid_retriever import get_retriever
            retriever = get_retriever()
            retrieval_result = retriever.predict_from_retrieval(text)
            retrieved = retrieval_result["retrieved"]
            if retrieval_result["confidence"] > result["confidence"]:
                result["intent"] = retrieval_result["intent"]
                result["confidence"] = retrieval_result["confidence"]
                source = "retrieval_fallback"
        except Exception as e:
            print(f"Retrieval fallback skipped: {e}")

    return {**result, "source": source, "retrieved_examples": retrieved}


@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(os.path.join(_api_dir, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "model": "bert-base-uncased-clinc150", "retrieval": _RETRIEVAL_AVAILABLE}


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
