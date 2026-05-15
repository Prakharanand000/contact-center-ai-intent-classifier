"""
Dialog State Tracker.

Maintains a 3-turn context window per session.
Tracks:
  - intent history (last N turns)
  - extracted slots (entities)
  - current dialog state (greeting / in_progress / resolved / escalate)

Reusable algorithm: SessionState is stateless JSON-serializable,
making it portable across any stateful conversation system.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import time

# Slot definitions per intent domain (extensible)
SLOT_PATTERNS = {
    "order_status":       ["order_number", "email"],
    "cancel_order":       ["order_number", "reason"],
    "track_package":      ["tracking_number", "carrier"],
    "refund_request":     ["order_number", "amount", "reason"],
    "account_login":      ["username"],
    "change_password":    ["username"],
    "bill_inquiry":       ["account_number", "billing_period"],
    "payment_issue":      ["account_number", "amount"],
    "product_inquiry":    ["product_name", "sku"],
    "store_hours":        ["location"],
}

MAX_TURNS = 3


@dataclass
class Turn:
    turn_id: int
    utterance: str
    intent: str
    confidence: float
    slots: dict = field(default_factory=dict)
    source: str = "bert"          # "bert" | "retrieval"
    timestamp: float = field(default_factory=time.time)


@dataclass
class SessionState:
    session_id: str
    turns: list = field(default_factory=list)
    resolved: bool = False
    escalate: bool = False

    @property
    def current_intent(self) -> Optional[str]:
        return self.turns[-1].intent if self.turns else None

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def dialog_state(self) -> str:
        if self.escalate:
            return "escalate"
        if self.resolved:
            return "resolved"
        if self.turn_count == 0:
            return "greeting"
        return "in_progress"

    def to_dict(self):
        d = asdict(self)
        d["current_intent"] = self.current_intent
        d["dialog_state"] = self.dialog_state
        return d


def extract_slots(utterance: str, intent: str) -> dict:
    """
    Lightweight regex-free slot extractor.
    Looks for known keywords in the utterance for the given intent.
    In production this would use a dedicated NER model.
    """
    slots = {}
    required = SLOT_PATTERNS.get(intent, [])
    utt_lower = utterance.lower()

    for slot in required:
        # Simple presence detection — placeholder for NER
        if any(kw in utt_lower for kw in slot.replace("_", " ").split()):
            slots[slot] = "__detected__"

    return slots


class DialogStateTracker:
    """
    Manages per-session state across a sliding window of MAX_TURNS.

    Usage:
        tracker = DialogStateTracker()
        state = tracker.new_session("session_abc")
        state = tracker.update(state, utterance, intent, confidence, source)
        print(state.dialog_state)
    """

    def new_session(self, session_id: str) -> SessionState:
        return SessionState(session_id=session_id)

    def update(
        self,
        state: SessionState,
        utterance: str,
        intent: str,
        confidence: float,
        source: str = "bert",
    ) -> SessionState:
        slots = extract_slots(utterance, intent)
        turn = Turn(
            turn_id=state.turn_count,
            utterance=utterance,
            intent=intent,
            confidence=confidence,
            slots=slots,
            source=source,
        )
        state.turns.append(turn)

        # Keep only last MAX_TURNS in window
        if len(state.turns) > MAX_TURNS:
            state.turns = state.turns[-MAX_TURNS:]

        # Resolution logic
        state.resolved = self._check_resolved(state)
        state.escalate = self._check_escalate(state)
        return state

    def _check_resolved(self, state: SessionState) -> bool:
        """Mark resolved if same high-confidence intent appears 2 turns in a row."""
        if len(state.turns) < 2:
            return False
        last_two = state.turns[-2:]
        return (
            last_two[0].intent == last_two[1].intent
            and last_two[1].confidence >= 0.80
        )

    def _check_escalate(self, state: SessionState) -> bool:
        """Escalate if intent flips 3 times or all turns are low confidence."""
        if len(state.turns) < MAX_TURNS:
            return False
        intents = [t.intent for t in state.turns[-MAX_TURNS:]]
        all_low = all(t.confidence < 0.50 for t in state.turns[-MAX_TURNS:])
        intent_flips = len(set(intents)) == MAX_TURNS
        return all_low or intent_flips

    def get_avg_handle_turns(self, state: SessionState) -> float:
        """
        Proxy metric for average handle time.
        Returns number of turns needed to reach resolution.
        Lower = better. Baseline (no model) ~ 3.0 turns.
        """
        if state.resolved:
            resolved_turn = next(
                (i + 1 for i, t in enumerate(state.turns) if t.confidence >= 0.80), state.turn_count
            )
            return float(resolved_turn)
        return float(state.turn_count)
