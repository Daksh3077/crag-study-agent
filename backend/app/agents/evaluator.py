"""Relevance gating. These scores are signals, not calibrated probabilities."""
import re
from app.core.config import get_settings
from app.models.schemas import AgentState
from app.services.retrieval_service import has_topic_match

def evaluate_node(state: AgentState) -> AgentState:
    chunks = state.get("filtered_chunks", [])
    score = max((c.get("_rerank_score", 0) for c in chunks), default=0)
    question = state.get("resolved_question") or state.get("original_question") or state["question"]
    lexical_match = any(has_topic_match(question, c.get("content", "")) for c in chunks)
    lexical_recovery = bool(chunks) and score < get_settings().relevance_threshold and lexical_match and not state.get("generic_request")
    sufficient = bool(chunks) and (state.get("generic_request", False) or score >= get_settings().relevance_threshold or lexical_match)
    # Relevance is a routing signal, never a claim that the answer is verified.
    state["fast_answer"] = bool(
        state.get("verification_mode") == "adaptive" and state.get("allow_fast_answer", True)
        and state.get("mode") == "ask" and not state.get("generic_request")
        and not re.search(r'\b(?:all|every|compare|comparison)\b', question, re.I)
        and not state.get("warnings") and not state.get("correction_attempts", 0) and not lexical_recovery
        and score >= get_settings().fast_answer_relevance_threshold
        and not any("[illegible]" in c.get("content", "").lower() for c in chunks))
    reason = ("matching topic terms found despite a low ranking score; attempt an answer with a full source check."
              if lexical_recovery else "evidence available." if sufficient else "retrieve a better explanation of the requested topic.")
    state.update(relevance_score=score, context_sufficient=sufficient, needs_correction=not sufficient,
                 evaluation_reason=f"Best passage relevance was {score:.3f}; " + reason)
    state.setdefault("pipeline_log", []).append({"name": "Context evaluated", "status": "done", "detail": state["evaluation_reason"]})
    return state
