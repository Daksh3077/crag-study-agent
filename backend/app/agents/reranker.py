"""Configured local cross-encoder; neighbor expansion happens after ranking."""
import logging
import math
from app.core.config import get_settings
from app.models.schemas import AgentState
from app.services.retrieval_service import _tokenize
from app.services.local_models import cached_model, prefer_local

@cached_model
def _get_reranker():
    from sentence_transformers import CrossEncoder
    return prefer_local(CrossEncoder, get_settings().reranker_model)

def rerank_node(state: AgentState) -> AgentState:
    chunks = state.get("retrieved_chunks", [])
    if not chunks or state.get("generic_request"):
        state["filtered_chunks"] = chunks
        return state
    question = state.get("resolved_question") or state["question"]
    try:
        scores = _get_reranker().predict([(question, c["content"]) for c in chunks], batch_size=16, show_progress_bar=False)
        scored = [dict(c, _rerank_score=1 / (1 + math.exp(-max(-30, min(30, float(s))))))
                  for c, s in zip(chunks, scores)]
    except Exception:
        logging.getLogger(__name__).exception("Semantic reranking failed; using keyword ranking")
        query_tokens = set(_tokenize(question))
        scored = [dict(c, _rerank_score=len(query_tokens & set(_tokenize(c["content"]))) / max(len(query_tokens), 1)) for c in chunks]
        state.setdefault("warnings", []).append("The semantic reranker was unavailable; keyword ranking was used.")
    ranked = sorted(scored, key=lambda c: c["_rerank_score"], reverse=True)
    heading = next((c for c in ranked if c.get("topic_heading_match")), None)
    # Reserve one seed for an exact topic heading; keep the original score so
    # the evaluator still requires a full check for weak semantic matches.
    if heading:
        ranked = [heading] + [c for c in ranked if c["id"] != heading["id"]]
    seeds = ranked[:get_settings().top_k_final]
    state["filtered_chunks"] = seeds
    state.setdefault("pipeline_log", []).append({"name": "Context ranked", "status": "done",
        "detail": f"{len(seeds)} passages selected" + ("; includes an exact topic heading." if heading else ".")})
    return state
