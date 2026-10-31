"""Separate document-wide instructions from actual academic query topics."""
import re
from app.core.config import get_settings
from app.models.schemas import AgentState
from app.services.retrieval_service import _all_candidate_chunks, _get_bm25_index, hybrid_retrieve
from app.services.llm_client import chat

_REFERENCES = re.compile(r"\b(?:from|using|of|in|on|based on)?\s*(?:the |this |these |my |uploaded )+(?:uploaded )?(?:pdfs?|documents?|files?|notes?|material)\b", re.I)
_INSTRUCTIONS = re.compile(r"\b(?:give|show|tell|generate|create|make|write|prepare|produce|explain|summarize|summarise|provide|list|me|some|important|all|overview|brief|detailed|quick|revision|summary|summaries|mcqs?|quiz|quizzes|questions?|answers?|multiple.choice|flashcards?|please|exam|sheet|a|an|the|from|using|use|based|about|on|of|in|for|and|to)\b", re.I)

def _topic(question: str) -> str:
    text = _REFERENCES.sub(" ", question)
    text = _INSTRUCTIONS.sub(" ", text)
    # Remove counts associated with study output, but preserve topic numbers (IPv6, type 2, chapter 4).
    text = re.sub(r"^\s*\d+\s*", "", text)
    return re.sub(r"\s+", " ", text).strip(" .,?!:-")

def is_generic_document_request(question: str) -> bool:
    return not bool(_topic(question))

def _clean_retrieval_query(question: str, mode: str) -> str:
    return question if mode == "ask" else (_topic(question) or question)

def resolve_question_node(state: AgentState) -> AgentState:
    original = state["original_question"]
    history = state.get("history", [])
    resolved = original
    # Resolve pronouns and short follow-ups while leaving explicit new topics alone.
    if history and re.search(r"\b(it|its|they|their|them|that|those|above|more|again)\b", original, re.I):
        try:
            resolved = chat("Rewrite the last student question as a standalone question using the conversation only to resolve references. Preserve intent. Do not answer. Return only the question.",
                            f"Conversation: {history[-6:]}\nLast question: {original}", max_tokens=300)
        except Exception:
            previous = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
            resolved = f"{original} (Previous topic: {previous})"
    state.update(question=resolved, resolved_question=resolved, generic_request=is_generic_document_request(resolved))
    return state

def _generic_document_retrieve(document_ids: list[str], top_k: int) -> list[dict]:
    rows = sorted(_all_candidate_chunks(document_ids), key=lambda c: (c["document_id"], c["chunk_index"]))
    if len(rows) <= top_k:
        return rows
    if top_k == 1:
        return rows[:1]
    return [rows[round(i * (len(rows) - 1) / (top_k - 1))] for i in range(top_k)]

def retrieve_node(state: AgentState) -> AgentState:
    query = _clean_retrieval_query(state["question"], state["mode"])
    state.setdefault("tried_queries", []).append(query)
    if state.get("generic_request"):
        chunks = _generic_document_retrieve(state["document_ids"], get_settings().top_k_final)
        state.setdefault("warnings", []).append("Questions sample topics across the selected material; they are not a prediction of your exam.")
    else:
        chunks = hybrid_retrieve(query, state["document_ids"], get_settings().top_k_initial, state.setdefault("warnings", []))
    state['coverage_chunks'] = []
    if (state['mode'] == 'ask' and len(state['document_ids']) == 1
            and re.search(r'\b(?:all|every)\b', state['question'], re.I)):
        # Broad requests on a small cheatsheet need more than the top five
        # excerpts. Reuse the same scoped corpus already loaded for BM25.
        rows, _ = _get_bm25_index(state['document_ids'])
        if len(rows) <= 24:
            state['coverage_chunks'] = sorted(rows, key=lambda c: c['chunk_index'])
    state["retrieved_chunks"] = chunks
    state.setdefault("retrieval_history", []).append([c["id"] for c in chunks])
    state.setdefault("pipeline_log", []).append({"name": "Documents retrieved", "status": "done", "detail": f"{len(chunks)} candidate passages"})
    return state
