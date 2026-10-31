"""Bounded retrieval correction preserving the student's resolved intent."""
from app.models.schemas import AgentState
from app.services.llm_client import chat

def rewrite_query_node(state: AgentState) -> AgentState:
    state["correction_attempts"] = state.get("correction_attempts", 0) + 1
    original = state.get("resolved_question") or state["original_question"]
    try:
        query = chat("Rewrite a failed academic search query. Preserve the topic and intent; add useful synonyms, not unrelated topics. Return only the search query.",
                     f"Question: {original}\nPrevious searches: {state.get('tried_queries', [])}\nProblem: {state.get('evaluation_reason', '')}",
                     max_tokens=250)
    except Exception:
        query = original + " explanation definition examples"
    if query.casefold().strip() in {q.casefold().strip() for q in state.get("tried_queries", [])}:
        state["insufficient_context"] = True
    state["question"] = query
    state.setdefault("pipeline_log", []).append({"name": "Query rewritten", "status": "done", "detail": query})
    return state
