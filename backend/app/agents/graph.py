"""Bounded retrieval correction, complete answer validation, and evidence-based repair."""
import logging
import time
from functools import wraps
from langgraph.graph import StateGraph, END
from app.core.config import get_settings
from app.models.schemas import AgentState
from app.agents.retrieval import resolve_question_node, retrieve_node
from app.agents.reranker import rerank_node
from app.agents.evaluator import evaluate_node
from app.agents.query_rewriter import rewrite_query_node
from app.agents.generator import generate_node
from app.agents.validator import validate_node
from app.agents.summarizer import full_document_summary_node, repair_document_summary_node
from app.services.provider_limits import observe_waits
from app.services import answer_cache
from app.services.cited_answer import render_answer
from app.services.evidence import cited_ids

def insufficient_context_node(state: AgentState) -> AgentState:
    state.update(insufficient_context=True, grounded=False, groundedness_confidence=None,
                 validation_status="insufficient_context",
                 generated_answer="I couldn't find enough supporting information in the selected documents. Try a more specific question or check the extraction status of the relevant pages.")
    return state

def finish_node(state: AgentState) -> AgentState:
    if state.get('verified_answer_blocks') and state.get('validation_status') != 'supported':
        answer = render_answer(state['verified_answer_blocks'])
        note = 'Partial answer: the remaining requested details could not be verified. Previously checked content is retained.'
        state.update(generated_answer=answer + '\n\n*' + note + '*',
                     cited_source_ids=cited_ids(answer), grounded=True, groundedness_confidence=None,
                     validation_status='supported', validation_input_error='')
        state.setdefault('warnings', []).append(note)
        return state
    if state.get("validation_input_error"):
        state["generated_answer"] = "The answer's citations could not be checked. The document is still indexed; the source-check details below explain what blocked this answer."
        state["cited_source_ids"] = []
        state.setdefault("warnings", []).append(state["validation_input_error"])
        return state
    if state.get("validation_status") == "unsupported":
        state["generated_answer"] = "I found related material, but couldn't produce a fully supported answer after checking it. Review the source pages or ask a narrower question."
        state["cited_source_ids"] = []
    if state.get('answer_cache_key') and not state.get('answer_cache_hit'):
        answer_cache.put(state['answer_cache_key'], state)
    return state

def route_start(state):
    if not state["document_ids"]:
        return "insufficient"
    if state.get("generic_request") and state["mode"] in {"quick_revision", "detailed_summary"}:
        return "summary"
    return "retrieve"

def route_evaluate(state):
    if not state.get("needs_correction"):
        return "generate"
    history = state.get("retrieval_history", [])
    unchanged = len(history) > 1 and history[-1] == history[-2]
    if unchanged or state.get("insufficient_context") or state.get("correction_attempts", 0) >= get_settings().max_retries:
        return "insufficient"
    return "rewrite"

def route_validate(state):
    summary = state.get("generic_request") and state["mode"] in {"quick_revision", "detailed_summary"}
    limit = get_settings().generation_retries
    # A local citation-format failure has not checked factual support yet.
    # Reserve at most one extra factual repair after fixing that format error.
    # Repeated format failures and whole-document summaries keep their old cap.
    if limit and not summary and 'answer_blocks' not in state and not state.get("validation_input_error"):
        limit += min(1, state.get("citation_format_repairs", 0))
    partial = (state.get('validation_status') == 'supported' and state.get('answer_blocks')
               and (state.get('unsupported_claims') or state.get('coverage_missing_topics')))
    if (state.get("validation_status") == "unsupported" or state.get("validation_input_error") or partial) and state.get("generation_attempts", 0) <= limit:
        if summary:
            return "summary_repair"
        return "generate"
    return "finish"

def timed_node(name, node):
    @wraps(node)
    def run(state):
        started = time.perf_counter()
        result = node(state)
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        timings = dict(result.get("stage_timings_ms", {}))
        timings[name] = round(timings.get(name, 0) + elapsed, 1)
        result["stage_timings_ms"] = timings
        logging.getLogger(__name__).info("Answer stage %s: %.1f ms", name, elapsed)
        return result
    return run

def build_graph():
    answer_cache.clear()
    graph = StateGraph(AgentState)
    for name, node in {"resolve": resolve_question_node, "retrieve": retrieve_node, "rerank": rerank_node,
                       "evaluate": evaluate_node, "rewrite": rewrite_query_node, "generate": generate_node,
                       "validate": validate_node, "summary": full_document_summary_node, "summary_repair": repair_document_summary_node,
                       "insufficient": insufficient_context_node, "finish": finish_node}.items():
        graph.add_node(name, timed_node(name, node))
    graph.set_entry_point("resolve")
    graph.add_conditional_edges("resolve", route_start, {"summary": "summary", "retrieve": "retrieve", "insufficient": "insufficient"})
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "evaluate")
    graph.add_conditional_edges("evaluate", route_evaluate, {"generate": "generate", "rewrite": "rewrite", "insufficient": "insufficient"})
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", "validate")
    graph.add_edge("summary", "validate")
    graph.add_edge("summary_repair", "validate")
    graph.add_conditional_edges("validate", route_validate, {"generate": "generate", "summary_repair": "summary_repair", "finish": "finish"})
    graph.add_edge("insufficient", END)
    graph.add_edge("finish", END)
    return graph.compile()

_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph

def run_agent(question: str, document_ids: list[str], mode: str = "ask", strict_mode: bool = True,
              history: list[dict[str, str]] | None = None, verification_mode: str = "adaptive",
              allow_fast_answer: bool = True) -> AgentState:
    waits = []
    with observe_waits(waits.append):
        result = get_graph().invoke({"question": question, "original_question": question, "document_ids": document_ids,
        "mode": mode, "strict_mode": strict_mode, "history": history or [], "correction_attempts": 0,
        "verification_mode": verification_mode, "allow_fast_answer": allow_fast_answer, "stage_timings_ms": {},
        "generation_attempts": 0, "tried_queries": [], "retrieval_history": [], "warnings": [], "pipeline_log": [],
        "validation_status": "unavailable", "groundedness_confidence": None}, {"recursion_limit": 40})
    if waits:
        result.setdefault('pipeline_log', []).append({'name': 'Provider quota waits', 'status': 'warning',
            'detail': f'The model provider requested {sum(waits):.1f}s of quota waits. These waits are included in the generation and validation times.'})
    return result
