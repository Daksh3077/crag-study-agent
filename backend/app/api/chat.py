"""Authenticated chat, persisted history, source citations and honest status."""
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from app.core.auth import current_user
from app.core.database import get_supabase
from app.models.schemas import ChatRequest, ChatResponse, SourceCitation, PipelineStep
from app.agents.graph import run_agent
from app.services.llm_client import public_error
from app.services import chat_progress
from app.services.provider_limits import observe_quota
from app.services.usage_limits import consume_usage
router = APIRouter(prefix="/api/chat", tags=["chat"])
logger = logging.getLogger(__name__)

@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest, user_id: str = Depends(current_user)):
    if request.request_id is None:
        return _answer(request, user_id)
    request_id = str(request.request_id)
    try:
        chat_progress.begin(user_id, request_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except OverflowError as exc:
        raise HTTPException(429, str(exc)) from exc
    try:
        with observe_quota(lambda key, seconds: chat_progress.quota(user_id, request_id, key, seconds)):
            response = _answer(request, user_id)
    except Exception:
        chat_progress.finish(user_id, request_id, failed=True)
        raise
    chat_progress.finish(user_id, request_id)
    return response


@router.get('/requests/{request_id}')
def request_progress(request_id: uuid.UUID, user_id: str = Depends(current_user)):
    status = chat_progress.get(user_id, str(request_id))
    if status is None:
        raise HTTPException(404, 'Request status not found.')
    return JSONResponse(status, headers={'Cache-Control': 'no-store'})


def _answer(request: ChatRequest, user_id: str):
    if not request.question.strip():
        raise HTTPException(422, "Enter a question.")
    if not request.strict_mode:
        raise HTTPException(422, "This workspace answers from uploaded documents only.")
    sb = get_supabase()
    docs = sb.table("documents").select("*").eq("user_id", user_id).execute().data or []
    requested = {str(i) for i in request.document_ids}
    if requested - {d["id"] for d in docs}:
        raise HTTPException(404, "One or more selected documents were not found.")
    selected = [d for d in docs if not requested or d["id"] in requested]
    ready = [d for d in selected if d["status"] in {"ready", "partial"}]
    if not ready:
        raise HTTPException(409, "Upload a document and wait until its readable pages are indexed. Older documents need to be re-uploaded.")
    warnings = []
    if len(ready) != len(selected):
        warnings.append("Documents still processing, failed, or requiring reindexing were excluded.")
    for doc in ready:
        report = doc.get("ingestion") or {}
        failed = [str(p["page"]) for p in report.get("pages", []) if p.get("status") == "failed"]
        pending = [str(p["page"]) for p in report.get("pages", []) if p.get("status") == "pending"]
        uncertain = [str(p["page"]) for p in report.get("pages", []) if p.get("warning")]
        if failed:
            warnings.append(f"{doc['name']}: pages {', '.join(failed)} could not be read and are excluded.")
        if pending:
            warnings.append(f"{doc['name']}: pages {', '.join(pending)} are waiting for provider quota and are excluded. Resume processing from Documents.")
        if uncertain:
            warnings.append(f"{doc['name']}: check unclear handwriting on pages {', '.join(uncertain)}.")
    conversation_id = str(request.conversation_id) if request.conversation_id else str(uuid.uuid4())
    history = []
    if request.conversation_id:
        owned = sb.table("conversations").select("id").eq("id", conversation_id).eq("user_id", user_id).execute().data
        if not owned:
            raise HTTPException(404, "Conversation not found.")
        history = sb.table("messages").select("role,content").eq("conversation_id", conversation_id).order("created_at", desc=True).limit(12).execute().data or []
        history.reverse()
    consume_usage(user_id, "questions")
    if not request.conversation_id:
        sb.table("conversations").insert({"id": conversation_id, "user_id": user_id, "title": request.question[:60]}).execute()
    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    try:
        state = run_agent(request.question.strip(), [d["id"] for d in ready], request.mode, request.strict_mode, history,
                          verification_mode=request.verification_mode,
                          allow_fast_answer=not warnings and all(d["status"] == "ready" for d in ready))
    except Exception as exc:
        logger.exception("Agent failed")
        raise HTTPException(503, public_error(exc)) from exc
    cited = set(state.get("cited_source_ids", []))
    sources = [SourceCitation(citation_id=c["citation_id"], document_id=c["document_id"],
                 document_name=(c.get("metadata") or {}).get("document_name", "Document"),
                 page=c.get("page_number"), chunk_index=c.get("chunk_index"), snippet=c["content"])
               for c in state.get("evidence_chunks", []) if c.get("citation_id") in cited]
    answer = state.get("generated_answer", "")
    missing = any(p.get("status") in {"failed", "pending"} for d in ready for p in (d.get("ingestion") or {}).get("pages", []))
    if missing and request.mode in {"detailed_summary", "quick_revision"}:
        answer = "**Partial coverage: some pages have not been extracted. This summary covers the readable pages only.**\n\n" + answer
    response = ChatResponse(conversation_id=conversation_id, answer=answer,
        sources=sources, relevance_score=state.get("relevance_score", 0),
        groundedness_confidence=state.get("groundedness_confidence"), validation_status=state.get("validation_status", "unavailable"),
        warnings=list(dict.fromkeys(warnings + state.get("warnings", []))), correction_attempts=state.get("correction_attempts", 0),
        generation_attempts=state.get("generation_attempts", 0),
        pipeline=[PipelineStep(name="Documents searched", status="done",
                    detail="; ".join(f"{d['name']} ({d.get('chunk_count', 0)} passages)" for d in ready))]
                 + [PipelineStep(**s) for s in state.get("pipeline_log", [])],
        stage_timings_ms=state.get("stage_timings_ms", {}),
        response_time_ms=int((time.perf_counter() - start) * 1000))
    try:
        sb.table("messages").insert([
            {"id": str(uuid.uuid4()), "conversation_id": conversation_id, "role": "user", "content": request.question, "metadata": {}, "created_at": started_at.isoformat()},
            {"id": str(uuid.uuid4()), "conversation_id": conversation_id, "role": "assistant", "content": response.answer,
             "metadata": response.model_dump(), "created_at": max(datetime.now(timezone.utc), started_at + timedelta(microseconds=1)).isoformat()}]).execute()
    except Exception:
        logger.exception("Conversation persistence failed")
        response.warnings.append("This answer could not be saved to conversation history.")
    try:
        sb.table("retrieval_logs").insert({"user_id": user_id, "conversation_id": conversation_id, "query": request.question,
            "retrieved_chunks": [c["id"] for c in state.get("evidence_chunks", []) if c.get("citation_id") in cited],
            "confidence_score": response.groundedness_confidence, "correction_attempts": response.correction_attempts,
            "final_status": response.validation_status, "response_time": response.response_time_ms,
            "metadata": {"mode": request.mode, "generation_attempts": response.generation_attempts,
                         "document_ids": [d["id"] for d in ready], "relevance_score": response.relevance_score,
                         "candidate_chunk_ids": [c["id"] for c in state.get("retrieved_chunks", [])],
                         "ranked_passages": [{"id": c["id"], "score": c.get("_rerank_score")} for c in state.get("filtered_chunks", [])],
                         "validation_input_error": state.get("validation_input_error")}}).execute()
    except Exception:
        logger.exception("Analytics persistence failed")
    return response

@router.get("/conversations")
def conversations(user_id: str = Depends(current_user)):
    return get_supabase().table("conversations").select("id,title,created_at").eq("user_id", user_id).order("created_at", desc=True).limit(50).execute().data

@router.get("/conversations/{conversation_id}")
def conversation(conversation_id: uuid.UUID, user_id: str = Depends(current_user)):
    sb = get_supabase()
    owned = sb.table("conversations").select("id").eq("id", str(conversation_id)).eq("user_id", user_id).execute().data
    if not owned:
        raise HTTPException(404, "Conversation not found.")
    return sb.table("messages").select("role,content,metadata").eq("conversation_id", str(conversation_id)).order("created_at").order("id").execute().data
