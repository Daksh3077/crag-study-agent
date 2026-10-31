"""API contracts and all values carried across LangGraph edges."""
from datetime import datetime
from typing import Any, Literal, TypedDict
from uuid import UUID
from pydantic import BaseModel, Field

StudyMode = Literal["ask", "quick_revision", "detailed_summary", "exam_answer", "important_questions", "mcq"]
ValidationStatus = Literal["supported", "unsupported", "unavailable", "insufficient_context", "generation_failed", "not_checked"]

class DocumentOut(BaseModel):
    id: str
    user_id: str
    name: str
    subject: str | None = None
    file_path: str
    created_at: datetime
    status: str = "ready"
    page_count: int = 0
    processed_pages: int = 0
    chunk_count: int = 0
    ingestion: dict[str, Any] = Field(default_factory=dict)

class ChatRequest(BaseModel):
    request_id: UUID | None = None
    conversation_id: UUID | None = None
    question: str = Field(min_length=1, max_length=4000)
    document_ids: list[UUID] = Field(default_factory=list, max_length=30)
    mode: StudyMode = "ask"
    strict_mode: bool = True
    verification_mode: Literal["adaptive", "full"] = "adaptive"

class SourceCitation(BaseModel):
    citation_id: str
    document_id: str
    document_name: str
    page: int | None = None
    chunk_index: int | None = None
    snippet: str

class PipelineStep(BaseModel):
    name: str
    status: Literal["done", "skipped", "running", "error", "warning"]
    detail: str | None = None

class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    sources: list[SourceCitation]
    relevance_score: float
    groundedness_confidence: float | None
    validation_status: ValidationStatus
    warnings: list[str] = Field(default_factory=list)
    correction_attempts: int
    generation_attempts: int = 0
    used_external_fallback: bool = False
    pipeline: list[PipelineStep]
    response_time_ms: int
    stage_timings_ms: dict[str, float] = Field(default_factory=dict)

class AnalyticsSummary(BaseModel):
    total_documents: int
    total_questions: int
    average_confidence: float
    average_correction_attempts: float
    fallback_usage_rate: float = 0
    average_response_time_ms: float
    supported_answers: int = 0
    validation_unavailable: int = 0

class AgentState(TypedDict, total=False):
    question: str
    original_question: str
    resolved_question: str
    document_ids: list[str]
    mode: StudyMode
    strict_mode: bool
    verification_mode: Literal["adaptive", "full"]
    allow_fast_answer: bool
    fast_answer: bool
    stage_timings_ms: dict[str, float]
    history: list[dict[str, str]]
    retrieved_chunks: list[dict[str, Any]]
    filtered_chunks: list[dict[str, Any]]
    coverage_chunks: list[dict[str, Any]]
    evidence_chunks: list[dict[str, Any]]
    evidence_context: str
    cited_source_ids: list[str]
    generic_request: bool
    relevance_score: float
    context_sufficient: bool
    needs_correction: bool
    evaluation_reason: str
    insufficient_context: bool
    correction_attempts: int
    generation_attempts: int
    citation_format_repairs: int
    used_external_fallback: bool
    tried_queries: list[str]
    retrieval_history: list[list[str]]
    generated_answer: str
    grounded: bool
    groundedness_confidence: float | None
    unsupported_claims: list[str]
    validation_status: ValidationStatus
    generation_error: str
    generation_format_error: str
    answer_blocks: list[dict[str, Any]]
    verified_answer_blocks: list[dict[str, Any]]
    coverage_topics: list[str]
    coverage_missing_topics: list[str]
    answer_coverage_gaps: list[str]
    answer_cache_key: str
    answer_cache_hit: bool
    validation_input_error: str
    summary_notes: list[str]
    summary_section_count: int
    warnings: list[str]
    pipeline_log: list[dict[str, str]]
