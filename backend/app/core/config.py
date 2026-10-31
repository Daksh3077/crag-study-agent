"""Validated configuration, loaded from the backend directory regardless of cwd."""
from functools import lru_cache
from pathlib import Path
from typing import Literal
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=Path(__file__).resolve().parents[2] / ".env", extra="ignore")
    llm_provider: Literal["groq", "openai", "anthropic"] = "groq"
    groq_api_key: str = ""
    gemini_api_key: str = ""
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    llm_model: str = "openai/gpt-oss-120b"
    validator_model: str = ""
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    embedding_batch_size: int = Field(64, ge=1, le=128)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    supabase_url: str = ""
    supabase_service_key: str = ""
    supabase_anon_key: str = ""
    storage_bucket: str = "study-originals"
    environment: str = "development"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    max_retries: int = Field(1, ge=0, le=3)
    generation_retries: int = Field(1, ge=0, le=2)
    groundedness_threshold: float = Field(.7, ge=0, le=1)
    relevance_threshold: float = Field(.45, ge=0, le=1)
    fast_answer_relevance_threshold: float = Field(.85, ge=.7, le=1)
    fast_answer_context_tokens: int = Field(3200, ge=1000, le=6000)
    fast_answer_max_tokens: int = Field(1600, ge=500, le=3500)
    top_k_initial: int = Field(24, ge=5, le=80)
    top_k_final: int = Field(8, ge=1, le=20)
    vector_similarity_threshold: float = Field(.25, ge=0, le=1)
    rrf_k: int = Field(60, ge=1)
    neighbor_window: int = Field(1, ge=0, le=2)
    bm25_cache_ttl_seconds: int = Field(60, ge=0)
    bm25_cache_entries: int = Field(16, ge=1)
    context_token_budget: int = Field(6000, ge=1000, le=16000)
    answer_max_tokens: int = Field(3500, ge=500, le=8000)
    summary_section_tokens: int = Field(2500, ge=500, le=5000)
    summary_reduce_tokens: int = Field(6000, ge=2000, le=16000)
    summary_max_sections: int = Field(120, ge=1, le=500)
    vision_provider: Literal["gemini", "groq"] = "gemini"
    vision_model: str = "gemini-3.5-flash-lite"
    gemini_ocr_requests_per_minute: int = Field(10, ge=1, le=1000)
    gemini_ocr_max_output_tokens: int = Field(4096, ge=1024, le=16384)
    gemini_ocr_batch_pages: int = Field(2, ge=1, le=2)
    vision_max_output_tokens: int = Field(1800, ge=256, le=8192)
    ocr_min_words_threshold: int = Field(30, ge=0)
    ocr_render_dpi: int = Field(180, ge=100, le=300)
    ocr_call_delay_seconds: float = Field(3, ge=0)
    ocr_provider_retries: int = Field(6, ge=0, le=10)
    ocr_max_wait_seconds: float = Field(300, ge=0, le=1800)
    provider_max_wait_seconds: float = Field(120, ge=0, le=600)
    validation_batch_tokens: int = Field(3200, ge=1000, le=6000)
    ask_validation_batch_tokens: int = Field(4500, ge=1000, le=6000)
    provider_timeout_seconds: float = Field(90, gt=0)
    provider_retries: int = Field(2, ge=0, le=4)
    max_upload_mb: int = Field(30, ge=1, le=100)
    max_document_pages: int = Field(150, ge=1, le=500)
    chunk_size: int = Field(350, ge=64, le=450)
    chunk_overlap: int = Field(50, ge=0, le=100)

    @model_validator(mode="after")
    def check_limits(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.top_k_final > self.top_k_initial:
            raise ValueError("TOP_K_FINAL must not exceed TOP_K_INITIAL")
        if self.embedding_dim != 384:
            raise ValueError("Migrate vector(384) before changing EMBEDDING_DIM")
        return self

@lru_cache
def get_settings() -> Settings:
    return Settings()
