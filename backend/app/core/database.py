"""
Supabase client singleton. All persistence (documents, chunks, conversations,
messages, retrieval_logs) goes through this client.
"""
from functools import lru_cache
from supabase import create_client, Client
from app.core.config import get_settings


@lru_cache
def get_supabase() -> Client:
    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_service_key:
        raise RuntimeError(
            "SUPABASE_URL / SUPABASE_SERVICE_KEY are not set. "
            "Copy .env.example to .env and fill them in."
        )
    return create_client(settings.supabase_url, settings.supabase_service_key)
