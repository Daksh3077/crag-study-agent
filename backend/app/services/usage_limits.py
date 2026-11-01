"""Durable guest allowances; only the authenticated backend can spend them."""
import logging
from fastapi import HTTPException
from app.core.database import get_supabase

logger = logging.getLogger(__name__)
QUESTION_LIMIT = 5
UPLOAD_LIMIT = 2


def consume_usage(user_id: str, kind: str) -> None:
    if kind not in {"questions", "uploads"}:
        raise ValueError("Unknown usage kind")
    try:
        allowed = get_supabase().rpc("consume_guest_usage", {
            "p_user_id": user_id, "p_kind": kind,
        }).execute().data
        if not isinstance(allowed, bool):
            raise ValueError("Invalid usage response")
    except Exception as exc:
        logger.exception("Usage reservation failed")
        raise HTTPException(503, "Could not check your session allowance. Please try again shortly.") from exc
    if not allowed:
        limit = QUESTION_LIMIT if kind == "questions" else UPLOAD_LIMIT
        raise HTTPException(429, f"This guest session has reached its limit of {limit} {kind}. "
                            "You can still view your existing documents and answers.")


def get_usage(user_id: str) -> dict:
    try:
        rows = get_supabase().table("guest_usage").select("questions,uploads").eq("user_id", user_id).execute().data
        row = rows[0] if rows else {"questions": 0, "uploads": 0}
        return {"questions_used": row["questions"], "questions_limit": QUESTION_LIMIT,
                "uploads_used": row["uploads"], "uploads_limit": UPLOAD_LIMIT}
    except Exception as exc:
        logger.exception("Usage read failed")
        raise HTTPException(503, "Could not check your session allowance. Please try again shortly.") from exc
