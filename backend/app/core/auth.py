"""All ownership is derived from a verified Supabase bearer token."""
from uuid import UUID
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from app.core.database import get_supabase

bearer = HTTPBearer(auto_error=False)

def current_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> str:
    if credentials is None:
        raise HTTPException(401, "Your private guest session is missing. Reload the page to reconnect.")
    try:
        user = get_supabase().auth.get_user(credentials.credentials).user
        if user is None:
            raise ValueError("Missing user")
        return str(UUID(user.id))
    except Exception as exc:
        raise HTTPException(401, "Your private guest session could not be verified. Reload the page to reconnect.") from exc

def owned_document(document_id: str, user_id: str) -> dict:
    result = get_supabase().table("documents").select("*").eq("id", document_id).eq("user_id", user_id).execute()
    if not result.data:
        raise HTTPException(404, "Document not found.")
    return result.data[0]
