"""Per-user analytics computed in SQL, without row-limit truncation."""
from fastapi import APIRouter, Depends
from app.core.auth import current_user
from app.core.database import get_supabase
from app.models.schemas import AnalyticsSummary
router = APIRouter(prefix="/api/analytics", tags=["analytics"])

@router.get("/summary", response_model=AnalyticsSummary)
def summary(user_id: str = Depends(current_user)):
    data = get_supabase().rpc("study_analytics", {"owner_id": user_id}).execute().data
    return data[0] if isinstance(data, list) else data
