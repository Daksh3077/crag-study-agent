from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from app.core.auth import current_user
from app.services.usage_limits import get_usage

router = APIRouter(prefix="/api/usage", tags=["usage"])


@router.get("")
def usage(user_id: str = Depends(current_user)):
    return JSONResponse(get_usage(user_id), headers={"Cache-Control": "no-store"})
