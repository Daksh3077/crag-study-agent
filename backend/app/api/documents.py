"""Private original files, bounded uploads and background page extraction."""
import logging
import math
import uuid
from pathlib import PurePosixPath
from typing import Literal
from uuid import UUID
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from app.core.auth import current_user, owned_document
from app.core.config import get_settings
from app.core.database import get_supabase
from app.models.schemas import DocumentOut
from app.services.document_processor import process_document
from app.services.provider_limits import saved_retry_seconds, quota_matches
from app.services.retrieval_service import invalidate_bm25_cache
from app.services.usage_limits import consume_usage
router = APIRouter(prefix="/api/documents", tags=["documents"])
MIME = {".pdf": "application/pdf", ".txt": "text/plain", ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}

def document_for_display(document):
    """A previous provider's deadline must not disable Resume in the dashboard."""
    settings = get_settings()
    report = dict(document.get("ingestion") or {})
    pages = []
    for original in report.get("pages") or []:
        page = dict(original)
        if not quota_matches(page, settings.vision_provider, settings.vision_model):
            page.pop("retry_at", None)
        pages.append(page)
    report["pages"] = pages
    return {**document, "ingestion": report}

@router.post("/upload", status_code=202, response_model=DocumentOut)
def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...),
                    subject: str = Form(""), ocr_mode: Literal["auto", "force"] = Form("auto"),
                    user_id: str = Depends(current_user)):
    settings, sb = get_settings(), get_supabase()
    filename = PurePosixPath((file.filename or "").replace("\\", "/")).name
    extension = PurePosixPath(filename).suffix.lower()
    if extension not in MIME:
        raise HTTPException(400, "Upload a PDF, TXT, JPG, PNG or WEBP file.")
    filename = PurePosixPath(filename).stem[:220] + extension
    queued = sb.table("documents").select("id", count="exact").eq("user_id", user_id).in_("status", ["queued", "processing"]).execute()
    if (queued.count or 0) >= 5:
        raise HTTPException(429, "Wait for some of your queued documents to finish before uploading more.")
    data = file.file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if not data or len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"Upload a non-empty file up to {settings.max_upload_mb} MB.")
    if extension == ".pdf" and not data.lstrip().startswith(b"%PDF-"):
        raise HTTPException(400, "This file is not a valid PDF.")
    consume_usage(user_id, "uploads")
    document_id = str(uuid.uuid4())
    path = f"{user_id}/{document_id}/original{extension}"
    bucket = sb.storage.from_(settings.storage_bucket)
    bucket.upload(path, data, {"content-type": MIME[extension]})
    try:
        result = sb.table("documents").insert({"id": document_id, "user_id": user_id, "name": filename[:240],
                    "subject": subject[:200] or None, "file_path": path, "status": "queued",
                    "ingestion": {"ocr_mode": ocr_mode}}).execute()
    except Exception:
        bucket.remove([path])
        raise
    background_tasks.add_task(process_document, document_id)
    return result.data[0]

@router.get("", response_model=list[DocumentOut])
def list_documents(user_id: str = Depends(current_user)):
    rows = get_supabase().table("documents").select("*").eq("user_id", user_id).order("created_at", desc=True).execute().data
    return [document_for_display(row) for row in rows]

@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: UUID, user_id: str = Depends(current_user)):
    return document_for_display(owned_document(str(document_id), user_id))

@router.get("/{document_id}/source")
def document_source(document_id: UUID, user_id: str = Depends(current_user)):
    doc = owned_document(str(document_id), user_id)
    try:
        result = get_supabase().storage.from_(get_settings().storage_bucket).create_signed_url(doc["file_path"], 300)
        return {"url": result["signedURL"]}
    except Exception as exc:
        raise HTTPException(404, "Original file unavailable. Re-upload this older document to enable source previews.") from exc

@router.post("/{document_id}/retry", status_code=202)
def retry_document(document_id: UUID, background_tasks: BackgroundTasks, user_id: str = Depends(current_user)):
    doc = owned_document(str(document_id), user_id)
    if doc["status"] in ("processing", "queued"):
        raise HTTPException(409, "This document is already being processed.")
    if doc["status"] == "needs_reindex":
        raise HTTPException(409, "Re-upload this older document so it can be extracted and indexed consistently.")
    settings = get_settings()
    remaining = saved_retry_seconds((doc.get("ingestion") or {}).get("pages") or [], settings.vision_provider, settings.vision_model)
    if remaining > 0:
        raise HTTPException(429, "OCR is paused for provider quota. Resume after the displayed retry time; completed pages are saved.",
                            headers={"Retry-After": str(math.ceil(remaining))})
    # Atomic state predicate prevents two concurrent retry requests from starting two jobs.
    result = get_supabase().table("documents").update({"status": "queued"}).eq("id", str(document_id)).eq("status", doc["status"]).execute()
    if not result.data:
        raise HTTPException(409, "The document status changed. Refresh and try again.")
    background_tasks.add_task(process_document, str(document_id))
    return {"status": "queued"}

@router.delete("/{document_id}")
def delete_document(document_id: UUID, user_id: str = Depends(current_user)):
    doc = owned_document(str(document_id), user_id)
    if doc["status"] in ("processing", "queued"):
        raise HTTPException(409, "Wait for processing to finish before deleting this document.")
    result = get_supabase().table("documents").delete().eq("id", str(document_id)).eq("user_id", user_id).eq("status", doc["status"]).execute()
    if not result.data:
        raise HTTPException(409, "The document status changed. Refresh and try again.")
    try:
        get_supabase().storage.from_(get_settings().storage_bucket).remove([doc["file_path"], f"{user_id}/{document_id}/pages.json"])
    except Exception:
        logging.getLogger(__name__).exception("Original file cleanup failed")
    invalidate_bm25_cache()
    return {"status": "deleted"}
