"""Resumable page extraction; failed OCR is never indexed as study content."""
import json
import logging
import threading
import time
import uuid
from typing import Callable

import pymupdf as fitz
from app.core.config import get_settings
from app.core.database import get_supabase
from app.services.chunking import chunk_text, clean_text
from app.services.pdf_layout import native_page_text, NATIVE_LAYOUT_VERSION
from app.services.embeddings import EMBEDDING_VERSION, embed_texts, warm_models
from app.services.llm_client import public_error, ProviderConfigurationError, IncompleteOutputError
from app.services.provider_limits import RateLimitDeferred, observe_waits, saved_retry_seconds
from app.services.vision_ocr import transcribe_image, transcribe_page_batch

logger = logging.getLogger(__name__)
_processing_slots = threading.BoundedSemaphore(2)

def usable_text(text: str) -> bool:
    cleaned = text.replace("[illegible]", "").replace("[blank]", "").strip()
    return sum(c.isalnum() for c in cleaned) >= 3

def needs_ocr(page, native: str, force: bool) -> bool:
    if force or len(native.split()) < get_settings().ocr_min_words_threshold:
        return True
    if "\ufffd" in native or sum(c.isalnum() for c in native) / max(len(native), 1) < .35:
        return True
    # A large embedded image can contain handwriting beside a valid printed header.
    area = page.rect.width * page.rect.height
    return any(info["bbox"] and fitz.Rect(info["bbox"]).get_area() / max(area, 1) > .35
               for info in page.get_image_info())

def extract_pages(file_bytes: bytes, filename: str, force_ocr: bool = False,
                  cached: dict | None = None, on_page: Callable | None = None) -> list[dict]:
    settings = get_settings()
    cached = cached or {}
    is_pdf = filename.lower().endswith(".pdf")
    is_text = filename.lower().endswith(".txt")
    doc = fitz.open(stream=file_bytes, filetype="pdf") if is_pdf else None
    try:
        if doc and doc.needs_pass:
            raise ValueError("Password-protected PDFs must be unlocked before upload")
        count = len(doc) if doc is not None else 1
        if count > settings.max_document_pages:
            raise ValueError(f"Maximum {settings.max_document_pages} pages per document")
        pages = []
        batch_text = {}
        batch_disabled = False
        remaining = saved_retry_seconds(cached.values(), settings.vision_provider, settings.vision_model)
        deferred = RateLimitDeferred(remaining) if remaining > 0 else None
        for index in range(count):
            number = index + 1
            previous = cached.get(str(number))
            if (previous and previous.get("status") in {"ok", "blank"} and not previous.get("warning")
                    and (previous.get('method') != 'native' or previous.get('layout_version') == NATIVE_LAYOUT_VERSION)):
                result = previous
            else:
                native = clean_text(native_page_text(doc[index])) if doc is not None else ""
                result = {"page": number, "text": "", "method": "native", "status": "ok", 'layout_version': NATIVE_LAYOUT_VERSION}
                try:
                    if is_text:
                        result["text"] = clean_text(file_bytes.decode("utf-8-sig"))
                    elif doc is None or needs_ocr(doc[index], native, force_ocr):
                        result["method"] = "vision"
                        result.update(ocr_provider=settings.vision_provider, ocr_model=settings.vision_model)
                        if deferred is not None and number not in batch_text:
                            raise deferred
                        image = doc[index].get_pixmap(dpi=settings.ocr_render_dpi).tobytes("png") if doc is not None else file_bytes
                        def waiting(seconds):
                            if on_page:
                                on_page({**result, "status": "waiting", "retry_at": RateLimitDeferred(seconds).retry_at}, pages, count)
                        with observe_waits(waiting):
                            # Pair adjacent uncached scans; native/cached pages never consume OCR quota.
                            if (number not in batch_text and not batch_disabled and doc is not None
                                    and settings.vision_provider == "gemini" and settings.gemini_ocr_batch_pages == 2
                                    and index + 1 < count):
                                next_saved = cached.get(str(number + 1), {})
                                next_page = doc[index + 1]
                                next_native = clean_text(native_page_text(next_page))
                                if (not (next_saved.get("status") in {"ok", "blank"} and not next_saved.get("warning"))
                                        and needs_ocr(next_page, next_native, force_ocr)):
                                    try:
                                        batch_text.update(transcribe_page_batch({number: image, number + 1:
                                            next_page.get_pixmap(dpi=settings.ocr_render_dpi).tobytes("png")}))
                                    except IncompleteOutputError:
                                        # Dense pages that do not fit still use the proven crop recovery.
                                        batch_disabled = True
                                        logger.info("OCR batch incomplete; using individual pages for this document")
                            text = batch_text.pop(number, None)
                            if text is None:
                                text = transcribe_image(image)
                            elif "[illegible]" in text.lower():
                                try:
                                    refined = transcribe_image(image)
                                    if len(refined) >= len(text) * .8 and refined.lower().count("[illegible]") < text.lower().count("[illegible]"):
                                        text = refined
                                except Exception as exc:
                                    logger.warning("Batch page refinement unavailable: %s", type(exc).__name__)
                                    if isinstance(exc, RateLimitDeferred):
                                        deferred = exc
                            result["text"] = clean_text(text)
                        if result["text"].strip() == "[blank]":
                            result.update(text="", status="blank")
                        elif not usable_text(result["text"]):
                            result.update(text="", status="failed", error="This page could not be read clearly.")
                    else:
                        result["text"] = native
                    if "[illegible]" in result["text"]:
                        result["warning"] = "Some handwriting was illegible; check the original page."
                except Exception as exc:
                    if isinstance(exc, ProviderConfigurationError):
                        raise
                    logger.warning("Page %d extraction failed: %s", number, type(exc).__name__)
                    result.update(text="", status="failed", error=public_error(exc))
                    if isinstance(exc, RateLimitDeferred):
                        deferred = exc
                        result.update(status="pending", retry_at=exc.retry_at, error_code="rate_limit",
                                      quota_kind=exc.quota_kind, quota_limit=exc.quota_limit)
                    # A failed retry must not destroy readable text from a previous pass.
                    if previous and previous.get("status") == "ok" and usable_text(previous.get("text", "")):
                        result = {**previous, "error": result["error"],
                                  "warning": previous.get("warning") or "Retained the previous transcription after a retry failed.",
                                  **({"retry_at": exc.retry_at, "error_code": "rate_limit",
                                      "ocr_provider": settings.vision_provider, "ocr_model": settings.vision_model}
                                     if isinstance(exc, RateLimitDeferred) else {})}
            pages.append(result)
            if on_page:
                on_page(result, pages, count)
        return pages
    finally:
        if doc is not None:
            doc.close()

def _storage():
    return get_supabase().storage.from_(get_settings().storage_bucket)

def _store_pages(path: str, pages: list[dict]):
    payload = json.dumps({str(p["page"]): p for p in pages}, ensure_ascii=False).encode()
    _storage().upload(path, payload, {"content-type": "application/json", "upsert": "true"})

def process_document(document_id: str) -> None:
    # Bound simultaneous OCR/model work within the documented single-worker server.
    with _processing_slots:
        _process_document(document_id)

def _process_document(document_id: str) -> None:
    """One persisted job. Original and successful page OCR survive backend restarts."""
    started = time.perf_counter()
    sb = get_supabase()
    rows = sb.table("documents").select("*").eq("id", document_id).execute().data
    if not rows:
        return
    document = rows[0]
    report = document.get("ingestion") or {}
    pages_path = f"{document['user_id']}/{document_id}/pages.json"
    sb.table("documents").update({"status": "processing"}).eq("id", document_id).execute()
    try:
        file_bytes = _storage().download(document["file_path"])
        try:
            cached = json.loads(_storage().download(pages_path))
        except Exception:
            cached = {}
        cache_state = dict(cached)
        cache_dirty = False
        # Model initialization overlaps the otherwise idle network OCR time.
        model_ready = warm_models()
        def progress(page, pages, total):
            nonlocal cache_dirty
            if page["status"] == "waiting":
                logger.info("Page %s waiting for provider quota until %s", page["page"], page["retry_at"])
                report["waiting"] = {"page": page["page"], "retry_at": page["retry_at"]}
                sb.table("documents").update({"ingestion": report, "page_count": total}).eq("id", document_id).execute()
                return
            was_waiting = report.pop("waiting", None)
            logger.info("Page %s/%s: %s", page["page"], total, page["status"])
            # Failed pages can be retried; successful pages never need another OCR call.
            changed = cache_state.get(str(page["page"])) != page
            if changed:
                cache_state[str(page["page"])] = page
                cache_dirty = True
            # One quota pause may defer dozens of pages. Persist them together at
            # the end instead of making two storage/database calls per skipped page.
            if page["status"] == "pending" and len(pages) != total:
                return
            if cache_dirty:
                _store_pages(pages_path, list(cache_state.values()))
                cache_dirty = False
            # Reusing a cached page needs no network roundtrip. Flush progress at the end.
            if not changed and not was_waiting and len(pages) != total:
                return
            report["pages"] = [{k: v for k, v in p.items() if k != "text"} for p in pages]
            completed = sum(p["status"] != "pending" for p in pages)
            sb.table("documents").update({"processed_pages": completed, "page_count": total, "ingestion": report}).eq("id", document_id).execute()
        extraction_started = time.perf_counter()
        pages = extract_pages(file_bytes, document["name"], report.get("ocr_mode") == "force", cached, progress)
        timings = {"extraction_and_progress": round(time.perf_counter() - extraction_started, 3)}
        indexing_started = time.perf_counter()
        model_ready.result()
        rows = []
        for page in pages:
            if page["status"] != "ok":
                continue
            for piece in chunk_text(page["text"]):
                rows.append({"id": str(uuid.uuid4()), "document_id": document_id, "content": piece,
                             "page_number": page["page"], "chunk_index": len(rows),
                             "metadata": {"document_name": document["name"], "subject": document.get("subject"),
                                          "extraction_method": page["method"], "embedding_model": get_settings().embedding_model,
                                          "embedding_version": EMBEDDING_VERSION}})
        if not rows:
            if any(p["status"] == "pending" for p in pages):
                raise RateLimitDeferred(saved_retry_seconds(pages))
            raise ValueError("No readable text was extracted. Check failed pages, then retry or upload a clearer scan.")
        timings["model_wait_and_chunking"] = round(time.perf_counter() - indexing_started, 3)
        embedding_started = time.perf_counter()
        batch_size = get_settings().embedding_batch_size
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset + batch_size]
            vectors = embed_texts([r["content"] for r in batch])
            if len(vectors) != len(batch):
                raise ValueError("Embedding count did not match the extracted passages. Retry indexing.")
            for row, vector in zip(batch, vectors):
                row["embedding"] = vector
        timings["embedding"] = round(time.perf_counter() - embedding_started, 3)
        storage_started = time.perf_counter()
        # While processing, the document is not eligible for retrieval.
        sb.table("document_chunks").delete().eq("document_id", document_id).execute()
        for offset in range(0, len(rows), 100):
            sb.table("document_chunks").insert(rows[offset:offset + 100]).execute()
        timings["chunk_storage"] = round(time.perf_counter() - storage_started, 3)
        report["stage_seconds"] = timings
        logger.info("Ingestion stage seconds: %s", timings)
        report.update(pages=[{k: v for k, v in p.items() if k != "text"} for p in pages],
                      embedding_model=get_settings().embedding_model, embedding_version=EMBEDDING_VERSION)
        report["processing_seconds"] = round(time.perf_counter() - started, 2)
        report["reused_pages"] = sum(p == cached.get(str(p["page"])) for p in pages)
        report.pop("error", None)
        status = "partial" if any(p["status"] in {"failed", "pending"} or p.get("warning") for p in pages) else "ready"
        sb.table("documents").update({"status": status, "ingestion": report, "chunk_count": len(rows)}).eq("id", document_id).execute()
    except Exception as exc:
        logger.exception("Document processing failed: %s", document_id)
        report["error"] = str(exc) if isinstance(exc, ValueError) else public_error(exc)
        sb.table("documents").update({"status": "failed", "ingestion": report}).eq("id", document_id).execute()
    finally:
        from app.services.retrieval_service import invalidate_bm25_cache
        invalidate_bm25_cache()
