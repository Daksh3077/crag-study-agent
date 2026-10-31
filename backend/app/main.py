"""FastAPI service. Run one worker; page jobs are persisted and retryable."""
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from app.api import chat, documents, analytics
from app.core.config import get_settings
from app.core.database import get_supabase
from app.services.embeddings import warm_models
from app.services.document_processor import process_document

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def recover_interrupted():
    sb = get_supabase()
    docs = sb.table("documents").select("id,status,ingestion").in_("status", ["queued", "processing"]).execute().data or []
    recovered = []
    for doc in docs:
        report = doc.get("ingestion") or {}
        report.pop('error', None)
        result = sb.table("documents").update({"status": "queued", "ingestion": report}).eq("id", doc["id"]).eq('status', doc['status']).execute()
        if result.data:
            recovered.append(doc['id'])
    return recovered

async def resume_documents(document_ids):
    # Single worker only. Database rows and saved pages survive process death.
    for document_id in document_ids:
        try:
            await run_in_threadpool(process_document, document_id)
        except Exception:
            logger.exception('Restart recovery failed for document %s', document_id)
            try:
                get_supabase().table('documents').update({'status': 'failed'}).eq('id', document_id).in_('status', ['queued', 'processing']).execute()
            except Exception:
                logger.exception('Could not save recovery failure; queued row will retry next startup')

@asynccontextmanager
async def lifespan(app):
    def warmed(future):
        if future.cancelled() or future.exception() is not None:
            logger.warning("Local model warmup failed; the next indexing/query request can retry")
        else:
            logger.info("Local study models are warm")
    app.state.model_warmup = warm_models()
    app.state.model_warmup.add_done_callback(warmed)
    app.state.database_initialized = False
    interrupted = []
    try:
        interrupted = await run_in_threadpool(recover_interrupted)
        app.state.database_initialized = True
    except Exception:
        logger.exception("Database startup check failed; apply migrations and check configuration")
    recovery = asyncio.create_task(resume_documents(interrupted))
    try:
        yield
    finally:
        if not recovery.done():
            recovery.cancel()
        await asyncio.gather(recovery, return_exceptions=True)

app = FastAPI(title="CRAG Study Agent", version="0.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware,
    allow_origins=[s.strip() for s in get_settings().cors_origins.split(",") if s.strip()],
    allow_credentials=False, allow_methods=["GET", "POST", "DELETE"], allow_headers=["Authorization", "Content-Type"])
app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(analytics.router)

@app.get("/")
@app.get("/health")
def health():
    return {"status": "healthy", "service": "crag-study-agent", "version": "0.2.0"}

@app.get('/ready')
def ready():
    future = getattr(app.state, 'model_warmup', None)
    models_ready = bool(future and future.done() and not future.cancelled() and future.exception() is None)
    database_ready = bool(getattr(app.state, 'database_initialized', False))
    available = models_ready and database_ready
    return JSONResponse(status_code=200 if available else 503,
        content={'status': 'ready' if available else 'not_ready',
                 'models_loaded': models_ready, 'startup_database_check': database_ready})
