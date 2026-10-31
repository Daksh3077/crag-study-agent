"""Integration through the real API and ingestion service with in-memory persistence."""
import copy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4
from fastapi.testclient import TestClient

from app.main import app
from app.core.auth import current_user
from app.services import document_processor
from test_pipeline import WordTokenizer

class MemoryQuery:
    def __init__(self, db, name):
        self.db, self.name = db, name
        self.filters, self.operation, self.payload = [], "select", None
    def select(self, *args, **kwargs): return self
    def eq(self, key, value):
        self.filters.append(lambda r: r.get(key) == value)
        return self
    def in_(self, key, values):
        self.filters.append(lambda r: r.get(key) in values)
        return self
    def order(self, *args, **kwargs): return self
    def limit(self, *args): return self
    def insert(self, payload):
        self.operation, self.payload = "insert", payload
        return self
    def update(self, payload):
        self.operation, self.payload = "update", payload
        return self
    def delete(self):
        self.operation = "delete"
        return self
    def execute(self):
        table = self.db.rows.setdefault(self.name, [])
        matches = [r for r in table if all(f(r) for f in self.filters)]
        if self.operation == "insert":
            matches = copy.deepcopy(self.payload if isinstance(self.payload, list) else [self.payload])
            if len({frozenset(row) for row in matches}) > 1:
                raise ValueError("PostgREST bulk inserts require matching object keys")
            for row in matches:
                row.setdefault("created_at", datetime.now(timezone.utc).isoformat())
            table.extend(matches)
        elif self.operation == "update":
            for row in matches: row.update(copy.deepcopy(self.payload))
        elif self.operation == "delete":
            self.db.rows[self.name] = [r for r in table if r not in matches]
        return SimpleNamespace(data=copy.deepcopy(matches), count=len(matches))

class MemoryDB:
    def __init__(self):
        self.rows, self.files = {}, {}
        self.storage = self
    def table(self, name): return MemoryQuery(self, name)
    def from_(self, bucket): return self
    def upload(self, path, data, options): self.files[path] = data
    def download(self, path): return self.files[path]
    def remove(self, paths):
        for path in paths: self.files.pop(path, None)
    def create_signed_url(self, path, expiry):
        if path not in self.files: raise FileNotFoundError(path)
        return {"signedURL": "https://fixture.invalid/source"}

class DocumentWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.db, self.owner = MemoryDB(), str(uuid4())
        app.dependency_overrides[current_user] = lambda: self.owner
        self.addCleanup(app.dependency_overrides.clear)
        for target in ("app.api.documents.get_supabase", "app.core.auth.get_supabase",
                       "app.api.chat.get_supabase", "app.services.document_processor.get_supabase"):
            mock = patch(target, return_value=self.db)
            mock.start()
            self.addCleanup(mock.stop)
        tokenizer = patch("app.services.chunking.get_tokenizer", return_value=WordTokenizer())
        tokenizer.start()
        self.addCleanup(tokenizer.stop)
        embeddings = patch.object(document_processor, "embed_texts", side_effect=lambda texts: [[0.1] * 384 for _ in texts])
        embeddings.start()
        self.addCleanup(embeddings.stop)
        warmup = patch.object(document_processor, "warm_models")
        warmup.start()
        self.addCleanup(warmup.stop)
        self.client = TestClient(app)

    def upload(self):
        return self.client.post("/api/documents/upload",
                                files={"file": ("notes.txt", b"Spring force F = k x. k = 125 N/m, x = 0.08 m, F = 10 N.", "text/plain")})

    def test_upload_index_chat_history_and_delete(self):
        response = self.upload()
        self.assertEqual(response.status_code, 202)
        doc_id = response.json()["id"]
        doc = self.client.get("/api/documents/" + doc_id).json()
        self.assertEqual(doc["status"], "ready")
        self.assertEqual((doc["processed_pages"], doc["page_count"], doc["chunk_count"]), (1, 1, 1))
        self.assertIn(doc["file_path"], self.db.files)
        source = self.db.rows["document_chunks"][0]
        state = {"generated_answer": "The force is 10 N. [S1]", "validation_status": "supported",
                 "groundedness_confidence": .9, "cited_source_ids": ["S1"],
                 "evidence_chunks": [dict(source, citation_id="S1")]}
        with patch("app.api.chat.run_agent", return_value=state) as agent:
            answer = self.client.post("/api/chat", json={"question": "What force?", "document_ids": [doc_id]})
        self.assertEqual(answer.status_code, 200)
        self.assertEqual(agent.call_args.args[1], [doc_id])
        self.assertEqual(answer.json()["sources"][0]["page"], 1)
        self.assertEqual(answer.json()["pipeline"][0]["name"], "Documents searched")
        self.assertIn("notes.txt", answer.json()["pipeline"][0]["detail"])
        self.assertEqual(self.db.rows["retrieval_logs"][0]["metadata"]["document_ids"], [doc_id])
        history = self.client.get("/api/chat/conversations/" + answer.json()["conversation_id"]).json()
        self.assertEqual([m["role"] for m in history], ["user", "assistant"])
        self.assertEqual(history[1]["metadata"]["sources"][0]["document_id"], doc_id)
        self.assertLess(history[0]["created_at"], history[1]["created_at"])
        self.assertEqual(self.client.get("/api/documents/" + doc_id + "/source").status_code, 200)
        self.assertEqual(self.client.delete("/api/documents/" + doc_id).status_code, 200)
        self.assertEqual(self.db.files, {})

    def test_uploaded_document_is_private(self):
        doc_id = self.upload().json()["id"]
        self.owner = str(uuid4())
        self.assertEqual(self.client.get("/api/documents").json(), [])
        for method, path in [("get", ""), ("get", "/source"), ("post", "/retry"), ("delete", "")]:
            response = getattr(self.client, method)("/api/documents/" + doc_id + path)
            self.assertEqual(response.status_code, 404)
        self.assertEqual(len(self.db.files), 2)

    def test_partial_retry_uses_successful_page_cache(self):
        pages = [{"page":1,"text":"Review on day 3.","status":"ok","method":"vision"},
                 {"page":2,"text":"","status":"failed","method":"vision","error":"Unreadable"}]
        def initial_extract(data, name, force, cached, progress):
            for i, page in enumerate(pages): progress(page, pages[:i+1], 2)
            return pages
        with patch.object(document_processor, "extract_pages", side_effect=initial_extract):
            doc_id = self.upload().json()["id"]
        self.assertEqual(self.client.get("/api/documents/" + doc_id).json()["status"], "partial")
        self.assertEqual(len(self.db.rows["document_chunks"]), 1)
        def retry_extract(data, name, force, cached, progress):
            self.assertEqual(cached["1"]["text"], "Review on day 3.")
            self.assertEqual(cached["2"]["status"], "failed")
            recovered = [cached["1"], {"page":2,"text":"Then review on day 8.","status":"ok","method":"vision"}]
            for i, page in enumerate(recovered): progress(page, recovered[:i+1], 2)
            return recovered
        with patch.object(document_processor, "extract_pages", side_effect=retry_extract):
            self.assertEqual(self.client.post("/api/documents/" + doc_id + "/retry").status_code, 202)
        final = self.client.get("/api/documents/" + doc_id).json()
        self.assertEqual(final["status"], "ready")
        self.assertEqual(final["chunk_count"], 2)
        self.assertEqual(len(self.db.rows["document_chunks"]), 2)

    def test_all_failed_pages_are_not_indexed(self):
        failed = [{"page":1,"text":"","status":"failed","method":"vision","error":"Unreadable"}]
        def extract(data, name, force, cached, progress):
            progress(failed[0], failed, 1)
            return failed
        with patch.object(document_processor, "extract_pages", side_effect=extract):
            doc_id = self.upload().json()["id"]
        doc = self.client.get("/api/documents/" + doc_id).json()
        self.assertEqual(doc["status"], "failed")
        self.assertEqual(self.db.rows.get("document_chunks", []), [])
        self.assertEqual(self.client.post("/api/chat", json={"question":"Guess an answer"}).status_code, 409)

    def test_retry_respects_saved_quota_without_starting_job(self):
        doc_id = self.upload().json()["id"]
        doc = self.db.rows["documents"][0]
        doc.update(status="partial", ingestion={"pages": [{"page": 2, "status": "pending",
                   "ocr_provider": "gemini", "ocr_model": "gemini-3.5-flash-lite",
                   "retry_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()}]})
        with patch("app.api.documents.process_document") as process:
            response = self.client.post("/api/documents/" + doc_id + "/retry")
        self.assertEqual(response.status_code, 429)
        self.assertGreater(int(response.headers["Retry-After"]), 3500)
        self.assertEqual(doc["status"], "partial")
        process.assert_not_called()

    def test_all_quota_pending_is_not_reported_as_unreadable(self):
        from app.services.provider_limits import RateLimitDeferred
        deferred = RateLimitDeferred(3600)
        pending = [{"page": i, "text": "", "status": "pending", "method": "vision",
                    "retry_at": deferred.retry_at, "error_code": "rate_limit"} for i in range(1, 53)]
        def extract(data, name, force, cached, progress):
            for i, page in enumerate(pending):
                progress(page, pending[:i + 1], len(pending))
            return pending
        with patch.object(document_processor, "extract_pages", side_effect=extract), \
             patch.object(document_processor, "_store_pages", wraps=document_processor._store_pages) as store:
            doc_id = self.upload().json()["id"]
        doc = self.client.get("/api/documents/" + doc_id).json()
        self.assertIn("quota", doc["ingestion"]["error"].lower())
        self.assertNotIn("clearer scan", doc["ingestion"]["error"])
        self.assertEqual(doc["processed_pages"], 0)
        self.assertEqual(len(doc["ingestion"]["pages"]), 52)
        self.assertEqual(store.call_count, 1)
        self.assertEqual(self.db.rows.get("document_chunks", []), [])
