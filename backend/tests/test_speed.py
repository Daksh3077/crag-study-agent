"""Performance regression tests retain complete evidence and unchanged page caches."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
from PIL import Image
from app.core.config import Settings
from app.services import summary_cache, vision_ocr, provider_limits, embeddings
from app.agents import validator, summarizer
import test_document_workflow as workflows

class SpeedTests(unittest.TestCase):
    def test_concurrent_ingestions_share_one_model_warmup(self):
        future=MagicMock()
        future.done.return_value=False
        with patch.object(embeddings,"_warm_future",None), patch.object(embeddings._warm_executor,"submit",return_value=future) as submit:
            first=embeddings.warm_models()
            second=embeddings.warm_models()
        self.assertIs(first,second)
        self.assertEqual(submit.call_count,1)

    def test_failed_model_warmup_can_be_retried(self):
        failed=MagicMock()
        failed.done.return_value=True
        failed.exception.return_value=RuntimeError("Temporary download error")
        with patch.object(embeddings,"_warm_future",failed), patch.object(embeddings._warm_executor,"submit") as submit:
            result=embeddings.warm_models()
        self.assertIs(result,submit.return_value)
        self.assertEqual(submit.call_count,1)

    def test_positive_quota_does_not_wait_for_full_refill(self):
        with patch.object(provider_limits,"cooldown") as wait:
            provider_limits.observe_headers("vision", {"x-ratelimit-remaining-tokens":"859", "x-ratelimit-reset-tokens":"56.545s"}, 2300)
        wait.assert_not_called()

    def test_normal_stop_without_sentinel_is_not_reprocessed(self):
        response=SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop",message=SimpleNamespace(content="Readable notes, including the final line."))])
        with patch.object(vision_ocr,"completion_request",return_value=response) as call:
            text=vision_ocr._request(Image.new("RGB",(100,100)))
        self.assertIn("final line",text)
        self.assertEqual(call.call_count,1)

    def test_real_output_truncation_still_requires_recovery(self):
        response=SimpleNamespace(choices=[SimpleNamespace(finish_reason="length",message=SimpleNamespace(content="Partial text [[END]]"))])
        with patch.object(vision_ocr,"completion_request",return_value=response):
            with self.assertRaises(vision_ocr.IncompleteOutputError):
                vision_ocr._request(Image.new("RGB",(100,100)))

    def test_shared_evidence_packs_twenty_checks_into_four(self):
        config = Settings(_env_file=None, validation_batch_tokens=1000)
        chunks = [{"citation_id":f"S{i}", "content":"source " * 190} for i in (1,2)]
        paragraphs = ["claim " * 115 + f"marker{i} [S{i%2+1}]" for i in range(20)]
        with patch.object(validator,"get_settings",return_value=config), patch.object(validator,"token_count",side_effect=lambda t:len(t.split())):
            batches = validator.validation_batches("\n\n".join(paragraphs),chunks)
        self.assertEqual(len(batches),4)
        text = "\n\n".join(b[0] for b in batches)
        self.assertTrue(all(f"marker{i} " in text for i in range(20)))
        self.assertTrue(all(len((a+" "+b).split()) <= 1000 for a,b in batches))

    def test_summary_cache_scope_model_content_and_expiry(self):
        base = [["doc"],"groq","model","system","source"]
        key = summary_cache.section_key(*base)
        summary_cache.put(key, "Saved")
        self.assertEqual(summary_cache.get(key), "Saved")
        for index,new in [(0,["another-doc"]),(2,"another-model"),(4,"changed-source")]:
            values=base.copy()
            values[index]=new
            self.assertNotEqual(summary_cache.section_key(*values),key)
        with patch.object(summary_cache.time,"monotonic",return_value=10**15):
            self.assertIsNone(summary_cache.get(key))

    def test_small_summary_uses_original_sources_in_one_generation(self):
        summary_cache._cache.clear()
        self.addCleanup(summary_cache._cache.clear)
        chunks = [{"id":"a","document_id":"testdoc","chunk_index":0,"content":"Evidence", "page_number":1}]
        config=Settings(_env_file=None)
        with patch.object(summarizer,"get_settings",return_value=config), patch.object(summarizer,"_all_candidate_chunks",return_value=chunks), patch.object(summarizer,"token_count",return_value=10), patch.object(summarizer,"chat",return_value="Supported notes [S1]") as chat:
            for _ in range(2):
                summarizer.full_document_summary_node({"document_ids":["testdoc"],"mode":"detailed_summary","original_question":"Summarize"})
        self.assertEqual(chat.call_count,2)  # One direct synthesis per request.
        self.assertIn('Evidence', chat.call_args.args[1])
        self.assertIn('[S1]', chat.call_args.args[1])

class CachedProgressTests(workflows.DocumentWorkflowTests):
    # Run only this additional workflow test; base regressions live in their original module.
    test_upload_index_chat_history_and_delete = None
    test_uploaded_document_is_private = None
    test_partial_retry_uses_successful_page_cache = None
    test_all_failed_pages_are_not_indexed = None

    def test_reused_pages_do_not_reupload_ocr_cache(self):
        doc_id=self.upload().json()["id"]
        self.db.rows["documents"][0]["status"]="partial"
        with patch("app.services.document_processor._store_pages", wraps=__import__("app.services.document_processor",fromlist=["_store_pages"])._store_pages) as store:
            response=self.client.post("/api/documents/"+doc_id+"/retry")
        self.assertEqual(response.status_code,202)
        store.assert_not_called()
        self.assertEqual(self.db.rows["documents"][0]["status"],"ready")
