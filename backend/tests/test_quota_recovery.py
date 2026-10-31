"""Regression checks for quota handling, retained OCR, and grouped citations."""
import io
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch
import pymupdf
from PIL import Image
from app.core.config import Settings
from app.services import provider_limits as limits, llm_client, document_processor, vision_ocr, evidence
from app.agents import validator

class QuotaError(RuntimeError):
    status_code = 429
    def __init__(self, message="rate limit", headers=None):
        super().__init__(message)
        self.response = SimpleNamespace(headers=headers or {})

class RateLimitTests(unittest.TestCase):
    def setUp(self):
        limits._cooldowns.clear()
        self.addCleanup(limits._cooldowns.clear)

    def test_duration_and_daily_limit(self):
        self.assertAlmostEqual(limits.duration_seconds("2m59.56s"), 179.56)
        self.assertEqual(limits.duration_seconds("1h2m3s"), 3723)
        self.assertGreaterEqual(limits.retry_delay(QuotaError("tokens per day (TPD)")), 86400)

    def test_retry_after_is_not_capped_at_ninety_seconds(self):
        clock = [1000.]
        def sleep(seconds): clock[0] += seconds
        calls = []
        def request():
            calls.append(clock[0])
            if len(calls) == 1: raise QuotaError(headers={"retry-after":"150"})
            return "ok"
        with patch.object(limits.time, "monotonic", side_effect=lambda: clock[0]), patch.object(limits.time, "sleep", side_effect=sleep):
            self.assertEqual(llm_client.with_retry(request, key="vision", max_wait=200), "ok")
        self.assertGreaterEqual(calls[1] - calls[0], 150)

    def test_long_quota_exhaustion_stops_without_sleep(self):
        error = QuotaError(headers={"retry-after":"3600"})
        with patch.object(limits.time, "sleep") as sleep:
            with self.assertRaises(limits.RateLimitDeferred) as raised:
                llm_client.with_retry(lambda: (_ for _ in ()).throw(error), key="vision", max_wait=300)
            self.assertGreaterEqual(raised.exception.retry_after, 3600)
            sleep.assert_not_called()

    def test_exhausted_retry_budget_blocks_next_request_on_same_model(self):
        error = QuotaError(headers={"retry-after":"120"})
        with self.assertRaises(limits.RateLimitDeferred):
            llm_client.with_retry(lambda: (_ for _ in ()).throw(error), key="vision", retries=0)
        with patch.object(limits.time, "sleep"), patch("builtins.print") as next_request:
            with self.assertRaises(limits.RateLimitDeferred):
                llm_client.with_retry(next_request, key="vision", max_wait=1)
            next_request.assert_not_called()

    def test_success_headers_schedule_next_request(self):
        with patch.object(limits.time, "monotonic", return_value=100):
            limits.observe_headers("vision", {"x-ratelimit-remaining-tokens":"0", "x-ratelimit-reset-tokens":"40s"}, 2000)
            self.assertEqual(limits._cooldowns["vision"], 141)

class DeferredPageTests(unittest.TestCase):
    def setUp(self):
        # These exercise individual-page recovery; batch behavior has dedicated tests.
        settings = document_processor.get_settings().model_copy(update={"gemini_ocr_batch_pages": 1})
        self.settings_patch = patch.object(document_processor, "get_settings", return_value=settings)
        self.settings_patch.start()
        self.addCleanup(self.settings_patch.stop)

    def pdf(self):
        doc = pymupdf.open()
        for _ in range(4): doc.new_page()
        data = doc.tobytes()
        doc.close()
        return data

    def test_stop_later_ocr_calls_keep_cached_good_pages(self):
        cache = {"1":{"page":1,"text":"Good saved text","method":"vision","status":"ok"},
                 "4":{"page":4,"text":"Good last page","method":"vision","status":"ok"}}
        with patch.object(document_processor, "transcribe_image", side_effect=limits.RateLimitDeferred(3600)) as ocr:
            pages = document_processor.extract_pages(self.pdf(), "notes.pdf", True, cache)
        self.assertEqual(ocr.call_count, 1)
        self.assertEqual([p["status"] for p in pages], ["ok","pending","pending","ok"])
        self.assertEqual(pages[2]["text"], "")
        self.assertIn("retry_at", pages[2])

    def test_persisted_quota_survives_lost_in_memory_cooldown(self):
        cache = {"1": {"page": 1, "text": "Saved transcription", "method": "vision", "status": "ok"},
                 "2": {"page": 2, "status": "pending", "text": "", "method": "vision",
                       "ocr_provider": "gemini", "ocr_model": "gemini-3.5-flash-lite",
                       "retry_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()}}
        with patch.object(document_processor, "transcribe_image") as ocr:
            pages = document_processor.extract_pages(self.pdf(), "notes.pdf", True, cache)
        ocr.assert_not_called()
        self.assertEqual(pages[0], cache["1"])
        self.assertEqual([p["status"] for p in pages], ["ok", "pending", "pending", "pending"])

    def test_expired_quota_allows_ocr_again(self):
        cache = {"1": {"page": 1, "status": "pending", "text": "", "method": "vision",
                       "ocr_provider": "gemini", "ocr_model": "gemini-3.5-flash-lite",
                       "retry_at": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()}}
        with patch.object(document_processor, "transcribe_image", return_value="Recovered study notes") as ocr:
            pages = document_processor.extract_pages(self.pdf(), "notes.pdf", True, cache)
        self.assertEqual(ocr.call_count, 4)
        self.assertTrue(all(p["status"] == "ok" for p in pages))

    def test_invalid_saved_deadlines_do_not_block_processing(self):
        self.assertEqual(limits.saved_retry_seconds([
            {"status": "pending", "retry_at": value} for value in (None, "bad", "2026-01-01", 123)
        ]), 0)

    def test_failed_retry_preserves_unclear_but_readable_text(self):
        cache = {"1":{"page":1,"text":"A readable formula and [illegible] label","method":"vision","status":"ok","warning":"Unclear label"}}
        with patch.object(document_processor, "transcribe_image", side_effect=limits.RateLimitDeferred(3600)):
            pages = document_processor.extract_pages(self.pdf(), "notes.pdf", True, cache)
        self.assertEqual(pages[0]["text"], cache["1"]["text"])
        self.assertEqual(pages[0]["status"], "ok")

    def test_unclear_page_gets_bounded_close_up(self):
        stream = io.BytesIO()
        Image.new("RGB",(700,1000),"white").save(stream,format="PNG")
        with patch.object(vision_ocr, "_request", side_effect=["Upper words and [illegible]", "Upper words and", "readable label"]) as request:
            result = vision_ocr.transcribe_image(stream.getvalue())
        self.assertEqual(request.call_count, 3)
        self.assertNotIn("[illegible]", result)
        self.assertIn("Upper words", result)

class CitationTests(unittest.TestCase):
    def test_ranges_and_groups_preserve_all_evidence(self):
        self.assertEqual(evidence.cited_ids("Arrays [S2–S5], stacks [S5-S8], types [S1, S9]."),
                         ["S2","S3","S4","S5","S6","S7","S8","S1","S9"])
        self.assertEqual(evidence.cited_ids("[S3‑S4] [S4; S7]"), ["S3","S4","S7"])

    def test_unknown_range_member_is_not_silently_ignored(self):
        with self.assertRaises(ValueError):
            validator.validation_batches("Claim [S1-S3]", [{"citation_id":"S1","content":"Known"}])

    def test_validation_budget_includes_answer_not_just_evidence(self):
        config = Settings(_env_file=None, validation_batch_tokens=1000)
        chunks = [{"citation_id":f"S{i}","content":"source " * 300} for i in range(1,4)]
        answer = "\n\n".join("claim " * 150 + f"[S{i}]" for i in range(1,4))
        with patch.object(validator,"get_settings",return_value=config), patch.object(validator,"token_count",side_effect=lambda text:len(text.split())):
            batches = validator.validation_batches(answer,chunks)
        self.assertEqual(len(batches),2)
        self.assertTrue(all(len((text+source).split()) <= 1000 for text,source in batches))
        self.assertTrue(all(f"[S{i}]" in "\n".join(b[0] for b in batches) for i in range(1,4)))
