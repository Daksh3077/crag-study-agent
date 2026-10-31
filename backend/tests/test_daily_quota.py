"""Regression for a real Gemini 20-RPD error carrying a misleading 20-second retry."""
import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, patch
from app.services import provider_limits as limits, llm_client, document_processor
from app.core.config import Settings
from app.api.documents import document_for_display


def quota_error(kind="Day"):
    error = RuntimeError("Quota exceeded; retry in 20s")
    error.status_code = 429
    error.body = [{"error": {"details": [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{
            "quotaId": f"GenerateRequestsPer{kind}PerProjectPerModel-FreeTier", "quotaValue": "20"}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "20s"}]}}]
    return error


class DailyQuotaTests(unittest.TestCase):
    def setUp(self):
        limits._cooldowns.clear()
        self.addCleanup(limits._cooldowns.clear)

    def test_daily_error_stops_immediately_even_with_large_wait_budget(self):
        request = Mock(side_effect=quota_error())
        with patch.object(limits, "gemini_daily_reset_seconds", return_value=3600), patch.object(limits.time, "sleep") as sleep:
            with self.assertRaises(limits.RateLimitDeferred) as raised:
                llm_client.with_retry(request, key="gemini:fixture", retries=6, max_wait=5000)
        request.assert_called_once()
        sleep.assert_not_called()
        self.assertEqual(raised.exception.retry_after, 3601)
        self.assertEqual(raised.exception.quota_kind, "daily")
        self.assertIn("daily limit: 20", llm_client.public_error(raised.exception))

    def test_minute_error_keeps_short_retry(self):
        self.assertEqual(limits.retry_delay(quota_error("Minute")), 21)

    def test_daily_wins_when_minute_violation_is_also_present(self):
        error = quota_error()
        error.body[0]["error"]["details"][0]["violations"].append({"quotaId": "GenerateRequestsPerMinute", "quotaValue": "5"})
        self.assertEqual(limits.quota_details(error), {"quota_kind": "daily", "quota_limit": "20"})

    def test_pacific_reset_and_daylight_saving_transitions(self):
        for timestamp, seconds in [("2026-01-20T07:00:00+00:00", 3600), ("2026-07-20T06:00:00+00:00", 3600),
                                   ("2026-03-08T08:00:00+00:00", 23 * 3600), ("2026-11-01T07:00:00+00:00", 25 * 3600)]:
            with self.subTest(timestamp=timestamp):
                self.assertEqual(limits.gemini_daily_reset_seconds(datetime.fromisoformat(timestamp)), seconds)

    def test_switch_to_lite_reuses_saved_pages_and_ignores_flash_quota(self):
        import pymupdf
        config = Settings(_env_file=None)
        with pymupdf.open() as pdf:
            pdf.new_page()
            pdf.new_page()
            data = pdf.tobytes()
        deadline = limits.RateLimitDeferred(3600).retry_at
        pages = {"1": {"page": 1, "text": "Saved handwriting", "status": "ok", "method": "vision"},
                 "2": {"page": 2, "text": "", "status": "pending", "method": "vision", "retry_at": deadline,
                       "ocr_provider": "gemini", "ocr_model": "gemini-3.6-flash", "quota_kind": "daily"}}
        with patch.object(document_processor, "get_settings", return_value=config), patch.object(document_processor, "transcribe_image", return_value="New handwritten page") as ocr:
            result = document_processor.extract_pages(data, "scan.pdf", True, pages)
        ocr.assert_called_once()
        self.assertEqual(result[0], pages["1"])
        self.assertEqual(result[1]["ocr_model"], "gemini-3.5-flash-lite")
        self.assertEqual(result[1]["status"], "ok")
        with patch("app.api.documents.get_settings", return_value=config):
            shown = document_for_display({"ingestion": {"pages": list(pages.values())}})
        self.assertNotIn("retry_at", shown["ingestion"]["pages"][1])

    def test_daily_reason_is_persisted_for_every_deferred_page(self):
        import pymupdf
        with pymupdf.open() as pdf:
            pdf.new_page()
            pdf.new_page()
            data = pdf.tobytes()
        error = limits.RateLimitDeferred(3600, quota_kind="daily", quota_limit="20")
        with patch.object(document_processor, "transcribe_page_batch", side_effect=error):
            pages = document_processor.extract_pages(data, "scan.pdf", True)
        self.assertTrue(all(p["quota_kind"] == "daily" and p["status"] == "pending" for p in pages))
        self.assertTrue(all(p["text"] == "" for p in pages))
