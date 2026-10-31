"""Gemini OCR routing and quota contracts, using a local HTTP transport only."""
import io
import json
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import openai
import pymupdf
from PIL import Image

from app.core.config import Settings
from app.api.documents import document_for_display
from app.services import llm_client, provider_limits, vision_ocr, document_processor


class GeminiOCRTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(_env_file=None, gemini_api_key="fixture-gemini", groq_api_key="fixture-groq")
        provider_limits._cooldowns.clear()
        provider_limits._pacing_deadlines.clear()
        llm_client.get_client.cache_clear()
        self.addCleanup(provider_limits._cooldowns.clear)
        self.addCleanup(provider_limits._pacing_deadlines.clear)
        self.addCleanup(llm_client.get_client.cache_clear)
        for module in (llm_client, vision_ocr, document_processor):
            mock = patch.object(module, "get_settings", return_value=self.settings)
            mock.start()
            self.addCleanup(mock.stop)

    def test_ocr_request_serializes_for_google_with_no_groq_options(self):
        def respond(request):
            self.assertEqual(request.url.host, "generativelanguage.googleapis.com")
            self.assertEqual(request.url.path, "/v1beta/openai/chat/completions")
            body = json.loads(request.content)
            self.assertEqual(body["model"], "gemini-3.5-flash-lite")
            self.assertEqual(body["reasoning_effort"], "low")
            self.assertEqual(body["max_tokens"], 4096)
            self.assertNotIn("include_reasoning", body)
            self.assertTrue(body["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
            return httpx.Response(200, json={"id": "fixture", "object": "chat.completion", "created": 0,
                "model": body["model"], "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Force F = ma."}}],
                "usage": {"prompt_tokens": 200, "completion_tokens": 10, "total_tokens": 210}})
        client = openai.OpenAI(api_key="fixture", base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                               http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        self.addCleanup(client.close)
        with patch.object(llm_client, "get_client", return_value=client) as get_client:
            self.assertEqual(vision_ocr._request(Image.new("RGB", (100, 100))), "Force F = ma.")
        get_client.assert_called_once_with("gemini")
        self.assertIn("gemini:gemini-3.5-flash-lite", provider_limits._pacing_deadlines)
        self.assertNotIn("gemini:gemini-3.5-flash-lite", provider_limits._cooldowns)
        self.assertNotIn("groq:openai/gpt-oss-120b", provider_limits._cooldowns)

    def test_clients_have_separate_keys_and_endpoints(self):
        with patch("openai.OpenAI") as factory:
            llm_client.get_client("gemini")
            self.assertEqual(factory.call_args.kwargs["api_key"], "fixture-gemini")
            self.assertIn("googleapis.com", factory.call_args.kwargs["base_url"])
            llm_client.get_client("groq")
            self.assertEqual(factory.call_args.kwargs["api_key"], "fixture-groq")
            self.assertIn("groq.com", factory.call_args.kwargs["base_url"])

    def run_timed_requests(self, duration, limited=False):
        clock, starts = [100.0], []
        response = SimpleNamespace(usage=SimpleNamespace(total_tokens=200))
        def request(**kwargs):
            starts.append(clock[0])
            if limited and len(starts) == 1:
                error = RuntimeError("Rate limited")
                error.status_code = 429
                error.response = SimpleNamespace(headers={"retry-after": "20"})
                raise error
            clock[0] += duration
            return SimpleNamespace(headers={}, parse=lambda: response)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(with_raw_response=SimpleNamespace(create=request))))
        waits = []
        with patch.object(provider_limits.time, "monotonic", side_effect=lambda: clock[0]), \
             patch.object(provider_limits.time, "sleep", side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)), \
             patch.object(llm_client, "get_client", return_value=client), \
             provider_limits.observe_waits(waits.append):
            for _ in range(1 if limited else 3):
                llm_client.completion_request("gemini", {"model": "gemini-3.5-flash-lite"}, ocr=True)
        return starts, waits

    def test_slow_ocr_has_no_added_spacing_or_quota_notice(self):
        starts, waits = self.run_timed_requests(8)
        self.assertEqual(starts, [100, 108, 116])
        self.assertEqual(waits, [])

    def test_fast_ocr_remains_within_configured_rpm_without_quota_notice(self):
        starts, waits = self.run_timed_requests(2)
        self.assertAlmostEqual(starts[1] - starts[0], 6.1)
        self.assertAlmostEqual(starts[2] - starts[1], 6.1)
        self.assertEqual(waits, [])

    def test_gemini_actual_rate_limit_still_waits_and_notifies(self):
        starts, waits = self.run_timed_requests(2, limited=True)
        self.assertGreaterEqual(starts[1] - starts[0], 20)
        self.assertEqual(len(waits), 1)
        self.assertGreaterEqual(waits[0], 20)

    def test_answers_still_use_groq(self):
        response = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="Answer [S1]"))])
        with patch.object(llm_client, "get_client"), patch.object(llm_client, "completion_request", return_value=response) as complete:
            self.assertEqual(llm_client.chat("Use the notes", "What is force?"), "Answer [S1]")
        self.assertEqual(complete.call_args.args[0], "groq")
        self.assertEqual(complete.call_args.args[1]["model"], "openai/gpt-oss-120b")

    def test_existing_25_flash_configuration_disables_thinking(self):
        self.settings.vision_model = "gemini-2.5-flash"
        response = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="Transcription"))])
        with patch.object(vision_ocr, "completion_request", return_value=response) as complete:
            vision_ocr._request(Image.new("RGB", (100, 100)))
        self.assertEqual(complete.call_args.args[1]["extra_body"], {"reasoning_effort": "none"})

    def test_missing_gemini_key_stops_once_with_setup_message(self):
        self.settings.gemini_api_key = ""
        data = io.BytesIO()
        Image.new("RGB", (100, 100)).save(data, format="PNG")
        with patch("openai.OpenAI") as factory:
            with self.assertRaisesRegex(llm_client.ProviderConfigurationError, "GEMINI_API_KEY"):
                document_processor.extract_pages(data.getvalue(), "scan.png")
        factory.assert_not_called()

    def test_gemini_retry_info_is_honored(self):
        error = RuntimeError("Resource exhausted")
        error.body = {"error": {"details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "123.5s"}]}}
        self.assertEqual(provider_limits.retry_delay(error), 124.5)
        self.assertEqual(provider_limits.retry_delay(RuntimeError("Please retry in 42.5s.")), 43.5)
        self.assertGreaterEqual(provider_limits.retry_delay(RuntimeError("GenerateRequestsPerDayPerProjectPerModel-FreeTier")), 86400)

    def test_old_groq_deadline_does_not_disable_gemini_resume(self):
        deadline = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        page = {"page": 2, "status": "pending", "retry_at": deadline}
        self.assertGreater(provider_limits.saved_retry_seconds([page], "groq", "qwen/qwen3.8-27b"), 0)
        self.assertEqual(provider_limits.saved_retry_seconds([page], "gemini", "gemini-2.5-flash"), 0)
        with patch("app.api.documents.get_settings", return_value=self.settings):
            shown = document_for_display({"ingestion": {"pages": [page]}})
        self.assertNotIn("retry_at", shown["ingestion"]["pages"][0])
        self.assertEqual(page["retry_at"], deadline)

    def test_new_gemini_deadline_is_retained(self):
        page = {"status": "pending", "ocr_provider": "gemini", "ocr_model": "gemini-2.5-flash",
                "retry_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()}
        self.assertGreater(provider_limits.saved_retry_seconds([page], "gemini", "gemini-2.5-flash"), 0)
        self.assertEqual(provider_limits.saved_retry_seconds([page], "gemini", "gemini-2.5-flash-lite"), 0)

    def test_switch_resumes_pending_pages_and_keeps_completed_groq_ocr(self):
        pdf = pymupdf.open()
        for _ in range(2):
            pdf.new_page()
        data = pdf.tobytes()
        pdf.close()
        cached = {"1": {"page": 1, "status": "ok", "method": "vision", "text": "Original saved notes"},
                  "2": {"page": 2, "status": "pending", "method": "vision", "text": "",
                        "retry_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()}}
        with patch.object(document_processor, "transcribe_image", return_value="Newly transcribed notes") as ocr:
            pages = document_processor.extract_pages(data, "notes.pdf", True, cached)
        ocr.assert_called_once()
        self.assertEqual(pages[0], cached["1"])
        self.assertEqual(pages[1]["status"], "ok")
        self.assertEqual(pages[1]["ocr_provider"], "gemini")
        self.assertNotIn("retry_at", pages[1])
