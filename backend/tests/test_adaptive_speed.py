"""Speed changes must preserve page boundaries, source scope and honest validation."""
import io
import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pymupdf
from PIL import Image
from app.core.config import Settings
from app.agents import graph, generator, validator
from app.services import document_processor as processor, vision_ocr, retrieval_service, evidence
from app.services.llm_client import IncompleteOutputError
from app.services.provider_limits import RateLimitDeferred


def response(text, reason="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=reason, message=SimpleNamespace(content=text))])


class BatchOCRTests(unittest.TestCase):
    def setUp(self):
        self.config = Settings(_env_file=None, gemini_ocr_batch_pages=2)
        for module in (processor, vision_ocr):
            p = patch.object(module, "get_settings", return_value=self.config)
            p.start()
            self.addCleanup(p.stop)

    def image(self):
        data = io.BytesIO()
        Image.new("RGB", (700, 1000), "white").save(data, format="PNG")
        return data.getvalue()

    def pdf(self, count=4):
        with pymupdf.open() as doc:
            for _ in range(count):
                doc.new_page()
            return doc.tobytes()

    def test_exact_page_ids_and_separate_transcriptions(self):
        text = json.dumps({"pages": [{"page": 19, "text": "Second formula"}, {"page": 18, "text": "First formula"}]})
        with patch.object(vision_ocr, "completion_request", return_value=response(text)) as call:
            result = vision_ocr.transcribe_page_batch({18: self.image(), 19: self.image()})
        self.assertEqual(result, {18: "First formula", 19: "Second formula"})
        self.assertEqual(call.call_count, 1)
        self.assertEqual(call.call_args.args[1]["max_tokens"], 8192)

    def test_missing_duplicate_unknown_empty_and_truncated_pages_are_rejected(self):
        for rows in ([{"page": 1, "text": "Only first"}],
                     [{"page": 1, "text": "First"}, {"page": 1, "text": "Duplicated"}],
                     [{"page": 1, "text": "First"}, {"page": 3, "text": "Unknown"}],
                     [{"page": 1, "text": "First"}, {"page": 2, "text": ""}]):
            with self.subTest(rows=rows), patch.object(vision_ocr, "completion_request", return_value=response(json.dumps({"pages": rows}))):
                with self.assertRaises(IncompleteOutputError):
                    vision_ocr.transcribe_page_batch({1: self.image(), 2: self.image()})
        with patch.object(vision_ocr, "completion_request", return_value=response("{}", "length")):
            with self.assertRaises(IncompleteOutputError):
                vision_ocr.transcribe_page_batch({1: self.image(), 2: self.image()})

    def test_four_pages_use_two_requests_and_all_progress_is_reported(self):
        progress = []
        with patch.object(processor, "transcribe_page_batch", side_effect=lambda pages: {n: f"Notes from page {n}" for n in pages}) as batch, patch.object(processor, "transcribe_image") as single:
            pages = processor.extract_pages(self.pdf(), "notes.pdf", True, on_page=lambda page, *_: progress.append(page["page"]))
        self.assertEqual(batch.call_count, 2)
        single.assert_not_called()
        self.assertEqual(progress, [1, 2, 3, 4])
        self.assertEqual([p["text"] for p in pages], [f"Notes from page {n}" for n in progress])

    def test_bad_batch_falls_back_once_without_losing_pages(self):
        with patch.object(processor, "transcribe_page_batch", side_effect=IncompleteOutputError()) as batch, patch.object(processor, "transcribe_image", return_value="Recovered complete page") as single:
            pages = processor.extract_pages(self.pdf(), "notes.pdf", True)
        self.assertEqual(batch.call_count, 1)
        self.assertEqual(single.call_count, 4)
        self.assertTrue(all(p["status"] == "ok" for p in pages))

    def test_quota_defers_uncached_pages_without_individual_retries(self):
        cached = {"4": {"page": 4, "text": "Saved notes", "status": "ok", "method": "vision"}}
        with patch.object(processor, "transcribe_page_batch", side_effect=RateLimitDeferred(3600)) as batch, patch.object(processor, "transcribe_image") as single:
            pages = processor.extract_pages(self.pdf(), "notes.pdf", True, cached)
        self.assertEqual(batch.call_count, 1)
        single.assert_not_called()
        self.assertEqual([p["status"] for p in pages], ["pending", "pending", "pending", "ok"])
        self.assertEqual(pages[-1], cached["4"])

    def test_refinement_quota_does_not_discard_already_received_second_page(self):
        with patch.object(processor, "transcribe_page_batch", return_value={1: "A formula and [illegible] label", 2: "Readable second page"}), patch.object(processor, "transcribe_image", side_effect=RateLimitDeferred(3600)):
            pages = processor.extract_pages(self.pdf(), "notes.pdf", True)
        self.assertEqual([p["status"] for p in pages], ["ok", "ok", "pending", "pending"])
        self.assertEqual(pages[1]["text"], "Readable second page")


class AdaptiveAnswerTests(unittest.TestCase):
    def run_answer(self, *, score=.95, verification="adaptive", allow=True, answer="Photosynthesis converts light into chemical energy. [S1]", mode="ask"):
        chunks = [{"id": "1", "document_id": "doc", "content": "Photosynthesis converts light into chemical energy.",
                   "chunk_index": 0, "page_number": 1, "_rerank_score": score}]
        def retrieve(state):
            state.update(retrieved_chunks=chunks, filtered_chunks=chunks)
            return state
        with patch.object(graph, "_graph", None), patch.object(graph, "retrieve_node", side_effect=retrieve), patch.object(graph, "rerank_node", side_effect=lambda state: state), \
             patch.object(evidence, "_expand_neighbors", return_value=[]), patch.object(evidence, "token_count", side_effect=lambda t: len(t.split())), \
             patch.object(validator, "token_count", side_effect=lambda t: len(t.split())), \
             patch.object(generator, "chat_json", return_value=({"items": [{"text": answer, "source_ids": evidence.cited_ids(answer) or ["S1"]}], "insufficient_context": False} if mode in {"ask", "quick_revision", "exam_answer"} else {"answer": answer, "insufficient_context": False})) as generate, \
             patch.object(validator, "chat_json", return_value=({"items": [{"id": "B1", "grounded": True, "confidence": .95, "reason": "Supported", "evidence_ids": ["S1:P1"], "unsupported_claims": []}]} if mode in {"ask", "quick_revision", "exam_answer"} else {"grounded": True, "confidence": .95, "unsupported_claims": []})) as validate:
            result = graph.run_agent("What is photosynthesis?", ["doc"], verification_mode=verification, allow_fast_answer=allow, mode=mode)
        return result, generate.call_count, validate.call_count

    def test_strong_match_uses_one_call_and_does_not_claim_verification(self):
        result, generates, validations = self.run_answer()
        self.assertEqual((generates, validations), (1, 0))
        self.assertEqual(result["validation_status"], "not_checked")
        self.assertIsNone(result["groundedness_confidence"])
        self.assertFalse(result["grounded"])
        self.assertTrue(result["stage_timings_ms"]["generate"] >= 0)

    def test_full_mode_weaker_matches_partial_documents_and_study_modes_keep_checking(self):
        for options in ({"verification": "full"}, {"score": .6}, {"allow": False}, {"mode": "exam_answer"}):
            with self.subTest(options=options):
                result, generates, validations = self.run_answer(**options)
                self.assertEqual((generates, validations), (1, 1))
                self.assertEqual(result["validation_status"], "supported")

    def test_invalid_citations_never_pass_adaptive_gate(self):
        result, _, _ = self.run_answer(answer="A claim [S99]")
        self.assertEqual(result["validation_status"], "unsupported")
        self.assertEqual(result["cited_source_ids"], [])

    def test_uncited_paragraph_triggers_full_check(self):
        result, _, validations = self.run_answer(answer="An uncited claim.\n\nA cited claim [S1]")
        self.assertEqual(validations, 1)
        self.assertNotEqual(result["validation_status"], "not_checked")

    def test_heading_does_not_hide_an_uncited_paragraph(self):
        result, _, validations = self.run_answer(answer="# Explanation\nAn uncited claim.\n\nA cited claim [S1]")
        self.assertEqual(validations, 1)
        self.assertNotEqual(result["validation_status"], "not_checked")


class RetrievalSpeedTests(unittest.TestCase):
    def tearDown(self):
        retrieval_service.invalidate_bm25_cache()

    def test_neighbors_reuse_scoped_corpus_without_another_database_read(self):
        row = {"id": "1", "document_id": "doc", "content": "Photosynthesis", "chunk_index": 0}
        with patch.object(retrieval_service, "_all_candidate_chunks", return_value=[row]) as read:
            retrieval_service.keyword_search("Photosynthesis", ["doc", "doc2"], 5)
            self.assertEqual(retrieval_service._expand_neighbors([row], ["doc2", "doc"]), [row])
            self.assertEqual(read.call_count, 1)
            retrieval_service.keyword_search("Photosynthesis", ["different-user-doc"], 5)
            self.assertEqual(read.call_count, 2)

    def test_searches_overlap_without_changing_failure_handling(self):
        barrier = threading.Barrier(2, timeout=3)
        def search(*args):
            barrier.wait()
            return []
        with patch.object(retrieval_service, "vector_search", side_effect=search), patch.object(retrieval_service, "keyword_search", side_effect=search):
            warnings = []
            self.assertEqual(retrieval_service.hybrid_retrieve("query", ["doc"], 5, warnings), [])
            self.assertEqual(warnings, [])

