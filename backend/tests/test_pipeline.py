"""Offline regression tests. No provider or live database requests are made."""
import io
import re
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4
import pymupdf
from PIL import Image
from fastapi.testclient import TestClient

from app.services import chunking, vision_ocr, document_processor, retrieval_service, evidence
from app.agents import retrieval, generator, validator, summarizer, graph
from app.models.schemas import ChatRequest
from app.core.config import Settings
from app.core.auth import current_user
from app.main import app

class WordTokenizer:
    def __call__(self, text, **kwargs):
        return {"offset_mapping": [m.span() for m in re.finditer(r"\w+|[^\w\s]", text)]}

def chunk(index=0, doc="doc", content="Photosynthesis converts light energy into chemical energy."):
    return {"id": f"{doc}-{index}", "document_id": doc, "content": content,
            "page_number": index + 1, "chunk_index": index, "metadata": {"document_name": "notes.pdf"}}

class ChunkTests(unittest.TestCase):
    def setUp(self):
        self.patch = patch.object(chunking, "get_tokenizer", return_value=WordTokenizer())
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_oversized_paragraph_is_bounded(self):
        text = " ".join(f"term{i}" for i in range(1200))
        pieces = chunking.chunk_text(text, 200, 20)
        self.assertTrue(all(len(p.split()) <= 200 for p in pieces))
        self.assertIn("term1199", pieces[-1])

    def test_zero_overlap_and_full_coverage(self):
        text = " ".join(f"word{i}" for i in range(60))
        self.assertEqual(" ".join(chunking.chunk_text(text, 10, 0)), text)

    def test_preserves_formula_and_line_structure(self):
        text = "# Force\nF = m * a\n\nMass is measured in kg."
        self.assertEqual(chunking.chunk_text(text, 100, 0), [text])

    def test_invalid_overlap_rejected(self):
        with self.assertRaises(ValueError):
            chunking.chunk_text("hello", 10, 10)

    def test_empty_document(self):
        self.assertEqual(chunking.chunk_text(" \n "), [])

class QueryTests(unittest.TestCase):
    def test_one_word_topics_survive(self):
        for q in ("Photosynthesis", "Generate 5 MCQs about photosynthesis"):
            self.assertEqual(retrieval._clean_retrieval_query(q, "mcq").lower(), "photosynthesis")

    def test_summary_and_revision_route(self):
        for q in ("Quick revision of this document", "Give me a detailed summary of this PDF",
                  "Summarize this PDF", "Generate 5 MCQs from these notes"):
            self.assertTrue(retrieval.is_generic_document_request(q), q)

    def test_targeted_topic_stays_targeted(self):
        for q in ("Explain REST from these notes", "Give me 5 MCQs about virtualization from this PDF"):
            self.assertFalse(retrieval.is_generic_document_request(q))

    def test_topic_numbers_survive(self):
        self.assertIn("type 2", retrieval._clean_retrieval_query("Generate MCQs about type 2 diabetes", "mcq"))

    def test_followup_resolves_before_retrieval(self):
        state = {"original_question": "What are its drawbacks?", "history": [{"role": "user", "content": "Explain virtualization"}]}
        with patch.object(retrieval, "chat", return_value="What are the drawbacks of virtualization?"):
            result = retrieval.resolve_question_node(state)
        self.assertIn("virtualization", result["resolved_question"])

class OCRTests(unittest.TestCase):
    def setUp(self):
        # These exercise individual-page recovery; batch behavior has dedicated tests.
        settings = document_processor.get_settings().model_copy(update={"gemini_ocr_batch_pages": 1})
        self.settings_patch = patch.object(document_processor, "get_settings", return_value=settings)
        self.settings_patch.start()
        self.addCleanup(self.settings_patch.stop)

    def image(self):
        image = Image.new("RGB", (700, 1000), "white")
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        return stream.getvalue()

    def pdf(self):
        doc = pymupdf.open()
        for _ in range(2):
            page = doc.new_page()
            page.insert_text((40, 50), "A printed page header.")
        result = doc.tobytes()
        doc.close()
        return result

    def test_truncated_ocr_is_split_without_losing_tail(self):
        with patch.object(vision_ocr, "_request", side_effect=[
            vision_ocr.IncompleteOutputError(), "First line\nShared line", "Shared line\nFinal formula F = ma"
        ]) as request:
            text = vision_ocr.transcribe_image(self.image())
        self.assertEqual(request.call_count, 3)
        self.assertEqual(text.count("Shared line"), 1)
        self.assertIn("Final formula", text)

    def test_incomplete_after_max_depth_raises(self):
        with patch.object(vision_ocr, "_request", side_effect=vision_ocr.IncompleteOutputError()):
            with self.assertRaises(vision_ocr.IncompleteOutputError):
                vision_ocr.transcribe_image(self.image())

    def test_force_ocr_even_with_native_text(self):
        with patch.object(document_processor, "transcribe_image", return_value="Handwritten formula F = ma.") as ocr, patch.object(document_processor.time, "sleep"):
            pages = document_processor.extract_pages(self.pdf(), "notes.pdf", force_ocr=True)
        self.assertEqual(ocr.call_count, 2)
        self.assertTrue(all(p["method"] == "vision" for p in pages))

    def test_ocr_failures_are_not_searchable_error_text(self):
        with patch.object(document_processor, "transcribe_image", side_effect=RuntimeError("Secret provider diagnostic")), patch.object(document_processor.time, "sleep"):
            pages = document_processor.extract_pages(self.pdf(), "notes.pdf", True)
        self.assertTrue(all(p["status"] == "failed" and p["text"] == "" for p in pages))
        self.assertNotIn("Secret", str(pages))

    def test_cached_pages_skip_ocr_but_failed_pages_retry(self):
        cache = {"1": {"page": 1, "text": "A valid cached page.", "status": "ok", "method": "vision"},
                 "2": {"page": 2, "text": "", "status": "failed", "method": "vision"}}
        with patch.object(document_processor, "transcribe_image", return_value="A recovered handwritten page.") as ocr, patch.object(document_processor.time, "sleep"):
            pages = document_processor.extract_pages(self.pdf(), "notes.pdf", True, cache)
        self.assertEqual(ocr.call_count, 1)
        self.assertEqual(pages[0]["text"], "A valid cached page.")
        self.assertEqual(pages[1]["status"], "ok")

    def test_illegible_only_is_a_failed_page(self):
        with patch.object(document_processor, "transcribe_image", return_value="[illegible]"):
            pages = document_processor.extract_pages(self.image(), "page.png")
        self.assertEqual(pages[0]["status"], "failed")

    def test_blank_page_is_not_a_provider_failure(self):
        with patch.object(document_processor, "transcribe_image", return_value="[blank]"):
            pages = document_processor.extract_pages(self.image(), "page.png")
        self.assertEqual(pages[0]["status"], "blank")

class RetrievalTests(unittest.TestCase):
    def tearDown(self):
        retrieval_service.invalidate_bm25_cache()

    def test_empty_selection_never_searches_everyones_documents(self):
        with patch.object(retrieval_service, "get_supabase") as db:
            self.assertEqual(retrieval_service.hybrid_retrieve("question", [], 10), [])
            self.assertEqual(retrieval_service._all_candidate_chunks([]), [])
            db.assert_not_called()

    def test_pagination_reads_beyond_first_response(self):
        db = MagicMock()
        builder = db.table.return_value.select.return_value.in_.return_value.eq.return_value.eq.return_value.order.return_value
        builder.range.return_value.execute.side_effect = [SimpleNamespace(data=[chunk(i) for i in range(500)]),
                                                        SimpleNamespace(data=[chunk(500)])]
        with patch.object(retrieval_service, "get_supabase", return_value=db):
            rows = retrieval_service._all_candidate_chunks(["doc"])
        self.assertEqual(len(rows), 501)
        self.assertEqual(builder.range.call_args_list[-1].args, (500, 999))

    def test_single_page_bm25_is_not_discarded(self):
        with patch.object(retrieval_service, "_all_candidate_chunks", return_value=[chunk()]):
            hits = retrieval_service.keyword_search("photosynthesis", ["doc"], 5)
        self.assertEqual(len(hits), 1)

    def test_vector_outage_keeps_keyword_path(self):
        warnings = []
        with patch.object(retrieval_service, "vector_search", side_effect=RuntimeError("offline")), patch.object(retrieval_service, "keyword_search", return_value=[chunk()]):
            hits = retrieval_service.hybrid_retrieve("photosynthesis", ["doc"], 5, warnings)
        self.assertEqual(len(hits), 1)
        self.assertTrue(warnings)

    def test_citation_map_has_no_sliced_passages(self):
        with patch.object(evidence, "token_count", side_effect=lambda s: len(s.split())):
            chunks, context = evidence.pack_evidence([chunk(), chunk(1, content="Another complete passage.")], expand=False, budget=15)
        self.assertTrue(chunks)
        self.assertIn(chunks[0]["content"], context)
        self.assertEqual(chunks[0]["citation_id"], "S1")

class AnswerTests(unittest.TestCase):
    def test_structured_mcq_accepts_valid_options(self):
        q = generator.MCQ(question="What is converted?", options={"A": "Light", "B": "Sound", "C": "Heat", "D": "Pressure"},
                          answer="A", explanation="The notes describe light energy.", source_ids=["S1"])
        self.assertEqual(q.answer, "A")

    def test_duplicate_options_are_rejected(self):
        with self.assertRaises(ValueError):
            generator.MCQ(question="Q", options=dict.fromkeys("ABCD", "same"), answer="A", explanation="E", source_ids=["S1"])

    def test_validator_checks_answer_tail(self):
        answer = "Intro " + "word " * 1100 + " Unsupported tail [S1]"
        state = {"generated_answer": answer, "evidence_chunks": [dict(chunk(), citation_id="S1")]}
        with patch.object(validator, "token_count", return_value=30), patch.object(validator, "chat_json", return_value={"grounded": False, "confidence": .1, "unsupported_claims": ["Unsupported tail"]}) as llm:
            validator.validate_node(state)
        self.assertIn("Unsupported tail", llm.call_args.args[1])
        self.assertEqual(state["validation_status"], "unsupported")

    def test_unknown_citation_is_rejected_without_call(self):
        with patch.object(validator, "chat_json") as llm:
            state = validator.validate_node({"generated_answer": "Claim [S9]", "evidence_chunks": [dict(chunk(), citation_id="S1")]})
        self.assertEqual(state["validation_status"], "unsupported")
        llm.assert_not_called()

    def test_validator_outage_is_not_zero_confidence(self):
        with patch.object(validator, "token_count", return_value=20), patch.object(validator, "chat_json", side_effect=RuntimeError()):
            state = validator.validate_node({"generated_answer": "Claim [S1]", "evidence_chunks": [dict(chunk(), citation_id="S1")]})
        self.assertEqual(state["validation_status"], "unavailable")
        self.assertIsNone(state["groundedness_confidence"])

    def test_unsupported_answer_is_withheld(self):
        state = graph.finish_node({"validation_status": "unsupported", "generated_answer": "An invented answer", "cited_source_ids": ["S1"]})
        self.assertNotIn("invented", state["generated_answer"])
        self.assertEqual(state["cited_source_ids"], [])

    def test_repair_is_bounded(self):
        state = {"validation_status": "unsupported", "generation_attempts": 1, "mode": "ask"}
        self.assertEqual(graph.route_validate(state), "generate")
        state["generation_attempts"] = 2
        self.assertEqual(graph.route_validate(state), "finish")

    def test_graph_state_keeps_evaluator_explanation(self):
        from langgraph.graph import StateGraph, END
        from app.models.schemas import AgentState
        received = []
        def first(state):
            state["evaluation_reason"] = "Need a definition"
            return state
        def second(state):
            received.append(state["evaluation_reason"])
            return state
        builder = StateGraph(AgentState)
        builder.add_node("first", first); builder.add_node("second", second)
        builder.set_entry_point("first"); builder.add_edge("first", "second"); builder.add_edge("second", END)
        builder.compile().invoke({"question": "test"})
        self.assertEqual(received, ["Need a definition"])

class SummaryTests(unittest.TestCase):
    def test_sections_never_mix_documents_or_drop_content(self):
        chunks = [chunk(i, doc=f"doc{i}", content="x " * 100) for i in range(7)]
        with patch.object(summarizer, "token_count", side_effect=lambda t: len(t.split())):
            sections = summarizer._split_into_sections(chunks, 200)
        self.assertEqual(len(sections), 7)
        self.assertEqual([c["id"] for section in sections for c in section], [c["id"] for c in chunks])

    def test_long_document_has_all_sections(self):
        chunks = [chunk(i, content="x " * 300) for i in range(60)]
        with patch.object(summarizer, "token_count", side_effect=lambda t: len(t.split())):
            sections = summarizer._split_into_sections(chunks, 1000)
        self.assertEqual(sum(len(s) for s in sections), 60)
        self.assertGreater(len(sections), 6)

    def test_synthesis_groups_preserve_all_notes(self):
        notes = ["x " * 200 + str(i) for i in range(20)]
        with patch.object(summarizer, "token_count", side_effect=lambda t: len(t.split())):
            groups = summarizer._reduce_groups(notes, 500)
        self.assertEqual([note for group in groups for note in group], notes)

class APITests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        app.dependency_overrides.clear()

    def tearDown(self):
        app.dependency_overrides.clear()

    def test_anonymous_api_requests_rejected(self):
        for path in ("/api/documents", "/api/analytics/summary", "/api/chat/conversations"):
            self.assertEqual(self.client.get(path).status_code, 401)
        self.assertEqual(self.client.post("/api/chat", json={"question": "test"}).status_code, 401)

    def test_mode_and_document_identifiers_validated(self):
        with self.assertRaises(ValueError):
            ChatRequest(question="hello", mode="unknown")
        with self.assertRaises(ValueError):
            ChatRequest(question="hello", document_ids=["not-an-id"])

    def test_other_users_document_selection_rejected(self):
        app.dependency_overrides[current_user] = lambda: str(uuid4())
        db = MagicMock()
        db.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
        with patch("app.api.chat.get_supabase", return_value=db):
            response = self.client.post("/api/chat", json={"question": "test", "document_ids": [str(uuid4())]})
        self.assertEqual(response.status_code, 404)

    def test_no_ready_documents_is_actionable(self):
        app.dependency_overrides[current_user] = lambda: str(uuid4())
        db = MagicMock()
        db.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
        with patch("app.api.chat.get_supabase", return_value=db):
            response = self.client.post("/api/chat", json={"question": "test"})
        self.assertEqual(response.status_code, 409)
        self.assertIn("Upload", response.json()["detail"])

if __name__ == "__main__":
    unittest.main()
