"""Regression coverage for large cited summaries and bounded whole-document repair."""
import unittest
from unittest.mock import patch
from app.agents import graph, summarizer, validator
from app.core.config import get_settings


class SummaryValidationTests(unittest.TestCase):
    def setUp(self):
        settings = get_settings().model_copy(update={'validation_batch_tokens': 45, 'generation_retries': 1})
        for target in (validator, graph):
            p = patch.object(target, 'get_settings', return_value=settings)
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(validator, 'token_count', side_effect=lambda text: len(text.split()))
        p.start()
        self.addCleanup(p.stop)
        self.chunks = [{'citation_id': f'S{i}', 'content': f'topic{i} ' * 20} for i in range(1, 4)]

    def test_contiguous_bullets_split_without_dropping_claims(self):
        answer = '# Summary\n\n' + '\n'.join(f'- Fact {i} [S{i}]' for i in range(1, 4)) + '\n\nUncited tail claim.'
        batches = validator.validation_batches(answer, self.chunks)
        self.assertEqual(len(batches), 3)
        self.assertIn('# Summary', batches[0][0])
        self.assertIn('Uncited tail claim.', batches[-1][0])
        for i, (text, evidence) in enumerate(batches, 1):
            self.assertIn(f'Fact {i}', text)
            self.assertIn(f'topic{i}', evidence)
            self.assertLessEqual(len(text.split()) + len(evidence.split()), 45)

    def test_table_rows_keep_headers_and_cited_evidence(self):
        answer = '| Topic | Meaning |\n|---|---|\n' + '\n'.join(f'| Item{i} | Fact [S{i}] |' for i in range(1, 4))
        batches = validator.validation_batches(answer, self.chunks)
        self.assertEqual(len(batches), 3)
        for i, (text, evidence) in enumerate(batches, 1):
            self.assertIn('| Topic | Meaning |', text)
            self.assertIn(f'Item{i}', text)
            self.assertIn(f'topic{i}', evidence)

    def test_oversized_claim_is_check_unavailable_and_repairable(self):
        state = {'generated_answer': 'Combined claim [S1] [S2] [S3]', 'evidence_chunks': self.chunks,
                 'generic_request': True, 'mode': 'detailed_summary', 'generation_attempts': 1}
        with patch.object(validator, 'chat_json') as model:
            validator.validate_node(state)
        model.assert_not_called()
        self.assertEqual(state['validation_status'], 'unavailable')
        self.assertIsNone(state['groundedness_confidence'])
        self.assertIn('single claim', state['pipeline_log'][0]['detail'])
        self.assertEqual(graph.route_validate(state), 'summary_repair')
        state['generation_attempts'] = 2
        self.assertEqual(graph.route_validate(state), 'finish')
        self.assertNotIn('Combined claim', graph.finish_node(state)['generated_answer'])

    def test_repair_uses_all_summary_notes_not_top_k(self):
        state = {'original_question': 'Summarize this document', 'mode': 'detailed_summary',
                 'summary_notes': ['First section [S1]', 'Last section [S61]'],
                 'summary_section_count': 9, 'generation_attempts': 1,
                 'unsupported_claims': ['Unknown citation'], 'validation_input_error': 'Unknown citation'}
        with patch.object(summarizer, 'chat', return_value='First fact [S1].\n\nLast fact [S61].') as model:
            summarizer.repair_document_summary_node(state)
        prompt = model.call_args.args[1]
        self.assertIn('First section [S1]', prompt)
        self.assertIn('Last section [S61]', prompt)
        self.assertIn('Unknown citation', prompt)
        self.assertEqual(state['generation_attempts'], 2)
        self.assertEqual(state['validation_input_error'], '')
        self.assertEqual(state['cited_source_ids'], ['S1', 'S61'])

    def test_split_batches_still_check_each_claim(self):
        answer = '\n'.join(f'- Fact {i} [S{i}]' for i in range(1, 4))
        verdicts = [{'grounded': True, 'confidence': .95, 'unsupported_claims': []}] * 2
        verdicts += [{'grounded': False, 'confidence': .1, 'unsupported_claims': ['Fact 3 is unsupported']}]
        with patch.object(validator, 'chat_json', side_effect=verdicts) as model:
            state = validator.validate_node({'generated_answer': answer, 'evidence_chunks': self.chunks})
        self.assertEqual(model.call_count, 3)
        self.assertEqual(state['validation_status'], 'unsupported')
        self.assertIn('Fact 3 is unsupported', state['unsupported_claims'])

    def test_contents_pages_are_not_used_as_chapter_explanations(self):
        chunks = [
            {'document_id': 'a', 'page_number': 1, 'content': '# Table of Contents\nGraphs ........ 63\nQueues ........ 45'},
            {'document_id': 'a', 'page_number': 2, 'content': 'DBMS\nJoins ......... 48\nTransactions ......... 70'},
            {'document_id': 'a', 'page_number': 3, 'content': 'A queue is a FIFO list. ' * 20},
            {'document_id': 'a', 'page_number': 4, 'content': 'Experimental results\nTrial one ........ 10\nTrial two ........ 20'},
            {'document_id': 'b', 'page_number': 1, 'content': 'Graph ......... 1\nTree ......... 2'},
        ]
        self.assertEqual(summarizer._contents_pages(chunks), {('a', 1), ('a', 2)})

    def test_index_heading_does_not_exclude_page_with_substantive_explanations(self):
        text = 'Table of contents\nGraph ......... 1\nTree ......... 2\n' + 'A graph consists of vertices and edges. ' * 10
        self.assertEqual(summarizer._contents_pages([{'document_id': 'a', 'page_number': 1, 'content': text}]), set())
