"""Legacy study modes retain a bounded factual repair after a citation-only retry."""
import unittest
from unittest.mock import patch
from app.agents import graph, generator, validator
from app.core.config import Settings
from app.services import evidence


class ClassificationRepairTests(unittest.TestCase):
    def run_case(self, drafts, verdicts, retries=1):
        chunk = {'id': 'one', 'document_id': 'ml', 'chunk_index': 0, 'page_number': 1,
                 'content': 'Naive Bayes assumes feature independence.', '_rerank_score': .763}
        def retrieve(state):
            state.update(retrieved_chunks=[chunk], filtered_chunks=[chunk])
            return state
        settings = Settings(_env_file=None, generation_retries=retries)
        with patch.object(graph, '_graph', None), patch.object(graph, 'get_settings', return_value=settings), \
             patch.object(graph, 'retrieve_node', side_effect=retrieve), \
             patch.object(graph, 'rerank_node', side_effect=lambda s: s), \
             patch.object(evidence, '_expand_neighbors', return_value=[]), \
             patch.object(evidence, 'token_count', side_effect=lambda t: len(t.split())), \
             patch.object(validator, 'token_count', side_effect=lambda t: len(t.split())), \
             patch.object(generator, 'chat_json', side_effect=[{'answer': d, 'insufficient_context': False} for d in drafts]) as generate, \
             patch.object(validator, 'chat_json', side_effect=verdicts) as validate:
            result = graph.run_agent('Explain all linear and non-linear algorithms', ['ml'], mode='important_questions')
        return result, generate, validate

    def test_citation_fix_then_factual_fix_keeps_source_validation(self):
        unsupported = {'grounded': False, 'confidence': .1, 'unsupported_claims': ['Non-linear classification is not established by the source.']}
        supported = {'grounded': True, 'confidence': .95, 'unsupported_claims': []}
        result, generate, validate = self.run_case([
            'Naive Bayes assumes feature independence.',
            'Naive Bayes is non-linear. [S1]',
            'Naive Bayes assumes feature independence. [S1]',
        ], [unsupported, supported])
        self.assertEqual((generate.call_count, validate.call_count), (3, 2))
        self.assertEqual(result['validation_status'], 'supported')
        self.assertNotIn('is non-linear', result['generated_answer'])
        self.assertIn('Non-linear classification is not established', generate.call_args.args[1])
        self.assertEqual(result['citation_format_repairs'], 1)

    def test_repeated_missing_citations_stop_without_model_validation(self):
        result, generate, validate = self.run_case(['No citations.', 'Still none.'], [])
        self.assertEqual(generate.call_count, 2)
        validate.assert_not_called()
        self.assertEqual(result['validation_status'], 'unsupported')
        self.assertEqual(result['cited_source_ids'], [])

    def test_factual_errors_after_citation_fix_remain_bounded_and_withheld(self):
        bad = {'grounded': False, 'confidence': .1, 'unsupported_claims': ['Unsupported classification']}
        result, generate, validate = self.run_case(['No citations.', 'Non-linear. [S1]', 'Non-linear again. [S1]'], [bad, bad])
        self.assertEqual((generate.call_count, validate.call_count), (3, 2))
        self.assertEqual(result['validation_status'], 'unsupported')
        self.assertEqual(result['cited_source_ids'], [])

    def test_zero_retry_setting_disables_all_repairs(self):
        result, generate, _ = self.run_case(['No citations.'], [], retries=0)
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(result['validation_status'], 'unsupported')
