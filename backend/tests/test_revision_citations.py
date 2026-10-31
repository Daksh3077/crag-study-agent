import unittest
from unittest.mock import patch
from app.agents import generator, validator


class RevisionCitationTests(unittest.TestCase):
    def test_topic_revision_uses_bounded_context(self):
        from app.core.config import Settings
        settings = Settings(_env_file=None)
        state = dict(mode='quick_revision', original_question='Revise stacks', filtered_chunks=[])
        with patch.object(generator, 'get_settings', return_value=settings), patch.object(generator, 'pack_evidence', return_value=([], '')) as pack:
            generator.generate_node(state)
        self.assertEqual(pack.call_args.kwargs['budget'], min(settings.context_token_budget, settings.fast_answer_context_tokens))

    def test_revision_has_citations_and_checks_each_concept(self):
        state = dict(mode='quick_revision', original_question='Revise variance',
                     evidence_context='Variance is sensitivity to training data changes.',
                     evidence_chunks=[dict(citation_id='S1', content='Variance is sensitivity to training data changes.')])
        with patch.object(generator, 'chat_json', return_value={
                'items':[dict(text='Variance is sensitivity to training data changes.', source_ids=['S1'])],
                'insufficient_context':False, 'missing_information':[]}):
            result = generator.generate_node(state)
        self.assertIn('[S1]', result['generated_answer'])
        self.assertEqual(len(result['answer_blocks']), 1)
        with patch.object(validator, 'token_count', side_effect=len), patch.object(validator, 'chat_json',
                return_value={'items':[dict(id='B1', grounded=True, confidence=.95, unsupported_claims=[], reason='Definition supported', evidence_ids=['S1:P1'])]}) as check:
            validated = validator.validate_node(result)
        check.assert_called_once()
        self.assertEqual(validated['validation_status'], 'supported')

    def test_revision_rejects_unknown_citation_ids(self):
        state = dict(mode='quick_revision', original_question='Revise variance',
                     evidence_context='Source', evidence_chunks=[dict(citation_id='S1', content='Source')])
        with patch.object(generator, 'chat_json', return_value={
                'items':[dict(text='Claim', source_ids=['S99'])], 'insufficient_context':False}):
            result = generator.generate_node(state)
        self.assertTrue(result['generation_format_error'])
        self.assertFalse(result['cited_source_ids'])
