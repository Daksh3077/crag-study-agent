import unittest
from unittest.mock import patch
from app.core.config import Settings
from app.services import answer_cache


class AnswerCacheTests(unittest.TestCase):
    def test_generator_reuses_checked_answer_but_changed_evidence_calls_model(self):
        from app.agents import generator, graph, validator
        answer_cache.clear()
        def state(content='A stack uses LIFO.'):
            return dict(mode='ask', original_question='Define stack', document_ids=['doc'],
                evidence_context=content, evidence_chunks=[dict(citation_id='S1', content=content)],
                generation_attempts=0)
        payload={'items':[{'text':'A stack uses LIFO.', 'source_ids':['S1']}],
                 'insufficient_context':False, 'missing_information':[]}
        with patch.object(generator, 'chat_json', return_value=payload) as model:
            first=generator.generate_node(state())
            first.update(validation_status='supported', grounded=True, groundedness_confidence=.95)
            graph.finish_node(first)
            cached=generator.generate_node(state())
            self.assertTrue(cached['answer_cache_hit'])
            with patch.object(validator, 'chat_json') as checker:
                validator.validate_node(cached)
                checker.assert_not_called()
            self.assertEqual(model.call_count, 1)
            generator.generate_node(state('Changed source evidence.'))
            self.assertEqual(model.call_count, 2)

    def setUp(self):
        answer_cache.clear()
        self.state = dict(validation_status='supported', grounded=True,
            generated_answer='A stack uses LIFO. [S1]', cited_source_ids=['S1'])

    def test_scope_evidence_question_and_validator_affect_key(self):
        settings=Settings(_env_file=None)
        original=answer_cache.key('question',['a'],'source',settings)
        for question,scope,context in [('changed',['a'],'source'),('question',['b'],'source'),('question',['a'],'changed')]:
            self.assertNotEqual(original,answer_cache.key(question,scope,context,settings))
        settings.validator_model='different'
        self.assertNotEqual(original,answer_cache.key('question',['a'],'source',settings))

    def test_partial_unchecked_and_failed_answers_are_not_cached(self):
        for changes in ({'warnings':['Partial']},{'unsupported_claims':['bad']},
                        {'answer_coverage_gaps':['missing']},{'validation_status':'not_checked'},
                        {'grounded':False},{'cited_source_ids':[]}):
            answer_cache.put('key',dict(self.state,**changes))
            self.assertIsNone(answer_cache.get('key'))

    def test_values_are_isolated_and_expire(self):
        with patch.object(answer_cache.time,'monotonic',return_value=10):
            answer_cache.put('key',self.state)
            cached=answer_cache.get('key')
            cached['cited_source_ids'].append('S99')
            self.assertEqual(answer_cache.get('key')['cited_source_ids'],['S1'])
        with patch.object(answer_cache.time,'monotonic',return_value=911):
            self.assertIsNone(answer_cache.get('key'))
