"""Ask latency is reduced without accepting unchecked or unsupported claims."""
import json
import unittest
from unittest.mock import Mock, patch

from app.agents import graph, generator, validator, evaluator
from app.agents.answer_checker import check_answer_items, item_batches, ItemVerdict, references_match, source_passages
from app.core.config import Settings
from app.services import evidence
from app.services.cited_answer import InvalidAnswerFormat, parse_answer, render_answer


class CitedAnswerTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(_env_file=None)
        self.sources = [dict(id='1', document_id='ml', chunk_index=0, page_number=1,
                             citation_id='S1', content='Linear regression is linear.', _rerank_score=.763),
                        dict(id='2', document_id='ml', chunk_index=1, page_number=2,
                             citation_id='S2', content='Naive Bayes assumes independence.')]
        self.items = [{'text': 'Linear regression is linear.', 'source_ids': ['S1']},
                      {'text': 'Naive Bayes is non-linear.', 'source_ids': ['S2']}]

    def payload(self, items):
        return {'items': items, 'insufficient_context': False}

    def state(self):
        blocks, _ = parse_answer(self.payload(self.items), self.sources)
        return dict(answer_blocks=blocks, evidence_chunks=self.sources,
                    generated_answer=render_answer(blocks), cited_source_ids=['S1', 'S2'])

    def verdict(self, bad=False):
        return {'items': [{'id': 'B1', 'grounded': not bad, 'confidence': .1 if bad else .95, 'reason': 'Unsupported' if bad else 'Source states this.', 'evidence_ids': ['S1:P1'], 'unsupported_claims': ['Unsupported'] if bad else []},
                          {'id': 'B2', 'grounded': False, 'confidence': .1, 'reason': 'Classification absent from source.', 'evidence_ids': [], 'unsupported_claims': ['Classification absent from source.']}]}

    def test_plain_text_gets_deterministic_citations_and_math_is_preserved(self):
        blocks, _ = parse_answer(self.payload([{'text': 'Formula [x+y].', 'source_ids': ['[S1]', 'S1']}]), self.sources)
        self.assertEqual(render_answer(blocks), 'Formula [x+y]. [S1]')

    def test_broad_question_keeps_factual_check_even_with_high_relevance(self):
        result = evaluator.evaluate_node({'question': 'Explain all linear and non-linear algorithms',
            'filtered_chunks': [dict(self.sources[0], _rerank_score=.99)],
            'mode': 'ask', 'verification_mode': 'adaptive'})
        self.assertFalse(result['fast_answer'])

    def test_malformed_citations_or_empty_answers_are_rejected(self):
        for items in ([], [{'text': 'Claim', 'source_ids': []}],
                      [{'text': 'Claim', 'source_ids': ['S99']}],
                      [{'text': 'Claim [S2]', 'source_ids': ['S1']}],
                      [{'text': '[S1]', 'source_ids': ['S1']}],
                      [{'text': ' ', 'source_ids': ['S1']}]):
            with self.subTest(items=items), self.assertRaises(InvalidAnswerFormat):
                parse_answer(self.payload(items), self.sources)

    def test_unsupported_item_is_removed_with_explicit_coverage_warning(self):
        call = Mock(return_value=self.verdict())
        result = check_answer_items(self.state(), call, len, self.settings)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(result['validation_status'], 'supported')
        self.assertNotIn('Naive Bayes', result['generated_answer'])
        self.assertEqual(result['cited_source_ids'], ['S1'])
        self.assertIn('Partial answer', result['warnings'][0])
        self.assertIn('omitted', result['generated_answer'])

    def test_uncovered_requested_topics_are_visible_even_if_items_are_supported(self):
        state=self.state()
        state['answer_coverage_gaps']=['Application and presentation tier roles']
        result=check_answer_items(state, Mock(return_value=self.verdict()), len, self.settings)
        self.assertIn('Not covered by this answer: Application', result['generated_answer'])

    def test_mcq_source_labels_are_normalized_without_accepting_unknown_ids(self):
        quiz=generator.MCQ.model_validate({'question':'What ordering?',
            'options':{'A':'LIFO','B':'FIFO','C':'Random','D':'Sorted'},'answer':'A',
            'explanation':'A stack follows LIFO.','source_ids':['[S1]','\u3010S1\u3011','S99']})
        self.assertEqual(quiz.source_ids,['S1','S99'])

    def test_missing_duplicate_or_invented_verdict_ids_are_not_supported(self):
        good = self.verdict()['items'][0]
        for verdicts in ([good], [good, good], [good, dict(good, id='B99')]):
            with self.subTest(verdicts=verdicts):
                result = check_answer_items(self.state(), Mock(return_value={'items': verdicts}), len, self.settings)
                self.assertEqual(result['validation_status'], 'unavailable')
                self.assertFalse(result['grounded'])

    def test_invented_references_and_contradictory_verdicts_are_rejected(self):
        for changes in ({'evidence_ids': []},
                        {'evidence_ids': ['S1:P99']},
                        {'evidence_ids': ['S2:P1']},
                        {'unsupported_claims': ['Some clauses are unsupported.']}):
            with self.subTest(changes=changes):
                verdict = self.verdict()
                verdict['items'][0].update(changes)
                result = check_answer_items(self.state(), Mock(return_value=verdict), len, self.settings)
                self.assertEqual(result['validation_status'], 'unsupported')
                self.assertFalse(result['grounded'])

    def test_reference_ids_accept_brackets(self):
        verdict = self.verdict()
        verdict['items'][0]['evidence_ids'] = ['[S1:P1]']
        result = check_answer_items(self.state(), Mock(return_value=verdict), len, self.settings)
        self.assertEqual(result['validation_status'], 'supported')
        self.assertEqual(result['cited_source_ids'], ['S1'])

    def test_numbered_passages_preserve_math_and_all_source_text(self):
        text = 'Heading\n\n' + ('Text with ?? and punctuation: ' * 30) + '\n\nFinal fact.'
        passages = source_passages(dict(self.sources[0], content=text))
        self.assertEqual('\n\n'.join(p['text'] for p in passages), text)
        self.assertEqual(len(set(p['id'] for p in passages)), len(passages))

    def test_batches_preserve_full_sources_and_every_item(self):
        state = self.state()
        groups = item_batches(state['answer_blocks'], self.sources, 4500, len)
        self.assertEqual(len(groups), 1)
        self.assertEqual([b['id'] for b in groups[0][0]], ['B1', 'B2'])
        for source in self.sources:
            self.assertIn(source['content'], groups[0][1])
        shared = [dict(b, source_ids=['S1']) for b in state['answer_blocks']]
        source_text = item_batches(shared, self.sources, 4500, len)[0][1].split('\n\nAnswer items:')[0]
        self.assertEqual(source_text.count(self.sources[0]['content']), 1)

    def test_later_batch_outage_cannot_claim_success(self):
        state = self.state()
        for source in state['evidence_chunks']:
            source['content'] = 'x' * 2300
        first = self.verdict()['items'][0]
        call = Mock(side_effect=[{'items': [first]}, RuntimeError('offline')])
        result = check_answer_items(state, call, len, self.settings)
        self.assertEqual(call.call_count, 2)
        self.assertEqual(result['validation_status'], 'unavailable')
        self.assertFalse(result['grounded'])

    def run_graph(self, payload, verdict):
        def retrieve(state):
            state.update(retrieved_chunks=self.sources, filtered_chunks=self.sources)
            return state
        with patch.object(graph, '_graph', None), patch.object(graph, 'get_settings', return_value=self.settings), \
             patch.object(graph, 'retrieve_node', side_effect=retrieve), \
             patch.object(graph, 'rerank_node', side_effect=lambda s: s), \
             patch.object(evidence, '_expand_neighbors', return_value=[]), \
             patch.object(evidence, 'token_count', side_effect=lambda t: len(t.split())), \
             patch.object(validator, 'token_count', side_effect=lambda t: len(t.split())), \
             patch.object(generator, 'chat_json', return_value=payload) as generate, \
             patch.object(validator, 'chat_json', return_value=verdict) as validate:
            result = graph.run_agent('Explain all linear and non-linear algorithms', ['ml'], verification_mode='full')
        return result, generate.call_count, validate.call_count

    def test_partial_answer_gets_one_bounded_repair_then_discloses_omissions(self):
        result, generates, checks = self.run_graph(self.payload(self.items), self.verdict())
        self.assertEqual((generates, checks), (2, 2))
        self.assertEqual(result['validation_status'], 'supported')
        self.assertEqual(result['cited_source_ids'], ['S1'])

    def test_all_unsupported_is_bounded_and_withheld(self):
        result, generates, checks = self.run_graph(self.payload(self.items), self.verdict(bad=True))
        self.assertEqual((generates, checks), (2, 2))
        self.assertEqual(result['validation_status'], 'unsupported')
        self.assertEqual(result['cited_source_ids'], [])

    def test_format_failures_never_get_a_third_generation(self):
        result, generates, checks = self.run_graph(self.payload([]), {})
        self.assertEqual((generates, checks), (2, 0))
        self.assertEqual(result['validation_status'], 'unsupported')
