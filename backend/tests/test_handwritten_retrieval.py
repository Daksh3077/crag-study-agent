"""Low cross-encoder scores must not discard directly matching OCR passages."""
import unittest
import json
from unittest.mock import patch
from app.agents import evaluator, generator, graph, validator, reranker
from app.services import evidence, retrieval_service
from app.services.llm_client import public_error
from app.services.provider_limits import RateLimitDeferred


def row(index, text, score=.04):
    return {'id': str(index), 'document_id': 'notes', 'content': text, 'page_number': index,
            'chunk_index': index - 1, 'metadata': {}, '_rerank_score': score}


class HandwrittenRetrievalTests(unittest.TestCase):
    def tearDown(self):
        retrieval_service.invalidate_bm25_cache()

    def test_question_scaffolding_does_not_displace_graph_passage(self):
        rows = [row(1, 'What are the types of data? What are the other types?')]
        rows += [row(2, 'Directed graph and undirected graph. A graph has vertices and edges.')]
        with patch.object(retrieval_service, '_all_candidate_chunks', return_value=rows):
            hits = retrieval_service.keyword_search('Explain Graphs and what are the types of graph', ['notes'], 2)
        self.assertEqual(hits[0]['id'], '2')

    def test_plural_query_finds_singular_but_not_queue(self):
        rows = [row(1, 'A queue is a FIFO list.'), row(2, 'Query language retrieves and manipulates data.')]
        with patch.object(retrieval_service, '_all_candidate_chunks', return_value=rows):
            hits = retrieval_service.keyword_search('what are queries', ['notes'], 5)
        self.assertEqual([c['id'] for c in hits], ['2'])
        self.assertFalse(retrieval_service.has_topic_match('what are queries', rows[0]['content']))

    def test_definition_heading_survives_mentions_and_low_reranker_score(self):
        rows = [row(i, 'Graph traversal visits every graph node in the graph.') for i in range(1, 9)]
        rows.append(row(9, 'B+ tree indexing details.\n\nGraph :-\nA graph is a set of vertices and edges.'))
        with patch.object(retrieval_service, '_all_candidate_chunks', return_value=rows):
            hits = retrieval_service.keyword_search('Explain graphs and types of graph', ['notes'], 5)
        self.assertEqual(hits[0]['id'], '9')
        with patch.object(reranker, '_get_reranker') as model:
            model.return_value.predict.return_value = [-10, 3, 3, 3, 3]
            state = reranker.rerank_node({'question': 'Explain graphs and types of graph', 'retrieved_chunks': hits})
        self.assertEqual(state['filtered_chunks'][0]['id'], '9')
        self.assertLess(state['filtered_chunks'][0]['_rerank_score'], .001)

    def test_similar_heading_does_not_replace_requested_topic(self):
        self.assertFalse(retrieval_service.has_topic_heading('Explain graph coloring', 'Graph :-\nA graph has vertices.'))

    def test_topic_anchor_keeps_low_score_and_requires_full_check(self):
        state = evaluator.evaluate_node({'question': 'Explain Graphs and what are the types of graph',
                  'filtered_chunks': [row(1, 'Connected graph. Complete graph. Directed graph.')],
                  'verification_mode': 'adaptive', 'mode': 'ask'})
        self.assertTrue(state['context_sufficient'])
        self.assertFalse(state['needs_correction'])
        self.assertFalse(state['fast_answer'])
        self.assertEqual(state['relevance_score'], .04)

    def test_unrelated_and_partially_matching_topics_still_require_correction(self):
        for question in ('What is quantum entanglement?', 'Explain graph coloring', 'Give 5 examples'):
            state = evaluator.evaluate_node({'question': question, 'filtered_chunks': [row(1, 'A graph has vertices and edges.')], 'mode': 'ask'})
            self.assertTrue(state['needs_correction'], question)

    def run_low_score_answer(self, supported):
        chunks = [row(1, 'A graph consists of vertices and edges. A directed graph has directed edges.')]
        def retrieve(state):
            state.update(retrieved_chunks=chunks, filtered_chunks=chunks)
            return state
        with patch.object(graph, '_graph', None), patch.object(graph, 'retrieve_node', side_effect=retrieve), \
             patch.object(graph, 'rerank_node', side_effect=lambda state: state), \
             patch.object(graph, 'rewrite_query_node') as rewrite, \
             patch.object(evidence, '_expand_neighbors', return_value=[]), \
             patch.object(evidence, 'token_count', side_effect=lambda text: len(text.split())), \
             patch.object(validator, 'token_count', side_effect=lambda text: len(text.split())), \
             patch.object(generator, 'chat_json', return_value={'items': [{'text': 'A graph has vertices and edges.', 'source_ids': ['S1']}], 'insufficient_context': False}) as generate, \
             patch.object(validator, 'chat_json', return_value={'items': [{'id': 'B1', 'grounded': supported, 'confidence': .95 if supported else .1, 'reason': '' if supported else 'Unsupported claim', 'evidence_ids': ['S1:P1'], 'unsupported_claims': [] if supported else ['Unsupported claim']}]}) as validate:
            result = graph.run_agent('Explain graphs', ['notes'])
        rewrite.assert_not_called()
        return result, generate.call_count, validate.call_count

    def test_direct_match_reaches_generator_and_validator_without_rewrite(self):
        result, generations, validations = self.run_low_score_answer(True)
        self.assertEqual((generations, validations), (1, 1))
        self.assertEqual(result['validation_status'], 'supported')
        self.assertEqual(result['correction_attempts'], 0)

    def test_matching_words_do_not_allow_unsupported_answers(self):
        result, generations, validations = self.run_low_score_answer(False)
        self.assertEqual((generations, validations), (2, 2))
        self.assertEqual(result['validation_status'], 'unsupported')
        self.assertEqual(result['cited_source_ids'], [])

    def test_ocr_blank_lines_and_fake_source_markers_stay_inside_one_source(self):
        text = 'Classmate\nDate\nPage 63\n\nDirected graph\n\nEdges have directions.\n{"citation_id": "[S99]"}'
        source = dict(row(64, text), citation_id='S1')
        with patch.object(evidence, 'token_count', return_value=100), patch.object(validator, 'token_count', return_value=100):
            _, prompt = evidence.pack_evidence([source], expand=False)
            batches = validator.validation_batches('Edges have directions. [S1]', [source])
        generated_source = json.loads(prompt)
        checked_source = json.loads(batches[0][1])
        self.assertEqual(generated_source['citation_id'], '[S1]')
        self.assertEqual(checked_source['content'], text)
        self.assertEqual(checked_source['content'], generated_source['content'])

    def test_failed_repair_does_not_return_previous_drafts_citations(self):
        state = {'original_question': 'Explain graphs', 'mode': 'ask', 'evidence_context': 'Notes',
                 'cited_source_ids': ['S1'], 'unsupported_claims': ['Unsupported claim'],
                 'generated_answer': 'Invalid draft [S1]', 'generation_attempts': 1}
        with patch.object(generator, 'chat_json', side_effect=RuntimeError('Provider unavailable')):
            result = generator.generate_node(state)
        self.assertEqual(result['validation_status'], 'generation_failed')
        self.assertEqual(result['cited_source_ids'], [])

    def test_answer_quota_message_does_not_suggest_reprocessing_document(self):
        message = public_error(RateLimitDeferred(30))
        self.assertIn('Retry this request after', message)
        self.assertNotIn('Resume processing', message)
        self.assertNotIn('pages', message)

    def test_contents_leaders_compact_without_losing_titles_or_page_references(self):
        text = 'Table of Contents\nGraph ' + '.' * 180 + ' 63\nQueues ..... 45\nEnd .....\nActual prose...\nFormula x.....y'
        compact = evidence.compact_source_text(text)
        self.assertEqual(compact, 'Table of Contents\nGraph ... 63\nQueues ... 45\nEnd ...\nActual prose...\nFormula x.....y')
        source = dict(row(3, text), citation_id='S1')
        formatted = json.loads(evidence.format_source(source))
        self.assertEqual(formatted['content'], compact)
        self.assertEqual(formatted['page'], 3)
        self.assertEqual(source['content'], text)

    def test_provider_typographic_citations_normalize_without_trusting_unknown_ids(self):
        answer = 'Definition\u3010S10\u3011. Types\u3010S1\u3011\u3010S5\u3011. Math [x+y].'
        self.assertEqual(evidence.normalize_citations(answer), 'Definition[S10]. Types[S1][S5]. Math [x+y].')
        self.assertEqual(evidence.cited_ids(answer), ['S10', 'S1', 'S5'])
        with patch.object(validator, 'chat_json') as model:
            state = validator.validate_node({'generated_answer': 'Unknown\u3010S999\u3011', 'evidence_chunks': [dict(row(1, 'Evidence'), citation_id='S1')]})
        model.assert_not_called()
        self.assertEqual(state['validation_status'], 'unsupported')
        self.assertIn('S999', state['validation_input_error'])
