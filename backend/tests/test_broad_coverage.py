import unittest
from unittest.mock import patch
from app.agents import retrieval, generator
from app.services import evidence


class BroadCoverageTests(unittest.TestCase):
    def state(self, scope=None, question='Explain all algorithms'):
        return {'question':question,'mode':'ask','document_ids':scope or ['ml']}

    def test_single_small_document_reuses_scoped_corpus(self):
        rows=[{'id':str(i),'document_id':'ml','chunk_index':i,'content':f'Concept {i}'} for i in range(14)]
        with patch.object(retrieval,'hybrid_retrieve',return_value=rows[:5]), \
             patch.object(retrieval,'_get_bm25_index',return_value=(rows,None)) as corpus:
            state=retrieval.retrieve_node(self.state())
        self.assertEqual(len(state['coverage_chunks']),14)
        corpus.assert_called_once_with(['ml'])

    def test_multi_document_and_narrow_questions_do_not_expand_scope(self):
        for state in (self.state(['ml','dsa']),self.state(question='Define regression')):
            with patch.object(retrieval,'hybrid_retrieve',return_value=[]), patch.object(retrieval,'_get_bm25_index') as corpus:
                self.assertEqual(retrieval.retrieve_node(state)['coverage_chunks'],[])
                corpus.assert_not_called()

    def test_large_document_does_not_silently_expand_all_passages(self):
        with patch.object(retrieval,'hybrid_retrieve',return_value=[]), \
             patch.object(retrieval,'_get_bm25_index',return_value=([{}]*25,None)):
            self.assertEqual(retrieval.retrieve_node(self.state())['coverage_chunks'],[])

    def test_generation_includes_unranked_small_document_passages(self):
        rows=[{'id':str(i),'document_id':'ml','chunk_index':i,'content':f'Unique concept {i}'} for i in range(14)]
        state=dict(self.state(),original_question='Explain all algorithms',filtered_chunks=rows[:5],coverage_chunks=rows)
        with patch.object(evidence,'token_count',return_value=20), patch.object(evidence,'_expand_neighbors') as expand, \
             patch.object(generator,'chat_json',return_value={'items':[{'text':'Concept 13','source_ids':['S14']}],'insufficient_context':False}):
            result=generator.generate_node(state)
        self.assertEqual(len(result['evidence_chunks']),14)
        self.assertIn('Unique concept 13',result['evidence_context'])
        expand.assert_not_called()
