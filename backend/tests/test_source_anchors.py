import unittest
from unittest.mock import patch, Mock
from app.services.cited_answer import parse_answer, InvalidAnswerFormat, answer_schema
from app.services.evidence import source_passages, format_source, with_section_context, pack_evidence
from app.agents import generator
from app.agents.answer_checker import check_answer_items
from app.core.config import Settings

class SourceAnchorTests(unittest.TestCase):
    def test_method_inventory_uses_visible_document_headings(self):
        from app.services.evidence import algorithm_topics
        chunks = with_section_context([dict(id='a', document_id='d', chunk_index=0, citation_id='S1',
            content='Linear Algorithms\n\nLinear Regression\n\nA linear model.\n\nLogistic Regression\n\nA classifier.\n\nInteresting Resources\n\nExternal Reading')])
        self.assertEqual(algorithm_topics(chunks), ['Linear Regression', 'Logistic Regression'])
        self.assertEqual(algorithm_topics([dict(chunks[0], visible_passage_ids=[])]), [])

    def test_missing_method_coverage_gets_only_bounded_repair(self):
        from app.agents.graph import route_validate
        state = dict(mode='ask', validation_status='supported', answer_blocks=[{'text': 'Checked'}],
                     coverage_missing_topics=['Decision Trees'], generation_attempts=1)
        with patch('app.agents.graph.get_settings', return_value=Settings(_env_file=None, generation_retries=1)):
            self.assertEqual(route_validate(state), 'generate')
            self.assertEqual(route_validate(dict(state, generation_attempts=2)), 'finish')

    def test_topic_label_is_part_of_the_checked_claim(self):
        payload = self.payload()
        payload['items'][0]['topic'] = 'Decision Trees'
        blocks, _ = parse_answer(payload, self.sources)
        self.assertTrue(blocks[0]['text'].startswith('Decision Trees:'))
        self.assertEqual(blocks[0]['topic'], 'Decision Trees')

    def setUp(self):
        self.sources = [dict(citation_id='S1', content='Bias is a systematic error in a model.'),
                        dict(citation_id='S2', content='Variance is sensitivity to changes in training data.')]

    def payload(self, anchors=None):
        return dict(items=[dict(text='Variance is sensitivity to changes in training data.',
                               source_ids=['S1'], evidence_ids=anchors or ['S2:P1'])],
                    insufficient_context=False, missing_information=[])

    def test_passage_anchor_fixes_wrong_citation_without_changing_claim(self):
        blocks, _ = parse_answer(self.payload(), self.sources)
        self.assertEqual(blocks[0]['source_ids'], ['S1', 'S2'])
        self.assertEqual(blocks[0]['text'], self.sources[1]['content'])

    def test_invalid_anchor_hint_never_adds_a_foreign_or_invented_source(self):
        for anchors in [['S2:P99'], ['S99:P1']]:
            with self.subTest(anchors=anchors):
                blocks, _ = parse_answer(self.payload(anchors), self.sources)
                self.assertEqual(blocks[0]['source_ids'], ['S1'])
                self.assertEqual(blocks[0]['evidence_ids'], [])

    def test_anchor_does_not_remove_joint_supporting_citation(self):
        payload = self.payload()
        payload['items'][0]['source_ids'] = ['S1', 'S2']
        blocks, _ = parse_answer(payload, self.sources)
        self.assertEqual(blocks[0]['source_ids'], ['S1', 'S2'])

    def test_repair_never_discards_a_previously_checked_concept(self):
        blocks, _ = parse_answer(self.payload(), self.sources)
        previous = dict(id='old', text='Bias is a systematic error in a model.', source_ids=['S1'])
        state = dict(answer_blocks=blocks, verified_answer_blocks=[previous], evidence_chunks=self.sources,
                     groundedness_confidence=.95)
        verdict = {'items':[dict(id='B1', grounded=True, confidence=.95, reason='Definition', evidence_ids=['S2:P1'], unsupported_claims=[])]}
        result = check_answer_items(state, Mock(return_value=verdict), len, Settings(_env_file=None))
        self.assertIn('Bias', result['generated_answer'])
        self.assertIn('Variance', result['generated_answer'])

    def test_failed_repair_returns_only_previously_checked_content(self):
        from app.agents.graph import finish_node
        result = finish_node(dict(validation_status='unsupported', validation_input_error='Bad new draft',
                                  verified_answer_blocks=[dict(id='B1', text='Checked definition.', source_ids=['S1'])]))
        self.assertIn('Checked definition.', result['generated_answer'])
        self.assertIn('Partial answer', result['generated_answer'])
        self.assertEqual(result['cited_source_ids'], ['S1'])

    def test_schema_requests_only_known_passages(self):
        field = answer_schema(['S1'], ['S1:P1'])['$defs']['AnswerItem']
        self.assertIn('evidence_ids', field['required'])
        self.assertIn('pattern', field['properties']['evidence_ids']['items'])

    def test_generation_and_checker_share_passages_without_losing_text(self):
        import json
        c = dict(self.sources[0], content='Heading\n\n' + 'Long definition. ' * 50 + '\n\nFinal fact.')
        self.assertEqual(json.loads(format_source(c, numbered=True))['passages'], source_passages(c))
        self.assertEqual('\n\n'.join(p['text'] for p in source_passages(c)), c['content'])

    def test_anchor_does_not_make_unsupported_inference_pass(self):
        blocks, _ = parse_answer(self.payload(), self.sources)
        blocks[0]['text'] = 'Variance proves that all models are accurate.'
        verdict = {'items':[dict(id='B1', grounded=False, confidence=.1, reason='Inference absent',
                                evidence_ids=['S2:P1'], unsupported_claims=['Accuracy not established'])]}
        result = check_answer_items(dict(answer_blocks=blocks, evidence_chunks=self.sources), Mock(return_value=verdict), len, Settings(_env_file=None))
        self.assertEqual(result['validation_status'], 'unsupported')

    def test_adjacent_method_sections_do_not_merge(self):
        chunks = [dict(id='a', document_id='d', chunk_index=0, citation_id='S1', content='Linear Regression\n\nA linear model.'),
                  dict(id='b', document_id='d', chunk_index=1, citation_id='S2', content='Learning:\nOLS estimates coefficients.\n\nLogistic Regression\n\nUse maximum likelihood.')]
        annotated = with_section_context(chunks)
        passages = source_passages(annotated[1])
        self.assertEqual(passages[0]['section'], 'Linear Regression')
        self.assertEqual(passages[1]['section'], 'Logistic Regression')
        self.assertNotIn('OLS', passages[1]['text'])

    def test_heading_does_not_cross_missing_chunks_or_documents(self):
        chunks = [dict(id='a', document_id='d', chunk_index=0, content='Linear Regression'),
                  dict(id='b', document_id='d', chunk_index=3, content='Some disconnected text.'),
                  dict(id='c', document_id='other', chunk_index=4, content='Another source.')]
        annotated = with_section_context(chunks)
        self.assertIsNone(annotated[1]['section_start'])
        self.assertIsNone(annotated[2]['section_start'])

    def test_garbled_formula_is_not_a_section_heading(self):
        chunks = [dict(id='a', document_id='d', chunk_index=0, content='Logistic Regression\n\n𝑒QRSQTUTS⋯SQVUV'),
                  dict(id='b', document_id='d', chunk_index=1, content='Training uses maximum likelihood.')]
        self.assertEqual(with_section_context(chunks)[1]['section_start'], 'Logistic Regression')

    def test_balanced_context_keeps_later_sources_and_complete_passages(self):
        import json
        rows = [dict(id=str(i), document_id='d', chunk_index=i,
                     content=('Long equation fragment ' * 50) + '\n\n' + f'Concept {i} has a readable definition here.') for i in range(3)]
        with patch('app.services.evidence.token_count', side_effect=len):
            chunks, context = pack_evidence(rows, expand=False, budget=1800, numbered=True, balanced=True)
        self.assertEqual(len(chunks), 3)
        self.assertLessEqual(sum(len(x) for x in context.split('\n\n')), 1800)
        for chunk in chunks:
            shown = json.loads(format_source(chunk, numbered=True))['passages']
            self.assertEqual(len(shown), 1)
            self.assertIn('readable definition', shown[0]['text'])
            self.assertIn('Long equation', chunk['content'])  # Full checker evidence retained.

    def test_hidden_anchor_hint_is_discarded(self):
        sources = [dict(self.sources[0], visible_passage_ids=[]), self.sources[1]]
        blocks, _ = parse_answer(self.payload(['S1:P1']), sources)
        self.assertEqual(blocks[0]['evidence_ids'], [])

    def test_exam_and_revision_keep_supported_concepts_when_another_fails(self):
        for mode in ['quick_revision', 'exam_answer']:
            state = dict(mode=mode, original_question='Explain bias and variance',
                         evidence_context='Sources', evidence_chunks=self.sources)
            payload = self.payload()
            payload['items'].append(dict(text='Bias vanishes automatically.', source_ids=['S1'], evidence_ids=['S1:P1']))
            with patch.object(generator, 'chat_json', return_value=payload):
                result = generator.generate_node(state)
            verdict = {'items': [dict(id='B1', grounded=True, confidence=.95, reason='Definition', evidence_ids=['S2:P1'], unsupported_claims=[]),
                                 dict(id='B2', grounded=False, confidence=.1, reason='Not stated', evidence_ids=[], unsupported_claims=['Not stated'])]}
            result = check_answer_items(result, Mock(return_value=verdict), len, Settings(_env_file=None))
            self.assertIn('Variance', result['generated_answer'])
            self.assertNotIn('vanishes', result['generated_answer'])
            self.assertTrue(any('Partial answer' in w for w in result['warnings']))
