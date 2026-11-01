"""Route tests ensure quotas block work, not just UI controls."""
from unittest.mock import patch
from uuid import uuid4
from test_document_workflow import DocumentWorkflowTests


class UsageLimitTests(DocumentWorkflowTests):
    def test_two_total_uploads_even_after_delete(self):
        first = self.upload().json()["id"]
        self.assertEqual(self.client.delete('/api/documents/' + first).status_code, 200)
        self.assertEqual(self.upload().status_code, 202)
        with patch('app.api.documents.process_document') as process:
            response = self.upload()
        self.assertEqual(response.status_code, 429)
        process.assert_not_called()
        self.assertEqual(self.client.get('/api/usage').json()['uploads_used'], 2)
        self.owner = str(uuid4())
        self.assertEqual(self.client.get('/api/usage').json()['uploads_used'], 0)

    def test_five_questions_across_conversations_and_modes(self):
        doc = self.upload().json()['id']
        with patch('app.api.chat.run_agent', return_value={}) as agent:
            for mode in ['ask', 'quick_revision', 'detailed_summary', 'mcq', 'exam_answer']:
                response = self.client.post('/api/chat', json={'question':'Explain', 'document_ids':[doc], 'mode':mode})
                self.assertEqual(response.status_code, 200)
            for extra in ({}, {'request_id':str(uuid4())}):
                response = self.client.post('/api/chat', json={'question':'One more', 'document_ids':[doc], **extra})
                self.assertEqual(response.status_code, 429)
            self.assertEqual(agent.call_count, 5)
        self.assertEqual(self.client.get('/api/usage').json()['questions_used'], 5)
        self.assertEqual(len(self.db.rows['conversations']), 5)

    def test_invalid_requests_do_not_spend_allowance(self):
        self.assertEqual(self.client.post('/api/chat', json={'question':'No documents'}).status_code, 409)
        self.assertEqual(self.client.post('/api/documents/upload', files={'file':('bad.exe',b'x')}).status_code, 400)
        self.assertEqual(self.client.get('/api/usage').json()['uploads_used'], 0)
        self.assertEqual(self.client.get('/api/usage').json()['questions_used'], 0)

    def test_usage_database_failure_blocks_expensive_work(self):
        with patch('app.services.usage_limits.get_supabase', side_effect=RuntimeError('offline')), \
             patch('app.api.documents.process_document') as process:
            with self.assertLogs('app.services.usage_limits', level='ERROR'):
                self.assertEqual(self.upload().status_code, 503)
            process.assert_not_called()
        self.assertEqual(self.db.files, {})

    def test_provider_failure_still_spends_reserved_question(self):
        self.upload()
        with patch('app.api.chat.run_agent', side_effect=RuntimeError('provider failed')):
            with self.assertLogs('app.api.chat', level='ERROR'):
                self.assertEqual(self.client.post('/api/chat', json={'question':'Explain'}).status_code, 503)
        self.assertEqual(self.client.get('/api/usage').json()['questions_used'], 1)
