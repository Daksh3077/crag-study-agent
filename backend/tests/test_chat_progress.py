import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from uuid import uuid4
from fastapi.testclient import TestClient
from app.main import app
from app.api import chat
from app.core.auth import current_user
from app.services import chat_progress, provider_limits
from test_document_workflow import MemoryDB


class ChatProgressTests(unittest.TestCase):
    def setUp(self):
        with chat_progress._lock:
            chat_progress._requests.clear()

    def test_owner_scope_expiry_and_duplicate_ids(self):
        chat_progress.begin('alice', 'r')
        self.assertIsNone(chat_progress.get('bob', 'r'))
        with self.assertRaises(ValueError):
            chat_progress.begin('alice', 'r')
        chat_progress.begin('bob', 'r')
        chat_progress.quota('alice', 'r', 'groq:model', 10)
        self.assertEqual(chat_progress.get('alice', 'r')['phase'], 'quota_wait')
        self.assertEqual(chat_progress.get('bob', 'r')['phase'], 'processing')
        with patch.object(chat_progress.time, 'time', return_value=time.time() + chat_progress.TTL + 1):
            self.assertIsNone(chat_progress.get('alice', 'r'))

    def test_provider_observer_reports_start_and_resume_without_changing_totals(self):
        statuses, waits = [], []
        with patch.dict(provider_limits._cooldowns, {'groq:test':1}), \
             patch.object(provider_limits.time, 'monotonic', side_effect=[0, 0, 1]), \
             provider_limits.observe_quota(lambda key, seconds: statuses.append((key, seconds))), \
             provider_limits.observe_waits(waits.append):
            provider_limits.wait_for_model('groq:test', 2)
        self.assertEqual(statuses, [('groq:test', 1), ('groq:test', None)])
        self.assertEqual(waits, [1])
        self.assertIsNone(provider_limits._quota_callback.get())

    def test_quota_observer_reaches_langgraph_nodes(self):
        from langgraph.graph import StateGraph, END
        events = []
        def node(state):
            provider_limits._quota_callback.get()('groq:test', 5)
            return state
        graph = StateGraph(dict)
        graph.add_node('wait', node)
        graph.set_entry_point('wait')
        graph.add_edge('wait', END)
        with provider_limits.observe_quota(lambda key, seconds: events.append((key, seconds))):
            graph.compile().invoke({})
        self.assertEqual(events, [('groq:test', 5)])

    def test_status_requires_authentication(self):
        client = TestClient(app)
        try:
            self.assertEqual(client.get('/api/chat/requests/' + str(uuid4())).status_code, 401)
        finally:
            client.close()

    def test_status_can_be_polled_while_answer_is_waiting_and_is_private(self):
        owner, other, doc, request_id = [str(uuid4()) for _ in range(4)]
        db = MemoryDB()
        db.rows['documents'] = [dict(id=doc, user_id=owner, name='Notes', status='ready', ingestion={})]
        waiting, resume = threading.Event(), threading.Event()
        def answer(*args, **kwargs):
            notify = provider_limits._quota_callback.get()
            notify('groq:test', 5)
            waiting.set()
            if not resume.wait(5):
                raise RuntimeError('Test did not resume')
            notify('groq:test', None)
            return dict(generated_answer='Test answer', validation_status='not_checked')
        client = TestClient(app)
        app.dependency_overrides[current_user] = lambda: owner
        try:
            with patch.object(chat, 'get_supabase', return_value=db), patch.object(chat, 'run_agent', side_effect=answer), ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(client.post, '/api/chat', json={'question':'Question', 'document_ids':[doc], 'request_id':request_id})
                try:
                    self.assertTrue(waiting.wait(3))
                    status = client.get('/api/chat/requests/' + request_id)
                    self.assertEqual(status.status_code, 200)
                    self.assertEqual(status.headers['cache-control'], 'no-store')
                    self.assertEqual(status.json()['phase'], 'quota_wait')
                    self.assertEqual(status.json()['provider'], 'Groq')
                    self.assertGreater(status.json()['retry_after_seconds'], 0)
                    app.dependency_overrides[current_user] = lambda: other
                    self.assertEqual(client.get('/api/chat/requests/' + request_id).status_code, 404)
                finally:
                    app.dependency_overrides[current_user] = lambda: owner
                    resume.set()
                self.assertEqual(future.result(timeout=3).status_code, 200)
                self.assertEqual(client.get('/api/chat/requests/' + request_id).json()['phase'], 'complete')
        finally:
            resume.set()
            app.dependency_overrides.clear()
            client.close()

    def test_failure_finishes_progress_and_old_clients_still_work(self):
        from app.models.schemas import ChatRequest
        request_id = str(uuid4())
        with patch.object(chat, '_answer', side_effect=RuntimeError('offline')):
            with self.assertRaises(RuntimeError):
                chat.chat(ChatRequest(question='Q', request_id=request_id), 'owner')
        self.assertEqual(chat_progress.get('owner', request_id)['phase'], 'failed')
        with patch.object(chat, '_answer', return_value='answer'):
            self.assertEqual(chat.chat(ChatRequest(question='Q'), 'owner'), 'answer')
