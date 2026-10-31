import unittest
import asyncio
from concurrent.futures import Future
from unittest.mock import patch
from app import main
from test_document_workflow import MemoryDB

class ReadinessRecoveryTests(unittest.TestCase):
    def test_recovery_failure_does_not_prevent_later_jobs(self):
        db = MemoryDB()
        db.rows['documents'] = [dict(id='a', status='queued'), dict(id='b', status='queued')]
        with patch.object(main, 'get_supabase', return_value=db), patch.object(main, 'process_document', side_effect=[RuntimeError('failed'), None]) as process:
            asyncio.run(main.resume_documents(['a', 'b']))
        self.assertEqual(process.call_count, 2)
        self.assertEqual(db.rows['documents'][0]['status'], 'failed')

    def test_readiness_requires_models_and_successful_startup(self):
        future = Future()
        with patch.object(main.app.state, 'model_warmup', future, create=True), patch.object(main.app.state, 'database_initialized', True, create=True):
            self.assertEqual(main.ready().status_code, 503)
            future.set_result(None)
            self.assertEqual(main.ready().status_code, 200)
            main.app.state.database_initialized = False
            self.assertEqual(main.ready().status_code, 503)

    def test_failed_warmup_is_not_ready(self):
        future = Future()
        future.set_exception(RuntimeError('failed'))
        with patch.object(main.app.state, 'model_warmup', future, create=True), patch.object(main.app.state, 'database_initialized', True, create=True):
            self.assertEqual(main.ready().status_code, 503)

    def test_restart_requeues_only_interrupted_documents_and_preserves_pages(self):
        db = MemoryDB()
        db.rows['documents'] = [dict(id=s, status=s, ingestion={'pages':[{'page':1, 'text':'Saved'}], 'error':'old'})
                                for s in ['queued', 'processing', 'ready', 'partial', 'failed']]
        with patch.object(main, 'get_supabase', return_value=db):
            recovered = main.recover_interrupted()
        self.assertEqual(set(recovered), {'queued', 'processing'})
        self.assertEqual(db.rows['documents'][1]['status'], 'queued')
        self.assertEqual(db.rows['documents'][1]['ingestion'], {'pages':[{'page':1, 'text':'Saved'}]})
        self.assertEqual(db.rows['documents'][2]['status'], 'ready')
