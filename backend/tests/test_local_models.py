import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock
from app.services.local_models import cached_model, prefer_local


class LocalModelTests(unittest.TestCase):
    def test_concurrent_misses_initialize_once(self):
        started, release = threading.Event(), threading.Event()
        calls = []
        @cached_model
        def load():
            calls.append(1)
            started.set()
            self.assertTrue(release.wait(2))
            return object()
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(load)
            self.assertTrue(started.wait(2))
            second = pool.submit(load)
            release.set()
            self.assertIs(first.result(), second.result())
        self.assertEqual(len(calls), 1)

    def test_cached_weights_need_no_network_load(self):
        factory = Mock(return_value='model')
        self.assertEqual(prefer_local(factory, 'name'), 'model')
        factory.assert_called_once_with('name', local_files_only=True)

    def test_missing_weights_can_download_but_other_errors_surface(self):
        factory = Mock(side_effect=[OSError('not cached'), 'model'])
        self.assertEqual(prefer_local(factory, 'name'), 'model')
        self.assertEqual(factory.call_count, 2)
        factory = Mock(side_effect=ValueError('broken model'))
        with self.assertRaises(ValueError):
            prefer_local(factory, 'name')
        self.assertEqual(factory.call_count, 1)
