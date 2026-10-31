import unittest
from unittest.mock import Mock,patch
from app.agents import graph
from app.services import provider_limits


class WaitReportingTests(unittest.TestCase):
    def test_nested_observers_and_pipeline_report_each_wait_once(self):
        outer=[]
        def invoke(*args):
            provider_limits._wait_callback.get()(2.5)
            return {'pipeline_log':[]}
        with provider_limits.observe_waits(outer.append), patch.object(graph,'get_graph',return_value=Mock(invoke=invoke)):
            result=graph.run_agent('question',['doc'])
        self.assertEqual(outer,[2.5])
        self.assertIn('2.5s',result['pipeline_log'][-1]['detail'])
        self.assertIsNone(provider_limits._wait_callback.get())
