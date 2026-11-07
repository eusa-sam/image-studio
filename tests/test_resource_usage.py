import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from backend.app.resource_usage import MemorySamples, process_memory, record_resources


class ResourceUsageTests(unittest.TestCase):
    def test_samples_keep_job_peak_not_lifetime_and_failures_are_explicit(self):
        samples=MemorySamples()
        with patch('backend.app.resource_usage.process_memory',side_effect=[
            {'working_set_bytes':10,'private_commit_bytes':20},
            {'working_set_bytes':30,'private_commit_bytes':25},
            OSError('unavailable'),{'working_set_bytes':5,'private_commit_bytes':15}]):
            for _ in range(4):samples.observe()
        row=samples.result()
        self.assertEqual(row['samples'],3)
        self.assertEqual(row['peak'],{'working_set_bytes':30,'private_commit_bytes':25})
        self.assertEqual(row['start']['working_set_bytes'],10)
        self.assertEqual(row['end']['working_set_bytes'],5)
        self.assertEqual(row['error'],'unavailable')

    def test_report_and_exception_preserve_original_behavior(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'request.json';report=Path(folder)/'report.json'
            @record_resources
            def successful(request):
                report.write_text('{"model_sha256":"unchanged"}')
                return 42
            self.assertEqual(successful(path),42)
            saved=json.loads(report.read_text())
            self.assertEqual(saved['model_sha256'],'unchanged')
            self.assertGreaterEqual(saved['resources']['ram']['samples'],2)
            @record_resources
            def failed(request):raise ValueError('inference failed')
            with self.assertRaisesRegex(ValueError,'inference failed'):failed(path)
            self.assertTrue((Path(folder)/'resources.json').exists())

    @unittest.skipUnless(os.name=='nt','Windows process counters')
    def test_real_windows_counters(self):
        row=process_memory()
        self.assertGreater(row['working_set_bytes'],0)
        self.assertGreater(row['private_commit_bytes'],0)
