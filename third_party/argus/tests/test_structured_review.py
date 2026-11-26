import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from review import structured


class StructuredReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.job = Path(self.temp.name)
        self.source = self.job / 'run_final/out/episode_000000.json'
        self.source.parent.mkdir(parents=True)
        self.source.write_text(json.dumps({'parse_ok': True, 'labels': {
            'scene': {'setting': 'kitchen', 'objects': []},
            'timeline': [{'start_s': 0, 'end_s': 1, 'action': 'reach'}]}}))
        self.env = patch.dict(os.environ, {'DASHSCOPE_API_KEY': 'test',
                                          'DASHSCOPE_BASE_URL': 'https://example.test/v1'}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_output_saved_and_reused_without_another_call(self):
        def fake(prepared, model, endpoint, key, timeout, *, receipt, result_validator=None, on_attempt=None):
            receipt.update(provider='dashscope', usage={'prompt_tokens': 100, 'completion_tokens': 30})
            return {**prepared, 'actions': {}, 'scene_classification': {}}
        with patch.object(structured.extraction, 'call_qwen', side_effect=fake) as call:
            outputs = structured.run(self.job, 'qwen3.8-27b')
            structured.run(self.job, 'qwen3.8-27b')
            self.assertEqual(call.call_count, 1)
        self.assertEqual(outputs, [self.job / 'episode_000000_structured.json'])
        claim = self.job / '.episode_000000_structured.claim.json'
        self.assertTrue(claim.exists())
        saved = json.loads(outputs[0].read_text())
        self.assertNotIn('extraction', saved)
        meta = json.loads(claim.read_text())
        self.assertEqual(meta['model'], 'qwen3.8-27b')
        self.assertEqual(meta['status'], 'complete')
        self.assertEqual(meta['usage']['prompt_tokens'], 100)

    def test_unresolved_call_never_retries_automatically(self):
        with patch.object(structured.extraction, 'call_qwen', side_effect=TimeoutError('timeout')) as call:
            with self.assertRaises(TimeoutError):
                structured.run(self.job, 'qwen3.8-27b')
            with self.assertRaisesRegex(RuntimeError, 'may have been billed'):
                structured.run(self.job, 'qwen3.8-27b')
            self.assertEqual(call.call_count, 1)

    def test_separate_output_directory_keeps_claim_in_job_and_reuses_result(self):
        output_dir = self.job / 'annotation'
        output_dir.mkdir()
        with patch.object(structured.extraction, 'call_qwen',
                          side_effect=lambda p, m, *a, **kw: {**p}) as call:
            output = structured.run(self.job, 'qwen3.8-27b', output_dir=output_dir)[0]
            structured.run(self.job, 'qwen3.8-27b', output_dir=output_dir)
            self.assertEqual(call.call_count, 1)
        self.assertEqual(output, output_dir / 'episode_000000_structured.json')
        self.assertTrue((self.job / '.episode_000000_structured.claim.json').exists())
        self.assertEqual(list(output_dir.iterdir()), [output])

    def test_changed_source_does_not_overwrite(self):
        with patch.object(structured.extraction, 'call_qwen',
                          side_effect=lambda p, m, *a, **kw: {**p}) as call:
            structured.run(self.job, 'qwen3.8-27b')
            self.source.write_text(self.source.read_text() + ' ')
            with self.assertRaises(FileExistsError):
                structured.run(self.job, 'qwen3.8-27b')
            self.assertEqual(call.call_count, 1)

    def test_changed_output_is_not_reused_or_redispatched(self):
        with patch.object(structured.extraction, 'call_qwen',
                          side_effect=lambda p, m, *a, **kw: {**p}) as call:
            output = structured.run(self.job, 'qwen3.8-27b')[0]
            output.write_text(output.read_text() + ' ')
            with self.assertRaisesRegex(FileExistsError, 'unverifiable'):
                structured.run(self.job, 'qwen3.8-27b')
            self.assertEqual(call.call_count, 1)

    def test_output_with_incomplete_claim_is_not_reused_or_redispatched(self):
        with patch.object(structured.extraction, 'call_qwen',
                          side_effect=lambda p, m, *a, **kw: {**p}) as call:
            structured.run(self.job, 'qwen3.8-27b')
            claim = self.job / '.episode_000000_structured.claim.json'
            meta = json.loads(claim.read_text())
            meta['status'] = 'dispatching'
            claim.write_text(json.dumps(meta))
            with self.assertRaisesRegex(FileExistsError, 'unverifiable'):
                structured.run(self.job, 'qwen3.8-27b')
            self.assertEqual(call.call_count, 1)

    def test_capture_context_saved_separately_and_reused(self):
        from review import capture_context
        snapshot = {'identity': {'file': 'setup_config.yaml', 'sha256': 'abc'},
                    'context': {'task': '烧水', 'scene': {'name': '厨房'}, 'objects': []}}
        video = self.job / 'video.mp4'
        with patch.object(capture_context, 'snapshot_for_video', return_value=snapshot), \
             patch.object(structured.extraction, 'call_qwen', side_effect=lambda p, m, *a, **kw: {**p}) as call:
            output = structured.run(self.job, 'qwen3.8-27b', capture_snapshot=snapshot, capture_video=video)[0]
            structured.run(self.job, 'qwen3.8-27b', capture_snapshot=snapshot, capture_video=video)
            self.assertEqual(call.call_count, 1)
        self.assertEqual(json.loads(output.read_text(encoding='utf-8'))['capture_context'], snapshot['context'])
        meta = json.loads((self.job / '.episode_000000_structured.claim.json').read_text())
        self.assertEqual(meta['capture_config'], snapshot['identity'])

    def test_capture_change_during_model_call_blocks_output_and_retry(self):
        from review import capture_context
        before = {'identity': {'file': 'setup_config.yaml', 'sha256': 'a'}, 'context': {'task': 'one'}}
        after = {'identity': {'file': 'setup_config.yaml', 'sha256': 'b'}, 'context': {'task': 'two'}}
        current = [before]
        def fake(*args, **kwargs):
            current[0] = after
            return {'subtask': []}
        with patch.object(capture_context, 'snapshot_for_video', side_effect=lambda _: current[0]), \
             patch.object(structured.extraction, 'call_qwen', side_effect=fake) as call:
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                structured.run(self.job, 'qwen3.8-27b', capture_snapshot=before, capture_video=self.job / 'video.mp4')
            self.assertFalse((self.job / 'episode_000000_structured.json').exists())
            current[0] = before
            with self.assertRaisesRegex(RuntimeError, 'may have been billed'):
                structured.run(self.job, 'qwen3.8-27b', capture_snapshot=before, capture_video=self.job / 'video.mp4')
            self.assertEqual(call.call_count, 1)

    def test_missing_output_does_not_dispatch(self):
        self.source.unlink()
        with patch.object(structured.extraction, 'call_qwen') as call:
            with self.assertRaisesRegex(ValueError, 'No completed episode'):
                structured.run(self.job, 'qwen3.8-27b')
            call.assert_not_called()

    def test_budget_refusal_removes_undispatched_claim(self):
        from review.cny_budget import BudgetExceeded
        with patch.object(structured.extraction, 'call_qwen', side_effect=BudgetExceeded('not sent')):
            with self.assertRaises(BudgetExceeded):
                structured.run(self.job, 'qwen3.8-27b')
        self.assertFalse((self.job / '.episode_000000_structured.claim.json').exists())


if __name__ == '__main__':
    unittest.main()
