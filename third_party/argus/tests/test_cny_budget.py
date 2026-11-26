import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import MagicMock, Mock, patch

from review import cny_budget


class CnyBudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'estimated_budget_cny.json'
        self.source = Path(self.tmp.name) / 'video.mp4'
        self.source.write_bytes(b'video')
        cny_budget.initialize(self.path, source=self.source, model='qwen3.8-27b', cap=0.6,
                              reserve=0.15, prior_paid=False)
        self.environment = patch.dict(os.environ, {cny_budget.ENV: str(self.path)})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.body = {'model': 'qwen3.8-27b', 'messages': [{'role': 'user', 'content': 'hello'}],
                     'max_tokens': 1000}

    def ledger(self):
        return json.loads(self.path.read_text(encoding='utf-8'))

    def test_video_and_postprocess_share_actual_usage_and_reservation(self):
        response = {'usage': {'prompt_tokens': 100, 'completion_tokens': 200}}
        cny_budget.execute(self.body, 'video', lambda: response)
        cny_budget.execute(self.body, 'postprocess', lambda: response)
        calls = self.ledger()['calls']
        self.assertEqual([c['stage'] for c in calls], ['video', 'postprocess'])
        self.assertEqual([c['cost_micro_cny'] for c in calls], [2700, 2700])
        self.assertEqual([c['status'] for c in calls], ['settled', 'settled'])

    def test_refuses_before_dispatch_when_reserve_exceeds_remaining(self):
        self.body['max_tokens'] = 40000
        dispatch = Mock()
        with self.assertRaises(cny_budget.BudgetExceeded):
            cny_budget.execute(self.body, 'video', dispatch)
        dispatch.assert_not_called()
        self.assertEqual(self.ledger()['calls'], [])
        # The same request can use the reserved amount during the postprocess stage.
        dispatch.return_value = {'usage': {'prompt_tokens': 1, 'completion_tokens': 1}}
        cny_budget.execute(self.body, 'postprocess', dispatch)
        dispatch.assert_called_once()

    def test_unknown_outcome_blocks_later_calls(self):
        with self.assertRaises(TimeoutError):
            cny_budget.execute(self.body, 'video', Mock(side_effect=TimeoutError()))
        self.assertEqual(self.ledger()['calls'][0]['status'], 'pending')
        dispatch = Mock()
        with self.assertRaisesRegex(RuntimeError, 'unknown outcome'):
            cny_budget.execute(self.body, 'postprocess', dispatch)
        dispatch.assert_not_called()

    def test_missing_usage_blocks_later_calls(self):
        with self.assertRaisesRegex(RuntimeError, 'usage missing'):
            cny_budget.execute(self.body, 'video', lambda: {'choices': []})
        self.assertEqual(self.ledger()['calls'][0]['status'], 'pending')

    def test_prior_paid_and_changed_settings_are_rejected(self):
        other = self.path.parent / 'other.json'
        with self.assertRaisesRegex(ValueError, 'Earlier paid'):
            cny_budget.initialize(other, source=self.source, model='qwen3.8-27b', cap=0.6,
                                  reserve=0.15, prior_paid=True)
        with self.assertRaisesRegex(ValueError, 'different settings'):
            cny_budget.initialize(self.path, source=self.source, model='qwen3.8-27b', cap=0.7,
                                  reserve=0.15, prior_paid=True)

    def test_different_model_and_unsupported_image_are_refused(self):
        dispatch = Mock()
        self.body['model'] = 'other'
        with self.assertRaisesRegex(ValueError, 'model'):
            cny_budget.execute(self.body, 'video', dispatch)
        self.body['model'] = 'qwen3.8-27b'
        self.body['messages'] = [{'content': [{'type': 'image_url', 'image_url': {'url': 'https://x'}}]}]
        with self.assertRaisesRegex(ValueError, 'embedded'):
            cny_budget.execute(self.body, 'video', dispatch)
        dispatch.assert_not_called()

    def test_image_dimensions_contribute_to_input_estimate(self):
        image = types.SimpleNamespace(size=(320, 160))
        opened = MagicMock()
        opened.__enter__.return_value = image
        opened.__exit__.return_value = False
        pil = types.ModuleType('PIL')
        pil.Image = types.SimpleNamespace(open=Mock(return_value=opened))
        body = {'max_tokens': 1000, 'messages': [{'content': [
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,YQ=='}}]}]}
        with patch.dict('sys.modules', {'PIL': pil}):
            prompt, maximum = cny_budget.estimate(body)
        self.assertEqual(prompt, 1024 + len('content') + 2 * (10 * 5 + 2))
        self.assertEqual(maximum, 1000)

    def test_underestimate_does_not_authorize_further_calls_after_overrun(self):
        cny_budget.execute(self.body, 'video', lambda: {'usage': {'prompt_tokens': 200000,
                                                                  'completion_tokens': 1000}})
        dispatch = Mock()
        with self.assertRaises(cny_budget.BudgetExceeded):
            cny_budget.execute(self.body, 'postprocess', dispatch)
        dispatch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
