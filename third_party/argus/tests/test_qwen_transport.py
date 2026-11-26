"""Transport checks without dependencies, credentials or paid API calls."""
import ast
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch


class QwenTransportTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1] / 'label' / 'harness.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        names = {'is_qwen_model', 'is_qwen_key', 'is_openrouter_key', '_call_qwen', 'get_keys'}
        module = ast.Module(body=[node for node in tree.body
                                 if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
        self.post = Mock(return_value={'usage': {'prompt_tokens': 10, 'completion_tokens': 5}})
        self.ns = {'os': os, '_post': self.post}
        exec(compile(module, str(source), 'exec'), self.ns)

    def test_xmsxb_gateway_keeps_images_and_uses_requested_model(self):
        content = [{'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,AA=='}}]
        with patch.dict(os.environ, {'XMSXB_API_KEY': 'test-gateway-key'}, clear=True):
            response = self.ns['_call_qwen'](content, 'qwen3.6-plus', 'test-gateway-key', 16000, 120)
            self.assertEqual(self.ns['get_keys']('qwen3.6-plus'), ['test-gateway-key'])
        args, _ = self.post.call_args
        self.assertEqual(args[0], 'https://zyapi.xmsxb.com/v1/chat/completions')
        self.assertEqual(args[1]['model'], 'qwen3.6-plus')
        self.assertEqual(args[1]['messages'][0]['content'], content)
        self.assertNotIn('chat_template_kwargs', args[1])
        self.assertEqual(response['provider'], 'XMSXB')
        self.assertNotIn('est_cost_cny', response['usage'])

    def test_gateway_model_never_falls_back_to_ali_credentials(self):
        with patch.dict(os.environ, {'DASHSCOPE_API_KEY': 'old-key'}, clear=True):
            self.assertEqual(self.ns['get_keys']('qwen3.6-plus'), [])
            with self.assertRaises(ValueError):
                self.ns['_call_qwen']([], 'qwen3.6-plus', 'old-key', 100, 120)
        self.post.assert_not_called()

    def test_qwen_request_and_unknown_cost(self):
        with patch.dict(os.environ, {'DASHSCOPE_API_KEY': 'test-key',
                                    'DASHSCOPE_BASE_URL': 'https://example.test/compatible-mode/v1'}, clear=True):
            response = self.ns['_call_qwen']([], 'qwen3.8-27b', 'test-key', 4096, 120, attempts=1)
        args, kwargs = self.post.call_args
        self.assertEqual(args[0], 'https://example.test/compatible-mode/v1/chat/completions')
        self.assertEqual(args[1]['model'], 'qwen3.8-27b')
        self.assertNotIn('reasoning_effort', args[1])
        self.assertEqual(kwargs['attempts'], 1)
        self.assertEqual(response['provider'], 'DashScope')
        self.assertNotIn('list_cost', response['usage'])
        self.assertAlmostEqual(response['usage']['est_cost_cny'], 0.00009)

    def test_other_models_and_missing_usage_have_no_fabricated_price(self):
        with patch.dict(os.environ, {'DASHSCOPE_API_KEY': 'test-key',
                                    'DASHSCOPE_BASE_URL': 'https://example.test/v1'}, clear=True):
            response = self.ns['_call_qwen']([], 'qwen-other', 'test-key', 4096, 120)
            self.assertNotIn('est_cost_cny', response['usage'])
            self.post.return_value = {'usage': {'prompt_tokens': 10}}
            response = self.ns['_call_qwen']([], 'qwen3.8-27b', 'test-key', 4096, 120)
            self.assertNotIn('est_cost_cny', response['usage'])

    def test_qwen_keys_are_selected_even_with_other_providers(self):
        with patch.dict(os.environ, {'DASHSCOPE_API_KEY': 'qwen-key', 'OPENAI_API_KEY': 'other'}, clear=True):
            self.assertEqual(self.ns['get_keys']('qwen3.8-27b'), ['qwen-key'])

    def test_wrong_model_or_missing_endpoint_never_dispatches(self):
        with patch.dict(os.environ, {'DASHSCOPE_API_KEY': 'test-key'}, clear=True):
            for model in ('openai/gpt-6-sol', 'qwen3.8-27b'):
                with self.assertRaises(ValueError):
                    self.ns['_call_qwen']([], model, 'test-key', 4096, 120)
        self.post.assert_not_called()

    def test_budgeted_transport_reserves_and_sends_once(self):
        from review import cny_budget
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = root / 'budget.json'
            cny_budget.initialize(ledger, source=root / 'video.mp4', model='qwen3.8-27b',
                                  cap=0.6, reserve=0.15, prior_paid=False)
            with patch.dict(os.environ, {'DASHSCOPE_API_KEY': 'test-key',
                                        'DASHSCOPE_BASE_URL': 'https://example.test/v1',
                                        cny_budget.ENV: str(ledger)}, clear=True):
                self.ns['_call_qwen']([], 'qwen3.8-27b', 'test-key', 4096, 120, attempts=6)
            self.assertEqual(self.post.call_args.kwargs['attempts'], 1)
            import json
            self.assertEqual(json.loads(ledger.read_text())['calls'][0]['status'], 'settled')


if __name__ == '__main__':
    unittest.main()
