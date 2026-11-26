"""Exercise review orchestration without external dependencies or paid calls."""
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from review import __main__ as review


class NoBoardTests(unittest.TestCase):
    def run_review(self, free, qwen=False, budget=False, capture=False, l1_default=False, structured_model=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = (root / 'cp_001/videos/chunk-000/observation.images.head_left_rectified/episode_000000.mp4'
                     if l1_default else root / 'video.mp4')
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b'input')
            annotation = video.parents[3] / 'annotation' if l1_default else root / 'annotation'
            job = annotation / 'argus' if l1_default else annotation
            steps, environments = [], {}
            board = types.ModuleType('board')
            board.clips = types.SimpleNamespace()
            checks = types.ModuleType('checks')
            checks.EXIT_PARTIAL = 3
            checks.timebase = types.SimpleNamespace()
            prepare = types.ModuleType('prepare')
            prepare.formats = types.SimpleNamespace()
            dictionary = types.SimpleNamespace(prepare=lambda *a, **kw: None,
                                               remaining_label_budget=lambda *a: 10,
                                               dictionary_spend=lambda *a: {'complete': True, 'cost_usd': 0, 'reserved_usd': 0},
                                               label_spend=lambda *a: 0,
                                               label_spend_complete=lambda *a: True)

            def stitch(job, eps, long_eps, out):
                out.mkdir(parents=True, exist_ok=True)
                (out / 'episode_000000.json').write_text('{"parse_ok": true}')
                return {'incomplete': [], 'unlabelled': []}

            pieces = types.SimpleNamespace(write_units=lambda *a: {}, stitch_run=stitch)
            rep = {'episodes': [{'episode_id': 'episode_000000', 'state_kind': 'none'}]}
            preparation = {'policies': {}, 'source': {'test': 1}}
            argv = ['review', '--data', str(video), '--rig', 'ego_head', '--no-board']
            if not l1_default:
                argv += ['--out', str(job)]
            if free:
                argv.append('--free')
            if qwen:
                argv += ['--model', 'qwen3.8-27b', '--cap', '0']
            if structured_model:
                argv += ['--structured-model', structured_model, '--structured-timeout', '45']
            if budget:
                argv += ['--concurrency', '1', '--max-tokens', '16384', '--estimated-cap-cny', '0.6']
            import label
            from review import structured, capture_context
            snapshot = {'identity': {'file': 'setup_config.yaml', 'sha256': 'abc'},
                        'context': {'task': '烧水', 'scene': {'name': '厨房灶台'}, 'objects': []}} if capture else None
            def step(job, name, command, env, **kwargs):
                steps.append(name)
                environments[name] = dict(env)
                return 0
            def extract_result(*args, **kwargs):
                environments['structured'] = dict(os.environ)
                return [kwargs.get('output_dir', job) / 'episode_000000_structured.json']
            with patch.dict(sys.modules, {'board': board, 'checks': checks, 'prepare': prepare}), \
                 patch.dict(os.environ, {'DASHSCOPE_API_KEY': 'test', 'DASHSCOPE_BASE_URL': 'https://example.test'}), \
                 patch.object(label, 'pieces', pieces, create=True), \
                 patch.object(label, 'dictionary_stage', dictionary, create=True), \
                 patch.object(capture_context, 'snapshot_for_video', return_value=snapshot), \
                 patch.object(review, 'verified_preparation', return_value=(rep, preparation)) as verify, \
                 patch.object(review, 'preparation_checkpoint'), \
                 patch.object(review, 'source_identity', return_value={'test': 1}), \
                 patch.object(review, 'run_step', side_effect=step), \
                 patch.object(review, 'build_review_board', side_effect=AssertionError('board invoked')), \
                 patch.object(structured, 'run', side_effect=extract_result) as extract, \
                 patch.object(sys, 'argv', argv):
                self.assertEqual(review.main(), 0)
                if qwen and not free:
                    expected = {'output_dir': annotation} if l1_default else {}
                    extract.assert_called_once_with(job, structured_model or 'qwen3.8-27b', timeout=45 if structured_model else 120,
                                                    capture_snapshot=snapshot, capture_video=video, **expected)
                else:
                    extract.assert_not_called()
                if capture:
                    self.assertEqual(verify.call_args.kwargs['capture_identity'], snapshot['identity'])
                    self.assertIn('烧水', verify.call_args.args[4]['dataset_note'])
            self.assertIn('checks_capture', steps)
            self.assertIn('dry_run', steps)
            self.assertNotIn('clips', steps)
            for name in ('clips', 'qa', 'manifest.json', 'BUILT.json'):
                self.assertFalse((job / name).exists())
            if free:
                self.assertNotIn('label', steps)
                self.assertFalse((job / 'run_final').exists())
            else:
                self.assertIn('label', steps)
                self.assertTrue(json.loads((job / 'run_final/out/episode_000000.json').read_text())['parse_ok'])
                self.assertEqual(json.loads((job / 'run_final/run.json').read_text())['status'], 'done')
            if budget:
                from review.cny_budget import ENV
                self.assertEqual(environments['label'][ENV], str(job / 'estimated_budget_cny.json'))
                self.assertEqual(environments['structured'][ENV], environments['label'][ENV])

    def test_paid_review_stitches_labels_without_board(self):
        self.run_review(False)

    def test_free_review_skips_board_without_calling_model(self):
        self.run_review(True)

    def test_qwen_review_automatically_extracts(self):
        self.run_review(False, qwen=True)

    def test_separate_structured_model_and_timeout(self):
        self.run_review(False, qwen=True, structured_model='qwen3.6-plus')

    def test_qwen_free_review_never_extracts(self):
        self.run_review(True, qwen=True)

    def test_qwen_budgeted_review(self):
        self.run_review(False, qwen=True, budget=True)

    def test_capture_context_is_forwarded_to_video_and_structured_stages(self):
        self.run_review(False, qwen=True, budget=True, capture=True)

    def test_default_l1_job_and_structured_paths(self):
        self.run_review(False, qwen=True, budget=True, capture=True, l1_default=True)

    def test_existing_unverified_structured_result_cleared_before_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'cp_001'
            video = root / 'videos/chunk-000/observation.images.head_left_rectified/episode_000000.mp4'
            video.parent.mkdir(parents=True)
            video.write_bytes(b'input')
            annotation = root / 'annotation'
            annotation.mkdir()
            (annotation / 'episode_000000_structured.json').write_text('{}')
            argv = ['review', '--data', str(video), '--rig', 'ego_head', '--model', 'qwen3.8-27b', '--cap', '0']
            board = types.ModuleType('board')
            board.clips = types.SimpleNamespace()
            checks = types.ModuleType('checks')
            checks.EXIT_PARTIAL = 3
            checks.timebase = types.SimpleNamespace()
            prepare = types.ModuleType('prepare')
            prepare.formats = types.SimpleNamespace()
            import label
            with patch.object(sys, 'argv', argv), \
                 patch.dict(os.environ, {'DASHSCOPE_API_KEY': 'test', 'DASHSCOPE_BASE_URL': 'https://example.test'}), \
                 patch.dict(sys.modules, {'board': board, 'checks': checks, 'prepare': prepare}), \
                 patch.object(label, 'pieces', types.SimpleNamespace(), create=True), \
                 patch.object(label, 'dictionary_stage', types.SimpleNamespace(), create=True), \
                 patch.object(review, 'verified_preparation', side_effect=RuntimeError('stop before paid calls')) as verify:
                with self.assertRaisesRegex(RuntimeError, 'stop before paid calls'):
                    review.main()
                verify.assert_called_once()
                self.assertFalse((annotation / 'episode_000000_structured.json').exists())


if __name__ == '__main__':
    unittest.main()
