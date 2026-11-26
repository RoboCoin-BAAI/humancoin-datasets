import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from review import __main__ as review, batch


class BatchReviewTests(unittest.TestCase):
    def test_duplicate_discovery_paths_dispatch_once(self):
        video = self.video('cp_001')
        with patch.object(Path, 'rglob', return_value=iter([video, video])):
            groups = batch.discover(self.root)
        self.assertEqual(list(groups.values()), [[video]])
        with patch.object(Path, 'rglob', return_value=iter([video, video])), \
             patch.object(batch.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as dispatch:
            self.assertEqual(batch.run(self.root, 1, [], {}), 0)
        self.assertEqual(dispatch.call_count, 1)

    def test_session_allowlist_excludes_existing_unselected_videos(self):
        a = self.video('cp_001')
        self.video('cp_002')
        allowlist = self.root / 'selected.json'
        allowlist.write_text(json.dumps([str(a.parents[3])]))
        with patch.object(batch.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as dispatch:
            self.assertEqual(batch.run(self.root, 2, [], {}, sessions_file=allowlist), 0)
        self.assertEqual(dispatch.call_count, 1)
        self.assertEqual(dispatch.call_args.args[0][-1], str(a))
        self.assertEqual(batch.child_arguments(['--sessions-file', 'selected.json', '--workers=2', '--rig', 'ego_head']),
                         ['--rig', 'ego_head'])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def video(self, session, episode='episode_000000', camera=batch.CAMERA):
        path = self.root / session / 'videos' / 'chunk-000' / camera / f'{episode}.mp4'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'video')
        return path

    def test_discovery_accepts_batch_and_one_session_and_only_left_rectified(self):
        a = self.video('cp_001')
        b = self.video('cp_002')
        self.video('cp_002', camera='observation.images.head_right_rectified')
        self.assertEqual(set(batch.discover(self.root)), {a.parents[3] / 'annotation', b.parents[3] / 'annotation'})
        self.assertEqual(list(batch.discover(a.parents[3]).values()), [[a]])

    def test_forwards_options_without_recursive_batch_arguments(self):
        args = ['--source', '/input', '--workers=8', '--rig', 'ego_head', '--cap', '0',
                '--estimated-cap-cny', '0.6']
        self.assertEqual(batch.child_arguments(args), ['--rig', 'ego_head', '--cap', '0',
                                                       '--estimated-cap-cny', '0.6'])

    def test_two_sessions_run_concurrently_in_separate_processes(self):
        a = self.video('cp_001')
        b = self.video('cp_002')
        barrier = threading.Barrier(2, timeout=5)
        commands, guard = [], threading.Lock()

        def fake_run(command, **kwargs):
            with guard:
                commands.append((command, kwargs))
            barrier.wait()
            return subprocess.CompletedProcess(command, 0)

        with patch.object(batch.subprocess, 'run', side_effect=fake_run):
            self.assertEqual(batch.run(self.root, 2, ['--cap', '0', '--estimated-cap-cny', '0.6'], {'test': 'env'}), 0)
        self.assertEqual({cmd[-1] for cmd, _ in commands}, {str(a), str(b)})
        for cmd, kwargs in commands:
            self.assertEqual(cmd[:3], [sys.executable, '-m', 'review'])
            self.assertIn('--estimated-cap-cny', cmd)
            self.assertEqual(kwargs['env'], {'test': 'env'})
        for video in (a, b):
            annotation = video.parents[3] / 'annotation'
            self.assertTrue((annotation / 'argus' / '.argus_batch.log').exists())
            self.assertFalse((annotation / 'argus' / '.argus_batch.lock').exists())

    def test_multiple_videos_same_session_refused_without_dispatch(self):
        self.video('cp_001')
        self.video('cp_001', 'episode_000001')
        with patch.object(batch.subprocess, 'run') as dispatch:
            self.assertEqual(batch.run(self.root, 8, [], {}), 1)
            dispatch.assert_not_called()

    def test_existing_lock_and_failed_child_are_reported(self):
        video = self.video('cp_001')
        annotation = video.parents[3] / 'annotation'
        (annotation / 'argus').mkdir(parents=True)
        lock = annotation / 'argus' / '.argus_batch.lock'
        lock.write_text('already running')
        with patch.object(batch.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as dispatch:
            self.assertEqual(batch.run(self.root, 1, [], {}), 0)
            dispatch.assert_called_once()
        self.assertFalse(lock.exists())
        with patch.object(batch.subprocess, 'run', return_value=subprocess.CompletedProcess([], 1)):
            self.assertEqual(batch.run(self.root, 1, [], {}), 1)
        self.assertFalse(lock.exists())

    def test_review_cli_starts_batch_with_per_video_budget_options(self):
        argv = ['review', '--source', str(self.root), '--workers', '8', '--rig', 'ego_head', '--no-board',
                '--model', 'qwen3.8-27b', '--cap', '0', '--concurrency', '1', '--max-tokens', '16384',
                '--estimated-cap-cny', '0.6']
        with patch.object(sys, 'argv', argv), \
             patch.dict(os.environ, {'DASHSCOPE_API_KEY': 'test', 'DASHSCOPE_BASE_URL': 'https://example.test'}), \
             patch.object(batch, 'run', return_value=0) as run:
            self.assertEqual(review.main(), 0)
        self.assertEqual(run.call_args.args[0:2], (self.root, 8))
        self.assertIn('--estimated-cap-cny', run.call_args.args[2])
        self.assertNotIn('--source', run.call_args.args[2])


if __name__ == '__main__':
    unittest.main()
