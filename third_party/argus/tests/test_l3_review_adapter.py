import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from review.L3.run import plan, main, run_one


class L3ReviewTests(unittest.TestCase):
    def test_successful_existing_annotation_deletes_work_without_model_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session, video = self.fixture(root)
            with patch.dict('os.environ', {'ARGUS_L3_WORK_ROOT': str(root / 'work')}):
                item = plan(video, [], structured_only=True)
            final = session / 'annotation/episode_000000_structured.json'
            final.parent.mkdir()
            final.write_text('{}')
            job = Path(item['job'])
            job.mkdir(parents=True)
            (job / 'debug.log').write_text('discard')
            with patch('review.base.run.validate'), patch('review.L3.run._run_with_workfiles') as model:
                self.assertEqual(run_one(item)['status'], 'done')
                model.assert_not_called()
            self.assertFalse((root / 'work').exists())
            self.assertEqual(final.read_text(), '{}')

    def test_external_work_publishes_only_final_json(self):
        from review.L3.run import publish_structured
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session, video = self.fixture(root)
            with patch.dict('os.environ', {'ARGUS_L3_WORK_ROOT': str(root / 'work')}):
                item = plan(video, [], structured_only=True)
            annotation = Path(item['job']).parent.parent
            annotation.mkdir(parents=True)
            (annotation / 'episode_000000_structured.json').write_text('{}')
            (annotation / 'debug.log').write_text('work only')
            publish_structured(item)
            self.assertEqual([p.name for p in (session / 'annotation').iterdir()],
                             ['episode_000000_structured.json'])

    def test_duplicate_discovery_with_force_dispatches_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            session, video = self.fixture(Path(tmp))
            with patch.object(Path, 'rglob', return_value=iter([video, video])), \
                 patch('review.L3.run.run_one', return_value=dict(status='done')) as dispatch:
                self.assertEqual(main(['--source', str(session), '--force']), 0)
            dispatch.assert_called_once()

    def fixture(self, root, index=0):
        session = root / 'batch_0000001/L3-001-2026-10-19-18-10-53'
        video = session / 'videos/chunk-000/observation.images.headband_left_rgb_rectified' / f'episode_{index:06d}.mp4'
        video.parent.mkdir(parents=True, exist_ok=True)
        video.touch()
        (session / 'meta').mkdir(exist_ok=True)
        (session / 'meta/collect_info.jsonl').write_text(json.dumps(dict(
            episode_index=index, scene='table', task_name='weigh cups', notes='planned steps', objects=[])), encoding='utf-8')
        qc = dict(episode_index=index, qc_report=dict(streams=dict(marker=dict(stats=dict(markers=dict(
            items=[dict(name='EXEC_START', tag='EXEC_START', t_offset=10.193434)]))))))
        (session / 'meta/qc_reports.jsonl').write_text(json.dumps(qc), encoding='utf-8')
        return session, video

    def test_plan_context_and_no_model_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            session, video = self.fixture(Path(tmp))
            item = plan(video, ['--free'])
            self.assertIn('--free', item['command'])
            self.assertIn('--annotation-start-s', item['command'])
            self.assertEqual(item['identity']['execution']['start_s'], 10.193434)
            self.assertIn('weigh cups', item['command'][-1])
            self.assertIn('not verified observations', item['command'][-1])
            self.assertFalse(Path(item['job']).is_relative_to(session))
            self.assertEqual(Path(item['job']).name, 'chunk-000_episode_000000')
            with patch('review.L3.run.subprocess.run') as backend:
                self.assertEqual(main(['--source', str(session), '--dry-run']), 0)
                backend.assert_not_called()
            self.assertFalse((session / 'annotation').exists())

    def test_run_and_changed_context_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            session, video = self.fixture(Path(tmp))
            item = plan(video, ['--free'])
            with patch('review.L3.run.subprocess.run', return_value=SimpleNamespace(returncode=0)) as backend:
                self.assertEqual(run_one(item)['status'], 'done')
                self.assertEqual(backend.call_args.args[0], item['command'])
            path = session / 'meta/collect_info.jsonl'
            row = json.loads(path.read_text()); row['notes'] = 'changed'
            path.write_text(json.dumps(row))
            with patch('review.L3.run.subprocess.run', return_value=SimpleNamespace(returncode=0)) as backend:
                self.assertEqual(run_one(plan(video, []))['status'], 'failed')
                backend.assert_not_called()

    def test_parent_cleanup_handoff_preserves_adapter_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            session, video = self.fixture(Path(tmp))
            item = plan(video, [])
            job = Path(item['job'])
            job.mkdir(parents=True)
            (job / 'old_failed_request.json').write_text('{}')
            def dispatch(command, **kwargs):
                self.assertIn('--parent-prepared-job', command)
                self.assertFalse((job / 'old_failed_request.json').exists())
                self.assertTrue((job / '.l3_review.lock').exists())
                return SimpleNamespace(returncode=0)
            with patch('review.L3.run.subprocess.run', side_effect=dispatch):
                self.assertEqual(run_one(item)['status'], 'done')
            self.assertFalse((job / '.l3_review.lock').exists())

    def test_episode_selection_and_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            session, first = self.fixture(Path(tmp))
            _, second = self.fixture(Path(tmp), 1)
            with self.assertRaisesRegex(ValueError, 'Expected one collect_info'):
                plan(first, [])
            row = json.loads((session / 'meta/collect_info.jsonl').read_text())
            row0 = dict(row, episode_index=0, task_name='first')
            (session / 'meta/collect_info.jsonl').write_text(json.dumps(row0)+'\n'+json.dumps(row))
            qcpath = session / 'meta/qc_reports.jsonl'
            qc = json.loads(qcpath.read_text())
            qcpath.write_text(json.dumps(dict(qc, episode_index=0))+'\n'+json.dumps(qc))
            a,b = plan(first, []),plan(second, [])
            self.assertNotEqual(a['job'], b['job'])
            self.assertEqual(a['identity']['capture']['context']['task_name'], 'first')

    def test_missing_duplicate_and_invalid_execution_marker_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            session, video = self.fixture(Path(tmp))
            path = session / 'meta/qc_reports.jsonl'
            row = json.loads(path.read_text())
            markers = row['qc_report']['streams']['marker']['stats']['markers']
            for items in ([], [dict(name='EXEC_START', t_offset=-1)],
                          [dict(name='EXEC_START', t_offset=1)] * 2):
                markers['items'] = items
                path.write_text(json.dumps(row))
                with self.assertRaises(ValueError):
                    plan(video, [])


if __name__ == '__main__':
    unittest.main()
