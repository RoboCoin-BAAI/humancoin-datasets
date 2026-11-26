import json
from pathlib import Path
import tempfile
import unittest

from review import capture_context


class CaptureContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.session = Path(self.temp.name) / 'cp_0001_session_test'
        self.annotation = self.session / 'annotation'
        self.annotation.mkdir(parents=True)
        self.video = self.session / 'videos/chunk-000/observation.images.head_left_rectified/episode_000000.mp4'
        self.video.parent.mkdir(parents=True)
        self.video.write_bytes(b'video')
        self.config = self.annotation / '厨房_烧水_config.yaml'

    def test_no_config_is_optional_and_unrelated_videos_are_ignored(self):
        self.assertIsNone(capture_context.snapshot_for_video(self.video))
        (self.annotation / 'other.yaml').write_text('bad: [', encoding='utf-8')
        self.assertIsNone(capture_context.snapshot_for_video(self.video))
        self.assertIsNone(capture_context.snapshot_for_video(self.session / 'video.mp4'))
        self.assertEqual(capture_context.dataset_note('original', None), 'original')

    def test_reads_relevant_fields_and_marks_plans_as_unverified(self):
        self.config.write_text('''task: 烧水
scene:
  name: 厨房灶台
  notes: 把水壶放到底座
  taskName: 烧水
  opts: {renderer: webgl}
  objects:
    - name: 电热水壶
      color: "#EDEDED"
      material: 金属
      shape: cyl
      dims: {d: 19, h: 23}
      trail: [{x: 1, y: 2}]
      weight: 729.5
''', encoding='utf-8')
        snapshot = capture_context.snapshot_for_video(self.video)
        ctx = snapshot['context']
        self.assertEqual(ctx['task'], '烧水')
        self.assertEqual(ctx['scene']['name'], '厨房灶台')
        self.assertEqual(ctx['scene']['notes'], '把水壶放到底座')
        self.assertEqual(ctx['objects'][0]['dims'], {'d': 19, 'h': 23})
        self.assertNotIn('trail', ctx['objects'][0])
        self.assertNotIn('opts', ctx['scene'])
        note = capture_context.dataset_note('user note', snapshot)
        self.assertIn('user note', note)
        self.assertIn('烧水', note)
        self.assertIn('not verified observations', note)
        self.assertIn('do not infer that a planned step happened', note)
        self.assertEqual(len(snapshot['identity']['sha256']), 64)
        json.dumps(ctx, allow_nan=False)

    def test_multiple_and_malformed_configs_are_refused(self):
        self.config.write_text('scene: {}', encoding='utf-8')
        (self.annotation / 'other_config.yaml').write_text('scene: {}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Multiple'):
            capture_context.snapshot_for_video(self.video)
        (self.annotation / 'other_config.yaml').unlink()
        self.config.write_text('scene: [', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Cannot parse'):
            capture_context.snapshot_for_video(self.video)
        self.config.write_text('scene: {objects: 5}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'scene.objects'):
            capture_context.snapshot_for_video(self.video)

    def test_same_context_file_change_updates_identity(self):
        self.config.write_text('scene: {}\n# one\n', encoding='utf-8')
        before = capture_context.snapshot_for_video(self.video)
        self.config.write_text('scene: {}\n# two\n', encoding='utf-8')
        after = capture_context.snapshot_for_video(self.video)
        self.assertEqual(before['context'], after['context'])
        self.assertNotEqual(before['identity'], after['identity'])


if __name__ == '__main__':
    unittest.main()
