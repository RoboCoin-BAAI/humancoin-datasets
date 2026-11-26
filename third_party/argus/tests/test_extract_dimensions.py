import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from review.extract_dimensions import action_text, call_qwen, prepare, validate, write_new
from review.l1_paths import default_l1_output


class ExtractDimensionsTests(unittest.TestCase):
    def test_action_text_combines_object_without_duplicate(self):
        for action, obj, expected in [
            ('releasing', 'power cord', 'releasing power cord'),
            ('holding power cord', 'power cord', 'holding power cord'),
            ('lifting kettle', 'electric kettle', 'lifting kettle'),
            ('resting', None, 'resting'),
            (None, 'cup', 'cup'),
        ]:
            with self.subTest(action=action, obj=obj):
                self.assertEqual(action_text(action, obj), expected)

    def test_l1_output_stays_under_matching_session(self):
        session = Path('cp_0001_session_20261009_172052').resolve()
        video = session / 'videos/chunk-000/observation.images.head_left_rectified/episode_000000.mp4'
        self.assertEqual(default_l1_output(str(video)), session / 'annotation')
        right = session / 'videos/chunk-000/observation.images.head_right_rectified/episode_000000.mp4'
        self.assertEqual(default_l1_output(str(right)), session / 'annotation')
        with self.assertRaises(ValueError):
            default_l1_output('https://example.org/video.mp4')

    def setUp(self):
        self.prepared = {'source': {'sha256': 'abc'}, 'scene': 'A small kitchen counter',
                         'objects': [{'name': 'kettle', 'attributes': ['white', 'plastic', 'uncertain'],
                                      'location': 'on counter'}],
                         'subtask': [{'start_s': 1, 'end_s': 2, 'action': 'grasping and lifting kettle',
                                      'object': 'electric kettle', 'arm': 'left'}]}
        self.answer = {'task_name': 'Lift kettle', 'actions': [{'index': 0, 'labels': ['grasp', 'lift']}],
                       'scene': {'category': 'household', 'area': 'kitchen'},
                       'objects': [{'index': 0, 'attributes': [{'index': 0, 'kind': 'color'},
                                                               {'index': 1, 'kind': 'material'},
                                                               {'index': 2, 'kind': 'unknown'}]}]}

    def test_preserves_exact_source_and_compound_action(self):
        result = validate(self.answer, self.prepared)
        self.assertEqual(result['actions'], {'vocabulary': ['grasp', 'lift']})
        self.assertEqual(result['subtask'][0]['text'], 'grasping and lifting kettle')
        self.assertEqual(result['subtask'][0]['action'], ['grasp', 'lift'])
        self.assertNotIn('labels', result['subtask'][0])
        self.assertEqual(result['subtask'][0]['start_s'], 1)
        self.assertEqual(result['subtask'][0]['arm'], 'left')
        self.assertNotIn('labels', self.prepared['subtask'][0])
        self.assertEqual(result['scene'], 'A small kitchen counter')
        self.assertEqual(result['scene_classification']['original_description'], 'A small kitchen counter')
        item = result['objects'][0]
        self.assertEqual(item['attributes']['color'], 'white')
        self.assertEqual(item['attributes']['material'], 'plastic')
        self.assertIsNone(item['attributes']['shape'])
        self.assertEqual(item['attributes']['unclassified'], ['uncertain'])
        self.assertEqual(item['original_attributes'], ['white', 'plastic', 'uncertain'])

    def test_accepts_expanded_scene_names(self):
        answer = {**self.answer, 'scene': {'category': 'commercial_service', 'area': 'hair_salon'}}
        result = validate(answer, self.prepared)
        self.assertEqual(result['scene_classification']['category'], 'commercial_service')
        self.assertEqual(result['scene_classification']['area'], 'hair_salon')

    def test_rejects_invalid_expanded_scene_names(self):
        for category in ['salon', 'one_two_three_four', 'Hair Salon']:
            with self.assertRaises(ValueError):
                validate({**self.answer, 'scene': {'category': category, 'area': 'hair_salon'}}, self.prepared)

    def test_removes_requested_source_fields_in_prepared_and_structured_output(self):
        raw = {'parse_ok': True, 'input_identity': {'source_proof': [{'name': 'example'}]},
               'labels': {'timeline': self.prepared['subtask'], 'scene': {'objects': [], 'setting': 'kitchen'}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'episode.json'
            path.write_text(json.dumps(raw), encoding='utf-8')
            prepared = prepare(path)
        self.assertEqual(set(prepared['source']), {'file', 'sha256'})
        previous_prepared = {**self.prepared, 'source': {'file': 'episode.json', 'sha256': 'abc',
                                                        'source_proof': [{'name': 'example'}], 'note': 'old'}}
        result = validate(self.answer, previous_prepared)
        self.assertEqual(result['source'], {'file': 'episode.json', 'sha256': 'abc'})
        self.assertEqual(result['actions'], {'vocabulary': ['grasp', 'lift']})

    def test_vocabulary_uses_original_action_order_and_deduplicates(self):
        prepared = {**self.prepared, 'subtask': self.prepared['subtask'] * 2}
        answer = {**self.answer, 'actions': [{'index': 1, 'labels': ['lift', 'place']},
                                             {'index': 0, 'labels': ['grasp', 'lift']}]}
        result = validate(answer, prepared)
        self.assertEqual(result['actions']['vocabulary'], ['grasp', 'lift', 'place'])
        self.assertEqual([item['action'] for item in result['subtask']], [['grasp', 'lift'], ['lift', 'place']])

    def test_existing_source_labels_are_not_overwritten(self):
        original = {**self.prepared['subtask'][0], 'labels': ['old']}
        prepared = {**self.prepared, 'subtask': [original]}
        with self.assertRaisesRegex(ValueError, 'refusing to overwrite'):
            validate(self.answer, prepared)
        self.assertEqual(original['labels'], ['old'])
        raw = {'parse_ok': True, 'labels': {'timeline': [original], 'scene': {}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'episode.json'
            path.write_text(json.dumps(raw), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'refusing to overwrite'):
                prepare(path)

    def test_rejects_missing_actions_or_incompatible_scene(self):
        for answer in ({'actions': [], 'scene': self.answer['scene']},
                       {'actions': self.answer['actions'], 'scene': {'category': 'kitchen', 'area': 'desk_area'}}):
            with self.assertRaises(ValueError):
                validate(answer, self.prepared)

    def test_rejects_missing_or_duplicate_attribute_classification(self):
        import copy
        for assignments in ([{'index': 0, 'kind': 'color'}],
                            [{'index': 0, 'kind': 'color'}, {'index': 0, 'kind': 'shape'},
                             {'index': 1, 'kind': 'material'}, {'index': 2, 'kind': 'unknown'}],
                            [{'index': 0, 'kind': 'made_up'}, {'index': 1, 'kind': 'material'},
                             {'index': 2, 'kind': 'unknown'}]):
            answer = copy.deepcopy(self.answer)
            answer['objects'][0]['attributes'] = assignments
            with self.assertRaises(ValueError):
                validate(answer, self.prepared)

    def test_mocked_request_and_no_overwrite(self):
        answer = {**self.answer, 'objects': [{'index': 0, 'attributes': [{'index': 2, 'kind': 'unknown'}]}]}
        payload = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(answer)}}],
                   'usage': {'prompt_tokens': 10, 'completion_tokens': 10}}
        receipt = {}
        with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(payload).encode())):
            result = call_qwen(self.prepared, 'qwen3.8-27b', 'https://example.com', 'test-key', 1,
                               receipt=receipt)
        self.assertNotIn('extraction', result)
        self.assertEqual(result['subtask'][0]['action'], ['grasp', 'lift'])
        self.assertEqual(receipt['usage'], payload['usage'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'out.json'
            write_new(path, result)
            with self.assertRaises(FileExistsError):
                write_new(path, result)

    def test_prepare_rejects_unsuccessful_argus_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.json'
            path.write_text('{"parse_ok": false}', encoding='utf-8')
            with self.assertRaises(ValueError):
                prepare(path)

    def test_text_extraction_uses_shared_budget_and_output_limit(self):
        from review import cny_budget
        answer = {**self.answer, 'objects': [{'index': 0, 'attributes': [{'index': 2, 'kind': 'unknown'}]}]}
        payload = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(answer)}}],
                   'usage': {'prompt_tokens': 100, 'completion_tokens': 50}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = root / 'budget.json'
            cny_budget.initialize(ledger, source=root / 'video.mp4', model='qwen3.8-27b', cap=0.6,
                                  reserve=0.15, prior_paid=False)
            with patch.dict(os.environ, {cny_budget.ENV: str(ledger)}), \
                 patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(payload).encode())) as urlopen:
                call_qwen(self.prepared, 'qwen3.8-27b', 'https://example.com', 'test-key', 1)
            call = json.loads(ledger.read_text())['calls'][0]
            self.assertEqual(call['stage'], 'postprocess')
            self.assertEqual(call['cost_micro_cny'], 900)
            self.assertEqual(json.loads(urlopen.call_args.args[0].data)['max_tokens'], 8192)


if __name__ == '__main__':
    unittest.main()
