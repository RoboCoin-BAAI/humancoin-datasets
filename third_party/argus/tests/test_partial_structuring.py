from unittest.mock import patch
import io
import json

import pytest

from review.extract_dimensions import completion_plan, merge_completion, validate, call_qwen


def test_exact_values_still_request_overall_task_name():
    prepared = dict(scene='kitchen', objects=[dict(name='cup', attributes=['red', 'plastic'])],
                    subtask=[dict(action='grasp', start_s=1., end_s=2., object='cup')])
    payload = {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(
        {'task_name': 'Pick up cup', 'actions': [], 'objects': []})}}]}
    with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(payload).encode())) as request:
        result = call_qwen(prepared, 'test-model', 'https://example.test/v1/chat/completions', 'unused', 1)
    request.assert_called_once()
    assert result['task_name'] == 'Pick up cup'
    assert result['subtask'][0]['action'] == ['grasp']
    assert result['subtask'][0]['start_s'] == 1.
    assert result['objects'][0]['attributes']['color'] == 'red'


def test_model_receives_only_unresolved_indices():
    prepared = dict(scene='kitchen', objects=[dict(name='cup', attributes=['red', 'shiny'])],
                    subtask=[dict(action='rest'), dict(action='picks up cup')])
    local, pending = completion_plan(prepared)
    assert pending['timeline'] == [dict(index=1, action='picks up cup')]
    assert pending['objects'][0]['attributes'] == [dict(index=1, value='shiny')]
    assert pending['scene_description'] == 'kitchen'
    assert pending['scene_category'] == 'kitchen'
    response = dict(task_name='Pick up cup', actions=[dict(index=1, labels=['grasp', 'lift'])],
                    objects=[dict(index=0, attributes=[dict(index=1, kind='unknown')])])
    result = validate(merge_completion(local, pending, response), prepared)
    assert result['subtask'][0]['action'] == ['rest']
    assert result['subtask'][1]['action'] == ['grasp', 'lift']
    response['actions'].append(dict(index=0, labels=['transport']))
    with pytest.raises(ValueError, match='indices'):
        merge_completion(local, pending, response)


def test_task_name_uses_full_observed_context_and_validates_short_output():
    prepared = dict(scene='kitchen', task_description='Sort matching drinks onto cards',
                    instruction_variants=['Match drinks to their cards.'],
                    tasks=[], objects=[], subtask=[dict(action='place', object='juice')])
    local, pending = completion_plan(prepared)
    assert pending['timeline'] == []
    assert pending['task_context']['timeline'] == prepared['subtask']
    assert pending['task_context']['task_description'] == prepared['task_description']
    assert pending['task_context']['instruction_variants'] == prepared['instruction_variants']
    with pytest.raises(ValueError, match='include task_name'):
        merge_completion(local, pending, {})
    for value in ('', 'x'*101, 'word '*13, ['name'], 'one\ntwo'):
        merged = merge_completion(local, pending, {'task_name': value})
        with pytest.raises(ValueError, match='task_name'):
            validate(merged, prepared)
    result = validate(merge_completion(local, pending, {'task_name': None}), prepared)
    assert result['task_name'] is None
    result = validate(merge_completion(local, pending, {'task_name': 'Match beverages to labeled cards'}), prepared)
    assert result['task_name'] == 'Match beverages to labeled cards'


def test_task_inferred_scene_without_description():
    prepared = dict(scene=None, objects=[], subtask=[dict(action='place')],
                    task_description='Label and organize cardboard packages')
    local, pending = completion_plan(prepared)
    assert pending['scene_description'] is None
    assert pending['task_context']['task_description'] == prepared['task_description']
    response = dict(task_name='Label packages', scene=dict(category='warehouse', area='packing_station'))
    result = validate(merge_completion(local, pending, response), prepared)
    assert result['scene_classification']['area'] == 'packing_station'
    assert result['subtask'][0]['action'] == ['place']


def test_exact_category_accepts_inferred_area_but_rejects_category_change():
    prepared = dict(scene='warehouse', objects=[], subtask=[])
    local, pending = completion_plan(prepared)
    response = dict(task_name='Label packages', scene=dict(category='warehouse', area='packing_station'))
    result = validate(merge_completion(local, pending, response), prepared)
    assert result['scene_classification']['area'] == 'packing_station'
    response['scene']['category'] = 'kitchen'
    with pytest.raises(ValueError, match='preserve'):
        merge_completion(local, pending, response)
