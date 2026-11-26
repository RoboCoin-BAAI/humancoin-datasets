import copy
import io
import json

import pytest
from jsonschema import Draft202012Validator

from review.extract_dimensions import call_qwen, completion_plan, merge_completion, validate


def source(attributes=None, arm='none'):
    return dict(scene='household', objects=[dict(name='cup', attributes=attributes or [])],
                subtask=[dict(action='grasp', arm=arm)])


def response(attributes=None):
    return dict(task_name='Pick up cup', actions=[],
                scene=dict(category='household', area='kitchen'),
                objects=[dict(index=0, attributes=attributes)] if attributes is not None else [])


def test_unknown_null_retains_text_or_unknown_and_provenance():
    prepared = source([None, 'shiny', ''])
    original = copy.deepcopy(prepared)
    answer = response([dict(index=i, kind='unknown', value=None) for i in range(3)])
    local, pending = completion_plan(prepared)
    result = validate(merge_completion(local, pending, answer), prepared)
    obj = result['objects'][0]
    assert obj['attributes']['unclassified'] == ['unknown', 'shiny', 'unknown']
    assert obj['original_attributes'] == [None, 'shiny', '']
    assert prepared == original
    assert result['subtask'][0]['arm'] is None


@pytest.mark.parametrize(('kind', 'value', 'expected'), [
    ('color', 'grey', 'gray'), ('material', 'wooden', 'wood'),
    ('shape', 'box-shaped', 'cuboid'),
])
def test_completion_aliases_are_normalized_before_enum_validation(kind, value, expected):
    prepared = source(['ambiguous'])
    answer = response([dict(index=0, kind=kind, value=value)])
    answer['actions'] = [dict(index=0, labels=['grasp'])]
    result = validate(answer, prepared)
    assert result['objects'][0]['attributes'][kind] == expected
    answer['objects'][0]['attributes'][0]['value'] = 'unsupported'
    with pytest.raises(ValueError, match='outside enum'):
        validate(answer, prepared)


def test_wrong_indices_report_object_missing_extra_and_duplicates():
    prepared = source(['red', 'shiny', 'metal', 'mottled'])
    local, pending = completion_plan(prepared)
    answer = response([dict(index=i, kind='unknown') for i in [0, 0]])
    with pytest.raises(ValueError) as error:
        merge_completion(local, pending, answer)
    for detail in ['object=0', 'expected=[1, 3]', 'actual=[0, 0]',
                   'missing=[1, 3]', 'extra=[0]', 'duplicates=[0]']:
        assert detail in str(error.value)


def test_invalid_arm_is_not_silently_normalized():
    result = validate(dict(task_name=None, actions=[dict(index=0, labels=['grasp'])],
                           objects=[], scene=dict(category='household', area='kitchen')),
                      dict(subtask=[dict(action='grasp', arm='invalid')], objects=[]))
    assert result['subtask'][0]['arm'] == 'invalid'
    with pytest.raises(Exception, match='not one of'):
        Draft202012Validator({'enum': ['left', 'right', 'both', None]}).validate('invalid')


@pytest.mark.parametrize('failure', ['indices', 'scene', 'schema'])
def test_retry_sends_specific_validation_feedback(monkeypatch, failure):
    from review import structured_retry
    monkeypatch.setattr(structured_retry.time, 'sleep', lambda _: None)
    monkeypatch.setenv('ARGUS_STRUCTURED_MAX_ATTEMPTS', '2')
    prepared = source(['shiny'])
    good = response([dict(index=0, kind='unknown')])
    bad = copy.deepcopy(good)
    if failure == 'indices':
        bad['objects'][0]['attributes'][0]['index'] = 9
    elif failure == 'scene':
        bad['scene']['area'] = 'hair_salon'
    calls = []

    def send(request, timeout):
        calls.append(json.loads(request.data))
        answer = bad if len(calls) == 1 else good
        return io.BytesIO(json.dumps(dict(choices=[dict(finish_reason='stop',
            message=dict(content=json.dumps(answer)))])).encode())

    validations = []

    def validate_schema(result):
        validations.append(result)
        if failure == 'schema' and len(validations) == 1:
            Draft202012Validator({'type': 'string'}).validate(None)

    monkeypatch.setattr('urllib.request.urlopen', send)
    call_qwen(prepared, 'test-model', 'https://example.test', 'unused', 1,
              result_validator=validate_schema)
    assert len(calls) == 2
    assert len(calls[0]['messages']) == 2
    feedback = calls[1]['messages'][-1]['content']
    assert 'failed validation' in feedback
    assert {'indices': 'missing=[0]', 'scene': 'area_owner=',
            'schema': "None is not of type 'string'"}[failure] in feedback
