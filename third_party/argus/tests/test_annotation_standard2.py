import copy
import json
import os
from unittest.mock import patch

import pytest

from review import standard2 as s2
from review import structured
from review.base import run as base


def annotation():
    return dict(schema_version='1.0', episode_dir=None,
        source=dict(file='source.json', sha256='a'*64), timesteps_s=[], finish_reason='stop',
        objects=[dict(name='two red cups', location=None, attributes=dict(color='red', material=None, shape=None,
            unclassified=[], additional={}), original_attributes=[])], scene='Washing cups',
        subtask=[dict(start_s=0.0, end_s=1.0, text='Wash the red cups', arm=None, action=['washing'],
            object='two red cups', destination=None, spatial_relation=None, contribution=None, progress=None,
            hands_visible=None, hands_wearing=None)], task_description='Wash cups',
        key_events=[], state_changes=[], tasks=[], scene_graph=[], recovery=[], instruction_variants=[],
        data_issues=[], viewpoint='first_person', performance_review=None, task_name='washing red cups',
        actions=dict(vocabulary=['washing']), scene_classification=dict(category='other', area='other', original_description='Washing cups'))


def response():
    return dict(scene=dict(category='household', area='kitchen'),
        objects=[dict(index=0, name='cups', attributes=[])],
        actions=[dict(index=0, labels=['wash', 'wash'])], task=dict(verb='wash', object_index=0))


def test_task_name_is_subtask_verb_plus_canonical_main_object():
    source = annotation()
    before = copy.deepcopy(source)
    result = s2.assemble(source, response())
    assert result['task_name'] == 'wash cup'
    assert result['actions']['vocabulary'] == ['wash']
    assert result['subtask'][0]['action'] == ['wash']
    assert result['subtask'][0]['text'] == 'Wash the red cups'
    assert result['subtask'][0]['object'] == 'two red cups'
    assert result['objects'][0]['attributes']['color'] == 'red'
    assert s2.grouping_key(result) == ('household', 'kitchen', 'wash cup')
    assert source == before
    base.validate(result, standard=2)


def test_color_count_size_variants_have_identical_group_key():
    keys = []
    for noun in ('two red cups', 'large blue cup', 'cup'):
        completion = response()
        completion['objects'][0]['name'] = noun
        keys.append(s2.grouping_key(s2.assemble(annotation(), completion)))
    assert len(set(keys)) == 1


@pytest.mark.parametrize('task', [dict(verb='pick', object_index=0), dict(verb='wash', object_index=5),
                                dict(verb='wash', object_index=True), dict(verb='wash and rinse', object_index=0)])
def test_atomic_or_invalid_task_names_are_rejected(task):
    completion = response()
    completion['task'] = task
    with pytest.raises(ValueError):
        s2.assemble(annotation(), completion)


def test_unknown_task_is_not_automatically_merged():
    completion = response()
    completion['task'] = None
    result = s2.assemble(annotation(), completion)
    assert result['task_name'] is None
    assert s2.grouping_key(result) is None


def test_scenes_only_commercial_can_extend_and_retired_areas_rejected():
    s2.validate_scene('commercial_service', 'shoe_repair_shop')
    s2.validate_scene('professional_work', 'conference_center')
    for category, area in [('household', 'home_gym'), ('professional_work', 'meeting_room'),
                           ('public_space', 'badminton_court'), ('commercial_service', 'bedroom')]:
        with pytest.raises(ValueError):
            s2.validate_scene(category, area)


def test_action_opposites_not_merged_and_recurrence_aliases_normalized():
    assert s2.normalize_verb('retrim') == 'trim'
    assert s2.normalize_verb('trimming') == 'trim'
    assert s2.normalize_verb('closing') == 'close'
    assert s2.normalize_verb('picked') == 'pick'
    for verb in ('open', 'close', 'fold', 'unfold', 'screw', 'unscrew', 'remove', 'move'):
        assert s2.normalize_verb(verb) == verb
    assert s2.normalize_object_name('first aid kit') == 'first aid kit'
    assert s2.normalize_object_name('washing machine') == 'washing machine'


def test_raw_argus_attributes_are_typed_without_inventing_new_source_attributes():
    source = annotation()
    source['objects'][0]['attributes'] = ['red']
    source['objects'][0].pop('original_attributes')
    completion = response()
    completion['objects'][0]['attributes'] = [dict(index=0, kind='color', value='red')]
    result = s2.assemble(source, completion)
    assert result['objects'][0]['attributes']['color'] == 'red'
    assert result['objects'][0]['original_attributes'] == ['red']


def test_task_actor_cannot_replace_main_object():
    completion = response()
    completion['objects'][0]['name'] = 'customer'
    with pytest.raises(ValueError, match='actor'):
        s2.assemble(annotation(), completion)


def test_opt_in_structured_pipeline_uses_standard2_and_reuses_receipt(tmp_path):
    source = tmp_path / 'run_final/out/episode_000000.json'
    source.parent.mkdir(parents=True)
    original = annotation()
    original['objects'][0]['attributes'] = []
    source.write_text(json.dumps(dict(parse_ok=True, labels=dict(scene=dict(setting='Washing cups', objects=original['objects']),
        timeline=original['subtask'], task_summary='Wash cups', viewpoint='first_person'))))
    def fake(prepared, *args, receipt, result_validator, **kwargs):
        result = s2.assemble(prepared, response())
        result_validator(result)
        return result
    with patch.dict(os.environ, {'XMSXB_API_KEY': 'test'}, clear=True), patch.object(s2, 'call_qwen', side_effect=fake) as call:
        outputs = structured.run(tmp_path, 'qwen3.7-plus', standard=2)
        structured.run(tmp_path, 'qwen3.7-plus', standard=2)
    assert call.call_count == 1
    receipt = json.loads((tmp_path / '.episode_000000_structured.claim.json').read_text())
    assert receipt['annotation_standard'] == 2
    assert json.loads(outputs[0].read_text())['task_name'] == 'wash cup'
