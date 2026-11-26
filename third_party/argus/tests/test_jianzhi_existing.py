import json
import pytest
from review.jianzhi.structure_existing import prepare, assemble


@pytest.mark.parametrize('success', [True, False])
def test_source_deleted_only_after_success(tmp_path, monkeypatch, success):
    from review.jianzhi import structure_existing as module
    (tmp_path/'meta').mkdir()
    (tmp_path/'data/chunk-000/episode_000000').mkdir(parents=True)
    source = tmp_path/'meta/annotations.json'
    source.write_text(json.dumps(dict(episodes=[dict(episode_index=0, annotation=dict(
        bold_mark='Cut hair', sst='business/beauty_personal_care/barber_shop/daily_work',
        segments_info=[dict(end_time_s=2, fine_label='Cut hair', is_success=True)]))])))
    monkeypatch.setattr(module.tempfile, 'gettempdir', lambda: str(tmp_path.parent/'external-text-test'))
    monkeypatch.setattr('review.qwen_endpoint.settings', lambda model: ('key','endpoint','provider'))
    monkeypatch.setenv('ARGUS_STRUCTURED_MAX_ATTEMPTS','1')
    def request(*args):
        if not success:
            raise TimeoutError('failed')
        return dict(task_name='Cut hair',scene='Hair salon',
            scene_classification=dict(category='commercial_service',area='hair_salon',original_description='Hair salon'),
            objects=[],subtasks=[dict(index=0,action=['cut'],object=None,destination=None,spatial_relation=None)])
    monkeypatch.setattr(module,'request',request)
    if success:
        module.process(tmp_path,'qwen3.7-plus')
        assert not source.exists()
        assert module.process(tmp_path,'qwen3.7-plus')[0]['reused']
    else:
        with pytest.raises(TimeoutError):
            module.process(tmp_path,'qwen3.7-plus')
        assert source.exists()


def test_existing_segments_are_preserved_and_unknowns_remain_null(tmp_path):
    source = tmp_path/'annotations.json'
    source.write_text(json.dumps(dict(episodes=[dict(episode_index=0, annotation=dict(
        bold_mark='Cut hair', sst='business/beauty_personal_care/barber_shop/daily_work',
        segments_info=[dict(end_time_s=2, fine_label='Cut hair', sub_segments_info=[
            dict(end_time_s=1, fine_label='Trim hair', is_success=True),
            dict(start_time_s=1, end_time_s=2, fine_label='Check hair', is_success=True)])]))])))
    prepared = prepare(source, 0)
    patch = dict(task_name='Cut hair', scene='Hair cutting service',
                 scene_classification=dict(category='commercial_service', area='hair_salon',
                                           original_description='Hair cutting service'), objects=[],
                 subtasks=[dict(index=i, action=[action], object=None, destination=None,
                                spatial_relation=None) for i, action in enumerate(['trim','inspect'])])
    result = assemble(prepared, patch)
    assert [(s['start_s'],s['end_s'],s['text']) for s in result['subtask']] == [
        (0,1,'Trim hair'),(1,2,'Check hair')]
    assert result['subtask'][0]['hands_visible'] is None
    assert result['subtask'][0]['progress'] is None
    assert result['tasks'][0]['outcome'] == 'success'
    patch['subtasks'][0]['start_s'] = 10
    with pytest.raises(ValueError):
        assemble(prepared, patch)
