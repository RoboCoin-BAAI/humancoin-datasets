import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from review.base import run


def annotation():
    return dict(schema_version='1.0', episode_dir=None,
                source=dict(file='source.json', sha256='a'*64), timesteps_s=[], finish_reason='stop',
                objects=[], scene=None, subtask=[], task_description=None, key_events=[],
                state_changes=[], tasks=[], scene_graph=[], recovery=[], instruction_variants=[],
                data_issues=[], viewpoint='first_person', performance_review=None, task_name=None,
                actions=dict(vocabulary=[]), scene_classification=dict(category='other', area='other',
                                                                      original_description=None))


@pytest.fixture
def item(tmp_path, monkeypatch):
    monkeypatch.setattr(run.tempfile, 'gettempdir', lambda: str(tmp_path / 'temp'))
    video = tmp_path / 'input.mp4'
    video.touch()
    return run.plan(video, tmp_path / 'annotation/result.json')


@pytest.mark.parametrize('keep', [False, True])
def test_publish_then_cleanup(item, monkeypatch, keep):
    item['keep_work'] = keep
    job = Path(item['job'])
    job.mkdir(parents=True)
    source = job / 'episode_000000_structured.json'
    source.write_text(json.dumps(annotation()))
    monkeypatch.setattr(run.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0))
    monkeypatch.setattr('review.completed.completed_result', lambda p: dict(outputs=[str(source)]))
    assert run.execute(item)['status'] == 'done'
    assert json.loads(Path(item['output']).read_text()) == annotation()
    assert job.exists() == keep


def test_failure_preserves_work(item, monkeypatch):
    job = Path(item['job'])
    job.mkdir(parents=True)
    monkeypatch.setattr(run.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=1))
    assert run.execute(item)['status'] == 'failed'
    assert job.exists()
    assert not Path(item['output']).exists()


def test_invalid_output_preserves_work(item, monkeypatch):
    job = Path(item['job'])
    job.mkdir(parents=True)
    source = job / 'episode_000000_structured.json'
    source.write_text('{}')
    monkeypatch.setattr(run.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0))
    monkeypatch.setattr('review.completed.completed_result', lambda p: dict(outputs=[str(source)]))
    with pytest.raises(Exception):
        run.execute(item)
    assert job.exists()
    assert not Path(item['output']).exists()


def test_reuse_does_not_call_model(item, monkeypatch):
    output = Path(item['output'])
    output.parent.mkdir(parents=True)
    output.write_text(json.dumps(annotation()))
    monkeypatch.setattr(run.subprocess, 'run', lambda *a, **k: pytest.fail('model dispatched'))
    assert run.execute(item)['reused']


def test_default_model(item):
    command = item['command']
    assert command[command.index('--model') + 1] == 'qwen3.7-plus'


def test_free_preserves_work(item, monkeypatch):
    item['command'].append('--free')
    job = Path(item['job'])
    job.mkdir(parents=True)
    monkeypatch.setattr(run.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0))
    assert run.execute(item)['status'] == 'prepared'
    assert job.exists()
