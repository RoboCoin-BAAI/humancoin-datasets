import json
import pytest

from review.jianzhi.run import CAMERA, main, plan


def video(tmp_path, camera=CAMERA):
    (tmp_path / 'meta').mkdir(exist_ok=True)
    (tmp_path / 'meta/info.json').write_text('{}')
    path = tmp_path / 'videos/chunk-000' / camera / 'episode_000000.mp4'
    path.parent.mkdir(parents=True)
    path.touch()
    return path


def test_explicit_job_supports_hash_session(tmp_path):
    path = video(tmp_path)
    item = plan(path, ['--free', '--no-board'])
    assert not __import__('pathlib').Path(item['job']).is_relative_to(tmp_path)
    assert item['output'] == str(tmp_path / 'annotation/episode_000000_structured.json')
    assert item['command'][item['command'].index('--rig') + 1] == 'ego_head'
    assert '--free' in item['command']
    assert '--no-board' in item['command']
    assert item['command'][item['command'].index('--model') + 1] == 'qwen3.7-plus'


def test_raw_video_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        plan(video(tmp_path, 'observation.images.head_left'), [])


def test_dry_run_creates_no_annotation(tmp_path, capsys):
    path = video(tmp_path)
    assert main(['--source', str(tmp_path), '--dry-run', '--free']) == 0
    assert json.loads(capsys.readouterr().out)['video'] == str(path)
    assert not (tmp_path / 'annotation').exists()


def test_missing_rectification_fails(tmp_path):
    with pytest.raises(SystemExit):
        main(['--source', str(tmp_path), '--dry-run'])


def test_adapter_delegates_execution_to_base(tmp_path, monkeypatch):
    from review.jianzhi import run
    item = plan(video(tmp_path), ['--keep-work'])
    assert item['keep_work']
    assert 'argus-base' in item['job']
    calls = []
    def execute_base(value):
        calls.append(value)
        return dict(status='done', output=value['output'])
    monkeypatch.setattr(run.base, 'execute', execute_base)
    assert run.execute(item)['status'] == 'done'
    assert calls == [item]


def test_plan_matches_shared_base(tmp_path):
    from review.base.run import plan as base_plan
    path = video(tmp_path)
    item = plan(path, ['--sample-fps', '1'])
    expected = base_plan(path, tmp_path / 'annotation/episode_000000_structured.json',
                         ['--dataset', tmp_path.name, '--sample-fps', '1'])
    assert item == expected
