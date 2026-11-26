import hashlib
import json

import pytest

from review.completed import completed_result
from review.retention import compact


def test_compaction_preserves_verified_outputs_and_reuse(tmp_path):
    job = tmp_path / 'job'
    job.mkdir()
    output = job / 'episode_000000_structured.json'
    output.write_text('{"task": "test"}')
    receipt = dict(version=1, status='done', previous_model='qwen3.6-plus',
                   outputs={output.name: hashlib.sha256(output.read_bytes()).hexdigest()})
    (job / 'structured_complete.json').write_text(json.dumps(receipt))
    (job / 'l3_input.json').write_text('{}')
    (job / 'frames').mkdir()
    (job / 'frames/frame.jpg').write_bytes(b'image')
    (job / 'review.log').write_text('log')
    compact(job)
    assert {p.name for p in job.iterdir()} == {
        output.name, 'structured_complete.json', 'l3_input.json'}
    assert completed_result(job)['status'] == 'done'
    compact(job)
    output.write_text('{}')
    assert completed_result(job) is None
    with pytest.raises(ValueError, match='not verified'):
        compact(job)


def test_incomplete_job_is_retained(tmp_path):
    (tmp_path / 'error.log').write_text('failed')
    with pytest.raises(ValueError, match='not verified'):
        compact(tmp_path)
    assert (tmp_path / 'error.log').exists()


def test_export_to_annotation_and_reuse(tmp_path):
    job = tmp_path / 'annotation/argus/chunk-000_episode_000000'
    job.mkdir(parents=True)
    output = job / 'episode_000000_structured.json'
    output.write_text('{"subtask":[]}')
    (job / 'structured_complete.json').write_text(json.dumps(dict(
        version=1, status='done', outputs={output.name:
            hashlib.sha256(output.read_bytes()).hexdigest()})))
    compact(job, annotation_output=True)
    final = tmp_path / 'annotation' / output.name
    assert final.exists() and not output.exists()
    assert completed_result(job)['outputs'] == [str(final)]
    compact(job, annotation_output=True)
    final.write_text('{}')
    assert completed_result(job) is None


def test_full_success_can_be_compacted(tmp_path, monkeypatch):
    from review import extract_dimensions
    source = tmp_path / 'run_final/out/episode_000000.json'
    source.parent.mkdir(parents=True)
    source.write_text('{}')
    (tmp_path / 'run_final/run.json').write_text('{"status":"done"}')
    monkeypatch.setattr(extract_dimensions, 'prepare',
                        lambda p: dict(source=dict(sha256='source-hash')))
    output = tmp_path / 'episode_000000_structured.json'
    output.write_text('{"subtask":[]}')
    (tmp_path / '.episode_000000_structured.claim.json').write_text(json.dumps(dict(
        status='complete', source_sha256='source-hash',
        output_sha256=hashlib.sha256(output.read_bytes()).hexdigest())))
    compact(tmp_path)
    assert not source.exists()
    assert completed_result(tmp_path)['outputs'] == [str(output)]


def test_cleanup_preserves_symlink_targets(tmp_path):
    job = tmp_path / 'job'
    job.mkdir()
    outside = tmp_path / 'source.mp4'
    outside.write_bytes(b'original video')
    folder = job / 'upload_selected'
    folder.mkdir()
    try:
        (folder / 'video.mp4').symlink_to(outside)
        (job / 'source_link').symlink_to(outside)
    except OSError:
        pytest.skip('Symlink creation unavailable')
    output = job / 'episode_000000_structured.json'
    output.write_text('{}')
    (job / 'structured_complete.json').write_text(json.dumps(dict(
        version=1, status='done', outputs={output.name:
            hashlib.sha256(output.read_bytes()).hexdigest()})))
    compact(job)
    assert outside.read_bytes() == b'original video'
    assert not folder.exists()
    assert not (job / 'source_link').exists()
