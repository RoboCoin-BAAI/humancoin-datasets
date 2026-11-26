import hashlib
import json
from review.completed import completed_result


def test_completed_annotation_reuses_old_model_and_checks_integrity(tmp_path, monkeypatch):
    from review import extract_dimensions
    job=tmp_path/'argus'
    source=job/'run_final/out/episode_000000.json'
    source.parent.mkdir(parents=True)
    source.write_text('{}')
    (job/'run_final/run.json').write_text(json.dumps(dict(status='done',annotation_policy=dict(model='old-model'))))
    monkeypatch.setattr(extract_dimensions,'prepare',lambda path: dict(source=dict(sha256='source-hash')))
    output=tmp_path/'episode_000000_structured.json'
    output.write_text('{"subtask": []}')
    receipt=job/'.episode_000000_structured.claim.json'
    receipt.write_text(json.dumps(dict(status='complete',source_sha256='source-hash',
        output_sha256=hashlib.sha256(output.read_bytes()).hexdigest())))
    result=completed_result(job,tmp_path)
    assert result['status']=='done' and result['reused']
    assert result['previous_model']=='old-model'
    output.write_text('{}')
    assert completed_result(job,tmp_path) is None
    receipt.unlink()
    assert completed_result(job,tmp_path) is None


def test_failed_or_missing_run_not_reused(tmp_path):
    assert completed_result(tmp_path) is None
    (tmp_path/'run_final').mkdir()
    (tmp_path/'run_final/run.json').write_text('{"status":"failed"}')
    assert completed_result(tmp_path) is None
