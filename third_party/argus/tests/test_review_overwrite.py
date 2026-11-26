from pathlib import Path
import pytest
from review.overwrite import archive_job, job_lock


def test_archive_preserves_receipts_and_external_output(tmp_path):
    job = tmp_path / 'argus'
    job.mkdir()
    (job / 'preparation_receipt.json').write_text('old receipt')
    (job / '.argus_batch.lock').write_text('stale')
    external = tmp_path / 'episode_000000_structured.json'
    external.write_text('old result')
    archive = archive_job(job, external)
    assert not job.exists() and not external.exists()
    assert (archive / 'job/preparation_receipt.json').read_text() == 'old receipt'
    assert (archive / external.name).read_text() == 'old result'
    assert (archive / 'job/.argus_batch.lock').exists()


def test_os_lock_blocks_concurrent_runs_and_releases(tmp_path):
    job = tmp_path / 'argus'
    with job_lock(job):
        with pytest.raises(RuntimeError, match='running'):
            with job_lock(job):
                pass
    with job_lock(job):
        pass


def test_force_batch_archives_stale_lock_before_dispatch(tmp_path, monkeypatch):
    from review import batch
    from types import SimpleNamespace
    annotation = tmp_path / 'annotation'
    job = annotation / 'argus'
    job.mkdir(parents=True)
    (job / '.argus_batch.lock').write_text('stale')
    def dispatch(command, **kwargs):
        assert '--force' not in command
        assert list((annotation / '.review_history').glob('*/job/.argus_batch.lock'))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(batch.subprocess, 'run', dispatch)
    assert batch.run_one(Path('episode_000000.mp4'), annotation, ['--force'], {})['status'] == 'done'



def test_cleanup_deletes_only_argus_records(tmp_path):
    from review.overwrite import clear_incomplete_job
    annotation=tmp_path/'annotation'
    job=annotation/'argus'
    job.mkdir(parents=True)
    (job/'failed_request.json').write_text('{}')
    external=annotation/'episode_000000_structured.json'
    external.write_text('{}')
    unrelated=annotation/'quality_report.json'
    unrelated.write_text('{}')
    clear_incomplete_job(job,external)
    assert not job.exists() and not external.exists()
    assert unrelated.exists()
