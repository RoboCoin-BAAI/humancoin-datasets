"""Explicit fresh review runs; previous artifacts are retained for inspection."""
from contextlib import contextmanager
from pathlib import Path
import os
from uuid import uuid4


@contextmanager
def job_lock(job):
    job = Path(job).resolve()
    job.parent.mkdir(parents=True, exist_ok=True)
    # Outside the job so archiving cannot replace the locked inode.
    path = job.parent / ('.' + job.name + '.execution.lock')
    with path.open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            stream.seek(0, 2)
            if stream.tell() == 0:
                stream.write(b'0'); stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(f'Review job is running: {job}') from exc
        else:
            import fcntl
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f'Review job is running: {job}') from exc
        try:
            yield
        finally:
            if os.name == 'nt':
                stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def archive_job(job, structured=None):
    job = Path(job).absolute()
    if job.is_symlink() or (job.exists() and not job.is_dir()):
        raise ValueError(f'Invalid review output directory: {job}')
    if structured is not None:
        structured = Path(structured)
        if structured.is_symlink():
            raise ValueError(f'Invalid structured output: {structured}')
    if not job.exists() and (structured is None or not structured.exists()):
        return None
    archive_root = job.parent / '.review_history'
    archive_root.mkdir(exist_ok=True)
    archive = archive_root / (job.name + '-' + uuid4().hex)
    archive.mkdir()
    if job.exists():
        job.rename(archive / 'job')
    if structured is not None and structured.exists():
        structured.rename(archive / structured.name)
    print(f'Previous review archived: {archive}; starting a fresh run', flush=True)
    return archive



def clear_incomplete_job(job, structured=None):
    """Delete only the selected annotation job and its external structured result."""
    import shutil
    job=Path(job).absolute()
    resolved=job.resolve()
    if job.is_symlink() or resolved==Path(resolved.anchor) or job.name in ('', '.', '..'):
        raise ValueError(f'Unsafe review cleanup target: {job}')
    if job.exists() and not job.is_dir():
        raise ValueError(f'Review job is not a directory: {job}')
    if structured is not None:
        structured=Path(structured).absolute()
        if structured.is_symlink() or structured.parent.resolve()!=job.parent.resolve() or not structured.name.endswith('_structured.json'):
            raise ValueError(f'Unsafe structured cleanup target: {structured}')
    if job.exists():
        shutil.rmtree(job)
    if structured is not None:
        structured.unlink(missing_ok=True)
    print(f'Cleared prior incomplete annotation: {job}; starting from scratch',flush=True)
