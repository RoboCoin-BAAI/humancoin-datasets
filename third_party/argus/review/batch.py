"""Run separate L1 review jobs concurrently, one annotation directory per session."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys

from review.l1_paths import default_l1_output

CAMERA = 'observation.images.head_left_rectified'


def child_arguments(arguments: list[str]) -> list[str]:
    """Forward the review settings without recursively forwarding batch options."""
    result = []
    index = 0
    while index < len(arguments):
        arg = arguments[index]
        if arg in ('--source', '--workers', '--sessions-file'):
            index += 2
        elif arg.startswith(('--source=', '--workers=', '--sessions-file=')):
            index += 1
        else:
            result.append(arg)
            index += 1
    return result


def discover(source: Path) -> dict[Path, list[Path]]:
    source = source.resolve()
    if not source.is_dir():
        raise ValueError(f'L1 source directory does not exist: {source}')
    groups: dict[Path, list[Path]] = {}
    seen: set[Path] = set()
    for video in sorted(source.rglob(f'{CAMERA}/episode_*.mp4')):
        if any(part.startswith('.') for part in video.relative_to(source).parts):
            continue
        if not video.is_file():
            continue
        video = video.resolve()
        if video in seen:
            continue
        try:
            annotation = default_l1_output(str(video))
        except ValueError:
            continue
        seen.add(video)
        groups.setdefault(annotation, []).append(video)
    return groups


def run_one(video: Path, annotation: Path, arguments: list[str], env: dict) -> dict:
    job = annotation / 'argus'
    job.mkdir(parents=True, exist_ok=True)
    lock = job / '.argus_batch.lock'
    log = job / '.argus_batch.log'
    try:
        with lock.open('x', encoding='utf-8') as handle:
            handle.write(f'{video}\n')
    except FileExistsError:
        return {'video': str(video), 'status': 'failed', 'error': f'job locked: {lock}'}
    try:
        command = [sys.executable, '-m', 'review', *arguments, '--parent-prepared-job', '--data', str(video)]
        with log.open('a', encoding='utf-8') as handle:
            handle.write(f'\n== batch video: {video}\n')
            handle.flush()
            result = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
        return {'video': str(video), 'status': 'done' if result.returncode == 0 else 'failed',
                'returncode': result.returncode, 'annotation': str(annotation), 'log': str(log)}
    except (OSError, ValueError) as error:
        return {'video': str(video), 'status': 'failed', 'error': str(error), 'log': str(log)}
    finally:
        lock.unlink(missing_ok=True)



_original_run_one = run_one


def run_one(video, annotation, arguments, env):
    from review.overwrite import job_lock, archive_job
    job = annotation / 'argus'
    try:
        with job_lock(job.with_name(job.name + '.adapter')):
            if '--force' not in arguments:
                from review.completed import completed_result
                with job_lock(job):
                    complete = completed_result(job, annotation)
                if complete:
                    return dict(video=str(video), annotation=str(annotation), **complete)
                if '--free' not in arguments and '--retry-received' not in arguments:
                    from review.overwrite import clear_incomplete_job
                    with job_lock(job):
                        clear_incomplete_job(job, annotation / f'{video.stem}_structured.json')
            if '--force' in arguments:
                if '--free' in arguments:
                    raise ValueError('--force cannot be combined with --free')
                with job_lock(job):
                    archive_job(job, annotation / f'{video.stem}_structured.json')
                arguments = [a for a in arguments if a != '--force']
            return _original_run_one(video, annotation, arguments, env)
    except (OSError, RuntimeError, ValueError) as exc:
        return dict(video=str(video), status='failed', error=str(exc))


def run(source: Path, workers: int, arguments: list[str], env: dict, sessions_file: Path | None = None) -> int:
    try:
        groups = discover(source)
        if sessions_file is not None:
            values = json.loads(sessions_file.read_text(encoding="utf-8"))
            if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
                raise ValueError("Invalid session allowlist")
            sessions = {Path(v).resolve() for v in values}
            if any(not s.is_relative_to(source.resolve()) for s in sessions):
                raise ValueError("Selected session outside source")
            groups = {a: videos for a, videos in groups.items()
                      if any(v.is_relative_to(s) for v in videos for s in sessions)}
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr, flush=True)
        return 1
    if not groups:
        print(f'No L1 {CAMERA} MP4 videos found under {source}', file=sys.stderr, flush=True)
        return 1
    jobs, failures = [], 0
    for annotation, videos in groups.items():
        if len(videos) != 1:
            failures += len(videos)
            print(json.dumps({'annotation': str(annotation), 'status': 'failed',
                              'error': 'multiple left rectified videos share this annotation directory; '
                                       'automatic batch review cannot safely share its intermediate files',
                              'videos': [str(video) for video in videos]}, ensure_ascii=False), flush=True)
        else:
            jobs.append((videos[0], annotation))
    print(f'== batch: {len(jobs)} video jobs, up to {workers} in parallel; budget is per video', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_one, video, annotation, arguments, env): video
                   for video, annotation in jobs}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as error:
                result = {'video': str(futures[future]), 'status': 'failed', 'error': str(error)}
            failures += result['status'] == 'failed'
            print(json.dumps(result, ensure_ascii=False), flush=True)
    print(json.dumps({'total_videos': sum(map(len, groups.values())), 'failed': failures}, ensure_ascii=False), flush=True)
    return int(bool(failures))
