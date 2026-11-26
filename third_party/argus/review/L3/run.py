"""Standalone L3 review adapter; existing review scripts remain unchanged."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import shutil
import sys
from threading import Event

STOP = Event()

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))

CAMERA = 'observation.images.headband_left_rgb_rectified'


def execution_start(session, video):
    path = session / 'meta/qc_reports.jsonl'
    rows = [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    index = int(video.stem.removeprefix('episode_'))
    selected = [r for r in rows if r.get('episode_index') == index]
    if len(selected) != 1:
        raise ValueError(f'Expected one qc_reports record for episode {index}')
    markers = selected[0]['qc_report']['streams']['marker']['stats']['markers']['items']
    starts = [m for m in markers if m.get('name') == 'EXEC_START' or m.get('tag') == 'EXEC_START']
    if len(starts) != 1:
        raise ValueError(f'Expected one EXEC_START marker in {path}')
    offset = starts[0].get('t_offset')
    if type(offset) not in (int, float) or not math.isfinite(offset) or offset < 0:
        raise ValueError(f'Invalid EXEC_START t_offset in {path}')
    return dict(source_file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                start_s=float(offset), timebase='seconds since RUN_START; retain original video clock')


def video_session(video):
    video = Path(video).resolve()
    if (not video.is_file() or video.suffix != '.mp4' or
            not re.fullmatch(r'episode_\d+', video.stem) or video.parent.name != CAMERA or
            not re.fullmatch(r'chunk-\d+', video.parent.parent.name) or
            video.parents[2].name != 'videos' or
            not re.fullmatch(r'L3-\d{3}-\d{4}(?:-\d{2}){5}', video.parents[3].name)):
        raise ValueError(f'Expected an L3 rectified left video: {video}')
    return video.parents[3]


def capture_context(session, video):
    path = session / 'meta/collect_info.jsonl'
    if not path.is_file():
        return None
    raw = path.read_bytes()
    if len(raw) > 1_000_000:
        raise ValueError(f'Capture metadata too large: {path}')
    index = int(video.stem.removeprefix('episode_'))
    rows = [json.loads(s) for s in raw.decode('utf-8-sig').splitlines() if s.strip()]
    selected = [r for r in rows if r.get('episode_index') == index]
    if len(selected) != 1:
        raise ValueError(f'Expected one collect_info record for episode {index}')
    row = selected[0]
    context = {k: row[k] for k in ('episode_index', 'session', 'scene', 'task_name', 'notes', 'objects') if k in row}
    json.dumps(context, allow_nan=False)
    return dict(source_file=str(path), sha256=hashlib.sha256(raw).hexdigest(), context=context)


def plan(video, arguments, note=None, structured_only=False):
    video = Path(video).resolve()
    session = video_session(video)
    snapshot = capture_context(session, video)
    execution = execution_start(session, video)
    if any(a.split('=')[0] == '--annotation-start-s' for a in arguments):
        raise ValueError('L3 annotation start is managed by EXEC_START')
    from review.capture_context import dataset_note
    combined = dataset_note(note, snapshot)
    work_root = Path(os.environ.get('ARGUS_L3_WORK_ROOT') or
                     (Path.home() / '.cache/argus/L3' / hashlib.sha256(str(session).encode()).hexdigest()))
    job = work_root / 'annotation/argus' / f'{video.parent.parent.name}_{video.stem}'
    command = [sys.executable, '-m', 'review', '--data', str(video), '--rig', 'ego_head',
               '--out', str(job), '--parent-prepared-job', *arguments,
               '--annotation-start-s', str(execution['start_s'])]
    if combined:
        command.extend(['--dataset-note', combined])
    identity = dict(video=str(video), capture=snapshot, execution=execution)
    return dict(video=str(video), job=str(job), command=command, identity=identity,
                structured_only=structured_only)


def run_one(item):
    job = Path(item['job'])
    job.mkdir(parents=True, exist_ok=True)
    lock = job / '.l3_review.lock'
    try:
        with lock.open('x', encoding='utf-8') as stream:
            stream.write(str(os.getpid()))
    except FileExistsError:
        return dict(video=item['video'], status='failed', error=f'Job locked: {lock}')
    try:
        receipt = job / 'l3_input.json'
        if receipt.exists():
            if json.loads(receipt.read_text(encoding='utf-8')) != item['identity']:
                raise ValueError('L3 capture metadata changed; use a fresh annotation job')
        else:
            receipt.write_text(json.dumps(item['identity'], ensure_ascii=False, indent=2), encoding='utf-8')
        log = job / 'l3_review.log'
        print(f'[start] {item["video"]} -> {job}', flush=True)
        with log.open('a', encoding='utf-8') as stream:
            result = subprocess.run(item['command'], cwd=ROOT, stdout=stream,
                                    stderr=subprocess.STDOUT, check=False)
        return dict(video=item['video'], status='done' if result.returncode == 0 else 'failed',
                    returncode=result.returncode, job=str(job), log=str(log))
    except (OSError, ValueError) as exc:
        return dict(video=item['video'], status='failed', error=str(exc))
    finally:
        lock.unlink(missing_ok=True)



_original_run_one = run_one


def publish_structured(item):
    """Publish only final structured JSON; work artifacts stay outside the dataset."""
    source = Path(item['job']).parent.parent
    target = video_session(item['video']) / 'annotation'
    target.mkdir(parents=True, exist_ok=True)
    for path in source.glob('episode_*_structured.json'):
        temporary = target / ('.' + path.name + '.tmp')
        shutil.copyfile(path, temporary)
        os.replace(temporary, target / path.name)


def run_one(item):
    if STOP.is_set():
        return dict(video=item['video'], status='failed', error='Batch interrupted; task not started')
    from review.overwrite import job_lock, archive_job
    job = Path(item['job'])
    try:
        with job_lock(job.with_name(job.name + '.adapter')):
            if '--force' not in item['command']:
                receipt = job / 'l3_input.json'
                if receipt.exists() and json.loads(receipt.read_text(encoding='utf-8')) != item['identity']:
                    raise ValueError('L3 input/EXEC_START policy changed; rerun with --force to archive prior annotation')
                from review.completed import completed_result
                with job_lock(job):
                    complete = completed_result(job)
                    if complete and item.get('structured_only'):
                        from review.retention import compact
                        complete = compact(job, annotation_output=True)
                if complete:
                    if item.get('structured_only'):
                        publish_structured(item)
                    return dict(video=item['video'], job=str(job), **complete)
                if (job / 'structured_complete.json').exists():
                    raise ValueError('Compacted structured outputs failed verification; inspect outputs or use --force')
                if '--free' not in item['command'] and '--retry-received' not in item['command']:
                    from review.overwrite import clear_incomplete_job
                    with job_lock(job):
                        clear_incomplete_job(job)
            if '--force' in item['command']:
                if '--free' in item['command']:
                    raise ValueError('--force cannot be combined with --free')
                with job_lock(job):
                    # The final JSON lives directly under annotation; archive it
                    # together with its job before a forced model change.
                    annotation = job.parent.parent
                    if job.parent.name != 'argus' or annotation.name != 'annotation':
                        raise ValueError('Expected annotation/argus/job layout')
                    archive_job(job, annotation / f'{Path(item["video"]).stem}_structured.json')
                item = {**item, 'command': [a for a in item['command'] if a != '--force']}
            result = _original_run_one(item)
            if result['status'] == 'done' and item.get('structured_only'):
                from review.retention import compact
                with job_lock(job):
                    compact(job, annotation_output=True)
                publish_structured(item)
                result.pop('log', None)
            return result
    except (OSError, RuntimeError, ValueError) as exc:
        return dict(video=item['video'], status='failed', error=str(exc))


_run_with_workfiles = run_one


def run_one(item):
    final = video_session(item['video']) / 'annotation' / f'{Path(item["video"]).stem}_structured.json'
    if item.get('structured_only') and '--force' not in item['command'] and final.is_file():
        from review.base.run import validate
        validate(json.loads(final.read_text(encoding='utf-8')))
        result = dict(video=item['video'], status='done', outputs=[str(final)])
    else:
        result = _run_with_workfiles(item)
    if result.get('status') == 'done' and item.get('structured_only'):
        job = Path(item['job'])
        if job.exists():
            shutil.rmtree(job)
        (job.parent.parent / final.name).unlink(missing_ok=True)
        for name in (f'.{job.name}.execution.lock', f'.{job.name}.adapter.execution.lock'):
            (job.parent / name).unlink(missing_ok=True)
        for parent in (job.parent, job.parent.parent, job.parent.parent.parent):
            try:
                parent.rmdir()
            except OSError:
                break
        result.pop('log', None)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--source', type=Path, help='L3 session, batch or root directory')
    inputs.add_argument('--data', type=Path, help='One L3 rectified left-eye MP4')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--dataset-note')
    parser.add_argument('--structured-only', action='store_true',
                        help='After verified success, retain structured JSON and reuse metadata only')
    parser.add_argument('--sessions-file', type=Path, help='JSON allowlist of sessions under --source')
    parser.add_argument('--dry-run', action='store_true', help='Print plans; no files or model calls')
    args, extra = parser.parse_known_args(argv)
    STOP.clear()
    if args.workers < 1:
        parser.error('workers must be positive')
    if any(x.split('=')[0] in ('--out', '--rig') for x in extra):
        parser.error('--out and --rig are managed by this L3 adapter')
    if args.sessions_file and not args.source:
        parser.error('--sessions-file requires --source')
    try:
        if args.data:
            videos = [args.data]
        else:
            source = args.source.resolve()
            if not source.is_dir():
                raise ValueError(f'Source directory does not exist: {source}')
            if args.sessions_file:
                values = json.loads(args.sessions_file.read_text(encoding='utf-8'))
                if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
                    raise ValueError('sessions-file must be a JSON list of paths')
                selected = {Path(v).resolve() for v in values}
                if any(not s.is_relative_to(source) for s in selected):
                    raise ValueError('Selected session outside source')
                videos = sorted(v for session in selected
                                for v in (session / 'videos').glob(f'chunk-*/{CAMERA}/episode_*.mp4')
                                if video_session(v) in selected)
            else:
                videos = sorted(p for p in source.rglob(f'{CAMERA}/episode_*.mp4')
                                if not any(x.startswith('.') for x in p.relative_to(source).parts))
        if not videos:
            raise ValueError(f'No {CAMERA} videos found; run L3 rectification first')
        # Preparation can retain symlinks to the original video under annotation.
        # Resolve aliases before planning: --force must still run each source once.
        videos = sorted({Path(video).resolve() for video in videos})
        plans = [plan(v, extra, args.dataset_note, args.structured_only) for v in videos]
        if len({item['job'] for item in plans}) != len(plans):
            raise ValueError('Multiple source videos map to the same annotation job')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        for item in plans:
            print(json.dumps(item, ensure_ascii=False), flush=True)
        return 0
    failed = 0
    pool = ThreadPoolExecutor(max_workers=args.workers)
    futures = [pool.submit(run_one, item) for item in plans]
    try:
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                result = dict(status='failed', error=str(exc))
            failed += result['status'] == 'failed'
            print(json.dumps(result, ensure_ascii=False), flush=True)
    except BaseException:
        STOP.set()
        for future in futures:
            future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    print(json.dumps(dict(total=len(plans), failed=failed)))
    return int(bool(failed))


if __name__ == '__main__':
    raise SystemExit(main())

# /share/project/liuyou/humancoin/envs/ego/bin/python \
#   /share/project/liuyou/humancoin/code/argus/review/L3/run.py \
#   --source /share/project/liuyou/humancoin/data/L3_lerobot/batch_0000001 \
#   --workers 4 --no-board --model qwen3.8-27b --reasoning none \
#   --sample-fps 2 --max-tokens 16384 --concurrency 1 --cap 0 --estimated-cap-cny 0.6
# Add --dry-run to inspect plans without calling the model or writing files.
# DASHSCOPE_API_KEY and DASHSCOPE_BASE_URL must be configured in the environment.
