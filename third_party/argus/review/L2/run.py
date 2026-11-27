"""Annotate L2 rectified left videos over the full video timeline using HumanCOIN Actions V3."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from review.L3.run import CAMERA, capture_context
from review.base.run import execute, plan
from review.capture_context import dataset_note


MODEL = 'qwen3.7-plus'


def make_plan(video, options):
    video = video.resolve()
    session = video.parents[3]
    if (not video.is_file() or video.parent.name != CAMERA or video.parents[2].name != 'videos'
            or not re.fullmatch(r'chunk-\d+', video.parents[1].name)
            or not re.fullmatch(r'episode_\d+\.mp4', video.name)
            or not re.fullmatch(r'L2-\d{3}-\d{4}(?:-\d{2}){5}', session.name)):
        raise ValueError(f'Expected an L2 rectified left video: {video}')
    note = dataset_note(None, capture_context(session, video))
    arguments = ['--model', MODEL, '--annotation-standard', '3', '--no-board', *options]
    if not any(arg.split('=', 1)[0] == '--sample-fps' for arg in options):
        arguments.extend(['--sample-fps', '2'])
    if note:
        arguments.extend(['--dataset-note', note])
    output = session / 'annotation' / f'{video.stem}_structured.json'
    return plan(video, output, arguments)


def resume_structured(item):
    """Retry a saved, rejected V3 response without relabeling the video."""
    import hashlib
    from review.base.run import validate
    from review.overwrite import job_lock
    from review.structured import finish_claim, run

    job, output = Path(item['job']), Path(item['output'])
    with job_lock(job.with_name(job.name + '.adapter')), job_lock(job):
        if output.is_file():
            validate(json.loads(output.read_text(encoding='utf-8')), 3)
            return {'status': 'done', 'output': str(output), 'reused': True}
        sources = list((job / 'run_final/out').glob('episode_*.json'))
        if len(sources) != 1:
            raise ValueError(f'Expected one saved video annotation in {job}')
        source = sources[0]
        claim = job / f'.{source.stem}_structured.claim.json'
        if claim.exists():
            receipt = json.loads(claim.read_text(encoding='utf-8'))
            attempts = receipt.get('structured_attempts', [])
            if receipt.get('status') != 'complete':
                if (receipt.get('model') != MODEL or receipt.get('annotation_standard') != 3
                        or receipt.get('source_sha256') != hashlib.sha256(source.read_bytes()).hexdigest()
                        or not attempts or any(a.get('status') != 'failed'
                                              or a.get('error_type') not in ('ValueError', 'JSONDecodeError', 'ValidationError')
                                              for a in attempts)):
                    raise ValueError(f'Cannot verify that the prior model response was rejected: {claim}')
                from uuid import uuid4
                claim.rename(claim.with_name(f'{claim.name}.{uuid4().hex}.rejected'))
        paths = run(job, MODEL, standard=3)
        if len(paths) != 1:
            raise ValueError(f'Expected one structured result in {job}')
        data = json.loads(paths[0].read_text(encoding='utf-8'))
        validate(data, 3)
        output.parent.mkdir(parents=True, exist_ok=True)
        finish_claim(output, data)
        return {'status': 'done', 'output': str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--data', type=Path)
    source.add_argument('--sessions-file', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume-structured', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args, extra = parser.parse_known_args(argv)
    if args.resume_structured and (not args.data or extra):
        parser.error('--resume-structured requires --data and accepts no video model options')
    if args.workers < 1:
        parser.error('--workers must be positive')
    managed = {'--model', '--structured-model', '--annotation-start-s', '--annotation-standard',
               '--dataset-note', '--no-structured', '--structured-output-dir'}
    if any(token.split('=', 1)[0] in managed for token in extra):
        parser.error('L2 manages its model, full video timeline, capture context and V3 standard')
    try:
        if args.data:
            videos = [args.data]
        else:
            sessions = json.loads(args.sessions_file.read_text(encoding='utf-8'))
            if not isinstance(sessions, list) or any(not isinstance(s, str) for s in sessions):
                raise ValueError('Expected a JSON list of session paths')
            videos = []
            for session in sorted({Path(s).resolve() for s in sessions}):
                found = sorted((session / 'videos').glob(f'chunk-*/{CAMERA}/episode_*.mp4'))
                if not found:
                    print(f'Missing rectified left video: {session}', file=sys.stderr, flush=True)
                videos.extend(found)
        plans = [make_plan(video, extra) for video in videos]
        if not plans or len({item['output'] for item in plans}) != len(plans):
            raise ValueError('Expected available videos with unique output paths')
    except (OSError, ValueError, KeyError, IndexError) as exc:
        parser.error(str(exc))
    print(f'{len(plans)} videos, model={MODEL}, standard=3, timeline=full video', flush=True)
    if args.dry_run:
        for item in plans:
            print(json.dumps(item, ensure_ascii=False), flush=True)
        return 0
    if args.resume_structured:
        print(json.dumps(resume_structured(plans[0]), ensure_ascii=False), flush=True)
        return 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute, item): item for item in plans}
        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {'status': 'failed', 'error': str(exc)}
            failed += result['status'] == 'failed'
            print(json.dumps({'video': item['video'], **result}, ensure_ascii=False), flush=True)
    return int(bool(failed))


if __name__ == '__main__':
    raise SystemExit(main())
