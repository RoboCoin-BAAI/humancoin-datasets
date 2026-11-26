"""Review Jianzhi rectified left-eye videos with the shared ego_head pipeline."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from review.base import run as base

CAMERA = 'observation.images.head_left_rectified'


def plan(video, arguments):
    video = Path(video).resolve()
    camera, chunk, videos = video.parent, video.parent.parent, video.parent.parent.parent
    session = videos.parent
    if (not video.is_file() or video.suffix.lower() != '.mp4' or camera.name != CAMERA
            or not re.fullmatch(r'episode_\d+', video.stem)
            or not re.fullmatch(r'chunk-\d+', chunk.name) or videos.name != 'videos'
            or not (session / 'meta/info.json').is_file()):
        raise ValueError(f'Expected a Jianzhi head_left_rectified episode MP4: {video}')
    output = session / 'annotation' / f'{video.stem}_structured.json'
    arguments = list(arguments)
    keep_work = '--keep-work' in arguments
    arguments = [a for a in arguments if a != '--keep-work']
    return base.plan(video, output, ['--dataset', session.name, *arguments], keep_work=keep_work)


def execute(item):
    return dict(base.execute(item), video=item['video'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--source', type=Path, help='Dataset or batch tree')
    inputs.add_argument('--data', type=Path, help='One rectified left-eye MP4')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--dry-run', action='store_true', help='Print commands; no files or model calls')
    args, extra = parser.parse_known_args(argv)
    reserved = {'--out', '--output', '--rig', '--dataset', '--source', '--data', '--workers', '--sessions-file'}
    if any(arg.split('=', 1)[0] in reserved for arg in extra):
        parser.error('Output, rig and dataset are managed by the Jianzhi entry point')
    if args.workers < 1:
        parser.error('--workers must be positive')
    try:
        if args.data:
            paths = [args.data]
        else:
            source = args.source.resolve()
            if not source.is_dir():
                raise ValueError(f'Source is not a directory: {source}')
            paths = sorted(p for p in source.rglob(f'{CAMERA}/episode_*.mp4')
                           if not any(part.startswith('.') for part in p.relative_to(source).parts))
        if not paths:
            raise ValueError('No rectified left-eye videos found; run Jianzhi rectification first')
        items = [plan(p, extra) for p in paths]
        if len({x['job'] for x in items}) != len(items):
            raise ValueError('Multiple inputs map to the same job')
        if len({x['output'] for x in items}) != len(items):
            raise ValueError('Repeated episode names across chunks map to the same structured output')
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        for item in items:
            print(json.dumps(item, ensure_ascii=False), flush=True)
        return 0
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute, item): item for item in items}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                result = dict(video=futures[future]['video'], status='failed', error=str(exc))
            failures += result['status'] == 'failed'
            print(json.dumps(result, ensure_ascii=False), flush=True)
    print(json.dumps(dict(total=len(items), failed=failures)))
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
