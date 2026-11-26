"""Backfill task_name and functional scene classification in historical annotations.

Default is preview without API calls. --execute backs up and updates files.
Only final annotation/episode_*_structured.json files are discovered recursively.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from review.extract_dimensions import SCENES, PROMPT
from review.qwen_endpoint import settings, gateway


def validate(response):
    if not isinstance(response, dict):
        raise ValueError('Expected object response')
    name = response.get('task_name')
    if (not isinstance(name, str) or not name.strip() or len(name) > 100
            or len(name.split()) > 12 or '\n' in name or '\r' in name):
        raise ValueError('Invalid task_name')
    scene = response.get('scene')
    if (not isinstance(scene, dict) or not isinstance(scene.get('category'), str)
            or scene['category'] not in SCENES
            or scene.get('area') not in SCENES[scene['category']]):
        raise ValueError('Invalid scene category/area')
    return name.strip(), scene


def infer(data, args):
    key, endpoint, _ = settings(args.model)
    if not endpoint.endswith('/chat/completions'):
        endpoint = endpoint.rstrip('/') + '/chat/completions'
    evidence = {k: data[k] for k in ('task_description', 'instruction_variants',
        'subtask', 'tasks', 'objects', 'scene', 'capture_context') if k in data}
    body = {'model': args.model.removeprefix('dashscope/'), 'temperature': 0,
        'max_tokens': 1024, 'messages': [
            {'role': 'system', 'content': PROMPT + '''
This is a HISTORICAL BACKFILL, not action normalization. Return ONLY
{"task_name":"English imperative verb + noun phrase", "scene":{"category":"...","area":"..."}}.
Use the supplied existing annotation as task_context. Choose a non-null area
from the taxonomy, using other/unknown only when the activity is unidentifiable.
Reconsider old conservative classifications from task evidence. Do not output
actions, objects, visual descriptions, model identity or any other fields.
'''}, {'role': 'user', 'content': json.dumps({'scenes': SCENES, 'task_context': evidence}, ensure_ascii=False)}]}
    if not gateway(args.model):
        body['chat_template_kwargs'] = {'enable_thinking': False}
    request = urllib.request.Request(endpoint, data=json.dumps(body).encode('utf-8'),
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=args.timeout) as stream:
        result = json.load(stream)
    choice = result['choices'][0]
    if choice.get('finish_reason') not in (None, 'stop'):
        raise ValueError('Incomplete model response')
    content = choice['message']['content'].strip()
    if content.startswith('```'):
        content = content.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    return validate(json.loads(content))


def process(path, args):
    try:
        raw = path.read_bytes()
        data = json.loads(raw.decode('utf-8-sig'))
        marker = data.get('task_scene_backfill', {})
        if marker.get('version') == 1 and not args.force:
            print('SKIP', path, flush=True)
            return 'skipped'
        if not args.execute:
            print('PLAN', path, flush=True)
            return 'planned'
        name, scene = infer(data, args)
        if not data.get('task_name'):
            data['task_name'] = name
        # Keep the original visual description and all object/action/timing fields.
        data['scene_classification'] = {**scene, 'original_description': data.get('scene')}
        data['task_scene_backfill'] = {'version': 1, 'basis': 'annotation_text_with_task_inference',
                                     'input_sha256': hashlib.sha256(raw).hexdigest()}
        if path.read_bytes() != raw:
            raise ValueError('Input changed during request; refusing overwrite')
        backup = path.with_name(path.name + '.before_task_scene.bak')
        try:
            with backup.open('xb') as stream:
                stream.write(raw)
        except FileExistsError:
            pass
        fd, tmp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, path)
        finally:
            Path(tmp).unlink(missing_ok=True)
        print('UPDATED', path, flush=True)
        return 'updated'
    except Exception as exc:
        # HTTP errors are printed without response bodies or credentials.
        print('FAILED', path, str(exc), file=sys.stderr, flush=True)
        return 'failed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', default='qwen3.8-27b')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--force', action='store_true', help='Reclassify scenes in previously backfilled files')
    args = parser.parse_args()
    if not args.root.exists() or args.workers < 1 or args.timeout <= 0:
        parser.error('Invalid root/workers/timeout')
    paths = ([args.root] if args.root.is_file() else
             [p for p in sorted(args.root.rglob('episode_*_structured.json')) if p.parent.name == 'annotation'])
    if not paths:
        parser.error('No final structured annotations found')
    if args.execute:
        settings(args.model)  # Validate configuration before starting workers.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda path: process(path, args), paths))
    print(json.dumps({k: results.count(k) for k in ('planned', 'updated', 'skipped', 'failed')}))
    return int('failed' in results)


if __name__ == '__main__':
    sys.exit(main())
