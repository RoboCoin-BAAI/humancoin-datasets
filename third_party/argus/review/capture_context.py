"""Read optional L1 capture setup as attributed, unverified context."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from review.l1_paths import default_l1_output

MAX_CONFIG_BYTES = 1_000_000
OBJECT_FIELDS = ('name', 'lib_name', 'obj_id', 'color', 'color2', 'shape', 'dims', 'material', 'weight')


def snapshot_for_video(video: Path) -> dict | None:
    """Only inspect the annotation folder of this L1 video; configuration is optional."""
    try:
        annotation = default_l1_output(str(video))
    except ValueError:
        return None
    if not annotation.is_dir():
        return None
    matches = sorted(path for path in annotation.glob('*_config.yaml') if path.is_file())
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f'Multiple *_config.yaml files in {annotation}; cannot determine the matching configuration')
    path = matches[0]
    raw = path.read_bytes()
    if len(raw) > MAX_CONFIG_BYTES:
        raise ValueError(f'Capture configuration exceeds {MAX_CONFIG_BYTES} bytes: {path}')
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError('PyYAML is required to read capture configuration; install PyYAML==6.0.3') from error
    try:
        data = yaml.safe_load(raw.decode('utf-8-sig'))
    except (UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f'Cannot parse capture configuration: {path}') from error
    if not isinstance(data, dict) or not isinstance(data.get('scene'), dict):
        raise ValueError(f'Capture configuration must contain a scene object: {path}')
    scene = data['scene']
    objects = scene.get('objects', [])
    if not isinstance(objects, list) or any(not isinstance(item, dict) for item in objects):
        raise ValueError(f'Capture configuration scene.objects must be an array of objects: {path}')
    context = {'source_file': path.name, 'task': data.get('task'),
               'scene': {key: scene[key] for key in ('name', 'taskName', 'notes') if key in scene},
               'objects': [{key: item[key] for key in OBJECT_FIELDS if key in item} for item in objects]}
    try:
        json.dumps(context, ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError, RecursionError) as error:
        raise ValueError(f'Capture configuration has unsupported field values: {path}') from error
    return {'identity': {'file': path.name, 'sha256': hashlib.sha256(raw).hexdigest()}, 'context': context}


def dataset_note(note: str | None, snapshot: dict | None) -> str | None:
    if snapshot is None:
        return note
    data = json.dumps(snapshot['context'], ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    config_note = ('Capture setup supplied by the collector, not verified observations: ' + data
                   + '. Planned steps and listed objects or attributes are contextual hints only. '
                   'Describe actions, objects and outcomes only when supported by the video; '
                   'do not infer that a planned step happened or a listed object is visible.')
    return (note.strip() + '\n' + config_note) if note and note.strip() else config_note
