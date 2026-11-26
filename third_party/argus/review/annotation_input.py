"""Parse existing Argus video annotation evidence."""
import json
import hashlib
from pathlib import Path

def prepare(source: Path) -> dict:
    raw = source.read_bytes()
    data = json.loads(raw.decode('utf-8-sig'))
    labels = data.get('labels')
    if data.get('parse_ok') is not True or not isinstance(labels, dict):
        raise ValueError('Input must be an Argus output with parse_ok=true and labels')
    timeline = labels.get('timeline')
    if not isinstance(timeline, list):
        raise ValueError('labels.timeline is missing')
    if any((not isinstance(item, dict) for item in timeline)):
        raise ValueError('Each timeline item must be an object')
    if any(('labels' in item for item in timeline)):
        raise ValueError('A source timeline item already has labels; refusing to overwrite it')
    scene = labels.get('scene') or {}
    if not isinstance(scene, dict):
        raise ValueError('labels.scene must be an object')
    return {'schema_version': '1.0', 'episode_dir': data.get('episode_dir'), 'source': {'file': str(source.resolve()), 'sha256': hashlib.sha256(raw).hexdigest()}, 'timesteps_s': data.get('config', {}).get('timesteps_s', []), 'finish_reason': data.get('finish_reason'), 'objects': scene.get('objects', []), 'scene': scene.get('setting'), 'subtask': timeline, 'task_description': labels.get('task_summary'), **{name: labels.get(name, []) for name in ('key_events', 'state_changes', 'tasks', 'scene_graph', 'recovery', 'instruction_variants', 'data_issues')}, 'viewpoint': labels.get('viewpoint'), 'performance_review': labels.get('performance_review')}

def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')

def action_text(action, obj) -> str:
    """Join the source action and object without repeating an existing object phrase."""
    import re
    action = action.strip() if isinstance(action, str) else ''
    obj = obj.strip() if isinstance(obj, str) else ''
    if not obj:
        return action
    if not action:
        return obj
    tokens = obj.split()
    if re.search('(?<!\\w)' + re.escape(obj) + '(?!\\w)', action, re.IGNORECASE):
        return action
    if any((re.search('(?<!\\w)' + re.escape(' '.join(tokens[i:])) + '[.!?\\s]*$', action, re.IGNORECASE) for i in range(1, len(tokens)))):
        return action
    return f'{action} {obj}'
