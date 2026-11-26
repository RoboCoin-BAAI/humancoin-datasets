"""Annotate one video, publish structured JSON, then remove verified work."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(__file__).resolve().parents[2]


def selected_standard(arguments):
    value = 1
    for index, argument in enumerate(arguments):
        if argument == '--annotation-standard':
            value = int(arguments[index + 1])
        elif argument.startswith('--annotation-standard='):
            value = int(argument.split('=', 1)[1])
    if value not in (1, 2, 3):
        raise ValueError('Annotation standard must be 1, 2 or 3')
    return value


def validate(data, standard=1):
    if standard >= 2:
        from review.standard2 import validate_final
        return validate_final(data, standard)
    from jsonschema import Draft202012Validator
    schema = json.loads((ROOT / 'review/schemas/episode_structured.schema.json').read_text(encoding='utf-8'))
    Draft202012Validator(schema).validate(data)
    from review.object_policy import validate_objects
    validate_objects(data['objects'])


def plan(video, output, arguments=(), keep_work=False):
    video = Path(video).resolve()
    output = Path(output).absolute()
    if not video.is_file() or video.suffix.lower() not in ('.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v'):
        raise ValueError('Expected an existing local video')
    if output.suffix.lower() != '.json' or output.resolve() == video:
        raise ValueError('Expected a separate output JSON path')
    arguments = list(arguments)
    if not any(a.split('=', 1)[0] == '--annotation-standard' for a in arguments):
        arguments += ['--annotation-standard', '3']
    selected_standard(arguments)
    reserved = {'--data', '--source', '--out', '--rig', '--workers', '--sessions-file', '--parent-prepared-job'}
    if any(a.split('=', 1)[0] in reserved for a in arguments):
        raise ValueError('Input, work directory and ego_head rig are managed by base')
    models = [a.split('=', 1)[1] if a.startswith('--model=') else arguments[i+1]
              for i, a in enumerate(arguments) if a == '--model' or a.startswith('--model=')]
    if models and not models[-1].startswith(('qwen', 'dashscope/')):
        raise ValueError('Structured extraction currently requires a Qwen model')
    if not models:
        arguments += ['--model', 'qwen3.7-plus']
    if '--no-board' not in arguments:
        arguments.append('--no-board')
    identity = hashlib.sha256((str(video) + '\n' + str(output)).encode()).hexdigest()
    job = Path(tempfile.gettempdir()) / 'argus-base' / identity
    if output.resolve().is_relative_to(job.resolve()) or video.is_relative_to(job.resolve()):
        raise ValueError('Input and final output must be outside the managed work directory')
    return dict(video=str(video), output=str(output), job=str(job), keep_work=keep_work,
                command=[sys.executable, '-m', 'review', '--data', str(video), '--out', str(job),
                         '--rig', 'ego_head', *arguments])


def execute(item):
    from review.completed import completed_result
    from review.overwrite import job_lock
    from review.structured import finish_claim
    output, job = Path(item['output']), Path(item['job'])
    standard = selected_standard(item['command'])
    # Only exact managed children may ever be removed. Check before dispatch too.
    expected = (Path(tempfile.gettempdir()) / 'argus-base').resolve()
    if job.is_symlink() or job.resolve().parent != expected:
        raise ValueError('Unsafe managed work directory')
    if output.is_symlink():
        raise ValueError('Output must not be a symlink')
    with job_lock(job.with_name(job.name + '.adapter')):
        if output.exists() and '--force' not in item['command']:
            validate(json.loads(output.read_text(encoding='utf-8')), standard=standard)
            return dict(status='done', output=str(output), reused=True)
        # Fail on missing schema dependencies before making model calls.
        from jsonschema import Draft202012Validator
        filename = 'episode_structured.standard2.schema.json' if standard >= 2 else 'episode_structured.schema.json'
        schema = json.loads((ROOT / 'review/schemas' / filename).read_text(encoding='utf-8'))
        Draft202012Validator.check_schema(schema)
        result = subprocess.run(item['command'], cwd=ROOT, check=False)
        if result.returncode:
            return dict(status='failed', returncode=result.returncode, job=str(job))
        if '--free' in item['command']:
            return dict(status='prepared', job=str(job))
        complete = completed_result(job)
        if not complete or len(complete['outputs']) != 1:
            raise ValueError(f'Expected one verified structured result; work retained at {job}')
        source = Path(complete['outputs'][0])
        if source.is_symlink() or source.resolve().parent != job.resolve():
            raise ValueError('Structured output escaped managed work directory')
        data = json.loads(source.read_text(encoding='utf-8'))
        validate(data, standard=standard)
        output.parent.mkdir(parents=True, exist_ok=True)
        finish_claim(output, data)
        if json.loads(output.read_text(encoding='utf-8')) != data:
            raise ValueError('Published output verification failed; work retained')
        if not item['keep_work']:
            if job.is_symlink() or job.resolve().parent != expected:
                raise ValueError('Unsafe cleanup target')
            shutil.rmtree(job)
        return dict(status='done', output=str(output), work_deleted=not item['keep_work'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--data', type=Path, required=True, help='One local video')
    parser.add_argument('--output', type=Path, required=True, help='Final structured JSON path')
    parser.add_argument('--keep-work', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args, extra = parser.parse_known_args(argv)
    try:
        item = plan(args.data, args.output, extra, args.keep_work)
        result = item if args.dry_run else execute(item)
    except Exception as exc:
        print(json.dumps(dict(status='failed', error=str(exc))), flush=True)
        return 1
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return int(result.get('status') == 'failed')


if __name__ == '__main__':
    raise SystemExit(main())
