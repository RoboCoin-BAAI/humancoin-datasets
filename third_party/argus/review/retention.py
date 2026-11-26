"""Compact completed jobs without losing verified reuse information."""
import hashlib
import json
from pathlib import Path
import shutil


def compact(job, *, annotation_output=False):
    from review.completed import completed_result
    from review.structured import finish_claim

    job = Path(job).absolute()
    if job.is_symlink() or not job.is_dir():
        raise ValueError('Invalid review job')
    job = job.resolve()
    result = completed_result(job)
    if not result:
        raise ValueError('Structured outputs are not verified; retaining job artifacts')
    outputs = [Path(p) for p in result['outputs']]
    annotation = job.parent.parent
    if annotation_output and (job.parent.name != 'argus' or annotation.name != 'annotation'):
        raise ValueError('Expected annotation/argus/job layout')
    allowed = {job, annotation} if annotation_output else {job}
    if any(p.is_symlink() or p.parent.resolve() not in allowed for p in outputs):
        raise ValueError('Structured output must be inside the selected job')
    # Copy first; commit the new location before deleting the previous copy.
    # Refuse conflicting files instead of overwriting another episode's output.
    if annotation_output:
        for source in outputs:
            target = annotation / source.name
            if target.is_symlink():
                raise ValueError(f'Invalid structured output: {target}')
            if target != source and target.exists() and target.read_bytes() != source.read_bytes():
                raise ValueError(f'Conflicting structured output: {target}')
        for source in outputs:
            target = annotation / source.name
            if target != source and not target.exists():
                from review.extract_dimensions import write_new
                write_new(target, json.loads(source.read_bytes()))
        outputs = [annotation / p.name for p in outputs]
    keep = {p.name for p in outputs if p.parent == job} | {'structured_complete.json', 'l3_input.json', '.l3_review.lock'}
    # Validate every deletion target before removing anything.
    targets = [p for p in job.iterdir() if p.name not in keep]
    for target in targets:
        if target.parent.resolve() != job:
            raise ValueError(f'Unsafe cleanup target: {target}')
    finish_claim(job / 'structured_complete.json', dict(
        version=1, status='done', previous_model=result.get('previous_model'),
        output_location='annotation' if annotation_output else 'job',
        outputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs}))
    for target in targets:
        if target.is_symlink():
            target.unlink()
        elif target.is_dir():
            # rmtree unlinks contained symlinks without traversing their targets.
            shutil.rmtree(target)
        else:
            target.unlink()
    return completed_result(job)
