"""Create one structured Qwen result per completed Argus episode."""
from __future__ import annotations

from functools import partial

import hashlib
import json
import os
from pathlib import Path
import tempfile

from review import annotation_input as extraction


def finish_claim(claim: Path, value: dict) -> None:
    """Atomically replace a dispatch claim only after the output has been saved."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=claim.parent,
                                         prefix=f'.{claim.name}.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, claim)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def run(job: Path, model: str, timeout: int = 120, *, capture_snapshot=None, capture_video=None,
        output_dir: Path | None = None, standard: int = 3) -> list[Path]:
    if standard != 3:
        raise ValueError('Only the current annotation standard (3) is supported')
    from review import standard as extractor
    def verify_capture():
        if capture_video is not None:
            from review.capture_context import snapshot_for_video
            if snapshot_for_video(capture_video) != capture_snapshot:
                raise RuntimeError('Capture configuration changed during structured extraction; no result was written')

    verify_capture()
    from review.qwen_endpoint import settings
    key, endpoint, provider = settings(model)
    processor = extractor.signature(standard)
    sources = sorted((job / 'run_final' / 'out').glob('episode_*.json'))
    if not sources:
        raise ValueError('No completed episode output is available for structured extraction')
    pending, outputs = [], []
    # Validate all input and existing output before dispatching any extra paid calls.
    for source in sources:
        prepared = extraction.prepare(source)
        output = (output_dir if output_dir is not None else job) / f'{source.stem}_structured.json'
        claim = job / f'.{source.stem}_structured.claim.json'
        identity = {'source_sha256': prepared['source']['sha256'], 'model': model,
                    'processor_sha256': processor}
        if standard >= 2:
            identity['annotation_standard'] = standard
        if capture_snapshot is not None:
            identity['capture_config'] = capture_snapshot['identity']
        if output.exists():
            try:
                raw = output.read_bytes()
                data = json.loads(raw.decode('utf-8'))
                meta = json.loads(claim.read_text(encoding='utf-8'))
            except (OSError, ValueError) as error:
                raise FileExistsError(f'Existing structured result has no verifiable receipt: {output}') from error
            if (not isinstance(data, dict) or 'extraction' in data or not isinstance(meta, dict)
                    or meta.get('status') != 'complete'
                    or meta.get('output_sha256') != hashlib.sha256(raw).hexdigest()
                    or any(meta.get(k) != v for k, v in identity.items())):
                raise FileExistsError(f'Existing structured result has a different or unverifiable identity: {output}')
            print(f'== structured result reused: {output}', flush=True)
        else:
            if claim.exists():
                raise RuntimeError(f'An earlier structured call may have been billed: {claim}; '
                                   'inspect its outcome before explicitly retrying')
            pending.append((prepared, output, claim, identity))
        outputs.append(output)
    for prepared, output, claim, identity in pending:
        verify_capture()
        local, missing = extractor.completion_plan(prepared)
        finish_claim(job / f'{output.stem}_draft.json', dict(
            prepared=prepared, resolved=local, pending=missing))
        print(f'== structured preparation: model={model}, '
              f'pending_actions={len(missing["timeline"])}, '
              f'pending_objects={len(missing["objects"])}, '
              f'pending_scene={local["scene"] is None}, timeout_s={timeout}', flush=True)
        # The claim is durable before dispatch and deliberately survives failures, preventing paid auto-retries.
        with claim.open('x', encoding='utf-8') as stream:
            json.dump({**identity, 'status': 'dispatching'}, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        print(f'== structured extraction: {output.name}', flush=True)
        receipt = {}
        try:
            verify_capture()
        except Exception:
            claim.unlink()
            raise
        try:
            if standard >= 2:
                validate_schema = partial(extractor.validate_final, standard=standard)
            else:
                from review.base.run import validate as validate_schema
            call = partial(extractor.call_qwen, standard=3) if standard == 3 else extractor.call_qwen
            result = call(prepared, model, endpoint, key, timeout,
                                          receipt=receipt, result_validator=validate_schema,
                                          on_attempt=lambda history: finish_claim(claim, {**identity,
                                              'status': 'dispatching', 'structured_attempts': history}))
        except Exception as error:
            from review.cny_budget import BudgetExceeded
            if isinstance(error, BudgetExceeded):
                # This particular exception guarantees no request was sent.
                claim.unlink()
            raise
        verify_capture()
        if capture_snapshot is not None:
            result['capture_context'] = capture_snapshot['context']
        extraction.write_new(output, result)
        finish_claim(claim, {**identity, 'status': 'complete', 'output_sha256':
                             hashlib.sha256(output.read_bytes()).hexdigest(), **receipt})
    return outputs
