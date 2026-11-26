"""Conservative, approximate per-job Qwen budget, in millionths of a yuan.

This is not a provider billing limit. Unknown outcomes block further calls.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import os
from pathlib import Path
import uuid

ENV = 'ARGUS_ESTIMATED_CNY_LEDGER'


class BudgetExceeded(RuntimeError):
    """A request was refused before it was sent."""


def micro_yuan(value) -> int:
    try:
        n = Decimal(str(value)) * 1_000_000
        if not n.is_finite() or n < 0 or n != n.to_integral_value():
            raise ValueError('amount must be nonnegative with at most six decimal places')
        return int(n)
    except (InvalidOperation, TypeError) as error:
        raise ValueError('invalid CNY amount') from error


@contextmanager
def locked(path: Path):
    lock = path.with_name(path.name + '.lock')
    try:
        with lock.open('x', encoding='utf-8') as handle:
            handle.write(str(os.getpid()))
    except FileExistsError as error:
        raise RuntimeError(f'Budget ledger is locked: {lock}; inspect it before retrying') from error
    try:
        yield
    finally:
        lock.unlink()


def _write(path: Path, ledger: dict):
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with tmp.open('x', encoding='utf-8') as handle:
            json.dump(ledger, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def initialize(path: Path, *, source: Path, model: str, cap, reserve, prior_paid: bool):
    cap_micro, reserve_micro = micro_yuan(cap), micro_yuan(reserve)
    if cap_micro <= 0 or reserve_micro >= cap_micro:
        raise ValueError('estimated CNY cap must be positive and larger than the postprocess reserve')
    identity = {'version': 1, 'source_sha256': hashlib.sha256(str(source.resolve()).encode()).hexdigest(),
                'model': model.removeprefix('dashscope/'), 'cap_micro_cny': cap_micro,
                'postprocess_reserve_micro_cny': reserve_micro}
    with locked(path):
        if path.exists():
            ledger = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(ledger, dict) or any(ledger.get(k) != v for k, v in identity.items()):
                raise ValueError('Existing CNY budget ledger has different settings or source; use a fresh job')
            if not isinstance(ledger.get('calls'), list):
                raise ValueError('Invalid CNY budget ledger')
        else:
            if prior_paid:
                raise ValueError('Earlier paid work has no CNY ledger; use a fresh job to apply a budget')
            _write(path, {**identity, 'calls': [], 'note': 'Estimates only; actual provider bill may be higher.'})


def estimate(body: dict) -> tuple[int, int]:
    """Estimate prompt tokens from text bytes and twice the 32x32 image grid."""
    output = body.get('max_tokens')
    if type(output) is not int or output <= 0:
        raise ValueError('Budgeted requests require an explicit positive max_tokens')
    text_bytes, image_tokens = 0, 0

    def visit(value):
        nonlocal text_bytes, image_tokens
        if isinstance(value, str):
            text_bytes += len(value.encode('utf-8'))
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            if value.get('type') == 'image_url':
                url = value.get('image_url', {}).get('url', '')
                if not isinstance(url, str) or not url.startswith('data:image/') or ';base64,' not in url:
                    raise ValueError('Budgeted requests only support embedded images')
                from PIL import Image
                raw = base64.b64decode(url.split(',', 1)[1], validate=True)
                with Image.open(io.BytesIO(raw)) as img:
                    width, height = img.size
                image_tokens += 2 * (((max(width, 64) + 31) // 32) * ((max(height, 64) + 31) // 32) + 2)
            else:
                for key, item in value.items():
                    text_bytes += len(str(key).encode('utf-8'))
                    visit(item)

    visit(body.get('messages', []))
    return text_bytes + image_tokens + 1024, output


def execute(body: dict, stage: str, dispatch):
    """Reserve before dispatch; settle using reported usage, or retain a blocking claim."""
    path_value = os.environ.get(ENV, '')
    if not path_value:
        return dispatch()
    path = Path(path_value)
    if stage not in ('video', 'postprocess'):
        raise ValueError('Unknown budget stage')
    prompt, completion = estimate(body)
    estimate_micro = prompt * 3 + completion * 12
    transaction = uuid.uuid4().hex
    with locked(path):
        ledger = json.loads(path.read_text(encoding='utf-8'))
        if ledger.get('model') != body.get('model', '').removeprefix('dashscope/'):
            raise ValueError('Request model does not match CNY budget ledger')
        calls = ledger['calls']
        if any(item.get('status') != 'settled' for item in calls):
            raise RuntimeError('An earlier CNY-budgeted request has an unknown outcome; inspect the ledger before retrying')
        spent = sum(item['cost_micro_cny'] for item in calls)
        remaining = ledger['cap_micro_cny'] - spent
        reserved = ledger['postprocess_reserve_micro_cny'] if stage == 'video' else 0
        if estimate_micro + reserved > remaining:
            raise BudgetExceeded(f'CNY estimated budget: {stage} request needs CNY {estimate_micro / 1e6:.6f} '
                                 f'plus CNY {reserved / 1e6:.6f} reserve; CNY {remaining / 1e6:.6f} remains. Request not sent.')
        calls.append({'id': transaction, 'stage': stage, 'status': 'pending',
                      'estimated_prompt_tokens': prompt, 'max_output_tokens': completion,
                      'estimated_cost_micro_cny': estimate_micro})
        _write(path, ledger)
    print(f'== CNY budget: {stage} estimated CNY {estimate_micro / 1e6:.6f}; '
          f'postprocess reserve CNY {reserved / 1e6:.6f}', flush=True)
    response = dispatch()
    usage = response.get('usage') if isinstance(response, dict) else None
    if not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] < 0
                                          for k in ('prompt_tokens', 'completion_tokens')):
        raise RuntimeError('Qwen usage missing; request may have been billed and budget is blocked')
    cost = usage['prompt_tokens'] * 3 + usage['completion_tokens'] * 12
    with locked(path):
        ledger = json.loads(path.read_text(encoding='utf-8'))
        found = [item for item in ledger['calls'] if item.get('id') == transaction and item.get('status') == 'pending']
        if len(found) != 1:
            raise RuntimeError('CNY budget transaction could not be verified')
        found[0].update(status='settled', prompt_tokens=usage['prompt_tokens'],
                        completion_tokens=usage['completion_tokens'], cost_micro_cny=cost)
        _write(path, ledger)
    print(f'== CNY budget: reported token usage at list rates CNY {cost / 1e6:.6f}; actual bill may differ', flush=True)
    return response
