"""Bounded retries for model dispatch plus parsing and structured validation."""
import http.client
import os
import time
import urllib.error


def max_attempts():
    value = int(os.environ.get('ARGUS_STRUCTURED_MAX_ATTEMPTS', '3'))
    if not 1 <= value <= 10:
        raise ValueError('ARGUS_STRUCTURED_MAX_ATTEMPTS must be between 1 and 10')
    return value


def run(operation, *, receipt=None, on_attempt=None):
    from jsonschema import ValidationError
    limit = max_attempts()
    history = []
    if receipt is not None:
        receipt['structured_attempts'] = history
    for attempt in range(1, limit + 1):
        record = dict(attempt=attempt, status='dispatching')
        history.append(record)
        if on_attempt:
            on_attempt(history)
        try:
            value = operation()
        except Exception as exc:
            record.update(status='failed', error_type=type(exc).__name__)
            if on_attempt:
                on_attempt(history)
            if isinstance(exc, urllib.error.HTTPError):
                retryable = exc.code in (408, 409, 425, 429, 500, 502, 503, 504)
            else:
                retryable = isinstance(exc, (TimeoutError, ConnectionError, urllib.error.URLError,
                    http.client.HTTPException, ValueError, KeyError, IndexError, TypeError, AttributeError, ValidationError))
            if not retryable or attempt == limit:
                raise
            delay = min(2 ** attempt, 8)
            print(f'== structured retry {attempt}/{limit}: {type(exc).__name__}; waiting {delay}s', flush=True)
            time.sleep(delay)
        else:
            record['status'] = 'complete'
            if on_attempt:
                on_attempt(history)
            return value
