"""One text-only API test. Python standard library; no Argus dependencies."""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from urllib.parse import urlsplit


def main(argv=None):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='replace')
        sys.stderr.reconfigure(errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='qwen3.6-plus')
    parser.add_argument('--base-url', default=os.environ.get('XMSXB_BASE_URL', 'https://zyapi.xmsxb.com/v1'))
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args(argv)
    key = os.environ.get('XMSXB_API_KEY', '').strip()
    if not key:
        parser.error('Set XMSXB_API_KEY to your actual API key first.')
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    endpoint = args.base_url.strip().rstrip('/')
    if not endpoint.endswith('/chat/completions'):
        if urlsplit(endpoint).path in ('', '/'):
            endpoint += '/v1'
        endpoint += '/chat/completions'
    body = {'model': args.model, 'messages': [{'role': 'user', 'content': 'Reply with only OK.'}],
            'max_tokens': 128, 'stream': False}
    request = urllib.request.Request(endpoint, data=json.dumps(body).encode('utf-8'),
                                     headers={'Authorization': 'Bearer ' + key,
                                              'Content-Type': 'application/json'}, method='POST')
    print(f'Text-only test: {endpoint}\nModel: {args.model}', flush=True)
    try:
        # Exactly one request; no automatic paid retries.
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            status = response.status
            raw = response.read().decode('utf-8', errors='replace')
        payload = json.loads(raw)
        if payload.get('error'):
            raise ValueError('API error: ' + json.dumps(payload['error'], ensure_ascii=False))
        choice = payload['choices'][0]
        message = choice['message']
        content = message.get('content')
        reasoning = message.get('reasoning_content')
        print(f'HTTP: {status}\nReturned model: {payload.get("model", "unknown")}')
        print('Reply:', content or '(empty)')
        print('Finish reason:', choice.get('finish_reason'))
        print('Usage:', json.dumps(payload.get('usage', {}), ensure_ascii=False))
        if not content:
            print('No text reply. Reasoning returned:', bool(reasoning))
            print('API responded, but a normal text completion was not verified.')
            return 1
        print('PASS: API returned a text completion.')
        return 0
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode('utf-8', errors='replace')[:3000].replace(key, '[REDACTED]')
        print(f'FAIL: HTTP {exc.code}\n{detail}', file=sys.stderr)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        print('FAIL: ' + str(exc).replace(key, '[REDACTED]'), file=sys.stderr)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
