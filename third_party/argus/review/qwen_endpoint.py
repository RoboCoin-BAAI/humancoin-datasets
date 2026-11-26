"""Explicit gateway routing, shared by video and structured Qwen requests."""
import os
from urllib.parse import urlsplit


def gateway(model=None):
    return (model or '').removeprefix('dashscope/') == 'qwen3.6-plus' or bool(os.environ.get('XMSXB_API_KEY', '').strip())


def settings(model=None):
    if gateway(model):
        key = os.environ.get('XMSXB_API_KEY', '').strip()
        base = os.environ.get('XMSXB_BASE_URL', 'https://zyapi.xmsxb.com/v1').strip()
        provider = 'XMSXB'
        if not key:
            raise ValueError('Set XMSXB_API_KEY to the actual API key, not its display name')
    else:
        key = os.environ.get('DASHSCOPE_API_KEY', '').strip()
        base = os.environ.get('DASHSCOPE_BASE_URL', '').strip()
        provider = 'DashScope'
        if not key or not base:
            raise ValueError('Set DASHSCOPE_API_KEY and DASHSCOPE_BASE_URL')
    base = base.rstrip('/')
    if not base.endswith('/chat/completions'):
        if urlsplit(base).path in ('', '/'):
            base += '/v1'
        base += '/chat/completions'
    return key, base, provider
