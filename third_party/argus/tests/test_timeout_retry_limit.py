import io
from pathlib import Path
import urllib.error
from unittest.mock import patch
import pytest
import ast
import json
import time
import urllib.request
from types import SimpleNamespace
source=Path(__file__).resolve().parents[1]/'label/harness.py'
tree=ast.parse(source.read_text(encoding='utf-8'))
ns=dict(urllib=urllib,json=json,time=time,is_key_exhausted=lambda *a:False)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_post'],type_ignores=[]),str(source),'exec'),ns)
harness=SimpleNamespace(_post=ns['_post'],urllib=urllib,time=time)


def test_video_timeout_stops_after_three_requests():
    with patch.object(harness.urllib.request,'urlopen',side_effect=TimeoutError('timeout')) as request, patch.object(harness.time,'sleep'):
        with pytest.raises(RuntimeError,match='3 HTTP attempts'):
            harness._post('https://example.test',{},'key',120,attempts=3)
    assert request.call_count==3


def test_content_rejection_is_not_retried():
    error=urllib.error.HTTPError('https://example.test',400,'rejected',{},io.BytesIO(b'DataInspectionFailed'))
    with patch.object(harness.urllib.request,'urlopen',side_effect=error) as request:
        with pytest.raises(RuntimeError,match='DataInspectionFailed'):
            harness._post('https://example.test',{},'key',120,attempts=3)
    assert request.call_count==1
