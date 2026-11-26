import urllib.error
import pytest
from review import structured_retry as retry


@pytest.mark.parametrize('error', [TimeoutError(), urllib.error.URLError('offline'),
    ValueError('incomplete JSON'), KeyError('choices'),
    urllib.error.HTTPError('https://test', 429, 'limited', {}, None)])
def test_retry_then_success(monkeypatch, error):
    monkeypatch.setenv('ARGUS_STRUCTURED_MAX_ATTEMPTS', '3')
    monkeypatch.setattr(retry.time, 'sleep', lambda _: None)
    calls = []
    receipt = {}
    def operation():
        calls.append(1)
        if len(calls) == 1:
            raise error
        return 'ok'
    assert retry.run(operation, receipt=receipt) == 'ok'
    assert len(calls) == 2
    assert [a['status'] for a in receipt['structured_attempts']] == ['failed', 'complete']


def test_exhausted_and_auth_not_retried(monkeypatch):
    monkeypatch.setenv('ARGUS_STRUCTURED_MAX_ATTEMPTS', '3')
    monkeypatch.setattr(retry.time, 'sleep', lambda _: None)
    for code, expected in [(503, 3), (401, 1)]:
        calls = []
        def operation():
            calls.append(1)
            raise urllib.error.HTTPError('https://test', code, 'error', {}, None)
        with pytest.raises(urllib.error.HTTPError):
            retry.run(operation)
        assert len(calls) == expected


def test_budget_not_retried(monkeypatch):
    from review.cny_budget import BudgetExceeded
    calls=[]
    def operation():
        calls.append(1)
        raise BudgetExceeded('budget exhausted')
    with pytest.raises(BudgetExceeded):
        retry.run(operation)
    assert len(calls)==1
