"""The reconciliation endpoint is private, bounded and entirely nonexecuting."""
from copy import deepcopy
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from api.routers import ide
from services.semantic.source_reconciliation import reconcile_capture
from tests.api.test_ide_router import client
from tests.source_review.test_source_reconciliation import observed


def test_endpoint_returns_the_same_observed_source_proof_without_reading_or_executing(client, observed, monkeypatch):
    import builtins
    import io
    import subprocess
    expected = reconcile_capture(observed).model_dump(mode='json')
    # Load the HTTP test runtime before prohibiting execution and import I/O.
    assert client.post('/api/ide/capture/reconcile', json=observed).status_code == 200
    def forbidden(*args, **kwargs):
        raise AssertionError('reconciliation must not read files or start execution')
    # Compiling a syntax-checked proposal is static; executing that code is not.
    monkeypatch.setattr(builtins, 'exec', forbidden)
    monkeypatch.setattr(builtins, 'eval', forbidden)
    monkeypatch.setattr(builtins, 'open', forbidden)
    monkeypatch.setattr(io, 'open', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    response = client.post('/api/ide/capture/reconcile', json=observed)
    assert response.status_code == 200, response.text
    assert response.headers['cache-control'] == 'no-store'
    assert response.json() == expected


@pytest.mark.parametrize('kind', ['json', 'source', 'intent', 'capture', 'extra'])
def test_invalid_reconciliation_never_echoes_source_or_validation_errors(client, observed, kind):
    payload = deepcopy(observed); secret = 'private_reconciliation_value'
    if kind == 'source': payload['sources']['model.py'] = secret
    if kind == 'intent': payload['intent']['value'] = secret
    if kind == 'capture': payload['current']['operations'][0]['target'] = {'private': secret}
    if kind == 'extra': payload['private'] = secret
    response = client.post('/api/ide/capture/reconcile', content=secret if kind == 'json' else json.dumps(payload))
    assert response.status_code == 400
    assert secret not in response.text and 'ValidationError' not in response.text


def test_streamed_reconciliation_limit_precedes_parsing(client, monkeypatch):
    monkeypatch.setattr(ide, 'CAPTURE_RECONCILE_INPUT_MAX_BYTES', 10)
    assert client.post('/api/ide/capture/reconcile', content=iter([b'123456', b'123456'])).status_code == 413


def test_reconciliation_requires_the_host_credential(client, observed, monkeypatch, tmp_path):
    monkeypatch.setenv('AUTH_DISABLED', 'false')
    monkeypatch.setenv('TENSORVIZ_LOCAL_TOKEN', 'test-reconciliation-token')
    monkeypatch.setenv('TENSORVIZ_WORKSPACE_DIR', str(tmp_path))
    app = FastAPI(); app.include_router(ide.router)
    with TestClient(app) as host:
        assert host.post('/api/ide/capture/reconcile', json=observed).status_code == 401
        response = host.post('/api/ide/capture/reconcile', json=observed, headers={'X-tensorViz-Token': 'test-reconciliation-token'})
        assert response.status_code == 200 and response.json()['postconditions']['status'] == 'verified'
