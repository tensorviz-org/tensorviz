"""Portable export uses only retained JSON and never returns private error inputs."""
from copy import deepcopy
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import ide
from services.semantic.portable import portable_capture
from tests.capture.test_semantic_contracts import snapshot
from tests.api.test_ide_router import client


def test_portable_api_preserves_computation_without_live_source_names(client, snapshot):
    original = deepcopy(snapshot)
    response = client.post('/api/ide/capture/portable', json=snapshot)
    assert response.status_code == 200
    assert response.json() == portable_capture(snapshot).model_dump(mode='json')
    assert response.headers['cache-control'] == 'no-store'
    assert snapshot == original


@pytest.mark.parametrize('broken', ['literal', 'invalid_json', 'invalid_capture'])
def test_errors_never_echo_submitted_literals_or_validation_inputs(client, snapshot, broken):
    secret = 'PRIVATE_TOKEN_MUST_NOT_APPEAR'
    if broken == 'literal':
        snapshot['operations'][0]['arguments']['kwargs']['token'] = dict(kind='str', value=secret)
        body = json.dumps(snapshot)
    elif broken == 'invalid_json':
        body = secret
    else:
        snapshot['status'] = secret
        body = json.dumps(snapshot)
    response = client.post('/api/ide/capture/portable', content=body)
    assert response.status_code == 400
    assert secret not in response.text
    assert 'Lossless export refused' in response.text


def test_input_and_output_limits_refuse_instead_of_truncating(client, snapshot, monkeypatch):
    monkeypatch.setattr(ide, 'PORTABLE_CAPTURE_INPUT_MAX_BYTES', 10)
    assert client.post('/api/ide/capture/portable', content=iter([b'123456', b'123456'])).status_code == 413
    monkeypatch.setattr(ide, 'PORTABLE_CAPTURE_INPUT_MAX_BYTES', 16 * 1024 * 1024)
    monkeypatch.setattr(ide, 'PORTABLE_CAPTURE_MAX_BYTES', 10)
    response = client.post('/api/ide/capture/portable', json=snapshot)
    assert response.status_code == 413
    assert 'Nothing was truncated' in response.text


def test_only_authenticated_host_can_prepare_capture(snapshot, monkeypatch, tmp_path):
    monkeypatch.setenv('AUTH_DISABLED', 'false')
    monkeypatch.setenv('TENSORVIZ_LOCAL_TOKEN', 'test-launch-token')
    monkeypatch.setenv('TENSORVIZ_WORKSPACE_DIR', str(tmp_path))
    app = FastAPI()
    app.include_router(ide.router)
    with TestClient(app) as client:
        assert client.post('/api/ide/capture/portable', json=snapshot).status_code == 401
        assert client.post('/api/ide/capture/portable', json=snapshot,
                           headers={'X-tensorViz-Token': 'test-launch-token'}).status_code == 200
