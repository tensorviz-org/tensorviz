from copy import deepcopy
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import ide
from services.semantic.document import capture_document
from services.semantic.source_edits import CapturedEditResolver
from services.ide.local_browser_protocol import _source_edit_is_valid
from tests.api.test_ide_router import client
from tests.source_review.test_semantic_source_edits import apply


FIXTURE = json.loads((Path(__file__).parent.resolve().parents[2] / 'tests/fixtures/contracts/captured-source-edit.json').read_text())
MODULE_FIXTURE = json.loads((Path(__file__).parent.resolve().parents[2] / 'tests/fixtures/contracts/captured-module-activation-edit.json').read_text())
ACTIVATION_FIXTURE = json.loads((Path(__file__).parent.resolve().parents[2] / 'tests/fixtures/contracts/captured-activation-edit.json').read_text())
IMPORT_FIXTURES = [json.loads((Path(__file__).parent.resolve().parents[2] / ('tests/fixtures/contracts/' + name + '.json')).read_text())
                   for name in ['captured-unary-insert', 'captured-unary-import', 'captured-unary-delete', 'captured-reconnection-edit', 'captured-activation-import', 'captured-module-activation-import', 'captured-constructor-edit', 'captured-constructor-keyword', 'captured-custom-constructor', 'captured-activation-options', 'captured-module-activation-options']]
CONFIG_FIXTURES = [json.loads((Path(__file__).parent.resolve().parents[2] / 'tests/fixtures/contracts' / name).read_text())
                   for name in ['captured-configuration-edit.json', 'captured-configuration-module.json']]
EDIT_FIXTURES = [*CONFIG_FIXTURES, FIXTURE, ACTIVATION_FIXTURE, MODULE_FIXTURE, *IMPORT_FIXTURES]


def request(fixture=FIXTURE):
    return deepcopy(dict(capture=fixture['capture'], sources={'model.py': fixture['source']}, intent=fixture['intent']))


@pytest.mark.parametrize('fixture', EDIT_FIXTURES)
def test_shared_fixture_matches_real_producer_document_capability_and_patch(fixture):
    resolver = CapturedEditResolver(fixture['capture'], {'model.py': fixture['source']})
    assert fixture['capability'] in [cap.model_dump(mode='json') for cap in resolver.capabilities(fixture['intent']['target']['semanticId'])]
    assert resolver.plan(fixture['intent']) == fixture['plan']
    assert apply(fixture['source'], fixture['plan']) == fixture['after']
    document, spans = capture_document(fixture['capture'], name='Model', graph_id=fixture['document']['semantic']['graph_id'], model_file='model.py',
        source_digests={'model.py': fixture['capture']['source']['files'][0]['digest']},
        source_graph={'nodes': [], 'edges': []}, source_spans={}, source_texts={'model.py': fixture['source']})
    assert document.model_dump(mode='json', by_alias=True) == fixture['document']
    assert spans == fixture['spans']
    assert all(span['kind'] == 'call' for span in spans.values())


def test_capability_budget_never_discards_computational_nodes_or_navigation(monkeypatch):
    monkeypatch.setattr('services.semantic.document.MAX_EDIT_CAPABILITY_BYTES', 1)
    document, spans = capture_document(FIXTURE['capture'], name='Model', graph_id='literal-fixture', model_file='model.py',
        source_digests={'model.py': FIXTURE['capture']['source']['files'][0]['digest']},
        source_graph={'nodes': [], 'edges': []}, source_spans={}, source_texts={'model.py': FIXTURE['source']})
    assert len(document.semantic.nodes) == len(FIXTURE['document']['semantic']['nodes'])
    assert spans == FIXTURE['spans']
    operations = [n for n in document.semantic.nodes if n.metadata['captureRef']['kind'] == 'operation']
    assert all(not n.metadata['sourceEditCapabilities'] and n.metadata['sourceEditLimitation'] == 'capability_limit' for n in operations)


@pytest.mark.parametrize('kind', ['set_constructor_literal', 'set_configuration_literal'])
def test_constructor_capability_budget_retains_available_fields_and_marks_omissions(monkeypatch, kind):
    fixture = next(f for f in EDIT_FIXTURES if f['intent']['kind'] == kind)
    node = next(n for n in fixture['document']['semantic']['nodes'] if n['metadata']['captureRef'] == fixture['intent']['target'])
    first = node['metadata']['sourceEditCapabilities'][0]
    monkeypatch.setattr('services.semantic.document.MAX_EDIT_CAPABILITY_BYTES', len(json.dumps(first).encode()) + 1)
    document, spans = capture_document(fixture['capture'], name='Model', graph_id='budget-fixture', model_file='model.py',
        source_digests={'model.py': fixture['capture']['source']['files'][0]['digest']},
        source_graph={'nodes': [], 'edges': []}, source_spans={}, source_texts={'model.py': fixture['source']})
    actual = next(n for n in document.semantic.nodes if n.node_id == node['node_id'])
    assert actual.metadata['sourceEditCapabilities'] == [first]
    assert actual.metadata['sourceEditLimitation'] == 'capability_limit'
    assert len(document.semantic.nodes) == len(fixture['document']['semantic']['nodes'])
    assert spans == fixture['spans']


@pytest.mark.parametrize('kind', ['set_constructor_literal', 'set_configuration_literal'])
@pytest.mark.parametrize('path,valid', [(['args', 0], True), (['args', 255], True), (['kwargs', '宽度'], True),
    (['kwargs', 'é'], True), (['kwargs', '𐐀' * 128], True), (['args', 256], False), (['args', True], False),
    (['args', 0.5], False), (['kwargs', 'p\n'], False), (['kwargs', 'p.x'], False), (['kwargs', '😀'], False),
    (['kwargs', 'a' * 129], False), (['kwargs', ''], False), (['args', 1, 0], False)])
def test_constructor_path_wire_guard_parity(path, valid, kind):
    fixture = next(f for f in EDIT_FIXTURES if f['intent']['kind'] == kind)
    assert _source_edit_is_valid({**fixture['intent'], 'nodeId': 'call', 'sourceArgumentPath': path}) == valid


@pytest.mark.parametrize('path,valid', [(['args', 0], True), (['args', 1], True), (['args', 2], False),
    (['args', True], False), (['args', 0.5], False), (['kwargs', 'input'], False), (['args', 1, 0], False)])
def test_reconnection_path_wire_guard_parity(path, valid):
    fixture = next(f for f in EDIT_FIXTURES if f['intent']['kind'] == 'reconnect_argument')
    assert _source_edit_is_valid({**fixture['intent'], 'nodeId': 'add', 'argumentPath': path}) == valid


@pytest.mark.parametrize('change', ['port', 'unavailable', 'name'])
def test_reconnection_api_refuses_substituted_operands_values_and_source_names(client, change):
    fixture = next(f for f in EDIT_FIXTURES if f['intent']['kind'] == 'reconnect_argument')
    payload = request(fixture)
    if change == 'port':
        payload['intent']['argumentPath'] = ['args', 0]
    elif change == 'unavailable':
        payload['intent']['value'] = fixture['capture']['operations'][-1]['output']
    else:
        payload['intent']['name'] = 'PRIVATE_SOURCE_DO_NOT_ECHO'
    response = client.post('/api/ide/capture/edit', json=payload)
    assert response.status_code in (400, 422)
    assert 'PRIVATE_SOURCE_DO_NOT_ECHO' not in response.text


@pytest.mark.parametrize('fixture', EDIT_FIXTURES)
def test_api_returns_exact_static_only_patch_and_noop(client, fixture):
    response = client.post('/api/ide/capture/edit', json=request(fixture))
    assert response.status_code == 200
    assert response.headers['cache-control'] == 'no-store'
    assert response.json() == fixture['plan']
    payload = request(fixture)
    payload['intent']['value'] = payload['intent']['expected']
    result = client.post('/api/ide/capture/edit', json=payload)
    if fixture['intent']['kind'] in ('insert_unary_op', 'delete_unary_op'):
        assert result.status_code in (400, 422)  # Structural requests cannot masquerade as same-value edits.
    else:
        assert result.status_code == 200 and result.json()['edits'] == []


@pytest.mark.parametrize('broken', ['json', 'schema', 'source', 'capability', 'extra_field'])
def test_invalid_requests_never_echo_source_or_private_input(client, broken):
    secret = 'PRIVATE_SOURCE_DO_NOT_ECHO'
    payload = request()
    if broken == 'schema':
        payload['intent']['value'] = {'kind': 'int', 'value': secret}
    elif broken == 'source':
        payload['sources']['model.py'] += secret
    elif broken == 'capability':
        payload['intent']['capabilityId'] = secret
    elif broken == 'extra_field':
        payload['path'] = secret
    response = client.post('/api/ide/capture/edit', content=secret if broken == 'json' else json.dumps(payload))
    assert response.status_code in (400, 422)
    assert secret not in response.text


def test_streamed_input_limit_refuses_without_parsing(client, monkeypatch):
    monkeypatch.setattr(ide, 'CAPTURE_EDIT_INPUT_MAX_BYTES', 10)
    assert client.post('/api/ide/capture/edit', content=iter([b'123456', b'123456'])).status_code == 413


def test_only_authenticated_host_can_prepare_source(client, monkeypatch, tmp_path):
    monkeypatch.setenv('AUTH_DISABLED', 'false')
    monkeypatch.setenv('TENSORVIZ_LOCAL_TOKEN', 'test-edit-token')
    monkeypatch.setenv('TENSORVIZ_WORKSPACE_DIR', str(tmp_path))
    app = FastAPI()
    app.include_router(ide.router)
    with TestClient(app) as host:
        assert host.post('/api/ide/capture/edit', json=request()).status_code == 401
        assert host.post('/api/ide/capture/edit', json=request(), headers={'X-tensorViz-Token': 'test-edit-token'}).status_code == 200


@pytest.mark.parametrize('fixture', EDIT_FIXTURES)
def test_broker_accepts_only_strict_bounded_captured_intents(fixture):
    edit = {**fixture['intent'], 'nodeId': 'captured:add'}
    assert _source_edit_is_valid(edit)
    for patch in [dict(value={'kind': 'int', 'value': '2\n'}), dict(scope='one_call'),
                  dict(path='model.py'), dict(argumentPath=['args', 2 if fixture['intent']['kind'] in ('reconnect_argument', 'insert_unary_op') else 1 if fixture['intent']['kind'] == 'delete_unary_op' else 0]), dict(value={'kind': 'float', 'value': -0.0})]:
        assert not _source_edit_is_valid({**edit, **patch})
    for kind in ['input', 'output', 'value'] + (['invocation'] if fixture is FIXTURE else []):
        assert not _source_edit_is_valid({**edit, 'target': {**edit['target'], 'kind': kind}})
