"""The authenticated broker binds candidate traffic to the actual view connection."""
import pytest
from services.ide.local_browser_protocol import local_browser_fixture_corpus
from tests.api.test_local_browser_broker import (
    client, env, mint, bootstrap, cookie_key_of, connect_view, connect_control_host,
    entered, envelope, target,
)


@pytest.mark.parametrize('action', ['list', 'reconcile'])
def test_candidate_and_source_ownership_cannot_be_forged_by_another_tab(client, action):
    session = mint(client)
    host = connect_control_host(client, session)
    a, ca, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session)
    b, cb, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session, n=60)
    with entered(a, b, host):
        aid, bid = ca['session']['viewId'], cb['session']['viewId']
        assert host.receive_json()['viewId'] == aid
        assert host.receive_json()['viewId'] == bid
        source = envelope('sourceEditRequest', session['workspaceSessionId'], 1100,
            viewId=aid, target=target(), requestId='review', edit={'kind': 'remove_layer', 'nodeId': 'fc'})
        b.send_json(source)
        assert b.receive_json()['code'] == 'INVALID_MESSAGE'
        a.send_json(source)
        assert host.receive_json() == source
        command = envelope('candidateRequest', session['workspaceSessionId'], 1101,
            viewId=aid, target=target(), requestId='read', proposalId='review', action={'kind': action})
        b.send_json(command)
        assert b.receive_json()['code'] == 'INVALID_MESSAGE'
        a.send_json(command)
        assert host.receive_json() == command
        payload = {'candidates': [], 'page': None}
        if action == 'reconcile':
            payload = next(item['result'] for item in local_browser_fixture_corpus()[1] if item['messageId'] == 'applied-source-checked')
        result = envelope('candidateResult', session['workspaceSessionId'], 1102,
            viewId=aid, target=target(), requestId='read', proposalId='review', result=payload, detail='Private.')
        b.send_json(result)
        assert b.receive_json()['code'] == 'INVALID_MESSAGE'
        host.send_json(result)
        assert a.receive_json() == result
        # A response for b proves a's result was never broadcast into b's queue.
        second = {**result, 'viewId': bid, 'messageId': 'second', 'requestId': 'second'}
        host.send_json(second)
        assert b.receive_json() == second


def test_review_progress_is_private_and_does_not_consume_the_apply_receipt(client):
    session = mint(client)
    host = connect_control_host(client, session)
    view, connected, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session)
    with entered(view, host):
        owner = connected['session']['viewId']
        assert host.receive_json()['viewId'] == owner
        request = envelope('sourceEditRequest', session['workspaceSessionId'], 1200,
            viewId=owner, target=target(), requestId='review', edit={'kind': 'remove_layer', 'nodeId': 'fc'})
        view.send_json(request)
        assert host.receive_json() == request
        progress = envelope('sourceEditReview', session['workspaceSessionId'], 1201,
            target=target(), requestId='review', candidate=None)
        host.send_json({**progress, 'target': {**target(), 'cls': 'Wrong'}})
        view.send_json(progress)
        assert view.receive_json()['code'] == 'INVALID_MESSAGE'
        host.send_json(progress)
        assert view.receive_json() == progress
        result = envelope('sourceEditResult', session['workspaceSessionId'], 1202,
            target=target(), requestId='review', status='applied', detail='Applied explicitly.')
        host.send_json(result)
        assert view.receive_json() == result
        # Historical inspection is directed, not tied to the consumed apply request.
        read = envelope('candidateRequest', session['workspaceSessionId'], 1203,
            viewId=owner, target=target(), requestId='history', proposalId='review', action={'kind': 'list'})
        view.send_json(read)
        assert host.receive_json() == read


def test_disconnect_notifies_host_to_revoke_historical_candidate_ownership(client):
    session = mint(client)
    host = connect_control_host(client, session)
    view, connected, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session)
    with entered(host):
        owner = connected['session']['viewId']
        assert host.receive_json()['viewId'] == owner
        view.__exit__(None, None, None)
        disconnected = host.receive_json()
        assert disconnected['type'] == 'viewState'
        assert disconnected['viewId'] == owner and disconnected['connected'] is False
        assert host.receive_json()['state'] == 'detached'
        # A late response for a gone view is dropped without echo or broadcast.
        host.send_json(envelope('candidateResult', session['workspaceSessionId'], 1300,
            viewId=owner, target=target(), requestId='late', proposalId='review', result=None, detail='Gone.'))
        new, newer, _ = connect_view(client, cookie_key_of(bootstrap(client, session)), session, n=70)
        with entered(new):
            assert newer['session']['viewId'] != owner
            assert host.receive_json()['viewId'] == newer['session']['viewId']
