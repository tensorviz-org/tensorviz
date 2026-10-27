"""Private candidate requests reuse normal capture and keep separate provenance."""
import json
import io
import os
from pathlib import Path
import subprocess
import sys
from contextlib import contextmanager

import pytest

from services.semantic.contracts import CandidateRequest, CandidateExecution, CaptureSnapshot
from services.semantic.source_edits import CapturedEditResolver
from tests.source_review.test_semantic_candidate import prepare, literal_proposal
from tests.source_review.test_semantic_source_edits import SOURCE, apply
from tests.source_review.test_semantic_constructor_edits import SOURCE as LINEAR, intent as constructor_intent


def payload(plan, capture, edit, after):
    return CandidateRequest(version=1, id='candidate:test', plan=plan, capture=capture, intent=edit,
        reviewedFile='model.py', reviewedSource=after).model_dump(mode='json')


def replay(root, request, tmp_path, extra=(), tamper=None):
    control = tmp_path / 'control'
    control.mkdir(mode=0o700)
    file = control / 'request.json'
    file.write_text(json.dumps(request))
    file.chmod(0o600)
    if tamper:
        tamper(file)
    backend = Path(__file__).parent.resolve().parents[1]
    process = subprocess.run([sys.executable, '-B', str(backend / 'services/ingest/bootstrap.py'),
        '--candidate-execution', '--workspace', str(root), '--file', str(root / 'model.py'),
        '--cls', 'Model', '--json', *extra], cwd=backend,
        env={**os.environ, 'TENSORVIZ_CANDIDATE_REQUEST': str(file),
            'TENSORVIZ_EXECUTION_PLAN': '/nonexistent/inherited_plan_must_not_run'},
        capture_output=True, text=True, timeout=30)
    envelope = json.loads(process.stdout)
    assert not list(control.glob('tensorviz-candidate-*')), 'candidate files survived worker completion'
    return process, envelope


def test_private_request_runs_exact_candidate_and_publishes_distinct_provenance(tmp_path):
    root = (tmp_path / 'original').resolve()
    plan, base = prepare(root)
    edit, after = literal_proposal(base)
    process, envelope = replay(root, payload(plan, base, edit, after), tmp_path)
    assert process.returncode == 0, process.stderr
    candidate = CandidateExecution.model_validate(envelope['worker']['candidateExecution'])
    capture = CaptureSnapshot.model_validate(envelope['result']['diagnostics']['semantic_capture'])
    assert envelope['worker']['candidateBaseCurrent'] is True
    assert candidate.baseCaptureId == base.id and candidate.baseAttemptId == base.attemptId
    assert capture.source == candidate.plan.source and capture.attemptId == candidate.id
    assert capture.operations[0].arguments.args[1].value == '2'
    postconditions = envelope['worker']['candidatePostconditions']
    assert postconditions['status'] == 'verified'
    assert postconditions['executionId'] == candidate.id and postconditions['captureId'] == capture.id
    assert postconditions['observedOperations'] == 1
    assert capture.source.workspaceId != base.source.workspaceId
    assert envelope['result']['diagnostics']['execution']['planCurrent'] is True
    assert (root / 'model.py').read_text() == SOURCE


@pytest.mark.parametrize('value,exit_code', [('8', 4), ('-1', 3)])
def test_candidate_keeps_forward_and_construction_failures_with_provenance(tmp_path, value, exit_code):
    root = (tmp_path / 'original').resolve()
    plan, base = prepare(root, LINEAR)
    resolver = CapturedEditResolver(base, {'model.py': LINEAR})
    cap = resolver.constructor_literal_capabilities(base.invocations[1].id)[1]
    edit = constructor_intent(cap, dict(kind='int', value=value))
    request = payload(plan, base, edit, apply(LINEAR, resolver.plan(edit)))
    process, envelope = replay(root, request, tmp_path)
    assert process.returncode == exit_code, process.stderr
    CandidateExecution.model_validate(envelope['worker']['candidateExecution'])
    assert envelope['worker']['candidateBaseCurrent'] is True
    assert envelope['error'] is not None
    assert envelope['worker']['candidatePostconditions']['reason'] == ('capture_incomplete' if exit_code == 4 else 'capture_unavailable')
    if exit_code == 4:
        capture = CaptureSnapshot.model_validate(envelope['result']['diagnostics']['semantic_capture'])
        assert capture.status == 'partial'
        assert any(d.code == 'invalid_shape' for d in capture.diagnostics)
    else:
        assert envelope['result'] is None
    assert (root / 'model.py').read_text() == LINEAR


@pytest.mark.parametrize('which', ['original', 'candidate'])
def test_runtime_source_changes_cannot_certify_freshness(tmp_path, which):
    root = (tmp_path / 'original').resolve()
    destination = repr(str(root / 'model.py')) if which == 'original' else '__file__'
    source = SOURCE.replace('import torch', 'import torch\nfrom pathlib import Path')
    source = source.replace('        a = h + 1',
        '        if Path(__file__).parent.name == "workspace":\n'
        f'            Path({destination}).write_text("# changed during forward")\n'
        '        a = h + 1')
    plan, base = prepare(root, source)
    edit, after = literal_proposal(base, source)
    process, envelope = replay(root, payload(plan, base, edit, after), tmp_path)
    assert process.returncode == 0, process.stderr
    assert envelope['worker']['candidateBaseCurrent'] is (which != 'original')
    assert envelope['result']['diagnostics']['execution']['planCurrent'] is (which != 'candidate')
    if which == 'candidate': assert (root / 'model.py').read_text() == source


@pytest.mark.parametrize('extra', [
    ['--execution', '{}'], ['--resolved-execution'], ['--capture-preparation'],
    ['--input', 'private_expression()'], ['--tensor-inputs', '[]'], ['--overrides', '{}'],
    ['--diagnostic-job', '{}'], ['--roots', '/private/root'], ['--no-frames'],
    ['--kind', 'train_step'], ['--tier', 'fake'], ['--cls', 'Other'],
])
def test_competing_execution_flags_refuse_without_import_or_private_text(tmp_path, extra):
    root = (tmp_path / 'original').resolve()
    plan, base = prepare(root)
    edit, after = literal_proposal(base)
    process, envelope = replay(root, payload(plan, base, edit, after), tmp_path, extra)
    assert process.returncode == 2
    assert envelope['result'] is None and envelope['error']['type'] == 'CandidateRefused'
    assert 'private_expression' not in process.stdout and '/private/root' not in process.stdout
    assert 'candidateExecution' not in envelope['worker']


@pytest.mark.parametrize('change', ['review', 'extra', 'capture', 'attempt', 'source', 'config', 'missing', 'large'])
def test_changed_or_unbounded_private_requests_never_run(tmp_path, change):
    root = (tmp_path / 'original').resolve()
    plan, base = prepare(root)
    edit, after = literal_proposal(base)
    request = payload(plan, base, edit, after)
    tamper = None
    if change == 'review': request['reviewedSource'] += '\nprivate_source_text()\n'
    if change == 'extra': request['arbitraryExpression'] = 'private_source_text()'
    if change == 'capture': request['capture']['attemptId'] = 'another'
    if change == 'attempt': request['id'] = plan.id
    if change == 'source': (root / 'model.py').write_text(SOURCE + '# changed')
    if change == 'config': (root / 'tensorviz.toml').write_text('# changed')
    if change == 'missing': tamper = lambda file: file.unlink()
    if change == 'large':
        def tamper(file):
            with file.open('r+b') as stream:
                stream.truncate(24 * 1024 * 1024 + 1)
    process, envelope = replay(root, request, tmp_path, tamper=tamper)
    assert process.returncode == 2
    assert envelope['result'] is None and envelope['error']['type'] == 'CandidateRefused'
    assert 'private_source_text' not in process.stdout


@pytest.mark.skipif(os.name == 'nt', reason='POSIX permissions; Windows host uses private ACLs')
@pytest.mark.parametrize('change', ['file_permissions', 'parent_permissions', 'symlink'])
def test_candidate_transport_requires_private_host_file_and_directory(tmp_path, change):
    root = (tmp_path / 'original').resolve()
    plan, base = prepare(root)
    edit, after = literal_proposal(base)
    def tamper(file):
        if change == 'file_permissions': file.chmod(0o644)
        if change == 'parent_permissions': file.parent.chmod(0o755)
        if change == 'symlink':
            target = file.with_suffix('.original')
            file.rename(target)
            file.symlink_to(target)
    process, envelope = replay(root, payload(plan, base, edit, after), tmp_path, tamper=tamper)
    assert process.returncode == 2
    assert envelope['error']['type'] == 'CandidateRefused'


def test_cleanup_failure_cannot_publish_success_then_a_second_envelope(tmp_path, monkeypatch):
    from services.ingest import candidate_worker
    from services.ingest.worker import _Envelope, build_parser
    root = (tmp_path / 'original').resolve()
    plan, base = prepare(root)
    edit, after = literal_proposal(base)
    control = tmp_path / 'control'
    control.mkdir(mode=0o700)
    file = control / 'request.json'
    file.write_text(json.dumps(payload(plan, base, edit, after)))
    file.chmod(0o600)
    monkeypatch.setenv('TENSORVIZ_CANDIDATE_REQUEST', str(file))
    original = candidate_worker.stage_candidate
    @contextmanager
    def cleanup_failure(*args, **kwargs):
        with original(*args, **kwargs) as staged:
            yield staged
        raise OSError('private_cleanup_error')
    def completed(_argv, *, envelope):
        envelope.result = {'completed': True}
        return envelope.emit(0)
    monkeypatch.setattr(candidate_worker, 'stage_candidate', cleanup_failure)
    monkeypatch.setattr(candidate_worker, 'main', completed)
    envelope = _Envelope()
    envelope._out = io.StringIO()
    args = build_parser().parse_args(['--candidate-execution', '--workspace', str(root),
        '--file', str(root / 'model.py'), '--cls', 'Model'])
    assert candidate_worker.run_candidate(args, envelope) == 2
    result = json.loads(envelope._out.getvalue())
    assert result['result'] is None and result['error']['type'] == 'CandidateRefused'
    assert 'private_cleanup_error' not in envelope._out.getvalue()
    assert 'TENSORVIZ_CANDIDATE_REQUEST' not in os.environ
    assert 'TENSORVIZ_EXECUTION_PLAN' not in os.environ
