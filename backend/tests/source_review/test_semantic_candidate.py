"""Candidate execution uses exact reviewed source without touching the live model."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from services.ingest.trace import trace_model
from services.semantic import candidate as staging
from services.semantic.candidate import CandidateRefused, CandidateCancelled, stage_candidate, verify_replay_base
from services.semantic.contracts import CaptureSnapshot, ResolvedExecutionPlan, CandidateExecution
from services.semantic.execution_plan import applicability_digest, profile_execution
from services.semantic.recorder import CaptureContext
from services.semantic.source_bindings import CapturedSourceEvidence
from services.semantic.source_edits import CapturedEditResolver
from services.validation.contracts import RunSpec
from services.validation.host_snapshot import snapshot
from tests.source_review.test_semantic_source_edits import SOURCE, intent, apply
from tests.source_review.test_semantic_constructor_edits import SOURCE as LINEAR, intent as constructor_intent
from tests.capture.test_execution_plan import CONFIG


def prepare(root, source=SOURCE, execution=None, profile=False):
    root.mkdir(exist_ok=True)
    root = root.resolve()
    (root / 'model.py').write_bytes(source.encode())
    request = dict(root=str(root), executable=sys.executable, environmentRevision=0,
        session='test-session', workspaceId='live-workspace', runId='base-run', generation=3,
        modelFile='model.py', className='Model', timeoutMs=30000,
        execution=execution or {'inputs': {'args': [{'shape': [2, 4], 'dtype': 'float32', 'fill': 'ones'}]}})
    if profile:
        (root / 'tensorviz.toml').write_text(CONFIG)
        spec = RunSpec.model_validate(snapshot({key: value for key, value in request.items() if key != 'execution'}
            | {'profile': 'small'})['resolution']['spec'])
        request.update(profile='small', expectedIdentity=spec.identity.model_dump(), execution=profile_execution(spec.profile))
    plan = ResolvedExecutionPlan.model_validate(snapshot(request)['plan'])
    evidence = CapturedSourceEvidence(plan.source, root)
    namespace = {'__name__': 'candidate_test_model', '__file__': str(root / 'model.py')}
    exec(compile(source, str(root / 'model.py'), 'exec', dont_inherit=True), namespace)
    with evidence.construction.observe():
        model = namespace['Model']()
    context = CaptureContext(plan.source, applicability_digest(plan), plan.id, plan.captureLimits, source_evidence=evidence)
    result = trace_model(model, (torch.ones(2, 4),), tier='real', semantic_context=context)
    capture = CaptureSnapshot.model_validate(result.semantic_capture)
    return plan, capture


def literal_proposal(capture, source=SOURCE):
    resolver = CapturedEditResolver(capture, {'model.py': source})
    operation = next(op for op in capture.operations if op.target == 'aten.add.Tensor'
        and op.arguments.args[1].kind == 'int')
    edit = intent(resolver.literal_capability(operation.id))
    return edit, apply(source, resolver.plan(edit))


def replay(root, plan, control):
    file = control / 'execution.json'
    file.write_text(json.dumps(dict(version=1, plan=plan.model_dump(mode='json'),
        applicabilityDigest=applicability_digest(plan))))
    file.chmod(0o600)
    backend = Path(__file__).parent.resolve().parents[1]
    process = subprocess.run([sys.executable, '-B', str(backend / 'services/ingest/bootstrap.py'),
        '--resolved-execution', '--workspace', str(root), '--file', str(root / plan.source.modelFile),
        '--cls', plan.source.className, '--json'], cwd=backend,
        env={**os.environ, 'TENSORVIZ_EXECUTION_PLAN': str(file)}, capture_output=True, text=True, timeout=30)
    return process, json.loads(process.stdout)


@pytest.mark.parametrize('profile', [False, True])
def test_staging_keeps_setup_exact_uses_distinct_identities_and_never_runs_source(tmp_path, profile):
    root = tmp_path / 'original'
    plan, capture = prepare(root, profile=profile)
    edit, after = literal_proposal(capture)
    with stage_candidate(str(root), plan, capture, edit, 'model.py', after) as staged:
        candidate = staged.execution
        assert candidate.baseCaptureId == capture.id and candidate.baseSourceId == capture.source.id
        assert candidate.basePlanDigest == applicability_digest(plan)
        assert candidate.plan.source.workspaceId != plan.source.workspaceId
        assert candidate.plan.source.modelId != plan.source.modelId
        assert candidate.plan.source.id != plan.source.id and candidate.plan.id != plan.id
        assert candidate.plan.source.sourceMode == 'saved'
        expected = plan.model_dump(exclude={'id', 'source'})
        assert candidate.plan.model_dump(exclude={'id', 'source'}) == expected
        assert (staged.workspace / 'model.py').read_bytes() == after.encode()
        assert (root / 'model.py').read_bytes() == SOURCE.encode()
        assert (root / 'model.py').stat().st_ino != (staged.workspace / 'model.py').stat().st_ino
        assert not (staged.workspace / '__pycache__').exists()
        if os.name != 'nt':
            assert staged.workspace.parent.stat().st_mode & 0o777 == 0o700
            assert (staged.workspace / 'model.py').stat().st_mode & 0o777 == 0o600
        # The working copy is writable, but the original and retained records stay exact.
        (staged.workspace / 'model.py').write_text('different copy')
        assert capture.source == plan.source
    assert not staged.workspace.parent.exists()
    assert (root / 'model.py').read_bytes() == SOURCE.encode()


def test_real_candidate_worker_changes_only_the_requested_literal(tmp_path):
    root = tmp_path / 'original'
    plan, capture = prepare(root)
    edit, after = literal_proposal(capture)
    with stage_candidate(str(root), plan, capture, edit, 'model.py', after) as staged:
        process, envelope = replay(staged.workspace, staged.execution.plan, tmp_path)
        assert process.returncode == 0, process.stderr
        diagnostics = envelope['result']['diagnostics']
        changed = CaptureSnapshot.model_validate(diagnostics['semantic_capture'])
        assert diagnostics['execution']['planCurrent'] is True
        assert changed.source == staged.execution.plan.source
        assert changed.planDigest == applicability_digest(staged.execution.plan)
        assert changed.operations[0].arguments.args[1].value == '2'
        assert changed.operations[1].arguments.args[0] == changed.operations[1].arguments.args[1]
        assert [op.target for op in changed.operations] == [op.target for op in capture.operations]
        assert changed.status == 'complete'
    assert (root / 'model.py').read_bytes() == SOURCE.encode()


def test_static_valid_constructor_candidate_retains_actual_shape_failure(tmp_path):
    root = tmp_path / 'original'
    plan, capture = prepare(root, LINEAR)
    resolver = CapturedEditResolver(capture, {'model.py': LINEAR})
    cap = resolver.constructor_literal_capabilities(capture.invocations[1].id)[1]
    edit = constructor_intent(cap)
    patch = resolver.plan(edit)
    assert patch['validation'] == 'static_only'
    with stage_candidate(str(root), plan, capture, edit, 'model.py', apply(LINEAR, patch)) as staged:
        process, envelope = replay(staged.workspace, staged.execution.plan, tmp_path)
        assert process.returncode == 4, process.stderr
        changed = CaptureSnapshot.model_validate(envelope['result']['diagnostics']['semantic_capture'])
        assert changed.status == 'partial'
        assert any(d.phase == 'forward' and d.code == 'invalid_shape' for d in changed.diagnostics)
        assert any(call.status == 'failed' for call in changed.invocations)
        assert envelope['result']['diagnostics']['execution']['planCurrent'] is True
    assert (root / 'model.py').read_bytes() == LINEAR.encode()
    assert capture.status == 'complete'


@pytest.mark.parametrize('change', ['source', 'new_source', 'config', 'environment', 'input', 'capture'])
def test_stale_or_substituted_base_refuses_before_staging(tmp_path, change):
    root = tmp_path / 'original'
    plan, capture = prepare(root)
    edit, after = literal_proposal(capture)
    copies = tmp_path / 'copies'
    copies.mkdir()
    if change == 'source': (root / 'model.py').write_text(SOURCE + '# changed')
    if change == 'new_source': (root / 'helper.py').write_text('raise RuntimeError("never import")')
    if change == 'config': (root / 'tensorviz.toml').write_text('# new')
    if change == 'environment': plan.environment.probe_digest = '0' * 64
    if change == 'input': plan.execution['seed'] = 42
    if change == 'capture': capture.attemptId = 'unrelated'
    with pytest.raises(ValueError):
        with stage_candidate(str(root), plan, capture, edit, 'model.py', after, temporary_parent=str(copies)):
            pytest.fail('stale candidate became executable')
    assert not list(copies.iterdir())


@pytest.mark.parametrize('change', ['file', 'source', 'capability', 'scope', 'unchanged'])
def test_reviewed_artifact_must_be_exact_closed_recipe(tmp_path, change):
    root = tmp_path / 'original'
    plan, capture = prepare(root)
    edit, after = literal_proposal(capture)
    file = 'model.py'
    if change == 'file': file = 'other.py'
    if change == 'source': after += '\nprint("extra edit")\n'
    if change == 'capability': edit['capabilityId'] = 'other'
    if change == 'scope': edit['scope'] = 'one_call'
    if change == 'unchanged': edit['value'] = edit['expected']
    with pytest.raises(ValueError):
        with stage_candidate(str(root), plan, capture, edit, file, after):
            pytest.fail('unreviewed change became executable')
    assert (root / 'model.py').read_bytes() == SOURCE.encode()


def test_cancellation_during_copy_removes_private_files(tmp_path):
    root = tmp_path / 'original'
    plan, capture = prepare(root)
    edit, after = literal_proposal(capture)
    copies = tmp_path / 'copies'
    copies.mkdir()
    def cancelled():
        return any(copies.glob('*/workspace/model.py'))
    with pytest.raises(CandidateCancelled):
        with stage_candidate(str(root), plan, capture, edit, 'model.py', after,
                temporary_parent=str(copies), cancelled=cancelled):
            pytest.fail('cancelled run was yielded')
    assert not list(copies.iterdir())
    assert (root / 'model.py').read_bytes() == SOURCE.encode()


def test_base_change_during_copy_refuses_and_cleans_up(tmp_path, monkeypatch):
    root = tmp_path / 'original'
    plan, capture = prepare(root)
    edit, after = literal_proposal(capture)
    copies = tmp_path / 'copies'
    copies.mkdir()
    calls = 0
    def check(base, workspace):
        nonlocal calls
        calls += 1
        if calls == 2: (root / 'model.py').write_text(SOURCE + '# changed while copying')
        verify_replay_base(base, workspace)
    monkeypatch.setattr(staging, 'verify_replay_base', check)
    with pytest.raises(CandidateRefused, match='base_changed'):
        with stage_candidate(str(root), plan, capture, edit, 'model.py', after, temporary_parent=str(copies)):
            pytest.fail('raced candidate was yielded')
    assert not list(copies.iterdir())


def test_copy_budget_is_independent_of_runtime_and_capture_limits(tmp_path, monkeypatch):
    root = tmp_path / 'original'
    plan, capture = prepare(root)
    edit, after = literal_proposal(capture)
    copies = tmp_path / 'copies'
    copies.mkdir()
    monkeypatch.setattr(staging, 'MAX_COPY_BYTES', 16)
    with pytest.raises(CandidateRefused, match='copy_limit'):
        with stage_candidate(str(root), plan, capture, edit, 'model.py', after, temporary_parent=str(copies)):
            pytest.fail('unbounded copy')
    assert not list(copies.iterdir())


@pytest.mark.parametrize('change', ['attempt', 'base_attempt', 'source', 'workspace', 'digest', 'file', 'noop'])
def test_candidate_provenance_cannot_impersonate_base(tmp_path, change):
    root = tmp_path / 'original'
    plan, capture = prepare(root)
    edit, after = literal_proposal(capture)
    with stage_candidate(str(root), plan, capture, edit, 'model.py', after) as staged:
        data = staged.execution.model_dump(mode='json')
        if change == 'attempt': data['id'] = 'unrelated'
        if change == 'base_attempt': data['plan']['id'] = data['id'] = data['baseAttemptId']
        if change == 'source': data['plan']['source']['id'] = data['baseSourceId']
        if change == 'workspace': data['plan']['source']['workspaceId'] = data['baseWorkspaceId']
        if change == 'digest': data['afterDigest'] = '0' * 64
        if change == 'file': data['file'] = 'other.py'
        if change == 'noop': data['afterDigest'] = data['beforeDigest']
        with pytest.raises(ValueError): CandidateExecution.model_validate(data)


def test_candidate_worker_refuses_tampered_copy_before_import(tmp_path):
    root = tmp_path / 'original'
    plan, capture = prepare(root)
    edit, after = literal_proposal(capture)
    with stage_candidate(str(root), plan, capture, edit, 'model.py', after) as staged:
        (staged.workspace / 'model.py').write_text('raise RuntimeError("do not import")')
        process, envelope = replay(staged.workspace, staged.execution.plan, tmp_path)
        assert process.returncode == 2
        assert 'do not import' not in json.dumps(envelope)
    assert (root / 'model.py').read_bytes() == SOURCE.encode()


def test_factory_fixture_checkpoint_and_cwd_replay_with_actual_input_and_state_identity(tmp_path):
    root = (tmp_path / 'original').resolve()
    source = '''import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4, bias=False)
    def forward(self, x):
        return self.proj(x) + 1
'''
    root.mkdir()
    (root / 'run').mkdir()
    (root / 'factory.py').write_text('from model import Model\ndef build(): return Model()\n')
    (root / 'not-tracked.txt').write_text('must not be copied implicitly')
    torch.save({'args': [torch.ones(2, 4)], 'kwargs': {}}, root / 'inputs.pt')
    torch.save({'proj.weight': torch.eye(4)}, root / 'state.pt')
    recipe = dict(inputs={'fixture': 'inputs.pt'}, setup={'file': 'factory.py', 'callable': 'build'},
        checkpoint={'file': 'state.pt'}, cwd='run', seed=7, mode='train', gradEnabled=True, deterministic=True)
    plan, _ = prepare(root, source, recipe)
    before_process, before_envelope = replay(root, plan, tmp_path)
    assert before_process.returncode == 0, before_process.stderr
    base = CaptureSnapshot.model_validate(before_envelope['result']['diagnostics']['semantic_capture'])
    edit, after = literal_proposal(base, source)
    with stage_candidate(str(root), plan, base, edit, 'model.py', after) as staged:
        assert (staged.workspace / 'run').is_dir()
        assert not (staged.workspace / 'not-tracked.txt').exists()
        for name in ['inputs.pt', 'state.pt', 'factory.py']:
            assert (staged.workspace / name).read_bytes() == (root / name).read_bytes()
            assert (staged.workspace / name).stat().st_ino != (root / name).stat().st_ino
        process, envelope = replay(staged.workspace, staged.execution.plan, tmp_path)
        assert process.returncode == 0, process.stderr
        changed = CaptureSnapshot.model_validate(envelope['result']['diagnostics']['semantic_capture'])
        assert changed.inputDigest == base.inputDigest and base.inputDigest is not None
        assert changed.stateDigest == base.stateDigest and base.stateDigest is not None
        assert changed.operations[-1].arguments.args[1].value == '2'
        assert staged.execution.plan.execution == plan.execution
    assert (root / 'model.py').read_bytes() == source.encode()


def test_exception_in_worker_owner_still_removes_candidate(tmp_path):
    root = tmp_path / 'original'
    plan, capture = prepare(root)
    edit, after = literal_proposal(capture)
    with pytest.raises(RuntimeError, match='worker failed'):
        with stage_candidate(str(root), plan, capture, edit, 'model.py', after) as staged:
            raise RuntimeError('worker failed')
    assert not staged.workspace.parent.exists()


def test_automatic_preparation_cannot_silently_become_an_exact_replay(tmp_path):
    root = tmp_path / 'original'
    plan, capture = prepare(root)
    edit, after = literal_proposal(capture)
    with pytest.raises(CandidateRefused, match='requires_resolved_plan'):
        with stage_candidate(str(root), None, capture, edit, 'model.py', after):
            pytest.fail('invented a plan from capture observations')


@pytest.mark.parametrize('kind', ['file', 'directory'])
def test_even_internal_symlink_dependencies_refuse_ambiguous_copy_semantics(tmp_path, kind):
    root = tmp_path / 'original'
    plan, capture = prepare(root)
    edit, after = literal_proposal(capture)
    # Same bytes and source inventory: replacing a path by an internal symlink
    # still must not silently flatten its __file__/relative resource semantics.
    if kind == 'file':
        (root / 'model.data').write_bytes(SOURCE.encode())
        (root / 'model.py').unlink()
        (root / 'model.py').symlink_to(root / 'model.data')
    else:
        (root / 'actual').mkdir()
        (root / 'linked').symlink_to(root / 'actual', target_is_directory=True)
        plan.execution['cwd'] = 'linked'
        capture.planDigest = applicability_digest(plan)
    with pytest.raises(CandidateRefused, match='symlink_dependency'):
        with stage_candidate(str(root), plan, capture, edit, 'model.py', after):
            pytest.fail('source alias was silently flattened')


def test_staging_does_not_execute_import_or_forward_side_effects(tmp_path):
    root = tmp_path / 'original'
    source = 'from pathlib import Path\nPath(__file__).with_suffix(".executed").touch()\n' + SOURCE
    plan, capture = prepare(root, source)
    assert (root / 'model.executed').exists()
    (root / 'model.executed').unlink()
    edit, after = literal_proposal(capture, source)
    with stage_candidate(str(root), plan, capture, edit, 'model.py', after) as staged:
        assert not (staged.workspace / 'model.executed').exists()
        assert not (root / 'model.executed').exists()


def test_utf16_and_crlf_reviewed_bytes_are_not_normalized_during_staging(tmp_path):
    root = tmp_path / 'original'
    source = SOURCE.replace('a = h + 1', 'label = "😀"; a = h + 1').replace('\n', '\r\n')
    plan, capture = prepare(root, source)
    edit, after = literal_proposal(capture, source)
    with stage_candidate(str(root), plan, capture, edit, 'model.py', after) as staged:
        assert (staged.workspace / 'model.py').read_bytes() == source.replace('h + 1', 'h + 2').encode()
    assert (root / 'model.py').read_bytes() == source.encode()


def test_candidate_keeps_the_exact_required_import_with_unary_insertion(tmp_path):
    from tests.source_review.test_semantic_unary_edits import add, insertion
    root = tmp_path / 'original'
    source = SOURCE.replace('import torch', 'from torch.nn import Module').replace('torch.nn.Module', 'Module')
    plan, capture = prepare(root, source)
    resolver = CapturedEditResolver(capture, {'model.py': source})
    cap = resolver.insert_unary_capabilities(add(capture).id)[1]
    assert cap.importPlan is not None
    edit = insertion(cap)
    after = apply(source, resolver.plan(edit))
    with stage_candidate(str(root), plan, capture, edit, 'model.py', after) as staged:
        process, envelope = replay(staged.workspace, staged.execution.plan, tmp_path)
        assert process.returncode == 0, process.stderr
        changed = CaptureSnapshot.model_validate(envelope['result']['diagnostics']['semantic_capture'])
        assert [op.target for op in changed.operations] == [
            'aten.add.Tensor', 'aten.gelu.default', 'aten.add.Tensor', 'aten.mul.Tensor']
        unary, binary = changed.operations[1:3]
        assert binary.arguments.args[1] == unary.output
        assert binary.arguments.args[0] == unary.arguments.args[0]


@pytest.mark.parametrize('profile', [False, True])
def test_retained_plan_host_preflight_preserves_profile_and_checks_dirty_buffers(tmp_path, profile):
    root = tmp_path / 'original'
    plan, _ = prepare(root, profile=profile)
    request = dict(root=str(root), replayPlan=plan.model_dump(mode='json'), documents={})
    result = snapshot(request)
    assert result['applicabilityDigest'] == applicability_digest(plan)
    assert result['plan']['origin'] == plan.origin
    assert result['plan']['profileName'] == plan.profileName
    request['documents'] = {'model.py': {'version': 2, 'dirty': True}}
    with pytest.raises(CandidateRefused, match='base_changed'): snapshot(request)


def test_retained_plan_preflight_cannot_accept_a_competing_new_recipe(tmp_path):
    root = tmp_path / 'original'
    plan, _ = prepare(root)
    with pytest.raises(ValueError, match='substitute'):
        snapshot(dict(root=str(root), replayPlan=plan.model_dump(mode='json'), execution={'inputs': {'args': []}}))
