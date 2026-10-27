"""Real JAX/Flax through saved profiles, private workers, and Canvas projection."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")

from services.ide.graph_setup_profiles import prepare_profile
from services.ide.graph_setup import discover
from services.ide.multifile import MultifileProjectionRequest
from services.ide.workspace_projection import project_workspace
from services.semantic.contracts import CaptureSnapshot, ResolvedExecutionPlan
from services.semantic.execution_plan import applicability_digest, profile_execution
from services.validation.contracts import RunSpec
from services.validation.host_snapshot import snapshot


def tensor(shape=(2, 3)):
    return dict(kind='tensor', shape=list(shape), dtype='float32', requires_grad=False,
                generator=dict(kind='full', value=1))


def request(source, name):
    digest = hashlib.sha256(source.encode()).hexdigest()
    return MultifileProjectionRequest(protocol_version=1,
        root=dict(file='model.py', cls=name, source_digest=digest),
        sources=[dict(file='model.py', source=source, digest=digest)])


def run_model(tmp_path, source, name='net', *, args=None, kwargs=None, alias=False, worker_timeout=45, **profile_changes):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'model.py').write_text(source)
    source_request = request(source, name)
    profile = dict(kind='model', framework='jax', model=f'model:{name}', mode='eval', device='cpu',
        grad_enabled=False, seed=0, init=dict(args=[], kwargs={}),
        inputs=dict(args=[tensor()] if args is None else args, kwargs=kwargs or {}))
    profile.update(profile_changes)
    prepared = prepare_profile(source_request, '', profile)
    assert prepared['ok'], prepared
    (workspace / 'tensorviz.toml').write_text(prepared['config'])
    if alias:
        linked = tmp_path / 'linked-workspace'
        linked.symlink_to(workspace, target_is_directory=True)
        workspace = linked
    params = dict(root=str(workspace), executable=sys.executable, environmentRevision=0,
        session='jax-test', workspaceId='jax-test', runId='jax-run', generation=1,
        modelFile='model.py', className=name, timeoutMs=worker_timeout * 1000, profile=prepared['profile'])
    resolution = snapshot(params)['resolution']
    assert resolution['status'] == 'resolved', resolution
    spec = RunSpec.model_validate(resolution['spec'])
    plan = ResolvedExecutionPlan.model_validate(snapshot(dict(params, expectedIdentity=spec.identity.model_dump(),
        execution=profile_execution(spec.profile)))['plan'])
    plan_file = tmp_path / 'plan.json'
    plan_file.write_text(json.dumps(dict(version=1, plan=plan.model_dump(mode='json'), applicabilityDigest=applicability_digest(plan))))
    plan_file.chmod(0o600)
    backend = Path(__file__).resolve().parents[2]
    process = subprocess.run([sys.executable, '-B', str(backend / 'services/ingest/bootstrap.py'),
        '--resolved-execution', '--workspace', str(workspace), '--file', str(workspace / 'model.py'),
        '--cls', name, '--json'], cwd=backend,
        env={**os.environ, 'TENSORVIZ_EXECUTION_PLAN': str(plan_file), 'PYTHONPATH': str(backend)},
        capture_output=True, text=True, timeout=worker_timeout)
    assert process.stdout, process.stderr
    envelope = json.loads(process.stdout)
    return process, envelope, source_request


@pytest.mark.parametrize('source,name', [
    ('import jax.numpy as jnp\ndef net(x): return jnp.tanh(x + x)\n', 'net'),
    ('import jax\nimport jax.numpy as jnp\n@jax.jit\ndef net(x): return jax.nn.relu(x @ jnp.ones((3,4)))\n', 'net'),
    ('from flax import linen as nn\nclass Net(nn.Module):\n    @nn.compact\n    def __call__(self,x): return nn.relu(nn.Dense(4)(x))\n', 'Net'),
])
def test_saved_jax_and_flax_capture_reaches_canvas(tmp_path, source, name):
    process, envelope, source_request = run_model(tmp_path, source, name)
    assert process.returncode == 0, (envelope.get('error'), process.stderr)
    diagnostics = envelope['result']['diagnostics']
    assert diagnostics['execution']['planCurrent'] is True
    capture = CaptureSnapshot.model_validate(diagnostics['semantic_capture'])
    assert capture.backend == 'jaxpr' and capture.torchVersion is None and capture.jaxVersion
    assert capture.coverage.scope == 'traced_program'
    assert capture.operations and capture.status == 'complete'
    assert capture.sourceBindings, 'Workspace source frames must remain navigable.'
    from services.semantic.portable import portable_capture
    portable = portable_capture(capture).capture
    assert portable.backend == 'jaxpr' and portable.coverage.scope == 'traced_program'
    assert all(value.device == 'abstract' and value.storageId is None and value.logicalBytes is None for value in capture.values)
    source_request.trace_result = envelope
    projected = project_workspace(source_request)
    assert projected['traced'] is True
    document = projected['document']
    assert document['semantic']['metadata']['capture']['backend'] == 'jaxpr'
    assert any(node['op'].startswith('jax.') for node in document['semantic']['nodes'])
    assert document['semantic']['edges']
    assert all(node['metadata']['shapeEvidence'] == 'inferred' for node in document['semantic']['nodes'])
    assert all(not node['metadata'].get('sourceEditCapabilities') for node in document['semantic']['nodes'])


def test_repeated_operands_and_nested_output_keep_value_identity(tmp_path):
    process, envelope, _ = run_model(tmp_path,
        'import jax.numpy as jnp\ndef net(x):\n    y=jnp.add(x,x)\n    return {"pair": (y,x), "nothing": None}\n')
    assert process.returncode == 0, envelope.get('error')
    capture = envelope['result']['diagnostics']['semantic_capture']
    add = next(op for op in capture['operations'] if op['target'] == 'jax.add')
    assert add['arguments']['args'][0] == add['arguments']['args'][1]
    assert capture['output']['kind'] == 'dict'
    pair = next(entry['value'] for entry in capture['output']['entries'] if entry['key']['value'] == 'pair')
    assert pair['kind'] == 'tuple' and pair['items'][1] == capture['inputs']['args'][0]


def test_control_flow_is_not_claimed_as_an_executed_branch(tmp_path):
    process, envelope, _ = run_model(tmp_path,
        'import jax\ndef net(x): return jax.lax.cond(x.sum()>0, lambda a:a+1, lambda a:a-1, x)\n')
    assert process.returncode == 0, envelope.get('error')
    capture = envelope['result']['diagnostics']['semantic_capture']
    assert capture['status'] == 'partial' and 'opaque_kernel' in capture['coverage']['omissions']
    assert any(op['target'] == 'jax.cond' for op in capture['operations'])


def test_invalid_shapes_fail_without_a_successful_graph(tmp_path):
    process, envelope, _ = run_model(tmp_path,
        'import jax.numpy as jnp\ndef net(x): return x @ jnp.ones((5,4))\n')
    assert process.returncode != 0
    assert envelope['error'] and envelope['result'] is None


def test_discovery_and_setup_do_not_execute_model_source(tmp_path):
    from services.ingest.repo_scan import scan_source, scan_repo
    source = '''import jax.numpy as jnp
from flax import linen as nn
raise RuntimeError("discovery must never import this")
def net(x): return jnp.sin(x)
def helper(x): return x
class Net(nn.Module):
    width: int
    @nn.compact
    def __call__(self,x): return nn.Dense(self.width)(x)
'''
    assert {item['name'] for item in scan_source(source)} == {'net', 'Net'}
    (tmp_path / 'model.py').write_text(source)
    assert {item['name'] for item in scan_repo(str(tmp_path)).class_occurrences if item['is_module']} == {'net', 'Net'}
    setup = discover(request(source, 'Net'))
    assert setup['init'][0]['name'] == 'width' and setup['init'][0]['required']
    assert setup['inputs'][0]['name'] == 'x'
    assert setup['resolution']['draft']['profile']['framework'] == 'jax'
    assert setup['resolution']['status'] == 'partial'


def test_static_keyword_uses_saved_value_without_tracer_boolean_error(tmp_path):
    process, envelope, _ = run_model(tmp_path,
        'import jax.numpy as jnp\ndef net(x, *, enabled=True): return jnp.sin(x) if enabled else jnp.cos(x)\n',
        kwargs={'enabled': False})
    assert process.returncode == 0, envelope.get('error')
    capture = envelope['result']['diagnostics']['semantic_capture']
    assert [op['target'] for op in capture['operations']] == ['jax.cos']
    assert capture['inputs']['kwargs']['enabled'] == dict(kind='bool', value=False)


@pytest.mark.parametrize('nnx', [False, True])
def test_flax_factory_preserves_configured_parameters_and_input_factory(tmp_path, nnx):
    source = '''from flax import linen as nn
import jax
import jax.numpy as jnp
class Net(nn.Module):
    @nn.compact
    def __call__(self,x): return nn.Dense(4)(x)
def build():
    model = Net()
    return model.bind(model.init(jax.random.key(3), jnp.ones((2,3))))
def inputs(): return {"x": jnp.ones((2,3))}
'''
    if nnx:
        source = '''from flax import nnx
import jax.numpy as jnp
class Net(nnx.Module):
    def __init__(self, rngs): self.dense = nnx.Linear(3,4,rngs=rngs)
    def __call__(self,x): return self.dense(x)
def build(): return Net(nnx.Rngs(3))
def inputs(): return {"x": jnp.ones((2,3))}
'''
    process, envelope, _ = run_model(tmp_path, source, 'Net', init=None,
        factory=dict(file='model.py', callable='build', kwargs={}),
        inputs=dict(factory=dict(file='model.py', callable='inputs', kwargs={})))
    assert process.returncode == 0, (envelope.get('error'), process.stderr)
    capture = envelope['result']['diagnostics']['semantic_capture']
    assert any(op['target'] == 'jax.dot_general' for op in capture['operations'])
    assert capture['values'][-1]['shape'] == [2, 4]


@pytest.mark.parametrize('kind', ['module', 'method', 'partial'])
def test_nnx_jit_state_updates_stay_inside_the_trace(tmp_path, kind):
    source = '''import functools
from flax import nnx
import jax
import jax.numpy as jnp
class Net(nnx.Module):
    def __init__(self): self.dense = nnx.Linear(3, 4, rngs=nnx.Rngs(3))
    @nnx.jit
    def __call__(self, x): return self.dense(x)
@jax.jit
def net(x): return x
def build(kind):
    model = Net()
    if kind == 'module': return model
    if kind == 'method': return model.__call__
    return functools.partial(nnx.jit(lambda model, x: model(x), donate_argnums=(0,)), model)
'''
    process, envelope, _ = run_model(tmp_path, source, init=None,
        factory=dict(file='model.py', callable='build', kwargs=dict(kind=kind)))
    assert process.returncode == 0, (envelope.get('error'), process.stderr)
    capture = envelope['result']['diagnostics']['semantic_capture']
    assert capture['status'] == 'complete'
    assert any(op['target'] == 'jax.dot_general' for op in capture['operations'])
    output = capture['output']
    assert next(value for value in capture['values'] if value['id'] == output['id'])['shape'] == [2, 4]


def test_compiled_parameter_operands_above_python_call_limit_are_retained(tmp_path):
    source = '''import functools
import jax
import jax.numpy as jnp
@jax.jit
def net(x): return x
def apply(weights, x):
    for weight in weights: x = x @ weight
    return x
def build():
    weights = tuple(jnp.full((3, 3), float(index + 1)) for index in range(300))
    return functools.partial(jax.jit(apply), weights)
'''
    process, envelope, _ = run_model(tmp_path, source, init=None,
        factory=dict(file='model.py', callable='build', kwargs={}))
    assert process.returncode == 0, envelope.get('error')
    capture = envelope['result']['diagnostics']['semantic_capture']
    assert len([op for op in capture['operations'] if op['target'] == 'jax.dot_general']) == 300
    operands = capture['invocations'][1]['arguments']['args'][0]
    assert operands['kind'] == 'tuple' and len(operands['items']) == 301
    assert len({item['id'] for item in operands['items']}) == 301


def test_jax_capture_cannot_claim_runtime_measurements(tmp_path):
    process, envelope, _ = run_model(tmp_path, 'import jax.numpy as jnp\ndef net(x): return jnp.sin(x)\n')
    assert process.returncode == 0
    capture = envelope['result']['diagnostics']['semantic_capture']
    capture['coverage']['scope'] = 'executed_path'
    with pytest.raises(ValueError, match='traced-program'):
        CaptureSnapshot.model_validate(capture)


def test_local_shadow_does_not_discover_an_unrelated_helper():
    from services.ingest.repo_scan import scan_source
    assert scan_source('import jax.numpy as jnp\ndef helper(jnp): return jnp.sin(0)\n') == []


def test_binding_total_limit_is_checked_before_array_allocation(monkeypatch):
    from services.jax_support import setup
    def unexpected_allocation(value):
        pytest.fail('Oversized bindings must be rejected before allocating arrays.')
    monkeypatch.setattr(setup, 'materialize', unexpected_allocation)
    with pytest.raises(ValueError, match='total elements'):
        setup.binding(dict(args=[tensor((2048, 2048)), tensor((1,))], kwargs={}))


def test_device_selection_never_falls_back_to_another_backend(monkeypatch):
    import jax
    from services.jax_support.setup import resolve_device
    cpu, gpu0, gpu1 = object(), object(), object()
    monkeypatch.setattr(jax, 'devices', lambda backend: {'cpu': [cpu], 'cuda': [gpu0, gpu1]}[backend])
    assert resolve_device('cpu') is cpu
    assert resolve_device('cuda:1') is gpu1
    with pytest.raises(ValueError, match='unavailable'):
        resolve_device('cuda:2')
    with pytest.raises(ValueError, match='cpu or cuda:N'):
        resolve_device('mps')
    def unavailable(backend):
        raise RuntimeError('CUDA backend failed to initialize')
    monkeypatch.setattr(jax, 'devices', unavailable)
    with pytest.raises(ValueError, match='selected Python and JAX accelerator installation'):
        resolve_device('cuda:0')


def test_unavailable_device_fails_before_importing_model(tmp_path):
    process, envelope, _ = run_model(tmp_path,
        'import jax.numpy as jnp\nraise AssertionError("model imported")\ndef net(x): return jnp.sin(x)\n',
        device='cuda:999')
    assert process.returncode != 0 and envelope['result'] is None
    assert 'JAX device cuda:999 is unavailable' in json.dumps(envelope['error'])
    assert 'model imported' not in json.dumps(envelope['error'])


def test_cpu_profile_scopes_import_factories_and_tracing(tmp_path):
    process, envelope, _ = run_model(tmp_path, '''import jax
import jax.numpy as jnp
assert jax.config.jax_default_device.platform == 'cpu'
def inputs():
    assert jax.config.jax_default_device.platform == 'cpu'
    return {"x": jnp.ones((2,3))}
def build():
    assert jax.config.jax_default_device.platform == 'cpu'
    return net
def net(x):
    assert jax.config.jax_default_device.platform == 'cpu'
    return jnp.sin(x)
''', init=None, factory=dict(file='model.py', callable='build', kwargs={}),
        inputs=dict(factory=dict(file='model.py', callable='inputs', kwargs={})))
    assert process.returncode == 0, envelope.get('error')
    assert envelope['result']['diagnostics']['execution']['device'] == 'cpu'


@pytest.mark.parametrize('expression', [
    'jax.grad(lambda a: jnp.sin(a).sum())(x)',
    'jax.vmap(lambda a: jnp.cos(a))(x)',
])
def test_explicit_jax_transforms_trace_the_selected_computation(tmp_path, expression):
    process, envelope, _ = run_model(tmp_path,
        f'import jax\nimport jax.numpy as jnp\ndef net(x): return {expression}\n')
    assert process.returncode == 0, envelope.get('error')
    capture = envelope['result']['diagnostics']['semantic_capture']
    assert capture['coverage']['scope'] == 'traced_program'
    assert capture['values'][-1]['shape'] == [2, 3]


def test_saved_operation_limit_stops_capture(tmp_path):
    process, envelope, _ = run_model(tmp_path,
        'import jax.numpy as jnp\ndef net(x): return jnp.cos(jnp.sin(x))\n',
        limits=dict(timeout_ms=30000, max_events=1, max_tensors=100, max_record_bytes=1048576))
    assert process.returncode != 0
    assert envelope['error']['diagnostic']['code'] == 'capture_limited'
    assert envelope['error']['diagnostic']['remedy'] == 'adjust_limits'
    assert envelope['error'] and envelope['result'] is None


def test_capture_byte_limit_has_actionable_diagnostic_and_separate_opt_in(tmp_path):
    source = 'import jax.numpy as jnp\ndef net(x): return jnp.sin(x)\n'
    low, high = tmp_path / 'low', tmp_path / 'high'
    low.mkdir()
    high.mkdir()
    limits = dict(timeout_ms=30000, max_events=100, max_tensors=100, max_record_bytes=1024)
    process, envelope, _ = run_model(low, source, limits=limits)
    assert process.returncode != 0 and envelope['result'] is None
    diagnostic = envelope['error']['diagnostic']
    assert diagnostic['code'] == 'capture_limited' and diagnostic['field'] == ['limits', 'max_capture_bytes']
    process, envelope, _ = run_model(high, source, limits={**limits, 'max_capture_bytes': 1048576})
    assert process.returncode == 0, envelope.get('error')


def test_static_discovery_never_loads_optional_frameworks(monkeypatch):
    import builtins
    from services.ingest.repo_scan import scan_source
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in ('jax', 'flax'):
            pytest.fail('Static discovery must not import optional frameworks.')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    assert {item['name'] for item in scan_source('import jax.numpy as jnp\ndef net(x): return jnp.sin(x)\n')} == {'net'}


def test_workspace_path_alias_reaches_the_same_saved_model(tmp_path):
    value = tensor()
    value['generator'] = dict(kind='normal', mean=0, std=1, seed=0)
    process, envelope, _ = run_model(tmp_path, 'import jax.numpy as jnp\ndef net(x): return jnp.sin(x)\n',
        args=[], kwargs={'x': value}, alias=True)
    assert process.returncode == 0, envelope.get('error')
    assert envelope['result']['diagnostics']['execution']['planCurrent']


def test_typed_rng_key_from_input_factory(tmp_path):
    process, envelope, _ = run_model(tmp_path,
        'import jax\ndef net(key): return jax.random.normal(key, (2,3))\n'
        'def inputs(): return {"key": jax.random.key(0)}\n',
        inputs=dict(factory=dict(file='model.py', callable='inputs', kwargs={})))
    assert process.returncode == 0, envelope.get('error')
    capture = envelope['result']['diagnostics']['semantic_capture']
    assert capture['values'][0]['dtype'].startswith('key<')
    assert capture['values'][-1]['shape'] == [2, 3]


def test_registered_pytree_boundaries_keep_all_tensor_leaves(tmp_path):
    # Gemma returns a Flax struct; decoding leaves into the original class can
    # also violate a custom container's invariants. Encode its registered paths.
    source = '''import jax
import jax.numpy as jnp
from flax import struct
@struct.dataclass
class Output:
    logits: object
    cache: object
    hidden_states: object
@jax.tree_util.register_pytree_with_keys_class
class Input:
    def __init__(self, x):
        assert hasattr(x, "shape"), "Only arrays are valid here"
        self.x = x
    def tree_flatten_with_keys(self): return ((jax.tree_util.GetAttrKey("x"), self.x),), None
    @classmethod
    def tree_unflatten(cls, aux, children): return cls(*children)
def inputs(): return {"batch": Input(jnp.ones((2,3)))}
def net(batch): return Output(jnp.sin(batch.x), {"key": batch.x}, None)
'''
    process, envelope, _ = run_model(tmp_path, source,
        inputs=dict(factory=dict(file='model.py', callable='inputs', kwargs={})))
    assert process.returncode == 0, (envelope.get('error'), process.stderr)
    capture = CaptureSnapshot.model_validate(envelope['result']['diagnostics']['semantic_capture'])
    from services.semantic.contracts import walk_argument
    output_refs = [v.id for _, v in walk_argument(capture.output) if v.kind == 'value']
    assert len(output_refs) == 2, 'Structured model outputs must not disappear as opaque leaves.'
    assert {tuple(v.shape) for v in capture.values if v.id in output_refs} == {(2, 3)}
    assert capture.status == 'complete'


def test_pytree_namedtuple_and_frozen_mapping_preserve_output_fields(tmp_path):
    process, envelope, _ = run_model(tmp_path, '''import jax.numpy as jnp
from collections import namedtuple
from flax.core import freeze
Output = namedtuple("Output", "logits state")
def net(x): return Output(jnp.cos(x), freeze({"value": x}))
''')
    assert process.returncode == 0, envelope.get('error')
    output = envelope['result']['diagnostics']['semantic_capture']['output']
    assert [entry['key']['value'] for entry in output['entries']] == ['logits', 'state']
    assert output['entries'][1]['value']['entries'][0]['value']['kind'] == 'value'


def test_input_keyword_order_and_kind_field_cannot_corrupt_references(tmp_path):
    process, envelope, _ = run_model(tmp_path, '''import jax.numpy as jnp
def inputs(): return {"z": {"kind": "value", "id": jnp.ones((2,3))}, "a": jnp.ones((4,5))}
def net(z, a): return a, jnp.add(z["id"], 1)
''', inputs=dict(factory=dict(file='model.py', callable='inputs', kwargs={})))
    assert process.returncode == 0, envelope.get('error')
    capture = envelope['result']['diagnostics']['semantic_capture']
    by_id = {value['id']: value for value in capture['values']}
    assert [by_id[value['id']]['shape'] for value in capture['output']['items']] == [[4,5], [2,3]]
    assert capture['inputs']['kwargs']['z']['kind'] == 'dict'


def test_checkpointed_convolution_and_gather_keep_primitive_details(tmp_path):
    process, envelope, _ = run_model(tmp_path, '''import jax
import jax.numpy as jnp
def net(x):
    def block(image):
        y = jax.lax.conv_general_dilated(image, jnp.ones((3,3,3,4)), (1,1), "SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"), precision=jax.lax.Precision.HIGHEST)
        return jnp.take(y, jnp.array([0,2]), axis=-1)
    return jax.checkpoint(block)(x)
''', args=[tensor((1,8,8,3))])
    assert process.returncode == 0, envelope.get('error')
    capture = envelope['result']['diagnostics']['semantic_capture']
    assert capture['status'] == 'complete', capture['coverage']
    targets = {op['target'] for op in capture['operations']}
    assert {'jax.conv_general_dilated', 'jax.gather'} <= targets
    assert 'jax.remat2' not in targets, 'Checkpointing must not hide the actual model layers.'
    conv = next(op for op in capture['operations'] if op['target'] == 'jax.conv_general_dilated')
    assert conv['arguments']['kwargs']['dimension_numbers']['kind'] == 'tuple'
    assert conv['arguments']['kwargs']['precision']['items'][0]['value'] == 'Precision.HIGHEST'


def test_moe_ragged_dot_dimension_metadata_is_retained(tmp_path):
    process, envelope, _ = run_model(tmp_path, '''import jax
import jax.numpy as jnp
def net(x):
    # Row groups select different expert matrices, as in a sparse MoE.
    dims = jax.lax.RaggedDotDimensionNumbers((((1,), (1,)), ((), ())), (0,), (0,))
    return jax.lax.ragged_dot_general(x, jnp.ones((2,3,4)), jnp.array([1,1]), dims)
''')
    assert process.returncode == 0, envelope.get('error')
    capture = envelope['result']['diagnostics']['semantic_capture']
    assert capture['status'] == 'complete', capture['coverage']
    op = next(op for op in capture['operations'] if op['target'] == 'jax.ragged_dot_general')
    metadata = op['arguments']['kwargs']['ragged_dot_dimension_numbers']
    assert metadata['kind'] == 'dict'
    assert {entry['key']['value'] for entry in metadata['entries']} == {
        'dot_dimension_numbers', 'lhs_ragged_dimensions', 'rhs_group_dimensions'}
