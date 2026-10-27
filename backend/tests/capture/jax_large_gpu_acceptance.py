"""Opt-in full-size JAX GPU acceptance. Requires a dedicated >= 140 GB CUDA GPU.

The real forward and each capture use separate processes to release model memory.
Run from the repo root with PYTHONPATH=backend. No cases are silently skipped.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import jax

from test_jax_capture import run_model
from jax_upstream_acceptance import signatures
from services.ide.workspace_projection import project_workspace
from services.ide.local_browser_wire import encode_message, decode_message
from services.semantic.contracts import CaptureSnapshot
from services.semantic.portable import portable_capture

PIN = 'a3f01618604ca1854874688b5149c18373dd10f5'
CASES = ('qwen3_32b', 'gemma4_31b', 'gemma4_26b_a4b')


def actual(case, size, mode):
    from flax import nnx
    from fixtures.jax_large_gpu import build, inputs, model_for
    gpu = jax.devices('cuda')[0]
    with jax.default_device(gpu):
        model = model_for(case)
        params = jax.tree.leaves(nnx.state(model, nnx.Param))
        assert params and all(x.dtype == jax.numpy.bfloat16 for x in params)
        assert all(x.devices() == {gpu} for x in params), 'Parameters must reside on the GPU'
        parameter_count = sum(x.size for x in params)
        assert parameter_count > 25_000_000_000, parameter_count
        arguments = inputs(case, size, mode)
        function = build(case, size, mode)
        first = jax.block_until_ready(function(**arguments))
        second = jax.block_until_ready(function(**arguments))
        leaves, repeated = jax.tree.leaves(first), jax.tree.leaves(second)
        assert leaves and len(leaves) == len(repeated)
        for x, y in zip(leaves, repeated):
            assert x.devices() == {gpu}, 'Forward outputs must reside on the GPU'
            assert bool(jax.numpy.isfinite(x).all()), 'Forward produced nonfinite outputs'
            assert bool(jax.numpy.array_equal(x, y)), 'Repeated GPU forwards differ'
        shapes = [(list(x.shape), str(x.dtype)) for x in leaves]
        config = repr(model.config)
        return dict(outputs=shapes, parameterCount=parameter_count,
            parameterBytes=sum(x.nbytes for x in params), gpu=str(gpu),
            deviceKind=gpu.device_kind, memory=gpu.memory_stats(), config=config,
            configSha256=hashlib.sha256(config.encode()).hexdigest(),
            weights='random initialization, seed 0', dtype='bfloat16', forwardRepeats=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--mode', choices=('eager', 'jit', 'decode'), default='eager')
    parser.add_argument('--forward-only', action='store_true')
    args = parser.parse_args()
    root = args.upstream.resolve()
    assert subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() == PIN
    assert not subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True).strip()
    origin = importlib.util.find_spec('tunix')
    assert origin and origin.origin and Path(origin.origin).resolve().is_relative_to(root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.forward_only:
        args.output.write_text(json.dumps(actual(args.case, args.size, args.mode), indent=2) + '\n')
        return
    start = time.monotonic()
    report = dict(case=args.case, size=args.size, mode=args.mode, upstream=PIN, status='failed',
        sourceCommit=os.environ.get('TENSORVIZ_SOURCE_COMMIT') or subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        platform=platform.platform(), python=platform.python_version(),
        versions={name: importlib.metadata.version(name) for name in ('jax', 'jaxlib', 'flax', 'torch')})
    backend = Path(__file__).resolve().parents[2]
    report['sourceSha256'] = {str(path.relative_to(backend)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(backend.rglob('*.py')) if '__pycache__' not in path.parts}
    try:
        forward = args.output.with_suffix('.forward.json')
        command = [sys.executable, __file__, '--upstream', str(root), '--output', str(forward),
            '--case', args.case, '--size', str(args.size), '--mode', args.mode, '--forward-only']
        with args.output.with_suffix('.forward.log').open('w') as log:
            subprocess.run(command, check=True, stdout=log, stderr=log, timeout=900)
        report['forward'] = json.loads(forward.read_text())
        expected = [(shape, dtype) for shape, dtype in report['forward']['outputs']]
        source = Path(__file__).with_name('fixtures').joinpath('jax_large_gpu.py').read_text()
        previous = None
        for repeat in range(2):
            with tempfile.TemporaryDirectory(prefix='tensorviz-jax-large-') as folder:
                options = dict(case=args.case, size=args.size, mode=args.mode)
                process, envelope, request = run_model(Path(folder).resolve(), source, init=None,
                    device='cuda:0', worker_timeout=900,
                    limits=dict(timeout_ms=900000, max_events=100000, max_tensors=100000,
                        max_record_bytes=4194304, max_capture_bytes=16777216),
                    factory=dict(file='model.py', callable='build', kwargs=options),
                    inputs=dict(factory=dict(file='model.py', callable='inputs', kwargs=options)))
                args.output.with_suffix(f'.capture-{repeat}.log').write_text(process.stderr)
                assert process.returncode == 0, envelope.get('error')
                diagnostics = envelope['result']['diagnostics']
                assert diagnostics['execution']['device'] == 'cuda:0'
                capture = CaptureSnapshot.model_validate(diagnostics['semantic_capture'])
                assert signatures(capture) == expected, (signatures(capture), expected)
                omissions = ['opaque_kernel'] if args.case == 'gemma4_26b_a4b' else []
                assert capture.coverage.omissions == omissions, capture.coverage.omissions
                assert capture.status == ('partial' if omissions else 'complete')
                assert signatures(portable_capture(capture).capture) == expected
                args.output.with_suffix('.capture.json').write_text(capture.model_dump_json())
                report.update(operations=len(capture.operations), values=len(capture.values),
                    captureBytes=len(capture.model_dump_json().encode()))
                request.trace_result = envelope
                document = project_workspace(request)
                wire = encode_message(document)
                assert decode_message(wire) == document
                report['wireBytes'] = len(wire.encode())
                assert document['traced'] and document['document']['semantic']['edges']
                assert all(n['metadata']['shapeEvidence'] == 'inferred' for n in document['document']['semantic']['nodes'])
                stable = dict(operations=[x.model_dump() for x in capture.operations],
                    values=[x.model_dump() for x in capture.values], output=capture.output.model_dump())
                digest = hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()
                assert previous is None or previous == digest, 'Repeated captures differ'
                previous = digest
                report.update(operations=len(capture.operations), values=len(capture.values),
                    coverage=capture.status, omissions=omissions, captureRepeats=repeat + 1)
                args.output.with_suffix('.capture.json').write_text(capture.model_dump_json())
                args.output.with_suffix('.projection.json').write_text(json.dumps(document))
        report['status'] = 'passed'
    except Exception as error:
        report['error'] = repr(error)
    report['seconds'] = round(time.monotonic() - start, 2)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k, v in report.items() if k not in ('forward', 'sourceSha256')}), flush=True)
    raise SystemExit(int(report['status'] != 'passed'))


if __name__ == '__main__':
    main()
