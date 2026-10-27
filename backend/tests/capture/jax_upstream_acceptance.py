"""Opt-in CPU acceptance against pinned upstream repositories (no skipped cases).

See docs/jax-model-acceptance.md for environment setup. Run from the repo root.
"""
import argparse
import importlib.metadata
import importlib.util
import json
import platform
from pathlib import Path
import subprocess
import tempfile
import time

import jax
import numpy as np

from test_jax_capture import run_model
from services.ide.workspace_projection import project_workspace
from services.semantic.contracts import CaptureSnapshot, walk_argument
from services.semantic.portable import portable_capture

PINS = {
    'gemma': '0f7e7420e84c6789d96ba465608d2a008ecb443d',
    'tunix': 'a3f01618604ca1854874688b5149c18373dd10f5',
    'big_vision': '0127fb6b337ee2a27bf4e54dea79cff176527356',
}
CASES = ['gemma3', 'gemma4', 'gemma4_moe', 'vit', 'vit_scan', 'siglip2',
         'tunix_qwen3', 'tunix_qwen3_remat', 'tunix_gemma4', 'tunix_gemma4_moe', 'tunix_qwen3_cache', 'tunix_gemma4_cache']


def signatures(capture):
    by_id = {v.id: v for v in capture.values}
    return [(list(by_id[value.id].shape), by_id[value.id].dtype)
            for _, value in walk_argument(capture.output) if value.kind == 'value']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstreams', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', choices=CASES, default=CASES)
    parser.add_argument('--sizes', nargs='+', type=int, default=[4, 8])
    args = parser.parse_args()
    for name, sha in PINS.items():
        root = args.upstreams / name
        assert subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() == sha
        assert not subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True).strip()
        origin = importlib.util.find_spec(name)
        assert origin and origin.origin and Path(origin.origin).resolve().is_relative_to(root.resolve()), (name, origin)
    source = Path(__file__).with_name('fixtures').joinpath('jax_upstream.py').read_text()
    from fixtures.jax_upstream import build, inputs
    import tensorflow as tf
    report = dict(pins=PINS, platform=platform.platform(), python=platform.python_version(),
        repeats=2, versions={name: importlib.metadata.version(name)
        for name in ('jax', 'jaxlib', 'flax', 'numpy', 'kauldron', 'torch')}, cases=[])
    report['versions']['tensorflow'] = tf.__version__
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in args.cases:
        for size in args.sizes:
            start = time.monotonic()
            record = dict(case=case, size=size, status='failed')
            try:
                actual = jax.block_until_ready(build(case, size)(**inputs(case, size)))
                leaves = jax.tree.leaves(actual)
                assert leaves and all(np.isfinite(np.asarray(leaf)).all() for leaf in leaves)
                expected = [(list(leaf.shape), str(leaf.dtype)) for leaf in leaves]
                previous = None
                for _ in range(2):
                    with tempfile.TemporaryDirectory(prefix='tensorviz-jax-upstream-') as folder:
                        process, envelope, request = run_model(Path(folder).resolve(), source, init=None,
                            worker_timeout=180, limits=dict(timeout_ms=180000, max_events=20000,
                                max_tensors=100000, max_record_bytes=4194304),
                            factory=dict(file='model.py', callable='build', kwargs=dict(case=case, size=size)),
                            inputs=dict(factory=dict(file='model.py', callable='inputs', kwargs=dict(case=case, size=size))))
                        assert process.returncode == 0, (envelope.get('error'), process.stderr[-2000:])
                        capture = CaptureSnapshot.model_validate(envelope['result']['diagnostics']['semantic_capture'])
                        assert signatures(capture) == expected, (signatures(capture), expected)
                        expected_omissions = ['opaque_kernel'] if case in ('vit_scan', 'tunix_gemma4_moe', 'gemma4_moe') else []
                        assert capture.coverage.omissions == expected_omissions, capture.coverage.omissions
                        assert capture.status == ('partial' if expected_omissions else 'complete')
                        exported = portable_capture(capture).capture
                        assert signatures(exported) == expected
                        request.trace_result = envelope
                        projected = project_workspace(request)
                        assert projected['traced'] and projected['document']['semantic']['edges']
                        assert all(n['metadata']['shapeEvidence'] == 'inferred' for n in projected['document']['semantic']['nodes'])
                        signature = ([(op.target, op.arguments.model_dump(), op.output.model_dump()) for op in capture.operations],
                                     [v.model_dump() for v in capture.values], capture.output.model_dump())
                        assert previous is None or previous == signature, 'Repeated captures differ'
                        previous = signature
                        record.update(coverage=capture.status, omissions=capture.coverage.omissions,
                            operations=len(capture.operations), values=len(capture.values), outputs=expected)
                        (args.output.parent / f'{case}-{size}-capture.json').write_text(capture.model_dump_json())
                record['status'] = 'passed'
            except Exception as error:
                record['error'] = repr(error)
            record['seconds'] = round(time.monotonic() - start, 2)
            report['cases'].append(record)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(record), flush=True)
            jax.clear_caches()
    raise SystemExit(int(any(item['status'] != 'passed' for item in report['cases'])))


if __name__ == '__main__':
    main()
