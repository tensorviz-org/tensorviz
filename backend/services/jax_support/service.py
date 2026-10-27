"""JAX entrypoint behind the existing saved-source execution plan and worker."""
import os
import sys
import time

from ..ingest.execution_spec import validate_execution_spec, workspace_directory
from ..ingest.spec import IngestDiagnostics, IngestResult
from ..validation.setup import setup_stage
from .capture import JaxprCapture
from .setup import prepare


def ingest_jax(ref, execution, workspace, plan, job=None):
    if plan is None or not workspace:
        raise ValueError('JAX capture requires a saved-source execution plan and explicit inputs.')
    if job is not None:
        raise ValueError('JAX graph tracing does not support runtime profiling or training diagnostics.')
    execution = validate_execution_spec(execution)
    started, previous_path, previous_cwd = time.perf_counter(), list(sys.path), os.getcwd()
    try:
        os.chdir(workspace_directory(workspace, execution['cwd']))
        model, args, kwargs, state = prepare(ref.file, ref.cls, workspace, execution)
        with setup_stage('forward', ['inputs'], 'inspect_failure'):
            capture = JaxprCapture(plan, workspace, state).capture(model, args, kwargs)
    finally:
        os.chdir(previous_cwd)
        sys.path[:] = previous_path
    diagnostics = IngestDiagnostics(tier='jaxpr', semantic_capture=capture, input_provenance='explicit',
        provenance=f'{ref.file}::{ref.cls}', seed=execution['seed'],
        execution=dict(framework='jax', seed=execution['seed'], device=execution['device'], mode='eval'),
        warnings=['JAX traced program: shapes are inferred; runtime timing and control-flow execution are not measured.'],
        timings_ms=dict(trace_ms=round((time.perf_counter() - started) * 1000, 1), input_candidates_tried=1))
    return IngestResult(ok=True, name=ref.name or ref.cls, graph=dict(nodes=[], edges=[]), diagnostics=diagnostics)
