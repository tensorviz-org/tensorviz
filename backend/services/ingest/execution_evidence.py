"""Bounded, value-free facts about a single forward, independent of live projections."""
from __future__ import annotations

from typing import Any

import torch
from ..semantic.identity import weak_get, weak_set

MAX_OPERATIONS = 8192
MAX_MODULE_CALLS = 4096
MAX_TENSORS_PER_OPERATION = 64


class ExecutionEvidence:
    def __init__(self, model: torch.nn.Module, args: tuple, kwargs: dict):
        self.operations: list[dict] = []
        self.module_calls: list[dict] = []
        self.failure: dict | None = None
        self.failure_error: BaseException | None = None
        self.failed_counts: dict[str, int] = {}
        self.module_errors: list[tuple[dict, BaseException]] = []
        self.operation_count = 0
        self.completion_count = 0
        self.module_digest_bytes = 0
        self.module_call_count = 0
        self.producers: dict[int, tuple[Any, dict]] = {}
        self.stack: list[dict | None] = []
        self.counts: dict[str, int] = {}
        self.completed_counts: dict[str, int] = {}
        self.reached: set[str] = set()
        self.truncated: set[str] = set()
        self.modules = [{'modulePath': path, 'operation': type(module).__name__}
                        for path, module in model.named_modules()]
        if len(self.modules) > MAX_OPERATIONS:
            self.modules = self.modules[:MAX_OPERATIONS]
            self.truncated.add('module_inventory')
        self._seed(args, 'args')
        self._seed(kwargs, 'kwargs')

    def _seed(self, value: Any, path: str) -> None:
        if isinstance(value, torch.Tensor):
            if weak_get(self.producers, value) is None:
                weak_set(self.producers, value, {'inputPath': path})
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                self._seed(item, f'{path}[{index}]')
        elif isinstance(value, dict):
            for key, item in value.items():
                self._seed(item, f'{path}[{key!r}]')

    def tensors(self, value: Any) -> list[dict]:
        from .trace import _tensors
        tensors = _tensors(value)
        if len(tensors) > MAX_TENSORS_PER_OPERATION:
            self.truncated.add('operation_tensors')
        result = []
        for tensor in tensors[:MAX_TENSORS_PER_OPERATION]:
            origin = weak_get(self.producers, tensor) or {}
            result.append({'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                           'device': str(tensor.device), **origin})
        return result

    def begin(self, operation: str, owner: str, invocation: int, args: tuple,
              kwargs: dict, frame: tuple | None) -> dict:
        index = self.operation_count
        self.operation_count += 1
        record = {'operationId': f'op-{index}', 'index': index, 'operation': operation,
                  'modulePath': owner, 'invocation': invocation,
                  'source': {'file': frame[0], 'line': frame[1], 'function': frame[2]} if frame else None,
                  'inputs': self.tensors((args, kwargs)), 'outputs': [], 'status': 'running'}
        if len(self.operations) < MAX_OPERATIONS:
            self.operations.append(record)
        else:
            self.truncated.add('operations')
        return record

    def finish(self, record: dict, output: Any) -> None:
        from .trace import _tensors
        record['status'] = 'completed'
        record['outputs'] = [{key: value for key, value in tensor.items() if key in ('shape', 'dtype', 'device')}
                             for tensor in self.tensors(output)]
        incoming, outgoing = record['inputs'], record['outputs']
        record['shapeChanged'] = (any(a['shape'] != b['shape'] for a in incoming for b in outgoing)
                                  if incoming and outgoing else None)
        for index, tensor in enumerate(_tensors(output)):
            weak_set(self.producers, tensor, {'producerOperationId': record['operationId'], 'outputIndex': index})

    def fail(self, record: dict, error: BaseException) -> None:
        record['status'] = 'failed'
        record['errorType'] = type(error).__name__[:128]
        # Retain the failure even if the prefix exhausted its operation budget.
        self.failure = record
        self.failure_error = error

    def enter(self, name: str, args: tuple, kwargs: dict) -> None:
        self.reached.add(name)
        invocation = self.counts.get(name, 0) + 1
        self.counts[name] = invocation
        index = self.module_call_count
        self.module_call_count += 1
        record = None
        if len(self.module_calls) < MAX_MODULE_CALLS:
            record = {'index': index, 'modulePath': name, 'invocation': invocation,
                      'inputs': self.tensors((args, kwargs)), 'outputs': [], 'status': 'entered'}
            self.module_calls.append(record)
        else:
            self.truncated.add('module_calls')
        self.stack.append(record)

    def leave(self, name: str, output: Any, error: BaseException | None = None) -> None:
        record = self.stack.pop()
        if error is not None:
            self.failed_counts[name] = self.failed_counts.get(name, 0) + 1
            if record is not None:
                record.update(status='failed', errorType=type(error).__name__[:128])
                self.module_errors.append((record, error))
            return
        self.completed_counts[name] = self.completed_counts.get(name, 0) + 1
        if record is not None:
            from .execution_values import capture_output, MAX_MODULE_DIGEST_BYTES
            observed = capture_output(output, max_bytes=max(0, MAX_MODULE_DIGEST_BYTES - self.module_digest_bytes), include_values=False)
            self.module_digest_bytes += observed['bytesCaptured']
            record['values'] = observed
            record['completedIndex'] = self.completion_count
            record['status'] = 'completed'
            record['outputs'] = self.tensors(output)
        self.completion_count += 1

    def snapshot(self, error: BaseException | None = None) -> dict:
        terminal = error
        seen = set()
        linked = False
        while terminal is not None and id(terminal) not in seen:
            seen.add(id(terminal))
            linked = linked or terminal is self.failure_error
            terminal = terminal.__cause__ or terminal.__context__
        for record, module_error in self.module_errors:
            record['disposition'] = ('terminal' if module_error is error else 'exception_chain') if id(module_error) in seen else ('recovered' if error is None else 'unknown')
        self.module_errors.clear()
        for record in self.operations:
            if record['status'] == 'failed':
                record['disposition'] = 'recovered' if error is None else 'unknown'
        if self.failure is not None:
            self.failure['disposition'] = ('terminal' if error is self.failure_error else 'exception_chain') if linked else ('recovered' if error is None else 'unknown')
        return {'version': 1, 'operations': self.operations, 'operationCount': self.operation_count,
                'moduleCalls': self.module_calls, 'moduleCallCount': self.module_call_count,
                'failure': self.failure if linked else None,
                'lastObservedFailure': self.failure,
                'coverage': [{**module, 'reached': module['modulePath'] in self.reached,
                              'completedCalls': self.completed_counts.get(module['modulePath'], 0),
                              'failedCalls': self.failed_counts.get(module['modulePath'], 0)} for module in self.modules],
                'truncated': sorted(self.truncated)}
