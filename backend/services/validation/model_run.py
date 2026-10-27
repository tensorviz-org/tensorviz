"""Direct model execution inside the approved worker only (never imported by resolution).

Both execution modes share the bounded eager runtime recorder.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

from . import contracts as c
from .trace import TraceRecorder
from .setup import bind_call, ExecutionRequirement


def failure(phase, code, source=None):
    return dict(phase=phase, code=code, source=source)


def empty_record(spec):
    return dict(version=1, identity=spec.identity.model_dump(), sequence=1,
                limits=spec.profile.limits.model_dump(), process=dict(state="running"),
                capture=dict(state="incomplete", rootInvocationId=None, reason="unsupported"),
                context=None, modules=[], invocations=[], tensors=[], edges=[], requirements=[])


def materialize(value, torch, device):
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, c.FloatValue):
        return value.value
    if isinstance(value, c.NoneValue):
        return None
    if isinstance(value, c.SequenceValue):
        items = [materialize(item, torch, device) for item in value.items]
        return tuple(items) if value.kind == "tuple" else items
    if isinstance(value, c.DictValue):
        return {key: materialize(item, torch, device) for key, item in value.items.items()}
    gen = value.generator
    options = dict(dtype=getattr(torch, value.dtype), device=device)
    if isinstance(gen, c.Values):
        tensor = torch.tensor(gen.values, **options).reshape(value.shape)
    elif isinstance(gen, c.Full):
        tensor = torch.full(value.shape, gen.value, **options)
    else:
        # A device without local generators fails setup; never use the global RNG
        # or silently generate on CPU and transfer.
        generator = torch.Generator(device=device).manual_seed(gen.seed)
        if isinstance(gen, c.RandInt):
            tensor = torch.randint(gen.low, gen.high, value.shape, generator=generator, **options)
        else:
            tensor = torch.empty(value.shape, **options)
            if isinstance(gen, c.Normal):
                tensor.normal_(gen.mean, gen.std, generator=generator)
            else:
                tensor.uniform_(gen.low, gen.high, generator=generator)
    return tensor.requires_grad_(value.requires_grad)


def binding(value, torch, device):
    return ([materialize(item, torch, device) for item in value.args],
            {key: materialize(item, torch, device) for key, item in value.kwargs.items()})


class SetupError(Exception):
    def __init__(self, field, code="invalid"):
        self.field, self.code = field, code


def bind(target, args, kwargs, field):
    try:
        bind_call(target, args, kwargs, field)
    except ExecutionRequirement as error:
        raise SetupError(field, "unsupported" if error.diagnostic["code"] == "unknown_signature" else "signature") from error


def execute_model(spec: c.RunSpec, sources=None, workspace_root: str | None = None) -> c.ExecutionRecord:
    """One construction and one call, no retry and no exception text in records."""
    record = empty_record(spec)
    profile = spec.profile
    phase, field = "setup", "model"
    torch = None
    importing = False
    try:
        if not isinstance(profile, c.ModelProfile):
            raise SetupError("kind", "unsupported")
        importing = True
        import torch
        importing = False
        from .setup import configure_runtime, factory_model
        field = "device"
        device = configure_runtime(torch, dict(device=profile.device, seed=profile.seed,
                                               deterministic=profile.deterministic))
        # Check even parameterless models and tensorless input bindings.
        torch.empty(0, device=device)
        field = "model"
        module_name, symbol = profile.model.split(":")
        importing = True
        target = importlib.import_module(module_name)
        module_source = getattr(target, "__file__", None)
        importing = False
        for part in symbol.split("."):
            target = getattr(target, part)
        if not callable(target):
            raise SetupError("model")
        if profile.factory is not None:
            field = "factory"
            if workspace_root is None or not module_source:
                raise SetupError(field, "missing")
            phase = "construction"
            model = factory_model(workspace_root, profile.factory.model_dump(), target, Path(module_source))
        else:
            field = "init"
            args, kwargs = binding(profile.init, torch, device)
            bind(target, args, kwargs, field)
            phase = "construction"
            model = target(*args, **kwargs)
        phase, field = "setup", "model"
        if not isinstance(model, torch.nn.Module):
            raise SetupError(field)
        field = "device"
        model.to(device=device)
        field = "mode"
        model.train(profile.mode == "train")
        field = "inputs"
        if isinstance(profile.inputs, c.InputFactory):
            from ..ingest.execution_setup import factory_inputs
            if workspace_root is None:
                raise SetupError(field, 'missing')
            args, kwargs = factory_inputs(workspace_root, profile.inputs.factory.model_dump(), device)
        else:
            args, kwargs = binding(profile.inputs, torch, device)
        bind(model.forward, args, kwargs, field)
        # Custom __call__ has its own Python binding, too.
        bind(model.__call__, args, kwargs, field)
        with torch.set_grad_enabled(profile.grad_enabled):
            phase = "forward"
            recorder = TraceRecorder(record, torch, sources)
            recorder.run(model, args, kwargs)
            if recorder.reason == "recorder_error":
                record["process"] = dict(state="failed", failure=failure("recorder", "recorder_failed"))

    except BaseException as error:
        # KeyboardInterrupt/SystemExit in approved code are failures too. The host
        # remains authoritative for actual process termination and cancellation.
        code = "import_failed" if importing else f"{phase}_failed"
        if isinstance(error, ModuleNotFoundError):
            code = "missing_dependency"
        if isinstance(error, MemoryError) or (torch is not None and isinstance(error, torch.OutOfMemoryError)):
            phase, code = "resource", "allocation_failed"
        if isinstance(error, SetupError):
            field, code = error.field, error.code
        if isinstance(error, ExecutionRequirement):
            field, code = error.diagnostic["field"][0], error.diagnostic["code"]
        try:
            source = sources.failure(error) if sources is not None else None
        except Exception:
            source = None  # Optional evidence must never replace the real error.
        record["process"] = dict(state="failed", failure=failure(phase, code, source))
        call = record["invocations"][0] if record["invocations"] else None
        if call is not None and call["outcome"] != "succeeded":
            call.update(outcome="failed", output=None, failure=failure(phase, code, source))
        if phase in ("setup", "construction"):
            record["requirements"] = [dict(
                profile=spec.identity.profile, field=["profiles", spec.identity.profile, field],
                phase="runtime", code=(error.code if isinstance(error, SetupError) else "environment"),
                reason="The configured model could not complete its required setup.",
                remedy="Check the selected environment, Python signature and explicit profile values; use a script for complex setup.")]
    if sources is not None:
        sources.clear()
    try:
        return c.ExecutionRecord.model_validate(record)
    except Exception:
        # Byte limits or unsupported metadata cannot be reported as model success.
        minimal = empty_record(spec)
        minimal["process"] = dict(state="failed", failure=failure("recorder", "record_unavailable"))
        size = len(json.dumps(record, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
        minimal["capture"]["reason"] = (
            "byte_limit" if size > profile.limits.max_record_bytes else "recorder_error")
        return c.ExecutionRecord.model_validate(minimal)
