"""Observe one existing eager module call without constructing or configuring it.

The selected call uses the same eager runtime recorder as direct model execution.
"""
from __future__ import annotations

import builtins
import importlib.machinery
import json
import sys
import threading
import types

from . import contracts as c
from .model_run import empty_record
from .trace import TraceRecorder


class ScriptRecorder:
    def __init__(self, spec, publish, sources=None):
        self.spec, self.publish = spec, publish
        self.sources = sources
        self.record = empty_record(spec)
        self.record["capture"]["reason"] = "target_not_observed"
        self.count = 0
        self.selected = False
        self.finished = False
        self.thread = None
        self.torch = self.original = self.wrapper = None

    def install(self, torch):
        self.torch = torch
        self.original = torch.nn.Module._call_impl

        def observe(module, *args, **kwargs):
            return self.call(module, args, kwargs)

        self.wrapper = observe
        torch.nn.Module._call_impl = observe

    def restore(self):
        if self.torch is not None and self.torch.nn.Module._call_impl is self.wrapper:
            self.torch.nn.Module._call_impl = self.original

    def recording_error(self):
        self.record["capture"]["reason"] = "recorder_error"

    def call(self, module, args, kwargs):
        # Never include another thread as a nested call in the selected tree.
        if self.finished or (self.thread is not None and threading.get_ident() != self.thread):
            return self.original(module, *args, **kwargs)
        root = False
        if not self.selected:
            symbol = f"{type(module).__module__}:{type(module).__qualname__}"
            if symbol == self.spec.profile.capture.model:
                self.count += 1
                if self.count == self.spec.profile.capture.call:
                    self.selected = root = True
                    self.thread = threading.get_ident()
                    self.record["capture"]["reason"] = "unsupported"
            if not root:
                return self.original(module, *args, **kwargs)
        # Remove selection instrumentation before the shared tracer installs its
        # nested-module wrapper. Restore both wrappers even when user code raises.
        self.restore()
        tracer = TraceRecorder(self.record, self.torch, self.sources)
        try:
            return tracer.run(module, args, kwargs, lambda: self.original(module, *args, **kwargs))
        finally:
            try:
                self.finish()
            except Exception:
                self.restore()

    def finish(self):
        if self.finished:
            return
        self.finished = True
        self.restore()
        if self.record["context"] is not None:
            self.record["context"]["devices"] = sorted({t["device"] for t in self.record["tensors"]})
        try:
            record = c.ExecutionRecord.model_validate(self.record)
        except Exception:
            size = len(json.dumps(self.record, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
            record = empty_record(self.spec)
            record["capture"]["reason"] = (
                "byte_limit" if size > self.spec.profile.limits.max_record_bytes else "recorder_error")
            record = c.ExecutionRecord.model_validate(record)
        self.publish(record)


class TorchObserver:
    """Install only after the program imports real torch, respecting import order."""
    def __init__(self, recorder):
        self.recorder = recorder

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "torch":
            return None
        sys.meta_path.remove(self)
        for finder in sys.meta_path:
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                loader = spec.loader
                observer = self

                class Loader:
                    def create_module(self, spec):
                        return loader.create_module(spec) if hasattr(loader, "create_module") else None

                    def exec_module(self, module):
                        module.__loader__ = module.__spec__.loader = loader
                        loader.exec_module(module)
                        try:
                            observer.recorder.install(module)
                        except Exception:
                            observer.recorder.recording_error()
                            observer.recorder.restore()

                spec.loader = Loader()
                return spec
        return None


def execute_script(spec, script, publish, sources=None):
    recorder = ScriptRecorder(spec, publish, sources)
    observer = TorchObserver(recorder)
    if "torch" in sys.modules:
        # A user's sitecustomize may already have imported it before bootstrap.
        try:
            recorder.install(sys.modules["torch"])
        except Exception:
            recorder.recording_error()
            recorder.restore()
    else:
        sys.meta_path.insert(0, observer)
    # A persistent __main__ preserves imports, pickling and atexit callbacks.
    main = types.ModuleType("__main__")
    main.__file__ = script
    main.__cached__ = None
    main.__builtins__ = builtins
    main.__annotations__ = {}
    main.__loader__ = importlib.machinery.SourceFileLoader("__main__", script)
    sys.modules["__main__"] = main
    sys.argv = [script, *spec.profile.args]
    try:
        with open(script, "rb") as stream:
            code = compile(stream.read(), script, "exec", dont_inherit=True)
        exec(code, main.__dict__)
    except ModuleNotFoundError:
        if not recorder.finished:
            recorder.record["requirements"] = [dict(
                profile=spec.identity.profile, field=["profiles", spec.identity.profile, "script"],
                phase="runtime", code="environment", reason="The program requires an unavailable dependency.",
                remedy="Install the program's dependency in the selected environment.")]
        raise
    finally:
        # CPython's file runner removes these launch attributes before atexit.
        main.__dict__.pop("__file__", None)
        main.__dict__.pop("__cached__", None)
        if observer in sys.meta_path:
            sys.meta_path.remove(observer)
        # Instrumentation/transport failure must never replace program outcomes.
        try:
            recorder.finish()
        except Exception:
            recorder.restore()
