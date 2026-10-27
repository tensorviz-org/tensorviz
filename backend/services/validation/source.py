"""Bounded, non-executing saved-source evidence for the private worker.

Filenames alone are not evidence: runtime code must match code compiled from a
RunSpec source with the expected digest. Locations are line-level (column zero).
No frames, exceptions, source text or constants are retained in the record.
"""
from __future__ import annotations

import hashlib
import inspect
import marshal
import os
from pathlib import Path
import stat
import sys
import types
import warnings
import weakref

MAX_FILES = 128
MAX_BYTES = 8 * 1024 * 1024
MAX_FILE_BYTES = 1024 * 1024
MAX_CODES = 8192
MAX_FRAMES = 64


def code_digest(code):
    # Marshal v2 has no object-reference flags: equivalent compiled code hashes
    # identically before and after execution. Bound and type-check its entire
    # input first, including replaced co_consts, without equality/repr callbacks.
    pending = [(code, 0)]
    size = count = 0
    while pending:
        value, depth = pending.pop()
        count += 1
        if count > MAX_CODES or depth > 32 or size > MAX_FILE_BYTES:
            return None
        kind = type(value)
        if kind is types.CodeType:
            size += len(value.co_code) + len(value.co_linetable) + len(getattr(value, 'co_exceptiontable', b''))
            pending.extend((item, depth + 1) for item in (
                value.co_consts, value.co_names, value.co_varnames, value.co_freevars,
                value.co_cellvars, value.co_filename, value.co_name, getattr(value, 'co_qualname', value.co_name)))
        elif kind in (tuple, frozenset):
            if len(value) + len(pending) + count > MAX_CODES:
                return None
            pending.extend((item, depth + 1) for item in value)
        elif kind in (str, bytes):
            size += len(value) * (4 if kind is str else 1)
        elif kind is int:
            size += value.bit_length() // 8 + 1
        elif kind not in (float, complex, bool, type(None), type(Ellipsis)):
            return None
    if size > MAX_FILE_BYTES:
        return None
    return hashlib.sha256(marshal.dumps(code, 2)).digest()


class SourceEvidence:
    def __init__(self, spec, workspace_root, *, on_source=None):
        self.files = {}
        self.cache = {}
        self.internal = {__file__, str(Path(__file__).with_name("trace.py"))}
        self.torch_directory = None
        # Prepare before importing/executing project code. Compilation never
        # executes it and must not emit duplicate warnings on the user's pipes.
        root = Path(workspace_root).resolve()
        budget = codes = examined = 0
        # Unrelated files must not displace the selected model's call sites.
        model_file = getattr(spec, "modelFile", None)
        for file in sorted(spec.files, key=lambda file: file.path != model_file):
            if file.role != "source":
                continue
            examined += 1
            if examined > MAX_FILES or budget >= MAX_BYTES or codes >= MAX_CODES:
                break
            try:
                path = (root / file.path).resolve(strict=True)
                if path.relative_to(root).as_posix() != file.path:
                    continue
                info = path.stat()
                if not stat.S_ISREG(info.st_mode) or info.st_size > min(MAX_FILE_BYTES, MAX_BYTES - budget):
                    continue
                descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(descriptor, "rb") as stream:
                    if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                        continue
                    data = stream.read(min(MAX_FILE_BYTES, MAX_BYTES - budget) + 1)
                budget += len(data)
                if len(data) > MAX_FILE_BYTES or budget > MAX_BYTES or hashlib.sha256(data).hexdigest() != file.digest:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    compiled = compile(data, str(path), "exec", dont_inherit=True)
                known = set()
                pending = [compiled]
                while pending and codes < MAX_CODES:
                    code = pending.pop()
                    codes += 1
                    fingerprint = code_digest(code)
                    if fingerprint is not None:
                        known.add(fingerprint)
                    pending.extend(item for item in code.co_consts if type(item) is types.CodeType)
                self.files[str(path)] = (file.path, known)
                if on_source is not None:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        on_source(file, data)
            except (OSError, ValueError, SyntaxError, RecursionError):
                continue

    def location(self, code, line):
        saved = self.files.get(code.co_filename)
        if saved is None or line is None or line < 1:
            return None
        # Cache by identity, not code-object equality (co_consts can be replaced
        # with arbitrary objects). Weak references cannot retain those values.
        prior = self.cache.get(id(code))
        if prior is None or prior[0]() is not code:
            if len(self.cache) >= MAX_CODES:
                return None
            try:
                fingerprint = code_digest(code)
            except (ValueError, TypeError, RecursionError):
                fingerprint = None
            valid = fingerprint is not None and fingerprint in saved[1]
            self.cache[id(code)] = (weakref.ref(code), valid)
        else:
            valid = prior[1]
        return dict(file=saved[0], line=line, column=0) if valid else None

    def module(self, module):
        # Do not bind forward or unwrap arbitrary descriptors/decorators.
        forward = inspect.getattr_static(module, "forward", None)
        if type(forward) in (staticmethod, classmethod, types.MethodType):
            forward = forward.__func__
        if type(forward) is types.FunctionType:
            code = forward.__code__
            return self.location(code, code.co_firstlineno)
        return None

    def callsite(self):
        frame = sys._getframe(1)
        try:
            for _ in range(MAX_FRAMES):
                if frame is None:
                    break
                filename = frame.f_code.co_filename
                if filename not in self.internal and not (
                        self.torch_directory and filename.startswith(self.torch_directory)):
                    # An unverified user/dynamic/external boundary is not
                    # attributed to an earlier caller with a convenient name.
                    return self.location(frame.f_code, frame.f_lineno)
                frame = frame.f_back
        finally:
            del frame
        return None

    def failure(self, error):
        traceback = BaseException.__traceback__.__get__(error)
        location = None
        try:
            for _ in range(MAX_FRAMES):
                if traceback is None:
                    return location
                found = self.location(traceback.tb_frame.f_code, traceback.tb_lineno)
                if found is not None:
                    location = found
                traceback = traceback.tb_next
            return None  # truncated traceback cannot identify the deepest site
        finally:
            del traceback

    def clear(self):
        self.cache.clear()
