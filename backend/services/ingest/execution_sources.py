"""Workspace source actually compiled by standard Python loaders in one worker."""
from __future__ import annotations

import hashlib
import importlib.machinery
import os
from contextlib import contextmanager
from pathlib import Path

MAX_SOURCE_FILES = 4096
MAX_SOURCE_BYTES = 64 * 1024 * 1024


@contextmanager
def workspace_imports(workspace: str | None):
    records: list[dict] = []
    if workspace is None:
        yield records
        return
    root = Path(workspace).resolve(strict=True)
    lexical_root = Path(os.path.abspath(workspace))
    source_loader = importlib.machinery.SourceFileLoader
    bytecode_loader = importlib.machinery.SourcelessFileLoader
    original = source_loader.get_code
    original_bytecode = bytecode_loader.get_code
    total = 0

    def location(filename):
        lexical = Path(os.path.abspath(filename))
        prefix = next((base for base in (lexical_root, root) if lexical.is_relative_to(base)), None)
        if prefix is None:
            return None
        relative = lexical.relative_to(prefix)
        parts = relative.parts
        # Interpreter installations are environment dependencies, not user recipes.
        if any(part in ('site-packages', 'dist-packages') for part in parts):
            return None
        resolved = lexical.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise ValueError('An imported Python source escapes the execution workspace.')
        if any(part == '.env' or part.startswith('.env.') for part in (*parts, *resolved.relative_to(root).parts)):
            raise ValueError('Environment files cannot be execution source.')
        return relative, resolved

    def get_code(loader, fullname):
        nonlocal total
        paths = location(loader.path)
        if paths is None:
            return original(loader, fullname)
        relative, resolved = paths
        if len(records) >= MAX_SOURCE_FILES:
            raise ValueError('Execution exceeds the imported-source file limit.')
        # Compile the exact recorded bytes; timestamp-valid stale .pyc must never
        # be labelled as the current source. Do not write or invalidate user caches.
        with resolved.open('rb') as stream:
            source = stream.read(MAX_SOURCE_BYTES - total + 1)
        total += len(source)
        if total > MAX_SOURCE_BYTES:
            raise ValueError('Execution exceeds the imported-source byte limit.')
        records.append({'file': relative.as_posix(),
                        'realPath': str(resolved), 'bytes': len(source),
                        'digest': hashlib.sha256(source).hexdigest()})
        return loader.source_to_code(source, loader.path)

    def get_bytecode(loader, fullname):
        if location(loader.path) is not None:
            raise ValueError('Workspace bytecode-only imports cannot certify current Python source.')
        return original_bytecode(loader, fullname)

    source_loader.get_code = get_code
    bytecode_loader.get_code = get_bytecode
    try:
        yield records
    finally:
        source_loader.get_code = original
        bytecode_loader.get_code = original_bytecode
