"""Saved-code-verified authored locations, not permission to edit an ATen event.

PEP 657 positions are resolved against bounded source read before importing the
model. Exact means that this saved expression was executing; decomposition can
associate many events with that expression. No frame, code, tensor or source text
is retained in a capture. SourceAnchor columns use zero-based UTF-16 code units.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
from itertools import islice
import os
from pathlib import Path
import sys
import types
import warnings
import weakref

from ..validation.source import MAX_CODES, MAX_FRAMES, SourceEvidence
from .contracts import SourceBinding

MAX_SITES = 8192
MAX_AST_NODES = 50000


def _binding_tokens(node):
    """Keep AST token identity consistent across supported Python versions.

    ast.dump started omitting empty list fields in 3.13. Captures and the
    nonexecuting edit resolver can run in different Python environments.
    """
    if isinstance(node, ast.AST):
        fields = []
        for name, value in ast.iter_fields(node):
            if isinstance(value, list) and not value:
                continue
            if value is None and getattr(type(node), name, ...) is None:
                continue
            fields.append(f"{name}={_binding_tokens(value)}")
        return f"{type(node).__name__}({', '.join(fields)})"
    if isinstance(node, list):
        return f"[{', '.join(_binding_tokens(value) for value in node)}]"
    return repr(node)


def source_binding(snapshot_id, file, lines, node, kind, qualified):
    """One exact AST anchor, shared by runtime location and edit resolvers."""
    tokens = _binding_tokens(node)
    if len(qualified) > 512 or len(tokens) > 65536:
        return None

    def column(line, byte_column):
        prefix = lines[line - 1].encode("utf-8")[:byte_column].decode("utf-8")
        return len(prefix.encode("utf-16-le")) // 2

    anchor = dict(file=file.path, digest=file.digest, startLine=node.lineno,
        startColumn=column(node.lineno, node.col_offset), endLine=node.end_lineno,
        endColumn=column(node.end_lineno, node.end_col_offset), scope=qualified or "<module>",
        tokenDigest=hashlib.sha256(tokens.encode()).hexdigest())
    identity = f"{snapshot_id}:{kind}:{file.path}:{node.lineno}:{node.col_offset}:{node.end_lineno}:{node.end_col_offset}"
    return SourceBinding(id="binding:" + hashlib.sha256(identity.encode()).hexdigest(),
        snapshotId=snapshot_id, kind=kind, quality="exact", anchors=[anchor]).model_dump(mode="json")


class SourceIndex:
    """Immutable, nonexecuting AST locations from one verified saved file."""

    def __init__(self, snapshot_id, file, data, *, max_sites=None, retain_nodes=False):
        if max_sites is None:
            max_sites = MAX_SITES
        tree = ast.parse(data)
        if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
            raise ValueError("source index exceeds node budget")
        lines = importlib.util.decode_source(data).split("\n")
        self.expressions, self.definitions = {}, {}
        self.callees = {}
        self.shape_reads = {}
        self.expanded_calls = set()
        # Only the edit resolver retains AST nodes. Capture keeps bounded
        # locations and simple call recipes without retaining the tree.
        self.bound_nodes = {}
        self.definition_nodes = {}
        self.tree = tree if retain_nodes else None
        scope = []
        forward_definitions = []
        count = 0

        def binding(node, kind, qualified):
            nonlocal count
            if count >= max_sites:
                return None
            found = source_binding(snapshot_id, file, lines, node, kind, qualified)
            if found is not None:
                count += 1
                if retain_nodes and kind == "call":
                    self.bound_nodes[found["id"]] = (found, node)
                if retain_nodes and kind == "definition":
                    self.definition_nodes[found["id"]] = (found, node)
                if isinstance(node, ast.Call):
                    if any(isinstance(arg, ast.Starred) for arg in node.args) or any(kw.arg is None for kw in node.keywords):
                        self.expanded_calls.add(found["id"])
                    parts, callee = [], node.func
                    while isinstance(callee, ast.Attribute) and len(parts) < 16:
                        parts.append(callee.attr)
                        callee = callee.value
                    if isinstance(callee, ast.Name):
                        parts.append(callee.id)
                        if len(parts) <= 16 and sum(map(len, parts)) <= 512:
                            self.callees[found["id"]] = tuple(reversed(parts))
                    arguments = [(["args", i], arg) for i, arg in enumerate(node.args)] + [
                        (["kwargs", arg.arg], arg.value) for arg in node.keywords if arg.arg is not None]
                    reads = 0
                    for path, argument in arguments:
                        leaves = [(path + ["items", i], item) for i, item in enumerate(argument.elts)] if isinstance(argument, (ast.List, ast.Tuple)) else [(path, argument)]
                        for leaf_path, leaf in leaves:
                            if (reads >= 32 or not forward_definitions or forward_definitions[-1] is None or not isinstance(leaf, ast.Subscript)
                                    or not isinstance(leaf.value, ast.Attribute) or leaf.value.attr != "shape"
                                    or not isinstance(leaf.value.value, ast.Name)):
                                continue
                            axis, sign = leaf.slice, 1
                            if isinstance(axis, ast.UnaryOp) and isinstance(axis.op, (ast.USub, ast.UAdd)):
                                sign = -1 if isinstance(axis.op, ast.USub) else 1
                                axis = axis.operand
                            if isinstance(axis, ast.Constant) and type(axis.value) is int and -32 <= sign * axis.value <= 31:
                                attribute = leaf.value
                                key = (qualified, attribute.lineno, attribute.end_lineno, attribute.col_offset, attribute.end_col_offset)
                                self.shape_reads[key] = dict(callId=found["id"], path=leaf_path, axis=sign * axis.value,
                                    definitionId=forward_definitions[-1])
                                reads += 1
            return found

        def visit(node):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = ".".join([*scope, node.name])
                found = binding(node, "definition", qualified)
                if found is not None:
                    self.definitions.setdefault(qualified, []).append(found)
                scope.append(node.name)
                if not isinstance(node, ast.ClassDef):
                    scope.append("<locals>")
                    body = node.body[1:] if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and type(node.body[0].value.value) is str else node.body
                    # A source call may run at most once in this actual forward
                    # frame. Loops, exception retries, comprehensions and helper
                    # frames cannot reuse an earlier property-read observation.
                    single_pass = (isinstance(node, ast.FunctionDef) and node.name == "forward" and not node.decorator_list
                        and body and len(body) <= 128 and isinstance(body[-1], ast.Return)
                        and all(isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)
                            or isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None for stmt in body[:-1])
                        and not any(isinstance(child, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
                            ast.Yield, ast.YieldFrom, ast.Await, ast.IfExp)) for stmt in body for child in ast.walk(stmt)))
                    forward_definitions.append(found["id"] if found and single_pass else None)
                # Expressions inside a function have its code scope, without
                # <locals>; nested definitions introduce that component.
                for child in node.body:
                    visit(child)
                if not isinstance(node, ast.ClassDef):
                    scope.pop()
                    forward_definitions.pop()
                scope.pop()
                return
            if isinstance(node, (ast.Call, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Subscript)):
                qualified = ".".join(scope[:-1] if scope and scope[-1] == "<locals>" else scope)
                found = binding(node, "call", qualified)
                if found is not None:
                    key = (qualified or "<module>", node.lineno, node.end_lineno, node.col_offset, node.end_col_offset)
                    self.expressions.setdefault(key, []).append(found)
            for child in ast.iter_child_nodes(node):
                visit(child)

        visit(tree)
        self.site_count = count

    def expression(self, code, position):
        candidates = self.expressions.get((code.co_qualname, *position), [])
        return candidates[0] if len(candidates) == 1 else None


class CapturedSourceEvidence(SourceEvidence):
    def __init__(self, snapshot, workspace_root):
        from .construction import ConstructionEvidence
        self.construction = ConstructionEvidence(self)
        self.indexes = {}
        self.sites = {}
        self.source_buffers = {}
        self.snapshot_id = snapshot.id
        self.remaining_sites = MAX_SITES

        def loaded(file, data):
            # Keep the bounded verified bytes, never reread a file after import.
            # Parse only code we actually encounter so unused repository files
            # cannot exhaust the expression budget before the model is reached.
            self.source_buffers[file.path] = (file, data)

        super().__init__(snapshot, workspace_root, on_source=loaded)
        directory = Path(__file__).parent
        self.internal.update({__file__, str(directory / "recorder.py"), str(directory / "source_shape_call.py"),
                              str(directory.parent / "ingest" / "trace.py")})
        import torch
        self.torch_directory = os.path.dirname(torch.__file__) + os.sep

    def source_index(self, relative):
        if relative in self.indexes:
            return self.indexes[relative]
        self.indexes[relative] = None
        saved = self.source_buffers.pop(relative, None)
        if saved is not None and self.remaining_sites > 0:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    index = SourceIndex(self.snapshot_id, *saved, max_sites=self.remaining_sites)
                self.indexes[relative] = index
                self.remaining_sites -= index.site_count
            except (ValueError, SyntaxError, RecursionError, UnicodeError):
                pass  # Unavailable mapping never invalidates captured computation.
        return self.indexes[relative]

    def expression(self, code, last_instruction):
        # Python 3.10 has line evidence but no instruction column ranges. Never
        # invent an expression binding (and source-edit capability) from a line.
        if not hasattr(code, 'co_positions'):
            return None
        location = self.location(code, code.co_firstlineno)
        index = self.source_index(location["file"]) if location else None
        if index is None or last_instruction < 0:
            return None
        key = (id(code), last_instruction)
        previous = self.sites.get(key)
        if previous is not None and previous[0]() is code:
            return previous[1]
        if len(self.sites) >= MAX_CODES:
            return None
        position = next(islice(code.co_positions(), last_instruction // 2, None), None)
        found = index.expression(code, position) if position and all(item is not None for item in position) else None
        self.sites[key] = (weakref.ref(code), found)
        return found

    def call_binding(self):
        frame = sys._getframe(1)
        try:
            for _ in range(MAX_FRAMES):
                if frame is None:
                    break
                filename = frame.f_code.co_filename
                if filename not in self.internal and not filename.startswith(self.torch_directory):
                    # Do not skip an unverified wrapper to attribute its work to
                    # a convenient outer saved call.
                    return self.expression(frame.f_code, frame.f_lasti)
                frame = frame.f_back
        finally:
            del frame
        return None

    def constructor_callee(self, binding):
        if binding is None:
            return None
        index = self.indexes.get(binding["anchors"][0]["file"])
        return index.callees.get(binding["id"]) if index else None

    def direct_shape_call(self, target, receiver=None, method=None):
        """Match the callable and its transient frame identity, without retaining a frame."""
        from .construction import _is_direct_class, _lookup, _MISSING
        frame = sys._getframe(1)
        try:
            for _ in range(MAX_FRAMES):
                if frame is None:
                    return None
                filename = frame.f_code.co_filename
                if filename in self.internal or filename.startswith(self.torch_directory):
                    frame = frame.f_back
                    continue
                binding = self.expression(frame.f_code, frame.f_lasti)
                parts = self.constructor_callee(binding)
                if binding is None or binding["id"] in self.indexes[binding["anchors"][0]["file"]].expanded_calls:
                    return None
                namespace = frame.f_locals
                if type(namespace) is not dict and not frame.f_code.co_flags & inspect.CO_OPTIMIZED:
                    return None
                def lookup(name):
                    candidate = _lookup(dict.items(namespace) if type(namespace) is dict else namespace.items(), name)
                    return _lookup(dict.items(frame.f_globals), name) if candidate is _MISSING else candidate
                if method is None:
                    if not _is_direct_class(parts, frame, target):
                        return None
                elif not parts or len(parts) != 2 or parts[1] != method or lookup(parts[0]) is not receiver:
                    return None
                return binding, id(frame)
        finally:
            del frame
        return None

    def shape_read_site(self, caller_codes):
        """A direct shape getter inside one observed standard Module call frame."""
        frame = sys._getframe(1)
        try:
            for _ in range(MAX_FRAMES):
                if frame is None:
                    return None
                filename = frame.f_code.co_filename
                if filename in self.internal or filename.startswith(self.torch_directory):
                    frame = frame.f_back
                    continue
                if frame.f_back is None or not any(frame.f_back.f_code is code for code in caller_codes):
                    return None
                code = frame.f_code
                if not hasattr(code, "co_positions") or frame.f_lasti < 0:
                    return None
                location = self.location(code, code.co_firstlineno)
                index = self.source_index(location["file"]) if location else None
                position = next(islice(code.co_positions(), frame.f_lasti // 2, None), None)
                site = index.shape_reads.get((code.co_qualname, *position)) if index and position else None
                return (site, id(frame)) if site is not None else None
        finally:
            del frame
        return None

    def definition_binding(self, module):
        cls = type(module)
        forward = inspect.getattr_static(cls, "forward", None)
        if type(forward) is not types.FunctionType:
            return None
        code = forward.__code__
        if getattr(code, 'co_qualname', None) != cls.__qualname__ + ".forward":
            return None  # Inheritance/decorators are not a proven class definition.
        location = self.location(code, code.co_firstlineno)
        index = self.source_index(location["file"]) if location else None
        # A verified function proves its own implementation location, not that
        # this runtime class was created by a particular class statement.
        candidates = index.definitions.get(code.co_qualname, []) if index else []
        return candidates[0] if len(candidates) == 1 else None

    def clear(self):
        super().clear()
        self.construction.clear()
        self.sites.clear()
        self.indexes.clear()
        self.source_buffers.clear()
