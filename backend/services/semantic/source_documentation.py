"""Declaration documentation from the bounded, supplied-source AST index.

This resolves lexical declarations, never imports project code or predicts a
runtime tensor type. Uncertain bindings deliberately have no documentation.
"""
from __future__ import annotations

import ast
from copy import copy
from dataclasses import dataclass


_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
_DEFINITIONS = (*_FUNCTIONS, ast.ClassDef)
_CONDITIONAL = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.TryStar,
                ast.With, ast.AsyncWith, ast.Match)
_JS_WHITESPACE = " \t\n\r\v\f\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"


class _Unavailable(Exception):
    def __init__(self, reason):
        self.reason = reason


@dataclass(frozen=True)
class _Module:
    name: str


@dataclass(frozen=True)
class _Instance:
    declaration: ast.ClassDef


def _utf16_size(value):
    return len(value.encode("utf-16-le", errors="surrogatepass")) // 2


def _docstring(value):
    filtered = "".join(char for char in value if (ord(char) >= 32 and ord(char) != 127
                      and not 0xd800 <= ord(char) <= 0xdfff) or char in "\n\t\r")
    cleaned = filtered.strip(_JS_WHITESPACE)
    changed = filtered != value
    if _utf16_size(cleaned) > 4096:
        cleaned = cleaned.encode("utf-16-le")[:8190].decode("utf-16-le", errors="ignore").rstrip() + "…"
        changed = True
    return cleaned, changed


class SourceDocumentation:
    """Reusable lookup tables; all per-query traversal state stays local."""

    def __init__(self, index):
        self.index = index
        self.by_id = {identity: node for node, identity in index.nodes.items()}
        self.classes, self.class_scopes, self.writes = {}, {}, {}
        self.unsafe_scopes, self.mutations = set(), []
        for file, tree in index.trees.items():
            for node in ast.walk(tree):
                scope = index.scopes.get(node)
                if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
                    self.unsafe_scopes.add(id(scope))
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if scope is None:
                        if node.func.id in ("exec", "eval", "globals", "locals", "vars", "setattr", "delattr"):
                            self.unsafe_scopes.add(id(index.roots[file]))
                        continue
                    if node.func.id in ("exec", "eval", "globals", "locals", "vars"):
                        self.unsafe_scopes.add(id(scope))
                        if node.func.id in ("exec", "eval", "globals"):
                            # These can affect a module namespace from a nested scope.
                            self.unsafe_scopes.add(id(index.roots[scope.file]))
                    elif node.func.id in ("setattr", "delattr"):
                        self.mutations.append(node)
        for node, identity in index.nodes.items():
            if isinstance(node, ast.ClassDef):
                self.classes[(index.scopes[node].file, index.records[identity]["anchor"]["scope"])] = node
            elif isinstance(node, ast.Attribute) and not isinstance(node.ctx, ast.Load):
                self.writes.setdefault(node.attr, []).append(node)
        for scope in index.scopes.values():
            if scope.kind == "class":
                self.class_scopes[(scope.file, scope.name)] = scope

    def query(self, seeds):
        candidates = set()
        for identity in seeds:
            node = self.by_id.get(identity)
            parent = self.index.parents.get(node)
            if isinstance(parent, ast.Call) and parent.func is node:
                node = parent
            if isinstance(node, (*_DEFINITIONS, ast.Call)):
                candidates.add(node)
        if not candidates:
            return None
        if len(candidates) != 1:
            return dict(status="unavailable", reason="ambiguous")
        try:
            return _Resolution(self).document(next(iter(candidates)))
        except _Unavailable as error:
            return dict(status="unavailable", reason=error.reason)
        except RecursionError:
            return dict(status="unavailable", reason="analysis_limit")


class _Resolution:
    def __init__(self, owner):
        self.owner, self.index, self.steps = owner, owner.index, 0
        self.validated_classes, self.validating_classes = set(), set()

    def document(self, selected):
        declaration = self.expression(selected.func, self.index.scopes[selected], set()) if isinstance(selected, ast.Call) else selected
        if isinstance(declaration, _Instance):
            declaration = declaration.declaration
        self.declaration(declaration)
        header = copy(declaration)
        header.decorator_list, header.body = [], [ast.Pass()]
        signature = ast.unparse(header).split("\n", 1)[0].removesuffix(":")
        if _utf16_size(signature) > 2048 or any(ord(char) < 32 or ord(char) == 127 for char in signature):
            raise _Unavailable("analysis_limit")
        identity = self.index.nodes[declaration]
        scope = self.index.scopes[declaration]
        qualified = self.index.records[identity]["anchor"]["scope"]
        if _utf16_size(qualified) > 512:
            raise _Unavailable("analysis_limit")
        result = dict(status="resolved", qualifiedName=qualified,
                      kind="class" if isinstance(declaration, ast.ClassDef) else "method" if scope.kind == "class" else "function",
                      signature=signature, truncated=False, declaration=dict(self.index.records[identity]["anchor"]))
        doc = ast.get_docstring(declaration, clean=True)
        if doc:
            cleaned, changed = _docstring(doc)
            if cleaned:
                result["docstring"] = cleaned
            result["truncated"] = changed
        return result

    def step(self):
        self.steps += 1
        if self.steps > 128:
            raise _Unavailable("analysis_limit")

    def unconditional(self, node):
        current = self.index.parents.get(node)
        while current and not isinstance(current, (ast.Module, *_DEFINITIONS)):
            if isinstance(current, _CONDITIONAL):
                raise _Unavailable("dynamic_binding")
            current = self.index.parents.get(current)

    def declaration(self, node):
        if not isinstance(node, _DEFINITIONS):
            raise _Unavailable("unsupported_callable")
        self.unconditional(node)
        scope = self.index.scopes[node]
        if id(self.index.roots[scope.file]) in self.owner.unsafe_scopes or id(scope) in self.owner.unsafe_scopes:
            raise _Unavailable("dynamic_binding")
        if len(scope.bindings.get(node.name, [])) != 1:
            raise _Unavailable("ambiguous")
        if node.decorator_list:
            raise _Unavailable("decorated")
        if isinstance(node, ast.ClassDef):
            if node in self.validating_classes:
                raise _Unavailable("dynamic_binding")
            if node in self.validated_classes:
                return
            if node.keywords or any(isinstance(child, _FUNCTIONS) and child.name in
                                   ("__getattr__", "__getattribute__", "__new__", "__setattr__") for child in node.body):
                raise _Unavailable("dynamic_binding")
            self.validating_classes.add(node)
            try:
                for base in node.bases:
                    try:
                        target = self.expression(base, scope, set())
                    except _Unavailable as error:
                        if error.reason not in ("unresolved_symbol", "unresolved_import", "unsupported_callable"):
                            raise
                    else:
                        if isinstance(target, ast.ClassDef):
                            self.declaration(target)
                self.validated_classes.add(node)
            finally:
                self.validating_classes.remove(node)
        elif scope.kind == "class":
            self.declaration(self.owner.classes[(scope.file, scope.name)])

    def binding(self, identities, trail):
        self.step()
        if len(identities) != 1:
            raise _Unavailable("ambiguous" if identities else "unresolved_symbol")
        identity = identities[0]
        if identity in trail:
            raise _Unavailable("dynamic_binding")
        trail = trail | {identity}
        node = self.owner.by_id[identity]
        self.unconditional(node)
        if isinstance(node, _DEFINITIONS):
            self.declaration(node)
            return node
        scope = self.index.scopes[node]
        if identity in self.index.imports:
            module, member, alias = self.index.imports[identity]
            if member is not None:
                return self.module_member(module, member, trail)
            if len(self.index.modules.get(module, [])) != 1:
                raise _Unavailable("unresolved_import")
            return _Module(module if alias else module.split(".")[0])
        parent = self.index.parents.get(node)
        if isinstance(node, (ast.Name, ast.Attribute)) and isinstance(parent, (ast.Assign, ast.AnnAssign)) and parent.value is not None:
            return self.expression(parent.value, scope, trail)
        raise _Unavailable("dynamic_binding")

    def module_member(self, module, name, trail):
        self.step()
        member_key = ("module", module, name)
        if member_key in trail:
            raise _Unavailable("dynamic_binding")
        trail = trail | {member_key}
        if self.attribute_writes(_Module(module), name, trail):
            raise _Unavailable("ambiguous")
        files = self.index.modules.get(module, [])
        if len(files) == 1:
            root = self.index.roots[files[0]]
            if id(root) in self.owner.unsafe_scopes:
                raise _Unavailable("dynamic_binding")
            if name in root.bindings:
                return self.binding(root.bindings[name], trail)
        elif len(files) > 1:
            raise _Unavailable("ambiguous")
        if len(self.index.modules.get(module + "." + name, [])) == 1:
            return _Module(module + "." + name)
        raise _Unavailable("unresolved_import")

    def receiver(self, scope, name):
        while scope:
            if name in scope.bindings:
                if len(scope.bindings[name]) == 1 and scope.receiver == name and scope.parent and scope.parent.kind == "class":
                    declaration = self.owner.classes[(scope.file, scope.parent.name)]
                    self.declaration(declaration)
                    return _Instance(declaration)
                return None
            scope = scope.parent
        return None

    def expression(self, node, scope, trail):
        self.step()
        if isinstance(node, ast.Name):
            receiver = self.receiver(scope, node.id)
            if receiver:
                return receiver
            current = scope
            while current:
                if current.kind == "opaque" or node.id in current.declarations or id(current) in self.owner.unsafe_scopes:
                    raise _Unavailable("dynamic_binding")
                if node.id in current.bindings:
                    return self.binding(current.bindings[node.id], trail)
                current = current.parent
                while current and current.kind == "class":
                    current = current.parent
            raise _Unavailable("unresolved_symbol")
        if isinstance(node, ast.Attribute):
            return self.member(self.expression(node.value, scope, trail), node.attr, trail)
        if isinstance(node, ast.Call):
            target = self.expression(node.func, scope, trail)
            if isinstance(target, ast.ClassDef):
                self.declaration(target)
                return _Instance(target)
        raise _Unavailable("unsupported_callable")

    def member(self, owner, name, trail):
        self.step()
        if isinstance(owner, _Module):
            return self.module_member(owner.name, name, trail)
        declaration = owner.declaration if isinstance(owner, _Instance) else owner
        if not isinstance(declaration, ast.ClassDef):
            raise _Unavailable("unsupported_callable")
        self.declaration(declaration)
        identity = self.index.nodes[declaration]
        member_key = (identity, name)
        if member_key in trail:
            raise _Unavailable("dynamic_binding")
        trail = trail | {member_key}
        key = (self.index.scopes[declaration].file, self.index.records[identity]["anchor"]["scope"])
        scope = self.owner.class_scopes.get(key)
        bindings = scope.bindings.get(name, []) if scope else []
        assignments = self.attribute_writes(owner, name, trail)
        if bindings or assignments:
            return self.binding([*bindings, *assignments], trail)
        if len(declaration.bases) > 1:
            raise _Unavailable("ambiguous")
        if declaration.bases:
            base = self.expression(declaration.bases[0], self.index.scopes[declaration], trail)
            return self.member(base, name, trail)
        raise _Unavailable("unresolved_symbol")

    @staticmethod
    def same_owner(left, right):
        if isinstance(left, _Module) or isinstance(right, _Module):
            return left == right
        left = left.declaration if isinstance(left, _Instance) else left
        right = right.declaration if isinstance(right, _Instance) else right
        return left is right

    def attribute_writes(self, owner, name, trail):
        assignments = []
        for mutation in self.owner.mutations:
            if not mutation.args:
                continue
            attribute = mutation.args[1] if len(mutation.args) > 1 else None
            if isinstance(attribute, ast.Constant) and attribute.value != name:
                continue
            try:
                base = self.expression(mutation.args[0], self.index.scopes[mutation], trail)
            except _Unavailable:
                continue
            if self.same_owner(base, owner):
                raise _Unavailable("dynamic_binding")
        for write in self.owner.writes.get(name, []):
            try:
                base = self.expression(write.value, self.index.scopes[write], trail)
            except _Unavailable:
                continue
            if self.same_owner(base, owner):
                assignments.append(self.index.nodes[write])
        return assignments
