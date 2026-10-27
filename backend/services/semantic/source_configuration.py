"""Nonexecuting lexical proof for directly assigned numeric configuration names.

No runtime value is substituted for a reference. Conditional/ambiguous writes,
parameters, closures, comprehensions and reflective namespaces need other recipes.
The caller owns literal validation, source anchors and whole-capture scope.
"""
import ast


class ConfigurationIndex:
    def __init__(self, tree):
        self.tree = tree
        self.parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        self.cache = {}
        self.dynamic = any(
            isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names)
            or isinstance(node, ast.Attribute) and node.attr == "__dict__"
            or isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in ("exec", "eval", "globals", "locals", "vars", "setattr", "delattr")
            for node in ast.walk(tree))

    def _owner(self, reference):
        parent = self.parents.get(reference)
        while parent is not None:
            if isinstance(parent, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.ClassDef)):
                return None
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                ancestor = self.parents.get(parent)
                while ancestor is not None:
                    if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                        return None  # A closure's name needs its own binding chain.
                    ancestor = self.parents.get(ancestor)
                return parent
            if isinstance(parent, ast.Module):
                return parent
            parent = self.parents.get(parent)
        return None

    def _names(self, owner, name):
        """Conservative: even a same-named binding in a nested scope refuses."""
        writes, reads = [], []
        for node in ast.walk(owner):
            if isinstance(node, ast.Name) and node.id == name:
                (reads if isinstance(node.ctx, ast.Load) else writes).append(node)
            elif isinstance(node, ast.arg) and node.arg == name:
                writes.append(node)
            elif node is not owner and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
                writes.append(node)
            elif isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names:
                writes.append(node)
            elif isinstance(node, (ast.Import, ast.ImportFrom)) and any(
                    (alias.asname or (alias.name.split(".")[0] if isinstance(node, ast.Import) else alias.name)) == name
                    for alias in node.names):
                writes.append(node)
            elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name == name:
                writes.append(node)
            elif isinstance(node, ast.MatchMapping) and node.rest == name:
                writes.append(node)
        return writes, reads

    def _statement(self, node, owner):
        while node in self.parents and self.parents[node] is not owner:
            node = self.parents[node]
        return node if node in owner.body else None

    def resolve(self, reference, scope):
        if self.dynamic or not isinstance(reference, ast.Name) or not isinstance(reference.ctx, ast.Load):
            return None
        owner = self._owner(reference)
        if owner is None:
            return None
        key = (owner, reference.id)
        if key not in self.cache:
            writes, reads = self._names(owner, reference.id)
            definition_owner = owner
            if not writes and owner is not self.tree:
                definition_owner = self.tree
                writes, reads = self._names(self.tree, reference.id)
            assignment = self.parents.get(writes[0]) if len(writes) == 1 else None
            single = (isinstance(assignment, ast.Assign) and len(assignment.targets) == 1 and assignment.targets[0] is writes[0]
                      or isinstance(assignment, ast.AnnAssign) and assignment.target is writes[0])
            self.cache[key] = ((assignment, definition_owner, len(reads)) if single
                               and assignment in definition_owner.body and assignment.value is not None else None)
        found = self.cache[key]
        if found is None:
            return None
        assignment, definition_owner, count = found
        statement = self._statement(reference, definition_owner)
        if statement is None or definition_owner.body.index(assignment) >= definition_owner.body.index(statement):
            return None
        return assignment.value, "<module>" if definition_owner is self.tree else scope, count
