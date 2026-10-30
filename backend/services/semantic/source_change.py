"""Closed integer-expression analysis over the existing lexical source index.

No project calls, imports, attribute reads or Python eval. Results are conditional
source arithmetic, not runtime values, tensor predictions or edit capabilities.
"""
import ast
from collections import deque


class _Unknown(ValueError):
    pass


class SourceChangeAnalysis:
    def __init__(self, index, seeds, after):
        self.index = index
        self.by_id = {identity: node for node, identity in index.nodes.items()}
        self.cache, self.visits = {}, 0
        if len(seeds) != 1:
            raise ValueError("one configuration definition required")
        self.seed, self.after = seeds[0], after
        value = self._definition(self.by_id[self.seed])
        before = self._literal(value)
        self.affected, pending = {self.seed}, deque([self.seed])
        while pending:
            for edge in index.outgoing[pending.popleft()]:
                if edge["target"] not in self.affected:
                    self.affected.add(edge["target"])
                    pending.append(edge["target"])
        self.metadata = dict(version=1, evidence="conditional_source_arithmetic", definitionId=self.seed,
            before=before, after=after, scope="source_records_on_this_page",
            assumptions=["Only the selected integer definition changes; supported direct lexical bindings retain their source meanings.",
                         "Branches, call order, import overrides and runtime mutation are not evaluated."],
            limitations=["Only integer literals, direct names/imports, unary signs and +, -, *, //, % are evaluated.",
                         "No execution, tensor shape, checkpoint compatibility or candidate-validity claim is made."])

    @staticmethod
    def _integer(value):
        if type(value) is not int or abs(value) > 9007199254740991:
            raise _Unknown("integer_limit")
        return value

    def _literal(self, node):
        if isinstance(node, ast.Constant):
            return self._integer(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = self._literal(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        raise _Unknown("unsupported_definition")

    def _definition(self, node):
        scope = self.index.scopes[node]
        parent = self.index.parents.get(node)
        owner = self.index.parents.get(parent)
        if (not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Store)
                or not isinstance(owner, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef))
                or scope.bindings[node.id] != [self.index.nodes[node]]
                or scope.declarations or self.index.roots[scope.file].declarations
                or not (isinstance(parent, ast.Assign) and len(parent.targets) == 1 and parent.targets[0] is node
                        or isinstance(parent, ast.AnnAssign) and parent.target is node and parent.value is not None)):
            raise _Unknown("unsupported_definition")
        return parent.value

    def _pair(self, node, depth=0):
        if depth > 64 or self.visits >= 4096:
            raise _Unknown("analysis_limit")
        if node in self.cache:
            return self.cache[node]
        self.visits += 1
        identity = self.index.nodes.get(node)
        if identity == self.seed:
            result = (self.metadata["before"], self.after)
        elif isinstance(node, ast.Constant):
            value = self._integer(node.value)
            result = (value, value)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            result = self._pair(self._definition(node), depth + 1)
        elif isinstance(node, (ast.Name, ast.Attribute, ast.alias)):
            record = self.index.records.get(identity, {})
            if record.get("bindingKind") == "parameter":
                raise _Unknown("default_parameter")
            if not isinstance(node, ast.alias) and record.get("resolution") != "lexical":
                raise _Unknown("unresolved_binding")
            edges = [edge for edge in self.index.incoming[identity] if edge["relation"] in ("reference", "import")]
            if len(edges) != 1 or edges[0]["certainty"] != "lexical":
                raise _Unknown("unresolved_binding")
            result = self._pair(self.by_id[edges[0]["source"]], depth + 1)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            values = self._pair(node.operand, depth + 1)
            result = tuple(-value if isinstance(node.op, ast.USub) else value for value in values)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)):
            left, right = self._pair(node.left, depth + 1), self._pair(node.right, depth + 1)
            def calculate(a, b):
                if isinstance(node.op, ast.Add): return a + b
                if isinstance(node.op, ast.Sub): return a - b
                if isinstance(node.op, ast.Mult): return a * b
                if not b: raise _Unknown("division_by_zero")
                return a // b if isinstance(node.op, ast.FloorDiv) else a % b
            result = tuple(self._integer(calculate(a, b)) for a, b in zip(left, right))
        else:
            raise _Unknown("unsupported_expression")
        self.cache[node] = result
        return result

    def _scalar(self, node):
        try:
            before, after = self._pair(node)
            return dict(status="derived", before=before, after=after, reason="integer_expression")
        except _Unknown as error:
            return dict(status="unknown", before=None, after=None, reason=str(error))

    def effect(self, identity):
        if identity not in self.affected:
            return None
        node = self.by_id[identity]
        if not isinstance(node, ast.Call):
            return dict(kind="scalar", **self._scalar(node))
        arguments, omitted, values, omitted_values = [], 0, [], 0
        candidates = [(["args", i], arg) for i, arg in enumerate(node.args)] + [
            (["kwargs", arg.arg] if arg.arg else ["unpacking"], arg.value) for arg in node.keywords]
        for path, argument in candidates:
            leaves = [(path + ["items", i], item) for i, item in enumerate(argument.elts)] if isinstance(argument, (ast.List, ast.Tuple)) else [(path, argument)]
            for leaf_path, leaf in leaves:
                if len(values) >= 32:
                    omitted_values += 1
                else:
                    values.append(dict(path=leaf_path, **self._scalar(leaf)))
            if not any(self.index.nodes.get(child) in self.affected for child in ast.walk(argument)):
                continue
            if len(arguments) >= 16:
                omitted += 1
            else:
                arguments.append(dict(path=path, **self._scalar(argument)))
        return dict(kind="call", arguments=arguments, omittedArguments=omitted,
            argumentValues=values, omittedArgumentValues=omitted_values,
            hasExpansion=any(isinstance(arg, ast.Starred) for arg in node.args) or any(arg.arg is None for arg in node.keywords))
