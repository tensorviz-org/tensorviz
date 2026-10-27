"""Nonexecuting editor focus ranges for a verified projection's source spans.

Parameter uses are direct references in the declaring class, not a claim about
all runtime aliases or which methods ran. Every range stays in the span's file.
"""
import ast


FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
SCOPES = (ast.ClassDef, *FUNCTIONS, ast.Lambda)
MAX_REFERENCES = 256


def _range(node):
    decorators = getattr(node, "decorator_list", [])
    return [min([node.lineno, *(item.lineno for item in decorators)]), node.end_lineno]


def _inside(node, start, end):
    return node.lineno <= start <= end <= node.end_lineno


def _walk_scope(node):
    """Do not attribute nested scopes' receiver bindings to this method."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, SCOPES):
            continue
        yield child
        yield from _walk_scope(child)


def _receiver(method):
    args = method.args.posonlyargs + method.args.args
    if not args or any(isinstance(item, ast.Name) and item.id in ("staticmethod", "classmethod")
                       for item in method.decorator_list):
        return None
    name = args[0].arg
    # Rebinding, deleting, or a comprehension target can change what the name
    # means. Refuse that method instead of guessing which object owns an access.
    if any(isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, (ast.Store, ast.Del))
           for node in _walk_scope(method)):
        return None
    return name


class _SourceIndex:
    def __init__(self, source):
        self.tree = ast.parse(source)
        self.parents = {child: parent for parent in ast.walk(self.tree) for child in ast.iter_child_nodes(parent)}
        self.classes = [node for node in ast.walk(self.tree) if isinstance(node, ast.ClassDef)]
        self.functions = [node for node in ast.walk(self.tree) if isinstance(node, FUNCTIONS)]

    def enclosing(self, candidates, start, end):
        return min((node for node in candidates if _inside(node, start, end)),
                   key=lambda node: node.end_lineno - node.lineno, default=None)

    def parameter(self, owner, start, end):
        methods = [node for node in owner.body if isinstance(node, FUNCTIONS)]
        init = next((node for node in methods if node.name == "__init__"), None)
        receiver = _receiver(init) if init else None
        if not receiver:
            return None
        declarations = []
        for node in _walk_scope(init):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
            for target in targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == receiver:
                    declarations.append((target.attr, node))
        matches = [(name, node) for name, node in declarations if _inside(node, start, end)]
        if len(matches) != 1:
            return None
        name, declaration = matches[0]
        if sum(other == name for other, _ in declarations) != 1:
            return None
        uses, functions = [], []
        limited = False
        for method in methods:
            receiver = _receiver(method)
            if not receiver:
                limited = True
                continue
            for node in _walk_scope(method):
                if not (isinstance(node, ast.Attribute) and node.attr == name
                        and isinstance(node.value, ast.Name) and node.value.id == receiver):
                    continue
                if _inside(declaration, node.lineno, node.end_lineno):
                    continue
                if isinstance(node.ctx, (ast.Store, ast.Del)):
                    # A later assignment can replace the captured parameter.
                    return dict(declaration=[_range(declaration)], uses=[],
                                declarationFunctions=[_range(init)], useFunctions=[],
                                limitation="Parameter reassignment makes its source uses ambiguous.")
                statement = node
                while not isinstance(statement, ast.stmt):
                    statement = self.parents[statement]
                use = _range(statement)
                if use not in uses:
                    uses.append(use)
                if _range(method) not in functions:
                    functions.append(_range(method))
                if len(uses) > MAX_REFERENCES:
                    return dict(declaration=[_range(declaration)], uses=[],
                                declarationFunctions=[_range(init)], useFunctions=[],
                                limitation="Too many parameter references; showing its declaration only.")
        return dict(declaration=[_range(declaration)], uses=sorted(uses),
                    declarationFunctions=[_range(init)], useFunctions=functions,
                    limitation="Direct references in this class only; aliases and nested scopes are not resolved."
                    if limited else "Direct references in this class only; runtime aliases are not resolved.")

    def focus(self, span, parameter):
        start, end = span["start"][0], span["end"][0]
        owner = self.enclosing(self.classes, start, end)
        method = self.enclosing(self.functions, start, end)
        result = {"class": [_range(owner)] if owner else [],
                  "function": [_range(method)] if method else [], "node": [[start, end]]}
        if parameter:
            result["parameter"] = self.parameter(owner, start, end) if owner else None
        return result


def enrich_focus_spans(spans, document, sources):
    """Attach context only to existing source navigation evidence, never labels."""
    parameter_ids = {node.node_id for node in document.semantic.nodes
                     if node.metadata.get("sourceKind") == "parameter"
                     or node.op.rsplit(".", 1)[-1].lower() in ("parameter", "buffer")}
    indexes, cache = {}, {}
    result = {}
    for node_id, original in spans.items():
        span = dict(original)
        file = span.get("file")
        if file in sources:
            if file not in indexes:
                try:
                    indexes[file] = _SourceIndex(sources[file])
                except (SyntaxError, ValueError, RecursionError):
                    indexes[file] = None
            index = indexes[file]
            if index is not None:
                key = (file, span["start"][0], span["end"][0], node_id in parameter_ids)
                if key not in cache:
                    cache[key] = index.focus(span, node_id in parameter_ids)
                span["focus"] = cache[key]
        result[node_id] = span
    return result
