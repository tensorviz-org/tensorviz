"""Bounded declaration evidence, independent of statically inferred model topology."""
import ast
from collections import Counter, defaultdict
from copy import deepcopy
from services.torch_imports import torch_imports, dotted_name, canonical_name
from services.ide.projection import utf16_columns
import hashlib
import json

MAX_DECLARATIONS = 4096
MAX_BYTES = 1024 * 1024


def review_declarations(sources):
    entries, complete = {}, True
    for file, source in sources.items():
        try:
            tree = ast.parse(source)
        except (SyntaxError, RecursionError):
            complete = False
            continue
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        names = Counter(node.name for node in classes)
        column = utf16_columns(source)
        aliases = torch_imports(tree)

        def span(node):
            # UTF-16 columns, as static node spans (which review joins by exact
            # location) and capture anchors (which constructors join) count them.
            return dict(file=file, start=[node.lineno, column(node.lineno, node.col_offset)],
                end=[node.end_lineno, column(node.end_lineno, node.end_col_offset)], kind='decl')

        for cls in classes:
            if names[cls.name] != 1:
                complete = False
                continue
            initializers = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '__init__']
            if len(initializers) != 1:
                continue
            writes = defaultdict(list)

            def collect(statements, guards=()):
                for statement in statements:
                    targets = statement.targets if isinstance(statement, ast.Assign) else (
                        [statement.target] if isinstance(statement, ast.AnnAssign) and statement.value is not None else [])
                    for target in targets:
                        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                            writes[target.attr].append((statement, guards, len(targets) != 1))
                    if isinstance(statement, (ast.If, ast.While, ast.For)):
                        condition = ast.dump(statement.iter if isinstance(statement, ast.For) else statement.test, include_attributes=False)
                        collect(statement.body, guards + ((type(statement).__name__, condition, 'body'),))
                        collect(statement.orelse, guards + ((type(statement).__name__, condition, 'else'),))
                    # A complete inventory must include handler, context and case bodies too.
                    elif isinstance(statement, ast.With):
                        condition = repr([ast.dump(item, include_attributes=False) for item in statement.items])
                        collect(statement.body, guards + (('With', condition, 'body'),))
                    elif isinstance(statement, (ast.Try, ast.TryStar)):
                        kind = type(statement).__name__
                        collect(statement.body, guards + ((kind, '', 'body'),))
                        for handler in statement.handlers:
                            condition = ast.dump(handler.type, include_attributes=False) if handler.type else ''
                            collect(handler.body, guards + ((kind, condition, 'except'),))
                        collect(statement.orelse, guards + ((kind, '', 'else'),))
                        collect(statement.finalbody, guards + ((kind, '', 'finally'),))
                    elif isinstance(statement, ast.Match):
                        subject = ast.dump(statement.subject, include_attributes=False)
                        for case in statement.cases:
                            condition = ast.dump(ast.match_case(pattern=case.pattern, guard=case.guard, body=[]), include_attributes=False)
                            collect(case.body, guards + (('Match', subject, condition),))

            collect(initializers[0].body)
            for name, declarations in writes.items():
                if not any(isinstance(statement.value, ast.Call) for statement, _, _ in declarations):
                    continue
                statement, _, chained = declarations[0]
                # Spelled as JSON.stringify spells it: renamed files are remapped by this prefix.
                key = '@declaration:' + json.dumps([file, cls.name, name], separators=(',', ':'), ensure_ascii=False)
                facts = [(ast.dump(stmt.value, include_attributes=False), guards) for stmt, guards, _ in declarations]
                entries[key] = dict(label=f'{cls.name}.{name}', digest=hashlib.sha256(repr(facts).encode()).hexdigest(),
                    ambiguous=len(declarations) != 1 or chained, span=span(statement),
                    constructor=span(statement.value) if isinstance(statement.value, ast.Call) else None)
                def children(value, parent_key, label):
                    # Only literal, source-proven torch containers. Comprehensions,
                    # helper factories and starred arguments do not define slots.
                    spelling = dotted_name(value.func) if isinstance(value, ast.Call) else None
                    if not spelling or spelling.split('.')[0] not in aliases or canonical_name(spelling, aliases) != 'torch.nn.Sequential' or value.keywords:
                        return
                    if not all(isinstance(child, ast.Call) for child in value.args):
                        return
                    entries[parent_key]['children'] = []
                    for index, child in enumerate(value.args):
                        child_key = parent_key + f'[{index}]'
                        entries[parent_key]['children'].append(child_key)
                        class Mask(ast.NodeTransformer):
                            def visit_Call(self, node):
                                if (node.lineno, node.col_offset) == (child.lineno, child.col_offset):
                                    return ast.Name(id='__tensorviz_review_slot__', ctx=ast.Load())
                                return self.generic_visit(node)
                        context = ast.dump(Mask().visit(deepcopy(statement.value)), include_attributes=False)
                        guards = declarations[0][1]
                        entries[child_key] = dict(label=f'{label}[{index}]', parent=parent_key,
                            digest=hashlib.sha256(repr((ast.dump(child, include_attributes=False), guards)).encode()).hexdigest(),
                            contextDigest=hashlib.sha256(repr((context, guards)).encode()).hexdigest(),
                            ambiguous=entries[parent_key]['ambiguous'], span=span(child), constructor=span(child))
                        children(child, child_key, f'{label}[{index}]')
                if isinstance(statement.value, ast.Call):
                    children(statement.value, key, f'{cls.name}.{name}')
                if len(entries) > MAX_DECLARATIONS:
                    return dict(entries={}, complete=False)
    if len(json.dumps(entries).encode()) > MAX_BYTES:
        return dict(entries={}, complete=False)
    return dict(entries=entries, complete=complete)


def captured_declaration_calls(capture, evidence, resolver):
    """Join exact recorded constructors, never registration indices or class labels."""
    if resolver is None:
        return {}
    def location(span):
        return span['file'], tuple(span['start']), tuple(span['end'])
    by_location = {location(fact['constructor']): key for key, fact in evidence['entries'].items()
                   if fact['constructor'] and not fact['ambiguous']}
    calls, result = defaultdict(list), defaultdict(list)
    for call in capture.invocations:
        calls[call.moduleId].append(call.id)
    for module in capture.modules:
        if not module.constructorClassVerified or len(module.constructorBindingIds) != 1 or len(module.paths) != 1:
            continue
        span = resolver.navigation(module.constructorBindingIds)
        key = by_location.get(location(span)) if span else None
        if key:
            result[key].extend(calls[module.id])
    return dict(result)
