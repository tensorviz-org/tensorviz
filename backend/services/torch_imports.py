"""Source-proven Torch names, shared by projection and source-edit authority."""
import ast
from collections import Counter


def dotted_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return prefix + "." + node.attr if prefix else None
    return None


def torch_imports(tree, targets=None):
    """Conservative import proof, never model imports or name evaluation.

    Only an unconditional Torch import with no source-visible rebinding is
    usable. Shadowing anywhere in this file refuses the root name, even in an
    unrelated scope. Dynamic namespace access cannot prove import-name safety.
    """
    candidates = {}
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                if alias.name in ("torch", "torch.nn", "torch.nn.functional"):
                    name = alias.asname or alias.name.split(".")[0]
                    candidates[name] = (statement, alias.name if alias.asname else "torch")
        elif isinstance(statement, ast.ImportFrom) and statement.level == 0:
            for alias in statement.names:
                target = (statement.module or "") + "." + alias.name
                if target in (targets or {"torch.nn", "torch.nn.functional"}) or statement.module == "torch.nn" and alias.name != "*":
                    candidates[alias.asname or alias.name] = (statement, target)
    invalid = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Name, ast.Attribute)) and isinstance(node.ctx, (ast.Store, ast.Del)):
            spelling = dotted_name(node)
            if spelling:
                invalid.add(spelling.split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            invalid.add(node.name)
        elif isinstance(node, ast.arg):
            invalid.add(node.arg)
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name:
            invalid.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            invalid.add(node.rest)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                name = alias.asname or (alias.name.split(".")[0] if isinstance(node, ast.Import) else alias.name)
                if name == "*":
                    invalid.update(candidates)
                # Importing a Torch submodule also binds the same top-level
                # package; DINOv2 imports torch.utils.checkpoint after torch.
                same_package = (isinstance(node, ast.Import) and node in tree.body
                                and not alias.asname and name == 'torch'
                                and (alias.name == 'torch' or alias.name.startswith('torch.'))
                                and name in candidates and candidates[name][1] == 'torch')
                if name in candidates and candidates[name][0] is not node and not same_package:
                    invalid.add(name)
            # Multiple aliases binding one root in the same import are ambiguous.
            names = [a.asname or (a.name.split(".")[0] if isinstance(node, ast.Import) else a.name) for a in node.names]
            invalid.update(name for name, count in Counter(names).items() if count > 1)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("setattr", "delattr") and node.args:
            spelling = dotted_name(node.args[0])
            if spelling:
                invalid.add(spelling.split(".")[0])
        if ((isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in ("exec", "eval", "globals", "locals", "vars"))
                or isinstance(node, ast.Attribute) and node.attr == "__dict__"):
            invalid.update(candidates)
    return {name: target for name, (_, target) in candidates.items() if name not in invalid}

def canonical_name(spelling, aliases):
    root, dot, tail = spelling.partition(".")
    return aliases.get(root, root) + (dot + tail if dot else "")
