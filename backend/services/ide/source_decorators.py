"""Recognize source-preserving library decorators without importing user code."""
from __future__ import annotations

import ast


def imported_symbol(tree, expression, package=""):
    """Resolve one untouched module import, including aliases and relative imports."""
    parts = ast.unparse(expression).split(".")
    if not all(part.isidentifier() for part in parts):
        return None
    root, candidates = parts[0], []

    def bindings(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node.name
            return
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield from (alias.asname or alias.name.split(".")[0] for alias in node.names)
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            yield node.id
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
            yield ast.unparse(node).split(".")[0]
        for child in ast.iter_child_nodes(node):
            yield from bindings(child)

    for statement in tree.body:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                if (alias.asname or alias.name.split(".")[0]) == root:
                    candidates.append(alias.name if alias.asname else root)
        elif isinstance(statement, ast.ImportFrom):
            if any(alias.name == "*" for alias in statement.names):
                return None
            module = statement.module or ""
            if statement.level:
                components = package.split(".") if package else []
                if statement.level > len(components):
                    continue
                module = ".".join(components[:len(components) - statement.level + 1] + ([module] if module else []))
            for alias in statement.names:
                if (alias.asname or alias.name) == root:
                    candidates.append(module + "." + alias.name)
        elif root in bindings(statement):
            return None
    return ".".join([candidates[0], *parts[1:]]) if len(candidates) == 1 else None


def known_decorators(tree, definition, package="", *, dataclass=False, constructor=False):
    allowed = {"transformers.utils.auto_docstring", "transformers.utils.auto_docstring.auto_docstring"}
    if dataclass:
        allowed.update({"dataclasses.dataclass", "huggingface_hub.dataclasses.strict"})
    if isinstance(definition, ast.FunctionDef):
        allowed.update({"transformers.utils.can_return_tuple", "transformers.utils.generic.can_return_tuple",
                        "transformers.utils.merge_with_config_defaults", "transformers.utils.generic.merge_with_config_defaults",
                        "transformers.utils.output_capturing.capture_outputs",
                        "torchvision.models._api.register_model",
                        "torchvision.models._utils.handle_legacy_interface"})
    if constructor:
        allowed.add("diffusers.configuration_utils.register_to_config")
    for decorator in definition.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        symbol = imported_symbol(tree, target, package)
        if symbol not in allowed:
            return False
        if isinstance(decorator, ast.Call) and symbol.endswith(".register_to_config"):
            return False
    return True
