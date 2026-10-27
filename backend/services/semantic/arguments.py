"""Bounded observation trees; never evaluate expressions or arbitrary object repr."""
from __future__ import annotations

from collections import OrderedDict
import math
import re


class ArgumentEncoder:
    def __init__(self, torch, tensor, omit):
        self.torch, self.tensor, self.omit = torch, tensor, omit
        self.nodes = 0
        from torch._subclasses.fake_tensor import FakeTensor
        self.fake_tensor = FakeTensor

    def opaque(self, value, reason="unsupported_type"):
        self.omit(reason)
        cls = type(value)
        if cls is self.fake_tensor:  # name the type the same program has with real tensors
            cls = self.torch.nn.Parameter if isinstance(value, self.torch.nn.Parameter) else self.torch.Tensor
        return dict(kind="opaque", reason=reason,
                    typeName=f"{cls.__module__}.{cls.__qualname__}"[:128])

    def literal(self, value):
        kind = type(value)
        if value is None:
            return dict(kind="none")
        if kind is bool:
            return dict(kind="bool", value=value)
        if kind is int:
            # Avoid unbounded decimal conversion (and Python's digit guard).
            if value.bit_length() > 13600:
                return self.opaque(value, "item_limit")
            return dict(kind="int", value=str(value))
        if kind is float:
            return dict(kind="float", value="-0" if value == 0 and math.copysign(1, value) < 0 else value if math.isfinite(value) else
                        "nan" if math.isnan(value) else "inf" if value > 0 else "-inf")
        if kind is str:
            return dict(kind="str", value=value) if len(value) <= 4096 else self.opaque(value, "item_limit")
        torch_types = {self.torch.dtype: "dtype", self.torch.device: "device",
                       self.torch.layout: "layout", self.torch.memory_format: "memory_format"}
        if kind in torch_types:
            return dict(kind=torch_types[kind], value=str(value))
        symbols = {self.torch.SymInt: "int", self.torch.SymFloat: "float", self.torch.SymBool: "bool"}
        if kind in symbols:
            expression = str(value)
            return (dict(kind="symbolic", type=symbols[kind], expression=expression)
                    if len(expression) <= 4096 else self.opaque(value, "item_limit"))
        return None

    def encode(self, value, path=(), depth=0):
        self.nodes += 1
        if depth >= 20:
            return self.opaque(value, "depth_limit")
        if isinstance(value, self.torch.Tensor):
            return self.tensor(value, list(path))
        literal = self.literal(value)
        if literal is not None:
            return literal
        mapping = isinstance(value, dict)
        if type(value) in (tuple, list) or mapping:
            # Leave room for each direct child; nested containers spend the same
            # budget. A rejected container is one explicit omitted subtree.
            count = dict.__len__(value) if mapping else len(value)
            if count > 4096 or self.nodes + count > 12000:
                return self.opaque(value, "item_limit")
            self.nodes += count
            if mapping:
                entries = []
                # Observe stored entries, never overridden mapping methods or
                # arbitrary object attributes. OrderedDict retains its own order.
                entries_view = OrderedDict.items(value) if isinstance(value, OrderedDict) else dict.items(value)
                for index, (key, item) in enumerate(entries_view):
                    if type(key) not in (type(None), bool, int, str):
                        return self.opaque(value)
                    encoded_key = self.literal(key)
                    if encoded_key["kind"] == "opaque":
                        return self.opaque(value, encoded_key["reason"])
                    entries.append(dict(key=encoded_key, value=self.encode(item,
                        (*path, "entries", index, "value"), depth + 1)))
                return dict(kind="dict", entries=entries)
            return dict(kind="tuple" if type(value) is tuple else "list",
                        items=[self.encode(item, (*path, "items", index), depth + 1)
                               for index, item in enumerate(value)])
        return self.opaque(value)

    def arguments(self, args, kwargs):
        self.nodes = len(args[:256]) + min(len(kwargs), 256)
        if len(args) > 256 or len(kwargs) > 256:
            self.omit("item_limit")
        positional = [self.encode(value, ("args", index)) for index, value in enumerate(args[:256])]
        keywords = {}
        for index, (key, value) in enumerate(kwargs.items()):
            if index >= 256:
                break
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", key):
                self.omit("unsupported_type")
                continue
            keywords[key] = self.encode(value, ("kwargs", key))
        return dict(args=positional, kwargs=keywords)

    def output(self, value):
        self.nodes = 0
        return self.encode(value)
