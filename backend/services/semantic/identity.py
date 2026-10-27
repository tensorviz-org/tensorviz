"""Weak runtime identity shared by capture adapters; never retain tensor payloads."""
from __future__ import annotations

import weakref


def weak_get(table: dict, value):
    entry = table.get(id(value))
    return entry[1] if entry is not None and entry[0]() is value else None


def weak_set(table: dict, value, metadata) -> None:
    key = id(value)

    def released(reference):
        # A stale callback must not remove a newer occupant of a reused address.
        entry = table.get(key)
        if entry is not None and entry[0] is reference:
            del table[key]

    table[key] = (weakref.ref(value, released), metadata)


def weak_identity(table: dict, counters: dict, value, prefix: str) -> str:
    identity = weak_get(table, value)
    if identity is None:
        sequence = counters.get(prefix, 0)
        counters[prefix] = sequence + 1
        identity = f"{prefix}{sequence}"
        weak_set(table, value, identity)
    return identity
