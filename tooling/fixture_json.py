"""Keep small fixture records on one line without changing their JSON values."""
import json


def dumps(value, depth=0):
    compact = json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
    if len(compact) + depth * 2 <= 500 or not isinstance(value, (dict, list)):
        return compact
    if isinstance(value, dict):
        rows = [json.dumps(key) + ": " + dumps(item, depth + 1) for key, item in value.items()]
        start, end = "{", "}"
    else:
        rows = [dumps(item, depth + 1) for item in value]
        start, end = "[", "]"
    return start + "\n" + ",\n".join("  " * (depth + 1) + row for row in rows) + "\n" + "  " * depth + end
