"""Export validation types/schema without loading an app or workspace code."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.validation.contracts import ValidationContracts


def schema_document(contract=ValidationContracts):
    schema = contract.model_json_schema(ref_template="#/definitions/{model}")
    schema["definitions"] = schema.pop("$defs")
    schema["$schema"] = "http://json-schema.org/draft-07/schema#"
    return schema


def openapi_document(contract=ValidationContracts):
    schema = contract.model_json_schema(ref_template="#/components/schemas/{model}")
    definitions = schema.pop("$defs")
    definitions[contract.__name__] = schema
    # openapi-typescript 6 does not understand JSON Schema patternProperties.
    # Preserve the value type for consumers rendering kwargs/autocast maps; the
    # separately exported validation schema still enforces the key pattern.
    def map_values(value):
        if isinstance(value, dict):
            patterns = value.get("patternProperties", {})
            if len(patterns) == 1:
                value["additionalProperties"] = next(iter(patterns.values()))
            for child in value.values():
                map_values(child)
        elif isinstance(value, list):
            for child in value:
                map_values(child)
    map_values(definitions)
    return {"openapi": "3.1.0", "info": {"title": contract.__name__, "version": "1"},
            "paths": {}, "components": {"schemas": definitions}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--openapi", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--contract", choices=["validation", "semantic"], default="validation")
    args = parser.parse_args()
    contract = ValidationContracts
    if args.contract == "semantic":
        from services.semantic.contracts import SemanticContracts
        contract = SemanticContracts
    for path, document in ((args.schema, schema_document(contract)), (args.openapi, openapi_document(contract))):
        content = json.dumps(document, indent=2, sort_keys=True) + "\n"
        if args.check:
            if not path.exists() or path.read_text() != content:
                raise SystemExit(f"Outdated {args.contract} contract: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)


if __name__ == "__main__":
    main()
