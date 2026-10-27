"""Generated structural contracts must follow Python policy without executing models."""
import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from scripts.export_validation_contracts import openapi_document, schema_document
from services.validation import contracts


FIXTURES = Path(__file__).parent.resolve().parents[2] / "tests/fixtures/contracts"
CORPUS = json.loads((FIXTURES / "validation-contracts.json").read_text())


def test_checked_in_schema_matches_current_python_contracts():
    expected = json.dumps(schema_document(), indent=2, sort_keys=True) + "\n"
    assert (FIXTURES / "validation.schema.json").read_text() == expected


@pytest.mark.parametrize("fixture", CORPUS["structural"] + CORPUS["relational"],
                         ids=lambda fixture: fixture["name"])
def test_shared_contract_corpus_matches_python_policy(fixture):
    adapter = TypeAdapter(getattr(contracts, fixture["schema"]))
    payload = json.dumps(fixture["value"])
    if fixture["valid"]:
        adapter.validate_json(payload)
    else:
        with pytest.raises(ValidationError):
            adapter.validate_json(payload)


@pytest.mark.parametrize("model,field", [("Binding", "kwargs"), ("CallContext", "autocast")])
def test_openapi_maps_keep_value_types_without_weakening_validation_schema(model, field):
    schema = schema_document()["definitions"][model]["properties"][field]
    exported = openapi_document()["components"]["schemas"][model]["properties"][field]
    # The v6 generator needs additionalProperties for typed maps. Runtime schema
    # still constrains keys; this compatibility transform must not mutate it.
    assert "additionalProperties" not in schema
    assert len(schema["patternProperties"]) == 1
    pattern, value = next(iter(schema["patternProperties"].items()))
    expected = json.loads(json.dumps(value).replace("#/definitions/", "#/components/schemas/"))
    assert exported["patternProperties"] == {pattern: expected}
    assert exported["additionalProperties"] == expected
    assert schema_document()["definitions"][model]["properties"][field] == schema
