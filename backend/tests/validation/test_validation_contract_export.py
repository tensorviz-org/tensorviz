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


@pytest.mark.parametrize('dtype', ['int64', 'float32'])
def test_tensor_grad_default_is_false_in_runtime_and_exported_schema(dtype):
    tensor = contracts.TensorValue.model_validate(dict(kind='tensor', shape=[2], dtype=dtype,
        generator=dict(kind='full', value=1)))
    assert tensor.requires_grad is False
    schema = schema_document()['definitions']['TensorValue']
    assert 'requires_grad' not in schema['required']
    assert schema['properties']['requires_grad']['default'] is False


def test_explicit_floating_tensor_gradients_remain_supported():
    tensor = contracts.TensorValue.model_validate(dict(kind='tensor', shape=[2], dtype='float32',
        requires_grad=True, generator=dict(kind='full', value=1)))
    assert tensor.requires_grad is True


@pytest.mark.parametrize('requires_grad', [True, 0, 'false', None])
def test_integer_tensors_still_reject_gradients_and_non_boolean_values(requires_grad):
    with pytest.raises(ValidationError):
        contracts.TensorValue.model_validate(dict(kind='tensor', shape=[2], dtype='int64',
            requires_grad=requires_grad, generator=dict(kind='full', value=1)))


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
