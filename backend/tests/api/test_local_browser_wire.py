import base64
import gzip
import json
from pathlib import Path

import pytest

from services.ide.local_browser_wire import (
    ENCODING, MAX_DECODED_BYTES, MAX_WIRE_BYTES, decode_message, encode_message,
)


def test_shared_cross_language_wire_fixture():
    fixture = json.loads((Path(__file__).parent.resolve().parents[2] /
        "tests/fixtures/contracts/local-browser-wire.json").read_text())
    assert decode_message(fixture["wire"]) == fixture["message"]


def test_large_message_is_lossless_and_small_messages_keep_plain_json():
    small = {"type": "ack", "label": "入力 👋"}
    assert json.loads(encode_message(small)) == small
    large = {"document": [small] * 1_100_000}
    assert len(json.dumps(large).encode()) > 32 * 1024 * 1024
    encoded = encode_message(large)
    assert len(encoded.encode()) < MAX_WIRE_BYTES
    assert json.loads(encoded)["encoding"] == ENCODING
    assert decode_message(encoded) == large


@pytest.mark.parametrize("text", [
    '{"encoding":"unknown","data":""}',
    '{"encoding":"tensorviz-gzip-json-v1","data":"!"}',
    '{"encoding":"tensorviz-gzip-json-v1","data":"eA=="}',
    '{"encoding":"tensorviz-gzip-json-v1","data":"","extra":1}',
    '"' + '界' * (MAX_WIRE_BYTES // 3 + 1) + '"',
])
def test_malformed_and_oversized_wire_messages_refuse(text):
    with pytest.raises(ValueError):
        decode_message(text)


def test_inflation_is_bounded_even_when_frame_is_small():
    encoded = json.dumps(dict(encoding=ENCODING, data=base64.b64encode(
        gzip.compress(b'"' + b'x' * MAX_DECODED_BYTES + b'"')).decode()))
    assert len(encoded) < MAX_WIRE_BYTES
    with pytest.raises(ValueError, match="oversized"):
        decode_message(encoded)
    with pytest.raises(ValueError, match="decoded message budget"):
        encode_message("x" * MAX_DECODED_BYTES)


def test_truncated_compression_cannot_become_a_partial_message():
    raw = gzip.compress(b'{"type":"ack"}')
    with pytest.raises(ValueError):
        decode_message(json.dumps(dict(encoding=ENCODING, data=base64.b64encode(raw[:-4]).decode())))
