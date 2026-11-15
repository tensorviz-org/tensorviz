import base64
import gzip
import json
import os
from pathlib import Path

import pytest

from services.ide.local_browser_protocol import (
    LOCAL_BROWSER_PUBLISH_FILE_MAX_LENGTH, LOCAL_BROWSER_PUBLISH_MAX_PACKED_BYTES,
)
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


def test_publish_largest_graph_file_fits_one_frame_though_gzip_cannot_shrink_it():
    file = base64.b64encode(os.urandom(LOCAL_BROWSER_PUBLISH_MAX_PACKED_BYTES)).decode("ascii")
    assert len(file) <= LOCAL_BROWSER_PUBLISH_FILE_MAX_LENGTH
    request = {"type": "graphPublishRequest", "protocolVersion": 1, "workspaceSessionId": "s" * 36,
               "messageId": "m" * 36, "requestId": "r" * 36,
               "target": {"workspaceFolder": "/ws", "file": "net.py"}, "file": file}
    encoded = encode_message(request)
    assert len(encoded.encode()) <= MAX_WIRE_BYTES
    assert decode_message(encoded) == request


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
