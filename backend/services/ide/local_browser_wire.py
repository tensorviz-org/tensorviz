"""Lossless, bounded encoding for the authenticated loopback message channel."""
import base64
import gzip
import json
import zlib


#: Fits Publish's graph file, gzipped in base64 and up to about 31 MiB on the wire.
MAX_WIRE_BYTES = 32 * 1024 * 1024
MAX_DECODED_BYTES = 96 * 1024 * 1024
COMPRESSION_THRESHOLD = 256 * 1024
ENCODING = "tensorviz-gzip-json-v1"


def encode_message(message):
    raw = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(raw) > MAX_DECODED_BYTES:
        raise ValueError("decoded message budget exceeded")
    if len(raw) >= COMPRESSION_THRESHOLD:
        raw = json.dumps(dict(encoding=ENCODING,
            data=base64.b64encode(gzip.compress(raw, compresslevel=1, mtime=0)).decode("ascii")),
            separators=(",", ":")).encode("ascii")
    if len(raw) > MAX_WIRE_BYTES:
        raise ValueError("wire message budget exceeded")
    return raw.decode("utf-8")


def decode_message(text):
    if len(text.encode("utf-8")) > MAX_WIRE_BYTES:
        raise ValueError("wire message budget exceeded")
    message = json.loads(text)
    if isinstance(message, dict) and "encoding" in message:
        if set(message) != {"encoding", "data"} or message["encoding"] != ENCODING or not isinstance(message["data"], str):
            raise ValueError("invalid wire encoding")
        compressed = base64.b64decode(message["data"], validate=True)
        inflater = zlib.decompressobj(wbits=31)
        try:
            raw = inflater.decompress(compressed, MAX_DECODED_BYTES + 1)
        except zlib.error as error:
            raise ValueError("invalid compressed message") from error
        if len(raw) > MAX_DECODED_BYTES or not inflater.eof or inflater.unused_data:
            raise ValueError("invalid or oversized compressed message")
        message = json.loads(raw.decode("utf-8"))
    return message
