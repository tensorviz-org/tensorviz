"""Small stdlib compatibility helpers for Python 3.10 capture workers."""
from contextlib import contextmanager
import hashlib
import os


@contextmanager
def chdir(directory):
    previous = os.getcwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)


def sha256_stream(stream):
    value = hashlib.sha256()
    for chunk in iter(lambda: stream.read(65536), b""):
        value.update(chunk)
    return value.hexdigest()
