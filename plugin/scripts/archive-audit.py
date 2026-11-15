"""Hash release archives in one bounded pass, without extracting files."""
import gzip
import hashlib
import json
from pathlib import PurePosixPath
import re
import stat
import sys
import tarfile
import zipfile


def audit_archive(filename, kind):
    files, manifest = {}, None
    limit = (256 if kind == "runtime" else 128) * 1024 * 1024

    def consume(name, size, stream):
        nonlocal manifest
        if (not re.fullmatch(r"[A-Za-z0-9_./-]+", name) or name.startswith("/")
                or ".." in name.split("/") or str(PurePosixPath(name)) != name):
            raise ValueError("Unsafe archive path: " + name)
        if name in files or len(files) >= 10000:
            raise ValueError("Duplicate or excessive archive entries: " + name)
        if kind == "runtime" and re.search(r"\.(?:py|pyc|pyo|cjs|ts|tsx|map|c|h|o)$", name):
            raise ValueError("Source file in runtime archive: " + name)
        if size < 0 or size > limit:
            raise ValueError("Archive entry exceeds its byte budget: " + name)
        digest, total, metadata = hashlib.sha256(), 0, bytearray()
        while chunk := stream.read(1024 * 1024):
            total += len(chunk)
            if total > size:
                raise ValueError("Archive entry size mismatch: " + name)
            digest.update(chunk)
            if kind == "runtime" and name == "release-manifest.json":
                if total > 1024 * 1024:
                    raise ValueError("Runtime manifest exceeds its byte budget")
                metadata.extend(chunk)
        if total != size:
            raise ValueError("Truncated archive entry: " + name)
        files[name] = digest.hexdigest()
        if metadata:
            manifest = json.loads(metadata)

    if kind == "runtime":
        with gzip.open(filename, "rb") as compressed:
            with tarfile.open(fileobj=compressed, mode="r|") as archive:
                for entry in archive:
                    if not entry.isfile():
                        raise ValueError("Runtime archive contains a link or special file: " + entry.name)
                    with archive.extractfile(entry) as stream:
                        consume(entry.name, entry.size, stream)
            # Read through the gzip trailer too, so truncation/CRC errors fail closed.
            while compressed.read(1024 * 1024):
                pass
        if manifest is None:
            raise ValueError("Runtime archive has no release manifest")
    elif kind == "zip":
        with zipfile.ZipFile(filename) as archive:
            for entry in archive.infolist():
                mode = entry.external_attr >> 16
                if entry.is_dir() or stat.S_IFMT(mode) not in (0, stat.S_IFREG):
                    raise ValueError("ZIP contains a link or special file: " + entry.filename)
                with archive.open(entry) as stream:
                    consume(entry.filename, entry.file_size, stream)
    else:
        raise ValueError("Unsupported archive kind")
    return {"files": files, **({"manifest": manifest} if kind == "runtime" else {})}


if __name__ == "__main__":
    try:
        if len(sys.argv) != 3:
            raise ValueError("Usage: archive-audit.py runtime|zip ARCHIVE")
        print(json.dumps(audit_archive(sys.argv[2], sys.argv[1])))
    except (OSError, ValueError, EOFError, tarfile.TarError, zipfile.BadZipFile) as error:
        sys.exit("Archive audit failed: " + str(error))
