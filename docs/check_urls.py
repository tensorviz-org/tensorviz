#!/usr/bin/env python3
"""Read-only, unauthenticated TensorViz documentation/install URL smoke check.

No install commands, browser sessions, credentials, or workflows are used.
Only public URL/status/hash/structured content evidence is written. Redirect
query strings, headers, cookies, full HTML, and exception text are not saved.
This tests delivery and static content, not rendered UI or installation.
"""

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import platform
from html.parser import HTMLParser
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request
import ssl

UA = "TensorViz-Mac-docs-check/1.0"
MAX_BYTES = 3 * 1024 * 1024
SEEDS = [
    "https://docs.tensorviz.dev/",
    "https://docs.tensorviz.dev/install.html",
    "https://docs.tensorviz.dev/quickstart.html",
    "https://docs.tensorviz.dev/agent-plugin.html",
    "https://docs.tensorviz.dev/local-setup.html",
    "https://docs.tensorviz.dev/plugin-build.html",
    "https://docs.tensorviz.dev/graph-diff-review/",
    "https://docs.tensorviz.dev/share-to-website.html",
    "https://tensorviz.dev/",
    "https://open-vsx.org/api/tensorviz/tensorviz-vscode/latest",
    "https://marketplace.visualstudio.com/items?itemName=tensorviz.tensorviz-vscode",
    "https://github.com/tensorviz-org/tensorviz-plugins",
    "https://github.com/tensorviz-org/tensorviz-plugins/releases/download/v0.1.44/release.json",
    "https://github.com/tensorviz-org/tensorviz-plugins/releases/download/v0.1.44/tensorviz-runtime-0.1.44-darwin-arm64.tar.gz.json",
]


def public_url(url):
    p = urllib.parse.urlsplit(url)
    query = p.query if p.netloc == "marketplace.visualstudio.com" and p.path == "/items" else ""
    return urllib.parse.urlunsplit((p.scheme, p.netloc, p.path, query, ""))


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.images = []
        self.ids = set()
        self.text = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if a.get("id"):
            self.ids.add(a["id"])
        if tag == "a" and a.get("href"):
            self.links.append(a["href"])
        if tag == "img" and a.get("src"):
            self.images.append(a["src"])

    def handle_data(self, text):
        self.text.append(text)


def fetch(url):
    row = {"url": public_url(url)}
    body = b""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=25) as response:
            body = response.read(MAX_BYTES + 1)
            row.update(status=response.status, final_url=public_url(response.url),
                       content_type=response.headers.get("Content-Type", "").split(";")[0],
                       bytes_read=len(body), truncated=len(body) > MAX_BYTES,
                       sha256=hashlib.sha256(body).hexdigest())
    except urllib.error.HTTPError as error:
        row.update(status=error.code, error="HTTPError")
    except Exception as error:
        row.update(status=None, error=type(error).__name__)
        reason = getattr(error, "reason", error)
        row["error_cause"] = type(reason).__name__
        if isinstance(reason, ssl.SSLCertVerificationError):
            row["tls_verify_code"] = reason.verify_code
            row["tls_verify_message"] = reason.verify_message
    return row, body


def batch(urls):
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        return dict(zip(urls, pool.map(fetch, urls)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("live-url-check.json"))
    args = parser.parse_args()
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    results = batch(SEEDS)
    parsed = {}
    docs_links = set()
    image_urls = set()
    link_origins = {}
    for url, (row, body) in results.items():
        if row.get("status") != 200 or row.get("content_type") != "text/html":
            continue
        page = Page()
        page.feed(body.decode("utf-8", errors="replace"))
        parsed[url] = page
        if urllib.parse.urlsplit(url).netloc not in {"docs.tensorviz.dev", "tensorviz.dev"}:
            continue
        for href in page.links:
            dest = urllib.parse.urljoin(url, href)
            split = urllib.parse.urlsplit(dest)
            if split.scheme == "https" and split.netloc == "docs.tensorviz.dev":
                bare = public_url(dest)
                docs_links.add(bare)
                link_origins.setdefault(dest, []).append(url)
        for src in page.images:
            dest = urllib.parse.urljoin(url, src)
            if urllib.parse.urlsplit(dest).netloc == "docs.tensorviz.dev":
                image_urls.add(public_url(dest))
    extra = sorted((docs_links | image_urls) - set(results))
    results.update(batch(extra))
    for url, (row, body) in results.items():
        if url not in parsed and row.get("status") == 200 and row.get("content_type") == "text/html":
            page = Page()
            page.feed(body.decode("utf-8", errors="replace"))
            parsed[url] = page
    missing_anchors = []
    for link, origins in sorted(link_origins.items()):
        split = urllib.parse.urlsplit(link)
        if split.fragment and public_url(link) in parsed:
            fragment = urllib.parse.unquote(split.fragment)
            if fragment not in parsed[public_url(link)].ids:
                missing_anchors.append({"url": link, "linked_from": sorted(set(origins))})
    snippets = {}
    phrases = ["0.1.45", "0.1.44", "glibc", "Linux x64", "Apple Silicon", "codex plugin", "claude plugin", "code --install-extension", "Source repository"]
    for url in SEEDS:
        if url in parsed and urllib.parse.urlsplit(url).netloc in {"docs.tensorviz.dev", "tensorviz.dev"}:
            text = " ".join(parsed[url].text)
            snippets[url] = {p: p in text for p in phrases}
    metadata = {}
    for url in SEEDS:
        row, body = results[url]
        if row.get("status") == 200 and (url.endswith(".json") or url.startswith("https://open-vsx.org/api/")):
            try:
                data = json.loads(body)
                keys = ("version", "namespace", "name", "timestamp", "sourceRevision", "formatVersion", "platform", "pythonABI", "sha256", "bytes", "url", "files")
                metadata[url] = {k: data[k] for k in keys if k in data}
                if isinstance(metadata[url].get("files"), dict):
                    metadata[url]["files"] = {k: public_url(v) if isinstance(v, str) and v.startswith("https://") else v for k, v in metadata[url]["files"].items()}
            except (ValueError, TypeError):
                metadata[url] = {"parse_error": True}
    report = {
        "checked_at_utc": started,
        "python_version": platform.python_version(),
        "openssl_version": ssl.OPENSSL_VERSION,
        "method": "Unauthenticated HTTPS GET; five workers; 25-second request timeout; 3 MiB response cap. No install or UI execution.",
        "scope": "Seed installation/docs pages, their directly linked documentation pages, and documentation image assets in those seed pages.",
        "results": [results[u][0] for u in sorted(results)],
        "missing_linked_anchor_ids": missing_anchors,
        "documentation_image_urls": sorted(image_urls),
        "static_text_presence": snippets,
        "public_metadata": metadata,
        "limitations": ["Delivery and static text checks do not prove rendered usability or installed behavior.", "Version metadata is a point-in-time observation.", "Unlinked or deeper pages/assets are outside this bounded check.", "Redirect query strings, response headers, cookies, raw HTML, local paths, and exception messages are omitted."],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    bad = [row for row in report["results"] if row.get("status") != 200]
    print(json.dumps({"checked": len(results), "http_200": len(results)-len(bad), "non_200": bad, "missing_anchor_count": len(missing_anchors), "documentation_images": len(image_urls)}, indent=2))


if __name__ == "__main__":
    main()
