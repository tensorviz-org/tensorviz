# Compact model context

MCP clients can call `get_model_context` with a current `modelId` and `revision`
from `list_models`. Native and ACP Canvas chat attach the same facts automatically
after a matching capture. Source-only and unsupported capture states explain which
facts are unavailable. Reading the tool never starts a capture or executes a model.

The packet contains module paths and aliases, complete retained per-module call
lists, unique recursive registered parameter element counts, ordered tensor leaves
and shapes, per-call modes, provenance, unknowns and capture omissions. Counts
include frozen registrations and must not be summed across overlapping modules.
These facts describe the recorded path; they do not establish numerical or training
correctness, or verify that current inputs, weights and external dependencies match.

The default limit is 4,000 UTF-8 bytes. Use an exact module id, path, qualified type
or invocation id as `focus`; `maxBytes` accepts 2,000–100,000. Omitted module/call
counts stay explicit. Use focused calls or `inspect_capture` for omitted details.
Source changes, an evicted capture or a newer unsuccessful attempt cannot borrow
older successful measurements. MCP awaits the host’s existing freshness guard,
which may recheck source/setup metadata.

Canvas excludes live facts from draft and historical selections. With available
compact facts, its general graph preload has an 8 KiB ceiling instead of 64 KiB;
continuation still retrieves omitted records. Human source review and explicit
execution controls remain in place. Payload limits do not establish a measured
agent speedup or token saving. See [source editing](source-editing.md).
