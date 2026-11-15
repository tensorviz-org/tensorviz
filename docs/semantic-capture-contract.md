# Semantic capture contract v1

`backend/services/semantic/contracts.py` owns the strict semantic schema. It
imports neither Torch nor user models. The schema version is independent of graph
layout and portable snapshot envelope versions. Live canonical producers and consumers
use this schema; legacy captures retain compatibility paths with explicit limitations.

Generate/check the shared extension and browser TypeScript with
`npm run semantic:generate --prefix extension` and
`npm run semantic:check --prefix extension`. Set `PYTHON` to the project's backend
interpreter when needed. The same exporter generates validation contracts
without changing their schema. Generated JSON schema checks structural types;
Python additionally checks relationships, event ordering and resource budgets.

## Resolved execution plans

Browser saved profiles and explicit MCP execution use `ResolvedExecutionPlan`.
Isolated preflight records saved source/configuration/native-file digests, selected
interpreter and package facts, exact execution and optional diagnostic recipe,
timeout and semantic capture budgets. Factories, checkpoints and input fixtures
must have the correct source/native identities. Execution files and working
directories must remain inside the workspace after symlink resolution; aliases
to excluded `.env*` files or directories are refused. Normal aliases to permitted
workspace files and directories remain supported. Selected profiles must match the
saved profile, target and applicability; explicit recipes record configuration as
absent or overridden. Automatic input guesses use `CapturePreparation` instead.

`CapturePreparation` is the preceding automatic-input stage. It preserves the exact
saved source/environment, configuration status (`absent` or `no_match`), bounded
original request and limits without claiming a chosen input recipe. The static host
resolver rejects dirty source/configuration, malformed configuration and matching
model/script profiles that require selection. It shares the plan's source identity,
applicability hashing and saved-file/environment rechecks. Browser/MCP automatic
requests, including trace-on-save, retain canonical evidence against this preparation.
Fake-tensor and non-model legacy paths do not claim this prepared canonical format.

Python owns the canonical applicability fingerprint. It excludes per-attempt IDs,
source occurrence IDs and editor document versions, while retaining saved bytes,
target, environment, recipe and limits. This is an applicability key, not proof of
identical runtime state: actual inputs, weights and RNG state have their separate
existing measured identities. Source scope remains explicitly partial.

The host transports the plan in a private 16 MiB-bounded temporary file, with mode
0600 inside a 0700 directory on POSIX and a restricted ACL on Windows. It cleans up
on every exit path. The worker verifies saved bytes and environment before importing
workspace code; host and worker reconcile again afterward. Stale captures remain
available without validating current settings. The MCP setup query pages plan files.
Workers enforce independent canonical operation, invocation, value and byte budgets.
Canonical runs bypass the legacy operation/reuse stream and folded display graph.
A separate diagnostic sidecar retains bounded operation/module/output evidence
for numerical and profiling consumers. Library/fake callers and older captures retain their compatibility path;
they do not claim canonical semantics.

## Preserved observations

- Operations are generic target/schema invocations, never an opcode admission list.
  Ordered positional and keyword argument trees preserve repeated references,
  containers, literal types, symbolic expressions and opaque boundaries.
- Integer literals are canonical decimal **strings** so JavaScript cannot round
  Python integers. Float tags preserve the float/int distinction and explicitly
  encode `nan`, `inf`, `-inf` and negative zero (`-0`). JavaScript JSON retention
  would otherwise turn numeric negative zero into positive zero. Symbolic expressions are descriptions, never code
  to evaluate. Unknown objects expose only a type name and omission reason.
- Argument and output paths address the serialized JSON tree. For example, the
  second tensor in `cat([a, b])` is `['args', 0, 'items', 1]`. Dictionary entries
  preserve order and typed keys; a value path includes `entries`, the entry index,
  and `value`. Rendering can give these paths friendly labels without merging them.
- Tensor facts reuse validation's object/storage/version identities and add producer
  output/effect paths, strides, storage offsets and parameter/buffer references.
  `version` is PyTorch's observed counter (nullable for inference tensors), while
  `logicalVersion` numbers this object's successive retained observations. A
  schema-declared write advances a logical version even if dispatch does not advance
  PyTorch's counter. `previousVersion` links the preceding value. They contain no
  tensor payload. Shape equality never establishes aliasing.
- Each operation has explicit `effects` for writable operands and already observed
  shared-storage aliases. Effects cite an exact argument path, prior value and new
  value or opaque limitation. Repeated operands keep distinct paths. A producer's
  `kind` selects the operation's `output` tree or its `effects` array; effect paths
  such as `[0, 'after']` never invent a returned tensor for a `None` result.
  These are conservative schema-declared write boundaries, not measurements of
  which tensor elements changed. A failed writable kernel remains partial even
  when the post-failure metadata is available.
- Definitions, runtime module instances, and particular calls have separate IDs.
  One module can have multiple registered paths and multiple invocations. Entry/exit
  sequence numbers describe containment, independently of computational dataflow.
- Optional `Value.logicalBytes` records actual `numel() * element_size()`, including
  scalar, empty and strided/view values. It is not allocated or peak memory.
  `Invocation.parameterCount` records unique recursive registered parameter objects
  at call entry, excluding buffers and counting shared registration only once per
  invocation. Missing legacy facts stay unknown. Optional
  `parameterCountUnavailable` distinguishes unsupported metadata and exhaustion
  of the separate 100,000-item per-capture metadata-read allowance; neither supplies
  a partial count or changes operation coverage. Browser and MCP summaries use these
  same facts, with no dtype-to-byte inference or first-output selection.
- Source bindings cite an immutable snapshot and exact file digest plus source/token
  anchors. Exact, candidate and ambiguous correspondence remain distinct. A binding
  alone does not grant edit capability. Current captures require saved source;
  buffer overlays have not been implemented.

- Fake tensors. The recorder observes a `FakeTensor` (PyTorch's shape-only tensor)
  exactly like a real tensor: a `Value` keeps its identity, shape, dtype, device,
  stride, storage offset, `logicalBytes`, producer and state references, and the
  argument trees, operations, invocations and parameter counts are the same. A
  FakeTensor's `.device` is a dispatched `prim.device` call; it is not recorded, as
  the legacy fold already filtered it. An unsupported FakeTensor is opaque under the
  type the same program has with real tensors (`torch.Tensor` or
  `torch.nn.Parameter`). Values never exist, so a capture that observed a FakeTensor
  value or registered parameter/buffer sets `valuesUnavailable: "fake_tensor"`, and
  the fingerprints of fake tensors are null (`stateDigest`, and `inputDigest` for fake
  inputs). Fingerprinting refuses fake and meta tensors before reading any memory
  (`execution_setup.TensorValuesUnavailable`); `state_identity` reports
  `unavailable: "fake_tensor"` or `"meta_tensor"`. Real captures omit the field rather
  than writing `null`, so they serialize exactly as before. One known difference:
  creating `nn.Parameter(...)` inside a forward dispatches extra `aten.detach` calls
  on a FakeTensor. Such dynamic parameters already leave the capture partial.

## Coverage and limits

A computational connection must come from a recorded value reference at a specific
argument path. Producer outputs must identify that value and occur before its use.
No layout edges belong in this contract.

Coverage describes only the executed path. `observed`, `failed`, `not_observed`,
`capture_limited`, `static_only`, `unresolved`, and `state` are separate states.
Coverage cannot label a module unobserved when its calls are present. Omitted events
require a reason. Unsupported arguments and exhausted budgets prevent a capture
from claiming completeness; a failed forward may retain a useful partial prefix.
Operations, invocations, values and serialized bytes have independent limits.

## Attempt state and retention

`RunAttempt` records the latest execution outcome separately from capture identity.
Its request digest and origin exist even when setup refuses. A preparation can
establish source identity while the plan digest remains null; an automatic guess or
invalid configuration never receives an invented plan. The worker reports its
resolved plan before launch so a timeout can retain that setup without a capture.
`CaptureState` separately names the last complete, last partial and displayed
captures. The host ledger uses these records for manual, automatic and MCP runs. A timeout
can terminate an attempt without producing a new capture; a projection failure can
retain a new capture while displaying an earlier one. Legacy evidence is explicitly
partial, even after a successful forward, because its completeness is not established.
It cannot fill the complete semantic-capture slot. Captures are immutable JSON copies,
bounded to 64 MiB each; only complete/partial/displayed slots remain retained. Oversize
captures retain their identity with an explicit retention-limit reason. The latest
resolved plan or preparation has a separate 16 MiB bound. These are host-session
records, not disk persistence. Planned and prepared automatic captures populate the complete/partial slots after
host schema, execution-identity, budget and basic completeness checks. Python remains
the owner of full relational validation. Invalid or duplicate canonical payloads
cannot replace complete evidence; their legacy result remains explicitly partial
with a host-private rejection reason.

Graph recording defaults to 16 MiB, independently of the validation-record budget
of 4 MiB. Explicit smaller `limits.max_capture_bytes` values remain authoritative.
The worker envelope has a 64 MiB bound because it also includes the legacy graph,
execution plan and evidence outside recorder accounting. The standalone service
response has a 128 MiB bound to accommodate its additional projection. These
transport bounds do not increase recorder, validation-record or browser-document
limits. An incomplete capture is never promoted to complete after transport.

The browser's bounded `validation.evidence` summary carries attempt origin/outcome,
source/plan identities, retained slot IDs and displayed-capture attribution. Its
optional resolved setup labels contain only profile name, device and mode. Full
recipes and interpreter paths remain host-owned. Reconnect rebuilds attribution from
the current ledger, and a late capture can update it without replacing a newer
refused attempt. Legacy and incomplete canonical captures are labelled incomplete
in the result panel. Capture completeness does not certify the legacy canvas projection.

Captured observations are not another execution-input language. Saved-input
materialization continues to use `services.validation.contracts.Binding`; explicit
setup/fixture/checkpoint requests continue through the existing execution machinery.

## Recorder and identity

`services/semantic/identity.py` is shared by the capture adapters. It uses
weak references and monotonically assigned IDs; producer identity does not require
keeping activations alive. Releasing an object clears its lookup entry, while already
serialized metadata remains. Input-boundary seeds use those same recorder identities.
The canonical recorder and legacy compatibility adapter share this identity implementation.

`services/semantic/tensors.py` owns the common weak tensor object/storage/version
observations used by the validation recorder. That legacy adapter still rejects
writes it cannot express using PyTorch counters. The canonical recorder extends
the snapshot key with a declared-write epoch and keeps weak per-storage alias
indices; no alias inventory keeps activations alive. The semantic argument encoder
preserves positional/keyword order, repeated references, typed literals, symbolic
descriptions and nested output paths. Its bounded built-in traversal records an
opaque subtree for unsupported objects, cycles/depth, large containers and exhausted
value budgets; it never evaluates an expression or serializes an object's repr.
Workers using resolved plans publish `diagnostics.semantic_capture`, validated against
the canonical contract, alongside the legacy graph. `semantic_capture_error` is an
explicit bounded reason when no canonical payload can be retained. The worker's
five-key/version-1 envelope is unchanged. Prepared automatic workers publish the
same contract with `planDigest=null`, a `preparationDigest` and selected-candidate
provenance/index/total attempts. Planned captures have a plan digest and null
automatic fields. Exactly one setup identity is required. Only the selected call
is retained, and search's reuse of model state is disclosed. Actual input and
pre-call state digests remain separate from the preparation's applicability key.
Imported source outside the saved snapshot discards canonical attribution with
`source_mismatch`; changes after importing original bytes retain their original
capture with a stale current flag. The host rechecks saved setup after worker exit
and after asynchronous graph projection. Source/import/configuration/environment
changes keep retained captures inspectable but cannot certify the current graph.

The existing dispatcher and module-call hooks feed `services/semantic/recorder.py`;
there is no second dispatch observer. Definitions, registered aliases, actual
module instances and calls are separate. Unregistered dynamic modules have an empty
`paths` list, not an invented source path. Parameter/buffer references are observed
from the entry inventory. Dynamic state without that identity remains incomplete.
Custom operations keep their target and dispatcher schema as atomic operations.
Python statements and kernel internals are outside this representation.

Canonical operations and failed invocations share the bounded forward
diagnostics. Proven operator shape/dtype constraints survive in the capture; arbitrary
exception prose is never parsed for dimensions or copied into diagnostic facts.
Exception causality distinguishes a propagated operator failure from a later assertion
or explicitly suppressed error. Identical safe facts can share a diagnostic record;
operation/invocation identities still distinguish the events. Diagnostic byte/item
exhaustion leaves an explicitly incomplete capture and a reserved generic failure link.
The recorder does not retain extra exception/traceback objects for this mapping.

The function boundary completes tensor metadata after PyTorch updates autograd and
mutation versions. Output/effect ports preserve producer paths and earlier values
remain immutable. Non-returned writable values have explicit effect producers.
Some custom dispatch kernels do not advance PyTorch's version counter as an ordinary
forward would; the capture records that observed counter without inventing a change,
and discloses its separate schema-write logical versions in normalization metadata.
Potentially partial writes from failed kernels remain incomplete. Escaped mutable
storage and unsupported tensor/argument types
also prevent completeness. Instrumentation errors do not replace a model's result.
No-capture byte errors can occur when even the immutable source/envelope exceeds
the requested budget. Conservative reservations bound incremental retention before
the final relational/byte validation.
Tensor producer paths are charged at their encoded size together with the output
or mutation effect that owns them. Event reservations cover only bounded terminal
fields and an opaque fallback; unused blanket padding must not discard a graph
that fits. A byte stop keeps that reason instead of also claiming the invocation
count was exhausted.

## Source correspondence

Source cards join exact saved-source captures through `SourceCorrespondence`:
registered paths identify module instances and retained invocations, shared aliases
remain explicit, and parameter/buffer values are not described as module calls.
Missing calls, capture limits, source-only declarations and unresolved expressions
have separate labels. A single shape is summarized only when every retained call
has the same direct tensor shape/dtype; multiple outputs and varying calls keep
their full trees without a misleading first-tensor summary. Failed calls keep their
measured input and no invented output. Joined observations bypass the legacy shape
rules and runtime fallback, including an explicit unknown output.

The shared validation loop passes its host attempt ID into plan resolution.
That exact ID is used by the plan and canonical capture, including manual, MCP and
save-driven attempts. Direct worker callers without a host attempt still receive a
fresh ID. The applicability digest continues to exclude per-attempt identity.

Contract tests cover repeated operands, ordered nested outputs, large integers,
symbolic values, invocation sharing, aliases, mutation versions, source revisions,
partial failures, contradictory coverage, budgets and generated-file parity.

## Reversible projection v1

`services/semantic/projection.py` derives `CaptureProjection` from a validated
capture without importing Torch. Every operation and invocation has its own node;
the projection refers back to their captured IDs rather than copying or lowering
their meaning again. Ports address exact serialized argument/output/effect paths.
Literal, symbolic, opaque and empty-container ports remain inspectable without
fake tensor connections. Multiple root input names for the same value remain
explicit aliases; Whisper-style distinct input tensors keep separate ports.

Connections exist only for captured tensor argument uses, effect predecessors and
returned values. Their source is the actual producer output/effect or an observed
input/state/external value. An unused model input is never connected by layout rank.
An identity module retains its invocation interfaces without an invented dispatched
operation or a claim that returning the same tensor creates a new producer.

Collapse retains every original node, port and connection. Only visibility and
display endpoints change. A boundary port cites the exact member port it replaces,
including the original destination argument position; internal connections keep
their original endpoints while hidden. Expanding regenerates the identical detailed
projection. Hidden-operation counts count actual retained records. State/external
source nodes are placed at the common scope of their uses so private state can hide
inside a collapsed region and shared state remains shared. This placement is a view
choice, not a change in parameter or module ownership.

The Python and generated TypeScript projection contracts validate ownership,
visibility, hidden counts, value identity, port direction and collapsed endpoints.
The producer, live document projection, canvas ports, MCP topology and portable
viewer use this contract. Legacy captures retain their original semantic limitations.

The GraphDocument handoff preserves declared ordered input/output port
IDs (including empty and unconnected ports) and exact edge handles. Legacy nodes
without declarations derive their port IDs from actual connections, falling back
to the old x/y defaults only when necessary. Python publishes semantic parent
ownership directly, and browser/MCP readers prefer it over legacy canvas copies.
The legacy recorder does not gain argument semantics it never captured.

## Live captured documents

`services/semantic/document.py` projects canonical captures directly into the
existing GraphDocument envelope, bypassing legacy source/trace edge splicing,
rank-based edges and destructive repeat folding. Both single-file and multi-file
projection use the same producer. Invalid or foreign captures refuse; they never
fall back to legacy wiring. A valid partial/identity capture is usable even when
the old visual recorder produced no nodes.

Each rendered record carries `CaptureReference` (capture/source, kind, semantic ID),
ordered `DisplayPort` records (exact argument paths, value identity, shape, dtype,
device), semantic parent ownership and actual descendant operation counts.
Occurrences are scoped to their capture, so reused event ordinals in a later run
cannot silently become the same selected card. Repeated/shared calls have separate
records and readable occurrence labels. Each edge keeps its original `ProjectionEdge`.

The host retains the full CaptureSnapshot for subsequent inspection. The document
has a compact capture summary and source-correspondence table, not a second copy of
the environment/setup/argument payload. Its independent 24 MiB decoded budget leaves
room for spans and message metadata within the transport's 32 MiB decoded ceiling.
Large messages use lossless gzip JSON while the broker's wire limit remains 8 MiB.
Overflow refuses with a display-budget message while retained evidence remains
available. No records or ports are removed to fit.

The browser keeps original ports/edges intact. Collapsed interfaces use ephemeral
`boundary:<memberPortId>` identities and one display connection per original use.
Repeated operands remain distinct. Port pins and both initial/dragged edge anchors
share deterministic geometry; card height accounts for the port count. Literal and
empty-container ports have no invented tensor edges. Root boundaries display each
typed port, including scalar `[]` and differently typed inputs, rather than a first
tensor summary. Expanded calls show their actual retained descendants; no repeat
representative replaces another captured occurrence.

Document validation checks references, typed-port consistency and recorded shape
summaries without running inference or constructing Torch layers. Its results
describe retained evidence, not a new execution or certification of other inputs.
Failed calls keep their own error evidence; the propagated terminal failure maps to
the canonical operation or root call. Static opening remains unchanged and never
imports user code. Source spans on uniquely registered calls permit navigation;
they do not grant an edit capability.

## Retained capture inspection

The host `CaptureInspector` reads only the ledger’s validated immutable canonical
records. MCP `list_captures` and `inspect_capture` expose the independent slots and
page the exact JSON tree at each container, preserving argument types/order,
repeated references, state aliases/versions, effects and coverage. Container entries
carry child counts; leaves keep exact scalars. Pages stay within 64 KiB and never
replace a large operation tree with an omitted-record placeholder. Queries require
exact capture/source identities and model scope; stale source does not destroy
retained evidence, while live focus still checks current revisions. The browser
uses this same inspector through private correlated request/reply messages and a
temporary Captured details dialog. Ordered argument/output records link to their
tensor facts; whole-capture browsing exposes state, coverage and diagnostics.

MCP runs retain the same validated payload under their independent existing
16 MiB-per-run/64 MiB-per-window allowance, 32-run count and 30-minute/session lifetime.
Live-slot eviction or model retargeting does not erase an owned retained run.
`inspect_run` uses this reader for its canonical `capture`, `operations` and `modules`
sections; legacy runs disclose unavailable semantics. Rejected capture identity
cannot be replaced with legacy operation/coverage claims. Run graph snapshots retain
their projected capture identity, module instance IDs and every shared registration
path. Capture and graph retention failures stay distinct. A second worker callback
cannot replace the first captured run evidence, even while projection is pending.

Failure inspection traverses recorded value producers (including nested argument
trees and mutation-effect predecessors) from operations sharing the root diagnostic.
These are explicit diagnostic candidates: equal safe diagnostic facts can describe
multiple caught failures, and do not establish exception causality. No operation is
invented for an assertion before dispatcher entry. State/argument-aware comparison
shares the canonical host/browser engine across MCP and the portable viewer.

## Portable capture v1

`PortableCapture` versions the detached payload independently from runtime and graph
file versions. `services/semantic/portable.py` validates and copies canonical evidence
without importing Torch or workspace code. Inputs, outputs, operations, invocations,
values, effects, aliases, module/definition identities, diagnostics and coverage stay
exact. Workspace/model/session identities are replaced by `portable`, editor versions
are cleared, and file references (including source anchors) use deterministic aliases.
The conventional configuration name remains `tensorviz.toml`. A digest identifies the
original source snapshot without publishing its paths; detached source cannot certify
an execution plan from the live workspace.

Literal privacy is lossless-or-refuse: absolute/path-like literals and literals under
recognized private/internal argument names refuse export with a fixed message.
Nested containers are checked; tensor references and None remain valid under those
names because they contain no payload. No operand is removed or replaced. Ordinary
model names and literals are computational evidence, so this is not a general secret
detector or an assurance that arbitrary user labels contain no personal information.
Source bodies, tensor payloads and environment paths are not capture fields.

Shared fixtures and executable corpus checks cover the producer and contract.
Version 4 files, live export, offline validation/inspection and canonical comparison
consume this payload. Uncertain correspondence remains explicit.

`packages/contracts/src/captureReader.ts` owns the nonexecuting structural reader shared
by host/MCP and the browser's checked generated copy. The offline client accepts an
already-validated capture and provides the same bounded pages to Captured details;
it has no network or source-edit capability. File import and the viewer use it.
UTF-8 byte limits, exact argument reconstruction, foreign-identity refusal and
prototype-safe own-property traversal apply identically in both environments.

The version 4 graph-file importer validates the detached payload with precompiled
Python-generated JSON-schema validators and explicit offline relationships/privacy
checks. Shared Python/browser acceptance cases cover producer/alias/version ordering,
ownership, boundaries, coverage, limits and non-returned mutation. Its document gate
checks every rendered event, exact argument/effect/result use, producer and value fact
against the payload. File layout can reorder display records but cannot omit events
or rewire them. The offline viewer uses the shared reader and discloses partial and
legacy evidence. The integrated canonical comparison preserves those limitations.

Live export uses the same retained record ownership checks as inspection. The host
rechecks displayed target/capture, eviction and connection epoch after Python detaches
the payload. Private request/reply transport never broadcasts the capture or retains
it in reconnect snapshots. One export can prepare at a time; cancellation aborts the
request. Authenticated preparation bounds raw input to 16 MiB and detached output to
6 MiB, with exact computational equality checks in the host and full offline validation
in the browser. The browser rechecks current projection and capture before download;
layout changes during preparation are preserved. No source is read or model executed.

## Verified authored source locations

Canonical definitions, calls and operations can cite exact saved-source bindings.
The existing code verifier supplies bounded, digest-checked source before model import;
PEP 657 instruction ranges are joined to one AST expression in the same verified code
scope. Definition bindings identify a known forward implementation. Coordinates are
one-based lines and zero-based UTF-16 columns. No source bodies or live frames are
retained in the capture. Unknown ranges/external helpers remain unmapped.

A binding is location evidence, not an edit capability: one expression may produce
several operations or repeated calls. Binding metadata has its own bounded share of
the overall capture budget. Supported constructor, argument and direct scalar
configuration bindings have explicit edit capabilities and shared scope;
ambiguous or unsupported recipes refuse.

## Stored dictionary outputs

Dictionary subclasses, including Transformers `ModelOutput` objects, expose their
stored entries as dictionary argument trees. Built-in `dict`/`OrderedDict` methods
read the storage directly, retaining OrderedDict order without calling overridden
`items`, length, iteration, lookup or repr methods. Custom object attributes and
computed mapping behavior are not reconstructed; the capture's normalizations
record this distinction. Existing key, item, depth and tensor budgets still apply.

Tensor outputs now retain their identities and measured shapes instead of becoming
opaque solely because the container is a dictionary subclass. Unsupported nested
objects remain explicit: for example, T5's `EncoderDecoderCache` still yields
partial coverage while its encoder and decoder tensor outputs remain visible.
