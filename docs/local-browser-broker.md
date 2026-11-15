# Local Browser Broker

The broker is the loopback relay between the one VS Code extension host and the browser views of a
workspace session. Each session uses a launch token and a one-time bootstrap credential.
The standalone plugin service binds `127.0.0.1` only, requires that token, and never
binds `0.0.0.0`. `start.sh` and `server_v2.py` use the same loopback default and refuse a public bind unless `TENSORVIZ_LOCAL_TOKEN` is set. `AUTH_DISABLED` does not grant that bind. See [service bind](local-setup.md#service-bind). This page documents the HTTP and WebSocket surface. Implementation:
`backend/services/ide/local_browser_broker.py` (relay) and
`backend/api/routers/local_browser.py` (HTTP routes). `backend/server_v2.py` mounts the
router outside the `get_current_user` group. `backend/standalone_server.py` mounts the same
router for the plugin, with `source_proposals=False` and `forwarded_loopback=True`. The browser cannot hold the launch token, so the router
carries its own session-scoped credential model.

## Endpoints

| Route | Caller | Auth | Purpose |
| ----- | ------ | ---- | ------- |
| `POST /local/sessions` | extension | per-launch token (see below) | mint a workspace session, TTL, credentials |
| `POST /local/bootstrap` | browser page | URL-fragment credential in the JSON body, once | trade the fragment for the session cookie |
| `WS /local/ws` | both roles | the session cookie plus the exact `Origin` it was issued to | relay typed protocol messages |

## Authentication and bootstrap

- The backend is only reachable in this model when launched with `TENSORVIZ_LOCAL_TOKEN`. Without
  that variable `POST /local/sessions` answers 404, so no sessions can exist and bootstrap plus the
  socket are unreachable in practice — the same inert-by-default property as the IDE middleware.
- The extension proves itself with the token in the `x-tensorviz-token` header, compared with
  `secrets.compare_digest`, checked at request time. The browser never sees this token.
- The mint response returns `workspaceSessionId`, `launchOrigin`, `expiresAt`, `protocolVersion`,
  the declared browser capabilities, `viewCredential` (meant only for the URL fragment), and
  `hostCookieKey` (extension-side only; it never enters a browser flow, and bootstrap refuses it).
- `launchOrigin` is normalized at mint (case-folded scheme/host, no credentials, path, query or
  fragment) so later checks are exact string equality against the header.
- The page posts the fragment credential once, same-origin, to `POST /local/bootstrap`. Refusal
  order is session lookup, origin, expiry, then credential digest. The successful response sets the
  session cookie and returns session info only — never a credential.
- The page reads the fragment once, at load. A new link opened in a tab that already shows the
  page differs only after `#`, so the browser does not load it. The page therefore reloads on a
  `hashchange` that carries a session and credential, and trades the new credential the same way
  (`canvas/src/ide/reloadOnNewLink.ts`). Pages whose host supplies its own transport, such as the
  chat panel, do not watch the fragment.
- The bootstrap `Origin` must equal the launch origin. The extension mints its session for the
  forwarded origin it opens, so the editor backend stays exact. The plugin's standalone service
  cannot know which local port an SSH `-L` forward or an editor's port forwarding will use, so
  its broker (`forwarded_loopback=True`) also accepts an `http` origin on `127.0.0.1` or
  `localhost` at any port, but only when that `Origin` names exactly the `Host` the request was
  sent to. Another local page presents its own origin, which never matches that `Host`, and a
  DNS-rebinding page's `Host` is not a loopback name; the service's trusted-host check also
  refuses it. The view's cookie key, cookie name and socket `Origin` are then bound to that
  forwarded origin, and its socket must arrive through the same forward.
- Browser view cookies use `tensorviz_local_browser_<origin-hash>` so loopback windows with
  different forwarded ports do not overwrite one another. The hash is the first 16 hexadecimal
  characters of SHA-256 of the exact origin the view bootstrapped from. Host sockets keep the fixed
  `tensorviz_local_browser` name in their explicit cookie header. Both use
  `HttpOnly; SameSite=Strict; Path=/local`, session-scoped, and
  `Secure` only when that origin is HTTPS. Its value is an opaque per-role cookie key, not the
  fragment credential, so after bootstrap the fragment secret exists nowhere the browser can reach.
- Capabilities are fixed to read-only: `readOnly`, `canNavigateSource` true; `canWriteSource`,
  `canRunAgent`, `canEditGraph` false. The optional `canProposeSourceEdits: true` enables typed proposals for native host review;
  it grants no direct source or graph write authority.

## Topology

- One `host` connection and many `view` connections per session. A second host handshake is refused
  with `HOST_ALREADY_CONNECTED`; existing views are unaffected.
- Direction is enforced per role: host state (`activeTarget`, `projection`, `syncState`,
  `staleState`, `focusNode`, `ack`, `error`) broadcasts to every view; view traffic
  (`snapshotRequest`, `revealNode`, `ack`, `error`) goes only to the host and is never echoed to
  other views. Broker-owned types (`connected`, `connectionState`, handshakes) cannot be injected.
- Each host-state broadcast is encoded once and the same bounded wire text is sent to
  every recipient. Inbound decoding, contract validation, session checks and ordering
  still run before broadcast; a failed recipient is removed without stopping the others.
- A view connecting before the host receives `connectionState: disconnected`; views are told
  `connected`/`disconnected` as the host arrives and leaves. A host disconnect clears the session's
  host, notifies every view, and restarts the ordering epoch when a new host registers.
- A connection whose registration was revoked mid-flight (a failed send marks its peer dead) stops
  relaying: the relay loop exits instead of letting a ghost peer feed state after the broker
  declared it gone.
- Host teardown and view teardown are idempotent; a view leaving also discards its pending snapshot
  requests.

## Ordering

- Only host-originated ordered state carries `hostSequence`, and it must strictly increase within
  one host epoch. A repeat or rewind is refused to the host with `INVALID_MESSAGE` and dropped —
  never relayed — so no view ever sees the same sequence twice. A new host epoch restarts ordering
  from scratch.

## Snapshot routing and reconnect

- A view refreshes by reusing its cookie on a brand-new socket and sending `snapshotRequest` with
  its last accepted `lastHostSequence`.
- Requests are routed to the host; the host's `snapshot` is delivered to the oldest live asker, one
  request per answer. A view asking twice keeps only its newest request. A host snapshot with no
  pending request is refused with `INVALID_MESSAGE`.
- Requests sent while no host is connected are queued and flushed, oldest first, when a host
  registers; asks from views that have since left are skipped.

## Source proposal and validation routing

`sourceEditRequest`/`sourceEditCancel` and `validateRequest`/`validateCancel` travel
view → host. The corresponding result returns only to the originating view,
correlated by request id and target, without a broadcast sequence. Reply kinds
cannot cross: a validation result cannot settle a source proposal.

A request without a host or with an in-flight duplicate id is refused. Only its
requesting view can cancel it. View disconnect synthesizes cancellation; host
disconnect clears pending requests and the page reports loss of service. Late
replies for cancelled requests are ignored. The broker relays typed intent and
results, never workspace writes. See [source editing](source-editing.md).

Capture inspection and portable export use the same private correlation/ownership
rules. Their reply kinds cannot settle each other or another request type. Detached
captures are bounded to 16 MiB UTF-8 JSON, never broadcast and never stored in reconnect
snapshots. A disconnected view cancels its export preparation.

Publish (`graphPublishRequest`/`graphPublishCancel`) follows the same rules. The request
carries the graph file gzipped in base64, bounded by the website's limits (96 MB, and 23 MB
gzipped); the broker remembers who asked, not the file. The host answers the asking view with
`graphPublishState` progress (`signing_in` with the website's http(s) connect page, then
`uploading`) and one terminal `graphPublishResult`: `published` with the graph's page and
Share page, `failed`, or `cancelled`. The view opens those pages only on the member's
click. See [Add graphs to tensorViz](share-to-website.md#publish-from-the-coding-agent-plugin).

## Expiry

- TTL defaults to 8 hours, bounded to [1s, 24h] at mint. When it fires, every session socket is
  closed with code 4402 / `CREDENTIAL_EXPIRED` and every credential — view keys and the host key —
  is revoked in one sweep.
- The session then remains an inert tombstone for a bounded retention window
  (`LOCAL_BROWSER_TOMBSTONE_RETENTION_SECONDS`, default 1 hour) so a straggling bootstrap is
  answered 410 `CREDENTIAL_EXPIRED` — actionable — rather than the `INVALID_CREDENTIALS` an unknown
  session gets. The reaper deletes the tombstone afterwards, so minted sessions cannot accumulate
  over the process's life.

## Refusals, close codes, and limits

- Transport-level failures (missing/wrong cookie, expired session, wrong origin) refuse before
  `accept()` with the application close code; there is no authenticated channel to carry an error.
  Protocol-level failures after the handshake are answered with the typed `error` message; single
  bad frames do not kill the connection, while handshake failures also close with the paired code.
- Close codes pair 1:1 with contract error codes: 4401 `INVALID_CREDENTIALS`, 4402
  `CREDENTIAL_EXPIRED`, 4403 `INVALID_ORIGIN`, 4404 `PROTOCOL_VERSION_MISMATCH`, 4405
  `SESSION_MISMATCH`, 4406 `INVALID_MESSAGE`, 4407 `UNKNOWN_MESSAGE`, 4408
  `HOST_ALREADY_CONNECTED`.
- Frames are capped at 32 MiB before JSON parsing, which fits Publish's graph file; binary frames
  are refused as `INVALID_MESSAGE`. Each service runs uvicorn with the same `ws_max_size`.
- Every inbound message is validated against the frozen contract once: wrong protocol version,
  cross-session envelope, malformed payload, and unknown types are refused in that order.
- No secret — launch token, view credential, or cookie key — ever appears in request logs, error
  bodies, protocol messages, or workspace files; refusal messages are static text that never quotes
  request content.

## Verification

Use the backend environment from [developer setup](local-setup.md):

```sh
./venv/bin/python -m pytest backend/tests/api/test_local_browser_broker.py backend/tests/api/test_local_browser_protocol.py -q
```

Coverage includes authentication, origin/session isolation, direction and ordering,
request ownership, reconnect, cancellation, expiry, redaction and frame limits.

Candidate review adds private nonterminal `sourceEditReview` updates on the existing
source-request route, preserving its Apply receipt. A supplied source `viewId` and
every `candidateRequest.viewId` must match the actual socket. Candidate replies
are host-only and directed to one connected view, never broadcast. The host owns
retained proposal/candidate authorization after the source request completes and
uses existing broker `viewState` disconnection notifications for revocation.
`test_local_browser_candidates.py` covers forged identities/replies, private progress,
historical reads and late replies after disconnect.
