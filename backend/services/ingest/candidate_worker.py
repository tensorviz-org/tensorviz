"""Candidate adapter for the ordinary worker, not another execution engine."""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat

from ..semantic.candidate import stage_candidate, verify_replay_base
from ..semantic.candidate_postconditions import check_postconditions
from ..semantic.contracts import CandidateRequest
from ..semantic.execution_plan import applicability_digest, consume_private_json
from ..validation.setup import workspace_file
from .worker import _Envelope, EXIT_USAGE, main

CANDIDATE_ENV = "TENSORVIZ_CANDIDATE_REQUEST"
MAX_REQUEST_BYTES = 24 * 1024 * 1024


class _CandidateEnvelope(_Envelope):
    def __init__(self, output, base, workspace, candidate, postconditions):
        super().__init__()
        self._out = output
        self.base = base
        self.workspace = workspace
        self.candidate = candidate
        self.postconditions = postconditions

    def emit(self, code):
        # Source/config/environment changes during import, construction, forward
        # or postprocessing keep the result inspectable but cannot certify Apply.
        current = False
        try:
            verify_replay_base(self.base, self.workspace)
            current = True
        except Exception:
            pass
        self.worker.update(candidateExecution=self.candidate.model_dump(mode="json"),
            candidateBaseCurrent=current)
        payload = (self.result or {}).get("diagnostics", {}).get("semantic_capture")
        self.worker["candidatePostconditions"] = check_postconditions(
            self.candidate, self.postconditions, payload).model_dump(mode="json")
        # Defer publication until the candidate directory has been removed. A
        # cleanup failure must not append a second JSON object after a success.
        return code


def run_candidate(args, envelope):
    """Called only by the private execution flag; never by static source review."""
    # Never let inherited bootstrap credentials alter the nested ordinary worker.
    os.environ.pop("TENSORVIZ_EXECUTION_PLAN", None)
    try:
        if (args.kind != "model" or args.tier != "real" or not args.workspace or not args.cls
                or args.resolved_execution or args.capture_preparation or args.execution is not None
                or args.diagnostic_job is not None or args.input is not None or args.tensor_inputs is not None
                or args.overrides is not None or args.capture or args.loss is not None or args.callable_ is not None
                or args.steps is not None or args.roots or args.no_frames):
            raise ValueError("competing candidate arguments")
        request_file, raw = consume_private_json(CANDIDATE_ENV, MAX_REQUEST_BYTES)
        request = CandidateRequest.model_validate(raw)
        # The host transport owns this outer private directory so it can remove
        # everything even after SIGKILL. Never put a candidate in a public parent.
        parent = request_file.resolve(strict=True).parent
        info = parent.stat()
        if os.name != "nt" and (info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700):
            raise ValueError("private candidate directory required")
        if (request.plan.source.className != args.cls
                or workspace_file(args.workspace, request.plan.source.modelFile) != Path(args.file).resolve(strict=True)):
            raise ValueError("candidate target mismatch")
        with stage_candidate(args.workspace, request.plan, request.capture,
                request.intent.model_dump(mode="json"), request.reviewedFile, request.reviewedSource,
                temporary_parent=str(parent), candidate_id=request.id) as staged:
            plan = staged.execution.plan
            plan_file = staged.workspace.parent / "execution.json"
            with plan_file.open("x", encoding="utf-8") as stream:
                json.dump(dict(version=1, plan=plan.model_dump(mode="json"),
                    applicabilityDigest=applicability_digest(plan)), stream, ensure_ascii=False)
            plan_file.chmod(0o600)
            os.environ["TENSORVIZ_EXECUTION_PLAN"] = str(plan_file)
            candidate_envelope = _CandidateEnvelope(envelope._out, request.plan, args.workspace,
                staged.execution, staged.postconditions)
            code = main(["--resolved-execution", "--workspace", str(staged.workspace),
                "--file", str(staged.workspace / plan.source.modelFile), "--cls", plan.source.className,
                "--depth", str(args.depth), "--json"], envelope=candidate_envelope)
        return _Envelope.emit(candidate_envelope, code)
    except Exception:
        # Arbitrary source, input values, filesystem paths and exception bodies
        # never enter this refusal. Ordinary runtime diagnostics retain their own
        # existing structured/redacted worker boundary.
        envelope.error = dict(type="CandidateRefused", message="The reviewed candidate or its saved setup is unavailable or changed.",
            frames=[], node_id=None)
        return envelope.emit(EXIT_USAGE)
    finally:
        os.environ.pop(CANDIDATE_ENV, None)
        os.environ.pop("TENSORVIZ_EXECUTION_PLAN", None)
