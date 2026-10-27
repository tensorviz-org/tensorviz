"""Template auto-generation staging service.

Takes template gaps identified by the supervisor and generates candidate
templates for human review and approval.  This is the human-in-the-loop
bridge between "the system detects it needs a new template" and "a new
template is registered in CanonicalTemplateRegistry".

Workflow:
1. Gap identified during deliberation → saved to artifact store
2. This service picks up the gap and generates a candidate template spec
3. Human reviews via staging endpoint (approve/reject/edit)
4. On approval, the template contract is ready for registry integration
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("uvicorn.error")


# ---------------------------------------------------------------------------
#  Staging models
# ---------------------------------------------------------------------------


class StagedTemplateNode(BaseModel):
    """Proposed node within a staged template."""
    suffix: str = Field(description="Node suffix ID (e.g. 'conv1', 'bn1')")
    layer_op: str = Field(description="PyTorch layer op (e.g. 'nn.Conv2d')")
    default_params: Dict[str, Any] = Field(default_factory=dict)


class StagedTemplateEdge(BaseModel):
    """Proposed edge within a staged template."""
    source_suffix: str
    target_suffix: str


class StagedTemplate(BaseModel):
    """A proposed template awaiting human approval."""
    staging_id: str = Field(description="Unique staging identifier")
    template_id: str = Field(description="Proposed canonical template ID")
    name: str = Field(description="Human-readable template name")
    family: str = Field(description="Architecture family")
    nodes: List[StagedTemplateNode] = Field(default_factory=list)
    edges: List[StagedTemplateEdge] = Field(default_factory=list)
    required_nodes: List[str] = Field(default_factory=list, description="For the TemplateContract")
    required_edges: List[str] = Field(default_factory=list, description="For the TemplateContract")
    allowed_residual_patterns: List[str] = Field(default_factory=list)
    training_minimums: List[str] = Field(default_factory=list)
    input_expectation: str = ""
    source_gap_id: Optional[str] = None
    status: str = "pending"  # pending, approved, rejected, edited
    created_at: float = Field(default_factory=time.time)
    reviewed_at: Optional[float] = None
    reviewer_notes: str = ""
    generated_by: str = "heuristic"  # heuristic or llm


class StagedTemplateSummary(BaseModel):
    """Summary of staged templates."""
    total: int = 0
    pending: int = 0
    approved: int = 0
    rejected: int = 0


# ---------------------------------------------------------------------------
#  Staging store (in-memory for now, can be persisted later)
# ---------------------------------------------------------------------------


class TemplateStagingStore:
    """Store for staged template candidates with optional JSON persistence."""

    def __init__(self, registry_path: Optional[str] = None) -> None:
        self._registry_path = os.path.abspath(registry_path) if registry_path else None
        self._staged: Dict[str, StagedTemplate] = {}
        self._counter = 0
        self._load()

    def add(self, template: StagedTemplate, allow_duplicate: bool = True) -> str:
        """Add a staged template and return its staging_id."""
        if not allow_duplicate and template.source_gap_id:
            existing = self.find_by_gap_template(
                gap_id=template.source_gap_id,
                template_id=template.template_id,
                statuses=["pending", "approved", "edited"],
            )
            if existing:
                logger.info(
                    "template.staging.add.skip_duplicate staging_id=%s gap_id=%s template_id=%s",
                    existing.staging_id,
                    template.source_gap_id,
                    template.template_id,
                )
                return existing.staging_id

        self._counter += 1
        template.staging_id = f"staged_{self._counter}_{template.template_id}"
        self._staged[template.staging_id] = template
        self._persist()
        logger.info(
            "template.staging.add staging_id=%s template_id=%s family=%s",
            template.staging_id, template.template_id, template.family,
        )
        return template.staging_id

    def get(self, staging_id: str) -> Optional[StagedTemplate]:
        return self._staged.get(staging_id)

    def find_by_gap_template(
        self,
        *,
        gap_id: str,
        template_id: str,
        statuses: Optional[List[str]] = None,
    ) -> Optional[StagedTemplate]:
        for staged in self._staged.values():
            if staged.source_gap_id != gap_id:
                continue
            if staged.template_id != template_id:
                continue
            if statuses and staged.status not in statuses:
                continue
            return staged
        return None

    def list_all(self, status: Optional[str] = None) -> List[StagedTemplate]:
        templates = list(self._staged.values())
        if status:
            templates = [t for t in templates if t.status == status]
        return sorted(templates, key=lambda t: t.created_at, reverse=True)

    def approve(self, staging_id: str, reviewer_notes: str = "") -> Optional[StagedTemplate]:
        template = self._staged.get(staging_id)
        if template and template.status == "pending":
            template.status = "approved"
            template.reviewed_at = time.time()
            template.reviewer_notes = reviewer_notes
            self._persist()
            logger.info("template.staging.approve staging_id=%s template_id=%s", staging_id, template.template_id)
            return template
        return None

    def reject(self, staging_id: str, reviewer_notes: str = "") -> Optional[StagedTemplate]:
        template = self._staged.get(staging_id)
        if template and template.status == "pending":
            template.status = "rejected"
            template.reviewed_at = time.time()
            template.reviewer_notes = reviewer_notes
            self._persist()
            logger.info("template.staging.reject staging_id=%s template_id=%s", staging_id, template.template_id)
            return template
        return None

    def summary(self) -> StagedTemplateSummary:
        all_templates = list(self._staged.values())
        return StagedTemplateSummary(
            total=len(all_templates),
            pending=sum(1 for t in all_templates if t.status == "pending"),
            approved=sum(1 for t in all_templates if t.status == "approved"),
            rejected=sum(1 for t in all_templates if t.status == "rejected"),
        )

    def _load(self) -> None:
        if not self._registry_path or not os.path.exists(self._registry_path):
            return
        try:
            with open(self._registry_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return
        raw_templates = payload.get("templates", []) if isinstance(payload, dict) else payload
        if not isinstance(raw_templates, list):
            return
        for row in raw_templates:
            try:
                template = StagedTemplate.model_validate(row)
            except Exception:
                continue
            self._staged[template.staging_id] = template
            self._counter = max(self._counter, self._extract_counter(template.staging_id))

    def _persist(self) -> None:
        if not self._registry_path:
            return
        directory = os.path.dirname(self._registry_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "templates": [
                template.model_dump(mode="json")
                for template in sorted(self._staged.values(), key=lambda t: t.staging_id)
            ]
        }
        temp_path = f"{self._registry_path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(temp_path, self._registry_path)

    @staticmethod
    def _extract_counter(staging_id: str) -> int:
        parts = staging_id.split("_")
        if len(parts) < 3:
            return 0
        try:
            return int(parts[1])
        except ValueError:
            return 0


# ---------------------------------------------------------------------------
#  Template auto-generation service
# ---------------------------------------------------------------------------


class TemplateAutoGenService:
    """Generates candidate templates from detected gaps.

    The heuristic generator creates reasonable template skeletons based on
    the gap's missing_templates and required_actions.  Future: LLM-powered
    generation using the paper text and architecture spec.
    """

    def __init__(self, store: Optional[TemplateStagingStore] = None) -> None:
        self._store = store or TemplateStagingStore()

    @property
    def store(self) -> TemplateStagingStore:
        return self._store

    def generate_from_gap(
        self,
        gap_id: str,
        missing_templates: List[str],
        required_actions: List[str],
        family: str = "unknown",
        allow_duplicates: bool = False,
    ) -> List[str]:
        """Generate staged template candidates from a gap.

        Returns list of staging_ids for the generated templates.
        """
        staging_ids: List[str] = []

        for template_id in missing_templates:
            staged = self._heuristic_generate(
                template_id=template_id,
                gap_id=gap_id,
                family=family,
                required_actions=required_actions,
            )
            sid = self._store.add(staged, allow_duplicate=allow_duplicates)
            staging_ids.append(sid)

        logger.info(
            "template.autogen.done gap_id=%s generated=%d templates=%s",
            gap_id, len(staging_ids), missing_templates,
        )
        return staging_ids

    def _heuristic_generate(
        self,
        template_id: str,
        gap_id: str,
        family: str,
        required_actions: List[str],
    ) -> StagedTemplate:
        """Generate a skeleton template using heuristic rules."""
        # Default: create a simple linear chain
        nodes = [
            StagedTemplateNode(suffix="identity", layer_op="nn.Identity", default_params={}),
            StagedTemplateNode(suffix="linear1", layer_op="nn.Linear", default_params={"in_features": 64, "out_features": 64}),
            StagedTemplateNode(suffix="activation", layer_op="nn.ReLU", default_params={"inplace": True}),
        ]
        edges = [
            StagedTemplateEdge(source_suffix="identity", target_suffix="linear1"),
            StagedTemplateEdge(source_suffix="linear1", target_suffix="activation"),
        ]

        # Add residual if detected in actions
        if any("residual" in a or "shortcut" in a for a in required_actions):
            nodes.append(StagedTemplateNode(suffix="add", layer_op="torch.add", default_params={}))
            edges.append(StagedTemplateEdge(source_suffix="activation", target_suffix="add"))
            edges.append(StagedTemplateEdge(source_suffix="identity", target_suffix="add"))

        return StagedTemplate(
            staging_id="",  # Will be set by store
            template_id=template_id,
            name=template_id.replace("_", " ").title(),
            family=family,
            nodes=nodes,
            edges=edges,
            required_nodes=[n.layer_op for n in nodes if n.layer_op != "nn.Identity"],
            required_edges=[f"{e.source_suffix} -> {e.target_suffix}" for e in edges],
            allowed_residual_patterns=["identity shortcut"] if any("residual" in a for a in required_actions) else [],
            training_minimums=["in_features > 0", "out_features > 0"],
            input_expectation="rank-2 tensor [batch, features]",
            source_gap_id=gap_id,
            generated_by="heuristic",
        )
