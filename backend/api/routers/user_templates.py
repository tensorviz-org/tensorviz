"""Owner-scoped personal templates ("starter files") API — Phase 5 §7.

Lets a user save the current graph as a reusable template and start new graphs
from it (the frontend fetches the template and creates a graph via the projects
API — duplicate-to-project). Mirrors projects.py: every endpoint authenticates
via ``get_current_user`` and scopes to ``user.uid``; foreign/missing rows → 404.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from auth import CurrentUser, get_current_user
from services.store_factory import get_project_store


router = APIRouter(prefix="/api/user-templates", tags=["user-templates"])
_store = get_project_store()


class CreateUserTemplateRequest(BaseModel):
    # A single-line label in the template list (200) and a short blurb under it
    # (2000) — both well under the 8000 projects.py uses for real long text.
    name: str = Field(min_length=1, max_length=200)
    canonical_graph: Optional[Dict[str, Any]] = None
    description: Optional[str] = Field(default=None, max_length=2000)

    @field_validator("name")
    @classmethod
    def _name_not_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("name must not be blank")
        return stripped


def _decode(template: Dict[str, Any]) -> Dict[str, Any]:
    """Parse the stored canonical_graph TEXT back into a JSON object for responses."""
    out = dict(template)
    raw = out.get("canonical_graph")
    if isinstance(raw, str):
        try:
            out["canonical_graph"] = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            out["canonical_graph"] = {}
    return out


@router.post("")
def create_user_template(request: CreateUserTemplateRequest, user: CurrentUser = Depends(get_current_user)):
    canonical = json.dumps(request.canonical_graph) if request.canonical_graph is not None else "{}"
    template = _store.create_user_template(
        user.uid,
        name=request.name,
        canonical_graph=canonical,
        description=request.description,
    )
    return {"template": _decode(template)}


@router.get("")
def list_user_templates(user: CurrentUser = Depends(get_current_user)):
    return {"templates": [_decode(t) for t in _store.list_user_templates(user.uid)]}


@router.get("/{template_id}")
def get_user_template(template_id: str, user: CurrentUser = Depends(get_current_user)):
    template = _store.get_user_template(user.uid, template_id)
    if template is None:
        raise HTTPException(status_code=404, detail="Template not found")
    return {"template": _decode(template)}


@router.delete("/{template_id}")
def delete_user_template(template_id: str, user: CurrentUser = Depends(get_current_user)):
    if not _store.delete_user_template(user.uid, template_id):
        raise HTTPException(status_code=404, detail="Template not found")
    return {"ok": True}
