"""FastAPI interface for the Cox hazard-ratio calculator."""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from cox_hr import check_proportional_hazards, cox_ph, forest_plot_data
from .base import AuditLogger
from .models import SystemTaskPayload
from .supervisor import SystemSupervisor


ROOT = Path(__file__).resolve().parent.parent
WEB_INDEX = ROOT / "web" / "index.html"
CORE_MODULE = ROOT / "cox_hr.py"

supervisor = SystemSupervisor(model_provider="mock")

app = FastAPI(
    title="Cox Hazard Ratio Calculator API",
    description="Cox proportional hazards regression and supporting local utilities.",
    version="1.1.0",
)


class CoxRequest(BaseModel):
    times: List[float] = Field(..., min_length=1)
    events: List[int] = Field(..., min_length=1)
    covariates: List[List[float]] = Field(..., min_length=1)
    labels: Optional[List[str]] = None


class ChatRequest(BaseModel):
    query: str


def _json_safe(value: Any) -> Any:
    """Convert non-finite floats to null-compatible values for JSON responses."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB_INDEX)


@app.get("/cox_hr.py", include_in_schema=False)
def browser_core():
    return FileResponse(CORE_MODULE, media_type="text/x-python")


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "cox-hazard-ratio-calculator",
        "version": app.version,
    }


@app.get("/metrics")
def metrics():
    return {
        "dossiers_processed_total": len(supervisor.dossier_registry),
        "audit_blocks_total": len(AuditLogger.get_trail()),
    }


@app.post("/api/cox")
def api_cox(request: CoxRequest):
    try:
        result = cox_ph(request.times, request.events, request.covariates)
        result["forest_plot"] = forest_plot_data(
            request.times,
            request.events,
            request.covariates,
            labels=request.labels,
        )
        return _json_safe(result)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/ph-check")
def api_ph_check(request: CoxRequest):
    try:
        return _json_safe(
            check_proportional_hazards(
                request.times, request.events, request.covariates
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/audit")
def api_audit(payload: SystemTaskPayload):
    return supervisor.process_task(payload).to_dict()


@app.post("/api/chat")
def api_chat(req: ChatRequest):
    try:
        return {"response": supervisor.query_supervisory_chat(req.query)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/audit/logs")
def api_audit_logs():
    return {
        "audit_trail": AuditLogger.get_trail(),
        "verified": AuditLogger.verify_integrity(),
    }
