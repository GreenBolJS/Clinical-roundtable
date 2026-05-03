from __future__ import annotations
import uuid
import asyncio
import structlog
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from app.schemas.clinical import (
    ClinicalQuery,
    ClinicalReport,
    PatientRecord,
    HITLOverride,
)
from app.pii.redactor import get_redactor
from app.graph.graph import get_graph
from app.graph.state import GraphState
from app.db.supabase import (
    upsert_clinical_report,
    get_clinical_report,
    create_patient,
    update_doctor_override,
)
from app.providers.groq_client import check_groq_health

logger = structlog.get_logger(__name__)

router = APIRouter()


# ── POST /consult ─────────────────────────────────────────────────────────────

@router.post("/consult", response_model=ClinicalReport, tags=["Clinical"])
async def consult(
    query: ClinicalQuery,
    background_tasks: BackgroundTasks,
) -> ClinicalReport:
    """
    Submit a clinical query to the Clinical Roundtable multi-agent system.
    Returns a comprehensive ClinicalReport.
    """
    session_id = str(uuid.uuid4())
    redactor = get_redactor()

    # ── PII Redaction ─────────────────────────────────────────────────────────
    logger.info("consult_start", session_id=session_id, patient_token=query.patient_token)

    # Redact history text
    redacted_history, pii_session = redactor.redact(query.history, session_id)

    # Redact each symptom string
    redacted_symptoms = [redactor.redact(s, session_id)[0] for s in query.symptoms]

    # Redact biosensor data if present
    redacted_biosensor = None
    if query.biosensor_data:
        redacted_biosensor, _ = redactor.redact_dict(query.biosensor_data, session_id)

    # ── Build Initial Graph State ─────────────────────────────────────────────
    initial_state: GraphState = {
        "session_id": session_id,
        "patient_token": query.patient_token,
        "symptoms": redacted_symptoms,
        "history": redacted_history,
        "current_medications": query.current_medications,
        "biosensor_data": redacted_biosensor,
        # Triage outputs (will be populated)
        "active_agents": [],
        "triage_reasoning": "",
        "urgency_level": "routine",
        "primary_concern": "",
        # Specialist outputs
        "differentials": [],
        "drug_interactions": [],
        "cited_findings": [],
        "biosensor_summary": "",
        # Agreement helpers
        "pathologist_conditions": [],
        "literature_conditions": [],
        "biosensor_conditions": [],
        # Citation
        "enriched_findings": [],
        # CMO
        "synthesis_narrative": "",
        "cmo_self_report": None,
        # Audit
        "audit_report": None,
        "previous_audit_feedback": "",
        # Confidence
        "confidence_signals": None,
        "confidence_score": 0.0,
        "hitl_required": False,
        "escalate_immediately": False,
        # Control
        "loop_count": 0,
        "hitl_triggered": False,
        "error": None,
    }

    # ── Run LangGraph ─────────────────────────────────────────────────────────
    graph = get_graph()

    try:
        # LangGraph's ainvoke is async-native
        final_state: GraphState = await graph.ainvoke(initial_state)
    except Exception as exc:
        logger.error("graph_execution_failed", session_id=session_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {exc}")

    # ── Assemble ClinicalReport ───────────────────────────────────────────────
    report = ClinicalReport(
        session_id=session_id,
        patient_token=query.patient_token,
        differentials=final_state.get("differentials", []),
        drug_interactions=final_state.get("drug_interactions", []),
        cited_findings=final_state.get("enriched_findings", final_state.get("cited_findings", [])),
        audit_report=final_state.get("audit_report"),
        confidence_signals=final_state.get("confidence_signals"),
        confidence_score=final_state.get("confidence_score", 0.0),
        hitl_required=final_state.get("hitl_required", False),
        escalate_immediately=final_state.get("escalate_immediately", False),
        biosensor_summary=final_state.get("biosensor_summary", ""),
        synthesis_narrative=final_state.get("synthesis_narrative", ""),
        loop_count=final_state.get("loop_count", 0),
    )

    # ── Persist to the database ───────────────────────────────────────────────
    background_tasks.add_task(upsert_clinical_report, report)

    logger.info(
        "consult_complete",
        session_id=session_id,
        confidence_score=report.confidence_score,
        hitl_required=report.hitl_required,
        escalate_immediately=report.escalate_immediately,
    )

    return report


# ── GET /session/{session_id} ─────────────────────────────────────────────────

@router.get("/session/{session_id}", response_model=ClinicalReport, tags=["Clinical"])
async def get_session(session_id: str) -> ClinicalReport:
    """Retrieve a past ClinicalReport by session ID from the database."""
    try:
        report = await get_clinical_report(session_id)
    except Exception as exc:
        logger.error("get_session_db_error", session_id=session_id, error=str(exc))
        raise HTTPException(status_code=503, detail="Database unavailable")

    if report is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    return report


# ── POST /patient ─────────────────────────────────────────────────────────────

@router.post("/patient", response_model=PatientRecord, tags=["Patients"])
async def create_patient_record(record: PatientRecord) -> PatientRecord:
    """Create a new tokenized patient record in the database (no PII stored)."""
    try:
        created = await create_patient(record)
    except Exception as exc:
        logger.error("create_patient_error", error=str(exc))
        raise HTTPException(status_code=503, detail="Database unavailable")

    return created


# ── GET /health ───────────────────────────────────────────────────────────────

@router.get("/health", tags=["System"])
async def health_check() -> dict:
    """Check provider connectivity for Groq."""
    groq_ok = await check_groq_health()

    status_code = 200 if groq_ok else 503

    body = {
        "status": "healthy" if groq_ok else "degraded",
        "providers": {
            "groq": {
                "status": "ok" if groq_ok else "error",
                "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
            },
        },
    }

    return JSONResponse(content=body, status_code=status_code)


# ── POST /webhook/hitl-response ───────────────────────────────────────────────

@router.post("/webhook/hitl-response", tags=["HITL"])
async def hitl_response(override: HITLOverride) -> dict:
    """
    Endpoint for a human doctor to POST their override decision.
    Updates the session record in the database.
    """
    override_payload = override.model_dump()

    try:
        await update_doctor_override(
            session_id=override.session_id,
            override_payload=override_payload,
        )
    except Exception as exc:
        logger.error("hitl_response_db_error", session_id=override.session_id, error=str(exc))
        raise HTTPException(status_code=503, detail="Database unavailable")

    logger.info(
        "hitl_override_saved",
        session_id=override.session_id,
        doctor_id=override.doctor_id,
        approved=override.approved,
    )

    return {
        "status": "accepted",
        "session_id": override.session_id,
        "message": "Doctor override recorded successfully",
    }
