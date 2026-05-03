from __future__ import annotations
"""
Celery task queue for asynchronous clinical agent dispatch.

While the main LangGraph pipeline runs synchronously per request,
Celery tasks are provided for:
- Long-running background re-analyses
- Batch processing multiple sessions
- Scheduled re-evaluation of flagged sessions

Start worker:
    celery -A app.tasks worker --loglevel=info
"""
import asyncio
import structlog
from celery import Celery
from app.config import get_settings

settings = get_settings()
logger = structlog.get_logger(__name__)

celery_app = Celery(
    "clinical_roundtable",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)


def _run_async(coro):
    """Helper to run async functions from sync Celery tasks."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(
    name="clinical.reanalyze_session",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
)
def reanalyze_session(self, session_id: str) -> dict:
    """
    Re-run the full clinical analysis for an existing session.
    Used when a doctor requests a second opinion or when new data arrives.
    """
    async def _inner():
        from app.db.supabase import get_clinical_report, upsert_clinical_report
        from app.graph.graph import get_graph

        report = await get_clinical_report(session_id)
        if report is None:
            logger.error("reanalyze_session_not_found", session_id=session_id)
            return {"error": "Session not found"}

        from app.graph.state import GraphState
        import uuid

        initial_state: GraphState = {
            "session_id": str(uuid.uuid4()),  # New session ID for re-analysis
            "patient_token": report.patient_token,
            "symptoms": [d.condition for d in report.differentials],
            "history": report.synthesis_narrative[:500],
            "current_medications": [f"{i.drug_a}" for i in report.drug_interactions],
            "biosensor_data": None,
            "active_agents": [],
            "triage_reasoning": "",
            "urgency_level": "routine",
            "primary_concern": "",
            "differentials": [],
            "drug_interactions": [],
            "cited_findings": [],
            "biosensor_summary": "",
            "pathologist_conditions": [],
            "literature_conditions": [],
            "biosensor_conditions": [],
            "enriched_findings": [],
            "synthesis_narrative": "",
            "cmo_self_report": None,
            "audit_report": None,
            "previous_audit_feedback": "",
            "confidence_signals": None,
            "confidence_score": 0.0,
            "hitl_required": False,
            "escalate_immediately": False,
            "loop_count": 0,
            "hitl_triggered": False,
            "error": None,
        }

        graph = get_graph()
        final_state = await graph.ainvoke(initial_state)

        from app.schemas.clinical import ClinicalReport
        new_report = ClinicalReport(
            session_id=initial_state["session_id"],
            patient_token=report.patient_token,
            differentials=final_state.get("differentials", []),
            drug_interactions=final_state.get("drug_interactions", []),
            cited_findings=final_state.get("enriched_findings", []),
            audit_report=final_state.get("audit_report"),
            confidence_signals=final_state.get("confidence_signals"),
            confidence_score=final_state.get("confidence_score", 0.0),
            hitl_required=final_state.get("hitl_required", False),
            escalate_immediately=final_state.get("escalate_immediately", False),
            biosensor_summary=final_state.get("biosensor_summary", ""),
            synthesis_narrative=final_state.get("synthesis_narrative", ""),
            loop_count=final_state.get("loop_count", 0),
        )

        await upsert_clinical_report(new_report)
        return {"session_id": new_report.session_id, "confidence_score": new_report.confidence_score}

    try:
        result = _run_async(_inner())
        logger.info("reanalyze_session_complete", session_id=session_id, result=result)
        return result
    except Exception as exc:
        logger.error("reanalyze_session_failed", session_id=session_id, error=str(exc))
        raise self.retry(exc=exc)


@celery_app.task(
    name="clinical.send_hitl_notification",
    bind=True,
    max_retries=3,
    default_retry_delay=15,
)
def send_hitl_notification(self, session_id: str, payload: dict) -> dict:
    """
    Fire the HITL webhook notification asynchronously via Celery.
    Used as an alternative to the background task in the main flow.
    """
    import httpx

    async def _inner():
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                settings.hitl_webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            return {"status_code": response.status_code, "session_id": session_id}

    try:
        result = _run_async(_inner())
        logger.info("hitl_notification_sent", session_id=session_id, status_code=result["status_code"])
        return result
    except Exception as exc:
        logger.error("hitl_notification_failed", session_id=session_id, error=str(exc))
        raise self.retry(exc=exc)
