from __future__ import annotations
import asyncio
import structlog
from typing import Any

from langgraph.graph import StateGraph, END
from langgraph.constants import Send

from app.graph.state import GraphState
from app.schemas.clinical import (
    ConfidenceSignals,
    InterAgentAgreement,
    CitationDensitySignal,
    AdversarialSignal,
    CMOSelfReport,
)
from app.agents.triage import run_triage
from app.agents.pathologist import run_pathologist
from app.agents.pharmaco import run_pharmaco
from app.agents.literature import run_literature
from app.agents.biosensor import run_biosensor
from app.agents.citation import run_citation
from app.agents.cmo import run_cmo
from app.agents.auditor import run_auditor
from app.confidence.engine import (
    ConfidenceEngine,
    compute_inter_agent_agreement,
    compute_citation_density_signals,
)
from app.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

_confidence_engine = ConfidenceEngine()


# ── Node: Triage ─────────────────────────────────────────────────────────────

async def triage_node(state: GraphState) -> dict:
    logger.info("node_start", node="triage", session_id=state["session_id"])
    try:
        result = await run_triage(
            symptoms=state["symptoms"],
            history=state["history"],
            current_medications=state["current_medications"],
            has_biosensor=state["biosensor_data"] is not None,
        )
        
        # Safety net: always include pharmaco if patient has medications
        active_agents = result.get("active_agents", [])
        if state.get("current_medications") and "pharmaco" not in active_agents:
            active_agents.append("pharmaco")
            result["active_agents"] = active_agents
            logger.info("triage_pharmaco_forced", reason="medications_present")
        
        return {
            "active_agents": result["active_agents"],
            "triage_reasoning": result.get("triage_reasoning", ""),
            "urgency_level": result.get("urgency_level", "routine"),
            "primary_concern": result.get("primary_concern", ""),
        }
    except Exception as exc:
        logger.error("node_error", node="triage", error=str(exc))
        return {
            "active_agents": ["pathologist", "pharmaco", "literature", "biosensor"],
            "triage_reasoning": f"Error: {exc}",
            "urgency_level": "routine",
            "primary_concern": "Triage error",
        }


# ── Fan-out routing ───────────────────────────────────────────────────────────

def fan_out_after_triage(state: GraphState) -> list[Send]:
    """Route to all active specialist agents in parallel using LangGraph Send API."""
    active = state.get("active_agents", ["pathologist", "literature"])
    sends = []

    if "pathologist" in active:
        sends.append(Send("pathologist_node", state))
    if "pharmaco" in active:
        sends.append(Send("pharmaco_node", state))
    if "literature" in active:
        sends.append(Send("literature_node", state))
    if "biosensor" in active and state.get("biosensor_data"):
        sends.append(Send("biosensor_node", state))

    # If somehow no sends, ensure at least pathologist
    if not sends:
        sends.append(Send("pathologist_node", state))

    return sends


# ── Node: Pathologist ─────────────────────────────────────────────────────────

async def pathologist_node(state: GraphState) -> dict:
    logger.info("node_start", node="pathologist", session_id=state["session_id"])
    try:
        differentials = await run_pathologist(
            symptoms=state["symptoms"],
            history=state["history"],
            current_medications=state["current_medications"],
        )
        # Preserve exact condition names for downstream agents
        conditions = [d.condition for d in differentials]
        return {
            "differentials": differentials,
            "pathologist_conditions": conditions,
        }
    except Exception as exc:
        logger.error("node_error", node="pathologist", error=str(exc))
        return {"differentials": [], "pathologist_conditions": []}


# ── Node: PharmacogenomicsSpecialist ─────────────────────────────────────────

async def pharmaco_node(state: GraphState) -> dict:
    logger.info("node_start", node="pharmaco", session_id=state["session_id"])
    try:
        # Pass top differentials as context
        differential_names = [d.condition for d in state.get("differentials", [])]

        print(f"[PHARMACO DEBUG] medications in state: {state.get('current_medications', 'KEY NOT FOUND')}")

        interactions = await run_pharmaco(
            current_medications=state["current_medications"],
            differentials=differential_names,
        )
        return {"drug_interactions": interactions}
    except Exception as exc:
        logger.error("node_error", node="pharmaco", error=str(exc))
        return {"drug_interactions": []}


# ── Node: Literature Critic ───────────────────────────────────────────────────

async def literature_node(state: GraphState) -> dict:
    logger.info("node_start", node="literature", session_id=state["session_id"])
    try:
        differential_names = [d.condition for d in state.get("differentials", [])]
        findings, conditions = await run_literature(
            symptoms=state["symptoms"],
            history=state["history"],
            differentials=differential_names,
        )
        return {
            "cited_findings": findings,
            "literature_conditions": conditions,
        }
    except Exception as exc:
        logger.error("node_error", node="literature", error=str(exc))
        return {"cited_findings": [], "literature_conditions": []}


# ── Node: Bio-Sensor ──────────────────────────────────────────────────────────

async def biosensor_node(state: GraphState) -> dict:
    logger.info("node_start", node="biosensor", session_id=state["session_id"])
    try:
        summary, conditions = await run_biosensor(
            biosensor_data=state.get("biosensor_data"),
            pathologist_differentials=state.get("pathologist_conditions", []),
        )
        return {
            "biosensor_summary": summary,
            "biosensor_conditions": conditions,
        }
    except Exception as exc:
        logger.error("node_error", node="biosensor", error=str(exc))
        return {"biosensor_summary": "Biosensor analysis failed.", "biosensor_conditions": []}


# ── Node: Citation ────────────────────────────────────────────────────────────

async def citation_node(state: GraphState) -> dict:
    logger.info("node_start", node="citation", session_id=state["session_id"])
    try:
        enriched = await run_citation(
            existing_findings=state.get("cited_findings", []),
            differentials=state.get("differentials", []),
            drug_interactions=state.get("drug_interactions", []),
        )
        return {"enriched_findings": enriched}
    except Exception as exc:
        logger.error("node_error", node="citation", error=str(exc))
        return {"enriched_findings": state.get("cited_findings", [])}


# ── Node: CMO ─────────────────────────────────────────────────────────────────

async def cmo_node(state: GraphState) -> dict:
    logger.info("node_start", node="cmo", session_id=state["session_id"], loop_count=state.get("loop_count", 0))
    try:
        narrative, self_report = await run_cmo(
            patient_token=state["patient_token"],
            symptoms=state["symptoms"],
            history=state["history"],
            differentials=state.get("differentials", []),
            drug_interactions=state.get("drug_interactions", []),
            cited_findings=state.get("enriched_findings", state.get("cited_findings", [])),
            biosensor_summary=state.get("biosensor_summary", ""),
            loop_count=state.get("loop_count", 0),
            previous_audit_feedback=state.get("previous_audit_feedback", ""),
        )
        return {
            "synthesis_narrative": narrative,
            "cmo_self_report": self_report,
        }
    except Exception as exc:
        logger.error("node_error", node="cmo", error=str(exc))
        return {
            "synthesis_narrative": f"CMO synthesis failed: {exc}",
            "cmo_self_report": CMOSelfReport(
                diagnosis_confidence=0.3,
                evidence_strength=0.3,
                contraindication_risk=0.5,
                differential_completeness=0.3,
            ),
        }


# ── Node: Auditor ─────────────────────────────────────────────────────────────

async def auditor_node(state: GraphState) -> dict:
    logger.info("node_start", node="auditor", session_id=state["session_id"])
    try:
        audit_report = await run_auditor(
            synthesis_narrative=state.get("synthesis_narrative", ""),
            differentials=state.get("differentials", []),
            drug_interactions=state.get("drug_interactions", []),
            symptoms=state["symptoms"],
            history=state["history"],
        )
        # Build audit feedback string for potential CMO re-loop
        feedback_lines = [f"Audit verdict: {audit_report.verdict}"]
        if audit_report.unanswered:
            feedback_lines.append("Unanswered challenges:")
            for uc in audit_report.unanswered:
                feedback_lines.append(f"  - {uc}")
        return {
            "audit_report": audit_report,
            "previous_audit_feedback": "\n".join(feedback_lines),
        }
    except Exception as exc:
        logger.error("node_error", node="auditor", error=str(exc))
        from app.schemas.clinical import AuditReport
        return {
            "audit_report": AuditReport(
                challenges_raised=0,
                challenges_answered=0,
                unanswered=[],
                verdict="pass",
            ),
            "previous_audit_feedback": "",
        }


# ── Node: Confidence ──────────────────────────────────────────────────────────

async def confidence_node(state: GraphState) -> dict:
    logger.info("node_start", node="confidence", session_id=state["session_id"])

    # Signal 2: Inter-Agent Agreement
    agreement_score = compute_inter_agent_agreement(
        pathologist_conditions=state.get("pathologist_conditions", []),
        literature_conditions=state.get("literature_conditions", []),
        biosensor_conditions=state.get("biosensor_conditions", []),
    )

    # Signal 3: Citation Density
    all_findings = state.get("enriched_findings", state.get("cited_findings", []))
    citation_count, recency_score, rct_ratio = compute_citation_density_signals(all_findings)

    # Signal 4: Adversarial
    audit = state.get("audit_report")
    if audit:
        challenges_raised = audit.challenges_raised
        challenges_answered = audit.challenges_answered
    else:
        challenges_raised = 0
        challenges_answered = 0

    # Signal 1: CMO Self-Report
    cmo_sr = state.get("cmo_self_report") or CMOSelfReport(
        diagnosis_confidence=0.5,
        evidence_strength=0.5,
        contraindication_risk=0.5,
        differential_completeness=0.5,
    )

    signals = ConfidenceSignals(
        cmo_self_report=cmo_sr,
        inter_agent_agreement=InterAgentAgreement(
            pathologist_differentials=state.get("pathologist_conditions", []),
            literature_differentials=state.get("literature_conditions", []),
            biosensor_differentials=state.get("biosensor_conditions", []),
            agreement_score=agreement_score,
        ),
        citation_density=CitationDensitySignal(
            citation_count=citation_count,
            recency_score=recency_score,
            rct_ratio=rct_ratio,
        ),
        adversarial=AdversarialSignal(
            challenges_raised=challenges_raised,
            challenges_answered=challenges_answered,
        ),
    )

    score = _confidence_engine.compute(signals)
    hitl_required = score < settings.hitl_threshold
    unanswered_challenges = max(0, challenges_raised - challenges_answered)
    escalate_immediately = score < settings.escalate_threshold and unanswered_challenges > 3

    logger.info(
        "confidence_result",
        session_id=state["session_id"],
        score=score,
        hitl_required=hitl_required,
        escalate_immediately=escalate_immediately,
        unanswered_challenges=unanswered_challenges,
    )

    return {
        "confidence_signals": signals,
        "confidence_score": score,
        "hitl_required": hitl_required,
        "escalate_immediately": escalate_immediately,
    }


# ── Conditional edge: re-loop or continue ─────────────────────────────────────

def should_reloop(state: GraphState) -> str:
    """
    After confidence_node: only loop back to CMO if audit verdict is fail,
    loop_count < 1, and there are more than 2 unanswered challenges.
    Otherwise → hitl_node.
    """
    audit = state.get("audit_report")
    loop_count = state.get("loop_count", 0)

    if (
        audit is not None
        and audit.verdict == "fail"
        and loop_count < 1
        and len(audit.unanswered) > 2
    ):
        logger.info("graph_reloop", loop_count=loop_count, unanswered=len(audit.unanswered))
        return "reloop"
    return "continue"


async def increment_loop_and_reloop(state: GraphState) -> dict:
    """Helper node that increments loop_count before returning to CMO."""
    return {"loop_count": state.get("loop_count", 0) + 1}


# ── Node: HITL ────────────────────────────────────────────────────────────────

async def hitl_node(state: GraphState) -> dict:
    logger.info("node_start", node="hitl", session_id=state["session_id"])
    if not state.get("hitl_required", False):
        return {"hitl_triggered": False}

    import httpx
    import json

    payload = {
        "session_id": state["session_id"],
        "patient_token": state["patient_token"],
        "confidence_score": state.get("confidence_score", 0.0),
        "escalate_immediately": state.get("escalate_immediately", False),
        "primary_concern": state.get("primary_concern", ""),
        "synthesis_narrative": state.get("synthesis_narrative", "")[:500],
        "differentials": [
            {"condition": d.condition, "probability": d.probability}
            for d in state.get("differentials", [])[:3]
        ],
        "audit_verdict": state.get("audit_report").verdict if state.get("audit_report") else "unknown",
        "unanswered_challenges": state.get("audit_report").unanswered if state.get("audit_report") else [],
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                settings.hitl_webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            logger.info(
                "hitl_webhook_fired",
                session_id=state["session_id"],
                status_code=response.status_code,
            )
    except Exception as exc:
        logger.error("hitl_webhook_failed", session_id=state["session_id"], error=str(exc))

    return {"hitl_triggered": True}


# ── Node: Output ──────────────────────────────────────────────────────────────

async def output_node(state: GraphState) -> dict:
    logger.info("node_start", node="output", session_id=state["session_id"])
    # This node is a final assembly passthrough; the final state IS the report
    return {}


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    builder = StateGraph(GraphState)

    # Add all nodes
    builder.add_node("triage_node", triage_node)
    builder.add_node("pathologist_node", pathologist_node)
    builder.add_node("pharmaco_node", pharmaco_node)
    builder.add_node("literature_node", literature_node)
    builder.add_node("biosensor_node", biosensor_node)
    builder.add_node("citation_node", citation_node)
    builder.add_node("cmo_node", cmo_node)
    builder.add_node("auditor_node", auditor_node)
    builder.add_node("confidence_node", confidence_node)
    builder.add_node("loop_increment_node", increment_loop_and_reloop)
    builder.add_node("hitl_node", hitl_node)
    builder.add_node("output_node", output_node)

    # Entry
    builder.set_entry_point("triage_node")

    # Triage → parallel fan-out
    builder.add_conditional_edges(
        "triage_node",
        fan_out_after_triage,
        # Possible target nodes from fan-out
        ["pathologist_node", "pharmaco_node", "literature_node", "biosensor_node"],
    )

    # All specialist nodes → citation_node
    builder.add_edge("pathologist_node", "citation_node")
    builder.add_edge("pharmaco_node", "citation_node")
    builder.add_edge("literature_node", "citation_node")
    builder.add_edge("biosensor_node", "citation_node")

    # Linear pipeline after citation
    builder.add_edge("citation_node", "cmo_node")
    builder.add_edge("cmo_node", "auditor_node")
    builder.add_edge("auditor_node", "confidence_node")

    # Conditional re-loop edge
    builder.add_conditional_edges(
        "confidence_node",
        should_reloop,
        {
            "reloop": "loop_increment_node",
            "continue": "hitl_node",
        },
    )

    # Re-loop goes back to CMO
    builder.add_edge("loop_increment_node", "cmo_node")

    # Final path
    builder.add_edge("hitl_node", "output_node")
    builder.add_edge("output_node", END)

    return builder.compile()


# Module-level compiled graph
_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph
