from __future__ import annotations
from typing import TypedDict, Annotated
import operator
from app.schemas.clinical import (
    DifferentialDiagnosis,
    DrugInteractionCheck,
    CitedFinding,
    AuditReport,
    ConfidenceSignals,
    CMOSelfReport,
)


class GraphState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    session_id: str
    patient_token: str
    symptoms: list[str]
    history: str
    current_medications: list[str]
    biosensor_data: dict | None

    # ── Triage ────────────────────────────────────────────────────────────────
    active_agents: list[str]
    triage_reasoning: str
    urgency_level: str
    primary_concern: str

    # ── Specialist Outputs ────────────────────────────────────────────────────
    differentials: list[DifferentialDiagnosis]
    drug_interactions: list[DrugInteractionCheck]
    cited_findings: list[CitedFinding]
    biosensor_summary: str

    # ── Agreement helpers for confidence scoring ──────────────────────────────
    pathologist_conditions: list[str]
    literature_conditions: list[str]
    biosensor_conditions: list[str]

    # ── Citation enriched findings ────────────────────────────────────────────
    enriched_findings: list[CitedFinding]

    # ── CMO ───────────────────────────────────────────────────────────────────
    synthesis_narrative: str
    cmo_self_report: CMOSelfReport | None

    # ── Audit ─────────────────────────────────────────────────────────────────
    audit_report: AuditReport | None
    previous_audit_feedback: str

    # ── Confidence ────────────────────────────────────────────────────────────
    confidence_signals: ConfidenceSignals | None
    confidence_score: float
    hitl_required: bool
    escalate_immediately: bool

    # ── Control ───────────────────────────────────────────────────────────────
    loop_count: int
    hitl_triggered: bool
    error: str | None
