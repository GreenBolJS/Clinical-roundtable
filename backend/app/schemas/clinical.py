from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, field_validator
import uuid


class ClinicalQuery(BaseModel):
    patient_token: str = Field(..., description="Tokenized patient identifier, e.g. PATIENT_001")
    symptoms: list[str] = Field(..., min_length=1, description="List of reported symptoms")
    history: str = Field(..., description="Relevant medical history narrative")
    current_medications: list[str] = Field(default_factory=list, description="List of current medications")
    biosensor_data: dict | None = Field(default=None, description="Optional biosensor readings as key-value pairs")


class DifferentialDiagnosis(BaseModel):
    condition: str = Field(..., description="Name of the condition")
    icd10_code: str = Field(..., description="ICD-10 code for the condition")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability estimate 0.0 to 1.0")
    reasoning: str = Field(..., description="Clinical reasoning supporting this diagnosis")


class DrugInteractionCheck(BaseModel):
    drug_a: str = Field(..., description="First drug name")
    drug_b: str = Field(..., description="Second drug name")
    severity: Literal["none", "mild", "moderate", "severe"] = Field(..., description="Interaction severity level")
    mechanism: str = Field(..., description="Pharmacological mechanism of the interaction")


class CitedFinding(BaseModel):
    claim: str = Field(..., description="The clinical claim or finding")
    source_title: str = Field(..., description="Title of the source publication")
    source_url: str | None = Field(default=None, description="URL to the source if available")
    paragraph_excerpt: str = Field(..., description="Relevant excerpt from the source")
    evidence_level: Literal["RCT", "meta-analysis", "cohort", "case-report", "expert-opinion"] = Field(
        ..., description="Level of evidence"
    )


class AuditReport(BaseModel):
    challenges_raised: int = Field(..., ge=0, description="Number of challenges raised by auditor")
    challenges_answered: int = Field(..., ge=0, description="Number of challenges answered satisfactorily")
    unanswered: list[str] = Field(default_factory=list, description="List of unanswered challenge descriptions")
    verdict: Literal["pass", "fail"] = Field(..., description="Overall audit verdict")


class CMOSelfReport(BaseModel):
    diagnosis_confidence: float = Field(..., ge=0.0, le=1.0)
    evidence_strength: float = Field(..., ge=0.0, le=1.0)
    contraindication_risk: float = Field(..., ge=0.0, le=1.0)
    differential_completeness: float = Field(..., ge=0.0, le=1.0)


class InterAgentAgreement(BaseModel):
    pathologist_differentials: list[str] = Field(default_factory=list)
    literature_differentials: list[str] = Field(default_factory=list)
    biosensor_differentials: list[str] = Field(default_factory=list)
    agreement_score: float = Field(default=0.0, ge=0.0, le=1.0)


class CitationDensitySignal(BaseModel):
    citation_count: int = Field(default=0, ge=0)
    recency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    rct_ratio: float = Field(default=0.0, ge=0.0, le=1.0)


class AdversarialSignal(BaseModel):
    challenges_raised: int = Field(default=0, ge=0)
    challenges_answered: int = Field(default=0, ge=0)


class ConfidenceSignals(BaseModel):
    cmo_self_report: CMOSelfReport
    inter_agent_agreement: InterAgentAgreement
    citation_density: CitationDensitySignal
    adversarial: AdversarialSignal


class ClinicalReport(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    patient_token: str
    differentials: list[DifferentialDiagnosis] = Field(default_factory=list)
    drug_interactions: list[DrugInteractionCheck] = Field(default_factory=list)
    cited_findings: list[CitedFinding] = Field(default_factory=list)
    audit_report: AuditReport | None = None
    confidence_signals: ConfidenceSignals | None = None
    confidence_score: float = Field(default=0.0, ge=0.0, le=100.0)
    hitl_required: bool = False
    escalate_immediately: bool = False
    biosensor_summary: str = ""
    synthesis_narrative: str = ""
    loop_count: int = 0


class PatientRecord(BaseModel):
    patient_token: str = Field(..., description="Tokenized patient identifier")
    age_range: str | None = Field(default=None, description="Age range (not exact DOB)")
    notes: str | None = Field(default=None, description="General notes, no PII")


class HITLOverride(BaseModel):
    session_id: str = Field(..., description="Session ID to override")
    doctor_id: str = Field(..., description="Identifier of the overriding physician")
    override_decision: str = Field(..., description="Doctor's override decision narrative")
    revised_diagnosis: str | None = Field(default=None, description="Revised primary diagnosis if changed")
    approved: bool = Field(..., description="Whether the AI recommendation is approved")
