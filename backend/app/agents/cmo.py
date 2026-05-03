from __future__ import annotations
import json
import structlog
from app.providers.router import get_router
from app.schemas.clinical import (
    ClinicalReport,
    DifferentialDiagnosis,
    DrugInteractionCheck,
    CitedFinding,
    CMOSelfReport,
)

logger = structlog.get_logger(__name__)

CMO_SYSTEM_PROMPT = """You are the Chief Medical Officer synthesizing findings from multiple specialist agents.
Rules:
- Your synthesis_narrative must be 3-5 sentences, clinically precise, no fluff
- confidence signals must be honest — if evidence is weak, score it low
- diagnosis_confidence above 0.85 only if top diagnosis has probability > 0.7 AND literature supports it
- differential_completeness above 0.85 only if 4+ conditions are listed with distinct reasoning
- Always integrate biosensor data if provided
- Output ONLY valid JSON. No markdown.

You are the Chief Medical Officer (CMO) AI — the senior clinical decision-maker responsible for synthesizing all specialist inputs into a coherent clinical report.

You receive inputs from: Diagnostic Pathologist, PharmacogenomicsSpecialist, Literature Critic, Bio-Sensor Analyst, and Citation Agent.

You MUST respond with ONLY valid JSON — no markdown, no explanation, no code fences.

Response schema:
{
  "synthesis_narrative": "A comprehensive 3-5 paragraph clinical synthesis narrative integrating all specialist inputs. Include the most likely diagnosis, supporting evidence, drug interaction concerns, and recommended next steps.",
  "primary_diagnosis": "The single most likely diagnosis based on all evidence",
  "confidence_self_report": {
    "diagnosis_confidence": 0.82,
    "evidence_strength": 0.75,
    "contraindication_risk": 0.15,
    "differential_completeness": 0.88
  }
}

Confidence self-report field definitions:
- diagnosis_confidence: How confident you are in the primary diagnosis (0.0 to 1.0)
- evidence_strength: Strength of the supporting evidence overall (0.0 to 1.0)
- contraindication_risk: Risk level of contraindications / dangerous interactions (0.0 to 1.0, will be INVERTED in scoring so high risk = lower confidence)
- differential_completeness: How complete and thorough the differential is (0.0 to 1.0)

Rules:
- All confidence values must be floats between 0.0 and 1.0
- synthesis_narrative must integrate ALL specialist inputs
- Be clinically precise and use medical terminology appropriately
- Explicitly reference drug interactions if severity is moderate or severe
- Do NOT include any text outside the JSON object
"""


async def run_cmo(
    patient_token: str,
    symptoms: list[str],
    history: str,
    differentials: list[DifferentialDiagnosis],
    drug_interactions: list[DrugInteractionCheck],
    cited_findings: list[CitedFinding],
    biosensor_summary: str,
    loop_count: int = 0,
    previous_audit_feedback: str = "",
) -> tuple[str, CMOSelfReport]:
    """
    CMO node: synthesize all agent outputs into a clinical narrative.
    Returns (synthesis_narrative, confidence_self_report).
    """
    router = get_router()

    differentials_text = "\n".join(
        f"  {i+1}. {d.condition} ({d.icd10_code}) — {d.probability:.0%} — {d.reasoning}"
        for i, d in enumerate(differentials)
    )
    interactions_text = "\n".join(
        f"  - {i.drug_a} + {i.drug_b}: {i.severity.upper()} — {i.mechanism}"
        for i in drug_interactions
    ) or "None identified"

    findings_text = "\n".join(
        f"  [{f.evidence_level}] {f.claim} — {f.source_title}"
        for f in cited_findings[:8]
    ) or "No findings available"

    audit_section = ""
    if previous_audit_feedback and loop_count > 0:
        audit_section = f"\n\nAudit Feedback (Loop {loop_count} — please address these issues):\n{previous_audit_feedback}"

    prompt = f"""Patient: {patient_token}
Symptoms: {', '.join(symptoms)}
Medical History: {history}

DIFFERENTIAL DIAGNOSES (from Pathologist):
{differentials_text if differentials_text else 'Not available'}

DRUG INTERACTIONS (from PharmacogenomicsSpecialist):
{interactions_text}

BIOSENSOR FINDINGS:
{biosensor_summary if biosensor_summary else 'No biosensor data'}

EVIDENCE-BASED LITERATURE (from Citation Agent):
{findings_text}
{audit_section}

Synthesize all specialist inputs into a comprehensive clinical report and provide your confidence self-assessment."""

    raw = await router.call_agent(
        agent_name="cmo",
        prompt=prompt,
        system_prompt=CMO_SYSTEM_PROMPT,
        max_tokens=3072,
    )

    parsed = _parse_cmo_response(raw)
    if parsed is None:
        logger.warning("cmo_json_parse_failed", raw=raw[:200])
        correction_prompt = (
            "Your previous response could not be parsed as JSON. "
            "Respond ONLY with a raw JSON object matching the schema, no markdown fences, no explanation.\n\n"
            f"Original request:\n{prompt}"
        )
        raw2 = await router.call_agent(
            agent_name="cmo",
            prompt=correction_prompt,
            system_prompt=CMO_SYSTEM_PROMPT,
            max_tokens=3072,
        )
        parsed = _parse_cmo_response(raw2)
        if parsed is None:
            logger.error("cmo_json_retry_failed")
            return "Clinical synthesis unavailable.", CMOSelfReport(
                diagnosis_confidence=0.5,
                evidence_strength=0.5,
                contraindication_risk=0.5,
                differential_completeness=0.5,
            )

    narrative = parsed.get("synthesis_narrative", "")
    self_report_raw = parsed.get("confidence_self_report", {})

    try:
        self_report = CMOSelfReport(**self_report_raw)
    except Exception:
        self_report = CMOSelfReport(
            diagnosis_confidence=0.5,
            evidence_strength=0.5,
            contraindication_risk=0.5,
            differential_completeness=0.5,
        )

    logger.info("cmo_complete", loop_count=loop_count)
    return narrative, self_report


def _parse_cmo_response(raw: str) -> dict | None:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    return None
