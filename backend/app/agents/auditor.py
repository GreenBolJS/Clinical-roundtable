from __future__ import annotations
import json
import structlog
from app.providers.router import get_router
from app.schemas.clinical import AuditReport, DifferentialDiagnosis, DrugInteractionCheck

logger = structlog.get_logger(__name__)

AUDITOR_SYSTEM_PROMPT = """You are a Chief Medical Officer conducting a peer review of a clinical report.
Your job is to find genuine clinical gaps — not generic concerns.
Rules:
- Raise MAXIMUM 4 challenges
- Each challenge must reference a SPECIFIC finding in the report
- Do not raise challenges about things not in scope (e.g. don't ask for MRI if patient just arrived)
- challenges_answered must reflect how many of your challenges the CMO report actually addresses
- If the report is clinically sound, raise 0-2 challenges and return verdict "pass"
- Only return verdict "fail" if there are unanswered safety-critical gaps
- Output ONLY valid JSON. No markdown.

You are an Adversarial Clinical Auditor AI. Your role is to rigorously challenge a CMO's clinical synthesis, identify logical gaps, unsupported claims, missed diagnoses, and dangerous oversights.

You must act as a devil's advocate — find every weakness in the clinical reasoning.

You MUST respond with ONLY valid JSON — no markdown, no explanation, no code fences.

Response schema:
{
  "challenges_raised": 4,
  "challenges_answered": 2,
  "unanswered": [
    "Why was pulmonary embolism not included in the differential given the patient's immobility?",
    "The drug interaction between metformin and contrast agents was not addressed"
  ],
  "verdict": "pass|fail",
  "audit_narrative": "A concise 2-3 sentence summary of the audit findings"
}

Field definitions:
- challenges_raised: Total number of clinical challenges you identified
- challenges_answered: Number of those challenges that were adequately addressed in the CMO synthesis
- unanswered: List of specific unanswered challenge descriptions (strings)
- verdict: "pass" if challenges_answered / challenges_raised >= 0.75, else "fail"
- audit_narrative: Summary of findings

Audit checklist to apply:
1. Are all serious/must-not-miss diagnoses included in the differential?
2. Are all significant drug interactions addressed?
3. Is the evidence quality adequate for the claims made?
4. Are there logical inconsistencies in the reasoning?
5. Are there contraindications not flagged?
6. Is the biosensor data properly integrated?
7. Are there missing workup recommendations?

Rules:
- Be rigorous but fair
- Specific challenges must be actionable, not vague
- If synthesis is excellent with no gaps, challenges_raised can be 0 with verdict "pass"
- Do NOT include any text outside the JSON object
"""


async def run_auditor(
    synthesis_narrative: str,
    differentials: list[DifferentialDiagnosis],
    drug_interactions: list[DrugInteractionCheck],
    symptoms: list[str],
    history: str,
) -> AuditReport:
    """
    Adversarial Auditor node: challenge the CMO synthesis.
    Returns AuditReport.
    """
    router = get_router()

    differentials_text = "\n".join(
        f"  {i+1}. {d.condition} ({d.icd10_code}) — {d.probability:.0%}"
        for i, d in enumerate(differentials)
    )
    interactions_text = "\n".join(
        f"  - {i.drug_a} + {i.drug_b}: {i.severity.upper()}"
        for i in drug_interactions
    ) or "None"

    prompt = f"""CMO Clinical Synthesis to Audit:
{synthesis_narrative}

Supporting Data:
Symptoms: {', '.join(symptoms)}
History: {history}

Differential Diagnoses:
{differentials_text if differentials_text else 'None provided'}

Drug Interactions:
{interactions_text}

Perform a rigorous adversarial audit of this clinical synthesis. Identify all clinical gaps, unsupported claims, and dangerous oversights."""

    raw = await router.call_agent(
        agent_name="auditor",
        prompt=prompt,
        system_prompt=AUDITOR_SYSTEM_PROMPT,
        max_tokens=2048,
    )

    parsed = _parse_audit_response(raw)
    if parsed is None:
        logger.warning("auditor_json_parse_failed", raw=raw[:200])
        correction_prompt = (
            "Your previous response could not be parsed as JSON. "
            "Respond ONLY with a raw JSON object matching the schema, no markdown fences, no explanation.\n\n"
            f"Original request:\n{prompt}"
        )
        raw2 = await router.call_agent(
            agent_name="auditor",
            prompt=correction_prompt,
            system_prompt=AUDITOR_SYSTEM_PROMPT,
            max_tokens=2048,
        )
        parsed = _parse_audit_response(raw2)
        if parsed is None:
            logger.error("auditor_json_retry_failed")
            return AuditReport(
                challenges_raised=0,
                challenges_answered=0,
                unanswered=[],
                verdict="pass",
            )

    try:
        report = AuditReport(
            challenges_raised=parsed.get("challenges_raised", 0),
            challenges_answered=parsed.get("challenges_answered", 0),
            unanswered=parsed.get("unanswered", []),
            verdict=parsed.get("verdict", "pass"),
        )
    except Exception as exc:
        logger.warning("auditor_schema_validation_error", error=str(exc))
        report = AuditReport(
            challenges_raised=parsed.get("challenges_raised", 0),
            challenges_answered=parsed.get("challenges_answered", 0),
            unanswered=parsed.get("unanswered", []),
            verdict="fail" if parsed.get("challenges_raised", 0) > 0 else "pass",
        )

    logger.info(
        "auditor_complete",
        challenges_raised=report.challenges_raised,
        challenges_answered=report.challenges_answered,
        verdict=report.verdict,
    )
    return report


def _parse_audit_response(raw: str) -> dict | None:
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
