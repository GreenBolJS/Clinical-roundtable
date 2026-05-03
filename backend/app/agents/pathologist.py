from __future__ import annotations
import json
import structlog
from app.providers.router import get_router
from app.schemas.clinical import DifferentialDiagnosis

logger = structlog.get_logger(__name__)

PATHOLOGIST_SYSTEM_PROMPT = """You are a senior diagnostic pathologist with 20 years of experience in internal medicine. 
Your job is to analyze patient symptoms and history and produce a ranked differential diagnosis.
Rules:
- Always list 3-5 conditions ranked by probability (highest first)
- Probability must reflect actual clinical likelihood based on symptom patterns
- ICD-10 codes must be specific (e.g. I21.4 not I21)
- Reasoning must cite specific symptoms that support OR argue against each diagnosis
- Never include conditions with probability below 0.1
- Always consider age, gender, medications, and comorbidities
- Output ONLY valid JSON matching the schema. No markdown, no explanation.

You MUST respond with ONLY valid JSON — no markdown, no explanation, no code fences.

Response schema (a JSON array of objects):
[
  {
    "condition": "Name of the condition",
    "icd10_code": "ICD-10 code (e.g., J18.9)",
    "probability": 0.75,
    "reasoning": "Detailed clinical reasoning supporting this diagnosis, referencing specific symptoms and history"
  }
]

Rules:
- Provide 3 to 6 differential diagnoses, ordered by probability (highest first)
- probability must be a float between 0.0 and 1.0
- icd10_code must be a valid ICD-10-CM code
- reasoning must be specific, citing the patient's symptoms and history
- Consider both common and serious/must-not-miss diagnoses
- Do NOT include any text outside the JSON array
"""


async def run_pathologist(
    symptoms: list[str],
    history: str,
    current_medications: list[str],
) -> list[DifferentialDiagnosis]:
    """
    Diagnostic Pathologist node: generate differential diagnoses.
    """
    router = get_router()

    prompt = f"""Patient Clinical Presentation:

Symptoms: {', '.join(symptoms)}

Medical History:
{history}

Current Medications: {', '.join(current_medications) if current_medications else 'None'}

Generate a ranked differential diagnosis list for this patient."""

    raw = await router.call_agent(
        agent_name="pathologist",
        prompt=prompt,
        system_prompt=PATHOLOGIST_SYSTEM_PROMPT,
        max_tokens=2048,
    )

    parsed = _parse_differentials(raw)
    if parsed is None:
        logger.warning("pathologist_json_parse_failed", raw=raw[:200])
        correction_prompt = (
            "Your previous response could not be parsed as JSON. "
            "Respond ONLY with a raw JSON array matching the schema, no markdown fences, no explanation.\n\n"
            f"Original request:\n{prompt}"
        )
        raw2 = await router.call_agent(
            agent_name="pathologist",
            prompt=correction_prompt,
            system_prompt=PATHOLOGIST_SYSTEM_PROMPT,
            max_tokens=2048,
        )
        parsed = _parse_differentials(raw2)
        if parsed is None:
            logger.error("pathologist_json_retry_failed")
            return []

    result = []
    for item in parsed:
        try:
            result.append(DifferentialDiagnosis(**item))
        except Exception as exc:
            logger.warning("pathologist_schema_validation_error", error=str(exc), item=item)

    logger.info("pathologist_complete", differential_count=len(result))
    return result


def _parse_differentials(raw: str) -> list[dict] | None:
    """Try to parse a JSON array from raw text."""
    raw = raw.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        # Sometimes model wraps in {"differentials": [...]}
        if isinstance(data, dict):
            for key in ("differentials", "diagnoses", "differential_diagnosis", "results"):
                if key in data and isinstance(data[key], list):
                    return data[key]
    except json.JSONDecodeError:
        pass
    return None
