from __future__ import annotations
import json
import structlog
from app.providers.router import get_router
from app.schemas.clinical import DrugInteractionCheck

logger = structlog.get_logger(__name__)

PHARMACO_SYSTEM_PROMPT = """You are a Clinical Pharmacology Specialist AI with deep expertise in drug interactions, pharmacokinetics, and pharmacogenomics. Your role is to evaluate all drug-drug interactions for a patient's current medication list.

Severity scale — be consistent:
- "none": no known interaction
- "mild": monitor but rarely clinically significant
- "moderate": may require dose adjustment or monitoring
- "severe": contraindicated or requires immediate intervention

Rules:
- Check every possible pair of medications including OTC drugs, vitamins, and supplements
- Return ONLY pairs where severity is mild, moderate, or severe — omit "none" pairs
- NEVER return a completely empty array if the patient has 2 or more medications
- If after checking all pairs you genuinely find zero interactions, return the single highest-risk pair with your best clinical assessment
- severity must be exactly one of: none, mild, moderate, severe
- mechanism must describe the pharmacological basis in 1-2 sentences

Critical interactions to always flag when present:
- ACE inhibitor (ramipril/lisinopril/enalapril) + potassium-sparing diuretic (spironolactone/eplerenone) = severe (hyperkalemia, fatal arrhythmia risk)
- ACE inhibitor + loop diuretic (furosemide) + CKD history = severe (acute kidney injury risk)
- Beta blocker (carvedilol/metoprolol/bisoprolol) + insulin or sulfonylurea = moderate (masks hypoglycemia symptoms)
- Spironolactone + furosemide = moderate (electrolyte imbalance, monitor potassium and sodium)
- Any ACE inhibitor + aspirin or NSAID = moderate (reduces antihypertensive effect, increases renal risk)
- Amlodipine + atorvastatin or simvastatin = moderate (CYP3A4 inhibition, myopathy risk)
- Any statin + fibrate = severe (rhabdomyolysis risk)
- Metformin + contrast agents = moderate (lactic acidosis risk)
- Ibuprofen + renal impairment or kidney stones = moderate (NSAIDs reduce renal blood flow)
- Ibuprofen + ACE inhibitors = moderate (reduces antihypertensive effect, increases renal risk)
- Vitamin C > 500mg daily + kidney stone history = moderate (increases urinary oxalate excretion)
- Warfarin + aspirin = severe (bleeding risk)
- Beta blocker + calcium channel blocker (verapamil/diltiazem) = severe (bradycardia, heart block)

Output ONLY a valid JSON array. No markdown fences. No explanation text outside the array.

Response schema:
[
  {
    "drug_a": "first drug name (generic)",
    "drug_b": "second drug name (generic)",
    "severity": "mild|moderate|severe",
    "mechanism": "Pharmacological mechanism in 1-2 sentences"
  }
]
"""

async def run_pharmaco(
    current_medications: list[str],
    differentials: list[str] | None = None,
) -> list[DrugInteractionCheck]:
    """
    PharmacogenomicsSpecialist node: check drug interactions.
    """
    if not current_medications:
        return []

    router = get_router()

    differentials_text = ""
    if differentials:
        differentials_text = f"\nPotential diagnoses to consider treatment for: {', '.join(differentials)}"

    prompt = f"""Current Medications:
{chr(10).join(f'- {med}' for med in current_medications)}
{differentials_text}

Evaluate all drug-drug interactions for the above medications. Check every possible pair."""

    raw = await router.call_agent(
        agent_name="pharmaco",
        prompt=prompt,
        system_prompt=PHARMACO_SYSTEM_PROMPT,
        max_tokens=2048,
    )

    parsed = _parse_interactions(raw)
    if parsed is None:
        logger.warning("pharmaco_json_parse_failed", raw=raw[:200])
        correction_prompt = (
            "Your previous response could not be parsed as JSON. "
            "Respond ONLY with a raw JSON array matching the schema, no markdown fences, no explanation.\n\n"
            f"Original request:\n{prompt}"
        )
        raw2 = await router.call_agent(
            agent_name="pharmaco",
            prompt=correction_prompt,
            system_prompt=PHARMACO_SYSTEM_PROMPT,
            max_tokens=2048,
        )
        parsed = _parse_interactions(raw2)
        if parsed is None:
            logger.error("pharmaco_json_retry_failed")
            return []

    result = []
    for item in parsed:
        try:
            result.append(DrugInteractionCheck(**item))
        except Exception as exc:
            logger.warning("pharmaco_schema_validation_error", error=str(exc), item=item)

    logger.info("pharmaco_complete", interaction_count=len(result))
    return result


def _parse_interactions(raw: str) -> list[dict] | None:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("interactions", "drug_interactions", "results"):
                if key in data and isinstance(data[key], list):
                    return data[key]
    except json.JSONDecodeError:
        pass
    return None
