from __future__ import annotations
import json
import structlog
from app.providers.router import get_router
from app.schemas.clinical import CitedFinding
from app.vector.chroma import query_documents

logger = structlog.get_logger(__name__)

LITERATURE_SYSTEM_PROMPT = """You are a Clinical Literature Critic AI specializing in evidence-based medicine. Your role is to identify and evaluate the highest-quality medical evidence relevant to a clinical case, AND to select the top differential diagnosis supported by literature.

You will be provided with:
1. The patient's clinical summary
2. A list of working differentials from the pathologist
3. Retrieved chunks from medical journals (if available)

CRITICAL REQUIREMENT - DIFFERENTIAL SELECTION:
You MUST always return at least one condition in literature_differentials. Pick the condition from the pathologist's differential list that has the strongest published evidence base. Use the EXACT same condition name string as provided in the pathologist's list. Never return an empty list.

Your response MUST include two sections in valid JSON:
1. The top differential(s) from the pathologist's list with strongest evidence
2. Cited findings supporting that differential

Response schema (a JSON object):
{
  "literature_differentials": ["exact_condition_name_from_pathologist_list"],
  "cited_findings": [
    {
      "claim": "The specific clinical claim or finding supported by this citation",
      "source_title": "Title of the source publication",
      "source_url": null,
      "paragraph_excerpt": "Relevant excerpt (1-3 sentences) from the source supporting the claim",
      "evidence_level": "RCT|meta-analysis|cohort|case-report|expert-opinion"
    }
  ]
}

Rules:
- evidence_level must be exactly one of: RCT, meta-analysis, cohort, case-report, expert-opinion
- Prioritize higher levels of evidence (RCT > meta-analysis > cohort > case-report > expert-opinion)
- Each claim should be directly relevant to the selected differential
- If retrieved journal chunks are provided, use them to ground your citations
- If no retrieved chunks are available, generate evidence from your medical knowledge but mark source_url as null
- Provide 3 to 8 cited findings
- ALWAYS populate literature_differentials with exactly 1 condition name from the pathologist's list
- Do NOT include any text outside the JSON object
- Do NOT invent condition names; use the EXACT strings from the pathologist's differential list
"""

# Top differentials to extract for agreement scoring
_COMMON_CONDITIONS = {
    "pneumonia", "influenza", "covid-19", "myocardial infarction", "heart failure",
    "pulmonary embolism", "sepsis", "appendicitis", "cholecystitis", "pancreatitis",
    "stroke", "tia", "hypertensive crisis", "anemia", "diabetes",
    "asthma", "copd", "atrial fibrillation", "deep vein thrombosis",
    "meningitis", "encephalitis", "urinary tract infection", "pyelonephritis",
    "gastroenteritis", "irritable bowel syndrome", "celiac disease", "crohn's disease",
}


async def run_literature(
    symptoms: list[str],
    history: str,
    differentials: list[str],
) -> tuple[list[CitedFinding], list[str]]:
    """
    Literature Critic node: retrieve evidence from ChromaDB then query Gemini.
    Returns (cited_findings, top_conditions_mentioned).
    """
    router = get_router()

    # Build retrieval query
    query_text = f"{' '.join(symptoms[:5])} {' '.join(differentials[:3])}"
    retrieved_chunks = query_documents(query_text=query_text, n_results=5)

    chunks_text = ""
    if retrieved_chunks:
        chunks_list = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            meta = chunk.get("metadata", {})
            title = meta.get("title", f"Source {i}")
            chunks_list.append(f"[Source {i}: {title}]\n{chunk['text']}")
        chunks_text = "\n\n".join(chunks_list)
        chunks_text = f"\n\nRetrieved Medical Literature Chunks:\n{chunks_text}"

    prompt = f"""Clinical Case Summary:
Symptoms: {', '.join(symptoms)}
Medical History: {history}
Working Differentials: {', '.join(differentials) if differentials else 'Not yet established'}
{chunks_text}

Identify and evaluate the strongest medical evidence relevant to this case. Select the top differential from the pathologist's list and provide cited findings with evidence levels."""

    raw = await router.call_agent(
        agent_name="literature",
        prompt=prompt,
        system_prompt=LITERATURE_SYSTEM_PROMPT,
        max_tokens=3072,
    )

    parsed_data = _parse_literature_response(raw)
    if parsed_data is None:
        logger.warning("literature_json_parse_failed", raw=raw[:200])
        correction_prompt = (
            "Your previous response could not be parsed as JSON. "
            "Respond ONLY with a raw JSON object (not array) matching the schema, no markdown fences, no explanation.\n\n"
            f"Original request:\n{prompt}"
        )
        raw2 = await router.call_agent(
            agent_name="literature",
            prompt=correction_prompt,
            system_prompt=LITERATURE_SYSTEM_PROMPT,
            max_tokens=3072,
        )
        parsed_data = _parse_literature_response(raw2)
        if parsed_data is None:
            logger.error("literature_json_retry_failed")
            return [], []

    findings_list = parsed_data.get("cited_findings", [])
    literature_differentials = parsed_data.get("literature_differentials", [])

    result = []
    for item in findings_list:
        try:
            result.append(CitedFinding(**item))
        except Exception as exc:
            logger.warning("literature_schema_validation_error", error=str(exc), item=item)

    # Use literature_differentials if provided by model, otherwise extract from findings
    if literature_differentials:
        top_conditions = literature_differentials
    else:
        top_conditions = _extract_conditions_from_findings(result)

    logger.info("literature_complete", finding_count=len(result), literature_differentials=top_conditions)
    return result, top_conditions


def _parse_literature_response(raw: str) -> dict | None:
    """Parse literature agent response containing literature_differentials and cited_findings."""
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


def _extract_conditions_from_findings(findings: list[CitedFinding]) -> list[str]:
    """Extract condition names from claims for inter-agent agreement scoring."""
    conditions = []
    for finding in findings:
        claim_lower = finding.claim.lower()
        for condition in _COMMON_CONDITIONS:
            if condition in claim_lower:
                conditions.append(condition)
    return list(set(conditions))[:5]
