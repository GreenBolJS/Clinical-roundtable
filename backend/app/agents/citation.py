from __future__ import annotations
import json
import structlog
from app.providers.router import get_router
from app.schemas.clinical import CitedFinding, DifferentialDiagnosis, DrugInteractionCheck

logger = structlog.get_logger(__name__)

CITATION_SYSTEM_PROMPT = """You are a Medical Citation Enrichment Agent. Your role is to review a set of clinical findings and enrich them with proper citations, linking each finding to the strongest available evidence source.

You will receive a list of existing cited findings and a list of differential diagnoses and drug interactions that may need additional citation support.

You MUST respond with ONLY valid JSON — no markdown, no explanation, no code fences.

Response schema (a JSON array of CitedFinding objects):
[
  {
    "claim": "The specific clinical claim or finding",
    "source_title": "Title of the source publication",
    "source_url": "https://pubmed.ncbi.nlm.nih.gov/?term=<search+terms>",
    "paragraph_excerpt": "Relevant supporting excerpt (1-3 sentences)",
    "evidence_level": "RCT|meta-analysis|cohort|case-report|expert-opinion"
  }
]

Rules:
- Always generate a realistic PubMed search URL for each claim using format: https://pubmed.ncbi.nlm.nih.gov/?term=<search+terms>
- Set evidence_level based on the type of study being cited — never default everything to expert-opinion
- You MUST return between 8 and 14 citations. Never return fewer than 8.
- Each differential diagnosis must have at least 2 citations supporting or refuting it.
- Each drug interaction flagged as moderate or severe must have at least 1 citation.
- Each citation must be directly tied to a specific differential diagnosis in the report
- Preserve and improve existing citations where possible
- Add new citations for any differentials or drug interactions not yet cited
- Prefer RCTs and meta-analyses over lower levels of evidence
- Do NOT duplicate claims already well-cited
- Return the complete enriched list (both old and new findings)
- Do NOT include any text outside the JSON array
"""


async def run_citation(
    existing_findings: list[CitedFinding],
    differentials: list[DifferentialDiagnosis],
    drug_interactions: list[DrugInteractionCheck],
) -> list[CitedFinding]:
    """
    Citation Agent node: enrich and link all findings to sources.
    """
    router = get_router()

    existing_json = json.dumps(
        [f.model_dump() for f in existing_findings],
        indent=2,
    )
    differentials_text = "\n".join(
        f"- {d.condition} ({d.icd10_code}): {d.reasoning[:100]}..."
        for d in differentials
    )
    interactions_text = "\n".join(
        f"- {i.drug_a} + {i.drug_b} [{i.severity}]: {i.mechanism}"
        for i in drug_interactions
    )

    prompt = f"""Existing Cited Findings:
{existing_json}

Differential Diagnoses requiring citation support:
{differentials_text if differentials_text else 'None'}

Drug Interactions requiring citation support:
{interactions_text if interactions_text else 'None'}

Enrich the citation list, ensuring every differential and significant drug interaction has supporting evidence. Return the complete updated list."""

    raw = await router.call_agent(
        agent_name="citation",
        prompt=prompt,
        system_prompt=CITATION_SYSTEM_PROMPT,
        max_tokens=3072,
    )

    parsed = _parse_findings(raw)
    if parsed is None:
        logger.warning("citation_json_parse_failed", raw=raw[:200])
        correction_prompt = (
            "Your previous response could not be parsed as JSON. "
            "Respond ONLY with a raw JSON array matching the schema, no markdown fences, no explanation.\n\n"
            f"Original request:\n{prompt}"
        )
        raw2 = await router.call_agent(
            agent_name="citation",
            prompt=correction_prompt,
            system_prompt=CITATION_SYSTEM_PROMPT,
            max_tokens=3072,
        )
        parsed = _parse_findings(raw2)
        if parsed is None:
            logger.error("citation_json_retry_failed")
            return existing_findings  # Fall back to existing findings

    result = []
    for item in parsed:
        try:
            result.append(CitedFinding(**item))
        except Exception as exc:
            logger.warning("citation_schema_validation_error", error=str(exc), item=item)

    logger.info("citation_complete", finding_count=len(result))
    return result


def _parse_findings(raw: str) -> list[dict] | None:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("findings", "cited_findings", "citations", "results"):
                if key in data and isinstance(data[key], list):
                    return data[key]
    except json.JSONDecodeError:
        pass
    return None
