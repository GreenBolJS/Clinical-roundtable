from __future__ import annotations
import json
import structlog
from app.providers.router import get_router

logger = structlog.get_logger(__name__)

TRIAGE_SYSTEM_PROMPT = """You are a Clinical Triage Coordinator AI. Your role is to analyze incoming patient data and determine which specialist agents are required for a comprehensive clinical evaluation.

Available specialists:
- pathologist: Diagnostic Pathologist for differential diagnosis
- pharmaco: PharmacogenomicsSpecialist for drug interactions and pharmacogenomics
- literature: Literature Critic for evidence-based medicine review
- biosensor: Bio-Sensor Analyzer for wearable/sensor data interpretation

You MUST respond with ONLY valid JSON — no markdown, no explanation, no code fences.

Response schema:
{
  "active_agents": ["pathologist", "pharmaco", "literature", "biosensor"],
  "pharmaco_priority": "low|normal|high",
  "triage_reasoning": "Brief explanation of which agents are needed and why",
  "urgency_level": "routine|urgent|emergent",
  "primary_concern": "Single sentence describing the main clinical concern"
}

Rules:
- Always include pathologist and literature in active_agents
- If biosensor_data is None or empty, skip biosensor_node
- If current_medications has fewer than 2 drugs, set pharmaco_priority to low
- urgency_level must be one of: routine, urgent, emergent
- CRITICAL ROUTING RULE: If the patient has ANY medications listed in current_medications (including OTC drugs, vitamins, supplements), you MUST always include "pharmaco" in the active_agents list. Never omit pharmaco when medications exist.
"""


async def run_triage(
    symptoms: list[str],
    history: str,
    current_medications: list[str],
    has_biosensor: bool,
) -> dict:
    """
    Triage node: decide which specialist agents are needed.
    Returns dict with keys: active_agents, triage_reasoning, urgency_level, primary_concern.
    """
    router = get_router()

    med_count = len(current_medications or [])
    pharmaco_priority = "low" if med_count < 2 else "normal"

    prompt = f"""Patient presentation:
Symptoms: {', '.join(symptoms)}
Medical History: {history}
Current Medications: {', '.join(current_medications) if current_medications else 'None'}
Biosensor Data Available: {has_biosensor}

Determine which specialist agents are needed for this case. Include pathologist and literature, decide whether pharmaco is required, and only include biosensor if data is available."""

    raw = await router.call_agent(
        agent_name="triage",
        prompt=prompt,
        system_prompt=TRIAGE_SYSTEM_PROMPT,
        max_tokens=512,
    )

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("triage_json_parse_failed", raw=raw[:200])
        # Retry with explicit correction
        correction_prompt = (
            f"Your previous response could not be parsed as JSON. "
            f"Respond ONLY with raw JSON, no markdown fences, no explanation.\n\n"
            f"Original request:\n{prompt}"
        )
        raw2 = await router.call_agent(
            agent_name="triage",
            prompt=correction_prompt,
            system_prompt=TRIAGE_SYSTEM_PROMPT,
            max_tokens=512,
        )
        try:
            result = json.loads(raw2)
        except json.JSONDecodeError as exc:
            logger.error("triage_json_retry_failed", error=str(exc))
            # Fallback: include all agents
            result = {
                "active_agents": ["pathologist", "pharmaco", "literature", "biosensor"],
                "triage_reasoning": "Fallback: all agents activated due to parse failure",
                "urgency_level": "routine",
                "primary_concern": "Unable to determine from triage",
            }

    # Ensure required keys exist
    if "active_agents" not in result:
        result["active_agents"] = ["pathologist", "pharmaco", "literature"]
    if "urgency_level" not in result:
        result["urgency_level"] = "routine"
    if "primary_concern" not in result:
        result["primary_concern"] = "Unspecified"
    if "triage_reasoning" not in result:
        result["triage_reasoning"] = ""
    if "pharmaco_priority" not in result:
        result["pharmaco_priority"] = pharmaco_priority

    # Always ensure pathologist and literature are included
    active = set(result["active_agents"])
    active.add("pathologist")
    active.add("literature")
    if not has_biosensor:
        active.discard("biosensor")
    if med_count == 0:
        active.discard("pharmaco")
    result["active_agents"] = list(active)

    logger.info("triage_complete", active_agents=result["active_agents"], urgency=result["urgency_level"], pharmaco_priority=result["pharmaco_priority"])
    return result
