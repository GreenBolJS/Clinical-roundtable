from __future__ import annotations
import json
import structlog
from app.providers.router import get_router

logger = structlog.get_logger(__name__)

BIOSENSOR_SYSTEM_PROMPT = """You are a clinical biosensor analyst. Given real-time patient vitals, you must:
- Identify which of the provided differential diagnoses the vitals SUPPORT or CONTRADICT
- CRITICAL: Your biosensor_differentials list must contain condition names taken EXACTLY from the pathologist_differentials list provided in the input. Do not invent new condition names. Copy the exact strings from the pathologist list that the vitals support.
- biosensor_differentials must use EXACTLY the same condition names as the pathologist used
- Flag any vital sign outside normal range with clinical significance
- Strict normal ranges — only flag if OUTSIDE these ranges:
  - Heart Rate: 60-100 bpm (flag if <60 or >100)
  - Blood Pressure systolic: 90-120 mmHg (flag if <90 or >140)
  - Blood Pressure diastolic: 60-80 mmHg (flag if <60 or >90)
  - Oxygen Saturation: >95% is NORMAL, only flag if ≤95%
  - Temperature: 36.5-37.5°C (flag if >37.8 as fever)
  - Respiratory Rate: 12-20 breaths/min (flag if <12 or >20)
  - Blood Glucose: 70-110 mg/dL fasting (flag if <70 or >126)
  - Pain Scale: 0-3 normal, 4-6 moderate, 7-10 severe
- Never flag a value as abnormal if it falls within the normal range above
- Output ONLY valid JSON. No markdown.

You are a Bio-Sensor Data Analyst AI specializing in interpreting wearable device data, continuous monitoring streams, and physiological sensor readings in clinical contexts.

Your role is to parse raw biosensor data and produce a structured clinical interpretation.

You MUST respond with ONLY valid JSON — no markdown, no explanation, no code fences.

Response schema:
{
  "summary": "2-3 sentence clinical summary of the biosensor findings",
  "abnormal_values": [
    {
      "parameter": "Parameter name (e.g., Heart Rate, SpO2)",
      "value": "The recorded value with units",
      "normal_range": "Expected normal range",
      "clinical_significance": "Brief clinical significance of this abnormality"
    }
  ],
  "trend_analysis": "1-2 sentences about any trends or patterns observed",
  "suggested_differentials": ["condition1", "condition2"],
  "alert_level": "normal|watch|warning|critical"
}

Rules:
- alert_level must be exactly one of: normal, watch, warning, critical
- abnormal_values should only include values outside normal clinical ranges
- suggested_differentials should reflect what the biosensor data suggests
- If biosensor_data is empty or null, return a minimal response with alert_level: normal
- Do NOT include any text outside the JSON object
"""


async def run_biosensor(
    biosensor_data: dict | None,
    pathologist_differentials: list[str] | None = None,
) -> tuple[str, list[str]]:
    """
    Bio-Sensor Streamer node: parse biosensor data into structured vitals summary.
    Returns (summary_text, suggested_differentials).
    """
    if not biosensor_data:
        return "No biosensor data provided.", []

    router = get_router()

    differential_text = ""
    if pathologist_differentials:
        differential_text = (
            "\nThe pathologist has identified these differential diagnoses:\n"
            + json.dumps(pathologist_differentials, indent=2)
            + "\n"
        )

    prompt = f"""Patient vitals: {json.dumps(biosensor_data)}

{differential_text}
From this EXACT list above, select which conditions the vital signs support. Copy the condition names EXACTLY as written above. Do not create new names.

Parse this biosensor data and provide a structured clinical interpretation with any abnormalities and their significance."""

    raw = await router.call_agent(
        agent_name="biosensor",
        prompt=prompt,
        system_prompt=BIOSENSOR_SYSTEM_PROMPT,
        max_tokens=1024,
    )

    parsed = _parse_biosensor_response(raw)
    if parsed is None:
        logger.warning("biosensor_json_parse_failed", raw=raw[:200])
        correction_prompt = (
            "Your previous response could not be parsed as JSON. "
            "Respond ONLY with a raw JSON object matching the schema, no markdown fences, no explanation.\n\n"
            f"Original request:\n{prompt}"
        )
        raw2 = await router.call_agent(
            agent_name="biosensor",
            prompt=correction_prompt,
            system_prompt=BIOSENSOR_SYSTEM_PROMPT,
            max_tokens=1024,
        )
        parsed = _parse_biosensor_response(raw2)
        if parsed is None:
            logger.error("biosensor_json_retry_failed")
            return "Biosensor data analysis failed.", []

    summary = parsed.get("summary", "")
    trend = parsed.get("trend_analysis", "")
    abnormals = parsed.get("abnormal_values", [])
    alert_level = parsed.get("alert_level", "normal")
    differentials = parsed.get("suggested_differentials", [])

    # Build a human-readable summary
    summary_parts = [f"[{alert_level.upper()}] {summary}"]
    if trend:
        summary_parts.append(f"Trend: {trend}")
    if abnormals:
        summary_parts.append("Abnormal Values:")
        for ab in abnormals:
            summary_parts.append(
                f"  - {ab.get('parameter', 'Unknown')}: {ab.get('value', 'N/A')} "
                f"(Normal: {ab.get('normal_range', 'N/A')}) — {ab.get('clinical_significance', '')}"
            )

    full_summary = "\n".join(summary_parts)
    logger.info("biosensor_complete", alert_level=alert_level, abnormal_count=len(abnormals))
    return full_summary, differentials


def _parse_biosensor_response(raw: str) -> dict | None:
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
