from __future__ import annotations
import time
import asyncio
import structlog
import google.generativeai as genai
from app.config import get_settings

logger = structlog.get_logger(__name__)

settings = get_settings()

genai.configure(api_key=settings.gemini_api_key)
GEMINI_MODEL = "gemini-2.0-flash"


async def call_gemini(
    agent_name: str,
    prompt: str,
    system_prompt: str,
    max_tokens: int = 4096,
) -> tuple[str, int, float]:
    """
    Call Gemini Flash API.
    Returns (response_text, tokens_used, latency_ms).
    """
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=system_prompt if system_prompt else None,
    )
    start = time.monotonic()
    response = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: model.generate_content(prompt),
    )
    elapsed_ms = (time.monotonic() - start) * 1000
    content = response.text or ""
    tokens_used = 0
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        tokens_used = getattr(response.usage_metadata, "total_token_count", 0)
    logger.info(
        "gemini_call",
        agent_name=agent_name,
        provider="gemini",
        model=GEMINI_MODEL,
        tokens_used=tokens_used,
        latency_ms=round(elapsed_ms, 2),
        success=True,
    )
    return content, tokens_used, elapsed_ms


async def check_gemini_health() -> bool:
    model = genai.GenerativeModel(model_name=GEMINI_MODEL)
    response = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: model.generate_content("ping"),
    )
    return bool(response.text)
