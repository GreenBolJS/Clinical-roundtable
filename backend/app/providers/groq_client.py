from __future__ import annotations
import time
import json
import structlog
from groq import AsyncGroq
from app.config import get_settings

logger = structlog.get_logger(__name__)

settings = get_settings()

_groq_client: AsyncGroq | None = None


def get_groq_client() -> AsyncGroq:
    global _groq_client
    if _groq_client is None:
        _groq_client = AsyncGroq(api_key=settings.groq_api_key)
    return _groq_client


GROQ_MODELS = {
    "heavy": "llama-3.3-70b-versatile",
    "light": "llama-3.1-8b-instant",
}


async def call_groq(
    agent_name: str,
    prompt: str,
    system_prompt: str,
    model_tier: str = "heavy",
    max_tokens: int = 4096,
) -> tuple[str, int, float]:
    """
    Call Groq API.
    Returns (response_text, tokens_used, latency_ms).
    """
    client = get_groq_client()
    model = GROQ_MODELS.get(model_tier, GROQ_MODELS["heavy"])
    start = time.monotonic()
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        elapsed_ms = (time.monotonic() - start) * 1000
        content = response.choices[0].message.content or ""
        tokens_used = response.usage.total_tokens if response.usage else 0
        logger.info(
            "groq_call",
            agent_name=agent_name,
            provider="groq",
            model=model,
            tokens_used=tokens_used,
            latency_ms=round(elapsed_ms, 2),
            success=True,
        )
        return content, tokens_used, elapsed_ms
    except Exception as exc:
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.error(
            "groq_call_failed",
            agent_name=agent_name,
            provider="groq",
            model=model,
            tokens_used=0,
            latency_ms=round(elapsed_ms, 2),
            success=False,
            error=str(exc),
        )
        raise


async def check_groq_health() -> bool:
    try:
        client = get_groq_client()
        response = await client.chat.completions.create(
            model=GROQ_MODELS["light"],
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        return True
    except Exception:
        return False
