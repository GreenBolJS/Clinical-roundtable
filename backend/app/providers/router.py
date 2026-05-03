from __future__ import annotations
import asyncio
from app.config import get_settings
from app.providers.groq_client import call_groq

settings = get_settings()

# Agents that use Groq heavy (70b)
GROQ_HEAVY_AGENTS = {
    "pathologist",
    "pharmaco",
    "cmo",
    "auditor",
    "literature",
}

# Agents that use Groq light (8b)
GROQ_LIGHT_AGENTS = {
    "triage",
    "biosensor",
    "citation",
}


class ProviderRouter:
    """
    Single unified async interface for all agent LLM calls.
    Enforces rate-limiting semaphores per provider.
    """

    def __init__(self) -> None:
        self._groq_semaphore = asyncio.Semaphore(settings.groq_max_concurrent)

    async def call_agent(
        self,
        agent_name: str,
        prompt: str,
        system_prompt: str,
        max_tokens: int = 4096,
    ) -> str:
        """
        Route an agent call to the appropriate provider and model.
        Returns the text response.
        """
        if agent_name in GROQ_HEAVY_AGENTS:
            async with self._groq_semaphore:
                text, _, _ = await call_groq(
                    agent_name=agent_name,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model_tier="heavy",
                    max_tokens=max_tokens,
                )
            return text

        elif agent_name in GROQ_LIGHT_AGENTS:
            async with self._groq_semaphore:
                text, _, _ = await call_groq(
                    agent_name=agent_name,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model_tier="light",
                    max_tokens=max_tokens,
                )
            return text

        else:
            # Default to Groq heavy for any unrecognised agent
            async with self._groq_semaphore:
                text, _, _ = await call_groq(
                    agent_name=agent_name,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model_tier="heavy",
                    max_tokens=max_tokens,
                )
            return text


# Module-level singleton
_router: ProviderRouter | None = None


def get_router() -> ProviderRouter:
    global _router
    if _router is None:
        _router = ProviderRouter()
    return _router
