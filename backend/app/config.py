from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    groq_api_key: str
    gemini_api_key: str
    database_url: str
    redis_url: str = "redis://localhost:6379"
    hitl_webhook_url: str = "http://localhost:8000/webhook/hitl-response"
    chroma_persist_path: str = "./chroma_db"
    hitl_threshold: float = 85.0
    escalate_threshold: float = 70.0
    max_loop_count: int = 1
    groq_max_concurrent: int = 2
    gemini_max_concurrent: int = 1
    max_retries: int = 3
    timeout: int = 30
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
