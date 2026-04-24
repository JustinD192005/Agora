
"""Shared configuration loaded from environment variables."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM providers
    gemini_api_key: str
    groq_api_key: str = "unused"
    tavily_api_key: str = "unused"

    # Infrastructure
    database_url: str
    redis_url: str


@lru_cache
def get_settings() -> Settings:
    return Settings()

