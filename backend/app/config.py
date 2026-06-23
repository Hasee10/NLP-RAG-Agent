"""Typed application settings, loaded from the repo-root .env."""

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py -> repo root is two parents up from this file's dir
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Supabase ──
    supabase_url: str
    supabase_service_role_key: str
    supabase_anon_key: str = ""

    # ── OpenRouter (LLM) ──
    openrouter_api_key: str = ""
    openrouter_model: str = "anthropic/claude-3.5-sonnet"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # ── Embeddings (own trained encoder) ──
    embedding_provider: str = "own_encoder"
    embedding_dim: int = 128

    # ── App ──
    app_env: str = "development"
    allowed_origins: str = "http://localhost:3000"

    # Paths to the trained encoder + vocab (reused from the NLP pipeline).
    @property
    def encoder_path(self) -> Path:
        return REPO_ROOT / "models" / "encoder_best.pt"

    @property
    def vocab_path(self) -> Path:
        return REPO_ROOT / "data" / "vocab.json"

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
