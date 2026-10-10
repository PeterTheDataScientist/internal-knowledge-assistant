"""Runtime configuration, loaded from environment variables or a .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, List, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ----- service -----
    app_name: str = "Internal Knowledge Assistant"
    app_version: str = "1.0.0"
    environment: Literal["local", "dev", "staging", "prod"] = "local"
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"

    # ----- corpus and index -----
    data_dir: Path = PROJECT_ROOT / "data"
    index_dir: Path = PROJECT_ROOT / ".index"
    chunk_size: int = Field(default=450, ge=200, le=4000)
    chunk_overlap: int = Field(default=80, ge=0, le=1000)

    # ----- embeddings -----
    # "auto" prefers the sentence-transformer model and silently falls back to the
    # dependency-free TF-IDF/SVD embedder when the model cannot be loaded.
    embedding_backend: Literal["auto", "sentence_transformer", "tfidf"] = "auto"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim_fallback: int = 256

    # ----- retrieval -----
    retrieval_top_k: int = Field(default=6, ge=1, le=20)
    candidate_pool: int = Field(default=20, ge=1, le=100)
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    # The usual value is 60, which is tuned for long result lists. This corpus produces
    # short ones, where 60 flattens the difference between rank 1 and rank 10.
    rrf_k: int = Field(default=20, ge=1)
    min_relevance_score: float = Field(default=0.28, ge=0.0, le=1.0)
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
    use_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ----- llm -----
    llm_provider: Literal["auto", "groq", "mock"] = "auto"
    # NoDecode keeps pydantic-settings from trying to JSON parse these, so they can be
    # supplied as ordinary comma separated environment variables.
    groq_api_keys: Annotated[List[str], NoDecode] = Field(default_factory=list)
    groq_model: str = "llama-3.3-70b-versatile"
    groq_fast_model: str = "llama-3.1-8b-instant"
    guard_model: str = "meta-llama/llama-prompt-guard-2-86m"
    use_model_guard: bool = True
    guard_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=700, ge=64, le=8000)
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    llm_max_retries: int = Field(default=3, ge=0, le=8)

    # ----- api -----
    max_question_length: int = Field(default=1000, ge=10, le=8000)
    rate_limit_per_minute: int = Field(default=30, ge=1)
    request_timeout_seconds: float = Field(default=45.0, gt=0)
    api_keys: Annotated[List[str], NoDecode] = Field(default_factory=list)
    require_api_key: bool = False
    cors_origins: Annotated[List[str], NoDecode] = Field(default_factory=lambda: ["*"])

    # ----- conversation memory -----
    memory_max_turns: int = Field(default=6, ge=0, le=20)
    memory_ttl_seconds: int = Field(default=1800, ge=60)
    memory_max_sessions: int = Field(default=1000, ge=1)

    @field_validator("groq_api_keys", "api_keys", "cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Accept either a JSON list or a comma separated string from the environment."""
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            if stripped.startswith("["):
                return value
            return [item.strip() for item in stripped.split(",") if item.strip()]
        return value

    @property
    def refusal_message(self) -> str:
        """The exact wording the assistant must use when it cannot answer."""
        return "I don't have sufficient information in the provided documents."


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
