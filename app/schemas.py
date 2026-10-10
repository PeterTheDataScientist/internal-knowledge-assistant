"""Request and response contracts for the public API."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import get_settings

MAX_QUESTION_LENGTH = get_settings().max_question_length


class Confidence(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    NONE = "None"


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        ...,
        min_length=3,
        max_length=MAX_QUESTION_LENGTH,
        description="Natural language question about Contoso Financial internal documentation.",
        examples=["How many leave days does an employee receive?"],
    )
    session_id: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Optional conversation identifier. When supplied, the previous turns "
        "of that conversation are used to resolve follow-up questions.",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Override the number of chunks passed to the model.",
    )

    @field_validator("question")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must contain non-whitespace characters")
        return cleaned

    @field_validator("session_id")
    @classmethod
    def _validate_session(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            return None
        if not all(char.isalnum() or char in "-_" for char in cleaned):
            raise ValueError("session_id may only contain letters, digits, hyphen and underscore")
        return cleaned


class Citation(BaseModel):
    """A single supporting passage, returned so a reader can verify the answer."""

    source: str = Field(..., description="Source file name, for example hr-leave-policy.md")
    chunk_id: str
    section: Optional[str] = Field(default=None, description="Heading the passage was taken from.")
    score: float = Field(..., description="Fused retrieval score, higher is more relevant.")
    excerpt: str = Field(..., description="Trimmed passage used to support the answer.")


class SafetyReport(BaseModel):
    input_flagged: bool = False
    context_sanitised: bool = False
    output_flagged: bool = False
    triggered_rules: List[str] = Field(default_factory=list)
    injection_score: Optional[float] = Field(
        default=None, description="Prompt Guard probability that the input is an injection."
    )


class AskResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "answer": "Permanent employees receive 22 working days of annual leave per calendar year.",
                "sources": ["hr-leave-policy.md"],
                "confidence": "High",
                "answered_from_context": True,
            }
        }
    )

    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: Confidence
    answered_from_context: bool
    refusal_reason: Optional[str] = Field(
        default=None,
        description="Why the assistant declined, when it declined. The answer text is the "
        "same in every refusal case so that a probing caller cannot learn which control "
        "fired; this field carries the detail for operators.",
    )
    citations: List[Citation] = Field(default_factory=list)
    safety: SafetyReport = Field(default_factory=SafetyReport)
    request_id: str
    latency_ms: int
    model: str
    session_id: Optional[str] = None


class HealthComponent(BaseModel):
    status: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    uptime_seconds: float
    documents: int
    chunks: int
    embedding_backend: str
    llm_provider: str
    llm_model: str
    components: Dict[str, HealthComponent]


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(..., min_length=1, max_length=64)
    helpful: bool
    comment: Optional[str] = Field(default=None, max_length=1000)


class FeedbackResponse(BaseModel):
    recorded: bool
    request_id: str


class MetricsResponse(BaseModel):
    requests_total: int
    refusals_total: int
    errors_total: int
    injection_blocks_total: int
    confidential_blocks_total: int
    refusal_rate: float
    refusal_reasons: Dict[str, int]
    latency_ms_avg: float
    latency_ms_p50: float
    latency_ms_p95: float
    confidence_counts: Dict[str, int]
    feedback_positive: int
    feedback_negative: int


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: Optional[str] = None
    fields: Optional[List[Dict[str, str]]] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
