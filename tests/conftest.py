"""Shared fixtures.

Tests run against the offline provider so they are deterministic and need no API key.
The knowledge base is built once for the session because building it is the slow part.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("USE_MODEL_GUARD", "false")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("EMBEDDING_BACKEND", "tfidf")

from app.config import Settings, get_settings  # noqa: E402
from app.llm.mock_provider import MockProvider  # noqa: E402
from app.rag.knowledge_base import build_knowledge_base  # noqa: E402
from app.rag.pipeline import RAGPipeline  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="session")
def settings() -> Settings:
    return get_settings()


@pytest.fixture(scope="session")
def knowledge_base(settings: Settings):
    return build_knowledge_base(settings, data_dir=DATA_DIR)


@pytest.fixture(scope="session")
def pipeline(settings: Settings, knowledge_base) -> RAGPipeline:
    return RAGPipeline(settings, knowledge_base, MockProvider())


@pytest.fixture(scope="session")
def client(pipeline):
    from fastapi.testclient import TestClient

    from app.main import create_app

    app = create_app()
    with TestClient(app) as test_client:
        # The lifespan builds its own pipeline. Replacing it keeps the API tests on the
        # deterministic offline provider regardless of what is in the environment.
        app.state.pipeline = pipeline
        yield test_client
