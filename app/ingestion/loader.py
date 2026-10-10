"""Corpus loading.

Every document is tagged with a trust tier at ingestion time. The corpus deliberately
contains a shared notice board whose entries are written by staff rather than by the
policy owners, and one of those entries is an injection attempt. Marking that document
as untrusted at ingestion is cheaper and far more reliable than trying to detect the
injection later in the prompt.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}

# Documents whose content is written by end users rather than by a policy owner.
UNTRUSTED_MARKERS = (
    "user-contributed",
    "user generated",
    "user-generated",
    "notice board",
    "not** official policy",
    "not official policy",
    "posted by staff",
)

# Documents that only describe the corpus itself and carry no policy content.
META_MARKERS = ("corpus notice", "created to test the assistant")

DOMAIN_BY_PREFIX = {
    "hr-": "human_resources",
    "aml-": "compliance",
    "credit-": "credit",
    "product-": "products",
    "sop-": "operations",
    "faq-": "faq",
    "data-": "information_governance",
    "information-": "information_security",
    "group-": "corporate",
    "staff-": "notices",
}


class TrustTier(str, Enum):
    AUTHORITATIVE = "authoritative"
    UNTRUSTED = "untrusted"
    META = "meta"


@dataclass(frozen=True)
class Document:
    doc_id: str
    source: str
    path: Path
    title: str
    text: str
    domain: str
    trust: TrustTier
    checksum: str
    loaded_at: str
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def is_trusted(self) -> bool:
        return self.trust is TrustTier.AUTHORITATIVE


def _title_of(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return fallback


def _domain_of(source: str) -> str:
    for prefix, domain in DOMAIN_BY_PREFIX.items():
        if source.startswith(prefix):
            return domain
    return "general"


def _trust_of(text: str) -> TrustTier:
    """Classify a document by who wrote it.

    The meta check runs first on purpose. The corpus notice happens to mention the phrase
    "user-generated content" while describing the corpus, and matching on that would
    misfile it as a notice board.
    """
    lowered = text.lower()
    if any(marker in lowered for marker in META_MARKERS):
        return TrustTier.META
    if any(marker in lowered for marker in UNTRUSTED_MARKERS):
        return TrustTier.UNTRUSTED
    return TrustTier.AUTHORITATIVE


def _normalise(text: str) -> str:
    """Collapse hard wrapping so that facts split across two lines stay searchable."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents(data_dir: Path) -> List[Document]:
    """Read every supported file in ``data_dir`` and return them sorted by file name."""
    if not data_dir.exists():
        raise FileNotFoundError(f"corpus directory not found: {data_dir}")

    documents: List[Document] = []
    for path in sorted(data_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            raw = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("skipping file that is not valid utf-8", extra={"source": path.name})
            continue

        text = _normalise(raw)
        if not text:
            logger.warning("skipping empty document", extra={"source": path.name})
            continue

        trust = _trust_of(text)
        document = Document(
            doc_id=hashlib.sha256(path.name.encode()).hexdigest()[:12],
            source=path.name,
            path=path,
            title=_title_of(text, path.stem),
            text=text,
            domain=_domain_of(path.name),
            trust=trust,
            checksum=hashlib.sha256(text.encode()).hexdigest(),
            loaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        documents.append(document)

    if not documents:
        raise FileNotFoundError(f"no readable documents in {data_dir}")

    untrusted = [d.source for d in documents if d.trust is TrustTier.UNTRUSTED]
    logger.info(
        "corpus loaded",
        extra={"documents": len(documents), "untrusted_documents": untrusted},
    )
    return documents
