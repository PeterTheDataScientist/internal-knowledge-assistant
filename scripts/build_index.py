"""Builds the index once and prints what was ingested.

Useful as a pre-flight check before starting the service, and as the place where an
index build would be wired into a CI job.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.observability.logging_config import configure_logging  # noqa: E402
from app.rag.knowledge_base import build_knowledge_base  # noqa: E402


def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level, "text")

    knowledge_base = build_knowledge_base(settings)

    print(f"documents        {knowledge_base.document_count}")
    print(f"chunks           {knowledge_base.chunk_count}")
    print(f"embedder         {knowledge_base.embedder.name} ({knowledge_base.embedder.dimension}d)")
    print(f"vector backend   {knowledge_base.vector_index.backend}")
    print(f"build time       {knowledge_base.build_seconds:.2f}s")

    per_source = Counter(chunk.source for chunk in knowledge_base.chunks)
    print("\nchunks per document")
    for source, count in sorted(per_source.items()):
        document = next(d for d in knowledge_base.documents if d.source == source)
        print(f"  {count:>2}  {source:<34} {document.domain:<22} {document.trust.value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
