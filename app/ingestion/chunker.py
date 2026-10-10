"""Chunking.

The corpus is short structured markdown, so the chunker splits on headings first and
only falls back to a sliding window when a section is longer than the target size. That
keeps each chunk about one topic, which is what makes the citations readable: a chunk
maps to a heading a human can find in the source file.

Bullet lists are kept with the paragraph that introduces them, because several policy
facts in this corpus live in a bullet directly under a lead-in sentence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from app.ingestion.loader import Document, TrustTier

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    title: str
    section: str
    text: str
    domain: str
    trust: TrustTier
    position: int
    token_estimate: int
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def is_trusted(self) -> bool:
        return self.trust is TrustTier.AUTHORITATIVE

    def as_context_block(self, index: int) -> str:
        """Render the chunk for the prompt with an explicit provenance header."""
        trust_label = "AUTHORITATIVE POLICY" if self.is_trusted else "UNTRUSTED USER CONTENT"
        return (
            f"[{index}] source={self.source} | section={self.section} | trust={trust_label}\n"
            f"{self.text}"
        )


def estimate_tokens(text: str) -> int:
    """Cheap token estimate. Good enough for budgeting without a tokeniser dependency."""
    return max(1, len(text) // 4)


def _split_sections(text: str) -> List[tuple[str, str]]:
    """Return (section_heading, body) pairs in document order."""
    sections: List[tuple[str, List[str]]] = []
    current_heading = "Introduction"
    current_body: List[str] = []

    for line in text.splitlines():
        match = HEADING_RE.match(line.strip())
        if match:
            if current_body:
                sections.append((current_heading, current_body))
            current_heading = match.group(2).strip() or current_heading
            current_body = []
        else:
            current_body.append(line)

    if current_body:
        sections.append((current_heading, current_body))

    return [(heading, "\n".join(body).strip()) for heading, body in sections if "".join(body).strip()]


def _split_paragraphs(body: str) -> List[str]:
    """Split on blank lines, then glue bullet blocks onto the paragraph above them."""
    blocks = [block.strip() for block in re.split(r"\n\s*\n", body) if block.strip()]
    merged: List[str] = []
    for block in blocks:
        is_list = all(
            line.strip().startswith(("-", "*", "+")) or re.match(r"^\d+\.", line.strip())
            for line in block.splitlines()
            if line.strip()
        )
        if is_list and merged:
            merged[-1] = f"{merged[-1]}\n{block}"
        else:
            merged.append(block)
    return merged


def _window(text: str, size: int, overlap: int) -> List[str]:
    """Sentence aware sliding window for sections that exceed the target chunk size."""
    sentences = SENTENCE_RE.split(text)
    windows: List[str] = []
    current: List[str] = []
    length = 0

    for sentence in sentences:
        sentence_length = len(sentence) + 1
        if length + sentence_length > size and current:
            windows.append(" ".join(current).strip())
            carry: List[str] = []
            carried = 0
            for previous in reversed(current):
                if carried >= overlap:
                    break
                carry.insert(0, previous)
                carried += len(previous) + 1
            current = carry
            length = carried
        current.append(sentence)
        length += sentence_length

    if current:
        windows.append(" ".join(current).strip())
    return [w for w in windows if w]


def chunk_document(document: Document, chunk_size: int, chunk_overlap: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    position = 0

    for heading, body in _split_sections(document.text):
        buffer = ""
        for paragraph in _split_paragraphs(body):
            candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph

            if len(candidate) <= chunk_size:
                buffer = candidate
                continue

            if buffer:
                chunks.append(_make_chunk(document, heading, buffer, position))
                position += 1
                buffer = ""

            if len(paragraph) <= chunk_size:
                buffer = paragraph
            else:
                for window in _window(paragraph, chunk_size, chunk_overlap):
                    chunks.append(_make_chunk(document, heading, window, position))
                    position += 1

        if buffer:
            chunks.append(_make_chunk(document, heading, buffer, position))
            position += 1

    if not chunks:
        chunks.append(_make_chunk(document, document.title, document.text, 0))

    return chunks


def _make_chunk(document: Document, section: str, text: str, position: int) -> Chunk:
    body = text.strip()
    # The heading is prepended to the indexed text so that a question phrased in the
    # words of the heading, for example "sanctions screening", still matches a chunk
    # whose body never repeats those words.
    indexed_text = (
        f"{document.title} > {section}\n{body}"
        if section != document.title
        else f"{document.title}\n{body}"
    )
    return Chunk(
        chunk_id=f"{document.source}#{position}",
        doc_id=document.doc_id,
        source=document.source,
        title=document.title,
        section=section,
        text=indexed_text,
        domain=document.domain,
        trust=document.trust,
        position=position,
        token_estimate=estimate_tokens(indexed_text),
    )


def chunk_documents(
    documents: Iterable[Document], chunk_size: int, chunk_overlap: int
) -> List[Chunk]:
    chunks: List[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, chunk_size, chunk_overlap))
    return chunks
