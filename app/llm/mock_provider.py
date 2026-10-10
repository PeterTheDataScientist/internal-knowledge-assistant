"""Offline provider.

The service has to be runnable with no API key, both for anyone trying it and for CI. This
provider does extractive answering over the same context blocks the real model sees and
returns the same JSON contract, so every layer above it is exercised identically. It is
deliberately conservative: when no sentence overlaps the question well enough it reports
insufficient context, which is the behaviour the pipeline must handle correctly anyway.
"""

from __future__ import annotations

import json
import re
import time
from typing import Dict, List, Optional, Tuple

from app.indexing.lexical_index import tokenize
from app.llm.base import LLMProvider, LLMResponse

BLOCK_RE = re.compile(
    r"^\[(\d+)\]\s+source=(?P<source>\S+)\s*\|\s*section=(?P<section>[^|]*)\|",
    re.MULTILINE,
)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
QUANTITY_QUESTION_RE = re.compile(
    r"\b(how\s+(many|much|long|often)|what\s+(is|are)\s+the\s+(maximum|minimum|limit|rate|"
    r"ratio|threshold|tenure|period))\b",
    re.IGNORECASE,
)


class MockProvider(LLMProvider):
    name = "mock"
    model = "offline-extractive-v1"

    def __init__(self, min_overlap: float = 0.28) -> None:
        self.min_overlap = min_overlap

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        started = time.perf_counter()
        question = self._extract_question(user_prompt)
        blocks = self._extract_blocks(user_prompt)
        payload = self._answer(question, blocks)
        return LLMResponse(
            text=json.dumps(payload),
            model=self.model,
            prompt_tokens=len(user_prompt) // 4,
            completion_tokens=len(payload.get("answer", "")) // 4,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    @staticmethod
    def _extract_question(prompt: str) -> str:
        match = re.search(r"QUESTION:\s*(.+?)(?:\n\s*\n|$)", prompt, re.DOTALL)
        return match.group(1).strip() if match else prompt.strip()

    @staticmethod
    def _extract_blocks(prompt: str) -> List[Dict[str, str]]:
        blocks: List[Dict[str, str]] = []
        matches = list(BLOCK_RE.finditer(prompt))
        # The last block runs to the start of the question, not to the end of the prompt.
        # Reading past that point pulled the question itself into the context and let the
        # offline provider answer a question with the question.
        markers = (prompt.find("\nQUESTION:"), prompt.find("\nCONVERSATION SO FAR"))
        tail = min((position for position in markers if position != -1), default=len(prompt))
        for position, match in enumerate(matches):
            start = match.end()
            end = matches[position + 1].start() if position + 1 < len(matches) else tail
            body = prompt[start:end]
            # Drop the remainder of the provenance line, then the "title > section"
            # heading the indexer prepends, so the extracted answer is document prose
            # rather than a heading followed by a sentence.
            body = body.split("\n", 1)[1] if "\n" in body else body
            first, _, rest = body.partition("\n")
            if rest and (" > " in first or len(first.split()) <= 8):
                body = rest
            blocks.append(
                {
                    "index": match.group(1),
                    "source": match.group("source"),
                    "text": body.strip(),
                }
            )
        return blocks

    def _answer(self, question: str, blocks: List[Dict[str, str]]) -> Dict[str, object]:
        if not blocks:
            return {"sufficient": False, "answer": "", "citations": []}

        question_terms = set(tokenize(question))
        if not question_terms:
            return {"sufficient": False, "answer": "", "citations": []}

        scored: List[Tuple[float, str, int]] = []
        for block in blocks:
            for sentence in SENTENCE_RE.split(block["text"].replace("\n", " ")):
                cleaned = sentence.strip(" -*\t")
                if len(cleaned) < 20:
                    continue
                terms = set(tokenize(cleaned))
                if not terms:
                    continue
                overlap = len(question_terms & terms) / len(question_terms)
                scored.append((overlap, cleaned, int(block["index"])))

        # A question asking for a quantity is only answered by a sentence containing one.
        # Without this the extractive baseline replies to "how many branches does Contoso
        # operate" with the most word-similar sentence it can find, which lists the
        # subsidiaries and answers nothing.
        if QUANTITY_QUESTION_RE.search(question):
            scored = [item for item in scored if any(char.isdigit() for char in item[1])]

        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored or scored[0][0] < self.min_overlap:
            return {"sufficient": False, "answer": "", "citations": []}

        picked = scored[:2]
        citations = sorted({index for _, _, index in picked})
        answer = " ".join(sentence for _, sentence, _ in picked)
        return {"sufficient": True, "answer": answer, "citations": citations}

    def health(self) -> Dict[str, str]:
        return {"status": "ok", "provider": self.name, "model": self.model}
