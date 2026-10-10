"""Deterministic pattern rules used by the guardrails.

These run before and after the model. They are cheap, they are testable, and unlike a
model they cannot be talked out of their behaviour. The model based guard adds recall on
top of them, it does not replace them.
"""

from __future__ import annotations

import re
from typing import List, Pattern, Tuple

# Attempts to override the system instruction or extract it.
INJECTION_RULES: List[Tuple[str, Pattern[str]]] = [
    ("ignore_prior_instructions", re.compile(
        r"(?:^|[.!?\n]\s*|\b(?:please|now|then|and|but)\s+)"
        r"(ignore|disregard|forget|override|bypass|discard)\b[^.]{0,40}\b"
        r"(previous|prior|above|earlier|all|any|your|the|everything)\b[^.]{0,30}\b"
        r"(instruction|instructions|prompt|prompts|rule|rules|guideline|guidelines|context|"
        r"directive|told|said|policy)",
        re.IGNORECASE)),
    ("system_prompt_extraction", re.compile(
        r"\b(reveal|show|print|repeat|output|display|what\s+(is|are|were))\b[^.]{0,40}\b"
        r"(system\s*(prompt|message|instruction)|initial\s*instruction|your\s*(prompt|instructions|rules))",
        re.IGNORECASE)),
    ("role_override", re.compile(
        r"\b(you\s+are\s+(now|no\s+longer)|from\s+now\s+on\s+you|pretend\s+to\s+be|"
        r"roleplay\s+as|role[-\s]?play\s+as|new\s+persona|you\s+have\s+no\s+"
        r"(restrictions|rules|guidelines))\b"
        r"|(?:^|[.!?\n]\s*)(act|behave|respond)\s+as\s+(if\s+you|an?\s+unrestricted)",
        re.IGNORECASE)),
    ("developer_mode", re.compile(
        r"\b(developer\s*mode|dan\s*mode|jailbreak|god\s*mode|unrestricted\s*mode|"
        r"sudo\s*mode|unfiltered\s*mode|an\s+unrestricted\s+(ai|assistant|model|bot))\b",
        re.IGNORECASE)),
    ("guardrail_disable", re.compile(
        r"\b(no|without|disable|turn\s*off|remove|drop)\b[^.]{0,25}\b"
        r"(restriction|restrictions|filter|filters|guardrail|guardrails|safety|censorship|limitation)",
        re.IGNORECASE)),
    ("fake_authority", re.compile(
        r"\b(i\s+am\s+(the\s+)?(ceo|admin|administrator|developer|your\s+creator|head\s+of\s+it)|"
        r"authorised\s+by\s+(management|compliance)\s+to\s+(see|access|receive))\b",
        re.IGNORECASE)),
    ("encoded_payload", re.compile(
        r"\b(base64|rot13|hex\s*decode|decode\s+(this|the\s+following))\b",
        re.IGNORECASE)),
]

# Requests for information the assistant must never produce, regardless of retrieval.
CONFIDENTIAL_REQUEST_RULES: List[Tuple[str, Pattern[str]]] = [
    ("customer_data_request", re.compile(
        r"\b(tell|show|give|send|list|provide|fetch|display|reveal|share|disclose|retrieve|"
        r"email|forward|print|export|download|pull\s+up|look\s+up|i\s+need|i\s+want|i\s+require|"
        r"what\s+is|what's|whats|how\s+much\s+is)\b[^?]{0,60}\b"
        r"(account\s+(balance|balances|number|numbers|details|holder|holders)|"
        r"customer\s+(balance|balances|data|details|record|records|information|list)|"
        r"balance\s+of\s+account|client\s+(data|records|list)|"
        r"(account|customer|client)\s+holders?\b|their\s+balances\b)",
        re.IGNORECASE)),
    ("credential_request", re.compile(
        r"\b(give|tell|show|send|list|provide|reveal|share|disclose|retrieve|email|forward|"
        r"i\s+need|i\s+want|i\s+require|what\s+is|what's|whats)\b"
        r"[^?]{0,40}\b(my|the|his|her|their|a|an|customer|customer's|client's|user's|staff|"
        r"admin|supervisor|branch|another\s+\w+)\b[^?]{0,30}\b"
        r"(pins?|passwords?|otps?|one[-\s]?time[-\s]?passwords?|credentials?|"
        r"api\s*keys?|private\s*keys?|encryption\s*keys?|cryptographic\s*keys?|master\s*keys?|"
        r"secrets?)\b",
        re.IGNORECASE)),
    ("credential_reset_request", re.compile(
        r"\b(reset|change|set)\b[^?]{0,30}\b(pin|password|otp)\b[^?]{0,20}\bto\b",
        re.IGNORECASE)),
    ("bulk_extraction", re.compile(
        r"\b(dump|export|extract|download|list\s+all|email\s+me|send\s+me)\b[^?]{0,40}\b"
        r"(all\s+)?(customer|account|client|user)s?\b",
        re.IGNORECASE)),
]

# Attributes of a control rather than the secret itself. "What is the minimum password
# length" is a question about the Information Security Policy, not a request for anybody's
# password, and refusing it would be a false positive that makes the assistant useless.
POLICY_ATTRIBUTE_RE = re.compile(
    r"\b(minimum|maximum|at\s+least|length|characters|complexity|expiry|expiration|"
    r"how\s+often|how\s+long\s+(must|are|is)|requirement|requirements|standard|standards|"
    r"threshold|thresholds|retention|retained|retain|kept\s+for|must\s+be|should\s+be|"
    r"are\s+required|is\s+required|reset\s+my|self-service|locked\s+out|who\s+approves|"
    r"who\s+can\s+approve|classified\s+as|disciplinary)\b",
    re.IGNORECASE)

# Framings that look like data requests but are legitimate policy questions. An employee
# asking whether they may share a balance is asking about the Code of Conduct, and the
# assistant should answer that from the documents rather than refuse.
POLICY_QUESTION_RULES: List[Pattern[str]] = [
    re.compile(r"^\s*(can|may|am\s+i|are\s+we|is\s+it|should\s+i|should\s+employees|"
               r"do\s+i\s+need|who\s+(approves|can\s+approve|is\s+allowed)|"
               r"what\s+(is|are)\s+the\s+(policy|policies|rule|rules|procedure|process|"
               r"classification|requirement|requirements))\b", re.IGNORECASE),
]

# Markers of a request for a specific record rather than a rule. When one of these is present
# the softener does not apply, however the question is dressed up.
DATA_TARGET_RE = re.compile(
    r"(\b\d{6,}\b|\baccount\s+(number|numbers)\b|\ball\s+(customer|account|client)s?\b|"
    r"\bfor\s+(customer|client|account)\s+(?-i:[A-Z0-9])|\bbelonging\s+to\b|\blist\s+all\b|"
    r"\bshow\s+me\b|\bgive\s+me\b|\bemail\s+me\b|\bsend\s+me\b|"
    r"\b(core\s+banking|admin|supervisor|branch)\s+(password|credentials|pin)\b|"
    r"\bpassword\s+for\s+the\b|\bcredentials\s+for\b)",
    re.IGNORECASE)

# Patterns that must never appear in an outgoing answer.
OUTPUT_LEAK_RULES: List[Tuple[str, Pattern[str]]] = [
    ("account_number_like", re.compile(r"\b\d{8,}\b")),
    ("card_number_like", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("credential_disclosure", re.compile(
        r"\b(pin|password|otp)\s*(is|:|=)\s*[\w!@#$%^&*]{3,}", re.IGNORECASE)),
    ("system_prompt_echo", re.compile(
        r"(you are the contoso|system prompt:|my instructions are|core rules:)", re.IGNORECASE)),
]

# Imperative sentences inside retrieved content are neutralised before they reach the model.
CONTEXT_INSTRUCTION_RE = re.compile(
    r"(?im)^[^\n]*\b("
    r"ignore\s+(previous|prior|all|any)|disregard\s+(previous|prior|all|any)|"
    r"you\s+(must|should|may)\s+(now\s+)?(share|disclose|reveal|state|say|tell)|"
    r"new\s+instructions?\s*:|system\s*:\s|override\s+the\s+(policy|instructions)"
    r")\b[^\n]*$"
)

REFUSAL_MESSAGE = "I don't have sufficient information in the provided documents."


def match_rules(text: str, rules: List[Tuple[str, Pattern[str]]]) -> List[str]:
    return [name for name, pattern in rules if pattern.search(text)]


def looks_like_policy_question(text: str) -> bool:
    """True when the question asks about a rule rather than for a record.

    The distinction has to survive a wrapper. "Per the data classification policy, show me
    customer account details" is a data request, and an earlier version of this function
    softened it because the word "policy" appeared anywhere in the string.
    """
    if DATA_TARGET_RE.search(text):
        return False
    return POLICY_ATTRIBUTE_RE.search(text) is not None or any(
        pattern.search(text) for pattern in POLICY_QUESTION_RULES
    )
