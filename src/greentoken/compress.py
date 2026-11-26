"""Lossless-first prompt compression.

Every prompt token costs prefill energy, and the KV cache it creates costs
memory bandwidth on every decode step. These passes remove tokens that carry
no information, in increasing order of aggressiveness:

1. ``normalize_whitespace`` – collapse runs of spaces/blank lines (lossless).
2. ``dedupe_lines`` – drop exact duplicate lines/paragraphs, common in
   RAG contexts that retrieve overlapping chunks (lossless for meaning).
3. ``strip_filler`` – remove conversational filler phrases (near-lossless).
4. ``fit_budget`` – keep the system prompt + newest turns that fit a token
   budget (lossy; explicit opt-in).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .tokens import count_tokens

_FILLER = [
    r"\bplease note that\b",
    r"\bit is important to note that\b",
    r"\bas (?:an|a) (?:ai|language model)[^.,]*[.,]?",
    r"\bi would like you to\b",
    r"\bcould you please\b",
    r"\bbasically\b",
    r"\bkind of\b",
    r"\bsort of\b",
    r"\bin order to\b",
]
_FILLER_RE = re.compile("|".join(_FILLER), re.IGNORECASE)
_REPLACEMENTS = {re.compile(r"\bin order to\b", re.IGNORECASE): "to"}


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def dedupe_lines(text: str, min_len: int = 20) -> str:
    """Drop repeated lines longer than ``min_len`` chars (keeps first copy).

    Short lines (headings, list bullets, code braces) are kept even if repeated
    because they often carry structure.
    """
    seen: set[str] = set()
    out = []
    for line in text.split("\n"):
        key = re.sub(r"\s+", " ", line.strip().lower())
        if len(key) >= min_len:
            if key in seen:
                continue
            seen.add(key)
        out.append(line)
    return "\n".join(out)


def strip_filler(text: str) -> str:
    for pat, rep in _REPLACEMENTS.items():
        text = pat.sub(rep, text)
    text = _FILLER_RE.sub("", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r" ([.,;:!?])", r"\1", text)
    return text.strip()


def fit_budget(messages: list[dict], budget: int) -> list[dict]:
    """Keep system messages + the most recent turns within ``budget`` tokens."""
    system = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    used = sum(count_tokens(m.get("content", "")) for m in system)
    kept: list[dict] = []
    for m in reversed(rest):
        cost = count_tokens(m.get("content", ""))
        if used + cost > budget and kept:
            break
        kept.append(m)
        used += cost
    return system + list(reversed(kept))


@dataclass
class CompressionResult:
    text: str
    tokens_before: int
    tokens_after: int

    @property
    def ratio(self) -> float:
        return self.tokens_after / self.tokens_before if self.tokens_before else 1.0

    @property
    def tokens_saved(self) -> int:
        return self.tokens_before - self.tokens_after


def compress(text: str, level: int = 2) -> CompressionResult:
    """Apply passes up to ``level`` (1=whitespace, 2=+dedupe, 3=+filler)."""
    before = count_tokens(text)
    out = normalize_whitespace(text)
    if level >= 2:
        out = dedupe_lines(out)
    if level >= 3:
        out = strip_filler(out)
    return CompressionResult(out, before, count_tokens(out))
