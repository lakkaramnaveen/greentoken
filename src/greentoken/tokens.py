"""Token counting with an optional exact backend.

Uses ``tiktoken`` when installed; otherwise falls back to a fast heuristic
(~4 characters per token for English, word-aware) that is accurate to within
roughly 10-15% for typical prose, which is enough for budgeting decisions.
"""

from __future__ import annotations

import re
from functools import lru_cache

_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@lru_cache(maxsize=1)
def _encoder():
    try:
        import tiktoken  # type: ignore

        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover - depends on optional install
        return None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    enc = _encoder()
    if enc is not None:  # pragma: no cover
        return len(enc.encode(text))
    total = 0
    for piece in _WORD_RE.findall(text):
        # long words split into multiple BPE tokens
        total += max(1, (len(piece) + 3) // 4) if piece.isalnum() else 1
    return total
