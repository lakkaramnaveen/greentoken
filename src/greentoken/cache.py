"""Response cache that accounts for the energy it saves.

The cheapest token is the one never generated. Production LLM traffic has a
long tail of repeated prompts (FAQ bots, retries, CI pipelines); an exact cache
on a normalized key avoids both prefill and decode for those requests.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable

from .energy import EnergyModel
from .tokens import count_tokens


def _normalize(prompt: str, model: str) -> str:
    key = re.sub(r"\s+", " ", prompt.strip().lower())
    return hashlib.sha256(f"{model}\x00{key}".encode()).hexdigest()


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    prompt_tokens_saved: int = 0
    output_tokens_saved: int = 0
    joules_saved: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


@dataclass
class TokenCache:
    max_entries: int = 10_000
    ttl_s: float | None = None
    energy: EnergyModel = field(default_factory=EnergyModel)
    stats: CacheStats = field(default_factory=CacheStats)

    def __post_init__(self) -> None:
        self._store: OrderedDict[str, tuple[float, str, int, int]] = OrderedDict()

    def get(self, prompt: str, model: str = "default") -> str | None:
        k = _normalize(prompt, model)
        entry = self._store.get(k)
        if entry is None:
            self.stats.misses += 1
            return None
        ts, response, p_tok, o_tok = entry
        if self.ttl_s is not None and time.monotonic() - ts > self.ttl_s:
            del self._store[k]
            self.stats.misses += 1
            return None
        self._store.move_to_end(k)
        self.stats.hits += 1
        self.stats.prompt_tokens_saved += p_tok
        self.stats.output_tokens_saved += o_tok
        self.stats.joules_saved += self.energy.estimate(p_tok, o_tok).facility_joules
        return response

    def put(self, prompt: str, response: str, model: str = "default") -> None:
        k = _normalize(prompt, model)
        self._store[k] = (time.monotonic(), response, count_tokens(prompt), count_tokens(response))
        self._store.move_to_end(k)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)

    def wrap(self, generate: Callable[[str], str], model: str = "default") -> Callable[[str], str]:
        """Decorate a ``prompt -> response`` function with this cache."""

        def cached(prompt: str) -> str:
            hit = self.get(prompt, model)
            if hit is not None:
                return hit
            out = generate(prompt)
            self.put(prompt, out, model)
            return out

        return cached

    def __len__(self) -> int:
        return len(self._store)
