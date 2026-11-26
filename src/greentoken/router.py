"""Energy-aware model routing.

Serving every request with the largest model wastes energy: decode energy per
token scales roughly linearly with active parameters. The router sends each
request to the *smallest* model whose capability tier covers the request's
estimated difficulty, and reports the energy delta versus always using the
largest model.

The difficulty heuristic is intentionally transparent (length, reasoning cues,
code, math). Swap in a learned classifier via ``difficulty_fn``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from .energy import EnergyModel, EnergyProfile, Facility
from .tokens import count_tokens

_REASONING = re.compile(
    r"\b(prove|derive|step[- ]by[- ]step|why|analy[sz]e|compare|trade-?offs?|design|architect|optimi[sz]e|debug)\b",
    re.IGNORECASE,
)
_CODE = re.compile(r"```|\bdef |\bclass |\bfunction\b|#include|SELECT .* FROM", re.IGNORECASE)
_MATH = re.compile(r"[=∑∫√^]|\b(integral|derivative|matrix|eigen|probability)\b", re.IGNORECASE)


def heuristic_difficulty(prompt: str) -> float:
    """Return a score in [0, 1]."""
    score = min(count_tokens(prompt) / 2000, 0.35)
    score += min(len(_REASONING.findall(prompt)) * 0.15, 0.35)
    if _CODE.search(prompt):
        score += 0.2
    if _MATH.search(prompt):
        score += 0.15
    return min(score, 1.0)


@dataclass(frozen=True)
class ModelTier:
    name: str
    max_difficulty: float
    profile: EnergyProfile


DEFAULT_TIERS = (
    ModelTier("small-1b", 0.25, EnergyProfile("1b", 0.008, 0.05, 0.3)),
    ModelTier("medium-8b", 0.6, EnergyProfile("8b", 0.05, 0.30, 1.0)),
    ModelTier("large-70b", 1.0, EnergyProfile("70b", 0.40, 2.50, 3.0)),
)


@dataclass
class RouteDecision:
    tier: ModelTier
    difficulty: float
    est_joules: float
    baseline_joules: float

    @property
    def joules_saved(self) -> float:
        return self.baseline_joules - self.est_joules


@dataclass
class EnergyRouter:
    tiers: tuple[ModelTier, ...] = DEFAULT_TIERS
    facility: Facility = field(default_factory=Facility)
    difficulty_fn: Callable[[str], float] = heuristic_difficulty
    expected_output_tokens: int = 300

    def __post_init__(self) -> None:
        if not self.tiers:
            raise ValueError("need at least one tier")
        self.tiers = tuple(sorted(self.tiers, key=lambda t: t.max_difficulty))

    def route(self, prompt: str, expected_output_tokens: int | None = None) -> RouteDecision:
        d = self.difficulty_fn(prompt)
        out_tok = self.expected_output_tokens if expected_output_tokens is None else expected_output_tokens
        p_tok = count_tokens(prompt)
        tier = next((t for t in self.tiers if d <= t.max_difficulty), self.tiers[-1])
        est = EnergyModel(tier.profile, self.facility).estimate(p_tok, out_tok).facility_joules
        base = EnergyModel(self.tiers[-1].profile, self.facility).estimate(p_tok, out_tok).facility_joules
        return RouteDecision(tier, d, est, base)
