"""Energy and carbon accounting for LLM inference.

The model separates the two phases of transformer inference because they have
very different energy profiles:

* **prefill** (processing the prompt) is compute-bound and highly parallel, so
  the energy per token is low;
* **decode** (generating output) is memory-bandwidth-bound and sequential, so
  each generated token costs several times more energy than a prompt token.

Facility overhead is captured with PUE (Power Usage Effectiveness), and
emissions with the grid carbon intensity in gCO2e/kWh.

Default coefficients are deliberately conservative order-of-magnitude values
for a ~7B-parameter model on a modern datacenter GPU; calibrate them with
``bench/apple_gpu_bench.py`` or your own power telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

JOULES_PER_KWH = 3.6e6


@dataclass(frozen=True)
class EnergyProfile:
    """Per-token energy coefficients for one model/hardware pair."""

    name: str = "7b-datacenter-gpu"
    prefill_j_per_token: float = 0.05
    decode_j_per_token: float = 0.30
    idle_overhead_j_per_request: float = 1.0  # scheduling, KV setup, networking

    def __post_init__(self) -> None:
        for attr in ("prefill_j_per_token", "decode_j_per_token", "idle_overhead_j_per_request"):
            if getattr(self, attr) < 0:
                raise ValueError(f"{attr} must be non-negative")


@dataclass(frozen=True)
class Facility:
    """Datacenter-level multipliers."""

    pue: float = 1.2
    grid_g_co2_per_kwh: float = 400.0
    water_l_per_kwh: float = 1.8  # on-site WUE; 0 for dry-cooled sites

    def __post_init__(self) -> None:
        if self.pue < 1.0:
            raise ValueError("PUE cannot be below 1.0")
        if self.grid_g_co2_per_kwh < 0 or self.water_l_per_kwh < 0:
            raise ValueError("intensities must be non-negative")


@dataclass
class Footprint:
    it_joules: float
    facility_joules: float
    kwh: float
    g_co2: float
    water_l: float

    def __add__(self, other: "Footprint") -> "Footprint":
        return Footprint(
            self.it_joules + other.it_joules,
            self.facility_joules + other.facility_joules,
            self.kwh + other.kwh,
            self.g_co2 + other.g_co2,
            self.water_l + other.water_l,
        )


@dataclass
class EnergyModel:
    profile: EnergyProfile = field(default_factory=EnergyProfile)
    facility: Facility = field(default_factory=Facility)

    def estimate(self, prompt_tokens: int, output_tokens: int, requests: int = 1) -> Footprint:
        if prompt_tokens < 0 or output_tokens < 0 or requests < 0:
            raise ValueError("token and request counts must be non-negative")
        it_j = (
            prompt_tokens * self.profile.prefill_j_per_token
            + output_tokens * self.profile.decode_j_per_token
            + requests * self.profile.idle_overhead_j_per_request
        )
        facility_j = it_j * self.facility.pue
        kwh = facility_j / JOULES_PER_KWH
        return Footprint(
            it_joules=it_j,
            facility_joules=facility_j,
            kwh=kwh,
            g_co2=kwh * self.facility.grid_g_co2_per_kwh,
            water_l=kwh * self.facility.water_l_per_kwh,
        )

    def savings(
        self,
        before: tuple[int, int],
        after: tuple[int, int],
        requests_before: int = 1,
        requests_after: int = 1,
    ) -> dict[str, float]:
        """Compare (prompt, output) token counts before/after an optimization."""
        b = self.estimate(*before, requests=requests_before)
        a = self.estimate(*after, requests=requests_after)
        saved_j = b.facility_joules - a.facility_joules
        return {
            "joules_saved": saved_j,
            "pct_saved": 100.0 * saved_j / b.facility_joules if b.facility_joules else 0.0,
            "g_co2_saved": b.g_co2 - a.g_co2,
        }
