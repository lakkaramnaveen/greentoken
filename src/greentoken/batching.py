"""Batch-size vs. energy/latency trade-off.

Decode is memory-bound: weights are streamed from HBM once per step regardless
of how many sequences are in the batch, so energy per token falls sharply with
batch size until the GPU becomes compute-bound. Past that point, batching only
adds latency. This module models that curve and picks the most energy-efficient
batch size that still meets a per-token latency SLO.

Model (per decode step, batch size b):
    t(b) = max(t_mem, b * t_compute)        # roofline
    P(b) = P_idle + (P_max - P_idle) * util(b)
    J/token(b) = P(b) * t(b) / b
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecodeHardware:
    t_mem_s: float = 0.020  # time to stream weights once (e.g. 14 GB / 700 GB/s)
    t_compute_s: float = 0.0004  # pure compute time per sequence per step
    p_idle_w: float = 100.0
    p_max_w: float = 700.0

    def step_time(self, b: int) -> float:
        return max(self.t_mem_s, b * self.t_compute_s)

    def power(self, b: int) -> float:
        util = min(1.0, (b * self.t_compute_s) / self.step_time(b))
        # memory traffic keeps power well above idle even at low compute util
        util = 0.35 + 0.65 * util
        return self.p_idle_w + (self.p_max_w - self.p_idle_w) * util

    def joules_per_token(self, b: int) -> float:
        if b < 1:
            raise ValueError("batch size must be >= 1")
        return self.power(b) * self.step_time(b) / b


def best_batch_size(hw: DecodeHardware, latency_slo_s: float, max_batch: int = 512) -> tuple[int, float]:
    """Return (batch, J/token) minimizing energy with step_time <= SLO."""
    best: tuple[int, float] | None = None
    for b in range(1, max_batch + 1):
        if hw.step_time(b) > latency_slo_s:
            break
        j = hw.joules_per_token(b)
        if best is None or j < best[1]:
            best = (b, j)
    if best is None:
        raise ValueError("latency SLO is below the memory-bound step time; unattainable")
    return best
