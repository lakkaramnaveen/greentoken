"""Measure prefill/decode energy per token on an Apple Silicon GPU.

Runs a small model with MLX on the Metal GPU and samples package power with
``powermetrics``. Output: J/token for prefill and decode, which can be used as
an ``EnergyProfile`` to calibrate greentoken's estimates.

    pip install mlx-lm
    sudo python bench/apple_gpu_bench.py --model mlx-community/Qwen2.5-0.5B-Instruct-4bit

``powermetrics`` needs root. Without sudo the script still measures throughput,
and estimates energy from ``--assumed-watts``, clearly labelled as an estimate.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time


class PowerSampler:
    """Background ``powermetrics`` sampler for GPU + CPU power (mW)."""

    _RE = re.compile(r"^(GPU|CPU|Combined) Power[^:]*:\s*([\d.]+)\s*mW", re.MULTILINE)

    def __init__(self, interval_ms: int = 100):
        self.interval_ms = interval_ms
        self.available = os.geteuid() == 0 and shutil.which("powermetrics") is not None
        self._proc: subprocess.Popen | None = None
        self._out = tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False)

    def __enter__(self):
        if self.available:
            self._proc = subprocess.Popen(
                ["powermetrics", "--samplers", "cpu_power,gpu_power", "-i", str(self.interval_ms)],
                stdout=self._out,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(0.3)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._t0
        if self._proc:
            self._proc.terminate()
            self._proc.wait()
        return False

    def mean_watts(self) -> float | None:
        if not self.available:
            return None
        self._out.seek(0)
        combined = [float(v) for k, v in self._RE.findall(self._out.read()) if k == "Combined"]
        return (sum(combined) / len(combined)) / 1000 if combined else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    ap.add_argument("--prompt-tokens", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--assumed-watts", type=float, default=30.0, help="used only without powermetrics")
    a = ap.parse_args()

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import stream_generate

    model, tok = load(a.model)
    prompt_ids = (tok.encode("The history of datacenter cooling. ") * a.prompt_tokens)[: a.prompt_tokens]

    # warm-up (compiles Metal kernels)
    for _ in stream_generate(model, tok, prompt_ids[:32], max_tokens=4):
        pass

    results = {}
    for phase, max_tokens in (("prefill", 1), ("full", a.max_tokens)):
        sampler = PowerSampler()
        n = 0
        with sampler:
            for _ in stream_generate(model, tok, prompt_ids, max_tokens=max_tokens):
                n += 1
            mx.eval()
        w = sampler.mean_watts()
        results[phase] = {
            "seconds": sampler.elapsed,
            "generated": n,
            "watts": w if w is not None else a.assumed_watts,
            "measured": w is not None,
        }

    pre, full = results["prefill"], results["full"]
    prefill_j = pre["watts"] * pre["seconds"]
    decode_s = max(full["seconds"] - pre["seconds"], 1e-9)
    decode_tokens = max(full["generated"] - 1, 1)
    report = {
        "model": a.model,
        "device": str(mx.default_device()),
        "power_measured": pre["measured"] and full["measured"],
        "prefill_tokens_per_s": a.prompt_tokens / pre["seconds"],
        "decode_tokens_per_s": decode_tokens / decode_s,
        "prefill_j_per_token": prefill_j / a.prompt_tokens,
        "decode_j_per_token": (full["watts"] * decode_s) / decode_tokens,
        "raw": results,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
