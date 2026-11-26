"""Command-line interface: ``greentoken {estimate,compress,route,batch}``."""

from __future__ import annotations

import argparse
import json
import sys

from .batching import DecodeHardware, best_batch_size
from .compress import compress
from .energy import EnergyModel, Facility
from .router import EnergyRouter


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="greentoken", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("estimate", help="energy/CO2 for a token workload")
    e.add_argument("--prompt-tokens", type=int, required=True)
    e.add_argument("--output-tokens", type=int, required=True)
    e.add_argument("--requests", type=int, default=1)
    e.add_argument("--pue", type=float, default=1.2)
    e.add_argument("--grid", type=float, default=400.0, help="gCO2e/kWh")

    c = sub.add_parser("compress", help="compress a prompt read from stdin or a file")
    c.add_argument("file", nargs="?")
    c.add_argument("--level", type=int, default=2, choices=[1, 2, 3])

    r = sub.add_parser("route", help="pick the smallest adequate model for a prompt")
    r.add_argument("prompt")

    b = sub.add_parser("batch", help="most energy-efficient batch size under a latency SLO")
    b.add_argument("--slo-ms", type=float, default=50.0)

    a = p.parse_args(argv)

    if a.cmd == "estimate":
        fp = EnergyModel(facility=Facility(pue=a.pue, grid_g_co2_per_kwh=a.grid)).estimate(
            a.prompt_tokens, a.output_tokens, a.requests
        )
        print(json.dumps(fp.__dict__, indent=2))
    elif a.cmd == "compress":
        text = open(a.file).read() if a.file else sys.stdin.read()
        res = compress(text, a.level)
        sys.stdout.write(res.text + "\n")
        print(
            f"tokens: {res.tokens_before} -> {res.tokens_after} ({100 * (1 - res.ratio):.1f}% saved)",
            file=sys.stderr,
        )
    elif a.cmd == "route":
        d = EnergyRouter().route(a.prompt)
        print(
            json.dumps(
                {
                    "model": d.tier.name,
                    "difficulty": round(d.difficulty, 3),
                    "est_joules": round(d.est_joules, 2),
                    "joules_saved_vs_largest": round(d.joules_saved, 2),
                },
                indent=2,
            )
        )
    elif a.cmd == "batch":
        bs, j = best_batch_size(DecodeHardware(), a.slo_ms / 1000)
        print(json.dumps({"batch_size": bs, "j_per_token": round(j, 4)}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
