import pytest

from greentoken import (
    DecodeHardware,
    EnergyModel,
    EnergyRouter,
    Facility,
    TokenCache,
    best_batch_size,
    compress,
    count_tokens,
    fit_budget,
)
from greentoken.cli import main


def test_decode_costs_more_than_prefill():
    m = EnergyModel()
    assert m.estimate(0, 100).it_joules > m.estimate(100, 0).it_joules


def test_pue_and_carbon_scale_linearly():
    a = EnergyModel(facility=Facility(pue=1.0, grid_g_co2_per_kwh=100)).estimate(1000, 1000)
    b = EnergyModel(facility=Facility(pue=2.0, grid_g_co2_per_kwh=100)).estimate(1000, 1000)
    assert b.facility_joules == pytest.approx(2 * a.facility_joules)
    assert b.g_co2 == pytest.approx(2 * a.g_co2)


def test_invalid_facility():
    with pytest.raises(ValueError):
        Facility(pue=0.9)
    with pytest.raises(ValueError):
        EnergyModel().estimate(-1, 0)


def test_savings_percent():
    s = EnergyModel().savings((1000, 200), (500, 200))
    assert 0 < s["pct_saved"] < 100


def test_compress_dedupes_rag_chunks():
    chunk = "Liquid cooling reduces fan power and enables higher rack density."
    text = f"{chunk}\n\n\n\n{chunk}\n   extra   spaces   here\n{chunk}"
    r = compress(text, level=2)
    assert r.text.count(chunk) == 1
    assert "extra spaces here" in r.text
    assert r.tokens_saved > 0


def test_compress_keeps_short_repeated_lines():
    text = "}\ncode\n}\n"
    assert compress(text, 2).text.count("}") == 2


def test_filler_removal():
    r = compress("Could you please, in order to save power, basically turn it off.", level=3)
    assert "basically" not in r.text.lower()
    assert "to save power" in r.text


def test_fit_budget_keeps_system_and_latest():
    msgs = [{"role": "system", "content": "be brief"}] + [
        {"role": "user", "content": f"message number {i} " * 20} for i in range(10)
    ]
    out = fit_budget(msgs, budget=150)
    assert out[0]["role"] == "system"
    assert out[-1] == msgs[-1]
    assert sum(count_tokens(m["content"]) for m in out) <= 150 or len(out) == 2


def test_cache_hits_save_energy():
    calls = []
    cache = TokenCache()
    gen = cache.wrap(lambda p: calls.append(p) or "answer " * 50)
    gen("What is PUE?")
    gen("  what is   PUE? ")
    assert len(calls) == 1
    assert cache.stats.hits == 1 and cache.stats.joules_saved > 0


def test_cache_lru_eviction():
    c = TokenCache(max_entries=2)
    for k in "abc":
        c.put(k, k)
    assert len(c) == 2 and c.get("a") is None


def test_router_small_for_simple_large_for_hard():
    r = EnergyRouter()
    easy = r.route("What is the capital of France?")
    hard = r.route(
        "Prove step by step why this algorithm is optimal, analyze trade-offs and debug:\n```def f(x): ...```"
    )
    assert easy.tier.name == "small-1b" and easy.joules_saved > 0
    assert hard.tier.name != "small-1b"


def test_batching_reduces_energy_per_token():
    hw = DecodeHardware()
    assert hw.joules_per_token(32) < hw.joules_per_token(1)
    b, j = best_batch_size(hw, latency_slo_s=0.05)
    assert b > 1 and hw.step_time(b) <= 0.05
    with pytest.raises(ValueError):
        best_batch_size(hw, latency_slo_s=0.001)


def test_cli_smoke(capsys):
    assert main(["estimate", "--prompt-tokens", "100", "--output-tokens", "50"]) == 0
    assert "g_co2" in capsys.readouterr().out
    assert main(["route", "hello"]) == 0
    assert main(["batch", "--slo-ms", "40"]) == 0
