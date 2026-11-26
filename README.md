# greentoken

**Measure and reduce the energy cost of every LLM token in the datacenter.**

AI inference is now one of the fastest-growing loads in datacenters, and most of
its energy is spent on tokens that didn't need to be processed: duplicated RAG
context, repeated prompts, oversized models for simple questions, and GPUs
running at batch sizes that waste memory bandwidth. `greentoken` is a small,
dependency-free Python toolkit that attacks each of those, and puts a number on
the savings in joules, kWh, gCO2e, and litres of water.

Companion project: [carbonshift](https://github.com/lakkaramnaveen/carbonshift)
decides *when and where* workloads run; `greentoken` cuts *how much energy each
request needs*.

## Where the energy goes

| Lever | Module | Why it saves energy |
|---|---|---|
| Accounting | `energy` | Splits prefill (compute-bound, cheap/token) from decode (memory-bound, ~10× costlier/token), then applies PUE, grid carbon, and water intensity |
| Prompt compression | `compress` | Fewer prompt tokens means less prefill compute and a smaller KV cache read on *every* decode step |
| Response caching | `cache` | A cache hit skips both prefill and decode, and the cache tracks the joules it saved |
| Model routing | `router` | Decode energy scales roughly with active parameters, so easy prompts go to the smallest adequate model |
| Batch sizing | `batching` | Roofline model: picks the batch that minimizes J/token while meeting a latency SLO |

## Quick start

```bash
pip install -e ".[dev]"
pytest -q

greentoken estimate --prompt-tokens 2000 --output-tokens 500 --pue 1.1 --grid 350
greentoken route "What is the capital of France?"
greentoken batch --slo-ms 50
cat rag_context.txt | greentoken compress --level 2 > compressed.txt
```

```python
from greentoken import EnergyModel, TokenCache, EnergyRouter, compress

res = compress(rag_context, level=2)          # dedupe overlapping chunks
print(res.tokens_saved, EnergyModel().savings((res.tokens_before, 300), (res.tokens_after, 300)))

cache = TokenCache(ttl_s=3600)
generate = cache.wrap(my_llm_call)            # cache.stats.joules_saved grows on hits

decision = EnergyRouter().route(prompt)       # decision.tier.name, decision.joules_saved
```

## Measured on Apple Silicon (M4 Pro, 20-core GPU)

`bench/apple_gpu_bench.py` runs a model on the Metal GPU with MLX and samples
power with `powermetrics`:

```bash
pip install -e ".[apple]"
sudo python bench/apple_gpu_bench.py          # sudo enables real power telemetry
```

First run, Qwen2.5-0.5B-Instruct (4-bit), 1024-token prompt, 256 output tokens:

| Phase | Throughput | J/token* |
|---|---|---|
| Prefill | ~5,200–7,200 tok/s | ~0.004–0.006 |
| Decode | ~385–425 tok/s | ~0.07–0.08 |

\*Without sudo, power is **assumed** at 30 W, so the absolute J/token values
are estimates. The ratio is measured: on the same hardware, each decode token
costs **~12–18× the energy of a prompt token**. So cutting output length
(concise system prompts, `max_tokens`, structured outputs) is usually the
biggest win, followed by caching. Raw results are in `bench/results/`.

## Assumptions and limits

* The default `EnergyProfile` coefficients are order-of-magnitude values for a
  7B model on a datacenter GPU. Calibrate them with the benchmark or with your
  own telemetry (DCGM / `nvidia-smi --query-gpu=power.draw`).
* The router's difficulty heuristic is transparent but crude. Replace
  `difficulty_fn` with a learned classifier for production use.
* Level-3 compression (filler removal) changes wording slightly. Levels 1–2
  only remove whitespace and exact duplicates.

## Daily research

`research/` holds a dated log written every morning by a scheduled agent,
covering new work on efficient datacenters and inference, plus ideas to fold
back into this repo.

## License

MIT
