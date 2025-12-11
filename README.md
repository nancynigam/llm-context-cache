# LLM Context-Reuse Cache

Reuses transformer KV states across prompts to reduce prefill latency and accelerate Large Language Model inference

### Overview

LLMs already cache Key-Value (KV) attention tensors while generating tokens — but only within a single request.
LLM Context-Reuse Cache extends this to the system level, allowing reuse of cached attention states across multiple requests that share common prefixes.

→ Goal: cut prefill latency and boost throughput for repeated or overlapping prompts.
Perfect for chatbots, document QA, and multi-user assistants.

### Core Features

- Cross-Request Reuse: Skip recomputation for identical prefixes across sessions.
- Partial-Prefix Reuse: Detect and reuse longest shared prefix via token hashing.
- Adaptive Cache Policy: LRU + popularity-aware eviction to fit GPU memory budgets.
- Metrics & Observability: Track hit rates, latency savings, and memory footprint.
- Plug-and-Play: Works with Hugging Face models and can integrate with vLLM / TGI.


### Comparison

| Built-in KV Cache             | LLM Context-Reuse Cache           |
| ----------------------------- | --------------------------------- |
| Works only within one request | Works across requests / users     |
| Ephemeral                     | Persistent with eviction policies |
| No metrics                    | Observable + tunable              |
| Local only                    | Extensible to distributed setups  |


### Benchmarks

This benchmark simulates a realistic chat workload where many requests share a long prefix  
(e.g., system prompt + conversation history), but each request contains a small delta (user question).

**Setup**
- Model: `gpt2`
- Hardware: <your machine here, e.g., “M1 Pro CPU only” or “RTX 4090 GPU”>
- Shared prefix length: **~<PREFIX_TOKENS> tokens**
- Delta length: ~1–2 sentences
- New tokens generated: 32
- Number of queries: <N>

### Results

No-cache latencies (ms):
2032.4, 1877.6, 1934.1, 1925.9, 1944.1, 1899.9, 2276.7, 2120.5, 1907.4, 1864.6

With-prefix-cache latencies (ms):
948.5, 873.5, 864.8, 906.0, 971.9, 1398.5, 1119.2, 1010.8, 984.7, 944.4

Markdown summary table :

| Mode               | N   | Avg latency (ms) | Min (ms) | Max (ms) |
|--------------------|-----|------------------|----------|----------|
| No cache           | 10  |           1978.3 |   1864.6 |   2276.7 |
| With prefix cache  | 10  |           1002.2 |    864.8 |   1398.5 |

Estimated average speedup: **1.97×**

### Interpretation

With a huge shared prefix, prefill dominates latency.  
Caching the prefix’s KV attention state avoids reprocessing this long context on every request.

In this benchmark, prefix-KV reuse reduced average latency from  
**1978 ms → 1002 ms**, giving a **1.97x ** speedup.

As shared prefix length grows (e.g., longer history or more few-shot examples),  
the benefit of prefix KV caching increases proportionally.
