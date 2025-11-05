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

  
